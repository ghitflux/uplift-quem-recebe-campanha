import numpy as np
import pandas as pd
import pytest

from uplift_campanha import analise, curvas, dados, site


def _base_sintetica(n: int = 9_000, semente: int = 0) -> pd.DataFrame:
    """Base no formato da de Hillstrom, com efeito conhecido.

    E-mail feminino: +0,20 em visita só para quem já comprou a linha feminina (+0,10 na média).
    E-mail masculino: +0,10 em visita para todo mundo. A taxa de base cresce com o histórico de
    gasto, que não tem relação com o efeito.
    """
    gerador = np.random.default_rng(semente)
    grupo = gerador.choice(["E-mail feminino", "E-mail masculino", "Sem e-mail"], n)
    feminino = gerador.integers(0, 2, n)
    historico = gerador.lognormal(5.0, 0.8, n) + 30
    recencia = gerador.integers(1, 13, n)
    regiao = gerador.integers(0, 3, n)
    canal = gerador.integers(0, 3, n)
    taxa = 0.05 + 0.25 * (np.log(historico) > 5.2)
    taxa = taxa + np.where(grupo == "E-mail feminino", 0.20 * feminino, 0.0) + np.where(grupo == "E-mail masculino", 0.10, 0.0)
    visita = (gerador.uniform(size=n) < taxa).astype(int)
    conversao = visita * (gerador.uniform(size=n) < 0.2).astype(int)
    tabela = pd.DataFrame({
        "grupo": grupo,
        "recencia": recencia,
        "historico": historico,
        "log_historico": np.log(historico),
        "comprou_masculino": 1 - feminino * gerador.integers(0, 2, n),
        "comprou_feminino": feminino,
        "cliente_novo": gerador.integers(0, 2, n),
        "visita": visita,
        "conversao": conversao,
        "gasto": conversao * gerador.exponential(80, n),
    })
    for i, nome in enumerate(["urbana", "suburbana", "rural"]):
        tabela[f"regiao_{nome}"] = (regiao == i).astype(int)
    for i, nome in enumerate(["telefone", "web", "multicanal"]):
        tabela[f"canal_{nome}"] = (canal == i).astype(int)
    return tabela


@pytest.fixture(scope="module")
def base() -> pd.DataFrame:
    return _base_sintetica()


@pytest.fixture(scope="module")
def avaliacao_feminino(base):
    """Família logística só, para o teste ser rápido: escores, tabela de métodos e bootstrap."""
    recorte = dados.recorte_da_campanha(base, "E-mail feminino")
    treino, teste = dados.separar_treino_teste(recorte, semente=1)
    escores = analise.pontuar(treino, teste, "visita", familias=["logistica"]) | analise.regras(teste, "E-mail feminino")
    metodos, boot = analise.avaliar(teste, "visita", escores, base="resposta|logistica", repeticoes=200)
    return recorte, teste, escores, metodos, boot


def test_resumo_e_equilibrio_da_base_sorteada(base):
    resumo = analise.resumo_da_base(base)
    assert resumo["clientes"] == len(base)
    assert sum(resumo["por_grupo"].values()) == len(base)
    assert resumo["teste_da_divisao"]["p"] > 0.01
    equilibrio = analise.equilibrio(base)
    assert list(equilibrio.columns) == dados.CAMPANHAS
    assert np.abs(equilibrio.to_numpy()).max() < 0.1


def test_efeitos_medios_recuperam_os_efeitos_plantados(base):
    efeitos = analise.efeitos_medios(base)
    assert efeitos.loc[("E-mail feminino", "visita"), "efeito"] == pytest.approx(0.10, abs=0.03)
    assert efeitos.loc[("E-mail masculino", "visita"), "efeito"] == pytest.approx(0.10, abs=0.03)
    gasto = efeitos.loc[("E-mail masculino", "gasto")]
    assert gasto["ic_bootstrap_inferior"] < gasto["efeito"] < gasto["ic_bootstrap_superior"]
    poder = analise.poder_por_recorte(base, efeitos)
    minimos = poder["visita"]["minimo_detectavel"]
    assert minimos["um decil do teste"]["efeito"] > minimos["metade de teste"]["efeito"] > minimos["experimento inteiro"]["efeito"]


def test_uplift_supera_o_modelo_de_resposta_e_empata_com_a_regra_certa(avaliacao_feminino):
    _, _, _, metodos, _ = avaliacao_feminino
    assert metodos.loc["dois_modelos|logistica", "qini"] > 2 * metodos.loc["resposta|logistica", "qini"]
    assert metodos.loc["dois_modelos|logistica", "diferenca_ic_inferior"] > 0  # diferença pareada para a base
    assert metodos.loc["resposta|logistica", "diferenca_para_a_base"] == 0
    # A regra "já comprou a linha feminina" é o próprio efeito plantado: ninguém ganha dela com folga.
    assert metodos.loc[analise.REGRA_LINHA, "qini"] == pytest.approx(metodos.loc["dois_modelos|logistica", "qini"], rel=0.15)
    assert metodos.loc[analise.REGRA_RECENCIA, "qini_ic_inferior"] < 0 < metodos.loc[analise.REGRA_RECENCIA, "qini_ic_superior"]


def test_comparar_e_antissimetrico(avaliacao_feminino):
    _, teste, escores, _, boot = avaliacao_feminino
    ida = analise.comparar(teste, "visita", escores, boot, "dois_modelos|logistica", analise.REGRA_RECENCIA)
    volta = analise.comparar(teste, "visita", escores, boot, analise.REGRA_RECENCIA, "dois_modelos|logistica")
    assert ida["diferenca"] == pytest.approx(-volta["diferenca"])
    assert ida["ic_inferior"] == pytest.approx(-volta["ic_superior"])
    assert ida["ic_inferior"] > 0


def test_valor_da_politica_fecha_com_o_efeito_medio_e_com_o_acaso(avaliacao_feminino):
    _, teste, escores, _, boot = avaliacao_feminino
    nomes = ["dois_modelos|logistica", "resposta|logistica"]
    politica = analise.valor_da_politica(teste, "visita", escores, boot, nomes)
    efeito = 1000 * (teste.loc[teste.tratado == 1, "visita"].mean() - teste.loc[teste.tratado == 0, "visita"].mean())
    # Mandar para todos rende o efeito médio, qualquer que seja o escore.
    assert politica.loc[(1.0, "dois_modelos|logistica"), "por_mil_clientes"] == pytest.approx(efeito)
    assert politica.loc[(1.0, "acaso"), "por_mil_clientes"] == pytest.approx(efeito)
    # Ao acaso, metade da base rende metade do efeito.
    assert politica.loc[(0.5, "acaso"), "por_mil_clientes"] == pytest.approx(efeito / 2)
    assert politica.loc[(0.5, "acaso"), "parte_do_efeito_total"] == 0.5
    # O efeito plantado está todo em metade da base: o modelo de uplift pega quase tudo com 50% dos envios.
    metade = politica.loc[(0.5, "dois_modelos|logistica")]
    assert metade["parte_do_efeito_total"] > 0.8
    assert metade["acima_ic_inferior"] > 0
    assert metade["por_mil_emails"] == pytest.approx(2 * metade["por_mil_clientes"])
    assert metade["por_mil_clientes"] > politica.loc[(0.5, "resposta|logistica"), "por_mil_clientes"]


def test_quem_cada_escore_escolhe_mostra_a_taxa_de_base_de_cada_publico(avaliacao_feminino):
    _, teste, escores, _, _ = avaliacao_feminino
    perfil = analise.quem_cada_escore_escolhe(teste, "visita", escores, "dois_modelos|logistica", "resposta|logistica", "E-mail feminino", fracao=0.5)
    assert list(perfil.index) == ["Base inteira", "Modelo de resposta", "Modelo de uplift", "Regra"]
    assert perfil.loc["Modelo de uplift", "clientes"] == round(len(teste) * 0.5)
    # O modelo de resposta busca quem já visita muito; o de uplift, quem já comprou a linha feminina.
    assert perfil.loc["Modelo de resposta", "taxa_sem_email"] > perfil.loc["Modelo de uplift", "taxa_sem_email"]
    assert perfil.loc["Modelo de uplift", "uplift"] > perfil.loc["Modelo de resposta", "uplift"]
    assert perfil.loc["Modelo de uplift", "comprou_feminino"] > 0.9
    assert perfil.loc["Regra", "comprou_feminino"] == 1.0
    assert perfil.loc["Regra", "uplift"] == pytest.approx(0.20, abs=0.05)
    # Mandar para todos rende o efeito médio; cada público rende o uplift dele vezes o seu tamanho.
    assert perfil.loc["Base inteira", "incrementais_por_mil_clientes"] == pytest.approx(1000 * perfil.loc["Base inteira", "uplift"])
    assert perfil.loc["Base inteira", "parte_do_efeito_total"] == 1.0
    assert perfil.loc["Modelo de uplift", "incrementais_por_mil_clientes"] == pytest.approx(500 * perfil.loc["Modelo de uplift", "uplift"], rel=1e-3)
    assert perfil.loc["Regra", "parte_do_efeito_total"] > 0.8


def test_decis_caem_do_topo_para_a_base(avaliacao_feminino):
    _, teste, escores, _, _ = avaliacao_feminino
    decis = analise.decis(teste, "visita", escores["dois_modelos|logistica"])
    assert len(decis) == 10
    assert decis["uplift"].iloc[:4].mean() > decis["uplift"].iloc[-4:].mean() + 0.10


def test_placebo_zera_o_ganho(avaliacao_feminino):
    recorte, _, _, metodos, _ = avaliacao_feminino
    observado = float(metodos.loc["dois_modelos|logistica", "qini"])
    placebo = analise.placebo(recorte, "visita", "dois_modelos|logistica", observado, repeticoes=30)
    assert abs(placebo["qini_placebo_medio"]) < 0.25 * observado
    assert placebo["qini_placebo_maximo"] < observado
    assert placebo["p_valor"] == pytest.approx(1 / 31)
    assert abs(placebo["efeito_medio_placebo_medio"]) < 0.02


def test_efeito_igual_para_todos_nao_rende_ganho_sobre_o_acaso(base):
    # No e-mail masculino sintético o efeito é o mesmo para todos: nenhum escore deve se afastar do acaso.
    recorte = dados.recorte_da_campanha(base, "E-mail masculino")
    treino, teste = dados.separar_treino_teste(recorte, semente=1)
    escores = analise.pontuar(treino, teste, "visita", familias=["logistica"]) | analise.regras(teste, "E-mail masculino")
    metodos, _ = analise.avaliar(teste, "visita", escores, base="resposta|logistica", repeticoes=200)
    dentro = (metodos["qini_ic_inferior"] < 0) & (metodos["qini_ic_superior"] > 0)
    assert dentro.sum() >= len(metodos) - 1  # com 95% de confiança e 7 escores, tolera um fora


def test_resumo_das_repeticoes_conta_vitorias():
    repeticoes = pd.DataFrame({"a": [0.3, 0.2, -0.1], "base": [0.1, 0.25, -0.2]})
    resumo = analise.resumo_das_repeticoes(repeticoes, "base")
    assert resumo.loc["a", "vezes_acima_do_acaso"] == 2
    assert resumo.loc["a", "vezes_acima_da_base"] == 2
    assert resumo.loc["base", "vezes_acima_da_base"] == 0
    assert resumo.loc["a", "qini_mediano"] == pytest.approx(0.2)


def test_escolha_do_principal_usa_so_o_treino_e_ordena_por_qini(base):
    recorte = dados.recorte_da_campanha(base, "E-mail feminino")
    treino, _ = dados.separar_treino_teste(recorte, semente=1)
    escolha = analise.escolher_principal(treino.iloc[:2500], "visita", dobras=3)
    assert len(escolha) == 8  # quatro métodos × duas famílias; o modelo de resposta não concorre
    assert not any(nome.startswith("resposta") for nome in escolha.index)
    assert escolha["qini_medio"].is_monotonic_decreasing
    assert escolha["qini_medio"].iloc[0] > 0


def test_numeros_do_site_seguem_o_formato_de_cada_idioma():
    assert site._n(9.12, 1, "pt") == "9,1"
    assert site._n(9.12, 1, "en") == "9.1"
    assert site._n(-1.44, 1, "es") == "−1,4"
    pontos = site._pontos(curvas.GRADE_PADRAO, list(curvas.GRADE_PADRAO * 0.045))
    assert len(pontos) == 51 and pontos[0] == [0.0, 0.0] and pontos[-1] == [1.0, 45.0]


def test_nome_do_escore():
    assert analise.nome_do_escore("dois_modelos|logistica") == "Dois modelos (T-learner) · regressão logística"
    assert analise.nome_do_escore(analise.REGRA_LINHA, "E-mail feminino") == "Regra: quem já comprou a linha feminina"
