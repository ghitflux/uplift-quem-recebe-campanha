"""Cada tabela do README e do notebook sai de uma função daqui.

Ordem da análise: o sorteio foi bem feito? → qual o efeito médio de cada e-mail? → o efeito
varia de cliente para cliente a ponto de valer escolher quem recebe? → o que cada jeito de
escolher entrega, em visitas e compras incrementais por mil clientes?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from uplift_campanha import curvas, dados, experimento, uplift

SEMENTE = 42
FAMILIAS = ["logistica", "lightgbm"]
METODOS_DE_UPLIFT = ["dois_modelos", "classe_transformada", "modelo_unico", "x_learner"]
FRACOES = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
PONTOS_DO_BOOTSTRAP = 2000  # posições do ordenamento em que cada curva reamostrada é calculada (ver curvas.bootstrap_das_curvas)
REGRA_RECENCIA = "regra_recencia"
REGRA_LINHA = "regra_linha"
LINHA_DA_CAMPANHA = {"E-mail feminino": "comprou_feminino", "E-mail masculino": "comprou_masculino"}


def chave(metodo: str, familia: str) -> str:
    return f"{metodo}|{familia}"


def nome_do_escore(chave_do_escore: str, campanha: str | None = None) -> str:
    """Nome legível de um escore: método e família, ou a regra de bolso."""
    if chave_do_escore == REGRA_RECENCIA:
        return "Regra: quem comprou mais recentemente"
    if chave_do_escore == REGRA_LINHA:
        linha = "feminina" if campanha == "E-mail feminino" else "masculina" if campanha == "E-mail masculino" else "da campanha"
        return f"Regra: quem já comprou a linha {linha}"
    if chave_do_escore == "principal_treinado_em_visita":
        return "Modelo de uplift treinado em visita"
    metodo, familia = chave_do_escore.split("|")
    return f"{uplift.NOMES[metodo]} · {uplift.FAMILIAS[familia]}"


# --------------------------------------------------------------------------- 1. o sorteio


def resumo_da_base(tabela: pd.DataFrame) -> dict:
    """Tamanho da base, clientes por grupo e teste da divisão em três partes iguais."""
    contagens = tabela["grupo"].value_counts().to_dict()
    return {
        "clientes": int(len(tabela)),
        "por_grupo": {g: int(n) for g, n in contagens.items()},
        "proporcao": {g: n / len(tabela) for g, n in contagens.items()},
        "teste_da_divisao": experimento.teste_da_divisao(contagens),
    }


def equilibrio(tabela: pd.DataFrame) -> pd.DataFrame:
    """Diferença padronizada de cada atributo entre cada e-mail e o grupo sem e-mail."""
    saida = experimento.tabela_de_equilibrio(tabela[dados.ATRIBUTOS], tabela["grupo"], dados.CONTROLE)
    saida = saida[dados.CAMPANHAS]
    saida.index = [dados.ROTULOS[a] for a in saida.index]
    return saida


# --------------------------------------------------------------------------- 2. efeito médio


def efeitos_medios(tabela: pd.DataFrame) -> pd.DataFrame:
    """Efeito de cada e-mail em visita, compra e gasto, contra o grupo sem e-mail, com intervalo de 95%."""
    linhas = []
    for campanha in dados.CAMPANHAS:
        for desfecho in dados.DESFECHOS:
            tratados = tabela.loc[tabela["grupo"] == campanha, desfecho].to_numpy()
            controles = tabela.loc[tabela["grupo"] == dados.CONTROLE, desfecho].to_numpy()
            linha = {"campanha": campanha, "desfecho": desfecho, **experimento.efeito_medio(tratados, controles)}
            if desfecho == "gasto":
                inferior, superior = experimento.intervalo_bootstrap_da_diferenca(tratados, controles, semente=SEMENTE)
                linha |= {"ic_bootstrap_inferior": inferior, "ic_bootstrap_superior": superior}
            linhas.append(linha)
    return pd.DataFrame(linhas).set_index(["campanha", "desfecho"])


def poder_por_recorte(tabela: pd.DataFrame, efeitos: pd.DataFrame, fracao_teste: float = 0.5) -> dict:
    """O que cada recorte da base consegue detectar, comparado ao efeito médio observado.

    Para visita e compra: a taxa do grupo sem e-mail, o efeito médio de cada campanha e o menor
    efeito detectável (80% de poder) com a base inteira, com a metade de teste e com um décimo
    do teste. Um recorte só mostra um efeito maior que o seu mínimo detectável.
    """
    controle = tabela[tabela["grupo"] == dados.CONTROLE]
    n_grupo = len(controle)
    recortes = {
        "experimento inteiro": n_grupo,
        "metade de teste": n_grupo * fracao_teste,
        "30% do teste": n_grupo * fracao_teste * 0.3,
        "um decil do teste": n_grupo * fracao_teste * 0.1,
    }
    saida = {}
    for desfecho in ("visita", "conversao"):
        taxa = float(controle[desfecho].mean())
        saida[desfecho] = {
            "taxa_sem_email": taxa,
            "efeito_medio": {c: float(efeitos.loc[(c, desfecho), "efeito"]) for c in dados.CAMPANHAS},
            "minimo_detectavel": {
                nome: {"clientes_por_grupo": int(round(n)), "efeito": experimento.efeito_minimo_detectavel(taxa, n)}
                for nome, n in recortes.items()
            },
            "clientes_por_grupo_para_detectar_o_efeito_medio": {
                c: experimento.amostra_por_grupo(taxa, taxa + float(efeitos.loc[(c, desfecho), "efeito"])) for c in dados.CAMPANHAS
            },
        }
    return saida


# --------------------------------------------------------------------------- 3. modelos e escores


def regras(tabela: pd.DataFrame, campanha: str) -> dict[str, np.ndarray]:
    """Regras de bolso: quem comprou há menos tempo; quem já comprou a linha anunciada no e-mail."""
    return {
        REGRA_RECENCIA: -tabela["recencia"].to_numpy(dtype=float),
        REGRA_LINHA: tabela[LINHA_DA_CAMPANHA[campanha]].to_numpy(dtype=float),
    }


def pontuar(
    treino: pd.DataFrame, teste: pd.DataFrame, desfecho: str, familias: list[str] = FAMILIAS, semente: int = SEMENTE
) -> dict[str, np.ndarray]:
    """Treina cada método no treino e devolve o escore de cada cliente do teste."""
    x_treino, x_teste = dados.matriz(treino), dados.matriz(teste)
    escores = {}
    for familia in familias:
        modelos = uplift.ajustar_todos(familia, x_treino, treino["tratado"].to_numpy(), treino[desfecho].to_numpy(), semente)
        for metodo, modelo in modelos.items():
            escores[chave(metodo, familia)] = modelo.prever_uplift(x_teste)
    return escores


def escolher_principal(treino: pd.DataFrame, desfecho: str, dobras: int = 5, semente: int = SEMENTE) -> pd.DataFrame:
    """Validação cruzada dentro do treino: coeficiente Qini médio de cada método de uplift.

    O método principal de cada campanha é o de maior Qini médio aqui. O teste não entra na
    escolha. Com cerca de 2 mil clientes por grupo em cada dobra, a ordem entre métodos parecidos
    é em boa parte sorteio; a tabela mostra o desvio entre dobras para deixar isso à vista.
    """
    dobra = dados.dobras_estratificadas(treino, dobras, semente)
    por_dobra: dict[str, list[float]] = {}
    for d in range(dobras):
        parte_treino, parte_validacao = treino[dobra != d], treino[dobra == d]
        escores = pontuar(parte_treino, parte_validacao, desfecho, semente=semente)
        y, t = parte_validacao[desfecho].to_numpy(), parte_validacao["tratado"].to_numpy()
        for nome, escore in escores.items():
            if nome.split("|")[0] in METODOS_DE_UPLIFT:
                por_dobra.setdefault(nome, []).append(curvas.coeficiente_qini(y, t, escore))
    saida = pd.DataFrame({
        "qini_medio": {nome: float(np.mean(v)) for nome, v in por_dobra.items()},
        "desvio_entre_dobras": {nome: float(np.std(v, ddof=1)) for nome, v in por_dobra.items()},
        "dobras_acima_do_acaso": {nome: int(np.sum(np.array(v) > 0)) for nome, v in por_dobra.items()},
    })
    return saida.sort_values("qini_medio", ascending=False)


# --------------------------------------------------------------------------- 4. avaliação no teste


def avaliar(
    teste: pd.DataFrame, desfecho: str, escores: dict[str, np.ndarray], base: str, repeticoes: int = 1000, semente: int = SEMENTE
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """Qini e AUUC de cada escore no teste, com intervalo de 95% por bootstrap e comparação pareada com a linha de base.

    `base` é a chave do modelo de resposta usado como comparação. Devolve a tabela e as
    repetições do bootstrap (para as curvas e para o valor de política).
    """
    y, t = teste[desfecho].to_numpy(), teste["tratado"].to_numpy()
    repeticoes_boot = curvas.bootstrap_das_curvas(y, t, escores, repeticoes=repeticoes, semente=semente, max_pontos=PONTOS_DO_BOOTSTRAP)
    linhas = {}
    for nome, escore in escores.items():
        qini = curvas.coeficiente_qini(y, t, escore)
        boot = repeticoes_boot[nome]
        diferenca = boot["coef_qini"] - repeticoes_boot[base]["coef_qini"]
        linhas[nome] = {
            "qini": qini,
            "qini_ic_inferior": curvas.intervalo(boot["coef_qini"])[0],
            "qini_ic_superior": curvas.intervalo(boot["coef_qini"])[1],
            "prob_acima_do_acaso": float(np.mean(boot["coef_qini"] > 0)),
            "auuc": curvas.auuc(y, t, escore),
            "auuc_ic_inferior": curvas.intervalo(boot["auuc"])[0],
            "auuc_ic_superior": curvas.intervalo(boot["auuc"])[1],
            "diferenca_para_a_base": qini - curvas.coeficiente_qini(y, t, escores[base]),
            "diferenca_ic_inferior": curvas.intervalo(diferenca)[0],
            "diferenca_ic_superior": curvas.intervalo(diferenca)[1],
            "prob_acima_da_base": float(np.mean(diferenca > 0)),
        }
    return pd.DataFrame(linhas).T, repeticoes_boot


def comparar(teste: pd.DataFrame, desfecho: str, escores: dict[str, np.ndarray], repeticoes_boot: dict, a: str, b: str) -> dict:
    """Diferença de coeficiente Qini entre dois escores (a − b), com intervalo de 95% pareado no bootstrap."""
    y, t = teste[desfecho].to_numpy(), teste["tratado"].to_numpy()
    diferenca = repeticoes_boot[a]["coef_qini"] - repeticoes_boot[b]["coef_qini"]
    inferior, superior = curvas.intervalo(diferenca)
    return {
        "a": a,
        "b": b,
        "diferenca": curvas.coeficiente_qini(y, t, escores[a]) - curvas.coeficiente_qini(y, t, escores[b]),
        "ic_inferior": inferior,
        "ic_superior": superior,
        "prob_a_maior": float(np.mean(diferenca > 0)),
    }


def curvas_no_teste(
    teste: pd.DataFrame, desfecho: str, escores: dict[str, np.ndarray], repeticoes_boot: dict, nomes: list[str]
) -> dict:
    """Curva Qini de cada escore pedido, na grade de 0 a 100%, com a faixa de 95% do bootstrap."""
    y, t = teste[desfecho].to_numpy(), teste["tratado"].to_numpy()
    grade = curvas.GRADE_PADRAO
    saida = {"fracao": grade.tolist(), "series": {}}
    for nome in nomes:
        fracao, ganho = curvas.curva_qini(y, t, escores[nome])
        boot = repeticoes_boot[nome]["qini"]
        saida["series"][nome] = {
            "ganho": np.interp(grade, fracao, ganho).tolist(),
            "inferior": np.quantile(boot, 0.025, axis=0).tolist(),
            "superior": np.quantile(boot, 0.975, axis=0).tolist(),
        }
    efeito = float(y[t == 1].mean() - y[t == 0].mean())
    saida["acaso"] = (grade * efeito).tolist()
    saida["efeito_medio_no_teste"] = efeito
    return saida


def valor_da_politica(
    teste: pd.DataFrame, desfecho: str, escores: dict[str, np.ndarray], repeticoes_boot: dict, nomes: list[str], fracoes: list[float] = FRACOES
) -> pd.DataFrame:
    """Respostas incrementais por mil clientes da base se o e-mail for só para a fração do topo de cada escore.

    Mandar ao acaso para a mesma fração rende essa fração do efeito médio; mandar para todos
    rende o efeito médio. A coluna `acima_do_acaso` é a diferença pareada no bootstrap.
    """
    y, t = teste[desfecho].to_numpy(), teste["tratado"].to_numpy()
    grade = curvas.GRADE_PADRAO
    efeito = float(y[t == 1].mean() - y[t == 0].mean())
    linhas = []
    for fracao in fracoes:
        posicao = int(np.argmin(np.abs(grade - fracao)))
        efeito_boot = repeticoes_boot[nomes[0]]["uplift"][:, -1]
        acaso_boot = fracao * efeito_boot
        linhas.append({
            "fracao": fracao, "escore": "acaso", "por_mil_clientes": 1000 * fracao * efeito,
            "ic_inferior": 1000 * curvas.intervalo(acaso_boot)[0], "ic_superior": 1000 * curvas.intervalo(acaso_boot)[1],
            "por_mil_emails": 1000 * efeito,
            "parte_do_efeito_total": fracao, "parte_ic_inferior": fracao, "parte_ic_superior": fracao,
            "acima_do_acaso": 0.0, "acima_ic_inferior": 0.0, "acima_ic_superior": 0.0, "prob_acima_do_acaso": float("nan"),
        })
        for nome in nomes:
            valor = float(curvas.ganho_na_fracao(*curvas.curva_uplift(y, t, escores[nome]), fracao))
            boot = repeticoes_boot[nome]["uplift"][:, posicao]
            diferenca = boot - acaso_boot
            parte = boot / efeito_boot
            linhas.append({
                "fracao": fracao, "escore": nome, "por_mil_clientes": 1000 * valor,
                "ic_inferior": 1000 * curvas.intervalo(boot)[0], "ic_superior": 1000 * curvas.intervalo(boot)[1],
                "por_mil_emails": 1000 * valor / fracao,
                "parte_do_efeito_total": valor / efeito,
                "parte_ic_inferior": curvas.intervalo(parte)[0], "parte_ic_superior": curvas.intervalo(parte)[1],
                "acima_do_acaso": 1000 * (valor - fracao * efeito),
                "acima_ic_inferior": 1000 * curvas.intervalo(diferenca)[0], "acima_ic_superior": 1000 * curvas.intervalo(diferenca)[1],
                "prob_acima_do_acaso": float(np.mean(diferenca > 0)) if fracao < 1 else float("nan"),
            })
    return pd.DataFrame(linhas).set_index(["fracao", "escore"])


def quem_cada_escore_escolhe(
    teste: pd.DataFrame, desfecho: str, escores: dict[str, np.ndarray], principal: str, base: str, campanha: str, fracao: float = 0.5
) -> pd.DataFrame:
    """Perfil do público de cada escolha: a taxa sem e-mail (quem viria de qualquer jeito), a taxa com e-mail e o uplift.

    Linhas: a base inteira, o topo (`fracao`) do modelo de resposta, o topo do modelo de uplift e
    o segmento da regra "já comprou a linha do e-mail" (no tamanho que ele tem). É a tabela que
    separa os dois modelos: o de resposta busca taxa alta; o de uplift busca diferença alta
    entre quem recebeu e quem não recebeu.
    """
    def perfil(recorte: pd.DataFrame, rotulo: str) -> dict:
        efeito = experimento.efeito_medio(recorte.loc[recorte["tratado"] == 1, desfecho], recorte.loc[recorte["tratado"] == 0, desfecho])
        return {
            "rotulo": rotulo,
            "clientes": int(len(recorte)),
            "parte_da_base": len(recorte) / len(teste),
            "taxa_sem_email": efeito["media_controles"],
            "taxa_com_email": efeito["media_tratados"],
            "uplift": efeito["efeito"],
            "uplift_ic_inferior": efeito["ic_inferior"],
            "uplift_ic_superior": efeito["ic_superior"],
            # Mandar só para este público rende o uplift dele vezes a parte da base que ele representa.
            "incrementais_por_mil_clientes": 1000 * efeito["efeito"] * len(recorte) / len(teste),
            "comprou_feminino": float(recorte["comprou_feminino"].mean()),
            "comprou_masculino": float(recorte["comprou_masculino"].mean()),
            "cliente_novo": float(recorte["cliente_novo"].mean()),
            "recencia_media": float(recorte["recencia"].mean()),
            "historico_mediano": float(recorte["historico"].median()),
        }

    tamanho = int(round(len(teste) * fracao))
    parte = f"{fracao * 100:.0f}%"
    linha_do_email = "feminina" if campanha == "E-mail feminino" else "masculina"
    segmento = teste[teste[LINHA_DA_CAMPANHA[campanha]] == 1]
    linhas = {
        "Base inteira": perfil(teste, "Base inteira (mandar para todos)"),
        "Modelo de resposta": perfil(
            teste.iloc[np.argsort(-escores[base], kind="stable")[:tamanho]], f"Modelo de resposta: os {parte} de maior escore"
        ),
        "Modelo de uplift": perfil(
            teste.iloc[np.argsort(-escores[principal], kind="stable")[:tamanho]], f"Modelo de uplift: os {parte} de maior uplift previsto"
        ),
        "Regra": perfil(segmento, f"Regra: já comprou a linha {linha_do_email} ({len(segmento) / len(teste) * 100:.0f}% da base)"),
    }
    saida = pd.DataFrame(linhas).T
    saida["parte_do_efeito_total"] = saida["incrementais_por_mil_clientes"] / saida.loc["Base inteira", "incrementais_por_mil_clientes"]
    return saida


def decis(teste: pd.DataFrame, desfecho: str, escore: np.ndarray) -> pd.DataFrame:
    """Uplift observado em cada décimo do teste, do maior para o menor uplift previsto."""
    return curvas.uplift_por_faixa(teste[desfecho].to_numpy(), teste["tratado"].to_numpy(), escore, faixas=10)


# --------------------------------------------------------------------------- 5. robustez


def placebo(
    recorte: pd.DataFrame, desfecho: str, principal: str, observado: float, repeticoes: int = 100, semente: int = SEMENTE
) -> dict:
    """Teste de placebo: embaralha quem "recebeu" o e-mail e refaz tudo (separação, treino, avaliação).

    Com o rótulo embaralhado não existe efeito a encontrar. Se o procedimento estivesse
    fabricando ganho (vazamento, avaliação viciada), o Qini continuaria positivo. O p-valor é a
    fração de placebos com Qini maior ou igual ao observado, contando o próprio observado.
    """
    metodo, familia = principal.split("|")
    gerador = np.random.default_rng(semente + 1)
    qinis, efeitos = [], []
    for _ in range(repeticoes):
        embaralhado = recorte.copy()
        embaralhado["tratado"] = gerador.permutation(recorte["tratado"].to_numpy())
        treino, teste = dados.separar_treino_teste(embaralhado, semente=semente)
        modelo = uplift.metodos(familia, semente)[metodo]
        modelo.ajustar(dados.matriz(treino), treino["tratado"].to_numpy(), treino[desfecho].to_numpy())
        y, t = teste[desfecho].to_numpy(), teste["tratado"].to_numpy()
        qinis.append(curvas.coeficiente_qini(y, t, modelo.prever_uplift(dados.matriz(teste))))
        efeitos.append(float(y[t == 1].mean() - y[t == 0].mean()))
    qinis = np.array(qinis)
    return {
        "repeticoes": repeticoes,
        "qini_observado": observado,
        "qini_placebo_medio": float(qinis.mean()),
        "qini_placebo_desvio": float(qinis.std(ddof=1)),
        "qini_placebo_p2_5": float(np.quantile(qinis, 0.025)),
        "qini_placebo_p97_5": float(np.quantile(qinis, 0.975)),
        "qini_placebo_maximo": float(qinis.max()),
        "efeito_medio_placebo_medio": float(np.mean(efeitos)),
        "p_valor": float((np.sum(qinis >= observado) + 1) / (repeticoes + 1)),
        "qinis": qinis.tolist(),
    }


def repetir_separacao(recorte: pd.DataFrame, campanha: str, desfecho: str, repeticoes: int = 20, semente: int = SEMENTE) -> pd.DataFrame:
    """Refaz a separação treino/teste com outras sementes e guarda o Qini de cada escore (família logística e regras).

    O bootstrap do teste não enxerga a sorte do treino. Aqui muda tudo junto: quem treina e quem
    avalia. Serve para ver se a ordem entre os métodos se mantém ou é sorteio.
    """
    linhas = []
    for r in range(repeticoes):
        treino, teste = dados.separar_treino_teste(recorte, semente=semente + 1000 + r)
        escores = pontuar(treino, teste, desfecho, familias=["logistica"], semente=semente) | regras(teste, campanha)
        y, t = teste[desfecho].to_numpy(), teste["tratado"].to_numpy()
        linhas.append({nome: curvas.coeficiente_qini(y, t, escore) for nome, escore in escores.items()})
    return pd.DataFrame(linhas)


def resumo_das_repeticoes(repeticoes: pd.DataFrame, base: str) -> pd.DataFrame:
    """Mediana e faixa do Qini entre as separações, e em quantas cada escore ficou acima do acaso e da linha de base."""
    return pd.DataFrame({
        "qini_mediano": repeticoes.median(),
        "qini_minimo": repeticoes.min(),
        "qini_maximo": repeticoes.max(),
        "vezes_acima_do_acaso": (repeticoes > 0).sum(),
        "vezes_acima_da_base": repeticoes.gt(repeticoes[base], axis=0).sum(),
        "separacoes": len(repeticoes),
    })


# --------------------------------------------------------------------------- uma campanha inteira


def analisar_campanha(tabela: pd.DataFrame, campanha: str, repeticoes_boot: int = 1000, repeticoes_placebo: int = 100) -> dict:
    """Roda a análise de uplift de uma campanha (e-mail contra nenhum e-mail) e devolve tudo o que o relatório usa.

    Desfecho principal: visita. A compra entra como desfecho secundário, com os mesmos métodos
    treinados nela e, a mais, o modelo principal treinado em visita usado para ordenar a compra.
    """
    recorte = dados.recorte_da_campanha(tabela, campanha)
    treino, teste = dados.separar_treino_teste(recorte, semente=SEMENTE)

    escolha = escolher_principal(treino, "visita")
    principal = str(escolha.index[0])
    familia = principal.split("|")[1]
    base = chave("resposta", familia)
    destaques = [principal, base, REGRA_LINHA, REGRA_RECENCIA]

    saida: dict = {
        "campanha": campanha,
        "separacao": {
            "treino": int(len(treino)),
            "teste": int(len(teste)),
            "tratados_no_teste": int(teste["tratado"].sum()),
            "controles_no_teste": int((1 - teste["tratado"]).sum()),
            "visitas_dos_tratados_no_teste": int(teste.loc[teste["tratado"] == 1, "visita"].sum()),
            "visitas_dos_controles_no_teste": int(teste.loc[teste["tratado"] == 0, "visita"].sum()),
            "compras_dos_tratados_no_teste": int(teste.loc[teste["tratado"] == 1, "conversao"].sum()),
            "compras_dos_controles_no_teste": int(teste.loc[teste["tratado"] == 0, "conversao"].sum()),
        },
        "escolha_do_principal": escolha,
        "principal": principal,
        "base": base,
    }

    escores_visita = pontuar(treino, teste, "visita") | regras(teste, campanha)
    for desfecho in ("visita", "conversao"):
        if desfecho == "visita":
            escores = escores_visita
        else:
            escores = pontuar(treino, teste, desfecho) | regras(teste, campanha)
            escores["principal_treinado_em_visita"] = escores_visita[principal]
        metodos, boot = avaliar(teste, desfecho, escores, base, repeticoes=repeticoes_boot)
        em_destaque = destaques + (["principal_treinado_em_visita"] if desfecho == "conversao" else [])
        bloco = {
            "metodos": metodos,
            "principal_contra_a_regra": comparar(teste, desfecho, escores, boot, principal, REGRA_LINHA),
            "principal_contra_a_resposta": comparar(teste, desfecho, escores, boot, principal, base),
            "curvas": curvas_no_teste(teste, desfecho, escores, boot, em_destaque),
            "politica": valor_da_politica(teste, desfecho, escores, boot, em_destaque),
            "quem_escolhe": {
                f"{fracao:.1f}": quem_cada_escore_escolhe(teste, desfecho, escores, principal, base, campanha, fracao) for fracao in (0.3, 0.5)
            },
            "decis": decis(teste, desfecho, escores[principal]),
        }
        if desfecho == "visita":
            # A família LightGBM custa dez vezes mais por ajuste; o placebo dela roda com menos repetições.
            n_placebo = repeticoes_placebo if familia == "logistica" else max(20, repeticoes_placebo // 5)
            bloco["placebo"] = placebo(recorte, desfecho, principal, float(metodos.loc[principal, "qini"]), repeticoes=n_placebo)
            repeticoes = repetir_separacao(recorte, campanha, desfecho)
            bloco["repeticoes_da_separacao"] = resumo_das_repeticoes(repeticoes, chave("resposta", "logistica"))
            bloco["qini_por_separacao"] = repeticoes
        saida[desfecho] = bloco
    return saida
