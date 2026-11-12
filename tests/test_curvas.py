import numpy as np
import pytest

from uplift_campanha import curvas


def _dois_segmentos() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """4.000 clientes em dois segmentos iguais, com contagens exatas (sem ruído).

    Segmento A (escore 1): 1.000 tratados com 300 respostas, 1.000 controles com 100 → uplift 0,20.
    Segmento B (escore 0): 1.000 tratados com 100 respostas, 1.000 controles com 100 → uplift 0.
    Efeito médio: 0,10.
    """
    def bloco(respostas_t: int, respostas_c: int) -> tuple[np.ndarray, np.ndarray]:
        t = np.repeat([1, 0], 1000)
        y = np.concatenate([np.repeat([1, 0], [respostas_t, 1000 - respostas_t]), np.repeat([1, 0], [respostas_c, 1000 - respostas_c])])
        return y, t

    ya, ta = bloco(300, 100)
    yb, tb = bloco(100, 100)
    return np.concatenate([ya, yb]), np.concatenate([ta, tb]), np.repeat([1.0, 0.0], 2000)


def _efeito_continuo(n: int, semente: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x uniforme em (0, 1); P(resposta) = 0,1 sem campanha e 0,1 + 0,5·x com campanha. Uplift verdadeiro: 0,5·x."""
    gerador = np.random.default_rng(semente)
    x = gerador.uniform(0, 1, n)
    t = gerador.integers(0, 2, n)
    y = (gerador.uniform(0, 1, n) < 0.1 + 0.5 * x * t).astype(int)
    return y, t, x


def test_escore_perfeito_em_dois_segmentos_da_os_pontos_exatos():
    y, t, escore = _dois_segmentos()
    fracao, ganho = curvas.curva_qini(y, t, escore)
    assert fracao == pytest.approx([0.0, 0.5, 1.0])
    assert ganho == pytest.approx([0.0, 0.10, 0.10])  # todo o ganho vem da primeira metade
    fracao_u, ganho_u = curvas.curva_uplift(y, t, escore)
    assert fracao_u == pytest.approx([0.0, 0.5, 1.0])
    assert ganho_u == pytest.approx([0.0, 0.20 * 0.5, 0.10])
    assert curvas.coeficiente_qini(y, t, escore) == pytest.approx(0.075 - 0.05)
    assert curvas.auuc(y, t, escore) == pytest.approx(0.075)


def test_escore_invertido_fica_abaixo_do_acaso_na_mesma_medida():
    y, t, escore = _dois_segmentos()
    fracao, ganho = curvas.curva_qini(y, t, -escore)
    assert ganho == pytest.approx([0.0, 0.0, 0.10])
    assert curvas.coeficiente_qini(y, t, -escore) == pytest.approx(-0.025)


def test_escore_constante_e_a_diagonal_e_o_qini_e_zero():
    y, t, _ = _dois_segmentos()
    fracao, ganho = curvas.curva_qini(y, t, np.zeros(len(y)))
    assert fracao == pytest.approx([0.0, 1.0])
    assert ganho == pytest.approx([0.0, 0.10])
    assert curvas.coeficiente_qini(y, t, np.zeros(len(y))) == pytest.approx(0.0)
    assert curvas.auuc(y, t, np.zeros(len(y))) == pytest.approx(0.05)  # metade do efeito médio


def test_efeito_constante_segue_a_diagonal_mesmo_com_escore_que_preve_a_resposta():
    # A taxa de base varia muito com x, mas a campanha soma 0,10 para todo mundo.
    # Ordenar por x (um ótimo modelo de resposta) não pode render ganho sobre o acaso.
    gerador = np.random.default_rng(1)
    n = 200_000
    x = gerador.uniform(0, 1, n)
    t = gerador.integers(0, 2, n)
    y = (gerador.uniform(0, 1, n) < 0.05 + 0.6 * x + 0.10 * t).astype(int)
    fracao, ganho = curvas.curva_qini(y, t, x)
    assert np.max(np.abs(ganho - 0.10 * fracao)) < 0.006
    assert curvas.coeficiente_qini(y, t, x) == pytest.approx(0.0, abs=0.002)
    fracao_u, ganho_u = curvas.curva_uplift(y, t, x)
    longe_do_inicio = fracao_u > 0.02  # nos primeiros clientes a taxa é ruído puro
    assert np.max(np.abs(ganho_u - 0.10 * fracao_u)[longe_do_inicio]) < 0.006


def test_efeito_conhecido_reproduz_a_curva_e_as_areas_teoricas():
    # Uplift verdadeiro 0,5·x. Com o escore perfeito, o ganho na fração f é 0,5·(f − f²/2);
    # a AUUC é 1/6 e o Qini (área acima da diagonal) é 1/24.
    y, t, x = _efeito_continuo(400_000)
    fracao, ganho = curvas.curva_qini(y, t, x)
    esperado = 0.5 * (fracao - fracao**2 / 2)
    assert np.max(np.abs(ganho - esperado)) < 0.006
    assert curvas.coeficiente_qini(y, t, x) == pytest.approx(1 / 24, abs=0.002)
    assert curvas.auuc(y, t, x) == pytest.approx(1 / 6, abs=0.002)
    assert curvas.coeficiente_qini(y, t, -x) == pytest.approx(-1 / 24, abs=0.002)


def test_as_duas_curvas_terminam_no_efeito_medio():
    y, t, x = _efeito_continuo(5_000, semente=3)
    efeito = y[t == 1].mean() - y[t == 0].mean()
    assert curvas.curva_qini(y, t, x)[1][-1] == pytest.approx(efeito)
    assert curvas.curva_uplift(y, t, x)[1][-1] == pytest.approx(efeito)


def test_transformacao_monotona_do_escore_nao_muda_a_curva():
    y, t, x = _efeito_continuo(5_000, semente=4)
    assert curvas.coeficiente_qini(y, t, x) == pytest.approx(curvas.coeficiente_qini(y, t, np.exp(3 * x) + 7))


def test_empates_viram_um_ponto_por_valor_distinto():
    y, t, x = _efeito_continuo(5_000, semente=5)
    em_classes = np.floor(x * 4)  # quatro valores distintos
    fracao, ganho = curvas.curva_qini(y, t, em_classes)
    assert len(fracao) == 5
    assert np.all(np.diff(fracao) > 0)
    # A ordem das linhas dentro do empate não pode mudar o resultado.
    troca = np.random.default_rng(0).permutation(len(y))
    assert curvas.coeficiente_qini(y[troca], t[troca], em_classes[troca]) == pytest.approx(curvas.coeficiente_qini(y, t, em_classes))


def test_pesos_inteiros_equivalem_a_repetir_as_linhas():
    y, t, x = _efeito_continuo(2_000, semente=6)
    pesos = np.random.default_rng(6).integers(0, 4, len(y))
    repetido = np.repeat(np.arange(len(y)), pesos)
    assert curvas.coeficiente_qini(y, t, x, pesos) == pytest.approx(curvas.coeficiente_qini(y[repetido], t[repetido], x[repetido]))
    assert curvas.auuc(y, t, x, pesos) == pytest.approx(curvas.auuc(y[repetido], t[repetido], x[repetido]))


def test_ganho_na_fracao_interpola_entre_os_pontos():
    y, t, escore = _dois_segmentos()
    fracao, ganho = curvas.curva_qini(y, t, escore)
    assert curvas.ganho_na_fracao(fracao, ganho, 0.25) == pytest.approx(0.05)
    assert curvas.ganho_na_fracao(fracao, ganho, [0.5, 0.75]) == pytest.approx([0.10, 0.10])


def test_uplift_por_faixa_recupera_o_uplift_de_cada_segmento():
    y, t, escore = _dois_segmentos()
    faixas = curvas.uplift_por_faixa(y, t, escore, faixas=2)
    assert list(faixas["clientes"]) == [2000, 2000]
    assert faixas["uplift"].to_numpy() == pytest.approx([0.20, 0.0])
    assert faixas.loc[1, "taxa_com_campanha"] == pytest.approx(0.30)
    assert faixas.loc[1, "taxa_sem_campanha"] == pytest.approx(0.10)
    erro = np.sqrt(0.3 * 0.7 / 999 + 0.1 * 0.9 / 999)
    assert faixas.loc[1, "ic95_superior"] - faixas.loc[1, "uplift"] == pytest.approx(1.96 * erro)


def test_bootstrap_cobre_o_valor_verdadeiro_e_e_reprodutivel():
    y, t, x = _efeito_continuo(6_000, semente=7)
    escores = {"perfeito": x, "sem_informacao": np.random.default_rng(70).uniform(0, 1, len(x))}
    boot = curvas.bootstrap_das_curvas(y, t, escores, repeticoes=300, semente=11)
    inferior, superior = curvas.intervalo(boot["perfeito"]["coef_qini"])
    assert inferior < 1 / 24 < superior
    inferior_nulo, superior_nulo = curvas.intervalo(boot["sem_informacao"]["coef_qini"])
    assert inferior_nulo < 0 < superior_nulo
    # A comparação pareada separa os dois escores com folga.
    assert np.mean(boot["perfeito"]["coef_qini"] > boot["sem_informacao"]["coef_qini"]) > 0.99
    # Formato das curvas e ponto final igual ao efeito médio de cada reamostra, para os dois escores.
    assert boot["perfeito"]["qini"].shape == (300, len(curvas.GRADE_PADRAO))
    assert boot["perfeito"]["qini"][:, -1] == pytest.approx(boot["sem_informacao"]["qini"][:, -1])
    assert boot["perfeito"]["qini"][:, 0] == pytest.approx(0.0)
    de_novo = curvas.bootstrap_das_curvas(y, t, escores, repeticoes=300, semente=11)
    assert de_novo["perfeito"]["coef_qini"] == pytest.approx(boot["perfeito"]["coef_qini"])


@pytest.mark.parametrize("classes", [None, 5])
def test_bootstrap_reproduz_a_conta_direta_com_pesos(classes):
    y, t, x = _efeito_continuo(1_500, semente=9)
    escore = x if classes is None else np.floor(x * classes)  # sem empates e com empates
    boot = curvas.bootstrap_das_curvas(y, t, {"a": escore}, repeticoes=3, semente=2)["a"]
    # Refaz a primeira reamostra com o mesmo gerador e confere contra a função de ponto, com pesos.
    gerador = np.random.default_rng(2)
    pesos = np.zeros(len(y))
    for indices in (np.flatnonzero(t == 1), np.flatnonzero(t == 0)):
        pesos[indices] = np.bincount(gerador.integers(0, len(indices), len(indices)), minlength=len(indices))
    assert boot["coef_qini"][0] == pytest.approx(curvas.coeficiente_qini(y, t, escore, pesos))
    assert boot["auuc"][0] == pytest.approx(curvas.auuc(y, t, escore, pesos))
    fracao, ganho = curvas.curva_qini(y, t, escore, pesos)
    assert boot["qini"][0] == pytest.approx(np.interp(curvas.GRADE_PADRAO, fracao, ganho))


def test_bootstrap_em_menos_pontos_quase_nao_muda_as_areas_nem_a_faixa():
    y, t, x = _efeito_continuo(20_000, semente=8)
    completo = curvas.bootstrap_das_curvas(y, t, {"a": x}, repeticoes=200, semente=5)["a"]
    resumido = curvas.bootstrap_das_curvas(y, t, {"a": x}, repeticoes=200, semente=5, max_pontos=2000)["a"]
    for area in ("coef_qini", "auuc"):
        erro_amostral = completo[area].std(ddof=1)
        assert np.max(np.abs(completo[area] - resumido[area])) < 0.01 * erro_amostral
    # Limites da faixa de 95% em cada ponto da grade, comparados ao erro amostral daquele ponto.
    meio = slice(5, -1)  # de 5% a 99% da base; nas pontas o erro amostral é quase zero
    desvio = completo["qini"].std(axis=0, ddof=1)[meio]
    for quantil in (0.025, 0.975):
        diferenca = np.abs(np.quantile(completo["qini"], quantil, axis=0) - np.quantile(resumido["qini"], quantil, axis=0))[meio]
        assert np.max(diferenca / desvio) < 0.15


def test_blocos_resumidos_nao_separam_empates():
    escore = np.repeat(np.arange(50.0), 20)  # 50 valores distintos, 20 clientes em cada
    bloco, n_blocos = curvas._bloco_de_cada_cliente(curvas.ordenar(escore), max_pontos=7)
    assert n_blocos == 7
    assert all(len(set(bloco[escore == valor])) == 1 for valor in np.unique(escore))
    assert bloco[escore == 49.0][0] == 0 and bloco[escore == 0.0][0] == 6  # maior escore no primeiro bloco
    todos, n_todos = curvas._bloco_de_cada_cliente(curvas.ordenar(escore))
    assert n_todos == 50


def test_bootstrap_tem_a_dispersao_de_amostras_novas():
    # O desvio do Qini entre reamostras deve ser parecido com o desvio entre amostras independentes.
    qinis = [curvas.coeficiente_qini(*_efeito_continuo(3_000, semente=100 + s)) for s in range(120)]
    y, t, x = _efeito_continuo(3_000, semente=99)
    boot = curvas.bootstrap_das_curvas(y, t, {"perfeito": x}, repeticoes=300, semente=1)
    razao = boot["perfeito"]["coef_qini"].std(ddof=1) / np.std(qinis, ddof=1)
    assert 0.75 < razao < 1.3
