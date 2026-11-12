import numpy as np
import pytest

from uplift_campanha import curvas, uplift


def _sigmoide(v: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-v))


def _campanha_sintetica(n: int = 20_000, semente: int = 0, proporcao: float = 0.5, efeito_constante: bool = False):
    """Experimento sintético: x0 manda na taxa de base, x1 manda no efeito da campanha.

    Sem campanha, P(resposta) = sigmoide(−1,5 + x0). Com campanha, soma 0,25 para quem tem
    x1 > 0 e nada para os demais (ou 0,125 para todos, se `efeito_constante`).
    """
    gerador = np.random.default_rng(semente)
    X = gerador.normal(size=(n, 4))
    t = (gerador.uniform(size=n) < proporcao).astype(int)
    efeito = np.full(n, 0.125) if efeito_constante else 0.25 * (X[:, 1] > 0)
    y = (gerador.uniform(size=n) < _sigmoide(-1.5 + X[:, 0]) + t * efeito).astype(int)
    return X, t, y, efeito


@pytest.fixture(scope="module")
def treino_e_teste():
    return _campanha_sintetica(semente=0), _campanha_sintetica(semente=1)


@pytest.mark.parametrize("metodo", ["dois_modelos", "classe_transformada", "modelo_unico", "x_learner"])
def test_cada_metodo_com_logistica_encontra_quem_responde_a_campanha(treino_e_teste, metodo):
    (X, t, y, _), (X_teste, t_teste, y_teste, efeito_teste) = treino_e_teste
    modelo = uplift.metodos("logistica")[metodo].ajustar(X, t, y)
    previsto = modelo.prever_uplift(X_teste)
    # Ordena certo: o uplift previsto é maior onde o efeito verdadeiro existe.
    assert previsto[efeito_teste > 0].mean() - previsto[efeito_teste == 0].mean() > 0.10
    # Na escala certa: a média do uplift previsto fica perto do efeito médio verdadeiro (0,125).
    assert previsto.mean() == pytest.approx(0.125, abs=0.03)
    # Captura boa parte do Qini do escore perfeito.
    ideal = curvas.coeficiente_qini(y_teste, t_teste, efeito_teste)
    assert curvas.coeficiente_qini(y_teste, t_teste, previsto) > 0.7 * ideal


@pytest.mark.parametrize("metodo", ["dois_modelos", "classe_transformada", "modelo_unico", "x_learner"])
def test_cada_metodo_com_lightgbm_encontra_quem_responde_a_campanha(metodo):
    X, t, y, _ = _campanha_sintetica(n=6_000, semente=2)
    X_teste, t_teste, y_teste, efeito_teste = _campanha_sintetica(n=6_000, semente=3)
    previsto = uplift.metodos("lightgbm")[metodo].ajustar(X, t, y).prever_uplift(X_teste)
    assert previsto[efeito_teste > 0].mean() - previsto[efeito_teste == 0].mean() > 0.10
    assert curvas.coeficiente_qini(y_teste, t_teste, previsto) > 0.6 * curvas.coeficiente_qini(y_teste, t_teste, efeito_teste)


def test_modelo_de_resposta_ordena_pela_taxa_de_base_e_nao_pelo_efeito(treino_e_teste):
    (X, t, y, _), (X_teste, t_teste, y_teste, efeito_teste) = treino_e_teste
    resposta = uplift.metodos("logistica")["resposta"].ajustar(X, t, y).prever_uplift(X_teste)
    dois = uplift.metodos("logistica")["dois_modelos"].ajustar(X, t, y).prever_uplift(X_teste)
    # O escore de resposta acompanha x0 (taxa de base) muito mais do que x1 (efeito).
    assert abs(np.corrcoef(resposta, X_teste[:, 0])[0, 1]) > 0.8
    assert curvas.coeficiente_qini(y_teste, t_teste, dois) > 1.5 * curvas.coeficiente_qini(y_teste, t_teste, resposta)


def test_modelo_de_resposta_fica_abaixo_do_acaso_quando_quem_mais_compra_e_quem_menos_muda():
    # Quem tem x0 alto compra de qualquer jeito e a campanha não muda nada; o efeito está em quem tem x0 baixo.
    gerador = np.random.default_rng(5)
    n = 20_000
    X = gerador.normal(size=(n, 3))
    t = gerador.integers(0, 2, n)
    y = (gerador.uniform(size=n) < np.where(X[:, 0] > 0, 0.6, 0.05 + 0.25 * t)).astype(int)
    metade = n // 2
    modelos = uplift.metodos("logistica")
    resposta = modelos["resposta"].ajustar(X[:metade], t[:metade], y[:metade]).prever_uplift(X[metade:])
    dois = modelos["dois_modelos"].ajustar(X[:metade], t[:metade], y[:metade]).prever_uplift(X[metade:])
    assert curvas.coeficiente_qini(y[metade:], t[metade:], resposta) < -0.01
    assert curvas.coeficiente_qini(y[metade:], t[metade:], dois) > 0.01


def test_transformacao_de_classe_corrige_experimento_desbalanceado():
    # Com 30% de tratados, sem os pesos a média de 2·P(Z = 1) − 1 ficaria longe do efeito médio.
    X, t, y, _ = _campanha_sintetica(semente=6, proporcao=0.3)
    previsto = uplift.metodos("logistica")["classe_transformada"].ajustar(X, t, y).prever_uplift(X)
    assert previsto.mean() == pytest.approx(0.125, abs=0.03)
    z = t * y + (1 - t) * (1 - y)
    sem_pesos = 2 * uplift.LogisticaPadronizada().fit(X, z).predict_proba(X)[:, 1] - 1
    assert abs(sem_pesos.mean() - 0.125) > 0.10


def test_efeito_constante_da_previsoes_quase_iguais_para_todos():
    X, t, y, _ = _campanha_sintetica(semente=7, efeito_constante=True)
    previsto = uplift.metodos("logistica")["x_learner"].ajustar(X, t, y).prever_uplift(X)
    assert previsto.mean() == pytest.approx(0.125, abs=0.02)
    assert previsto.std() < 0.03


@pytest.mark.parametrize("familia", ["logistica", "lightgbm"])
def test_ajustar_todos_da_o_mesmo_resultado_de_ajustar_cada_metodo_sozinho(familia):
    X, t, y, _ = _campanha_sintetica(n=3_000, semente=9)
    X_novo = _campanha_sintetica(n=500, semente=10)[0]
    juntos = uplift.ajustar_todos(familia, X, t, y)
    for nome, sozinho in uplift.metodos(familia).items():
        assert juntos[nome].prever_uplift(X_novo) == pytest.approx(sozinho.ajustar(X, t, y).prever_uplift(X_novo), abs=1e-12)


def test_modelo_unico_monta_atributos_tratamento_e_interacoes():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    montado = uplift.ModeloUnicoComInteracoes._montar(X, np.array([1, 0]))
    assert montado.tolist() == [[1.0, 2.0, 1.0, 1.0, 2.0], [3.0, 4.0, 0.0, 0.0, 0.0]]


def test_logistica_padronizada_nao_depende_da_escala_do_atributo():
    X, t, y, _ = _campanha_sintetica(n=3_000, semente=8)
    em_outra_escala = X * np.array([1000.0, 0.001, 1.0, 50.0])
    a = uplift.LogisticaPadronizada().fit(X, y).predict_proba(X)[:, 1]
    b = uplift.LogisticaPadronizada().fit(em_outra_escala, y).predict_proba(em_outra_escala)[:, 1]
    assert a == pytest.approx(b, abs=1e-4)


def test_entrada_invalida_e_recusada():
    X = np.zeros((4, 2))
    with pytest.raises(ValueError):
        uplift.metodos("logistica")["dois_modelos"].ajustar(X, np.array([1, 1, 1, 1]), np.array([0, 1, 0, 1]))
    with pytest.raises(ValueError):
        uplift.metodos("logistica")["dois_modelos"].ajustar(X, np.array([1, 0, 1]), np.array([0, 1, 0]))
    with pytest.raises(ValueError):
        uplift.fabricas("rede_neural")
