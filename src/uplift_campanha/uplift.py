"""Modelos de uplift (efeito heterogêneo da campanha) montados sobre classificadores comuns.

Nenhum cliente é visto com e sem a campanha, então não existe um alvo "uplift" para treinar
direto. Os quatro métodos abaixo contornam isso de jeitos diferentes; todos devolvem, para cada
cliente, uma estimativa de P(resposta | com campanha) − P(resposta | sem campanha).

Os modelos de base entram por uma `fabrica`: uma função sem argumentos que devolve um estimador
novo com `fit(X, y, sample_weight=None)` e `predict_proba(X)` (classificador) ou `predict(X)`
(regressor). Assim o mesmo método roda com regressão logística ou com LightGBM.

Convenção: `tratado` é 0/1, `resposta` é 0/1, `X` é uma matriz NumPy só com atributos anteriores
à campanha.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression, Ridge

Fabrica = Callable[[], object]


# --------------------------------------------------------------------------- modelos de base


class LogisticaPadronizada:
    """Regressão logística com os atributos postos em média 0 e desvio 1 (a penalização L2 exige escala comum)."""

    def __init__(self, c: float = 1.0):
        self.c = c

    def _padronizar(self, X: np.ndarray) -> np.ndarray:
        return (X - self.media_) / self.desvio_

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "LogisticaPadronizada":
        X = np.asarray(X, dtype=float)
        self.media_ = X.mean(axis=0)
        desvio = X.std(axis=0)
        self.desvio_ = np.where(desvio > 0, desvio, 1.0)
        self.modelo_ = LogisticRegression(C=self.c, max_iter=2000).fit(self._padronizar(X), y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.modelo_.predict_proba(self._padronizar(np.asarray(X, dtype=float)))


class RidgePadronizada:
    """Regressão linear com penalização L2 sobre atributos padronizados (segundo estágio do X-learner)."""

    def __init__(self, alfa: float = 1.0):
        self.alfa = alfa

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "RidgePadronizada":
        X = np.asarray(X, dtype=float)
        self.media_ = X.mean(axis=0)
        desvio = X.std(axis=0)
        self.desvio_ = np.where(desvio > 0, desvio, 1.0)
        self.modelo_ = Ridge(alpha=self.alfa).fit((X - self.media_) / self.desvio_, y, sample_weight=sample_weight)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.modelo_.predict((np.asarray(X, dtype=float) - self.media_) / self.desvio_)


# Árvores rasas, folhas grandes e taxa de aprendizado baixa: o efeito da campanha é pequeno perto do
# ruído, e uma árvore funda decora diferenças entre tratados e controles que não se repetem.
# Valores fixados antes de olhar o teste; não foram ajustados depois.
PARAMETROS_LGBM = {
    "n_estimators": 150,
    "learning_rate": 0.05,
    "num_leaves": 8,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "n_jobs": 1,
    "verbose": -1,
}


def fabricas(familia: str, semente: int = 42) -> tuple[Fabrica, Fabrica]:
    """Fábricas de (classificador, regressor) de uma família: "logistica" ou "lightgbm"."""
    if familia == "logistica":
        return (lambda: LogisticaPadronizada()), (lambda: RidgePadronizada())
    if familia == "lightgbm":
        return (
            lambda: LGBMClassifier(**PARAMETROS_LGBM, random_state=semente),
            lambda: LGBMRegressor(**PARAMETROS_LGBM, random_state=semente),
        )
    raise ValueError(f"família desconhecida: {familia}")


def _probabilidade(modelo, X: np.ndarray) -> np.ndarray:
    return modelo.predict_proba(X)[:, 1]


def _checar(X, tratado, resposta) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, t, y = np.asarray(X, dtype=float), np.asarray(tratado).astype(int), np.asarray(resposta).astype(float)
    if not (len(X) == len(t) == len(y)):
        raise ValueError("X, tratado e resposta precisam ter o mesmo número de linhas")
    if set(np.unique(t)) != {0, 1}:
        raise ValueError("tratado precisa ter os dois grupos, codificados como 0 e 1")
    return X, t, y


# --------------------------------------------------------------------------- métodos de uplift


class DoisModelos:
    """T-learner: um modelo de resposta só com quem recebeu, outro só com quem não recebeu.

    Uplift = previsão do primeiro − previsão do segundo. Simples e sem viés de forma, mas cada
    modelo erra por conta própria e a diferença soma os dois erros.
    """

    nome = "Dois modelos (T-learner)"

    def __init__(self, fabrica: Fabrica):
        self.fabrica = fabrica

    def ajustar(self, X, tratado, resposta) -> "DoisModelos":
        X, t, y = _checar(X, tratado, resposta)
        self.modelo_tratado_ = self.fabrica().fit(X[t == 1], y[t == 1])
        self.modelo_controle_ = self.fabrica().fit(X[t == 0], y[t == 0])
        return self

    def prever_uplift(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return _probabilidade(self.modelo_tratado_, X) - _probabilidade(self.modelo_controle_, X)


class ClasseTransformada:
    """Transformação de classe (Jaśkowski e Jaroszewicz, 2012): um modelo só, com alvo trocado.

    Alvo Z = 1 para quem recebeu e respondeu ou não recebeu e não respondeu; 0 para os outros.
    Com os dois grupos pesando igual, P(Z = 1 | x) = ½ · [1 + uplift(x)], logo
    uplift(x) = 2 · P(Z = 1 | x) − 1. Quando o experimento não é meio a meio, cada grupo entra
    com peso inverso ao seu tamanho, o que devolve a igualdade.
    """

    nome = "Transformação de classe"

    def __init__(self, fabrica: Fabrica):
        self.fabrica = fabrica

    def ajustar(self, X, tratado, resposta) -> "ClasseTransformada":
        X, t, y = _checar(X, tratado, resposta)
        z = (t * y + (1 - t) * (1 - y)).astype(int)
        proporcao = t.mean()
        pesos = np.where(t == 1, 0.5 / proporcao, 0.5 / (1.0 - proporcao))
        self.modelo_ = self.fabrica().fit(X, z, sample_weight=pesos)
        return self

    def prever_uplift(self, X) -> np.ndarray:
        return 2.0 * _probabilidade(self.modelo_, np.asarray(X, dtype=float)) - 1.0


class ModeloUnicoComInteracoes:
    """S-learner com interações: um modelo com os atributos, o tratamento e atributo × tratamento.

    Uplift = previsão com tratamento ligado − previsão com tratamento desligado. As interações
    explícitas são o que permite a um modelo linear ter efeito diferente por cliente; sem elas,
    o modelo logístico daria a todos o mesmo efeito na escala do logito.
    """

    nome = "Modelo único com interações (S-learner)"

    def __init__(self, fabrica: Fabrica):
        self.fabrica = fabrica

    @staticmethod
    def _montar(X: np.ndarray, t: np.ndarray) -> np.ndarray:
        t = t.reshape(-1, 1).astype(float)
        return np.hstack([X, t, X * t])

    def ajustar(self, X, tratado, resposta) -> "ModeloUnicoComInteracoes":
        X, t, y = _checar(X, tratado, resposta)
        self.modelo_ = self.fabrica().fit(self._montar(X, t), y)
        return self

    def prever_uplift(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        com = _probabilidade(self.modelo_, self._montar(X, np.ones(len(X))))
        sem = _probabilidade(self.modelo_, self._montar(X, np.zeros(len(X))))
        return com - sem


class XLearner:
    """X-learner (Künzel e outros, 2019): dois estágios.

    1. Como no T-learner, um modelo de resposta por grupo.
    2. Para cada tratado, o efeito imputado é a resposta dele menos o que o modelo dos controles
       previa para ele; para cada controle, o que o modelo dos tratados previa menos a resposta
       dele. Dois regressores aprendem esses efeitos imputados a partir dos atributos.
    A previsão final é a média dos dois regressores, ponderada pela proporção de tratados (num
    experimento aleatorizado, a chance de receber é a mesma para todo mundo).
    """

    nome = "X-learner"

    def __init__(self, fabrica: Fabrica, fabrica_regressor: Fabrica):
        self.fabrica = fabrica
        self.fabrica_regressor = fabrica_regressor

    def ajustar(self, X, tratado, resposta, primeiro_estagio: DoisModelos | None = None) -> "XLearner":
        """`primeiro_estagio`: um T-learner já ajustado nos mesmos dados, para não treinar os mesmos dois modelos de novo."""
        X, t, y = _checar(X, tratado, resposta)
        if primeiro_estagio is None:
            primeiro_estagio = DoisModelos(self.fabrica).ajustar(X, t, y)
        modelo_tratado, modelo_controle = primeiro_estagio.modelo_tratado_, primeiro_estagio.modelo_controle_
        efeito_nos_tratados = y[t == 1] - _probabilidade(modelo_controle, X[t == 1])
        efeito_nos_controles = _probabilidade(modelo_tratado, X[t == 0]) - y[t == 0]
        self.efeito_tratados_ = self.fabrica_regressor().fit(X[t == 1], efeito_nos_tratados)
        self.efeito_controles_ = self.fabrica_regressor().fit(X[t == 0], efeito_nos_controles)
        self.proporcao_ = float(t.mean())
        return self

    def prever_uplift(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        g = self.proporcao_
        return g * self.efeito_controles_.predict(X) + (1.0 - g) * self.efeito_tratados_.predict(X)


# --------------------------------------------------------------------------- linhas de base


class ModeloDeResposta:
    """Linha de base: o modelo de propensão comum, treinado só com quem recebeu a campanha.

    Escore = P(resposta | recebeu). Ordena por quem mais responde, não por quem muda de
    comportamento: é o modelo que quase toda operação de CRM usa para escolher o público.
    """

    nome = "Modelo de resposta"

    def __init__(self, fabrica: Fabrica):
        self.fabrica = fabrica

    def ajustar(self, X, tratado, resposta, primeiro_estagio: DoisModelos | None = None) -> "ModeloDeResposta":
        """`primeiro_estagio`: um T-learner já ajustado nos mesmos dados; o modelo dos tratados dele é este mesmo modelo."""
        X, t, y = _checar(X, tratado, resposta)
        self.modelo_ = primeiro_estagio.modelo_tratado_ if primeiro_estagio is not None else self.fabrica().fit(X[t == 1], y[t == 1])
        return self

    def prever_uplift(self, X) -> np.ndarray:
        """Devolve o escore de resposta (não é um uplift; só serve para ordenar)."""
        return _probabilidade(self.modelo_, np.asarray(X, dtype=float))


def metodos(familia: str, semente: int = 42) -> dict[str, object]:
    """Os quatro métodos de uplift e o modelo de resposta, todos sobre a mesma família de modelo de base."""
    classificador, regressor = fabricas(familia, semente)
    return {
        "dois_modelos": DoisModelos(classificador),
        "classe_transformada": ClasseTransformada(classificador),
        "modelo_unico": ModeloUnicoComInteracoes(classificador),
        "x_learner": XLearner(classificador, regressor),
        "resposta": ModeloDeResposta(classificador),
    }


def ajustar_todos(familia: str, X, tratado, resposta, semente: int = 42) -> dict[str, object]:
    """Ajusta os quatro métodos e o modelo de resposta de uma família, treinando cada modelo de base uma vez só.

    O modelo dos tratados e o dos controles do T-learner são os mesmos do primeiro estágio do
    X-learner, e o dos tratados é o próprio modelo de resposta: são reaproveitados. O resultado é
    idêntico ao de ajustar cada método sozinho, com seis treinos em vez de nove.
    """
    modelos = metodos(familia, semente)
    dois = modelos["dois_modelos"].ajustar(X, tratado, resposta)
    modelos["classe_transformada"].ajustar(X, tratado, resposta)
    modelos["modelo_unico"].ajustar(X, tratado, resposta)
    modelos["x_learner"].ajustar(X, tratado, resposta, primeiro_estagio=dois)
    modelos["resposta"].ajustar(X, tratado, resposta, primeiro_estagio=dois)
    return modelos


NOMES = {
    "dois_modelos": DoisModelos.nome,
    "classe_transformada": ClasseTransformada.nome,
    "modelo_unico": ModeloUnicoComInteracoes.nome,
    "x_learner": XLearner.nome,
    "resposta": ModeloDeResposta.nome,
}
FAMILIAS = {"logistica": "regressão logística", "lightgbm": "LightGBM"}
