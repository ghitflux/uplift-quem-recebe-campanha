"""Curva Qini, curva de uplift, áreas, uplift por faixa e bootstrap. Tudo em NumPy.

A ideia comum: ordenar os clientes do maior para o menor escore e perguntar, para cada fração
do topo, quantas respostas a mais o grupo que recebeu a campanha teve em relação ao grupo que
não recebeu. Ninguém é observado nas duas situações, então o ganho só existe em grupo: é uma
diferença entre tratados e controles que caíram na mesma faixa de escore.

Unidade das duas curvas: respostas incrementais por cliente da base. Mandar a campanha para a
fração `f` do topo rende `ganho(f)` respostas a mais por cliente da base inteira; em `f = 1` as
duas curvas valem o efeito médio da campanha. Multiplicado por mil, é "respostas incrementais
por mil clientes".

Empates no escore (uma regra como "meses desde a última compra" só tem 12 valores) entram como
um bloco só: a curva tem um ponto por valor distinto e é reta dentro do bloco, que é o valor
esperado de desempatar no sorteio. Um escore constante dá, por construção, a diagonal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

GRADE_PADRAO = np.linspace(0.0, 1.0, 101)


@dataclass(frozen=True)
class Ordenacao:
    """Ordem decrescente de um escore e a última posição de cada bloco de empate."""

    ordem: np.ndarray
    limites: np.ndarray


def ordenar(escore: np.ndarray) -> Ordenacao:
    """Ordena do maior para o menor escore, guardando onde termina cada bloco de valores iguais."""
    escore = np.asarray(escore, dtype=float)
    ordem = np.argsort(-escore, kind="stable")
    ordenado = escore[ordem]
    limites = np.append(np.flatnonzero(np.diff(ordenado) != 0), len(ordenado) - 1)
    return Ordenacao(ordem=ordem, limites=limites)


def _acumulados(
    resposta: np.ndarray, tratado: np.ndarray, ordenacao: Ordenacao, pesos: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tratados, controles e respostas de cada grupo, acumulados do topo até o fim de cada bloco de empate."""
    y = np.asarray(resposta, dtype=float)[ordenacao.ordem]
    t = np.asarray(tratado, dtype=float)[ordenacao.ordem]
    w = np.ones_like(y) if pesos is None else np.asarray(pesos, dtype=float)[ordenacao.ordem]
    wt = w * t
    wc = w - wt
    lim = ordenacao.limites
    return np.cumsum(wt)[lim], np.cumsum(wc)[lim], np.cumsum(wt * y)[lim], np.cumsum(wc * y)[lim]


def _dividir(numerador: np.ndarray, denominador: np.ndarray) -> np.ndarray:
    """Divisão que devolve zero onde o denominador é zero (topo ainda sem ninguém de um dos grupos)."""
    return np.divide(numerador, denominador, out=np.zeros_like(numerador, dtype=float), where=denominador > 0)


def _com_origem(valores: np.ndarray) -> np.ndarray:
    """Acrescenta o ponto zero no começo da última dimensão."""
    return np.concatenate([np.zeros(valores.shape[:-1] + (1,)), valores], axis=-1)


def _fracao(nt: np.ndarray, nc: np.ndarray) -> np.ndarray:
    return _com_origem((nt + nc) / (nt[..., -1:] + nc[..., -1:]))


def _qini(nt: np.ndarray, nc: np.ndarray, yt: np.ndarray, yc: np.ndarray) -> np.ndarray:
    return _com_origem((yt - yc * _dividir(nt, nc)) / nt[..., -1:])


def _uplift(nt: np.ndarray, nc: np.ndarray, yt: np.ndarray, yc: np.ndarray) -> np.ndarray:
    return _com_origem((_dividir(yt, nt) - _dividir(yc, nc)) * (nt + nc) / (nt[..., -1:] + nc[..., -1:]))


def curva_qini(
    resposta: np.ndarray, tratado: np.ndarray, escore: np.ndarray, pesos: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Curva Qini (Radcliffe, 2007): respostas incrementais acumuladas, por tratado da base.

    Em cada ponto: respostas dos tratados no topo menos as respostas dos controles no topo,
    reescaladas para o mesmo número de pessoas (`n_tratados / n_controles` do topo), dividido
    pelo total de tratados. Devolve (fração da base, ganho), começando em (0, 0).
    """
    acumulados = _acumulados(resposta, tratado, ordenar(escore), pesos)
    return _fracao(*acumulados[:2]), _qini(*acumulados)


def curva_uplift(
    resposta: np.ndarray, tratado: np.ndarray, escore: np.ndarray, pesos: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Curva de uplift acumulado: (taxa dos tratados − taxa dos controles no topo) × fração do topo.

    É o valor da política "mandar só para a fração do topo": o uplift observado entre os
    escolhidos, vezes a parte da base que eles representam. Devolve (fração, ganho) desde (0, 0).
    """
    acumulados = _acumulados(resposta, tratado, ordenar(escore), pesos)
    return _fracao(*acumulados[:2]), _uplift(*acumulados)


def area_sob_a_curva(fracao: np.ndarray, ganho: np.ndarray) -> np.ndarray | float:
    """Área pela regra do trapézio (na última dimensão). Para a curva de uplift, é a AUUC."""
    fracao, ganho = np.asarray(fracao, dtype=float), np.asarray(ganho, dtype=float)
    area = np.sum(np.diff(fracao, axis=-1) * (ganho[..., 1:] + ganho[..., :-1]) / 2.0, axis=-1)
    return float(area) if area.ndim == 0 else area


def area_acima_do_acaso(fracao: np.ndarray, ganho: np.ndarray) -> np.ndarray | float:
    """Área entre a curva e a reta do acaso (de zero ao efeito médio). Para a curva Qini, é o coeficiente Qini.

    Escolher ao acaso rende, em cada fração, a mesma fração do efeito total: uma reta. Um escore
    que não ordena nada fica em zero; um que põe no topo quem menos responde fica negativo.
    """
    area = area_sob_a_curva(fracao, ganho) - np.asarray(ganho, dtype=float)[..., -1] / 2.0
    return float(area) if np.ndim(area) == 0 else area


def coeficiente_qini(resposta: np.ndarray, tratado: np.ndarray, escore: np.ndarray, pesos: np.ndarray | None = None) -> float:
    """Coeficiente Qini de um escore: área entre a curva Qini e a reta do acaso."""
    return area_acima_do_acaso(*curva_qini(resposta, tratado, escore, pesos))


def auuc(resposta: np.ndarray, tratado: np.ndarray, escore: np.ndarray, pesos: np.ndarray | None = None) -> float:
    """Área sob a curva de uplift acumulado (AUUC)."""
    return area_sob_a_curva(*curva_uplift(resposta, tratado, escore, pesos))


def ganho_na_fracao(fracao: np.ndarray, ganho: np.ndarray, alvo: float | np.ndarray) -> np.ndarray:
    """Valor da curva numa fração qualquer da base (interpolação linear entre os pontos)."""
    return np.interp(alvo, fracao, ganho)


def uplift_por_faixa(resposta: np.ndarray, tratado: np.ndarray, escore: np.ndarray, faixas: int = 10) -> pd.DataFrame:
    """Uplift observado em cada faixa de escore (faixa 1 = maior escore), com intervalo de 95%.

    Cada faixa tem o mesmo número de clientes. Dentro dela, o uplift é a diferença entre a taxa
    dos tratados e a dos controles, e o erro padrão é o da diferença de duas médias independentes.
    """
    y, t, s = (np.asarray(v, dtype=float) for v in (resposta, tratado, escore))
    ordem = np.argsort(-s, kind="stable")
    faixa = np.empty(len(s), dtype=int)
    faixa[ordem] = np.arange(len(s)) * faixas // len(s) + 1
    linhas = []
    for f in range(1, faixas + 1):
        na_faixa = faixa == f
        yt, yc = y[na_faixa & (t == 1)], y[na_faixa & (t == 0)]
        taxa_t, taxa_c = float(yt.mean()), float(yc.mean())
        erro = float(np.sqrt(yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc)))
        uplift = taxa_t - taxa_c
        linhas.append({
            "faixa": f,
            "clientes": int(na_faixa.sum()),
            "escore_medio": float(s[na_faixa].mean()),
            "taxa_com_campanha": taxa_t,
            "taxa_sem_campanha": taxa_c,
            "uplift": uplift,
            "ic95_inferior": uplift - 1.96 * erro,
            "ic95_superior": uplift + 1.96 * erro,
        })
    return pd.DataFrame(linhas).set_index("faixa")


def bootstrap_das_curvas(
    resposta: np.ndarray,
    tratado: np.ndarray,
    escores: dict[str, np.ndarray],
    repeticoes: int = 1000,
    semente: int = 42,
    grade: np.ndarray = GRADE_PADRAO,
    max_pontos: int | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Reamostra os clientes do teste (dentro de cada grupo do experimento) e recalcula as curvas de cada escore.

    Todos os escores veem a mesma reamostra em cada repetição, então a diferença entre dois
    métodos pode ser calculada repetição a repetição (comparação pareada). A reamostra é feita
    com contagens: cada cliente recebe o número de vezes em que foi sorteado, e as curvas são
    recalculadas com esses pesos, sem reordenar. Mede a incerteza de avaliar num teste finito;
    não inclui a incerteza de treinar o modelo em outra amostra.

    `max_pontos` limita em quantas posições do ordenamento cada curva reamostrada é calculada
    (None = todas, a conta exata). Na base deste estudo, 2.000 posições mudam a área em menos
    de 0,5% do erro amostral e os limites da faixa em menos de um décimo dele, e o bootstrap
    roda na metade do tempo.

    Devolve, por escore: `qini` e `uplift` (repetições × pontos da grade), `coef_qini` e `auuc`
    (um valor por repetição).
    """
    y, t = np.asarray(resposta, dtype=float), np.asarray(tratado, dtype=float)
    gerador = np.random.default_rng(semente)
    estratos = [np.flatnonzero(t == 1), np.flatnonzero(t == 0)]
    # Para cada escore: o código (bloco do ordenamento × grupo) de cada cliente e o número de blocos.
    codigos = {}
    for nome, escore in escores.items():
        bloco, n_blocos = _bloco_de_cada_cliente(ordenar(escore), max_pontos)
        codigos[nome] = (bloco * 2 + t.astype(int), n_blocos)
    saida = {
        nome: {
            "qini": np.empty((repeticoes, len(grade))),
            "uplift": np.empty((repeticoes, len(grade))),
            "coef_qini": np.empty(repeticoes),
            "auuc": np.empty(repeticoes),
        }
        for nome in escores
    }
    for b in range(repeticoes):
        pesos = np.zeros(len(y))
        for indices in estratos:
            pesos[indices] = np.bincount(gerador.integers(0, len(indices), len(indices)), minlength=len(indices))
        pesos_y = pesos * y
        for nome, (codigo, n_blocos) in codigos.items():
            # Uma passada conta as pessoas e outra as respostas, já separadas por bloco e por grupo.
            pessoas = np.bincount(codigo, weights=pesos, minlength=2 * n_blocos).reshape(n_blocos, 2).cumsum(axis=0)
            respostas = np.bincount(codigo, weights=pesos_y, minlength=2 * n_blocos).reshape(n_blocos, 2).cumsum(axis=0)
            acumulados = (pessoas[:, 1], pessoas[:, 0], respostas[:, 1], respostas[:, 0])  # nt, nc, yt, yc
            fracao, qini, uplift = _fracao(*acumulados[:2]), _qini(*acumulados), _uplift(*acumulados)
            saida[nome]["coef_qini"][b] = area_acima_do_acaso(fracao, qini)
            saida[nome]["auuc"][b] = area_sob_a_curva(fracao, uplift)
            saida[nome]["qini"][b] = np.interp(grade, fracao, qini)
            saida[nome]["uplift"][b] = np.interp(grade, fracao, uplift)
    return saida


def _bloco_de_cada_cliente(ordenacao: Ordenacao, max_pontos: int | None = None) -> tuple[np.ndarray, int]:
    """Número do bloco do ordenamento de cada cliente (0 = maior escore) e o total de blocos.

    Sem `max_pontos`, um bloco por valor distinto do escore. Com ele, blocos vizinhos são
    unidos até sobrarem no máximo `max_pontos`, de tamanhos parecidos; clientes com o mesmo
    escore nunca são separados.
    """
    limites = ordenacao.limites
    if max_pontos is not None and len(limites) > max_pontos:
        limites = limites[np.unique(np.linspace(0, len(limites) - 1, max_pontos).round().astype(int))]
    bloco = np.empty(len(ordenacao.ordem), dtype=int)
    bloco[ordenacao.ordem] = np.searchsorted(limites, np.arange(len(ordenacao.ordem)))
    return bloco, len(limites)


def intervalo(valores: np.ndarray, confianca: float = 0.95) -> tuple[float, float]:
    """Intervalo percentil de uma coleção de repetições do bootstrap."""
    cauda = (1.0 - confianca) / 2.0
    return float(np.quantile(valores, cauda)), float(np.quantile(valores, 1.0 - cauda))
