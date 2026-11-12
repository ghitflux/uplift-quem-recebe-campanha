"""Leitura de um experimento aleatorizado: checagem do sorteio, efeito médio com intervalo e poder."""

from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd
from scipy import stats


def diferenca_padronizada(a: np.ndarray, b: np.ndarray) -> float:
    """Diferença de médias entre dois grupos, em desvios padrão (média das duas variâncias).

    É a medida usual de equilíbrio de covariáveis: não depende do tamanho da amostra, ao
    contrário do p-valor. A regra de bolso trata |d| < 0,1 como equilíbrio.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    desvio = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return 0.0 if desvio == 0 else float((a.mean() - b.mean()) / desvio)


def tabela_de_equilibrio(atributos: pd.DataFrame, grupo: pd.Series, controle: str) -> pd.DataFrame:
    """Diferença padronizada de cada atributo entre cada grupo tratado e o controle."""
    saida = {}
    for nome in [g for g in pd.unique(grupo) if g != controle]:
        saida[nome] = {
            coluna: diferenca_padronizada(atributos.loc[grupo == nome, coluna], atributos.loc[grupo == controle, coluna])
            for coluna in atributos.columns
        }
    return pd.DataFrame(saida)


def teste_da_divisao(contagens: dict[str, int]) -> dict[str, float]:
    """Qui-quadrado contra a divisão igual entre os grupos (checagem de "sample ratio mismatch")."""
    observado = np.array(list(contagens.values()), dtype=float)
    qui2, p = stats.chisquare(observado)
    return {"qui2": float(qui2), "p": float(p), "graus_de_liberdade": len(observado) - 1}


def efeito_medio(tratados: np.ndarray, controles: np.ndarray, confianca: float = 0.95) -> dict[str, float]:
    """Diferença de médias (tratados − controles) com intervalo de confiança e teste z, variâncias separadas.

    Serve para taxa (0/1) e para valor (gasto): com dezenas de milhares de clientes por grupo, a
    média é aproximadamente normal mesmo com a distribuição do gasto muito assimétrica.
    """
    t, c = np.asarray(tratados, dtype=float), np.asarray(controles, dtype=float)
    diferenca = float(t.mean() - c.mean())
    erro = float(np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c)))
    z = stats.norm.ppf(1 - (1 - confianca) / 2)
    return {
        "media_tratados": float(t.mean()),
        "media_controles": float(c.mean()),
        "efeito": diferenca,
        "erro_padrao": erro,
        "ic_inferior": diferenca - z * erro,
        "ic_superior": diferenca + z * erro,
        "efeito_relativo": diferenca / float(c.mean()) if c.mean() != 0 else float("nan"),
        "p": float(2 * stats.norm.sf(abs(diferenca / erro))) if erro > 0 else float("nan"),
        "n_tratados": int(len(t)),
        "n_controles": int(len(c)),
    }


def intervalo_bootstrap_da_diferenca(
    tratados: np.ndarray, controles: np.ndarray, repeticoes: int = 2000, semente: int = 42
) -> tuple[float, float]:
    """Intervalo percentil de 95% da diferença de médias, reamostrando cada grupo. Conferência para o gasto."""
    t, c = np.asarray(tratados, dtype=float), np.asarray(controles, dtype=float)
    gerador = np.random.default_rng(semente)
    diferencas = np.empty(repeticoes)
    for b in range(repeticoes):
        diferencas[b] = t[gerador.integers(0, len(t), len(t))].mean() - c[gerador.integers(0, len(c), len(c))].mean()
    return float(np.quantile(diferencas, 0.025)), float(np.quantile(diferencas, 0.975))


def efeito_minimo_detectavel(taxa_base: float, n_por_grupo: float, alfa: float = 0.05, poder: float = 0.80) -> float:
    """Menor diferença de taxa (em pontos de proporção) que dois grupos de `n_por_grupo` detectam.

    Aproximação normal com a variância da taxa de base nos dois grupos: é a conta de bolso para
    saber se um recorte da base tem tamanho para mostrar um efeito.
    """
    z = stats.norm.ppf(1 - alfa / 2) + stats.norm.ppf(poder)
    return float(z * np.sqrt(2.0 * taxa_base * (1.0 - taxa_base) / n_por_grupo))


def amostra_por_grupo(taxa_controle: float, taxa_tratado: float, alfa: float = 0.05, poder: float = 0.80) -> int:
    """Clientes por grupo para detectar a diferença entre duas taxas (teste z bicaudal, variâncias separadas)."""
    z = stats.norm.ppf(1 - alfa / 2) + stats.norm.ppf(poder)
    variancia = taxa_controle * (1 - taxa_controle) + taxa_tratado * (1 - taxa_tratado)
    return ceil(z**2 * variancia / (taxa_tratado - taxa_controle) ** 2)
