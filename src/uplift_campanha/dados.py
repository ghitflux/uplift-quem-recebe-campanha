"""Carga da base de Hillstrom (MineThatData, 2008) e preparo dos atributos.

Cada linha é um cliente que comprou nos doze meses anteriores. Os 64.000 foram sorteados em três
grupos iguais: e-mail da linha masculina, e-mail da linha feminina e nenhum e-mail. Os desfechos
foram medidos nas duas semanas seguintes ao envio.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CAMINHO = Path("data") / "hillstrom_email_2008.csv"

CONTROLE = "Sem e-mail"
GRUPOS = {"Mens E-Mail": "E-mail masculino", "Womens E-Mail": "E-mail feminino", "No E-Mail": CONTROLE}
CAMPANHAS = ["E-mail feminino", "E-mail masculino"]
DESFECHOS = {"visita": "visita ao site", "conversao": "compra", "gasto": "gasto (US$)"}

_REGIAO = {"Urban": "urbana", "Surburban": "suburbana", "Rural": "rural"}  # "Surburban" é a grafia da base
_CANAL = {"Phone": "telefone", "Web": "web", "Multichannel": "multicanal"}

# Atributos conhecidos antes do envio. A faixa de histórico da base é o próprio `historico` em
# classes, então fica de fora.
ATRIBUTOS = [
    "recencia",
    "log_historico",
    "comprou_masculino",
    "comprou_feminino",
    "cliente_novo",
    "regiao_urbana",
    "regiao_suburbana",
    "regiao_rural",
    "canal_telefone",
    "canal_web",
    "canal_multicanal",
]

ROTULOS = {
    "recencia": "Meses desde a última compra",
    "log_historico": "Gasto no último ano (log)",
    "comprou_masculino": "Comprou linha masculina",
    "comprou_feminino": "Comprou linha feminina",
    "cliente_novo": "Cliente novo (12 meses)",
    "regiao_urbana": "Região urbana",
    "regiao_suburbana": "Região suburbana",
    "regiao_rural": "Região rural",
    "canal_telefone": "Comprou por telefone",
    "canal_web": "Comprou pela web",
    "canal_multicanal": "Comprou nos dois canais",
}


def carregar(caminho: Path | str = CAMINHO) -> pd.DataFrame:
    """Lê o CSV original e devolve a tabela com nomes em português e os atributos já numéricos."""
    bruto = pd.read_csv(caminho)
    tabela = pd.DataFrame({
        "grupo": bruto["segment"].map(GRUPOS),
        "recencia": bruto["recency"].astype(int),
        "historico": bruto["history"].astype(float),
        "log_historico": np.log(bruto["history"].astype(float)),
        "comprou_masculino": bruto["mens"].astype(int),
        "comprou_feminino": bruto["womens"].astype(int),
        "cliente_novo": bruto["newbie"].astype(int),
        "visita": bruto["visit"].astype(int),
        "conversao": bruto["conversion"].astype(int),
        "gasto": bruto["spend"].astype(float),
    })
    for original, nome in _REGIAO.items():
        tabela[f"regiao_{nome}"] = (bruto["zip_code"] == original).astype(int)
    for original, nome in _CANAL.items():
        tabela[f"canal_{nome}"] = (bruto["channel"] == original).astype(int)
    if tabela["grupo"].isna().any():
        raise ValueError("grupo desconhecido na coluna `segment`")
    return tabela


def recorte_da_campanha(tabela: pd.DataFrame, campanha: str) -> pd.DataFrame:
    """Clientes de uma campanha e do controle, com a coluna `tratado` (1 = recebeu o e-mail)."""
    recorte = tabela[tabela["grupo"].isin([campanha, CONTROLE])].reset_index(drop=True)
    recorte["tratado"] = (recorte["grupo"] == campanha).astype(int)
    return recorte


def separar_treino_teste(
    tabela: pd.DataFrame, fracao_teste: float = 0.5, semente: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sorteia treino e teste mantendo, nos dois, a proporção de tratados e as taxas de visita e de compra de cada grupo.

    A separação é estratificada por tratamento × desfecho (não visitou, visitou sem comprar,
    comprou): assim o efeito médio é o mesmo no treino e no teste, e uma diferença de acaso na
    taxa de base entre as duas metades não entra na conta como se fosse mérito do modelo.
    """
    gerador = np.random.default_rng(semente)
    no_teste = np.zeros(len(tabela), dtype=bool)
    estrato = tabela["tratado"].to_numpy() * 3 + tabela["visita"].to_numpy() + tabela["conversao"].to_numpy()
    for valor in np.unique(estrato):
        indices = gerador.permutation(np.flatnonzero(estrato == valor))
        no_teste[indices[: int(round(len(indices) * fracao_teste))]] = True
    return tabela[~no_teste].reset_index(drop=True), tabela[no_teste].reset_index(drop=True)


def dobras_estratificadas(tabela: pd.DataFrame, dobras: int = 5, semente: int = 42) -> np.ndarray:
    """Número da dobra (0 a dobras − 1) de cada linha, com a mesma estratificação da separação treino/teste."""
    gerador = np.random.default_rng(semente)
    dobra = np.empty(len(tabela), dtype=int)
    estrato = tabela["tratado"].to_numpy() * 3 + tabela["visita"].to_numpy() + tabela["conversao"].to_numpy()
    for valor in np.unique(estrato):
        indices = gerador.permutation(np.flatnonzero(estrato == valor))
        dobra[indices] = np.arange(len(indices)) % dobras
    return dobra


def matriz(tabela: pd.DataFrame) -> np.ndarray:
    """Matriz de atributos na ordem de `ATRIBUTOS`."""
    return tabela[ATRIBUTOS].to_numpy(dtype=float)
