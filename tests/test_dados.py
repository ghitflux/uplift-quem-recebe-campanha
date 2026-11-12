import io

import numpy as np
import pandas as pd
import pytest

from uplift_campanha import dados

CSV = """recency,history_segment,history,mens,womens,zip_code,newbie,channel,segment,visit,conversion,spend
10,2) $100 - $200,142.44,1,0,Surburban,0,Phone,Womens E-Mail,0,0,0
6,3) $200 - $350,329.08,1,1,Rural,1,Web,No E-Mail,1,0,0
7,2) $100 - $200,180.65,0,1,Urban,1,Multichannel,Mens E-Mail,1,1,29.99
"""


def _tabela_sintetica(n: int = 3_000, semente: int = 0) -> pd.DataFrame:
    gerador = np.random.default_rng(semente)
    visita = (gerador.uniform(size=n) < 0.15).astype(int)
    return pd.DataFrame({
        "cliente": np.arange(n),
        "tratado": gerador.integers(0, 2, n),
        "visita": visita,
        "conversao": visita * (gerador.uniform(size=n) < 0.1).astype(int),
    })


def test_carregar_traduz_colunas_e_cria_atributos_numericos():
    tabela = dados.carregar(io.StringIO(CSV))  # lê da memória: o teste não cria arquivo
    assert list(tabela["grupo"]) == ["E-mail feminino", "Sem e-mail", "E-mail masculino"]
    assert list(tabela["regiao_suburbana"]) == [1, 0, 0]
    assert list(tabela["canal_multicanal"]) == [0, 0, 1]
    assert tabela.loc[0, "log_historico"] == pytest.approx(np.log(142.44))
    assert tabela.loc[2, "gasto"] == pytest.approx(29.99)
    assert dados.matriz(tabela).shape == (3, len(dados.ATRIBUTOS))
    assert (tabela[["regiao_urbana", "regiao_suburbana", "regiao_rural"]].sum(axis=1) == 1).all()


def test_carregar_recusa_grupo_desconhecido():
    with pytest.raises(ValueError):
        dados.carregar(io.StringIO(CSV.replace("Mens E-Mail", "Outro E-Mail")))


def test_recorte_da_campanha_fica_com_a_campanha_e_o_controle():
    tabela = dados.carregar(io.StringIO(CSV))
    recorte = dados.recorte_da_campanha(tabela, "E-mail feminino")
    assert list(recorte["grupo"]) == ["E-mail feminino", "Sem e-mail"]
    assert list(recorte["tratado"]) == [1, 0]


def test_separacao_nao_repete_cliente_e_preserva_as_taxas_de_cada_grupo():
    tabela = _tabela_sintetica()
    treino, teste = dados.separar_treino_teste(tabela, fracao_teste=0.5, semente=1)
    assert len(treino) + len(teste) == len(tabela)
    assert set(treino["cliente"]).isdisjoint(teste["cliente"])
    assert abs(len(treino) - len(teste)) <= 6  # um arredondamento por estrato
    for coluna in ("tratado", "visita", "conversao"):
        assert treino[coluna].mean() == pytest.approx(teste[coluna].mean(), abs=0.002)
    for grupo in (0, 1):
        taxa_treino = treino.loc[treino["tratado"] == grupo, "visita"].mean()
        taxa_teste = teste.loc[teste["tratado"] == grupo, "visita"].mean()
        assert taxa_treino == pytest.approx(taxa_teste, abs=0.003)


def test_separacao_e_reprodutivel_e_muda_com_a_semente():
    tabela = _tabela_sintetica()
    _, teste_a = dados.separar_treino_teste(tabela, semente=1)
    _, teste_b = dados.separar_treino_teste(tabela, semente=1)
    _, teste_c = dados.separar_treino_teste(tabela, semente=2)
    assert list(teste_a["cliente"]) == list(teste_b["cliente"])
    assert list(teste_a["cliente"]) != list(teste_c["cliente"])


def test_dobras_cobrem_todos_uma_vez_e_ficam_equilibradas():
    tabela = _tabela_sintetica()
    dobra = dados.dobras_estratificadas(tabela, dobras=5, semente=3)
    assert set(dobra) == {0, 1, 2, 3, 4}
    tamanhos = np.bincount(dobra)
    assert tamanhos.max() - tamanhos.min() <= 6
    taxas = [tabela.loc[dobra == d, "tratado"].mean() for d in range(5)]
    assert max(taxas) - min(taxas) < 0.01
