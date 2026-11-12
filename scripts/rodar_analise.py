"""Roda a análise completa e grava reports/resultados.json e as figuras do README.

Uso: `uv run python scripts/rodar_analise.py` (a partir da raiz do projeto, depois de `baixar_dados.py`).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uplift_campanha import analise, dados, graficos, site, uplift

SAIDA = Path("reports")
FIGURAS = SAIDA / "figuras"


def _registros(tabela: pd.DataFrame) -> list[dict]:
    """Tabela como lista de registros, com o índice virando coluna."""
    return json.loads(tabela.reset_index().to_json(orient="records", force_ascii=False))


def _metodos(tabela: pd.DataFrame, campanha: str) -> list[dict]:
    com_nome = tabela.astype(float).copy()
    com_nome.insert(0, "nome", [analise.nome_do_escore(chave, campanha) for chave in com_nome.index])
    com_nome.index.name = "escore"
    return _registros(com_nome)


def _campanha_para_json(r: dict) -> dict:
    campanha = r["campanha"]
    escolha = r["escolha_do_principal"].copy()
    escolha.insert(0, "nome", [analise.nome_do_escore(chave, campanha) for chave in escolha.index])
    escolha.index.name = "escore"
    saida = {
        "separacao": r["separacao"],
        "principal": {"escore": r["principal"], "nome": analise.nome_do_escore(r["principal"], campanha)},
        "base": {"escore": r["base"], "nome": analise.nome_do_escore(r["base"], campanha)},
        "escolha_do_principal_por_validacao_cruzada_no_treino": _registros(escolha),
    }
    for desfecho in ("visita", "conversao"):
        bloco = r[desfecho]
        politica = bloco["politica"].reset_index()
        politica.insert(2, "nome", ["Ao acaso" if e == "acaso" else analise.nome_do_escore(e, campanha) for e in politica["escore"]])
        saida[desfecho] = {
            "metodos": _metodos(bloco["metodos"], campanha),
            "principal_contra_a_resposta": bloco["principal_contra_a_resposta"],
            "principal_contra_a_regra": bloco["principal_contra_a_regra"],
            "valor_da_politica": json.loads(politica.to_json(orient="records", force_ascii=False)),
            "quem_cada_escolha_alcanca": {fracao: _registros(tabela.rename_axis("escolha")) for fracao, tabela in bloco["quem_escolhe"].items()},
            "uplift_por_decil_do_principal": _registros(bloco["decis"]),
            "curva_qini": bloco["curvas"],
        }
        if "placebo" in bloco:
            saida[desfecho]["placebo"] = bloco["placebo"]
            repeticoes = bloco["repeticoes_da_separacao"].copy()
            repeticoes.insert(0, "nome", [analise.nome_do_escore(chave, campanha) for chave in repeticoes.index])
            saida[desfecho]["repeticoes_da_separacao"] = _registros(repeticoes.rename_axis("escore"))
            saida[desfecho]["qini_por_separacao"] = _registros(bloco["qini_por_separacao"].rename_axis("separacao"))
    return saida


def _conferir_a_narrativa(resultados: dict[str, dict]) -> None:
    """Para se o texto fixo das figuras e do site deixar de valer para os números desta execução."""
    fem, masc = resultados["E-mail feminino"], resultados["E-mail masculino"]
    regra = fem["visita"]["principal_contra_a_regra"]
    assert regra["ic_inferior"] < 0 < regra["ic_superior"], "feminino: o modelo de uplift não empata mais com a regra"
    assert fem["visita"]["principal_contra_a_resposta"]["ic_inferior"] > 0, "feminino: o modelo de uplift não supera mais o de resposta"
    assert (masc["visita"]["metodos"]["qini_ic_inferior"] < 0).all(), "masculino: algum escore passou a se distinguir do acaso"
    for r in (fem, masc):
        placebo = r["visita"]["placebo"]
        assert placebo["qini_placebo_p2_5"] < 0 < placebo["qini_placebo_p97_5"], "o placebo deixou de ficar em torno de zero"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    inicio = time.time()
    FIGURAS.mkdir(parents=True, exist_ok=True)

    tabela = dados.carregar()
    base = analise.resumo_da_base(tabela)
    equilibrio = analise.equilibrio(tabela)
    efeitos = analise.efeitos_medios(tabela)
    poder = analise.poder_por_recorte(tabela, efeitos)

    resultados = {}
    for campanha in dados.CAMPANHAS:
        resultados[campanha] = analise.analisar_campanha(tabela, campanha)
        print(f"{campanha}: análise pronta aos {time.time() - inicio:.0f} s", flush=True)
    _conferir_a_narrativa(resultados)

    fem = resultados["E-mail feminino"]
    figuras = {
        "qini_visita": graficos.curvas_qini(resultados, "visita", "No e-mail feminino dá para escolher quem recebe; no masculino, não"),
        "qini_compra": graficos.curvas_qini(resultados, "conversao", "Em compra, a incerteza é do tamanho do ganho: não dá para ordenar os métodos"),
        "efeito_medio": graficos.efeito_medio(efeitos, poder),
        "uplift_por_decil": graficos.uplift_por_decil(fem["visita"]["decis"], fem["visita"]["curvas"]["efeito_medio_no_teste"], "E-mail feminino"),
        "quem_cada_metodo_escolhe": graficos.quem_cada_metodo_escolhe(fem["visita"]["quem_escolhe"]["0.5"], "E-mail feminino", 0.5),
        "comparacao_dos_metodos": graficos.comparacao_dos_metodos(resultados),
    }
    for nome, figura in figuras.items():
        figura.savefig(FIGURAS / f"{nome}.png")
        plt.close(figura)

    saida = {
        "fonte": {
            "base": "Kevin Hillstrom, MineThatData E-Mail Analytics and Data Mining Challenge (2008)",
            "url": "https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html",
            "arquivo": str(dados.CAMINHO).replace("\\", "/"),
        },
        "configuracao": {
            "semente": analise.SEMENTE,
            "fracao_de_teste": 0.5,
            "desfecho_principal": "visita",
            "repeticoes_do_bootstrap": 1000,
            "posicoes_da_curva_no_bootstrap": analise.PONTOS_DO_BOOTSTRAP,
            "dobras_da_validacao_cruzada_no_treino": 5,
            "repeticoes_do_placebo": 100,
            "separacoes_repetidas": 20,
            "parametros_do_lightgbm": uplift.PARAMETROS_LGBM,
            "atributos": dados.ATRIBUTOS,
        },
        "base": base,
        "sorteio": {
            "diferenca_padronizada": _registros(equilibrio.rename_axis("atributo")),
            "maior_diferenca_absoluta": float(np.abs(equilibrio.to_numpy()).max()),
        },
        "efeito_medio": _registros(efeitos),
        "poder": poder,
        "campanhas": {campanha: _campanha_para_json(r) for campanha, r in resultados.items()},
        "para_o_site": site.montar(resultados, efeitos, base, poder),
    }
    (SAIDA / "resultados.json").write_text(json.dumps(saida, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(base)
    print("maior diferença padronizada:", saida["sorteio"]["maior_diferenca_absoluta"])
    print(efeitos[["media_tratados", "media_controles", "efeito", "ic_inferior", "ic_superior", "efeito_relativo", "p"]].to_string())
    for campanha, r in resultados.items():
        print(f"\n== {campanha}: principal = {analise.nome_do_escore(r['principal'], campanha)}")
        print((r["escolha_do_principal"] * [1000, 1000, 1]).to_string())
        for desfecho in ("visita", "conversao"):
            print(f"-- {desfecho}: Qini por mil clientes")
            print((r[desfecho]["metodos"][["qini", "qini_ic_inferior", "qini_ic_superior", "diferenca_para_a_base", "diferenca_ic_inferior", "diferenca_ic_superior"]].astype(float) * 1000).to_string())
            print(r[desfecho]["politica"][["por_mil_clientes", "ic_inferior", "ic_superior", "parte_do_efeito_total", "parte_ic_inferior", "parte_ic_superior", "acima_do_acaso", "acima_ic_inferior", "acima_ic_superior"]].to_string())
            print("principal contra a regra:", r[desfecho]["principal_contra_a_regra"])
        print(r["visita"]["quem_escolhe"]["0.5"].drop(columns="rotulo").astype(float).to_string())
        print(r["visita"]["quem_escolhe"]["0.3"].drop(columns="rotulo").astype(float).to_string())
        print(r["visita"]["decis"].to_string())
        print({k: v for k, v in r["visita"]["placebo"].items() if k != "qinis"})
        print(r["visita"]["repeticoes_da_separacao"].to_string())
    print(json.dumps({k: v for k, v in saida["para_o_site"].items() if k != "grafico"}, indent=2, ensure_ascii=False))
    print(f"\nGravado em {SAIDA / 'resultados.json'} e {FIGURAS}")
    print(f"Tempo total: {time.time() - inicio:.0f} s")


if __name__ == "__main__":
    main()
