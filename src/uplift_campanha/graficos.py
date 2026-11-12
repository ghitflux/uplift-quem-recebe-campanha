"""Figuras do estudo. Azul é sempre o modelo de uplift principal; laranja, o modelo de resposta
(a linha de base); verde-água, a regra de uma variável; cinza, o acaso e as referências."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator

from uplift_campanha import analise, dados
from uplift_campanha.estilo import APAGADO, AQUA, AZUL, EIXO, ESTILO, GRADE, LARANJA, SUPERFICIE, TINTA, TINTA_SECUNDARIA, nova_figura, numero

NOMES_CURTOS = {
    "dois_modelos": "Dois modelos (T-learner)",
    "classe_transformada": "Transformação de classe",
    "modelo_unico": "Modelo único com interações (S)",
    "x_learner": "X-learner",
    "resposta": "Modelo de resposta",
}
FAMILIAS_CURTAS = {"logistica": "logística", "lightgbm": "LightGBM"}
NOME_DO_DESFECHO = {"visita": "visitas", "conversao": "compras"}


def _nome_curto(chave: str, campanha: str) -> str:
    if chave == analise.REGRA_RECENCIA:
        return "Regra: comprou mais recentemente"
    if chave == analise.REGRA_LINHA:
        return f"Regra: já comprou a linha {'feminina' if campanha == 'E-mail feminino' else 'masculina'}"
    metodo, familia = chave.split("|")
    return f"{NOMES_CURTOS[metodo]} · {FAMILIAS_CURTAS[familia]}"


def _cor(chave: str) -> str:
    if chave.startswith("regra"):
        return AQUA
    return LARANJA if chave.startswith("resposta") else AZUL


def _figura_dupla(titulo: str, subtitulo: str, altura: float = 4.7, largura: float = 9.6, **opcoes) -> tuple[Figure, np.ndarray]:
    """Dois painéis lado a lado (um por campanha), com o título e o subtítulo do portfólio."""
    fig, eixos = plt.subplots(1, 2, figsize=(largura, altura), dpi=160, **opcoes)
    fig.subplots_adjust(top=0.76, left=0.075, right=0.985, bottom=0.24, wspace=0.07)
    fig.text(0.02, 0.955, titulo, fontsize=13, fontweight="bold", color=TINTA, va="top")
    fig.text(0.02, 0.885, subtitulo, fontsize=9.5, color=TINTA_SECUNDARIA, va="top")
    return fig, eixos


def _por_cento(valor: float, _posicao=None) -> str:
    return f"{valor * 100:.0f}%"


def curvas_qini(resultados: dict[str, dict], desfecho: str, titulo: str) -> Figure:
    """Curva Qini no teste, um painel por campanha: modelo de uplift (com faixa de 95%), modelo de resposta, regra e acaso."""
    with plt.rc_context(ESTILO):
        o_que = NOME_DO_DESFECHO[desfecho]
        fig, eixos = _figura_dupla(
            titulo,
            f"{o_que.capitalize()} incrementais por mil clientes, conforme a parte da base que recebe o e-mail (do maior para o menor escore).\n"
            "Metade de teste; faixa = intervalo de 95% do modelo de uplift (bootstrap).",
            sharey=True,
        )
        for ax, campanha in zip(eixos, dados.CAMPANHAS):
            r = resultados[campanha]
            bloco = r[desfecho]["curvas"]
            fracao = np.array(bloco["fracao"])
            principal, base = r["principal"], r["base"]
            ax.plot(fracao, 1000 * np.array(bloco["acaso"]), color=APAGADO, linewidth=1.4, zorder=2)
            serie = bloco["series"][principal]
            ax.fill_between(fracao, 1000 * np.array(serie["inferior"]), 1000 * np.array(serie["superior"]), color=AZUL, alpha=0.14, linewidth=0, zorder=1)
            ax.plot(fracao, 1000 * np.array(bloco["series"][analise.REGRA_LINHA]["ganho"]), color=AQUA, zorder=3)
            ax.plot(fracao, 1000 * np.array(bloco["series"][base]["ganho"]), color=LARANJA, zorder=4)
            ax.plot(fracao, 1000 * np.array(serie["ganho"]), color=AZUL, zorder=5)
            total = 1000 * bloco["efeito_medio_no_teste"]
            ax.plot(1.0, total, "o", color=TINTA_SECUNDARIA, markersize=6, zorder=6)
            ax.annotate(
                f"todos: {numero(total, 1 if desfecho == 'conversao' else 0)}", (1.0, total), xytext=(-10, 10), textcoords="offset points", ha="right",
                color=TINTA_SECUNDARIA, fontsize=8.5, zorder=8, bbox={"facecolor": SUPERFICIE, "edgecolor": "none", "pad": 1.5, "alpha": 0.85},
            )
            if desfecho == "visita":
                metade = r[desfecho]["politica"].loc[(0.5, principal)]
                ax.plot(0.5, metade["por_mil_clientes"], "o", color=AZUL, zorder=7)
                ax.annotate(
                    f"metade da base:\n{numero(metade['por_mil_clientes'], 0)} ({metade['parte_do_efeito_total'] * 100:.0f}% do total)",
                    (0.5, metade["por_mil_clientes"]), xytext=(-9, 9), textcoords="offset points", ha="right", va="bottom", color=TINTA, fontsize=8.5,
                )
            ax.set_title(campanha, loc="left", fontsize=10, fontweight="bold", color=TINTA, pad=8)
            ax.set_xlim(0, 1.02)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.xaxis.set_major_formatter(FuncFormatter(_por_cento))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: numero(v, 1 if desfecho == "conversao" else 0)))
            ax.set_xlabel("parte da base que recebe o e-mail")
        eixos[0].set_ylabel(f"{o_que} incrementais por mil clientes")
        fig.legend(
            handles=[
                Line2D([], [], color=AZUL, label="Modelo de uplift (escolhido por validação no treino)"),
                Line2D([], [], color=LARANJA, label="Modelo de resposta"),
                Line2D([], [], color=AQUA, label="Regra: já comprou a linha do e-mail"),
                Line2D([], [], color=APAGADO, linewidth=1.4, label="Ao acaso"),
            ],
            loc="lower center", ncol=4, fontsize=8.8, bbox_to_anchor=(0.5, 0.01), handlelength=1.6, columnspacing=1.6,
        )
        return fig


def efeito_medio(efeitos: pd.DataFrame, poder: dict) -> Figure:
    """Efeito de cada e-mail em visita e em compra, em pontos percentuais, com intervalo de 95% e o mínimo detectável num décimo do teste."""
    with plt.rc_context(ESTILO):
        fig, eixos = _figura_dupla(
            "Os dois e-mails funcionam, mas o efeito em compra é pequeno demais para fatiar",
            "Efeito médio contra o grupo sem e-mail, em pontos percentuais, com intervalo de 95%. O traço cinza é o menor efeito\n"
            "que um décimo da metade de teste consegue detectar (80% de poder): em compra, ele é maior que o efeito inteiro.",
            altura=3.6,
        )
        fig.subplots_adjust(left=0.15, bottom=0.2, top=0.70, wspace=0.12)
        for ax, desfecho, rotulo in zip(eixos, ("visita", "conversao"), ("Visita ao site", "Compra")):
            minimo = 100 * poder[desfecho]["minimo_detectavel"]["um decil do teste"]["efeito"]
            ax.grid(axis="y", visible=False)
            ax.grid(axis="x", visible=True)
            ax.axvline(minimo, color=APAGADO, linewidth=1.6, zorder=1)
            a_esquerda = desfecho == "conversao"  # em compra o traço fica à direita dos efeitos; o rótulo vai para dentro
            ax.annotate(
                f"mínimo detectável\nnum décimo: {numero(minimo, 2 if a_esquerda else 1)}", (minimo, 1.5), xytext=(-5 if a_esquerda else 5, 0),
                textcoords="offset points", ha="right" if a_esquerda else "left", color=TINTA_SECUNDARIA, fontsize=8.5, va="center",
            )
            for y, campanha in enumerate(dados.CAMPANHAS):
                linha = efeitos.loc[(campanha, desfecho)]
                ax.plot([100 * linha["ic_inferior"], 100 * linha["ic_superior"]], [y, y], color=AZUL, zorder=2)
                ax.plot(100 * linha["efeito"], y, "o", color=AZUL, zorder=3)
                ax.annotate(
                    f"+{numero(100 * linha['efeito'], 2 if desfecho == 'conversao' else 1)}",
                    (100 * linha["efeito"], y), xytext=(0, 9), textcoords="offset points", ha="center", color=TINTA, fontsize=9, fontweight="bold",
                )
            ax.set_title(rotulo, loc="left", fontsize=10, fontweight="bold", color=TINTA, pad=8)
            ax.set_yticks(range(len(dados.CAMPANHAS)), dados.CAMPANHAS if desfecho == "visita" else [""] * len(dados.CAMPANHAS))
            ax.set_ylim(-0.6, 1.9)
            ax.invert_yaxis()
            maior = max(minimo, 100 * max(efeitos.loc[(c, desfecho), "ic_superior"] for c in dados.CAMPANHAS))
            ax.set_xlim(0, maior * 1.08)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: numero(v, 1)))
            ax.set_xlabel("pontos percentuais a mais que o grupo sem e-mail")
        return fig


def uplift_por_decil(decis: pd.DataFrame, efeito_no_teste: float, campanha: str) -> Figure:
    """Uplift observado em cada décimo do teste, ordenado pelo uplift previsto, com intervalo de 95%."""
    with plt.rc_context(ESTILO):
        sem_efeito = int((decis["ic95_inferior"] <= 0).sum())
        fig, ax = nova_figura(
            f"A metade de cima concentra o efeito; em {sem_efeito} décimos o intervalo inclui o zero",
            f"{campanha}: visitas a mais por 100 clientes em cada décimo do teste, do maior para o menor uplift previsto. Intervalos de 95%.",
        )
        ax.axhline(0, color=EIXO, linewidth=1.2, zorder=1)
        ax.axhline(100 * efeito_no_teste, color=APAGADO, linewidth=1.4, zorder=1)
        ax.annotate(f"efeito médio: {numero(100 * efeito_no_teste, 1)}", (10.45, 100 * efeito_no_teste), xytext=(0, 4), textcoords="offset points", ha="right", color=TINTA_SECUNDARIA, fontsize=8.5)
        for faixa, linha in decis.iterrows():
            ax.plot([faixa, faixa], [100 * linha["ic95_inferior"], 100 * linha["ic95_superior"]], color=AZUL, zorder=2)
            ax.plot(faixa, 100 * linha["uplift"], "o", color=AZUL, zorder=3)
            ax.annotate(numero(100 * linha["uplift"], 1).replace("-", "−"), (faixa, 100 * linha["uplift"]), xytext=(7, 0), textcoords="offset points", va="center", color=TINTA_SECUNDARIA, fontsize=8.5)
        ax.set_xticks(range(1, 11), [f"{d}º" for d in range(1, 11)])
        ax.set_xlim(0.5, 10.6)
        ax.yaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: numero(v, 0).replace("-", "−")))
        ax.set_xlabel("décimo do teste (1º = maior uplift previsto)")
        ax.set_ylabel("visitas incrementais por 100 clientes")
        return fig


def quem_cada_metodo_escolhe(perfil: pd.DataFrame, campanha: str, fracao: float) -> Figure:
    """Taxa de visita sem e com e-mail no público escolhido por cada jeito de escolher."""
    with plt.rc_context(ESTILO):
        base_resposta = perfil.loc["Modelo de resposta", "taxa_sem_email"]
        base_uplift = perfil.loc["Modelo de uplift", "taxa_sem_email"]
        fig, ax = nova_figura(
            "O modelo de resposta escolhe quem já visitaria; o de uplift, quem muda com o e-mail",
            f"{campanha}, metade de teste: taxa de visita sem e-mail (círculo vazio) e com e-mail (círculo cheio) no público de cada escolha.\n"
            f"À direita, o efeito do e-mail em visitas por 100 clientes, com IC 95%. Sem e-mail visitam {numero(100 * base_resposta, 1)}% (resposta) e {numero(100 * base_uplift, 1)}% (uplift) dos escolhidos.",
            altura=3.9, largura=9.2,
        )
        fig.subplots_adjust(left=0.40, bottom=0.17, top=0.74)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        cores = {"Base inteira": APAGADO, "Modelo de resposta": LARANJA, "Modelo de uplift": AZUL, "Regra": AQUA}
        rotulos = []
        for y, (nome, linha) in enumerate(perfil.iterrows()):
            cor = next(c for chave, c in cores.items() if nome.startswith(chave))
            sem, com = 100 * linha["taxa_sem_email"], 100 * linha["taxa_com_email"]
            ax.plot([sem, com], [y, y], color=cor, zorder=2)
            ax.plot(sem, y, "o", markerfacecolor=SUPERFICIE, markeredgecolor=cor, markeredgewidth=2, zorder=3)
            ax.plot(com, y, "o", color=cor, zorder=3)
            ax.annotate(
                f"+{numero(100 * linha['uplift'], 1)}  [{numero(100 * linha['uplift_ic_inferior'], 1)} ; {numero(100 * linha['uplift_ic_superior'], 1)}]",
                (com, y), xytext=(9, 0), textcoords="offset points", va="center", color=TINTA, fontsize=9,
            )
            rotulos.append(linha["rotulo"])
        ax.set_yticks(range(len(perfil)), rotulos)
        ax.set_ylim(-0.6, len(perfil) - 0.4)
        ax.invert_yaxis()
        ax.set_xlim(0, max(100 * perfil["taxa_com_email"]) * 1.32)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.set_xlabel("clientes que visitaram o site nas duas semanas seguintes")
        return fig


def comparacao_dos_metodos(resultados: dict[str, dict], desfecho: str = "visita") -> Figure:
    """Coeficiente Qini de cada jeito de escolher, com intervalo de 95%, um painel por campanha. A faixa cinza é o placebo."""
    ordem = (
        [analise.chave(m, "logistica") for m in analise.METODOS_DE_UPLIFT]
        + [analise.chave(m, "lightgbm") for m in analise.METODOS_DE_UPLIFT]
        + [analise.chave("resposta", "logistica"), analise.chave("resposta", "lightgbm"), analise.REGRA_LINHA, analise.REGRA_RECENCIA]
    )
    posicoes = [0, 1, 2, 3, 4.6, 5.6, 6.6, 7.6, 9.2, 10.2, 11.8, 12.8]
    with plt.rc_context(ESTILO):
        fig, eixos = _figura_dupla(
            "Os quatro métodos empatam entre si, e uma regra de uma variável chega ao mesmo lugar",
            "Coeficiente Qini em visitas (por mil clientes) na metade de teste, com intervalo de 95% por bootstrap. Zero = escolher ao acaso.\n"
            "Faixa cinza: onde cai o Qini quando o rótulo de quem recebeu o e-mail é embaralhado (placebo, 95% das repetições).",
            altura=5.6, sharey=True, sharex=True,
        )
        fig.subplots_adjust(left=0.30, bottom=0.17, top=0.78, wspace=0.06)
        for ax, campanha in zip(eixos, dados.CAMPANHAS):
            r = resultados[campanha]
            metodos, placebo = r[desfecho]["metodos"], r[desfecho]["placebo"]
            ax.grid(axis="y", visible=False)
            ax.grid(axis="x", visible=True)
            ax.axvspan(1000 * placebo["qini_placebo_p2_5"], 1000 * placebo["qini_placebo_p97_5"], color=GRADE, alpha=0.9, linewidth=0, zorder=0)
            ax.axvline(0, color=EIXO, linewidth=1.2, zorder=1)
            for y, chave in zip(posicoes, ordem):
                linha = metodos.loc[chave]
                cor = _cor(chave)
                ax.plot([1000 * linha["qini_ic_inferior"], 1000 * linha["qini_ic_superior"]], [y, y], color=cor, zorder=2)
                ax.plot(1000 * linha["qini"], y, "o", color=cor, zorder=3)
                principal = chave == r["principal"]
                ax.annotate(
                    numero(1000 * linha["qini"], 1).replace("-", "−"), (1000 * linha["qini_ic_superior"], y), xytext=(6, 0), textcoords="offset points",
                    va="center", color=TINTA if principal else TINTA_SECUNDARIA, fontsize=8.5, fontweight="bold" if principal else "normal",
                )
            ax.set_title(campanha, loc="left", fontsize=10, fontweight="bold", color=TINTA, pad=8)
            ax.set_xlabel("coeficiente Qini (visitas por mil clientes)")
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: numero(v, 0).replace("-", "−")))
        eixos[0].set_yticks(posicoes, [_nome_curto(chave, "") .replace("linha masculina", "linha do e-mail") for chave in ordem])
        eixos[0].set_ylim(-0.7, 13.5)
        eixos[0].invert_yaxis()
        fig.legend(
            handles=[
                Line2D([], [], color=AZUL, marker="o", label="Métodos de uplift"),
                Line2D([], [], color=LARANJA, marker="o", label="Modelo de resposta"),
                Line2D([], [], color=AQUA, marker="o", label="Regras de bolso"),
                Patch(facecolor=GRADE, label="Placebo"),
            ],
            loc="lower center", ncol=4, fontsize=8.8, bbox_to_anchor=(0.62, 0.0), handlelength=1.6, columnspacing=1.6,
        )
        return fig
