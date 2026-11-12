"""Bloco `para_o_site` do resultados.json: o resumo trilíngue e o gráfico que o site do portfólio desenha."""

from __future__ import annotations

import numpy as np

from uplift_campanha import analise

FEMININO, MASCULINO = "E-mail feminino", "E-mail masculino"


def _n(valor: float, casas: int, idioma: str) -> str:
    """Número no formato do idioma: vírgula decimal em pt e es, ponto em en; menos tipográfico."""
    texto = f"{valor:.{casas}f}"
    if idioma != "en":
        texto = texto.replace(".", ",")
    return texto.replace("-", "−")


def _pontos(fracao: np.ndarray, valores: list[float], passo: int = 2) -> list[list[float]]:
    return [[round(float(f), 2), round(1000 * float(v), 2)] for f, v in zip(fracao[::passo], valores[::passo])]


def montar(resultados: dict[str, dict], efeitos, base_resumo: dict) -> dict:
    """Monta o bloco a partir dos resultados das duas campanhas. Todo número do texto sai daqui."""
    fem, masc = resultados[FEMININO], resultados[MASCULINO]
    principal, base = fem["principal"], fem["base"]
    politica = fem["visita"]["politica"]
    metade = politica.loc[(0.5, principal)]
    metade_resposta = politica.loc[(0.5, base)]
    qini = fem["visita"]["metodos"]["qini"]
    q_masc = masc["visita"]["metodos"].loc[masc["principal"]]
    conversao = sorted(100 * float(efeitos.loc[(c, "conversao"), "efeito"]) for c in (FEMININO, MASCULINO))

    p = 100 * metade["parte_do_efeito_total"]
    p_inf, p_sup = 100 * metade["parte_ic_inferior"], 100 * metade["parte_ic_superior"]
    p_resp = 100 * metade_resposta["parte_do_efeito_total"]
    q_regra, q_principal = 1000 * qini[analise.REGRA_LINHA], 1000 * qini[principal]
    qm, qm_inf, qm_sup = 1000 * q_masc["qini"], 1000 * q_masc["qini_ic_inferior"], 1000 * q_masc["qini_ic_superior"]
    clientes = base_resumo["clientes"]
    milhar = f"{clientes:,}".replace(",", ".")

    curvas = fem["visita"]["curvas"]
    fracao = np.array(curvas["fracao"])
    serie = curvas["series"][principal]

    return {
        "numero_chave": {"pt": f"{p:.0f}% com 50%", "en": f"{p:.0f}% with 50%", "es": f"{p:.0f}% con 50%"},
        "pergunta": {
            "pt": "Para quem mandar a campanha: para quem compra por causa dela, e não para quem compraria de qualquer jeito?",
            "en": "Who should get the campaign: the customers who act because of it, not the ones who would have bought anyway?",
            "es": "¿A quién enviarle la campaña: a quien compra gracias a ella y no a quien compraría de todos modos?",
        },
        "resultado": {
            "pt": (
                f"No e-mail da linha feminina, mandar só para a metade da base com maior uplift previsto preserva {p:.0f}% das visitas "
                f"incrementais (IC 95%: {p_inf:.0f}% a {p_sup:.0f}%); com a mesma metade, o modelo de resposta comum preserva {p_resp:.0f}%. "
                f"Mas a regra de uma variável “quem já comprou a linha feminina” empata com o modelo de uplift "
                f"(Qini {_n(q_regra, 1, 'pt')} contra {_n(q_principal, 1, 'pt')} por mil clientes)."
            ),
            "en": (
                f"For the women's-line e-mail, sending only to the half of the list with the highest predicted uplift keeps {p:.0f}% of the "
                f"incremental visits (95% CI: {p_inf:.0f}% to {p_sup:.0f}%); given the same half, the usual response model keeps {p_resp:.0f}%. "
                f"But a one-variable rule, “has bought from the women's line before”, ties with the uplift model "
                f"(Qini {_n(q_regra, 1, 'en')} vs {_n(q_principal, 1, 'en')} per thousand customers)."
            ),
            "es": (
                f"En el e-mail de la línea femenina, enviar solo a la mitad de la base con mayor uplift previsto conserva el {p:.0f}% de las "
                f"visitas incrementales (IC 95%: {p_inf:.0f}% a {p_sup:.0f}%); con esa misma mitad, el modelo de respuesta habitual conserva "
                f"el {p_resp:.0f}%. Pero una regla de una sola variable, «ya compró la línea femenina», empata con el modelo de uplift "
                f"(Qini {_n(q_regra, 1, 'es')} frente a {_n(q_principal, 1, 'es')} por mil clientes)."
            ),
        },
        "detalhe": {
            "pt": (
                f"No e-mail da linha masculina nenhum método se distingue do acaso (Qini do modelo de uplift: {_n(qm, 1, 'pt')} por mil, "
                f"IC 95% de {_n(qm_inf, 1, 'pt')} a {_n(qm_sup, 1, 'pt')}): o efeito é parecido em toda a base e o certo é mandar para todos. "
                f"Em compra, o efeito médio é de {_n(conversao[0], 1, 'pt')} a {_n(conversao[1], 1, 'pt')} ponto percentual e o experimento "
                f"não tem tamanho para separar os métodos."
            ),
            "en": (
                f"For the men's-line e-mail no method can be told apart from random targeting (uplift model Qini: {_n(qm, 1, 'en')} per thousand, "
                f"95% CI {_n(qm_inf, 1, 'en')} to {_n(qm_sup, 1, 'en')}): the effect is similar across the list, so the right call is to send to everyone. "
                f"On purchases the average effect is only {_n(conversao[0], 1, 'en')} to {_n(conversao[1], 1, 'en')} percentage points, "
                f"and the experiment is too small to separate the methods."
            ),
            "es": (
                f"En el e-mail de la línea masculina ningún método se distingue del azar (Qini del modelo de uplift: {_n(qm, 1, 'es')} por mil, "
                f"IC 95% de {_n(qm_inf, 1, 'es')} a {_n(qm_sup, 1, 'es')}): el efecto es parecido en toda la base y lo correcto es enviar a todos. "
                f"En compras, el efecto medio es de {_n(conversao[0], 1, 'es')} a {_n(conversao[1], 1, 'es')} punto porcentual y el experimento "
                f"no tiene tamaño para separar los métodos."
            ),
        },
        "dados": {
            "pt": f"Experimento aleatorizado de e-mail de Kevin Hillstrom (MineThatData, 2008): {milhar} clientes sorteados em três grupos, com visita, compra e gasto nas duas semanas seguintes.",
            "en": f"Kevin Hillstrom's randomized e-mail experiment (MineThatData, 2008): {clientes:,} customers randomly split into three groups, with visits, purchases and spend over the following two weeks.",
            "es": f"Experimento aleatorizado de e-mail de Kevin Hillstrom (MineThatData, 2008): {milhar} clientes sorteados en tres grupos, con visita, compra y gasto en las dos semanas siguientes.",
        },
        "etiquetas": ["inferência causal", "uplift", "experimento aleatorizado", "curva Qini", "meta-learners", "bootstrap"],
        "grafico": {
            "tipo": "linhas",
            "titulo": {
                "pt": "E-mail feminino: visitas incrementais conforme a parte da base que recebe",
                "en": "Women's e-mail: incremental visits by share of the list targeted",
                "es": "E-mail femenino: visitas incrementales según la parte de la base que lo recibe",
            },
            "eixo_x": {
                "pt": "parte da base que recebe o e-mail (maior escore primeiro)",
                "en": "share of the list that gets the e-mail (highest score first)",
                "es": "parte de la base que recibe el e-mail (mayor puntuación primero)",
            },
            "eixo_y": {
                "pt": "visitas incrementais por mil clientes",
                "en": "incremental visits per thousand customers",
                "es": "visitas incrementales por mil clientes",
            },
            "formato_x": "percentual",
            "formato_y": "numero",
            "series": [
                {"nome": {"pt": "Modelo de uplift", "en": "Uplift model", "es": "Modelo de uplift"}, "papel": "principal", "pontos": _pontos(fracao, serie["ganho"])},
                {"nome": {"pt": "Modelo de resposta", "en": "Response model", "es": "Modelo de respuesta"}, "papel": "base", "pontos": _pontos(fracao, curvas["series"][base]["ganho"])},
                {
                    "nome": {"pt": "Regra: já comprou a linha feminina", "en": "Rule: bought the women's line before", "es": "Regla: ya compró la línea femenina"},
                    "papel": "alternativa",
                    "pontos": _pontos(fracao, curvas["series"][analise.REGRA_LINHA]["ganho"]),
                },
                {"nome": {"pt": "Ao acaso", "en": "Random targeting", "es": "Al azar"}, "papel": "neutro", "pontos": _pontos(fracao, curvas["acaso"])},
            ],
            "faixa": {"serie": 0, "inferior": _pontos(fracao, serie["inferior"]), "superior": _pontos(fracao, serie["superior"])},
        },
    }
