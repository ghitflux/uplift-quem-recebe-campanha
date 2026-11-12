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


def montar(resultados: dict[str, dict], efeitos, base_resumo: dict, poder: dict | None = None) -> dict:
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
        "graficos_extras": graficos_extras(resultados, efeitos, poder) if poder is not None else [],
    }


IDIOMAS = ("pt", "en", "es")
PAPEIS = {"principal", "base", "alternativa", "neutro"}
FORMATOS = {"numero", "percentual", "moeda", "data"}


def problemas_do_grafico(spec: dict, extra: bool = True) -> list[str]:
    """Lista o que a especificação de gráfico viola nas regras do site (lista vazia = válida)."""
    erros = []
    campos = ["titulo", "eixo_x", "eixo_y"] + (["leitura"] if extra else [])
    for campo in campos:
        texto = spec.get(campo)
        if not isinstance(texto, dict) or any(not isinstance(texto.get(i), str) or not texto[i].strip() for i in IDIOMAS):
            erros.append(f"{campo}: precisa de texto em pt, en e es")
    if extra and isinstance(spec.get("titulo"), dict) and any(len(spec["titulo"].get(i, "")) > 90 for i in IDIOMAS):
        erros.append("titulo: mais de 90 caracteres")
    tipo = spec.get("tipo")
    if tipo not in {"linhas", "barras", "pontos"}:
        erros.append(f"tipo inválido: {tipo}")
    if spec.get("formato_x") not in FORMATOS or spec.get("formato_y") not in FORMATOS:
        erros.append("formato_x/formato_y inválido")
    series = spec.get("series", [])
    if not 1 <= len(series) <= 4:
        erros.append("de 1 a 4 séries")
    limite = 300 if tipo == "pontos" else 80
    for s in series:
        if s.get("papel") not in PAPEIS:
            erros.append(f"papel inválido: {s.get('papel')}")
        if not isinstance(s.get("nome"), dict) or any(not s["nome"].get(i) for i in IDIOMAS):
            erros.append("nome da série sem os três idiomas")
        pontos = s.get("pontos", [])
        if not 1 <= len(pontos) <= limite:
            erros.append(f"série com {len(pontos)} pontos (limite {limite})")
        for p in pontos:
            if len(p) != 2 or not isinstance(p[1], (int, float)) or (tipo != "barras" and not isinstance(p[0], (int, float))):
                erros.append(f"ponto mal formado: {p}")
                break
            if tipo == "barras" and not isinstance(p[0], str):
                erros.append(f"barras precisam de rótulo de texto: {p}")
                break
        if "rotulos" in s and any(len(s["rotulos"].get(i, [])) != len(pontos) for i in IDIOMAS):
            erros.append("rotulos não acompanham os pontos nos três idiomas")
    if "faixa" in spec:
        if tipo != "linhas":
            erros.append("faixa só em linhas")
        elif not 0 <= spec["faixa"].get("serie", -1) < len(series):
            erros.append("faixa aponta para série inexistente")
    return erros


def _t(pt: str, en: str, es: str) -> dict:
    return {"pt": pt, "en": en, "es": es}


def graficos_extras(resultados: dict[str, dict], efeitos, poder: dict) -> list[dict]:
    """Os outros achados do README como especificações de gráfico (mesmo formato de `grafico`, mais `leitura`).

    Ordem: comparação dos métodos, quem cada modelo escolhe, curva Qini do e-mail masculino, uplift por décimo e
    efeito médio contra o mínimo detectável. Todos os valores saem dos resultados desta execução.
    """
    fem, masc = resultados[FEMININO], resultados[MASCULINO]
    return [
        _comparacao_dos_metodos(fem, masc),
        _quem_cada_modelo_escolhe(fem),
        _qini_masculino(masc),
        _uplift_por_decil(fem),
        _efeito_medio_e_poder(efeitos, poder),
    ]


def _comparacao_dos_metodos(fem: dict, masc: dict) -> dict:
    chaves = [analise.chave(m, "logistica") for m in analise.METODOS_DE_UPLIFT] + [
        analise.chave("resposta", "logistica"), analise.chave("resposta", "lightgbm"), analise.REGRA_LINHA, analise.REGRA_RECENCIA,
    ]
    rotulos = {
        "pt": ["T-learner", "Transformação de classe", "S-learner", "X-learner", "Resposta (logística)", "Resposta (LightGBM)",
               "Regra: já comprou a linha", "Regra: recência"],
        "en": ["T-learner", "Class transformation", "S-learner", "X-learner", "Response (logistic)", "Response (LightGBM)",
               "Rule: bought the line before", "Rule: recency"],
        "es": ["T-learner", "Transformación de clase", "S-learner", "X-learner", "Respuesta (logística)", "Respuesta (LightGBM)",
               "Regla: ya compró la línea", "Regla: recencia"],
    }

    def serie(r: dict, nome: dict, papel: str) -> dict:
        qini = r["visita"]["metodos"]["qini"]
        pontos = [[rotulos["pt"][i], round(1000 * float(qini[c]), 2)] for i, c in enumerate(chaves)]
        return {"nome": nome, "papel": papel, "pontos": pontos, "rotulos": rotulos}

    q = 1000 * fem["visita"]["metodos"]["qini"]
    regra, principal = float(q[analise.REGRA_LINHA]), float(q[fem["principal"]])
    return {
        "tipo": "barras",
        "titulo": _t(
            "Os métodos de uplift empatam entre si, e a regra de uma variável chega junto",
            "The uplift methods tie with each other, and a one-variable rule keeps up",
            "Los métodos de uplift empatan entre sí y una regla de una variable llega igual",
        ),
        "eixo_x": _t("jeito de escolher quem recebe", "targeting approach", "forma de elegir a quién enviar"),
        "eixo_y": _t("coeficiente Qini (visitas por mil clientes)", "Qini coefficient (visits per thousand customers)",
                     "coeficiente Qini (visitas por mil clientes)"),
        "formato_x": "numero",
        "formato_y": "numero",
        "series": [
            serie(fem, _t("E-mail feminino", "Women's e-mail", "E-mail femenino"), "principal"),
            serie(masc, _t("E-mail masculino", "Men's e-mail", "E-mail masculino"), "neutro"),
        ],
        "leitura": _t(
            f"Qini na metade de teste (zero = escolher ao acaso). No e-mail feminino a regra fica em {_n(regra, 1, 'pt')} e o modelo "
            f"em {_n(principal, 1, 'pt')}, o dobro do modelo de resposta; no masculino tudo fica dentro do ruído.",
            f"Qini on the held-out half (zero = random targeting). For the women's e-mail the rule scores {_n(regra, 1, 'en')} and the "
            f"model {_n(principal, 1, 'en')}, twice the response model; for the men's e-mail everything sits within the noise.",
            f"Qini en la mitad de prueba (cero = elegir al azar). En el e-mail femenino la regla queda en {_n(regra, 1, 'es')} y el "
            f"modelo en {_n(principal, 1, 'es')}, el doble del modelo de respuesta; en el masculino todo queda dentro del ruido.",
        ),
    }


def _quem_cada_modelo_escolhe(fem: dict) -> dict:
    perfil = fem["visita"]["quem_escolhe"]["0.5"]
    ordem = ["Base inteira", "Modelo de resposta", "Modelo de uplift", "Regra"]
    rotulos = {
        "pt": ["Base inteira", "Modelo de resposta (50%)", "Modelo de uplift (50%)", "Regra: já comprou a linha feminina"],
        "en": ["Whole list", "Response model (top 50%)", "Uplift model (top 50%)", "Rule: bought women's line before"],
        "es": ["Toda la base", "Modelo de respuesta (50%)", "Modelo de uplift (50%)", "Regla: ya compró la línea femenina"],
    }

    def serie(coluna: str, nome: dict, papel: str) -> dict:
        pontos = [[rotulos["pt"][i], round(float(perfil.loc[c, coluna]), 4)] for i, c in enumerate(ordem)]
        return {"nome": nome, "papel": papel, "pontos": pontos, "rotulos": rotulos}

    sem_resp, sem_upl = 100 * float(perfil.loc["Modelo de resposta", "taxa_sem_email"]), 100 * float(perfil.loc["Modelo de uplift", "taxa_sem_email"])
    up_resp, up_upl = 100 * float(perfil.loc["Modelo de resposta", "uplift"]), 100 * float(perfil.loc["Modelo de uplift", "uplift"])
    return {
        "tipo": "barras",
        "titulo": _t(
            "O modelo de resposta escolhe quem já visitaria; o de uplift, quem muda com o e-mail",
            "The response model picks people who'd visit anyway; uplift picks those the e-mail moves",
            "El modelo de respuesta elige a quien ya visitaría; el de uplift, a quien el e-mail mueve",
        ),
        "eixo_x": _t("público escolhido (e-mail feminino)", "audience selected (women's e-mail)", "público elegido (e-mail femenino)"),
        "eixo_y": _t("clientes que visitaram o site", "customers who visited the site", "clientes que visitaron el sitio"),
        "formato_x": "numero",
        "formato_y": "percentual",
        "series": [
            serie("taxa_sem_email", _t("Sem e-mail", "Without e-mail", "Sin e-mail"), "neutro"),
            serie("taxa_com_email", _t("Com e-mail", "With e-mail", "Con e-mail"), "principal"),
        ],
        "leitura": _t(
            f"Sem e-mail, já visitam {_n(sem_resp, 1, 'pt')}% dos escolhidos pelo modelo de resposta e {_n(sem_upl, 1, 'pt')}% dos escolhidos "
            f"pelo de uplift; o e-mail soma {_n(up_resp, 1, 'pt')} e {_n(up_upl, 1, 'pt')} pontos.",
            f"Without the e-mail, {_n(sem_resp, 1, 'en')}% of the response model's picks visit anyway, against {_n(sem_upl, 1, 'en')}% of the "
            f"uplift model's; the e-mail adds {_n(up_resp, 1, 'en')} and {_n(up_upl, 1, 'en')} points.",
            f"Sin e-mail ya visitan el {_n(sem_resp, 1, 'es')}% de los elegidos por el modelo de respuesta y el {_n(sem_upl, 1, 'es')}% de los "
            f"elegidos por el de uplift; el e-mail suma {_n(up_resp, 1, 'es')} y {_n(up_upl, 1, 'es')} puntos.",
        ),
    }


def _qini_masculino(masc: dict) -> dict:
    curvas = masc["visita"]["curvas"]
    fracao = np.array(curvas["fracao"])
    serie = curvas["series"][masc["principal"]]
    linha = masc["visita"]["metodos"].loc[masc["principal"]]
    q, inf, sup = (1000 * float(linha[c]) for c in ("qini", "qini_ic_inferior", "qini_ic_superior"))
    return {
        "tipo": "linhas",
        "titulo": _t(
            "No e-mail masculino, nenhuma escolha bate o sorteio: o certo é mandar para todos",
            "For the men's e-mail no targeting beats random: send it to everyone",
            "En el e-mail masculino ninguna selección supera al azar: lo correcto es enviar a todos",
        ),
        "eixo_x": _t("parte da base que recebe o e-mail (maior escore primeiro)", "share of the list that gets the e-mail (highest score first)",
                     "parte de la base que recibe el e-mail (mayor puntuación primero)"),
        "eixo_y": _t("visitas incrementais por mil clientes", "incremental visits per thousand customers", "visitas incrementales por mil clientes"),
        "formato_x": "percentual",
        "formato_y": "numero",
        "series": [
            {"nome": _t("Modelo de uplift", "Uplift model", "Modelo de uplift"), "papel": "principal", "pontos": _pontos(fracao, serie["ganho"])},
            {"nome": _t("Modelo de resposta", "Response model", "Modelo de respuesta"), "papel": "base", "pontos": _pontos(fracao, curvas["series"][masc["base"]]["ganho"])},
            {
                "nome": _t("Regra: já comprou a linha masculina", "Rule: bought the men's line before", "Regla: ya compró la línea masculina"),
                "papel": "alternativa",
                "pontos": _pontos(fracao, curvas["series"][analise.REGRA_LINHA]["ganho"]),
            },
            {"nome": _t("Ao acaso", "Random targeting", "Al azar"), "papel": "neutro", "pontos": _pontos(fracao, curvas["acaso"])},
        ],
        "faixa": {"serie": 0, "inferior": _pontos(fracao, serie["inferior"]), "superior": _pontos(fracao, serie["superior"])},
        "leitura": _t(
            f"As curvas se confundem com a reta do acaso: Qini do modelo de uplift de {_n(q, 1, 'pt')} por mil (IC 95% de {_n(inf, 1, 'pt')} "
            f"a {_n(sup, 1, 'pt')}). O efeito é parecido em toda a base.",
            f"The curves overlap the random line: uplift model Qini of {_n(q, 1, 'en')} per thousand (95% CI {_n(inf, 1, 'en')} to "
            f"{_n(sup, 1, 'en')}). The effect is similar across the whole list.",
            f"Las curvas se confunden con la recta del azar: Qini del modelo de uplift de {_n(q, 1, 'es')} por mil (IC 95% de {_n(inf, 1, 'es')} "
            f"a {_n(sup, 1, 'es')}). El efecto es parecido en toda la base.",
        ),
    }


def _uplift_por_decil(fem: dict) -> dict:
    decis = fem["visita"]["decis"]
    rotulos = {"pt": [f"{d}º" for d in decis.index], "en": [f"{d}{'st' if d == 1 else 'nd' if d == 2 else 'rd' if d == 3 else 'th'}" for d in decis.index],
               "es": [f"{d}.º" for d in decis.index]}
    sem_efeito = int((decis["ic95_inferior"] <= 0).sum())
    metade = 100 * float(decis["uplift"].iloc[:5].mean())
    return {
        "tipo": "barras",
        "titulo": _t(
            "A metade com maior uplift previsto concentra o efeito do e-mail feminino",
            "The half with the highest predicted uplift holds most of the women's e-mail effect",
            "La mitad con mayor uplift previsto concentra el efecto del e-mail femenino",
        ),
        "eixo_x": _t("décimo do teste (1º = maior uplift previsto)", "test-set decile (1st = highest predicted uplift)",
                     "decil de la prueba (1.º = mayor uplift previsto)"),
        "eixo_y": _t("visitas a mais por cliente (pontos percentuais)", "extra visits per customer (percentage points)",
                     "visitas adicionales por cliente (puntos porcentuales)"),
        "formato_x": "numero",
        "formato_y": "percentual",
        "series": [{
            "nome": _t("Uplift observado", "Observed uplift", "Uplift observado"),
            "papel": "principal",
            "pontos": [[rotulos["pt"][i], round(float(v), 4)] for i, v in enumerate(decis["uplift"])],
            "rotulos": rotulos,
        }],
        "leitura": _t(
            f"Nos cinco primeiros décimos o e-mail soma em média {_n(metade, 1, 'pt')} pontos de visita; em {sem_efeito} décimos o "
            "intervalo de 95% inclui o zero.",
            f"In the top five deciles the e-mail adds {_n(metade, 1, 'en')} visit points on average; in {sem_efeito} deciles the "
            "95% interval includes zero.",
            f"En los cinco primeros deciles el e-mail suma en promedio {_n(metade, 1, 'es')} puntos de visita; en {sem_efeito} deciles el "
            "intervalo del 95% incluye el cero.",
        ),
    }


def _efeito_medio_e_poder(efeitos, poder: dict) -> dict:
    rotulos = {"pt": ["Visita ao site", "Compra"], "en": ["Site visit", "Purchase"], "es": ["Visita al sitio", "Compra"]}
    desfechos = ["visita", "conversao"]

    def serie(valores: list[float], nome: dict, papel: str) -> dict:
        return {"nome": nome, "papel": papel, "pontos": [[rotulos["pt"][i], round(v, 5)] for i, v in enumerate(valores)], "rotulos": rotulos}

    minimo = [float(poder[d]["minimo_detectavel"]["um decil do teste"]["efeito"]) for d in desfechos]
    fem = [float(efeitos.loc[(FEMININO, d), "efeito"]) for d in desfechos]
    masc = [float(efeitos.loc[(MASCULINO, d), "efeito"]) for d in desfechos]
    return {
        "tipo": "barras",
        "titulo": _t(
            "Os dois e-mails funcionam, mas o efeito em compra é pequeno demais para fatiar a base",
            "Both e-mails work, but the purchase effect is too small to slice the list",
            "Los dos e-mails funcionan, pero el efecto en compras es demasiado pequeño para segmentar",
        ),
        "eixo_x": _t("desfecho nas duas semanas seguintes", "outcome over the following two weeks", "resultado en las dos semanas siguientes"),
        "eixo_y": _t("diferença contra o grupo sem e-mail", "difference vs. the no-e-mail group", "diferencia frente al grupo sin e-mail"),
        "formato_x": "numero",
        "formato_y": "percentual",
        "series": [
            serie(fem, _t("E-mail feminino", "Women's e-mail", "E-mail femenino"), "principal"),
            serie(masc, _t("E-mail masculino", "Men's e-mail", "E-mail masculino"), "alternativa"),
            serie(minimo, _t("Mínimo detectável num décimo do teste", "Smallest effect a test decile can detect",
                             "Mínimo detectable en un decil de la prueba"), "neutro"),
        ],
        "leitura": _t(
            f"Em compra, um décimo da metade de teste só detecta efeitos a partir de {_n(100 * minimo[1], 2, 'pt')} ponto, mais que o efeito "
            f"inteiro de qualquer e-mail ({_n(100 * fem[1], 2, 'pt')} e {_n(100 * masc[1], 2, 'pt')}); por isso o desfecho principal é a visita.",
            f"On purchases, a tenth of the test half can only detect effects of {_n(100 * minimo[1], 2, 'en')} points or more, larger than "
            f"either e-mail's full effect ({_n(100 * fem[1], 2, 'en')} and {_n(100 * masc[1], 2, 'en')}); that's why visits are the main outcome.",
            f"En compras, un décimo de la mitad de prueba solo detecta efectos desde {_n(100 * minimo[1], 2, 'es')} punto, más que el efecto "
            f"total de cualquier e-mail ({_n(100 * fem[1], 2, 'es')} y {_n(100 * masc[1], 2, 'es')}); por eso el resultado principal es la visita.",
        ),
    }
