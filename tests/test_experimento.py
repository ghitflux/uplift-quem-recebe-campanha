import numpy as np
import pandas as pd
import pytest

from uplift_campanha import experimento


def test_diferenca_padronizada_exemplo_feito_a_mao():
    # Médias 1 e 2, variância 2 nos dois grupos → (1 − 2) / raiz(2).
    assert experimento.diferenca_padronizada([0.0, 2.0], [1.0, 3.0]) == pytest.approx(-1 / np.sqrt(2))
    assert experimento.diferenca_padronizada([1.0, 1.0], [1.0, 1.0]) == 0.0


def test_tabela_de_equilibrio_aponta_o_atributo_desequilibrado():
    gerador = np.random.default_rng(0)
    n = 6_000
    grupo = pd.Series(np.repeat(["A", "B", "controle"], n // 3))
    atributos = pd.DataFrame({"sorteado": gerador.normal(size=n), "viciado": gerador.normal(size=n)})
    atributos.loc[grupo == "A", "viciado"] += 0.5
    tabela = experimento.tabela_de_equilibrio(atributos, grupo, "controle")
    assert list(tabela.columns) == ["A", "B"]
    assert tabela.loc["viciado", "A"] == pytest.approx(0.5, abs=0.08)
    assert tabela.drop(index="viciado").abs().to_numpy().max() < 0.1
    assert abs(tabela.loc["viciado", "B"]) < 0.1


def test_teste_da_divisao_aceita_grupos_iguais_e_rejeita_divisao_torta():
    assert experimento.teste_da_divisao({"a": 1000, "b": 1000, "c": 1000})["p"] == pytest.approx(1.0)
    assert experimento.teste_da_divisao({"a": 1000, "b": 1010, "c": 990})["p"] > 0.5
    torta = experimento.teste_da_divisao({"a": 1200, "b": 1000, "c": 800})
    assert torta["p"] < 1e-6
    assert torta["qui2"] == pytest.approx((200**2 + 0 + 200**2) / 1000)


def test_efeito_medio_bate_com_a_conta_a_mao():
    tratados = np.repeat([1, 0], [30, 70])
    controles = np.repeat([1, 0], [20, 80])
    efeito = experimento.efeito_medio(tratados, controles)
    erro = np.sqrt(0.3 * 0.7 * 100 / 99 / 100 + 0.2 * 0.8 * 100 / 99 / 100)
    assert efeito["efeito"] == pytest.approx(0.10)
    assert efeito["erro_padrao"] == pytest.approx(erro)
    assert efeito["ic_inferior"] == pytest.approx(0.10 - 1.959964 * erro, abs=1e-6)
    assert efeito["ic_superior"] == pytest.approx(0.10 + 1.959964 * erro, abs=1e-6)
    assert efeito["efeito_relativo"] == pytest.approx(0.5)
    assert 0.05 < efeito["p"] < 0.15  # z = 1,63


def test_intervalo_do_efeito_medio_cobre_o_valor_verdadeiro_na_frequencia_certa():
    gerador = np.random.default_rng(1)
    cobriu = 0
    for _ in range(400):
        efeito = experimento.efeito_medio(gerador.uniform(size=500) < 0.15, gerador.uniform(size=500) < 0.10)
        cobriu += efeito["ic_inferior"] < 0.05 < efeito["ic_superior"]
    assert 0.92 < cobriu / 400 < 0.98


def test_intervalo_bootstrap_concorda_com_o_intervalo_normal():
    gerador = np.random.default_rng(2)
    tratados, controles = gerador.exponential(2.0, 3_000), gerador.exponential(1.5, 3_000)
    normal = experimento.efeito_medio(tratados, controles)
    inferior, superior = experimento.intervalo_bootstrap_da_diferenca(tratados, controles, repeticoes=600)
    assert inferior == pytest.approx(normal["ic_inferior"], abs=0.02)
    assert superior == pytest.approx(normal["ic_superior"], abs=0.02)


def test_amostra_por_grupo_valor_de_referencia():
    # 10% contra 15%, alfa 5% bicaudal, poder 80%: (1,96 + 0,8416)² · (0,09 + 0,1275) / 0,05² = 682,9.
    assert experimento.amostra_por_grupo(0.10, 0.15) == 683
    assert experimento.amostra_por_grupo(0.10, 0.11) > 10 * experimento.amostra_por_grupo(0.10, 0.15)


def test_efeito_minimo_detectavel_valor_de_referencia_e_escala():
    assert experimento.efeito_minimo_detectavel(0.10, 1000) == pytest.approx(2.8016 * np.sqrt(0.18 / 1000), rel=1e-3)
    # Quatro vezes mais clientes, metade do efeito mínimo.
    assert experimento.efeito_minimo_detectavel(0.10, 4000) == pytest.approx(experimento.efeito_minimo_detectavel(0.10, 1000) / 2)


def test_poder_simulado_confere_com_o_efeito_minimo_detectavel():
    gerador = np.random.default_rng(3)
    n, base = 2_000, 0.10
    minimo = experimento.efeito_minimo_detectavel(base, n)
    detectou = sum(
        experimento.efeito_medio(gerador.uniform(size=n) < base + minimo, gerador.uniform(size=n) < base)["p"] < 0.05
        for _ in range(500)
    )
    assert 0.72 < detectou / 500 < 0.86
