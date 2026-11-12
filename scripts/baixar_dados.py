"""Baixa a base de Hillstrom (MineThatData, 2008) para data/ e confere o hash.

A base tem cerca de 4 MB e já vem versionada no repositório; este script serve para refazer o
download a partir da fonte original e provar que o arquivo é o mesmo. Se o arquivo local já
existir com o hash certo, nada é baixado.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
DESTINO = Path("data") / "hillstrom_email_2008.csv"
SHA256 = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"
TAMANHO = 3_964_977  # bytes
LINHAS = 64_000


def _hash(conteudo: bytes) -> str:
    return hashlib.sha256(conteudo).hexdigest()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if DESTINO.exists() and _hash(DESTINO.read_bytes()) == SHA256:
        print(f"{DESTINO} já existe e confere com o hash esperado. Nada a baixar.")
        return

    print(f"Baixando {URL}")
    pedido = urllib.request.Request(URL, headers={"User-Agent": "portfolio-estudos/1.0"})
    with urllib.request.urlopen(pedido, timeout=60) as resposta:
        conteudo = resposta.read()

    if len(conteudo) != TAMANHO or _hash(conteudo) != SHA256:
        raise SystemExit(
            f"O arquivo baixado não confere: {len(conteudo)} bytes, sha256 {_hash(conteudo)}.\n"
            f"Esperado: {TAMANHO} bytes, sha256 {SHA256}. Nada foi gravado."
        )
    linhas = conteudo.count(b"\n") - 1
    if linhas != LINHAS:
        raise SystemExit(f"Esperava {LINHAS} clientes e vieram {linhas}. Nada foi gravado.")

    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    DESTINO.write_bytes(conteudo)
    tamanho, clientes = f"{len(conteudo):,}".replace(",", "."), f"{linhas:,}".replace(",", ".")
    print(f"Gravado em {DESTINO}: {tamanho} bytes, {clientes} clientes, hash conferido.")


if __name__ == "__main__":
    main()
