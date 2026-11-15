"""Investigação exploratória dos arquivos em data/raw/.

Ferramenta manual: não faz parte do pipeline. Serve para validar a estrutura real dos dados
antes de codificar a base de conhecimento. Usa só a biblioteca padrão, então roda sem a .venv:

    python3 scripts/investigar_dados.py
"""

import csv
import logging
import random
import re
from collections import Counter
from pathlib import Path

PASTA_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
ARQUIVO_REVIEWS = PASTA_RAW / "olist_order_reviews_dataset.csv"
EXEMPLOS_POR_NOTA = 2
SEMENTE = 1
TEM_LETRA = re.compile(r"[A-Za-zÀ-ÿ]")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def vazio(valor: str) -> bool:
    return valor.strip() == ""


def listar_arquivos() -> None:
    log.info("== Arquivos em %s", PASTA_RAW)
    for arquivo in sorted(PASTA_RAW.glob("*.csv")):
        log.info("%-45s %8.1f MB", arquivo.name, arquivo.stat().st_size / 1e6)


def verificar_encoding(bruto: bytes) -> None:
    log.info("== Encoding")
    log.info("BOM UTF-8: %s", bruto.startswith(b"\xef\xbb\xbf"))
    try:
        bruto.decode("utf-8")
        log.info("UTF-8 estrito: válido")
    except UnicodeDecodeError as erro:
        log.warning("UTF-8 estrito: inválido (%s)", erro)
    log.info("Fins de linha CRLF: %s | CR isolado: %s", bruto.count(b"\r\n"), bruto.count(b"\r") - bruto.count(b"\r\n"))
    log.info("Linhas físicas: %s", bruto.count(b"\n"))


def inferir_tipo(valores: list[str]) -> str:
    preenchidos = [v for v in valores if not vazio(v)]
    if not preenchidos:
        return "vazia"
    if all(v.isdigit() for v in preenchidos):
        return "inteiro"
    if all(re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", v) for v in preenchidos):
        return "data e hora"
    return "texto"


def investigar_reviews() -> None:
    bruto = ARQUIVO_REVIEWS.read_bytes()
    verificar_encoding(bruto)

    with ARQUIVO_REVIEWS.open(encoding="utf-8", newline="") as arquivo:
        linhas = list(csv.DictReader(arquivo))
    colunas = list(linhas[0])

    log.info("== Estrutura")
    log.info("Linhas (registros): %s", len(linhas))
    for coluna in colunas:
        valores = [linha[coluna] for linha in linhas]
        log.info("%-25s tipo=%-12s vazios=%s", coluna, inferir_tipo(valores), sum(vazio(v) for v in valores))

    com_mensagem = [l for l in linhas if not vazio(l["review_comment_message"])]
    com_titulo = [l for l in linhas if not vazio(l["review_comment_title"])]
    log.info("== Comentários")
    log.info("Com mensagem: %s (%.1f%%)", len(com_mensagem), 100 * len(com_mensagem) / len(linhas))
    log.info("Com título: %s | só título, sem mensagem: %s", len(com_titulo), sum(vazio(l["review_comment_message"]) for l in com_titulo))

    log.info("== Distribuição de review_score")
    for nome, grupo in (("todas", linhas), ("com mensagem", com_mensagem)):
        contagem = sorted(Counter(l["review_score"] for l in grupo).items())
        log.info("%-13s %s", nome, " | ".join(f"{nota}: {n}" for nota, n in contagem))

    log.info("== Datas")
    for coluna in ("review_creation_date", "review_answer_timestamp"):
        datas = [l[coluna] for l in linhas]
        log.info("%-25s %s a %s", coluna, min(datas), max(datas))

    log.info("== Duplicatas por review_id")
    contagem_ids = Counter(l["review_id"] for l in linhas)
    repetidos = {rid for rid, n in contagem_ids.items() if n > 1}
    grupos: dict[str, set[tuple[str, str]]] = {}
    for l in linhas:
        if l["review_id"] in repetidos:
            grupos.setdefault(l["review_id"], set()).add((l["review_score"], l["review_comment_message"]))
    identicos = sum(len(conteudo) == 1 for conteudo in grupos.values())
    log.info("IDs distintos: %s | repetidos: %s | linhas envolvidas: %s", len(contagem_ids), len(repetidos), sum(contagem_ids[r] for r in repetidos))
    log.info("Repetidos com nota e mensagem idênticas: %s de %s", identicos, len(repetidos))
    log.info("Mensagens únicas por review_id: %s", len({l["review_id"] for l in com_mensagem}))

    log.info("== Conteúdo das mensagens")
    mensagens = [l["review_comment_message"] for l in com_mensagem]
    log.info("Com quebra de linha interna: %s", sum("\n" in m or "\r" in m for m in mensagens))
    tamanhos = sorted(len(m.strip()) for m in mensagens)
    log.info(
        "Tamanho (caracteres) min=%s mediana=%s p90=%s max=%s",
        tamanhos[0], tamanhos[len(tamanhos) // 2], tamanhos[9 * len(tamanhos) // 10], tamanhos[-1],
    )
    log.info("Sem nenhuma letra: %s", sum(not TEM_LETRA.search(m) for m in mensagens))
    log.info("Textos mais repetidos: %s", Counter(m.strip() for m in mensagens).most_common(5))

    log.info("== Exemplos reais (semente %s)", SEMENTE)
    sorteio = random.Random(SEMENTE)
    for nota in "12345":
        grupo = [l for l in com_mensagem if l["review_score"] == nota]
        for l in sorteio.sample(grupo, EXEMPLOS_POR_NOTA):
            texto = " ".join(l["review_comment_message"].split())
            log.info("[nota %s] %s | %s", nota, l["review_comment_title"].strip() or "(sem título)", texto)


if __name__ == "__main__":
    listar_arquivos()
    investigar_reviews()
