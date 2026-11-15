"""Constrói a base de conhecimento: avaliações com comentário viram documentos com metadados,
recebem embeddings locais e são indexadas no Chroma persistido em chroma_db/.

Não depende da chave do Gemini. Reexecutar recria a coleção do zero.

    python src/base_conhecimento.py
"""

import csv
import logging
import re
from collections import Counter
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer

RAIZ = Path(__file__).resolve().parent.parent
ARQUIVO_REVIEWS = RAIZ / "data" / "raw" / "olist_order_reviews_dataset.csv"
PASTA_CHROMA = RAIZ / "chroma_db"
NOME_COLECAO = "avaliacoes"
MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LOTE_EMBEDDINGS = 128
TEM_LETRA = re.compile(r"[A-Za-zÀ-ÿ]")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def limpar_texto(texto: str) -> str:
    # Junta quebras de linha internas (\r\n em 3.834 comentários) e espaços repetidos.
    # Caixa, acentos e emojis ficam: o modelo transformer usa o texto natural.
    return " ".join(texto.split())


def montar_texto(titulo: str, mensagem: str) -> str:
    if not titulo:
        return mensagem
    separador = " " if titulo[-1] in ".!?" else ". "
    return f"{titulo}{separador}{mensagem}"


def carregar_documentos() -> list[dict]:
    """Lê o CSV e devolve um documento por review_id com comentário que contenha ao menos uma letra."""
    descartes: Counter[str] = Counter()
    vistos: set[str] = set()
    documentos: list[dict] = []

    with ARQUIVO_REVIEWS.open(encoding="utf-8", newline="") as arquivo:
        for linha in csv.DictReader(arquivo):
            mensagem = limpar_texto(linha["review_comment_message"])
            if not mensagem:
                descartes["sem mensagem"] += 1
                continue
            # 789 review_id aparecem em mais de um pedido da mesma compra, sempre com o mesmo texto e nota.
            if linha["review_id"] in vistos:
                descartes["review_id duplicado"] += 1
                continue
            if not TEM_LETRA.search(mensagem):
                descartes["sem letra"] += 1
                continue
            vistos.add(linha["review_id"])

            titulo = limpar_texto(linha["review_comment_title"])
            data = linha["review_creation_date"][:10]
            documentos.append({
                "id": linha["review_id"],
                "texto": montar_texto(titulo, mensagem),
                "metadados": {
                    "review_score": int(linha["review_score"]),
                    "order_id": linha["order_id"],
                    "data_avaliacao": data,
                    # Chroma só compara números em $gte/$lte, por isso o período também vai como inteiro.
                    "ano_mes": int(data[:4] + data[5:7]),
                    "tem_titulo": bool(titulo),
                },
            })

    total = len(documentos) + sum(descartes.values())
    log.info("Linhas lidas: %s", total)
    for motivo, quantidade in descartes.items():
        log.info("Descartadas (%s): %s", motivo, quantidade)
    log.info("Documentos: %s", len(documentos))
    notas = sorted(Counter(d["metadados"]["review_score"] for d in documentos).items())
    log.info("Documentos por nota: %s", notas)
    return documentos


def indexar(documentos: list[dict]) -> None:
    log.info("Gerando embeddings com %s", MODELO_EMBEDDINGS)
    modelo = SentenceTransformer(MODELO_EMBEDDINGS)
    # Vetores normalizados deixam a distância cosseno comparável entre consultas, base do limiar de evidência.
    embeddings = modelo.encode(
        [d["texto"] for d in documentos],
        batch_size=LOTE_EMBEDDINGS,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    cliente = chromadb.PersistentClient(path=str(PASTA_CHROMA))
    try:
        cliente.delete_collection(NOME_COLECAO)
        log.info("Coleção anterior removida")
    except NotFoundError:
        pass
    colecao = cliente.create_collection(
        NOME_COLECAO,
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )

    lote = cliente.get_max_batch_size()
    for inicio in range(0, len(documentos), lote):
        fatia = documentos[inicio:inicio + lote]
        colecao.add(
            ids=[d["id"] for d in fatia],
            documents=[d["texto"] for d in fatia],
            metadatas=[d["metadados"] for d in fatia],
            embeddings=embeddings[inicio:inicio + lote].tolist(),
        )

    if colecao.count() != len(documentos):
        raise RuntimeError(f"Índice com {colecao.count()} itens, esperado {len(documentos)}")
    log.info("Coleção '%s' em %s com %s documentos", NOME_COLECAO, PASTA_CHROMA, colecao.count())


if __name__ == "__main__":
    indexar(carregar_documentos())
