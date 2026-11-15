"""Pipeline RAG: pergunta → filtro de nota → embedding → recuperação no Chroma → limiar de evidência
→ deduplicação → contexto → resposta do LLM com as avaliações citadas.

    python -m src.pipeline_rag "O que os clientes relatam sobre problemas na entrega?"
"""

import logging
import re
import sys
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from src.base_conhecimento import MODELO_EMBEDDINGS, NOME_COLECAO, PASTA_CHROMA
from src.llm import Tema, gerar_resposta, gerar_temas

# Distância cosseno máxima para uma avaliação contar como evidência. Calibrado em scripts/calibrar_limiar.py.
LIMIAR_DISTANCIA = 0.35
TOP_K = 10
# Perguntas de frequência pedem uma amostra maior para o LLM agrupar por tema.
TOP_K_FREQUENCIA = 80
MAX_EVIDENCIAS_CONTEXTO = 40

PADRAO_FREQUENCIA = re.compile(r"frequen|mais comu|principa|recorrent|padr[õo]es|mais relatad|mais elogia|aparecem", re.I)
PADRAO_NEGATIVO = re.compile(r"insatisf|reclama|problema|nota baixa|negativ|ruim|defeito|atras", re.I)
PADRAO_POSITIVO = re.compile(r"elogi|positiv|(?<!in)satisfeit|valoriz|gostam|nota alta|recomend", re.I)
# Palavras de pergunta e de meta-linguagem ("relatados pelos clientes nas avaliações") afastam o embedding
# do texto das avaliações: na calibração, "principais problemas relatados..." ficou a 0,393 e "problemas" a 0,266.
PADRAO_META = re.compile(
    r"\b(qual|quais|que|o que|são|é|os|as|o|a|principais|mais|maior|frequ[eê]ncia|aparecem|com|"
    r"relatad[oa]s?|relatam|dizem|falam|pelos|dos|das|de|do|da|em|nas|nos|suas?|seus?|clientes?|consumidores|"
    r"avalia[cç][õo]es|podem|ser|identificad[oa]s|existem|há|padr[õo]es|recorrentes|sustentam|"
    r"relacionad[oa]s|à|ao|aspectos|outros|isso|algum|sobre|reclama[cç][õo]es|reclamam|elogiam|elogiad[oa]s)\b",
    re.I,
)

RESPOSTA_SEM_EVIDENCIA = (
    "Não encontrei avaliações suficientemente relacionadas a essa pergunta na base. "
    "Ela pode estar fora do escopo das avaliações de clientes da Olist."
)

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def carregar_modelo() -> SentenceTransformer:
    return SentenceTransformer(MODELO_EMBEDDINGS)


@lru_cache(maxsize=1)
def carregar_colecao() -> chromadb.Collection:
    if not PASTA_CHROMA.exists():
        sys.exit("Índice não encontrado. Rode antes: python src/base_conhecimento.py")
    return chromadb.PersistentClient(path=str(PASTA_CHROMA)).get_collection(NOME_COLECAO)


def transformar_pergunta(pergunta: str) -> str:
    """Remove a meta-linguagem para a consulta se parecer com o texto de uma avaliação."""
    consulta = " ".join(PADRAO_META.sub(" ", re.sub(r"[?!.,;]", " ", pergunta)).split())
    # Pergunta feita só de meta-linguagem ("Quais aspectos são mais elogiados?") fica sem nada: usa a original.
    return consulta or pergunta


def analisar_pergunta(pergunta: str) -> dict:
    """Decide consulta, filtro de nota e tamanho da amostra a partir de palavras da pergunta."""
    negativa = bool(PADRAO_NEGATIVO.search(pergunta))
    positiva = bool(PADRAO_POSITIVO.search(pergunta))
    filtro = None
    # Só filtra quando o sentido é inequívoco; pergunta mista busca em todas as notas.
    if negativa and not positiva:
        filtro = {"review_score": {"$lte": 2}}
    elif positiva and not negativa:
        filtro = {"review_score": {"$gte": 4}}
    frequencia = bool(PADRAO_FREQUENCIA.search(pergunta))
    return {
        "consulta": transformar_pergunta(pergunta),
        "filtro": filtro,
        "frequencia": frequencia,
        "top_k": TOP_K_FREQUENCIA if frequencia else TOP_K,
    }


def buscar(consulta: str, top_k: int, filtro: dict | None = None) -> list[dict]:
    """Devolve as avaliações mais próximas com a distância cosseno, sem aplicar o limiar."""
    vetor = carregar_modelo().encode([consulta], normalize_embeddings=True, show_progress_bar=False).tolist()
    resultado = carregar_colecao().query(query_embeddings=vetor, n_results=top_k, where=filtro)
    return [
        {"id": id_, "texto": texto, "distancia": distancia, **metadados}
        for id_, texto, distancia, metadados in zip(
            resultado["ids"][0], resultado["documents"][0], resultado["distances"][0], resultado["metadatas"][0]
        )
    ]


def chave_texto(texto: str) -> str:
    return re.sub(r"[^\w\s]", "", texto.lower()).strip()


def deduplicar(avaliacoes: list[dict]) -> list[dict]:
    """Agrupa textos idênticos ("Muito bom", "Bom.") mantendo a primeira ocorrência e a contagem."""
    grupos: dict[str, dict] = {}
    for avaliacao in avaliacoes:
        chave = chave_texto(avaliacao["texto"])
        if chave in grupos:
            grupos[chave]["ocorrencias"] += 1
            grupos[chave]["ids_repetidos"].append(avaliacao["id"])
        else:
            grupos[chave] = {**avaliacao, "ocorrencias": 1, "ids_repetidos": []}
    return list(grupos.values())


def montar_contexto(evidencias: list[dict]) -> str:
    linhas = []
    for numero, evidencia in enumerate(evidencias, start=1):
        repeticao = f", repetida {evidencia['ocorrencias']}x" if evidencia["ocorrencias"] > 1 else ""
        linhas.append(
            f"[E{numero}] (nota {evidencia['review_score']}, {evidencia['data_avaliacao']}{repeticao}) {evidencia['texto']}"
        )
    return "\n".join(linhas)


def resumir_temas(temas: list[Tema], evidencias: list[dict]) -> str:
    """Soma as ocorrências reais de cada tema e monta a resposta ordenada pela soma."""
    usadas: set[int] = set()
    linhas = []
    for tema in temas:
        # Descarta números inexistentes e evidência já usada em outro tema, para nenhuma ser contada duas vezes.
        numeros = [n for n in dict.fromkeys(tema.evidencias) if 1 <= n <= len(evidencias) and n not in usadas]
        usadas.update(numeros)
        if numeros:
            soma = sum(evidencias[n - 1]["ocorrencias"] for n in numeros)
            linhas.append((soma, tema.tema, tema.resumo, numeros))
    linhas.sort(key=lambda linha: linha[0], reverse=True)
    total = sum(e["ocorrencias"] for e in evidencias)
    texto = [f"Temas recorrentes na amostra de {total} avaliações recuperadas (não é uma contagem sobre a base inteira):"]
    for soma, nome, resumo, numeros in linhas:
        citacoes = ", ".join(f"E{n}" for n in numeros)
        texto.append(f"- {nome} ({soma} avaliações): {resumo} [{citacoes}]")
    return "\n".join(texto)


def responder(pergunta: str) -> dict:
    """Executa o pipeline e devolve a resposta com as evidências usadas e os parâmetros da busca."""
    analise = analisar_pergunta(pergunta)
    recuperadas = buscar(analise["consulta"], analise["top_k"], analise["filtro"])
    relevantes = [a for a in recuperadas if a["distancia"] <= LIMIAR_DISTANCIA]
    evidencias = deduplicar(relevantes)[:MAX_EVIDENCIAS_CONTEXTO]
    log.info(
        "consulta=%r filtro=%s frequencia=%s recuperadas=%s dentro_do_limiar=%s evidencias=%s",
        analise["consulta"], analise["filtro"], analise["frequencia"], len(recuperadas), len(relevantes), len(evidencias),
    )

    if not evidencias:
        # Sem evidência acima do limiar o LLM não é chamado: nada a fundamentar e nenhuma cota gasta.
        return {"resposta": RESPOSTA_SEM_EVIDENCIA, "evidencias": [], **analise}

    contexto = montar_contexto(evidencias)
    if analise["frequencia"]:
        resposta = resumir_temas(gerar_temas(pergunta, contexto), evidencias)
    else:
        resposta = gerar_resposta(pergunta, contexto)
    return {"resposta": resposta, "evidencias": evidencias, **analise}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if len(sys.argv) < 2:
        sys.exit('Uso: python -m src.pipeline_rag "sua pergunta"')
    saida = responder(" ".join(sys.argv[1:]))
    print(saida["resposta"])
    print()
    for numero, evidencia in enumerate(saida["evidencias"], start=1):
        print(f"[E{numero}] dist={evidencia['distancia']:.3f} nota={evidencia['review_score']} x{evidencia['ocorrencias']} {evidencia['texto']}")
