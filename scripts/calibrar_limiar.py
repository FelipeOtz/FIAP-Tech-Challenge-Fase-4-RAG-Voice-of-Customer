"""Calibração do limiar de distância usado em src/pipeline_rag.py.

Ferramenta manual. Mede a distância cosseno da avaliação mais próxima (com o mesmo filtro de nota
que o pipeline aplicaria) para perguntas dentro e fora do escopo, e quantas avaliações do top-k
passam em cada limiar candidato. Não chama o LLM.

    python -m scripts.calibrar_limiar          # transformação por regras
    python -m scripts.calibrar_limiar --llm    # reescrita pelo Gemini (1 chamada por pergunta)
"""

import logging
import sys
import time

from src.pipeline_rag import analisar_com_llm, analisar_pergunta, buscar

USAR_LLM = "--llm" in sys.argv
# O nível gratuito aceita 15 requisições por minuto; sem espaçar, parte das perguntas cai nas regras.
ESPERA_ENTRE_CHAMADAS_S = 4.5
LIMIARES_CANDIDATOS = (0.25, 0.30, 0.35, 0.40, 0.45)

DENTRO_DO_ESCOPO = (
    "Quais são os principais problemas relatados pelos clientes em suas avaliações?",
    "O que os clientes relatam sobre problemas relacionados à entrega?",
    "Quais aspectos da experiência de compra são mais elogiados pelos clientes?",
    "Quais padrões podem ser identificados nas avaliações de clientes insatisfeitos?",
    "Quais avaliações sustentam que existem problemas recorrentes com o prazo de entrega?",
    "Quais reclamações relacionadas à entrega aparecem com maior frequência?",
    "Existem padrões recorrentes nas avaliações relacionadas à qualidade dos produtos?",
    "Os clientes reclamam de produto com defeito?",
    "Recebi o produto errado, outros clientes relatam isso?",
    "O que dizem sobre o atendimento da loja?",
    "Há reclamações sobre embalagem danificada?",
    "Clientes elogiam a rapidez da entrega?",
    "Algum cliente comprou mais de um item e recebeu só parte do pedido?",
    "O que falam sobre o custo-benefício dos produtos?",
)

FORA_DO_ESCOPO = (
    "Qual é a capital da França?",
    "Quem ganhou a Copa do Mundo de 2002?",
    "Como faço um bolo de chocolate?",
    "Explique a teoria da relatividade.",
    "Qual a cotação do dólar hoje?",
    "Escreva um poema sobre o mar.",
    "Qual o melhor time de futebol do Brasil?",
    "Como instalar o Python no Windows?",
    "Qual a previsão do tempo para amanhã em São Paulo?",
    "Quantos planetas existem no sistema solar?",
    "Qual o faturamento anual da Olist?",
    "Quanto custa o frete para o Japão?",
)

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger(__name__)


def medir(perguntas: tuple[str, ...]) -> list[float]:
    distancias = []
    for pergunta in perguntas:
        if USAR_LLM:
            time.sleep(ESPERA_ENTRE_CHAMADAS_S)
        analise = analisar_com_llm(pergunta) if USAR_LLM else analisar_pergunta(pergunta)
        resultados = buscar(analise["consulta"], analise["top_k"], analise["filtro"])
        valores = [r["distancia"] for r in resultados]
        passam = " ".join(f"{sum(v <= l for v in valores):>2}" for l in LIMIARES_CANDIDATOS)
        log.info("%.3f  [%s]  %r | %s", valores[0], passam, analise["consulta"], resultados[0]["texto"][:50])
        distancias.append(valores[0])
    return distancias


if __name__ == "__main__":
    log.info("Distância do top-1 e quantas do top-k do pipeline (10, ou 80 em frequência) passam nos limiares %s", LIMIARES_CANDIDATOS)
    log.info("== Dentro do escopo")
    dentro = medir(DENTRO_DO_ESCOPO)
    log.info("== Fora do escopo")
    fora = medir(FORA_DO_ESCOPO)
    log.info("== Resumo")
    log.info("Dentro: maior top-1 = %.3f | Fora: menor top-1 = %.3f", max(dentro), min(fora))
    for limiar in LIMIARES_CANDIDATOS:
        log.info(
            "limiar %.2f: dentro respondidas %s/%s | fora bloqueadas %s/%s",
            limiar, sum(d <= limiar for d in dentro), len(dentro), sum(d > limiar for d in fora), len(fora),
        )
