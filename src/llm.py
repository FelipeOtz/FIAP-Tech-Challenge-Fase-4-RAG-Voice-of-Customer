"""Única porta de saída para o modelo de linguagem. Trocar de provedor exige mexer só neste arquivo."""

import logging
import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

ARQUIVO_ENV = Path(__file__).resolve().parent.parent / ".env"
MAX_TENTATIVAS = 2
ESPERA_ENTRE_TENTATIVAS_S = 5
# Só vale tentar de novo em sobrecarga momentânea; 404 (modelo errado) ou cota diária esgotada não melhoram.
CODIGOS_TRANSITORIOS = {500, 503}
TEMPERATURA = 0.2

INSTRUCAO_SISTEMA = """Você é um analista de Voice of Customer de um e-commerce brasileiro.
Responda em português, usando apenas as avaliações de clientes fornecidas como evidência.
Regras:
- Cada afirmação deve citar as evidências que a sustentam, no formato [E1], [E2].
- Não use conhecimento externo nem invente números, produtos, lojas ou causas que as avaliações não mencionam.
- Se as evidências não respondem à pergunta, ou só respondem em parte, diga isso explicitamente.
- Seja direto: no máximo 3 parágrafos curtos ou uma lista curta."""

INSTRUCAO_TEMAS = """Esta é uma pergunta sobre frequência ou padrões. Agrupe as evidências em no máximo 5 temas.
Cada evidência entra em um único tema; deixe de fora as que não tratam do assunto da pergunta.
Para cada tema devolva um nome curto, um resumo de uma frase baseado só nas evidências do tema
(sem citações no resumo) e a lista dos números das evidências (por exemplo, 3 para [E3]). Não conte nem some nada."""


INSTRUCAO_REESCRITA = """Você converte perguntas de analistas sobre avaliações de clientes de um e-commerce brasileiro
em parâmetros de busca semântica sobre o texto das avaliações.
- consulta: frase curta com as palavras que um cliente usaria numa avaliação sobre o assunto (por exemplo,
  "demora na entrega" ou "produto veio diferente da foto"), sem palavras de pergunta. Se a pergunta não tem relação
  com compras, entregas, produtos, lojas ou atendimento, devolva a própria pergunta, sem adaptá-la.
- filtro: "negativo" se a pergunta busca reclamações, problemas ou insatisfação; "positivo" se busca elogios ou
  satisfação; "todas" se é neutra ou mistura os dois sentidos.
- frequencia: true se a pergunta pede o que é mais comum, principal, recorrente ou um padrão; false caso contrário."""


class Tema(BaseModel):
    tema: str
    resumo: str
    evidencias: list[int]


class ConsultaReescrita(BaseModel):
    consulta: str
    filtro: Literal["negativo", "positivo", "todas"]
    frequencia: bool

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def carregar_cliente() -> tuple[genai.Client, str]:
    load_dotenv(ARQUIVO_ENV)
    chave = os.getenv("GEMINI_API_KEY", "").strip()
    modelo = os.getenv("GEMINI_MODEL", "").strip()
    if not chave or not modelo:
        sys.exit(
            "GEMINI_API_KEY e GEMINI_MODEL precisam estar no arquivo .env. "
            "Copie .env.example para .env e preencha com a chave do Google AI Studio."
        )
    return genai.Client(api_key=chave), modelo


def chave_configurada() -> bool:
    load_dotenv(ARQUIVO_ENV)
    return bool(os.getenv("GEMINI_API_KEY", "").strip() and os.getenv("GEMINI_MODEL", "").strip())


def reescrever_pergunta(pergunta: str) -> ConsultaReescrita:
    """Uma única tentativa, sem nova chamada em erro: quem chama volta para as regras e não gasta mais cota."""
    cliente, modelo = carregar_cliente()
    configuracao = types.GenerateContentConfig(
        system_instruction=INSTRUCAO_REESCRITA,
        temperature=0,
        response_mime_type="application/json",
        response_schema=ConsultaReescrita,
    )
    resposta = cliente.models.generate_content(model=modelo, contents=pergunta, config=configuracao)
    if resposta.parsed is None:
        raise ValueError(f"Resposta do modelo fora do formato de consulta: {resposta.text[:200]}")
    return resposta.parsed


def gerar_resposta(pergunta: str, contexto: str) -> str:
    configuracao = types.GenerateContentConfig(system_instruction=INSTRUCAO_SISTEMA, temperature=TEMPERATURA)
    return chamar_modelo(pergunta, contexto, configuracao).text


def gerar_temas(pergunta: str, contexto: str) -> list[Tema]:
    """Agrupa as evidências por tema em saída estruturada; a contagem fica com o código, não com o modelo."""
    configuracao = types.GenerateContentConfig(
        system_instruction=INSTRUCAO_SISTEMA + "\n\n" + INSTRUCAO_TEMAS,
        temperature=TEMPERATURA,
        response_mime_type="application/json",
        response_schema=list[Tema],
    )
    resposta = chamar_modelo(pergunta, contexto, configuracao)
    if resposta.parsed is None:
        raise ValueError(f"Resposta do modelo fora do formato de temas: {resposta.text[:200]}")
    return resposta.parsed


def chamar_modelo(pergunta: str, contexto: str, configuracao: types.GenerateContentConfig) -> types.GenerateContentResponse:
    cliente, modelo = carregar_cliente()
    conteudo = f"Avaliações recuperadas:\n{contexto}\n\nPergunta: {pergunta}"

    for tentativa in range(1, MAX_TENTATIVAS + 1):
        try:
            return cliente.models.generate_content(model=modelo, contents=conteudo, config=configuracao)
        except errors.APIError as erro:
            if erro.code not in CODIGOS_TRANSITORIOS or tentativa == MAX_TENTATIVAS:
                log.error("Falha no modelo %s (código %s): %s", modelo, erro.code, erro.message)
                raise
            log.warning("Modelo indisponível (código %s), nova tentativa em %ss", erro.code, ESPERA_ENTRE_TENTATIVAS_S)
            time.sleep(ESPERA_ENTRE_TENTATIVAS_S)
    raise RuntimeError("inalcançável")
