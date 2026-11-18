"""Interface Streamlit do Voice of Customer: pergunta, resposta fundamentada e avaliações usadas como evidência.

    streamlit run app.py
"""

import logging

import streamlit as st

from src.pipeline_rag import FILTRO_AUTOMATICO, LIMIAR_DISTANCIA, OPCOES_FILTRO, responder

EXEMPLOS = (
    "Quais reclamações relacionadas à entrega aparecem com maior frequência?",
    "Quais aspectos da experiência de compra são mais elogiados pelos clientes?",
    "Quais padrões podem ser identificados nas avaliações de clientes insatisfeitos?",
    "Os clientes reclamam de produto com defeito?",
    "Qual é a capital da França?",
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

st.set_page_config(page_title="Voice of Customer Olist", layout="wide")
st.title("Voice of Customer: avaliações Olist")
st.caption(
    "Perguntas em linguagem natural respondidas com base em 40.482 avaliações escritas de clientes (2016 a 2018). "
    "Cada afirmação cita as avaliações que a sustentam."
)

with st.sidebar:
    st.subheader("Exemplos")
    for exemplo in EXEMPLOS:
        if st.button(exemplo, use_container_width=True):
            st.session_state["pergunta"] = exemplo

with st.form("pergunta_form"):
    pergunta = st.text_area("Pergunta", key="pergunta", height=80)
    filtro_manual = st.radio(
        "Filtro de nota",
        list(OPCOES_FILTRO),
        horizontal=True,
        help="Automático decide pela pergunta: reclamação ou insatisfação busca notas 1 e 2, elogio busca notas 4 e 5.",
    )
    enviar = st.form_submit_button("Perguntar", type="primary")

if enviar and pergunta.strip():
    try:
        with st.spinner("Buscando avaliações e gerando a resposta..."):
            saida = responder(pergunta.strip(), filtro_manual)
    except SystemExit as erro:
        # As funções do pipeline encerram com mensagem de configuração (chave ausente, índice não construído).
        st.error(str(erro))
        st.stop()
    except Exception as erro:
        log.exception("Falha ao responder a pergunta")
        st.error(f"Não foi possível gerar a resposta: {erro}")
        st.stop()

    origem = "automático" if filtro_manual == FILTRO_AUTOMATICO else "escolhido manualmente"
    modo = "frequência (amostra maior, agrupada por tema)" if saida["frequencia"] else "pergunta direta"
    analise = "Gemini" if saida["origem"] == "llm" else "regras"
    st.info(
        f"Filtro de nota: **{saida['filtro_descricao']}** ({origem}) · Consulta usada na busca ({analise}): "
        f"*{saida['consulta']}* · Modo: {modo} · {saida['dentro_do_limiar']} de {saida['recuperadas']} "
        f"avaliações recuperadas dentro do limiar de distância {LIMIAR_DISTANCIA}"
    )

    st.subheader("Resposta")
    st.markdown(saida["resposta"])

    if saida["evidencias"]:
        st.subheader(f"Evidências ({len(saida['evidencias'])} textos distintos)")
        st.dataframe(
            [
                {
                    "Ref.": f"E{numero}",
                    "Avaliação": evidencia["texto"],
                    "Nota": evidencia["review_score"],
                    "Data": evidencia["data_avaliacao"],
                    "Repetições": evidencia["ocorrencias"],
                    "Distância": round(evidencia["distancia"], 3),
                    "review_id": evidencia["id"],
                }
                for numero, evidencia in enumerate(saida["evidencias"], start=1)
            ],
            hide_index=True,
            use_container_width=True,
        )
elif enviar:
    st.warning("Escreva uma pergunta.")
