"""Testes das partes determinísticas do pipeline (sem modelo de embeddings, Chroma ou LLM).

    python -m unittest discover tests
"""

import unittest
from unittest.mock import patch

from src.llm import ConsultaReescrita, Tema
from src.pipeline_rag import analisar_com_llm, analisar_pergunta, deduplicar, resumir_temas, transformar_pergunta


def avaliacao(id_: str, texto: str) -> dict:
    return {"id": id_, "texto": texto, "distancia": 0.1, "review_score": 1, "data_avaliacao": "2018-01-01"}


class TestPipeline(unittest.TestCase):
    def test_transformar_remove_meta_linguagem(self) -> None:
        self.assertEqual(
            transformar_pergunta("Quais são os principais problemas relatados pelos clientes em suas avaliações?"),
            "problemas",
        )

    def test_consulta_vazia_usa_sentimento_ou_original(self) -> None:
        analise = analisar_pergunta("Do que os clientes mais reclamam?")
        self.assertEqual(analise["consulta"], "reclamação problema")
        self.assertTrue(analise["frequencia"])
        pergunta = "Quais são os que mais existem?"
        self.assertEqual(analisar_pergunta(pergunta)["consulta"], pergunta)

    def test_filtro_de_nota(self) -> None:
        self.assertEqual(analisar_pergunta("Clientes insatisfeitos")["filtro"], {"review_score": {"$lte": 2}})
        self.assertEqual(analisar_pergunta("Que queixas aparecem sobre os Correios?")["filtro"], {"review_score": {"$lte": 2}})
        self.assertEqual(analisar_pergunta("O que elogiam?")["filtro"], {"review_score": {"$gte": 4}})
        self.assertIsNone(analisar_pergunta("Elogios e reclamações sobre entrega")["filtro"])
        self.assertTrue(analisar_pergunta("Quais reclamações aparecem com maior frequência?")["frequencia"])

    def test_filtro_manual_substitui_automatico(self) -> None:
        self.assertIsNone(analisar_pergunta("Clientes insatisfeitos", "Todas as notas")["filtro"])
        analise = analisar_pergunta("Como é a entrega?", "Notas 4 e 5")
        self.assertEqual(analise["filtro"], {"review_score": {"$gte": 4}})
        self.assertEqual(analise["filtro_descricao"], "notas 4 e 5")

    def test_deduplicar_conta_ocorrencias(self) -> None:
        evidencias = deduplicar([avaliacao("a", "Muito bom"), avaliacao("b", "muito bom!"), avaliacao("c", "Atrasou")])
        self.assertEqual([e["ocorrencias"] for e in evidencias], [2, 1])
        self.assertEqual(evidencias[0]["ids_repetidos"], ["b"])

    def test_resumir_temas_soma_e_ignora_repeticao(self) -> None:
        evidencias = deduplicar([avaliacao("a", "Atrasou"), avaliacao("b", "atrasou"), avaliacao("c", "Não chegou")])
        temas = [
            Tema(tema="Não entrega", resumo="r", evidencias=[2, 99]),
            Tema(tema="Atraso", resumo="r", evidencias=[1, 2]),
        ]
        texto = resumir_temas(temas, evidencias)
        # Atraso soma 2 (E1 repetida 2x) e fica antes; E2 já usada não conta de novo; E99 não existe.
        self.assertLess(texto.index("Atraso (2 avaliações)"), texto.index("Não entrega (1 avaliação)"))
        self.assertIn("[E1]", texto)
        self.assertNotIn("E99", texto)

    def test_reescrita_volta_para_regras_sem_chave_ou_em_falha(self) -> None:
        pergunta = "Clientes insatisfeitos com a entrega"
        with patch("src.pipeline_rag.chave_configurada", return_value=False):
            self.assertEqual(analisar_com_llm(pergunta)["origem"], "regras")
        with patch("src.pipeline_rag.chave_configurada", return_value=True), \
                patch("src.pipeline_rag.reescrever_pergunta", side_effect=RuntimeError("cota")):
            self.assertEqual(analisar_com_llm(pergunta), analisar_pergunta(pergunta))

    def test_reescrita_respeita_filtro_manual(self) -> None:
        reescrita = ConsultaReescrita(consulta="demora na entrega", filtro="negativo", frequencia=True)
        with patch("src.pipeline_rag.chave_configurada", return_value=True), \
                patch("src.pipeline_rag.reescrever_pergunta", return_value=reescrita):
            automatico = analisar_com_llm("Do que mais reclamam?")
            manual = analisar_com_llm("Do que mais reclamam?", "Todas as notas")
        self.assertEqual((automatico["consulta"], automatico["filtro"], automatico["top_k"]),
                         ("demora na entrega", {"review_score": {"$lte": 2}}, 80))
        self.assertIsNone(manual["filtro"])
        self.assertEqual(manual["origem"], "llm")


if __name__ == "__main__":
    unittest.main()
