"""Testes da camada de KB/RAG."""

from datetime import date

from credito_agentes.kb import KnowledgeBase, build_default_kbs, semantic_chunk
from credito_agentes.kb.retrieval import ChunkMetadata


def test_semantic_chunk_respeita_paragrafos():
    texto = "Seção A.\nlinha.\n\nSeção B.\nlinha."
    chunks = semantic_chunk(texto, max_chars=20)
    assert len(chunks) == 2


def test_filtro_vigencia_descarta_politica_revogada():
    kb = KnowledgeBase("teste")
    meta_revogada = ChunkMetadata(
        versao_politica="v1", vigencia_inicio=date(2020, 1, 1),
        vigencia_fim=date(2020, 12, 31), produto="x", segmento="todos",
    )
    kb.add_document("regra antiga sobre taxa", meta_revogada, prefixo="old")
    # Consulta na data atual: a política revogada não deve voltar.
    assert kb.retrieve("taxa", hoje=date.today()) == []


def test_recuperacao_retorna_chunk_relevante():
    kbs = build_default_kbs()
    res = kbs["politicas_concessao"].retrieve(
        "taxa mínima prazo máximo", hoje=date.today(), top_k=3
    )
    assert res
    assert any("taxa" in c.texto.lower() for c in res)


def test_metadados_obrigatorios_presentes():
    kbs = build_default_kbs()
    for c in kbs["faq_bancario"].chunks:
        assert c.metadata.versao_politica
        assert c.metadata.vigencia_inicio
        assert c.metadata.produto
        assert c.metadata.segmento
