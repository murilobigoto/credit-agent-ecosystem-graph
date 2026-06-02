"""Bases de conhecimento (RAG)."""

from credito_agentes.kb.registry import build_default_kbs
from credito_agentes.kb.retrieval import (
    Chunk,
    ChunkMetadata,
    KnowledgeBase,
    semantic_chunk,
)

__all__ = [
    "Chunk",
    "ChunkMetadata",
    "KnowledgeBase",
    "build_default_kbs",
    "semantic_chunk",
]
