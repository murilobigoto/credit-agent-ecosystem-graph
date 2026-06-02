"""Camada de abstração de LLM."""

from credito_agentes.llm.client import FakeLLM, LLMClient, RealLLM, get_llm

__all__ = ["FakeLLM", "LLMClient", "RealLLM", "get_llm"]
