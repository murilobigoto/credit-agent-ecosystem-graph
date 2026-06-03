"""
Camada de abstracao do LLM.

POR QUE ABSTRAIR
----------------
Se cada agente chamasse a API do provedor diretamente, trocar de modelo ou
rodar testes sem internet seria um pesadelo. Usamos o padrao interface +
implementacoes: um contrato (LLMClient) e duas implementacoes que o respeitam.

  FakeLLM  -- deterministica, offline, usada em testes e CI.
  RealLLM  -- chama a API Anthropic, configurada via variaveis de ambiente.

O parametro tier ("small" ou "large") permite usar modelos diferentes para
etapas baratas (roteamento, fallback, refinamento) e etapas criticas (decisao).

CONFIGURACAO
------------
Variaveis de ambiente lidas do arquivo .env (ou do ambiente do sistema):

  USE_FAKE_LLM            1 = FakeLLM (offline), 0 = RealLLM (API real)
  ANTHROPIC_API_KEY       chave da API Anthropic (obrigatoria se USE_FAKE_LLM=0)
  MODEL_SMALL             modelo usado para tier="small"
  MODEL_LARGE             modelo usado para tier="large"
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Literal

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

Tier = Literal["small", "large"]

_MODEL_SMALL_DEFAULT = "claude-haiku-4-5-20251001"
_MODEL_LARGE_DEFAULT = "claude-sonnet-4-6"


class LLMClient(ABC):
    """Interface minima de um cliente LLM usada por todos os agentes."""

    @abstractmethod
    def complete(self, system: str, user: str, *, tier: Tier = "small") -> str:
        """Recebe prompts de sistema e usuario e devolve texto.

        tier="small"  -- etapas baratas: roteamento, fallback, refinamento.
        tier="large"  -- etapas criticas: decisao de credito.
        """
        raise NotImplementedError

    def complete_json(self, system: str, user: str, *, tier: Tier = "small") -> dict:
        """Conveniencia: pede JSON e faz parse com tolerancia a cercas ```."""
        raw = self.complete(system, user, tier=tier)
        return _parse_json_lenient(raw)


def _parse_json_lenient(raw: str) -> dict:
    """Parse de JSON com tolerancia a cercas de codigo que LLMs inserem."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        linhas = cleaned.splitlines()
        linhas = [ln for ln in linhas if not ln.strip().startswith("```")]
        cleaned = "\n".join(linhas).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        inicio, fim = cleaned.find("{"), cleaned.rfind("}")
        if inicio != -1 and fim != -1:
            return json.loads(cleaned[inicio: fim + 1])
        raise


class FakeLLM(LLMClient):
    """LLM deterministico para testes e desenvolvimento offline.

    Usa regras baseadas em palavras-chave. A mesma entrada sempre produz
    a mesma saida, o que torna os testes rapidos e reprodutiveis no CI.
    """

    def complete(self, system: str, user: str, *, tier: Tier = "small") -> str:
        texto = user.lower()
        if "roteador" in system.lower():
            return json.dumps(self._fake_route(texto), ensure_ascii=False)
        if "refinamento" in system.lower():
            return user.strip()
        return user.strip()[:280]

    @staticmethod
    def _fake_route(texto: str) -> dict:
        consulta = (
            "saldo", "quanto estou usando", "uso do limite",
            "uso do cartao", "uso do cartão", "extrato", "fatura",
        )
        reneg = ("renegoci", "divida", "dívida", "atraso", "atrasad", "quitar")
        conc = (
            "emprestimo", "empréstimo", "credito novo", "crédito novo",
            "financiamento", "contratar", "novo cartao", "novo cartão",
            "aumentar limite", "pedir credito", "pedir crédito",
        )
        if any(p in texto for p in consulta):
            return {"intent": "bancario_geral", "confidence": 0.86,
                    "rationale": "Consulta de leitura sobre conta/limite."}
        if any(p in texto for p in reneg):
            return {"intent": "renegociacao", "confidence": 0.90,
                    "rationale": "Mencao a divida/renegociacao."}
        if any(p in texto for p in conc):
            return {"intent": "concessao", "confidence": 0.88,
                    "rationale": "Mencao a contratacao de novo produto."}
        if len(texto.strip()) < 8:
            return {"intent": "bancario_geral", "confidence": 0.40,
                    "rationale": "Mensagem curta ou ambigua."}
        return {"intent": "bancario_geral", "confidence": 0.80,
                "rationale": "Duvida bancaria generica (fallback)."}


class RealLLM(LLMClient):
    """Implementacao que chama a API Anthropic.

    Le os modelos de MODEL_SMALL e MODEL_LARGE no ambiente (ou .env).
    Requer ANTHROPIC_API_KEY configurada. Instancia somente quando
    USE_FAKE_LLM=0 e a chave esta presente.
    """

    def __init__(self) -> None:  # pragma: no cover
        self.small_model = os.getenv("MODEL_SMALL", _MODEL_SMALL_DEFAULT)
        self.large_model = os.getenv("MODEL_LARGE", _MODEL_LARGE_DEFAULT)
        try:
            import anthropic  # type: ignore
            self._client = anthropic.Anthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"]
            )
        except KeyError:
            raise RuntimeError(
                "ANTHROPIC_API_KEY nao configurada. "
                "Defina no arquivo .env ou como variavel de ambiente."
            ) from None
        except ImportError as exc:
            raise RuntimeError(
                "SDK anthropic nao instalado. Execute: pip install anthropic"
            ) from exc

    def complete(self, system: str, user: str, *, tier: Tier = "small") -> str:  # pragma: no cover
        model = self.large_model if tier == "large" else self.small_model
        resp = self._client.messages.create(
            model=model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(
            block.text
            for block in resp.content
            if getattr(block, "type", "") == "text"
        )


def get_llm() -> LLMClient:
    """Fabrica: decide qual implementacao usar conforme o ambiente.

    USE_FAKE_LLM=1 (ou ausente) -> FakeLLM.
    USE_FAKE_LLM=0 + ANTHROPIC_API_KEY -> RealLLM.
    USE_FAKE_LLM=0 sem chave -> FakeLLM com aviso no stderr.
    """
    if os.getenv("USE_FAKE_LLM", "1") == "1":
        return FakeLLM()
    if not os.getenv("ANTHROPIC_API_KEY"):
        import sys
        print(
            "AVISO: USE_FAKE_LLM=0 mas ANTHROPIC_API_KEY nao encontrada. "
            "Usando FakeLLM. Defina a chave no arquivo .env.",
            file=sys.stderr,
        )
        return FakeLLM()
    try:  # pragma: no cover
        return RealLLM()
    except Exception as exc:  # noqa: BLE001
        import sys
        print(f"AVISO: Falha ao inicializar RealLLM ({exc}). Usando FakeLLM.", file=sys.stderr)
        return FakeLLM()
