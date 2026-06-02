"""
Agente Bancário Simplificado (a rota de fallback inteligente).
==================================================================

PAPEL DESTE AGENTE (explicação a nível de aula)
-----------------------------------------------
Quando a demanda do cliente NÃO é concessão nem renegociação, o roteador manda
para cá. Este agente é uma "camada de contenção": resolve dúvidas comuns sem
acionar uma análise de crédito completa (que seria cara e desnecessária).

CARACTERÍSTICAS DE SEGURANÇA
----------------------------
  * READ-ONLY: só usa tools de leitura (saldo, uso de limite). Nunca altera
    estado. Sem transferências, sem pagamentos.
  * ESCOPO LIMITADO: se a pergunta foge do tema bancário, responde com uma
    mensagem padrão de redirecionamento em vez de tentar adivinhar.
  * FUNDAMENTADO EM FAQ: usa RAG sobre a KB de FAQ para responder com base em
    conteúdo real, reduzindo alucinação.

DEFINIÇÃO — "cache semântico": guardar respostas de perguntas já vistas para
devolvê-las rapidamente quando uma pergunta MUITO parecida chegar. Reduz
latência e custo em perguntas frequentes.
"""

from __future__ import annotations

import re
from datetime import date

from credito_agentes.kb import KnowledgeBase
from credito_agentes.llm import LLMClient
from credito_agentes.schemas import GraphState
from credito_agentes.tools import (
    READ_ONLY_BANKING_TOOLS,
    SaldoOutput,
    UsoLimiteOutput,
    with_timeout,
)

# Palavras que indicam intenção de leitura de dados da conta.
_INTENT_SALDO = re.compile(r"\bsaldo\b", re.I)
_INTENT_CARTAO = re.compile(r"(limite|uso).{0,15}cart[ãa]o", re.I)
_INTENT_CHEQUE = re.compile(r"(cheque especial|limite.{0,15}conta)", re.I)

# Termos que mantêm a conversa dentro do escopo bancário.
_ESCOPO_BANCARIO = re.compile(
    r"(saldo|cart[ãa]o|limite|conta|fatura|cheque|banco|pix|transfer|extrato)",
    re.I,
)

_REDIRECIONAMENTO = (
    "Esse assunto está fora do que consigo atender por aqui. Posso ajudar com "
    "saldo, limites de cartão e conta, e dúvidas sobre seus produtos. Quer falar "
    "sobre algum desses?"
)


class SemanticCache:
    """Cache semântico simples baseado em sobreposição de palavras.

    Em produção, troque por similaridade de embeddings. A interface é a mesma:
    `get(pergunta)` e `put(pergunta, resposta)`.
    """

    def __init__(self, *, limiar: float = 0.8) -> None:
        self.limiar = limiar
        self._itens: list[tuple[set[str], str]] = []

    def get(self, pergunta: str) -> str | None:
        toks = set(re.findall(r"\w+", pergunta.lower()))
        for chave, resposta in self._itens:
            uniao = toks | chave
            if uniao and len(toks & chave) / len(uniao) >= self.limiar:
                return resposta
        return None

    def put(self, pergunta: str, resposta: str) -> None:
        self._itens.append((set(re.findall(r"\w+", pergunta.lower())), resposta))


class BankingAgent:
    """Agente de fallback bancário."""

    def __init__(self, llm: LLMClient, faq_kb: KnowledgeBase) -> None:
        self.llm = llm
        self.faq_kb = faq_kb
        self.cache = SemanticCache()

    def run(self, state: GraphState) -> GraphState:
        msg = state.mensagem
        cliente_id = state.cliente_id

        # 0. Fora de escopo → redirecionamento padrão.
        if not _ESCOPO_BANCARIO.search(msg):
            state.raw_answer = _REDIRECIONAMENTO
            return state

        # 1. Cache semântico (responde rápido perguntas frequentes).
        cached = self.cache.get(msg)
        if cached is not None:
            state.raw_answer = cached
            return state

        # 2. Tools READ-ONLY conforme a intenção detectada.
        partes: list[str] = []
        if _INTENT_SALDO.search(msg):
            r = with_timeout(
                lambda: READ_ONLY_BANKING_TOOLS["consulta_saldo"](cliente_id),
                timeout_s=2.0,
                fallback=SaldoOutput(cliente_id=cliente_id, saldo=0.0),
                tool_name="consulta_saldo",
            )
            saldo = r.value.saldo  # type: ignore[attr-defined]
            partes.append(f"Seu saldo em conta é de R$ {saldo:.2f}.")

        if _INTENT_CARTAO.search(msg):
            r = with_timeout(
                lambda: READ_ONLY_BANKING_TOOLS["uso_limite_cartao"](cliente_id),
                timeout_s=2.0,
                fallback=UsoLimiteOutput(cliente_id=cliente_id, percentual_uso=0.0),
                tool_name="uso_limite_cartao",
            )
            uso = r.value.percentual_uso  # type: ignore[attr-defined]
            partes.append(f"Você está usando {uso * 100:.0f}% do limite do cartão.")

        if _INTENT_CHEQUE.search(msg):
            r = with_timeout(
                lambda: READ_ONLY_BANKING_TOOLS["uso_limite_cheque_especial"](cliente_id),
                timeout_s=2.0,
                fallback=UsoLimiteOutput(cliente_id=cliente_id, percentual_uso=0.0),
                tool_name="uso_limite_cheque_especial",
            )
            uso = r.value.percentual_uso  # type: ignore[attr-defined]
            partes.append(
                f"Você está usando {uso * 100:.0f}% do limite da conta/cheque especial."
            )

        # 3. Se nenhuma tool casou, responde via RAG sobre o FAQ.
        if not partes:
            trechos = self.faq_kb.retrieve(msg, hoje=date.today(), top_k=2)
            if trechos:
                contexto = "\n".join(c.texto for c in trechos)
                partes.append(self._responder_com_faq(msg, contexto))
            else:
                partes.append(_REDIRECIONAMENTO)

        resposta = " ".join(partes)
        self.cache.put(msg, resposta)
        state.raw_answer = resposta
        return state

    def _responder_com_faq(self, pergunta: str, contexto: str) -> str:
        """Usa o LLM (modelo pequeno) para responder ancorado no FAQ."""
        system = (
            "Você é um assistente bancário de fallback. Responda APENAS com base "
            "no contexto fornecido, de forma curta e objetiva."
        )
        user = f"Contexto:\n{contexto}\n\nPergunta: {pergunta}"
        return self.llm.complete(system, user, tier="small")
