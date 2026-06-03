"""
Roteador de Entrada (o "Supervisor" do padrão Supervisor + Workers).
==================================================================

PAPEL
----------------------------------
O roteador é a porta de entrada. Ele:
  1. Roda os guardrails de entrada (PII, injection, jailbreak).
  2. Classifica a intenção da mensagem em uma das três rotas.
  3. Produz uma SAÍDA ESTRUTURADA (RoutingDecision) — não texto livre.
  4. Se a confiança for menor que o limiar (0.7), pede clarificação em vez de
     arriscar um roteamento errado.

DEFINIÇÃO — "padrão Supervisor + Workers": um agente supervisor decide PARA ONDE
mandar o trabalho; agentes "workers" (as células) executam. É simples de
depurar porque o fluxo é determinístico: dado o RoutingDecision, sabemos
exatamente qual worker roda.

POR QUE LIMIAR DE CONFIANÇA
---------------------------
Rotear errado é caro: mandar uma renegociação para a concessão gera uma resposta
inútil. Quando o classificador não tem certeza (confiança < 0.7), é melhor
perguntar do que adivinhar. Essa é uma decisão de PRODUTO codificada em regra.
"""

from __future__ import annotations

from credito_agentes.guardrails import check_input
from credito_agentes.llm import LLMClient
from credito_agentes.schemas import GraphState, Intent, RoutingDecision

LIMIAR_CONFIANCA = 0.7

_SYSTEM_ROTEADOR = (
    "Você é o roteador de um atendimento bancário de crédito. Classifique a "
    "mensagem do cliente em uma de três intenções: 'concessao' (quer novo "
    "produto de crédito), 'renegociacao' (quer renegociar dívida) ou "
    "'bancario_geral' (qualquer outra dúvida). Responda SOMENTE com um JSON "
    "contendo: intent, confidence (0 a 1), rationale."
)


class Router:
    """Orquestrador de entrada."""

    def __init__(self, llm: LLMClient, *, limiar: float = LIMIAR_CONFIANCA) -> None:
        self.llm = llm
        self.limiar = limiar

    def run(self, state: GraphState) -> GraphState:
        # 1. Guardrails de entrada.
        for outcome in check_input(state.mensagem):
            state.add_flag("entrada", outcome.name, outcome.blocked, outcome.detail)
            if outcome.blocked:
                state.final_answer = outcome.safe_message
                state.routing = RoutingDecision(
                    intent=Intent.BANCARIO_GERAL, confidence=0.0,
                    rationale=f"Bloqueado na entrada: {outcome.name}.",
                    cliente_id=state.cliente_id, needs_clarification=False,
                )
                return state

        # 2. Classificação (LLM pequeno → JSON estruturado).
        data = self.llm.complete_json(_SYSTEM_ROTEADOR, state.mensagem, tier="small")
        try:
            intent = Intent(data["intent"])
            confidence = float(data["confidence"])
            rationale = str(data.get("rationale", ""))
        except (KeyError, ValueError):
            # Saída malformada do modelo → trata como baixa confiança.
            intent, confidence, rationale = Intent.BANCARIO_GERAL, 0.0, "Saída inválida do classificador."

        # 3. Limiar de confiança → pede clarificação.
        if confidence < self.limiar:
            state.routing = RoutingDecision(
                intent=intent, confidence=confidence, rationale=rationale,
                cliente_id=state.cliente_id, needs_clarification=True,
                clarification_question=(
                    "Só para eu te ajudar melhor: você quer contratar um novo "
                    "crédito, renegociar uma dívida existente, ou tirar outra "
                    "dúvida da sua conta?"
                ),
            )
            return state

        state.routing = RoutingDecision(
            intent=intent, confidence=confidence, rationale=rationale,
            cliente_id=state.cliente_id, needs_clarification=False,
        )
        return state
