"""
Grafo de Agentes (o fluxo determinístico que conecta tudo).
==================================================================

PADRÃO ARQUITETURAL (explicação a nível de aula)
------------------------------------------------
Implementamos um grafo determinístico no estilo Supervisor + Workers. Cada nó é
uma função que recebe o GraphState e devolve um GraphState atualizado. As
arestas (transições) são decididas por REGRAS em código a partir do
RoutingDecision — nunca por improviso do LLM.

POR QUE NÃO USAR LANGGRAPH DIRETAMENTE AQUI?
--------------------------------------------
O requisito cita LangGraph/Semantic Kernel como opções. Para manter o núcleo
SIMPLES, testável e sem dependência pesada, implementamos um mini-grafo próprio
com a MESMA semântica (nós + transições determinísticas + retry/timeout nas
tools). O arquivo `graph_langgraph.py` (opcional) mostra como portar este mesmo
desenho para LangGraph quase 1:1, preservando os contratos.

FLUXO DE UM TURNO
-----------------
  entrada → roteador → [clarificação?] →
      concessao | renegociacao | bancario_geral → refinamento → saída

A camada de refinamento é SEMPRE o último nó antes da resposta (exceto quando o
roteador já bloqueou na entrada ou pediu clarificação).
"""

from __future__ import annotations

from credito_agentes.agents.banking import BankingAgent
from credito_agentes.agents.concessao import ConcessaoCell
from credito_agentes.agents.refinement import RefinementLayer
from credito_agentes.agents.renegociacao import RenegociacaoCell
from credito_agentes.kb import build_default_kbs
from credito_agentes.llm import LLMClient, get_llm
from credito_agentes.observability import Tracer, get_tracer, new_correlation_id
from credito_agentes.orchestrator.router import Router
from credito_agentes.schemas import CustomerProfile, GraphState, Intent


class CreditAgentGraph:
    """O grafo completo, pronto para processar um turno de conversa."""

    def __init__(self, llm: LLMClient | None = None, tracer: Tracer | None = None) -> None:
        self.llm = llm or get_llm()
        self.tracer = tracer or get_tracer()
        kbs = build_default_kbs()

        # Instancia cada nó com suas dependências (injeção de dependência).
        self.router = Router(self.llm)
        self.banking = BankingAgent(self.llm, kbs["faq_bancario"])
        self.concessao = ConcessaoCell(self.llm, kbs["politicas_concessao"])
        self.renegociacao = RenegociacaoCell(self.llm, kbs["politicas_renegociacao"])
        self.refinement = RefinementLayer(self.llm)

    def invoke(
        self,
        mensagem: str,
        *,
        cliente_id: str,
        profile: CustomerProfile | None = None,
        correlation_id: str | None = None,
    ) -> GraphState:
        """Processa uma mensagem ponta a ponta e devolve o estado final."""
        cid = correlation_id or new_correlation_id()
        state = GraphState(
            correlation_id=cid,
            cliente_id=cliente_id,
            mensagem=mensagem,
            profile=profile or CustomerProfile(cliente_id=cliente_id),
        )

        # --- nó: roteador ---
        with self.tracer.span("roteamento", cid):
            state = self.router.run(state)

        # Bloqueado na entrada → já tem final_answer.
        if state.final_answer:
            return state

        # Pediu clarificação → devolve a pergunta e encerra o turno.
        if state.routing and state.routing.needs_clarification:
            state.final_answer = state.routing.clarification_question or (
                "Pode me dar mais detalhes do que você precisa?"
            )
            return state

        # --- nó: célula/worker conforme a intenção ---
        intent = state.routing.intent if state.routing else Intent.BANCARIO_GERAL
        with self.tracer.span("celula", cid, intent=intent.value):
            if intent == Intent.CONCESSAO:
                state = self.concessao.run(state)
            elif intent == Intent.RENEGOCIACAO:
                state = self.renegociacao.run(state)
            else:
                state = self.banking.run(state)

        # --- nó: refinamento (sempre antes da resposta) ---
        with self.tracer.span("refinamento", cid):
            state = self.refinement.run(state)

        return state
