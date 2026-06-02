"""
Port opcional para LangGraph (referência).
==================================================================

POR QUE ESTE ARQUIVO É OPCIONAL
-------------------------------
O núcleo do sistema (orchestrator/graph.py) usa um mini-grafo próprio para
manter ZERO dependências pesadas e máxima testabilidade. Este arquivo mostra
como o MESMO desenho — mesmos nós, mesmos contratos (GraphState), mesmas
transições determinísticas — se mapeia para LangGraph quase 1:1.

Requer `pip install "credito-agentes[real]"`. Se LangGraph não estiver
instalado, o import falha de forma clara e o restante do sistema segue
funcionando com o grafo nativo.

DEFINIÇÃO — "LangGraph": biblioteca para montar fluxos de agentes como grafos de
estado, com retry nativo, checkpoints e fácil depuração. O padrão Supervisor +
Workers é expresso como nós que leem/escrevem um estado compartilhado e arestas
condicionais que decidem o próximo nó.
"""

from __future__ import annotations

from credito_agentes.agents.banking import BankingAgent
from credito_agentes.agents.concessao import ConcessaoCell
from credito_agentes.agents.refinement import RefinementLayer
from credito_agentes.agents.renegociacao import RenegociacaoCell
from credito_agentes.kb import build_default_kbs
from credito_agentes.llm import get_llm
from credito_agentes.orchestrator.router import Router
from credito_agentes.schemas import GraphState, Intent


def build_langgraph_app():  # pragma: no cover - requer dependência opcional
    """Monta o mesmo fluxo usando LangGraph.

    Mapeamento direto:
      * cada `*.run(state)` vira um nó do StateGraph;
      * a transição por intenção vira uma aresta condicional;
      * o GraphState (Pydantic) é o estado do grafo.
    """
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "LangGraph não instalado. Use: pip install 'credito-agentes[real]'"
        ) from exc

    llm = get_llm()
    kbs = build_default_kbs()
    router = Router(llm)
    banking = BankingAgent(llm, kbs["faq_bancario"])
    concessao = ConcessaoCell(llm, kbs["politicas_concessao"])
    renegociacao = RenegociacaoCell(llm, kbs["politicas_renegociacao"])
    refinement = RefinementLayer(llm)

    g = StateGraph(GraphState)
    g.add_node("roteador", lambda s: router.run(s))
    g.add_node("concessao", lambda s: concessao.run(s))
    g.add_node("renegociacao", lambda s: renegociacao.run(s))
    g.add_node("bancario", lambda s: banking.run(s))
    g.add_node("refinamento", lambda s: refinement.run(s))

    g.set_entry_point("roteador")

    def _rota(state: GraphState) -> str:
        # Bloqueio ou clarificação encerram cedo (vão direto ao END).
        if state.final_answer or (state.routing and state.routing.needs_clarification):
            return "fim"
        intent = state.routing.intent if state.routing else Intent.BANCARIO_GERAL
        return {
            Intent.CONCESSAO: "concessao",
            Intent.RENEGOCIACAO: "renegociacao",
            Intent.BANCARIO_GERAL: "bancario",
        }[intent]

    g.add_conditional_edges("roteador", _rota, {
        "concessao": "concessao",
        "renegociacao": "renegociacao",
        "bancario": "bancario",
        "fim": END,
    })
    for celula in ("concessao", "renegociacao", "bancario"):
        g.add_edge(celula, "refinamento")
    g.add_edge("refinamento", END)

    return g.compile()
