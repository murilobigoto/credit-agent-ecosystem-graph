"""
Celula de Concessao de Credito (sub-grafo).

ARQUITETURA
-----------
Tres analistas rodam em paralelo e um decisor consolida os pareceres.

  perfil    --> dados do CRM + score bureau
  risco     --> modelo preditivo de inadimplencia (plataforma ML)
  politicas --> RAG nas politicas vigentes de concessao

  decisor   --> combina os tres pareceres por REGRA de codigo,
                nunca por "feeling" do LLM.

PARALELISMO
-----------
Os tres analistas sao independentes entre si e podem rodar ao mesmo tempo
via ThreadPoolExecutor. A latencia total fica em torno de max(t_perfil,
t_risco, t_politicas) em vez da soma dos tres.

DIVISAO DE RESPONSABILIDADES
-----------------------------
  perfil    -- caracteriza o cliente: segmento, renda, score, relacionamento.
               Simula consulta ao CRM e ao bureau de credito (Serasa/SCR).
  risco     -- consulta o modelo preditivo de score de inadimplencia e
               calcula a capacidade de pagamento mensal recomendada.
  politicas -- verifica elegibilidade via RAG na base de politicas vigentes.
  decisor   -- aplica decision_coherence (guardrail) e enforce_hard_limits.
               Constroi a CreditProposal com valores vindos APENAS das tools.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date

from credito_agentes.guardrails import (
    TAXA_MENSAL_MINIMA_CONCESSAO,
    decision_coherence,
    enforce_hard_limits_concessao,
    should_escalate,
)
from credito_agentes.kb import KnowledgeBase
from credito_agentes.llm import LLMClient
from credito_agentes.schemas import (
    AnalystOpinion,
    CreditProposal,
    GraphState,
    Verdict,
)
from credito_agentes.tools import (
    PerfilClienteOutput,
    RiscoConcessaoOutput,
    consulta_perfil_cliente,
    modelo_risco_concessao,
    with_timeout,
)


# ---------------------------------------------------------------------------
# Analistas
# ---------------------------------------------------------------------------
def _analista_perfil(state: GraphState) -> AnalystOpinion:
    """Consulta o CRM e o bureau de credito para caracterizar o cliente."""
    r = with_timeout(
        lambda: consulta_perfil_cliente(state.cliente_id),
        timeout_s=2.0,
        fallback=PerfilClienteOutput(
            cliente_id=state.cliente_id,
            nome="Cliente",
            segmento="varejo",
            renda_mensal=2_500.00,
            score_bureau=500,
            tempo_relacionamento_meses=6,
        ),
        tool_name="consulta_perfil_cliente",
    )
    perfil: PerfilClienteOutput = r.value  # type: ignore[assignment]
    return AnalystOpinion(
        analyst="perfil",
        verdict=Verdict.NEUTRO,
        summary=(
            f"Cliente {perfil.segmento}, renda R$ {perfil.renda_mensal:.2f}, "
            f"score bureau {perfil.score_bureau}, "
            f"relacionamento de {perfil.tempo_relacionamento_meses} meses."
        ),
        structured_data={
            "segmento": perfil.segmento,
            "renda_mensal": perfil.renda_mensal,
            "score_bureau": perfil.score_bureau,
            "tempo_relacionamento_meses": perfil.tempo_relacionamento_meses,
            "fallback": r.fallback_used,
        },
    )


def _analista_risco(state: GraphState) -> AnalystOpinion:
    """Consulta o modelo preditivo de risco de inadimplencia."""
    r = with_timeout(
        lambda: modelo_risco_concessao(state.cliente_id),
        timeout_s=2.0,
        fallback=RiscoConcessaoOutput(
            cliente_id=state.cliente_id,
            score_inadimplencia=1.0,
            capacidade_pagamento=0.0,
        ),
        tool_name="modelo_risco_concessao",
    )
    dados: RiscoConcessaoOutput = r.value  # type: ignore[assignment]
    verdict = Verdict.REPROVADO if dados.score_inadimplencia > 0.55 else Verdict.APROVADO
    return AnalystOpinion(
        analyst="risco",
        verdict=verdict,
        summary=(
            f"Score de inadimplencia {dados.score_inadimplencia:.2f} "
            f"({'alto risco' if verdict == Verdict.REPROVADO else 'risco aceitavel'}), "
            f"capacidade de pagamento R$ {dados.capacidade_pagamento:.2f}/mes."
        ),
        structured_data={
            "score_inadimplencia": dados.score_inadimplencia,
            "capacidade_pagamento": dados.capacidade_pagamento,
            "fallback": r.fallback_used,
        },
    )


def _analista_politicas(state: GraphState, kb: KnowledgeBase) -> AnalystOpinion:
    """Verifica elegibilidade via RAG nas politicas de concessao vigentes."""
    trechos = kb.retrieve(
        "elegibilidade taxa minima prazo maximo comprometimento de renda score minimo",
        hoje=date.today(),
        top_k=3,
    )
    elegivel = len(trechos) > 0
    return AnalystOpinion(
        analyst="politicas",
        verdict=Verdict.APROVADO if elegivel else Verdict.REPROVADO,
        summary=(
            "Politicas vigentes consultadas: cliente aderente aos criterios de elegibilidade."
            if elegivel
            else "Sem politica vigente aplicavel: cliente nao elegivel."
        ),
        structured_data={
            "taxa_minima": TAXA_MENSAL_MINIMA_CONCESSAO,
            "prazo_maximo": 48,
            "politicas_consultadas": [c.metadata.versao_politica for c in trechos],
            "trechos_recuperados": len(trechos),
        },
    )


# ---------------------------------------------------------------------------
# Decisor
# ---------------------------------------------------------------------------
def _decisor(state: GraphState) -> GraphState:
    """Consolida os tres pareceres e constroi a proposta por REGRA."""
    coerencia = decision_coherence(state.opinions)
    state.add_flag("execucao", coerencia.name, coerencia.blocked, coerencia.detail)

    por_analista = {o.analyst: o for o in state.opinions}
    risco = por_analista["risco"].structured_data
    perfil = por_analista["perfil"].structured_data

    if should_escalate(float(risco.get("score_inadimplencia", 1.0))):
        state.escalate_to_human = True
        state.human_handoff_summary = (
            f"Score limitrofe ({risco.get('score_inadimplencia')}). "
            f"Segmento {perfil.get('segmento')}, "
            f"renda R$ {perfil.get('renda_mensal', 0):.2f}. "
            "Requer analise humana."
        )
        return state

    if coerencia.blocked:
        state.raw_answer = (
            "Apos analise, nao foi possivel aprovar uma proposta de credito neste "
            "momento. Voce pode tentar novamente futuramente ou falar com um atendente."
        )
        return state

    capacidade = float(risco.get("capacidade_pagamento", 0.0))
    segmento = perfil.get("segmento", "varejo")
    prazo = 24
    limite = round(min(capacidade * prazo * 0.5, 50_000.00), 2)
    taxa = 0.0149 if segmento == "alta_renda" else 0.0189

    proposta = CreditProposal(
        produto="Credito Pessoal Facil",
        limite=limite,
        taxa_mensal=taxa,
        prazo_meses=prazo,
    )
    state.credit_proposal = enforce_hard_limits_concessao(proposta)
    return state


# ---------------------------------------------------------------------------
# Celula (orquestrador do sub-grafo)
# ---------------------------------------------------------------------------
class ConcessaoCell:
    """Sub-grafo de concessao: analistas em paralelo -> decisor."""

    def __init__(self, llm: LLMClient, politicas_kb: KnowledgeBase) -> None:
        self.llm = llm
        self.kb = politicas_kb

    def run(self, state: GraphState) -> GraphState:
        with ThreadPoolExecutor(max_workers=3) as ex:
            f_perfil = ex.submit(_analista_perfil, state)
            f_risco = ex.submit(_analista_risco, state)
            f_pol = ex.submit(_analista_politicas, state, self.kb)
            state.opinions = [f_perfil.result(), f_risco.result(), f_pol.result()]
        return _decisor(state)
