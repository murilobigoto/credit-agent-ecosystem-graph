"""
Celula de Renegociacao de Dividas (sub-grafo).

ARQUITETURA
-----------
Mesma estrutura da celula de concessao: tres analistas em paralelo + decisor.

  perfil    --> contratos ativos e saldo devedor (sistema de cobranca SICC)
  risco     --> modelo de probabilidade de cura e recovery score (plataforma ML)
  politicas --> RAG nas politicas de renegociacao vigentes

  decisor   --> gera ate N opcoes de renegociacao ordenadas pela menor parcela.

FORMULA DE AMORTIZACAO (Tabela Price)
--------------------------------------
P = S * i / (1 - (1+i)^-n)

  S = saldo devedor com desconto aplicado
  i = taxa mensal
  n = prazo em meses

Garante que o valor da parcela exibido ao cliente venha de CALCULO
deterministico, nao de texto inventado pelo LLM.

LOGICA DE DESCONTO
------------------
Quanto maior a probabilidade de cura (prob_cura do modelo), melhor o
desconto oferecido, ate o maximo permitido pela politica. Prazos curtos
recebem desconto maior para estimular quitacao rapida.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date

from credito_agentes.guardrails import (
    DESCONTO_MAXIMO,
    TAXA_MENSAL_MINIMA,
    enforce_hard_limits_reneg,
)
from credito_agentes.kb import KnowledgeBase
from credito_agentes.llm import LLMClient
from credito_agentes.schemas import (
    AnalystOpinion,
    GraphState,
    RenegotiationOption,
    Verdict,
)
from credito_agentes.tools import (
    ContratosOutput,
    RiscoRenegOutput,
    consulta_contratos_ativos,
    modelo_risco_renegociacao,
    with_timeout,
)


def _parcela_price(saldo: float, taxa_mensal: float, n: int) -> float:
    """Calcula a parcela pela Tabela Price (sistema de amortizacao frances)."""
    if taxa_mensal <= 0:
        return round(saldo / n, 2)
    fator = (1 + taxa_mensal) ** (-n)
    return round(saldo * taxa_mensal / (1 - fator), 2)


# ---------------------------------------------------------------------------
# Analistas
# ---------------------------------------------------------------------------
def _analista_perfil(state: GraphState) -> AnalystOpinion:
    """Consulta o sistema de cobranca para obter contratos e saldo devedor."""
    r = with_timeout(
        lambda: consulta_contratos_ativos(state.cliente_id),
        timeout_s=2.0,
        fallback=ContratosOutput(
            cliente_id=state.cliente_id,
            contratos=[],
            saldo_total=0.0,
            total_contratos=0,
            total_parcelas_atraso=0,
        ),
        tool_name="consulta_contratos_ativos",
    )
    dados: ContratosOutput = r.value  # type: ignore[assignment]

    contratos_desc = (
        ", ".join(f"{c.produto} (R$ {c.saldo_devedor:.2f})" for c in dados.contratos)
        if dados.contratos
        else "nenhum contrato localizado"
    )
    return AnalystOpinion(
        analyst="perfil",
        verdict=Verdict.NEUTRO,
        summary=(
            f"Cliente com {dados.total_contratos} contrato(s) em atraso. "
            f"Saldo devedor total: R$ {dados.saldo_total:.2f}. "
            f"Contratos: {contratos_desc}."
        ),
        structured_data={
            "saldo_devedor": dados.saldo_total,
            "contratos_ativos": dados.total_contratos,
            "total_parcelas_atraso": dados.total_parcelas_atraso,
            "fallback": r.fallback_used,
        },
    )


def _analista_risco(state: GraphState) -> AnalystOpinion:
    """Consulta o modelo de recuperacao de credito."""
    r = with_timeout(
        lambda: modelo_risco_renegociacao(state.cliente_id),
        timeout_s=2.0,
        fallback=RiscoRenegOutput(
            cliente_id=state.cliente_id,
            prob_cura=0.0,
            recovery_score=0.0,
        ),
        tool_name="modelo_risco_renegociacao",
    )
    dados: RiscoRenegOutput = r.value  # type: ignore[assignment]
    return AnalystOpinion(
        analyst="risco",
        verdict=Verdict.APROVADO,
        summary=(
            f"Probabilidade de regularizacao em 12 meses: {dados.prob_cura:.0%}. "
            f"Recovery score: {dados.recovery_score:.2f}."
        ),
        structured_data={
            "prob_cura": dados.prob_cura,
            "recovery_score": dados.recovery_score,
            "fallback": r.fallback_used,
        },
    )


def _analista_politicas(state: GraphState, kb: KnowledgeBase) -> AnalystOpinion:
    """Verifica opcoes e limites permitidos via RAG nas politicas de renegociacao."""
    trechos = kb.retrieve(
        "opcoes de renegociacao desconto maximo prazo taxa minima condicoes acesso",
        hoje=date.today(),
        top_k=3,
    )
    permitido = len(trechos) > 0
    return AnalystOpinion(
        analyst="politicas",
        verdict=Verdict.APROVADO if permitido else Verdict.REPROVADO,
        summary=(
            "Politicas de renegociacao vigentes consultadas: opcoes dentro dos limites permitidos."
            if permitido
            else "Sem politica vigente: renegociacao indisponivel."
        ),
        structured_data={
            "desconto_maximo": DESCONTO_MAXIMO,
            "taxa_minima": TAXA_MENSAL_MINIMA,
            "prazos_permitidos": [6, 12, 24, 48, 60],
            "politicas_consultadas": [c.metadata.versao_politica for c in trechos],
        },
    )


# ---------------------------------------------------------------------------
# Decisor
# ---------------------------------------------------------------------------
def _decisor(state: GraphState, n_propostas: int) -> GraphState:
    """Gera ate N opcoes de renegociacao ordenadas pela menor parcela."""
    por_analista = {o.analyst: o for o in state.opinions}
    risco = por_analista["risco"].structured_data
    perfil = por_analista["perfil"].structured_data
    politicas = por_analista["politicas"].structured_data

    if por_analista["politicas"].verdict == Verdict.REPROVADO:
        state.raw_answer = (
            "No momento nao ha opcao de renegociacao disponivel para o seu caso. "
            "Recomendo falar com um atendente para alternativas."
        )
        return state

    saldo = float(perfil.get("saldo_devedor", 0.0))
    if saldo <= 0:
        state.raw_answer = (
            "Nao foram encontradas dividas em atraso vinculadas a sua conta. "
            "Se acredita que ha um erro, entre em contato com nossa central."
        )
        return state

    prob_cura = float(risco.get("prob_cura", 0.0))
    desconto_base = min(DESCONTO_MAXIMO, round(0.10 + 0.30 * prob_cura, 2))

    opcoes: list[RenegotiationOption] = []
    prazos = sorted(politicas.get("prazos_permitidos", [12, 24, 48]))[:n_propostas]
    for prazo in prazos:
        desconto = desconto_base if prazo <= 12 else round(desconto_base * 0.5, 2)
        taxa = max(TAXA_MENSAL_MINIMA, round(0.012 + 0.0002 * prazo, 4))
        saldo_com_desconto = saldo * (1 - desconto)
        parcela = _parcela_price(saldo_com_desconto, taxa, prazo)
        opcao = RenegotiationOption(
            prazo_meses=prazo,
            taxa_mensal=taxa,
            desconto_aplicado=desconto,
            parcela=parcela,
        )
        opcoes.append(enforce_hard_limits_reneg(opcao))

    opcoes.sort(key=lambda o: o.parcela)
    state.renegotiation_options = opcoes
    return state


# ---------------------------------------------------------------------------
# Celula (orquestrador do sub-grafo)
# ---------------------------------------------------------------------------
class RenegociacaoCell:
    """Sub-grafo de renegociacao: analistas em paralelo -> decisor (N opcoes)."""

    def __init__(
        self,
        llm: LLMClient,
        politicas_kb: KnowledgeBase,
        *,
        n_propostas: int = 3,
    ) -> None:
        self.llm = llm
        self.kb = politicas_kb
        self.n_propostas = n_propostas

    def run(self, state: GraphState) -> GraphState:
        with ThreadPoolExecutor(max_workers=3) as ex:
            f_perfil = ex.submit(_analista_perfil, state)
            f_risco = ex.submit(_analista_risco, state)
            f_pol = ex.submit(_analista_politicas, state, self.kb)
            state.opinions = [f_perfil.result(), f_risco.result(), f_pol.result()]
        return _decisor(state, self.n_propostas)
