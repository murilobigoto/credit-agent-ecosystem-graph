"""
Ferramentas (tools) que os agentes podem chamar.

O QUE E UMA TOOL
----------------
Uma tool e uma funcao deterministica e auditavel que um agente chama para
obter um dado ou executar uma acao. O resultado de uma tool e FATO, nao
opiniao do LLM. Por isso, todo numero que aparece na resposta final precisa
vir de uma tool ou de um modelo, nunca ser inventado pelo LLM.

FONTES DE DADOS SIMULADAS
--------------------------
Cada tool simula a chamada a um sistema interno do banco:

  consulta_saldo              -> Core Banking System (CBS)
  uso_limite_cartao           -> CBS: modulo de cartoes
  uso_limite_cheque_especial  -> CBS: modulo de conta corrente
  consulta_perfil_cliente     -> CRM / sistema de onboarding
  consulta_contratos_ativos   -> Sistema de cobranca (SICC)
  modelo_risco_concessao      -> Plataforma ML: modelo de concessao
  modelo_risco_renegociacao   -> Plataforma ML: modelo de recuperacao

REGRAS APLICADAS
----------------
1. READ-ONLY no agente bancario: as tools de saldo/limite apenas LEEM.
   Nenhuma tool altera estado (sem transferencias, sem pagamentos).
2. TIMEOUT + FALLBACK: toda tool e embrulhada com with_timeout para que,
   se demorar ou falhar, a conversa nao trave.
3. SCHEMA VALIDADO: cada tool devolve um modelo Pydantic validado. O
   guardrail de execucao pode revalidar o payload a qualquer momento.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, Field

from credito_agentes.data.mock_db import (
    get_cliente,
    get_conta,
    get_contratos,
    get_scores_risco,
)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Embrulho de timeout + fallback
# ---------------------------------------------------------------------------
@dataclass
class ToolResult:
    """Resultado padronizado de uma tool, com indicacao de fallback."""

    value: object
    ok: bool
    fallback_used: bool = False
    detail: str = ""


def with_timeout(
    fn: Callable[[], T],
    *,
    timeout_s: float,
    fallback: T,
    tool_name: str,
) -> ToolResult:
    """Executa fn com limite de tempo; em falha ou timeout devolve fallback.

    Uma tool lenta ou indisponivel NUNCA congela a conversa.
    """
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(fn)
            value = fut.result(timeout=timeout_s)
        return ToolResult(value=value, ok=True)
    except FuturesTimeout:
        return ToolResult(value=fallback, ok=False, fallback_used=True,
                          detail=f"{tool_name}: timeout apos {timeout_s}s")
    except Exception as exc:  # noqa: BLE001
        return ToolResult(value=fallback, ok=False, fallback_used=True,
                          detail=f"{tool_name}: erro {type(exc).__name__}")


# ---------------------------------------------------------------------------
# SCHEMAS DE SAIDA (validados a cada chamada)
# ---------------------------------------------------------------------------
class SaldoOutput(BaseModel):
    """Saldo disponivel em conta corrente (Core Banking)."""

    cliente_id: str
    saldo: float = Field(description="Saldo disponivel em conta, em reais.")


class UsoLimiteOutput(BaseModel):
    """Percentual de uso de um limite de credito."""

    cliente_id: str
    percentual_uso: float = Field(ge=0, le=1, description="Fracao utilizada do limite, 0..1.")
    limite_total: float = Field(default=0.0, ge=0, description="Limite total contratado, em reais.")
    valor_utilizado: float = Field(default=0.0, ge=0, description="Valor ja utilizado, em reais.")


class PerfilClienteOutput(BaseModel):
    """Perfil detalhado do cliente (CRM + bureau de credito)."""

    cliente_id: str
    nome: str
    segmento: str
    renda_mensal: float = Field(ge=0, description="Renda mensal declarada/estimada, R$.")
    score_bureau: int = Field(ge=0, le=1000, description="Score Serasa/SCR, escala 0-1000.")
    tempo_relacionamento_meses: int = Field(ge=0)


class ContratoOutput(BaseModel):
    """Um contrato de credito ativo."""

    numero: str
    produto: str
    saldo_devedor: float = Field(ge=0)
    parcelas_em_atraso: int = Field(ge=0)
    valor_parcela: float = Field(ge=0)


class ContratosOutput(BaseModel):
    """Conjunto de contratos ativos de um cliente (sistema de cobranca)."""

    cliente_id: str
    contratos: list[ContratoOutput]
    saldo_total: float = Field(ge=0, description="Soma dos saldos devedores, R$.")
    total_contratos: int = Field(ge=0)
    total_parcelas_atraso: int = Field(ge=0)


class RiscoConcessaoOutput(BaseModel):
    """Score de risco para concessao de credito (modelo ML interno)."""

    cliente_id: str
    score_inadimplencia: float = Field(ge=0, le=1, description="0=risco minimo, 1=risco maximo.")
    capacidade_pagamento: float = Field(ge=0, description="Comprometimento mensal maximo recomendado, R$.")


class RiscoRenegOutput(BaseModel):
    """Score de recuperacao para renegociacao (modelo ML interno)."""

    cliente_id: str
    prob_cura: float = Field(ge=0, le=1, description="Probabilidade de regularizacao em 12 meses.")
    recovery_score: float = Field(ge=0, le=1, description="Potencial de recuperacao do valor devido.")


# ---------------------------------------------------------------------------
# TOOLS READ-ONLY - Conta Corrente / Cartoes (Core Banking)
# ---------------------------------------------------------------------------
def consulta_saldo(cliente_id: str) -> SaldoOutput:
    """READ-ONLY: saldo disponivel em conta corrente."""
    conta = get_conta(cliente_id)
    return SaldoOutput(cliente_id=cliente_id, saldo=conta.saldo_disponivel)


def uso_limite_cartao(cliente_id: str) -> UsoLimiteOutput:
    """READ-ONLY: uso do limite do cartao de credito."""
    conta = get_conta(cliente_id)
    percentual = (conta.uso_cartao / conta.limite_cartao) if conta.limite_cartao > 0 else 0.0
    return UsoLimiteOutput(
        cliente_id=cliente_id,
        percentual_uso=round(min(percentual, 1.0), 4),
        limite_total=conta.limite_cartao,
        valor_utilizado=conta.uso_cartao,
    )


def uso_limite_cheque_especial(cliente_id: str) -> UsoLimiteOutput:
    """READ-ONLY: uso do limite do cheque especial."""
    conta = get_conta(cliente_id)
    percentual = (conta.uso_cheque / conta.limite_cheque) if conta.limite_cheque > 0 else 0.0
    return UsoLimiteOutput(
        cliente_id=cliente_id,
        percentual_uso=round(min(percentual, 1.0), 4),
        limite_total=conta.limite_cheque,
        valor_utilizado=conta.uso_cheque,
    )


# Tools read-only disponiveis ao agente bancario de fallback.
READ_ONLY_BANKING_TOOLS: dict[str, Callable[[str], BaseModel]] = {
    "consulta_saldo": consulta_saldo,
    "uso_limite_cartao": uso_limite_cartao,
    "uso_limite_cheque_especial": uso_limite_cheque_especial,
}


# ---------------------------------------------------------------------------
# TOOLS DE PERFIL E CONTRATOS (CRM / Sistema de Cobranca)
# ---------------------------------------------------------------------------
def consulta_perfil_cliente(cliente_id: str) -> PerfilClienteOutput:
    """Retorna perfil detalhado do cliente (CRM + score bureau).

    Em producao: chamada HTTP ao servico de CRM com autenticacao interna.
    """
    cli = get_cliente(cliente_id)
    return PerfilClienteOutput(
        cliente_id=cliente_id,
        nome=cli.nome,
        segmento=cli.segmento,
        renda_mensal=cli.renda_mensal,
        score_bureau=cli.score_bureau,
        tempo_relacionamento_meses=cli.tempo_relacionamento_meses,
    )


def consulta_contratos_ativos(cliente_id: str) -> ContratosOutput:
    """Retorna contratos de credito ativos do cliente (sistema de cobranca SICC).

    Em producao: chamada HTTP ao SICC com cliente_id autenticado.
    """
    dados = get_contratos(cliente_id)
    contratos_out = [
        ContratoOutput(
            numero=c.numero,
            produto=c.produto,
            saldo_devedor=c.saldo_devedor,
            parcelas_em_atraso=c.parcelas_em_atraso,
            valor_parcela=c.valor_parcela,
        )
        for c in dados.contratos
    ]
    return ContratosOutput(
        cliente_id=cliente_id,
        contratos=contratos_out,
        saldo_total=dados.saldo_total_devedor,
        total_contratos=len(dados.contratos),
        total_parcelas_atraso=dados.total_parcelas_atraso,
    )


# ---------------------------------------------------------------------------
# MODELOS PREDITIVOS (Plataforma ML interna)
# ---------------------------------------------------------------------------
def modelo_risco_concessao(cliente_id: str) -> RiscoConcessaoOutput:
    """Score de risco para concessao de credito.

    Em producao: POST /v1/scoring/concessao com cliente_id e features.
    O sleep simula a latencia tipica de 10-50ms de um endpoint de scoring.
    """
    time.sleep(0.01)
    scores = get_scores_risco(cliente_id)
    return RiscoConcessaoOutput(
        cliente_id=cliente_id,
        score_inadimplencia=scores["score_inadimplencia"],
        capacidade_pagamento=scores["capacidade_pagamento"],
    )


def modelo_risco_renegociacao(cliente_id: str) -> RiscoRenegOutput:
    """Score de recuperacao para renegociacao de divida.

    Em producao: POST /v1/scoring/recuperacao com cliente_id e historico.
    """
    time.sleep(0.01)
    scores = get_scores_risco(cliente_id)
    return RiscoRenegOutput(
        cliente_id=cliente_id,
        prob_cura=scores["prob_cura"],
        recovery_score=scores["recovery_score"],
    )
