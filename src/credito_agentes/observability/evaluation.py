"""
Avaliação do sistema: métricas de qualidade e desempenho.
==================================================================

POR QUE MEDIR (explicação a nível de aula)
------------------------------------------
"O que não se mede, não se melhora." Este módulo calcula as métricas exigidas
pelo plano de avaliação:

  * PRECISÃO DE ROTEAMENTO: % de mensagens classificadas na rota correta.
    DEFINIÇÃO — "precisão" aqui = acertos / total no golden set rotulado.

  * TAXA DE ALUCINAÇÃO NUMÉRICA: % de respostas em que o guardrail de
    não-alucinação precisou bloquear/corrigir um número. Idealmente ~0%.

  * LATÊNCIA p50 e p95: o tempo de resposta mediano (p50) e o tempo abaixo do
    qual ficam 95% das respostas (p95). p95 captura a "cauda" lenta, que é o que
    o usuário sente nos piores casos.
    DEFINIÇÃO — "percentil p": valor abaixo do qual caem p% das observações.

Como rodar:  python -m credito_agentes.observability.evaluation
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from credito_agentes.orchestrator import CreditAgentGraph
from credito_agentes.schemas import CustomerProfile, Intent


@dataclass
class RoutingCase:
    mensagem: str
    intent_esperado: Intent


# Conjunto rotulado para medir precisão de roteamento.
ROUTING_GOLDEN: list[RoutingCase] = [
    RoutingCase("quero contratar um empréstimo pessoal", Intent.CONCESSAO),
    RoutingCase("preciso de um financiamento novo", Intent.CONCESSAO),
    RoutingCase("gostaria de pedir crédito", Intent.CONCESSAO),
    RoutingCase("preciso renegociar minha dívida", Intent.RENEGOCIACAO),
    RoutingCase("estou em atraso e quero quitar", Intent.RENEGOCIACAO),
    RoutingCase("qual é o meu saldo", Intent.BANCARIO_GERAL),
    RoutingCase("quanto uso do limite do cartão", Intent.BANCARIO_GERAL),
    RoutingCase("como funciona a fatura do cartão de crédito", Intent.BANCARIO_GERAL),
]


def _percentil(valores: list[float], p: float) -> float:
    if not valores:
        return 0.0
    ordenados = sorted(valores)
    k = (len(ordenados) - 1) * p
    f = int(k)
    c = min(f + 1, len(ordenados) - 1)
    return ordenados[f] + (ordenados[c] - ordenados[f]) * (k - f)


@dataclass
class EvalReport:
    precisao_roteamento: float
    taxa_alucinacao_numerica: float
    latencia_p50_ms: float
    latencia_p95_ms: float
    n_amostras: int


def run_evaluation(grafo: CreditAgentGraph | None = None) -> EvalReport:
    """Roda o golden set e devolve o relatório de métricas."""
    grafo = grafo or CreditAgentGraph()
    perfil = CustomerProfile(cliente_id="eval", nome="Eval", segmento="varejo",
                             autenticado=True)

    acertos = 0
    alucinacoes = 0
    latencias: list[float] = []

    for caso in ROUTING_GOLDEN:
        inicio = time.perf_counter()
        s = grafo.invoke(caso.mensagem, cliente_id="eval", profile=perfil)
        latencias.append((time.perf_counter() - inicio) * 1000.0)

        if s.routing and s.routing.intent == caso.intent_esperado:
            acertos += 1
        if any(f.name == "nao_alucinacao_numerica" and f.blocked
               for f in s.guardrail_flags):
            alucinacoes += 1

    n = len(ROUTING_GOLDEN)
    return EvalReport(
        precisao_roteamento=acertos / n,
        taxa_alucinacao_numerica=alucinacoes / n,
        latencia_p50_ms=_percentil(latencias, 0.50),
        latencia_p95_ms=_percentil(latencias, 0.95),
        n_amostras=n,
    )


def main() -> None:  # pragma: no cover
    rel = run_evaluation()
    print("=== Relatório de Avaliação ===")
    print(f"Amostras:                  {rel.n_amostras}")
    print(f"Precisão de roteamento:    {rel.precisao_roteamento:.1%}")
    print(f"Taxa de alucinação num.:   {rel.taxa_alucinacao_numerica:.1%}")
    print(f"Latência p50:              {rel.latencia_p50_ms:.1f} ms")
    print(f"Latência p95:              {rel.latencia_p95_ms:.1f} ms")


if __name__ == "__main__":  # pragma: no cover
    main()
