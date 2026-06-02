"""
Observabilidade: tracing de turno com IDs de correlação.
==================================================================

POR QUE OBSERVABILIDADE (explicação a nível de aula)
----------------------------------------------------
Quando algo dá errado em produção — uma decisão estranha, uma latência alta —
você precisa reconstruir EXATAMENTE o que aconteceu. Observabilidade é a
capacidade de enxergar o interior do sistema a partir do que ele emite (logs,
métricas, traces).

DEFINIÇÃO — "trace": o registro encadeado de todos os passos de um turno. Cada
passo é um "span" (intervalo) com início, fim e atributos.

DEFINIÇÃO — "ID de correlação": um identificador único do turno que aparece em
TODOS os logs daquele turno. Permite filtrar "me mostre tudo o que aconteceu na
conversa X" e ligar cliente ↔ sessão ↔ decisão para auditoria.

Em produção, troque o `ConsoleTracer` por um exportador OpenTelemetry ou
LangSmith. A interface (`Tracer`) é a mesma; só a implementação muda.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


def new_correlation_id() -> str:
    """Gera um ID de correlação único para um turno."""
    return f"trn_{uuid.uuid4().hex[:16]}"


@dataclass
class Span:
    """Um intervalo medido dentro de um turno (ex.: 'roteamento')."""

    name: str
    correlation_id: str
    start: float = field(default_factory=time.perf_counter)
    end: float | None = None
    attributes: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        fim = self.end if self.end is not None else time.perf_counter()
        return (fim - self.start) * 1000.0


class Tracer:
    """Coletor simples de spans, com saída para console.

    Mantém os spans em memória para que testes possam inspecioná-los e, ao
    mesmo tempo, imprime um resumo legível. É deliberadamente simples; a troca
    por um backend real é uma questão de implementar a mesma interface.
    """

    def __init__(self, *, verbose: bool = False) -> None:
        self.spans: list[Span] = []
        self.verbose = verbose

    @contextmanager
    def span(self, name: str, correlation_id: str, **attributes) -> Iterator[Span]:
        s = Span(name=name, correlation_id=correlation_id, attributes=dict(attributes))
        try:
            yield s
        finally:
            s.end = time.perf_counter()
            self.spans.append(s)
            if self.verbose:
                attrs = " ".join(f"{k}={v}" for k, v in s.attributes.items())
                print(
                    f"[trace {correlation_id}] {name} "
                    f"({s.duration_ms:.1f}ms) {attrs}".rstrip()
                )

    def durations_by_name(self) -> dict[str, float]:
        """Soma de durações por nome de span — útil para análise de latência."""
        agg: dict[str, float] = {}
        for s in self.spans:
            agg[s.name] = agg.get(s.name, 0.0) + s.duration_ms
        return agg


# Tracer global padrão (pode ser substituído na inicialização da app).
_default_tracer = Tracer(verbose=False)


def get_tracer() -> Tracer:
    return _default_tracer
