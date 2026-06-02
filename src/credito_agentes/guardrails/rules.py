"""
Guardrails: as barreiras de segurança determinísticas do sistema.
==================================================================

POR QUE GUARDRAILS SÃO EM CÓDIGO, NÃO NO LLM (explicação a nível de aula)
------------------------------------------------------------------------
Um LLM é probabilístico: dada a mesma entrada, pode responder diferente, e pode
ser manipulado (prompt injection). Decisões com impacto financeiro e regulatório
NÃO podem depender disso. Por isso as regras críticas vivem em CÓDIGO Python
puro, testável e auditável. O LLM redige texto; o código decide o que é
permitido.

DEFINIÇÃO — "guardrail": uma regra de segurança que valida ou bloqueia algo. Há
três momentos de aplicação:

  ENTRADA  — antes de processar: detecta PII indevida, prompt injection,
             jailbreak e exige autenticação para tools transacionais.
  EXECUÇÃO — durante: valida schema das tools, aplica rate limit por sessão e
             impõe a COERÊNCIA DE DECISÃO (a regra de negócio mais importante).
  SAÍDA    — antes de responder: filtra conteúdo tóxico, exige disclaimer
             regulatório e aplica a regra de NÃO-ALUCINAÇÃO NUMÉRICA.

A regra de ouro do decisor (em código, jamais no LLM):
  "NUNCA aprovar crédito se o risco veio reprovado OU as políticas indicam
   não-elegibilidade." Mais limites duros: taxa mínima, prazo máximo, desconto
   máximo.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from credito_agentes.schemas import (
    AnalystOpinion,
    CreditProposal,
    RenegotiationOption,
    Verdict,
)

# ---------------------------------------------------------------------------
# LIMITES DUROS (hard limits) — fonte única da verdade regulatória
# ---------------------------------------------------------------------------
TAXA_MENSAL_MINIMA = 0.009       # 0,9% a.m. (renegociação pode chegar aqui)
TAXA_MENSAL_MINIMA_CONCESSAO = 0.012  # 1,2% a.m.
PRAZO_MAXIMO_CONCESSAO = 48      # meses
PRAZO_MAXIMO_RENEG = 60          # meses
DESCONTO_MAXIMO = 0.40           # 40%
SCORE_INADIMPLENCIA_LIMITROFE = (0.45, 0.55)  # zona cinzenta → humano

DISCLAIMER_REGULATORIO = (
    "Esta é uma simulação. Sujeito a análise e aprovação. "
    "CET e condições finais podem variar conforme regulamentação vigente."
)


@dataclass
class GuardrailOutcome:
    """Resultado de uma checagem de guardrail."""

    name: str
    blocked: bool
    detail: str = ""
    safe_message: str = ""  # mensagem a devolver ao cliente quando bloqueado


# ===========================================================================
# 1. GUARDRAILS DE ENTRADA
# ===========================================================================
# Padrões simples de detecção. Em produção, complemente com classificadores
# dedicados; aqui usamos regex/heurística determinística e testável.
_PII_PATTERNS = [
    re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),      # CPF
    re.compile(r"\b\d{4} ?\d{4} ?\d{4} ?\d{4}\b"),         # cartão 16 díg.
]
_INJECTION_PATTERNS = [
    re.compile(r"ignore.{0,20}(as |todas as )?instru", re.I),
    re.compile(r"esque[çc]a.{0,20}(as |suas )?(regras|instru)", re.I),
    re.compile(r"(system prompt|prompt do sistema|aja como|finja que voc)", re.I),
    re.compile(r"(developer mode|modo desenvolvedor|jailbreak|DAN)", re.I),
]


def check_input(mensagem: str) -> list[GuardrailOutcome]:
    """Roda os guardrails de entrada sobre a mensagem do cliente."""
    out: list[GuardrailOutcome] = []

    if any(p.search(mensagem) for p in _PII_PATTERNS):
        out.append(GuardrailOutcome(
            name="pii_entrada",
            blocked=True,
            detail="PII sensível detectada na mensagem.",
            safe_message="Por segurança, não compartilhe CPF ou número de cartão "
                         "por aqui. Posso ajudar sem esses dados.",
        ))

    if any(p.search(mensagem) for p in _INJECTION_PATTERNS):
        out.append(GuardrailOutcome(
            name="prompt_injection",
            blocked=True,
            detail="Tentativa de prompt injection/jailbreak detectada.",
            safe_message="Não consigo atender a esse pedido. Posso ajudar com "
                         "assuntos da sua conta ou crédito.",
        ))

    return out


def require_authentication(autenticado: bool, *, tool_transacional: bool) -> GuardrailOutcome:
    """Exige autenticação antes de qualquer tool transacional.

    Como o agente bancário é READ-ONLY, `tool_transacional` será False nele.
    A regra existe para proteger expansões futuras do sistema.
    """
    if tool_transacional and not autenticado:
        return GuardrailOutcome(
            name="autenticacao",
            blocked=True,
            detail="Operação transacional sem autenticação.",
            safe_message="Para essa operação preciso que você esteja autenticado.",
        )
    return GuardrailOutcome(name="autenticacao", blocked=False)


# ===========================================================================
# 2. GUARDRAILS DE EXECUÇÃO
# ===========================================================================
class RateLimiter:
    """Rate limit por sessão (janela deslizante simples).

    DEFINIÇÃO — "rate limit": teto de requisições num intervalo, para conter
    abuso e loops descontrolados.
    """

    def __init__(self, *, max_calls: int = 30, window_s: float = 60.0) -> None:
        self.max_calls = max_calls
        self.window_s = window_s
        self._hits: dict[str, list[float]] = {}

    def allow(self, sessao_id: str) -> GuardrailOutcome:
        agora = time.monotonic()
        janela = [t for t in self._hits.get(sessao_id, []) if agora - t < self.window_s]
        if len(janela) >= self.max_calls:
            self._hits[sessao_id] = janela
            return GuardrailOutcome(
                name="rate_limit", blocked=True,
                detail=f"Limite de {self.max_calls} chamadas/{self.window_s:.0f}s.",
                safe_message="Você fez muitas solicitações em pouco tempo. "
                             "Tente novamente em instantes.",
            )
        janela.append(agora)
        self._hits[sessao_id] = janela
        return GuardrailOutcome(name="rate_limit", blocked=False)


def validate_tool_schema(model_cls, payload: dict) -> GuardrailOutcome:
    """Valida o payload de uma tool contra seu schema Pydantic.

    Se a tool devolver algo malformado, bloqueamos a propagação do dado ruim.
    """
    try:
        model_cls(**payload)
        return GuardrailOutcome(name="schema_tool", blocked=False)
    except Exception as exc:  # noqa: BLE001
        return GuardrailOutcome(
            name="schema_tool", blocked=True,
            detail=f"Schema inválido: {type(exc).__name__}",
        )


def decision_coherence(opinions: list[AnalystOpinion]) -> GuardrailOutcome:
    """A REGRA DE OURO, em código: nunca aprovar contra risco/políticas.

    Bloqueia a aprovação se:
      * o analista de RISCO reprovou, OU
      * o analista de POLÍTICAS reprovou (não-elegibilidade).
    Esta função NÃO é delegada ao LLM em hipótese alguma.
    """
    por_analista = {o.analyst: o for o in opinions}
    risco = por_analista.get("risco")
    politicas = por_analista.get("politicas")

    if risco and risco.verdict == Verdict.REPROVADO:
        return GuardrailOutcome(
            name="coerencia_decisao", blocked=True,
            detail="Risco reprovado: aprovação proibida.",
        )
    if politicas and politicas.verdict == Verdict.REPROVADO:
        return GuardrailOutcome(
            name="coerencia_decisao", blocked=True,
            detail="Políticas indicam não-elegibilidade: aprovação proibida.",
        )
    return GuardrailOutcome(name="coerencia_decisao", blocked=False)


def enforce_hard_limits_concessao(p: CreditProposal) -> CreditProposal:
    """Aplica limites duros a uma proposta de concessão (corrige se preciso)."""
    taxa = max(p.taxa_mensal, TAXA_MENSAL_MINIMA_CONCESSAO)
    prazo = min(p.prazo_meses, PRAZO_MAXIMO_CONCESSAO)
    return p.model_copy(update={"taxa_mensal": taxa, "prazo_meses": prazo})


def enforce_hard_limits_reneg(o: RenegotiationOption) -> RenegotiationOption:
    """Aplica limites duros a uma opção de renegociação (corrige se preciso)."""
    taxa = max(o.taxa_mensal, TAXA_MENSAL_MINIMA)
    prazo = min(o.prazo_meses, PRAZO_MAXIMO_RENEG)
    desconto = min(o.desconto_aplicado, DESCONTO_MAXIMO)
    return o.model_copy(update={
        "taxa_mensal": taxa, "prazo_meses": prazo, "desconto_aplicado": desconto
    })


def should_escalate(score_inadimplencia: float) -> bool:
    """Casos de borda: score na zona limítrofe vai para atendente humano."""
    lo, hi = SCORE_INADIMPLENCIA_LIMITROFE
    return lo <= score_inadimplencia <= hi


# ===========================================================================
# 3. GUARDRAILS DE SAÍDA
# ===========================================================================
_TOXIC_PATTERNS = [
    re.compile(r"\b(idiota|imbecil|burro|lixo)\b", re.I),
]
# Regex que encontra "números financeiros" no texto: R$, %, taxas.
_NUMERIC_IN_TEXT = re.compile(
    r"R\$\s?\d[\d.,]*\d"            # valores monetários: R$ 1.234,56 / R$ 450.50 / R$ 1000
    r"|\d+(?:[.,]\d+)?\s?%"         # percentuais: 12% / 1,49% / 1.20%
    r"|\d+[.,]\d+"                  # decimais soltos: 1234.50 / 12,5
)


def check_output_toxicity(texto: str) -> GuardrailOutcome:
    """Filtra conteúdo tóxico/discriminatório na resposta final."""
    if any(p.search(texto) for p in _TOXIC_PATTERNS):
        return GuardrailOutcome(
            name="toxicidade", blocked=True,
            detail="Conteúdo tóxico detectado na saída.",
            safe_message="Desculpe, vou reformular minha resposta.",
        )
    return GuardrailOutcome(name="toxicidade", blocked=False)


def _extract_numbers(texto: str) -> set[str]:
    """Extrai a forma canônica dos números financeiros de um texto."""
    achados = set()
    for m in _NUMERIC_IN_TEXT.finditer(texto):
        # normaliza: mantém só dígitos para comparar valores
        digitos = re.sub(r"[^\d]", "", m.group(0))
        if digitos:
            achados.add(digitos)
    return achados


def numeric_non_hallucination(
    texto: str, valores_autorizados: set[str]
) -> GuardrailOutcome:
    """NÃO-ALUCINAÇÃO NUMÉRICA: todo número financeiro no texto deve estar
    no conjunto de valores autorizados (vindos de tool/modelo/output estruturado).

    Como funciona o cross-check:
      1. Extraímos todos os números financeiros do texto da resposta.
      2. Comparamos com `valores_autorizados` (a fonte da verdade estruturada).
      3. Se aparecer um número que NÃO está autorizado, bloqueamos — o LLM
         provavelmente inventou.
    """
    no_texto = _extract_numbers(texto)
    nao_autorizados = {n for n in no_texto if n not in valores_autorizados}
    if nao_autorizados:
        return GuardrailOutcome(
            name="nao_alucinacao_numerica", blocked=True,
            detail=f"Números não autorizados na resposta: {sorted(nao_autorizados)}",
            safe_message="Vou recalcular os valores antes de responder.",
        )
    return GuardrailOutcome(name="nao_alucinacao_numerica", blocked=False)


def ensure_disclaimer(texto: str, *, is_credit_proposal: bool) -> str:
    """Garante o disclaimer regulatório em propostas de crédito."""
    if is_credit_proposal and DISCLAIMER_REGULATORIO not in texto:
        return f"{texto}\n\n{DISCLAIMER_REGULATORIO}"
    return texto


def authorized_numbers_from_proposal(p: CreditProposal) -> set[str]:
    """Deriva o conjunto de números autorizados a partir de uma proposta.

    Inclui variações de formatação (com/sem casas decimais) para que o
    cross-check aceite a mesma quantia escrita de formas diferentes.
    """
    nums: set[str] = set()
    for valor in (p.limite, p.taxa_mensal * 100, p.prazo_meses):
        nums |= _number_forms(valor)
    return nums


def authorized_numbers_from_options(opcoes: list[RenegotiationOption]) -> set[str]:
    nums: set[str] = set()
    for o in opcoes:
        for valor in (o.parcela, o.taxa_mensal * 100, o.prazo_meses,
                      o.desconto_aplicado * 100):
            nums |= _number_forms(valor)
    return nums


def _number_forms(valor: float) -> set[str]:
    """Gera as formas de dígitos que um valor pode assumir no texto.

    O cross-check compara apenas DÍGITOS (sem separadores). Por isso geramos
    aqui todas as representações plausíveis do mesmo valor: inteiro (.0f),
    uma casa (.1f) e duas casas (.2f). Ex.: a taxa 1.2% pode ser escrita como
    "1.20%" (-> 120) ou "1.2%" (-> 12); ambas precisam estar autorizadas.
    """
    formas = set()
    for fmt in (f"{valor:.0f}", f"{valor:.1f}", f"{valor:.2f}"):
        digitos = re.sub(r"[^\d]", "", fmt)
        if digitos:
            formas.add(digitos)
    return formas
