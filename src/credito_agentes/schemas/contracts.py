"""
Contratos de dados do ecossistema (a "linguagem comum" entre agentes).
==========================================================================

POR QUE ESTE ARQUIVO EXISTE (explicação a nível de aula)
---------------------------------------------------------
Em um sistema multi-agente, cada agente é uma caixa-preta que recebe algo e
devolve algo. Se cada agente inventar seu próprio formato de entrada/saída, o
sistema vira um emaranhado impossível de manter: mudar um agente quebra os
outros. A solução é definir *contratos* — estruturas de dados fixas e validadas
— que funcionam como um "idioma oficial". Todo agente fala esse idioma.

Usamos Pydantic v2 porque ele:
  1. Valida os dados em tempo de execução (se vier lixo, levanta erro na hora).
  2. Gera JSON Schema automaticamente (útil para documentação e guardrails).
  3. Documenta os campos de forma legível.

DEFINIÇÃO — "contrato de dados": um acordo formal sobre quais campos existem,
seus tipos e suas regras. É a fronteira entre dois componentes.

DEFINIÇÃO — "estado do grafo" (GraphState): o objeto único que atravessa todos
os nós do grafo de agentes. Cada nó lê o que precisa e escreve seu resultado
nele. Pense num prontuário médico que passa de médico em médico, cada um
anotando sua parte.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# 1. INTENÇÕES E ROTEAMENTO
# ---------------------------------------------------------------------------
class Intent(str, Enum):
    """As três rotas possíveis do sistema.

    DEFINIÇÃO — "intenção" (intent): a categoria do que o cliente quer. O
    orquestrador classifica a mensagem em uma destas três.
    """

    CONCESSAO = "concessao"          # quer um novo produto de crédito
    RENEGOCIACAO = "renegociacao"    # quer renegociar uma dívida existente
    BANCARIO_GERAL = "bancario_geral"  # fallback inteligente (dúvida genérica)


class RoutingDecision(BaseModel):
    """Saída estruturada do roteador.

    Repare que NÃO devolvemos texto livre. Devolvemos um objeto validado. Isso
    permite que o código (não o LLM) decida o que fazer a seguir. Essa é a
    diferença entre um sistema auditável e um chatbot imprevisível.
    """

    intent: Intent = Field(description="Rota escolhida para a mensagem.")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Nível de confiança do classificador, entre 0 e 1.",
    )
    rationale: str = Field(
        description="Justificativa curta do porquê desta classificação. "
        "Serve para auditoria: um humano consegue revisar a decisão."
    )
    cliente_id: str = Field(description="Identificador único do cliente.")
    needs_clarification: bool = Field(
        default=False,
        description="True quando a confiança ficou abaixo do limiar e o "
        "sistema deve perguntar antes de rotear.",
    )
    clarification_question: str | None = Field(
        default=None,
        description="Pergunta a fazer ao cliente quando needs_clarification=True.",
    )


# ---------------------------------------------------------------------------
# 2. PARECERES DOS ANALISTAS (sub-grafos de concessão e renegociação)
# ---------------------------------------------------------------------------
class Verdict(str, Enum):
    """Veredito padronizado que cada analista pode emitir.

    DEFINIÇÃO — "veredito": a conclusão binária/ternária de um analista. O
    decisor combina os vereditos por REGRA (em código), nunca pelo "feeling"
    do LLM.
    """

    APROVADO = "aprovado"
    REPROVADO = "reprovado"
    NEUTRO = "neutro"  # sem objeção, mas sem aprovação explícita


class AnalystOpinion(BaseModel):
    """Parecer de um único analista (perfil, risco OU políticas).

    Os três analistas rodam EM PARALELO e cada um devolve um destes objetos.
    O decisor recebe os três e aplica regras determinísticas.
    """

    analyst: Literal["perfil", "risco", "politicas"] = Field(
        description="Qual analista produziu este parecer."
    )
    verdict: Verdict = Field(description="Conclusão do analista.")
    summary: str = Field(description="Resumo textual do parecer (para humanos).")
    # Dados numéricos vêm SEMPRE de tools/modelos — nunca inventados pelo LLM.
    # Por isso ficam em um dicionário tipado e são a fonte da verdade numérica.
    structured_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Números e fatos vindos de tools/modelos preditivos. "
        "Esta é a ÚNICA fonte legítima de valores numéricos na resposta final.",
    )


# ---------------------------------------------------------------------------
# 3. PROPOSTAS (saída dos decisores)
# ---------------------------------------------------------------------------
class CreditProposal(BaseModel):
    """Proposta de concessão de crédito gerada pelo decisor de concessão."""

    produto: str = Field(description="Nome do produto de crédito proposto.")
    limite: float = Field(ge=0, description="Limite aprovado, em reais.")
    taxa_mensal: float = Field(
        ge=0, description="Taxa de juros mensal, em fração (ex.: 0.0199 = 1,99%)."
    )
    prazo_meses: int = Field(ge=1, description="Prazo em meses.")
    disclaimer: str = Field(
        default="",
        description="Disclaimer regulatório obrigatório (preenchido por guardrail).",
    )


class RenegotiationOption(BaseModel):
    """Uma das N opções de renegociação ordenadas pelo decisor."""

    prazo_meses: int = Field(ge=1)
    taxa_mensal: float = Field(ge=0)
    desconto_aplicado: float = Field(
        ge=0, le=1, description="Desconto sobre o saldo, em fração (0..1)."
    )
    parcela: float = Field(ge=0, description="Valor da parcela resultante, em reais.")
    disclaimer: str = Field(default="")


# ---------------------------------------------------------------------------
# 4. PERFIL DO CLIENTE (usado para personalização e guardrails)
# ---------------------------------------------------------------------------
class CustomerProfile(BaseModel):
    cliente_id: str
    nome: str = Field(default="Cliente")
    segmento: str = Field(default="varejo", description="Ex.: varejo, alta_renda, pj.")
    canal: str = Field(default="app", description="Ex.: app, web, agencia.")
    autenticado: bool = Field(
        default=False,
        description="Se False, tools transacionais devem ser bloqueadas pelos "
        "guardrails de entrada.",
    )


# ---------------------------------------------------------------------------
# 5. ESTADO DO GRAFO — o objeto que atravessa todo o pipeline
# ---------------------------------------------------------------------------
class GuardrailFlag(BaseModel):
    """Registro de um disparo de guardrail, para auditoria."""

    stage: Literal["entrada", "execucao", "saida"]
    name: str
    blocked: bool = Field(description="True se o guardrail interrompeu o fluxo.")
    detail: str = ""


class GraphState(BaseModel):
    """Estado compartilhado por todos os nós do grafo.

    Cada nó recebe este objeto, lê o que precisa e devolve uma versão
    atualizada. É a "memória de trabalho" de um turno de conversa.
    """

    # --- entrada ---
    correlation_id: str = Field(
        description="ID único do turno. Liga cliente↔sessão↔decisão nos logs. "
        "É a espinha dorsal da auditabilidade."
    )
    cliente_id: str
    mensagem: str = Field(description="A mensagem original do cliente.")
    profile: CustomerProfile | None = None

    # --- roteamento ---
    routing: RoutingDecision | None = None

    # --- pareceres dos analistas (preenchidos em paralelo) ---
    opinions: list[AnalystOpinion] = Field(default_factory=list)

    # --- saída das células ---
    credit_proposal: CreditProposal | None = None
    renegotiation_options: list[RenegotiationOption] = Field(default_factory=list)

    # --- resposta ---
    raw_answer: str = Field(
        default="", description="Resposta bruta antes do refinamento."
    )
    final_answer: str = Field(
        default="", description="Resposta final, refinada e validada."
    )

    # --- auditoria / controle de fluxo ---
    guardrail_flags: list[GuardrailFlag] = Field(default_factory=list)
    escalate_to_human: bool = Field(
        default=False,
        description="True quando um caso de borda exige atendente humano.",
    )
    human_handoff_summary: str = Field(default="")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("correlation_id", "cliente_id", "mensagem")
    @classmethod
    def _nao_vazio(cls, v: str) -> str:
        """Campos essenciais não podem chegar vazios — falha cedo e claro."""
        if not v or not v.strip():
            raise ValueError("campo obrigatório vazio")
        return v

    def add_flag(self, stage: str, name: str, blocked: bool, detail: str = "") -> None:
        """Atalho para registrar um disparo de guardrail no estado."""
        self.guardrail_flags.append(
            GuardrailFlag(stage=stage, name=name, blocked=blocked, detail=detail)  # type: ignore[arg-type]
        )
