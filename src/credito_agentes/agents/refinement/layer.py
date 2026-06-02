"""
Camada de Refinamento de Saída.
==================================================================

O QUE FAZ (explicação a nível de aula)
--------------------------------------
Depois que uma célula (ou o agente bancário) produz uma resposta BRUTA, ela não
vai direto ao cliente. Passa por esta camada, que:

  1. Constrói o texto final a partir do OUTPUT ESTRUTURADO (proposta/opções),
     não de números soltos — garantindo rastreabilidade.
  2. Reescreve em tom empático e linguagem acessível (LLM pequeno).
  3. Sumariza quando o texto fica longo.
  4. Aplica os GUARDRAILS DE SAÍDA: toxicidade, disclaimer regulatório e a
     regra de NÃO-ALUCINAÇÃO NUMÉRICA (cross-check contra o output estruturado).
  5. Personaliza com nome, segmento e canal do cliente.

POR QUE ESTA ORDEM IMPORTA
--------------------------
Construímos o texto a partir dos dados estruturados ANTES de deixar o LLM
"enfeitar". Assim sabemos exatamente quais números são legítimos e podemos
validar a versão final contra esse conjunto. Se o LLM introduzir um número novo,
o guardrail de não-alucinação bloqueia.
"""

from __future__ import annotations

from credito_agentes.guardrails import (
    authorized_numbers_from_options,
    authorized_numbers_from_proposal,
    check_output_toxicity,
    ensure_disclaimer,
    numeric_non_hallucination,
)
from credito_agentes.llm import LLMClient
from credito_agentes.schemas import GraphState


def _formatar_proposta(state: GraphState) -> str:
    p = state.credit_proposal
    assert p is not None
    return (
        f"Proposta: {p.produto}. Limite de R$ {p.limite:.2f}, "
        f"taxa de {p.taxa_mensal * 100:.2f}% ao mês, em {p.prazo_meses} meses."
    )


def _formatar_opcoes(state: GraphState) -> str:
    linhas = ["Opções de renegociação (da menor parcela para a maior):"]
    for i, o in enumerate(state.renegotiation_options, 1):
        linhas.append(
            f"{i}) {o.prazo_meses} meses, taxa {o.taxa_mensal * 100:.2f}% a.m., "
            f"desconto de {o.desconto_aplicado * 100:.0f}%, "
            f"parcela de R$ {o.parcela:.2f}."
        )
    return "\n".join(linhas)


class RefinementLayer:
    """Reescreve, valida e personaliza a resposta final."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, state: GraphState) -> GraphState:
        # Encaminhamento humano tem precedência: resposta fixa e segura.
        if state.escalate_to_human:
            state.final_answer = (
                "Vou transferir você para um de nossos especialistas, que dará "
                "continuidade ao seu atendimento com todo o contexto. Um instante."
            )
            return state

        # 1. Determina a fonte da verdade numérica e o texto-base.
        is_proposta = state.credit_proposal is not None
        is_opcoes = bool(state.renegotiation_options)
        autorizados: set[str] = set()

        if is_proposta:
            base = _formatar_proposta(state)
            autorizados = authorized_numbers_from_proposal(state.credit_proposal)  # type: ignore[arg-type]
        elif is_opcoes:
            base = _formatar_opcoes(state)
            autorizados = authorized_numbers_from_options(state.renegotiation_options)
        else:
            base = state.raw_answer

        # 2. Reescrita empática (LLM pequeno). O FakeLLM devolve o texto como veio.
        nome = state.profile.nome if state.profile else "Cliente"
        seg = state.profile.segmento if state.profile else "varejo"
        _ = state.profile.canal if state.profile else "app"  # canal disponível p/ personalização futura
        system = (
            "Você é a camada de refinamento. Reescreva a mensagem em tom empático "
            "e linguagem simples, SEM alterar nenhum número, valor, taxa ou prazo."
        )
        refinado = self.llm.complete(system, base, tier="small")

        # 3. Sumarização se ficou longo.
        if len(refinado) > 800:
            refinado = self.llm.complete(
                "Você é a camada de refinamento. Resuma mantendo todos os números.",
                refinado, tier="small",
            )

        # 4. Personalização (nome, segmento, canal).
        personalizado = f"Olá, {nome}! {refinado}"
        if seg == "alta_renda":
            personalizado += " Como cliente de alta renda, você conta com condições diferenciadas."

        # 5. Guardrails de saída.
        tox = check_output_toxicity(personalizado)
        state.add_flag("saida", tox.name, tox.blocked, tox.detail)
        if tox.blocked:
            state.final_answer = tox.safe_message
            return state

        personalizado = ensure_disclaimer(personalizado, is_credit_proposal=is_proposta)

        # Não-alucinação numérica: só validamos quando há fonte estruturada.
        if autorizados:
            # O disclaimer pode conter números regulatórios; validamos só o corpo.
            corpo = personalizado.split("\n\n")[0]
            checagem = numeric_non_hallucination(corpo, autorizados)
            state.add_flag("saida", checagem.name, checagem.blocked, checagem.detail)
            if checagem.blocked:
                # Fallback seguro: devolve o texto-base estruturado (confiável).
                state.final_answer = ensure_disclaimer(
                    f"Olá, {nome}! {base}", is_credit_proposal=is_proposta
                )
                return state

        state.final_answer = personalizado
        return state
