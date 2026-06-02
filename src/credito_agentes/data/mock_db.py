"""
Banco de dados mock que representa sistemas de um banco de varejo real.

Em producao, cada funcao get_* seria uma chamada a um servico interno:
  - get_cliente     -> CRM / sistema de onboarding (cadastro e segmentacao)
  - get_conta       -> Core Banking System (CBS): saldos e limites
  - get_contratos   -> Sistema de cobranca e gestao de credito (SICC)
  - get_scores_risco -> Plataforma de ML interna (endpoint HTTP REST)

Os dados abaixo seguem o mesmo formato de retorno que esses servicos usariam,
com campos reais de um core bancario brasileiro (saldo disponivel, limite de
credito, score Serasa/SCR, etc.).

Clientes disponveis para demonstracao:
  CLI001 - Ana Beatriz Costa      | varejo     | bom pagador, sem dividas
  CLI002 - Carlos Eduardo Rocha   | alta_renda | excelente historico
  CLI003 - Roberto Almeida Souza  | varejo     | alto risco, dividas em atraso
  CLI004 - Fernanda Lima Torres   | varejo     | bom historico, relacionamento longo
  cli_demo                        | varejo     | perfil padrao do modo demo/CLI
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ClienteDB:
    """Perfil do cliente conforme retornado pelo CRM."""

    nome: str
    segmento: str                     # varejo | alta_renda | pj
    renda_mensal: float               # R$
    score_bureau: int                 # escala Serasa 0-1000
    tempo_relacionamento_meses: int
    canal_preferencial: str           # app | web | agencia | cli


@dataclass
class ContaDB:
    """Dados de conta conforme retornados pelo Core Banking System."""

    saldo_disponivel: float    # saldo atual na conta corrente, R$
    limite_cartao: float       # limite total do cartao de credito, R$
    uso_cartao: float          # valor utilizado do cartao, R$
    limite_cheque: float       # limite do cheque especial, R$
    uso_cheque: float          # valor utilizado do cheque especial, R$


@dataclass
class ContratoCredito:
    """Um contrato de credito ativo no sistema de cobranca."""

    numero: str
    produto: str
    saldo_devedor: float       # saldo atualizado com juros e multas, R$
    parcelas_em_atraso: int
    valor_parcela: float       # valor da parcela original, R$


@dataclass
class ClienteContratos:
    """Conjunto de contratos ativos de um cliente."""

    contratos: list[ContratoCredito] = field(default_factory=list)

    @property
    def saldo_total_devedor(self) -> float:
        return round(sum(c.saldo_devedor for c in self.contratos), 2)

    @property
    def total_parcelas_atraso(self) -> int:
        return sum(c.parcelas_em_atraso for c in self.contratos)


# ---------------------------------------------------------------------------
# Perfis de clientes (CRM / onboarding)
# ---------------------------------------------------------------------------
CLIENTES: dict[str, ClienteDB] = {
    "CLI001": ClienteDB(
        nome="Ana Beatriz Costa",
        segmento="varejo",
        renda_mensal=3_800.00,
        score_bureau=682,
        tempo_relacionamento_meses=36,
        canal_preferencial="app",
    ),
    "CLI002": ClienteDB(
        nome="Carlos Eduardo Rocha",
        segmento="alta_renda",
        renda_mensal=21_500.00,
        score_bureau=841,
        tempo_relacionamento_meses=96,
        canal_preferencial="agencia",
    ),
    "CLI003": ClienteDB(
        nome="Roberto Almeida Souza",
        segmento="varejo",
        renda_mensal=2_100.00,
        score_bureau=398,
        tempo_relacionamento_meses=14,
        canal_preferencial="app",
    ),
    "CLI004": ClienteDB(
        nome="Fernanda Lima Torres",
        segmento="varejo",
        renda_mensal=5_600.00,
        score_bureau=761,
        tempo_relacionamento_meses=60,
        canal_preferencial="web",
    ),
    "cli_demo": ClienteDB(
        nome="Ana",
        segmento="varejo",
        renda_mensal=4_200.00,
        score_bureau=650,
        tempo_relacionamento_meses=24,
        canal_preferencial="cli",
    ),
}

# ---------------------------------------------------------------------------
# Dados de conta (Core Banking System)
# ---------------------------------------------------------------------------
CONTAS: dict[str, ContaDB] = {
    "CLI001": ContaDB(
        saldo_disponivel=1_840.30,
        limite_cartao=4_000.00,
        uso_cartao=1_620.00,   # 40,5% utilizado
        limite_cheque=800.00,
        uso_cheque=0.00,
    ),
    "CLI002": ContaDB(
        saldo_disponivel=14_320.00,
        limite_cartao=32_000.00,
        uso_cartao=8_750.00,   # 27,3% utilizado
        limite_cheque=8_000.00,
        uso_cheque=0.00,
    ),
    "CLI003": ContaDB(
        saldo_disponivel=87.45,
        limite_cartao=1_500.00,
        uso_cartao=1_410.00,   # 94% utilizado
        limite_cheque=500.00,
        uso_cheque=496.00,     # 99,2% utilizado
    ),
    "CLI004": ContaDB(
        saldo_disponivel=4_210.80,
        limite_cartao=8_000.00,
        uso_cartao=2_640.00,   # 33% utilizado
        limite_cheque=2_000.00,
        uso_cheque=0.00,
    ),
    "cli_demo": ContaDB(
        saldo_disponivel=2_340.50,
        limite_cartao=5_000.00,
        uso_cartao=2_050.00,   # 41% utilizado
        limite_cheque=1_000.00,
        uso_cheque=0.00,
    ),
}

# ---------------------------------------------------------------------------
# Contratos de credito ativos (Sistema de Cobranca - SICC)
# ---------------------------------------------------------------------------
CONTRATOS: dict[str, ClienteContratos] = {
    "CLI001": ClienteContratos(),
    "CLI002": ClienteContratos(),
    "CLI003": ClienteContratos(contratos=[
        ContratoCredito(
            numero="CP-2023-84521",
            produto="Credito Pessoal Facil",
            saldo_devedor=7_840.00,
            parcelas_em_atraso=4,
            valor_parcela=412.00,
        ),
        ContratoCredito(
            numero="CC-2022-31045",
            produto="Cartao de Credito Parcelado",
            saldo_devedor=1_230.00,
            parcelas_em_atraso=2,
            valor_parcela=185.00,
        ),
    ]),
    "CLI004": ClienteContratos(),
    "cli_demo": ClienteContratos(),
}

# ---------------------------------------------------------------------------
# Scores de risco (Plataforma de ML interna)
#
# score_inadimplencia: 0=risco minimo, 1=risco maximo (modelo de classificacao
#                      treinado com historico de pagamentos e bureau externo)
# capacidade_pagamento: comprometimento maximo mensal recomendado, R$
#                       (30% da renda disponivel estimada pelo modelo)
# prob_cura: probabilidade de regularizacao da divida em 12 meses (renegociacao)
# recovery_score: potencial de recuperacao do valor devido (0-1)
# ---------------------------------------------------------------------------
SCORES_RISCO: dict[str, dict[str, float]] = {
    "CLI001": {
        "score_inadimplencia": 0.18,
        "capacidade_pagamento": 1_140.00,
        "prob_cura": 0.85,
        "recovery_score": 0.78,
    },
    "CLI002": {
        "score_inadimplencia": 0.07,
        "capacidade_pagamento": 6_450.00,
        "prob_cura": 0.96,
        "recovery_score": 0.93,
    },
    "CLI003": {
        "score_inadimplencia": 0.76,
        "capacidade_pagamento": 210.00,
        "prob_cura": 0.28,
        "recovery_score": 0.25,
    },
    "CLI004": {
        "score_inadimplencia": 0.21,
        "capacidade_pagamento": 1_680.00,
        "prob_cura": 0.89,
        "recovery_score": 0.83,
    },
    "cli_demo": {
        "score_inadimplencia": 0.25,
        "capacidade_pagamento": 1_260.00,
        "prob_cura": 0.80,
        "recovery_score": 0.74,
    },
}

# ---------------------------------------------------------------------------
# Fallbacks usados quando o cliente_id nao esta no mock
# ---------------------------------------------------------------------------
_DEFAULT_CLIENTE = ClienteDB(
    nome="Cliente",
    segmento="varejo",
    renda_mensal=2_500.00,
    score_bureau=500,
    tempo_relacionamento_meses=6,
    canal_preferencial="app",
)
_DEFAULT_CONTA = ContaDB(
    saldo_disponivel=450.00,
    limite_cartao=1_000.00,
    uso_cartao=500.00,
    limite_cheque=300.00,
    uso_cheque=0.00,
)
_DEFAULT_CONTRATOS = ClienteContratos()
_DEFAULT_SCORE: dict[str, float] = {
    "score_inadimplencia": 0.50,
    "capacidade_pagamento": 500.00,
    "prob_cura": 0.45,
    "recovery_score": 0.40,
}


def get_cliente(cliente_id: str) -> ClienteDB:
    """Retorna o perfil do cliente (simula consulta ao CRM)."""
    return CLIENTES.get(cliente_id, _DEFAULT_CLIENTE)


def get_conta(cliente_id: str) -> ContaDB:
    """Retorna os dados de conta (simula consulta ao Core Banking)."""
    return CONTAS.get(cliente_id, _DEFAULT_CONTA)


def get_contratos(cliente_id: str) -> ClienteContratos:
    """Retorna contratos ativos (simula consulta ao sistema de cobranca)."""
    return CONTRATOS.get(cliente_id, _DEFAULT_CONTRATOS)


def get_scores_risco(cliente_id: str) -> dict[str, float]:
    """Retorna scores do modelo de risco (simula endpoint HTTP da plataforma ML)."""
    return SCORES_RISCO.get(cliente_id, _DEFAULT_SCORE)
