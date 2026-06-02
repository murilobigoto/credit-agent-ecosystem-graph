# Ecossistema Multi-Agente de Credito - Banco Poneiu

Ecossistema conversacional multi-agente para atendimento de credito bancario.
Orquestrador determinístico (padrao Supervisor + Workers) que roteia a mensagem
do cliente para uma de tres celulas de decisao, aplica guardrails em tres
momentos e refina a resposta antes de entregar ao cliente.

O projeto usa a API Anthropic (Claude) para roteamento e refinamento.
Todas as decisoes criticas de credito sao feitas por regras em codigo,
nunca delegadas ao LLM.

---

## Arquitetura

```
Mensagem do cliente
       |
       v  [guardrails ENTRADA: PII, injection, jailbreak, autenticacao]
  Roteador  ----  confianca < 0.7  ---->  pede clarificacao
       |   (LLM small: classifica intent em JSON)
       |
  RoutingDecision.intent
       |
       +---> concessao      ---> Celula de Concessao
       |                         (3 analistas paralelos + decisor)
       +---> renegociacao   ---> Celula de Renegociacao
       |                         (3 analistas paralelos + decisor)
       +---> bancario_geral ---> Agente Bancario
                                  (FAQ + tools read-only)
       |
       v  [guardrails EXECUCAO: schema, rate limit, coerencia, limites duros]
       |
  Camada de Refinamento
       |   (LLM small: empatia, sumarizacao, personalizacao)
       |
       v  [guardrails SAIDA: toxicidade, disclaimer, nao-alucinacao numerica]
       |
  Resposta ao cliente
```

---

## Estrutura do repositorio

```
credito-agentes-banco-poneiu/
├── .env                         variaveis de ambiente (API key, modelos)
├── .env.example                 template de configuracao
├── pyproject.toml               dependencias, build, pytest, ruff
├── requirements.txt             dependencias de runtime
├── README.md                    este arquivo
│
├── docs/
│   ├── FLUXO.md                 arquitetura detalhada + diagramas Mermaid
│   └── PLANO_DE_AVALIACAO.md    metricas e versionamento
│
├── scripts/
│   ├── setup_local.sh           cria venv e instala dependencias
│   └── validate_prompts.py      valida prompts criticos (CI)
│
├── src/credito_agentes/
│   ├── cli.py                   CLI interativa e modo demo
│   ├── data/
│   │   └── mock_db.py           banco de dados mock (CRM, CBS, SICC, ML)
│   ├── orchestrator/
│   │   ├── router.py            roteador: classifica intent via LLM
│   │   └── graph.py             grafo principal: conecta todos os nos
│   ├── agents/
│   │   ├── banking/             agente bancario (FAQ + tools read-only)
│   │   ├── concessao/           celula de concessao de credito
│   │   ├── renegociacao/        celula de renegociacao de dividas
│   │   └── refinement/          camada de refinamento de saida
│   ├── tools/
│   │   └── banking_tools.py     tools + modelos preditivos (usam mock_db)
│   ├── kb/
│   │   ├── retrieval.py         RAG: BM25 + denso + rerank + vigencia
│   │   └── registry.py          quatro KBs por dominio (FAQ, politicas, catalogo)
│   ├── guardrails/
│   │   └── rules.py             regras determinísticas (entrada/execucao/saida)
│   ├── schemas/
│   │   └── contracts.py         contratos Pydantic (GraphState, propostas, etc.)
│   ├── llm/
│   │   └── client.py            FakeLLM (testes) + RealLLM (API Anthropic)
│   └── observability/
│       ├── tracing.py           correlation_id + coleta de spans
│       └── evaluation.py        metricas: roteamento, alucinacao, latencia
│
└── tests/
    ├── unit/                    guardrails, KB, roteador
    └── golden/                  conversas ponta a ponta (golden set)
```

---

## Configuracao e execucao

### Pre-requisitos

- Python 3.10+
- Chave de API Anthropic (obtenha em console.anthropic.com)

### Setup

```bash
# 1. Clone e entre no diretorio
git clone <url-do-repo> && cd credito-agentes-banco-poneiu

# 2. Crie o ambiente virtual e instale as dependencias
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # dev: pytest, ruff
pip install anthropic             # SDK da API Anthropic

# 3. Configure as credenciais
cp .env.example .env
# Edite .env: defina ANTHROPIC_API_KEY com sua chave
```

### Executar

```bash
# Demo com roteiro de exemplos (usa API real se configurada)
python -m credito_agentes.cli --demo

# Modo interativo
python -m credito_agentes.cli

# Demo com cliente especifico
python -m credito_agentes.cli --demo --nome "Carlos" --segmento alta_renda
```

### Rodar sem API (modo offline)

```bash
# Sobrescreve a configuracao do .env para usar FakeLLM
USE_FAKE_LLM=1 python -m credito_agentes.cli --demo
```

---

## Clientes disponíveis no mock

O arquivo `src/credito_agentes/data/mock_db.py` define cinco perfis para
demonstracao. Cada perfil tem dados realistas de CRM, Core Banking, sistema
de cobranca e scores do modelo ML.

| ID       | Nome                   | Segmento   | Score | Situacao             |
|----------|------------------------|------------|-------|----------------------|
| CLI001   | Ana Beatriz Costa      | varejo     | 682   | Sem dividas, bom     |
| CLI002   | Carlos Eduardo Rocha   | alta_renda | 841   | Excelente historico  |
| CLI003   | Roberto Almeida Souza  | varejo     | 398   | Dividas em atraso    |
| CLI004   | Fernanda Lima Torres   | varejo     | 761   | Bom historico        |
| cli_demo | (padrao CLI)           | varejo     | 650   | Perfil neutro        |

---

## Testes

```bash
# Suite completa (offline, FakeLLM)
USE_FAKE_LLM=1 pytest -q

# Com cobertura
USE_FAKE_LLM=1 pytest --cov=credito_agentes -q

# Lint
ruff check src tests
```

Todos os testes rodam com `FakeLLM` (offline, deterministico).
A CI usa `USE_FAKE_LLM=1` e nunca requer credenciais.

---

## Guardrails

Tres momentos, todos determinísticos (codigo, nunca LLM):

**ENTRADA:** detecta CPF/cartao, prompt injection, jailbreak.

**EXECUCAO:**
- Validacao de schema das tools
- Rate limit por sessao (30 chamadas/min)
- `decision_coherence`: nunca aprova se risco reprovou ou politicas recusaram
- `enforce_hard_limits`: taxa minima, prazo maximo, desconto maximo
- `should_escalate`: score na zona limitrofe (0,45-0,55) vai para humano

**SAIDA:**
- Filtro de toxicidade
- Disclaimer regulatorio obrigatorio em propostas
- `numeric_non_hallucination`: cada numero da resposta e validado contra
  o output estruturado da celula (CreditProposal / RenegotiationOption)

---

## Arquitetura das celulas de credito

Ambas as celulas (concessao e renegociacao) seguem o mesmo padrao:

```
Analista perfil  (consulta_perfil_cliente / consulta_contratos_ativos)
Analista risco   (modelo_risco_concessao / modelo_risco_renegociacao)   -> em paralelo
Analista politicas (RAG na KB de politicas vigentes)

Decisor (regras em codigo: decision_coherence + enforce_hard_limits)
```

Os tres analistas rodam em paralelo via `ThreadPoolExecutor`.
A latencia total e `max(t_perfil, t_risco, t_politicas)`, nao a soma.

---

## Fontes de dados simuladas

Cada tool simula um servico interno do banco:

| Tool                        | Simula                                  |
|-----------------------------|-----------------------------------------|
| `consulta_saldo`            | Core Banking System (CBS)               |
| `uso_limite_cartao`         | CBS - modulo de cartoes                 |
| `uso_limite_cheque_especial`| CBS - modulo de conta corrente          |
| `consulta_perfil_cliente`   | CRM / sistema de onboarding             |
| `consulta_contratos_ativos` | Sistema de cobranca (SICC)              |
| `modelo_risco_concessao`    | Plataforma ML - endpoint de scoring     |
| `modelo_risco_renegociacao` | Plataforma ML - modelo de recuperacao   |

Em producao: substitua as funcoes do `mock_db.py` por chamadas HTTP reais.
A interface das tools e os schemas Pydantic permanecem identicos.

---

## Bases de conhecimento (RAG)

Quatro KBs independentes com conteudo versionado e filtro de vigencia:

| KB                       | Conteudo                                      |
|--------------------------|-----------------------------------------------|
| `faq_bancario`           | Saldo, PIX, cartao, cheque especial, TED, seguranca |
| `politicas_concessao`    | Elegibilidade, compliance, escalada para humano      |
| `politicas_renegociacao` | Descontos, prazos, taxas, fluxo de renegociacao      |
| `catalogo_produtos`      | Credito pessoal, consignado, cartao, cheque especial |

Pipeline de recuperacao: BM25 (lexical) + cosine (vetorial) + reranking.
Filtro de vigencia garante que politicas revogadas nunca sejam citadas.

---

## Adicionar uma nova celula

1. Crie `src/credito_agentes/agents/<nome>/cell.py` com classe que exponha
   `run(state: GraphState) -> GraphState`.
2. Devolva `AnalystOpinion` nos analistas e preencha o campo correto do
   `GraphState` no decisor. Nunca invente numeros: use tools.
3. Adicione o novo `Intent` em `schemas/contracts.py` e o ramo em
   `orchestrator/graph.py`.
4. Adicione casos no golden set (`tests/golden/`) e rode `pytest`.

---
