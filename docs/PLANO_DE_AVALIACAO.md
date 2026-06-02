# Plano de Avaliação

> Como medimos qualidade e desempenho, e como versionamos prompts, KBs e modelos.
> "O que não se mede, não se melhora."

## 1. Métricas

Todas são computadas por `src/credito_agentes/observability/evaluation.py`, que
roda um *golden set* rotulado de conversas. Para executar:

```bash
python -m credito_agentes.observability.evaluation
```

### 1.1 Precisão de roteamento

**DEFINIÇÃO.** Fração de mensagens classificadas na rota correta:
`acertos / total`, medida sobre o golden set rotulado (`ROUTING_GOLDEN`).

- **Meta:** ≥ 95%.
- **Resultado atual (FakeLLM determinístico):** 100% sobre 8 casos.
- **Como interpretar:** uma queda aqui indica prompt de roteamento degradado ou
  mudança de distribuição das mensagens reais. Aja revisando exemplos do prompt.

### 1.2 Taxa de alucinação numérica

**DEFINIÇÃO.** Fração de respostas em que o guardrail de não-alucinação numérica
precisou bloquear/reescrever porque apareceu um número financeiro **não
autorizado** (não derivado de tool/modelo/output estruturado).

- **Meta:** 0%.
- **Resultado atual:** 0%.
- **Como interpretar:** qualquer valor > 0% é uma regressão crítica — significa
  que números estão "vazando" do LLM para a resposta. Investigue o formatter e o
  conjunto de números autorizados em `guardrails/rules.py`.

### 1.3 Latência p50 e p95

**DEFINIÇÃO.** `p50` é a mediana; `p95` é o tempo abaixo do qual ficam 95% das
respostas (captura a cauda lenta — o pior caso que o usuário sente).

- **Meta (offline, FakeLLM):** p95 < 50 ms. **Atual:** p50 ≈ 11 ms, p95 ≈ 11 ms.
- **Meta (produção, LLM real):** p95 < 4 s para fallback/roteamento; < 8 s para
  decisão de crédito (modelo grande).
- **Por que paralelizar os analistas:** a célula gasta `max(perfil, risco,
  políticas)` em vez da soma — diferença direta no p95.

## 2. Golden set

O golden set vive em código (`ROUTING_GOLDEN` + testes em `tests/golden/`). Cada
caso é um par *(mensagem, resultado esperado)*. Regras:

- Toda correção de bug de roteamento/decisão **adiciona** um caso ao golden set
  (teste de regressão).
- O golden set roda no CI a cada push (ver `.github/workflows/ci.yml`).

## 3. Estratégia de versionamento

### 3.1 Prompts

- Prompts ficam versionados **no código** (constantes como `_SYSTEM_ROTEADOR`),
  então seguem o versionamento Git e passam por *code review*.
- O CI inclui um passo de **validação de prompts**: garante que cada prompt
  crítico existe e contém as instruções obrigatórias (ex.: "responda só com JSON").
- Mudança de prompt → roda golden set → compara precisão de roteamento antes/depois.

### 3.2 Bases de conhecimento (KBs)

- Cada chunk carrega `versao_politica` e `data_vigencia` (metadados obrigatórios).
- Ingestão **versionada**: nova versão entra com pipeline de aprovação; a versão
  antiga é mantida para *rollback*.
- O filtro de recuperação **sempre** restringe a políticas vigentes na data atual,
  evitando citar normas revogadas.
- Versionamento sugerido: `politica-concessao@v3 (vigência 2025-01-01)`.

### 3.3 Modelos

- O `tier` do modelo (`small`/`large`) é escolhido por etapa: médios
  (GPT-4o-mini / Claude Haiku) para roteamento, fallback e refinamento; grandes
  só para a **decisão de crédito**.
- Troca de modelo → re-rodar todas as métricas e comparar; registrar o número da
  versão do modelo no *trace* (correlation_id liga cliente ↔ sessão ↔ decisão).
- *Canary*: subir o novo modelo para uma fração do tráfego e comparar métricas
  antes do rollout completo.

## 4. Observabilidade

Cada turno gera *spans* (`observability/tracing.py`) com `correlation_id` ligando
cliente → sessão → decisão, para auditoria. Em produção, exporte os spans para
OpenTelemetry / LangSmith. As durações por etapa alimentam diretamente p50/p95.
