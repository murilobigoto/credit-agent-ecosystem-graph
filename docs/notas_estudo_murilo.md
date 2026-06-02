# Notas de Estudo — Fundamentos Teoricos do Ecossistema de Credito

Murilo Bigoto

---

## Indice

1. [Algebra Linear: o substrato matematico de tudo](#1-algebra-linear)
2. [Teoria da Informacao e Entropia](#2-teoria-da-informacao)
3. [Modelos de Linguagem: da probabilidade ao Transformer](#3-modelos-de-linguagem)
4. [Mecanismo de Atencao (Self-Attention)](#4-mecanismo-de-atencao)
5. [Embeddings e Semantica Distribucional](#5-embeddings)
6. [Recuperacao de Informacao: BM25 e Similaridade Vetorial](#6-recuperacao-de-informacao)
7. [Vector Databases e Busca Aproximada (ANN)](#7-vector-databases)
8. [RAG: Recuperacao Aumentada por Geracao](#8-rag)
9. [Teoria dos Grafos: fundamentos formais](#9-teoria-dos-grafos)
10. [O Grafo deste Projeto como Automato Finito](#10-o-grafo-do-projeto)
11. [Sistemas Multi-Agente: modelo formal](#11-sistemas-multi-agente)
12. [Guardrails como Restricoes de Otimizacao](#12-guardrails)
13. [Amortizacao Price: derivacao completa](#13-amortizacao-price)
14. [Metricas de Avaliacao](#14-metricas-de-avaliacao)
15. [Conexoes entre os Componentes](#15-conexoes)

---

## 1. Algebra Linear

Toda computacao em machine learning e, em ultima instancia, algebra linear.
Entender os espacos vetoriais com rigor e pre-requisito para tudo que segue.

### 1.1 Espaco Vetorial

**Definicao formal (axiomas de Peano/Moore).**
Um espaco vetorial V sobre um corpo F (tipicamente R ou C) e um conjunto
equipado com duas operacoes:

$$+ : V \times V \to V \quad \text{(adicao de vetores)}$$
$$* : F \times V \to V \quad \text{(multiplicacao por escalar)}$$

que satisfazem os 8 axiomas: comutatividade e associatividade da adicao,
existencia de elemento neutro e inverso aditivo, distributividade escalar
sobre vetor e sobre escalar, associatividade escalar, elemento neutro
multiplicativo.

No projeto: cada texto e convertido em um vetor em $\mathbb{R}^d$ (d tipicamente
768, 1536, ou 3072 dimensoes). Operacoes sobre esses vetores determinam
quais documentos sao "proximos" semanticamente.

### 1.2 Norma e Produto Interno

**Produto interno em $\mathbb{R}^n$:**

$$\langle u, v \rangle = u^T v = \sum_{i=1}^{n} u_i \cdot v_i$$

Propriedades: bilinearidade, simetria, positividade definitiva.

**Norma euclidiana (L2):**

$$\|v\|_2 = \sqrt{\langle v, v \rangle} = \sqrt{\sum_{i=1}^{n} v_i^2}$$

**Desigualdade de Cauchy-Schwarz** (fundamental para similaridade):

$$|\langle u, v \rangle| \leq \|u\|_2 \cdot \|v\|_2$$

com igualdade se e somente se u e v sao linearmente dependentes (paralelos).

**Similaridade de Cosseno** emerge diretamente:

$$\cos(\theta) = \frac{\langle u, v \rangle}{\|u\|_2 \cdot \|v\|_2}$$

$$\cos(u, v) \in [-1, 1]$$
$$\cos(u, v) = 1 \Rightarrow \text{vetores identicos (mesma direcao)}$$
$$\cos(u, v) = 0 \Rightarrow \text{vetores ortogonais (semanticamente nao relacionados)}$$
$$\cos(u, v) = -1 \Rightarrow \text{vetores antiparalelos (opostos semanticos)}$$

No projeto (kb/retrieval.py, funcao _cosine): usamos um bag-of-words como
vetor de embedding. Em producao, seria um vetor denso de 1536 dims do
text-embedding-3-large da OpenAI, ou equivalente Anthropic/Cohere.

### 1.3 Transformacoes Lineares e Matrizes

Uma transformacao linear $T: V \to W$ satisfaz:

$$T(u + v) = T(u) + T(v)$$
$$T(\alpha \cdot v) = \alpha \cdot T(v)$$

Representada por uma matriz $M \in \mathbb{R}^{m \times n}$. A composicao de transformacoes
lineares e multiplicacao de matrizes — operacao $O(n^3)$ naive, $O(n^{2.37})$ com
Coppersmith-Winograd.

**Por que importa no Transformer:** as matrizes de peso $W^Q$, $W^K$, $W^V$, $W^O$
sao transformacoes lineares aprendidas que projetam o espaco de entrada
em subespacos especializados para query, key e value.

### 1.4 Softmax: de Scores a Distribuicoes de Probabilidade

$$\text{softmax}(z)_i = \frac{\exp(z_i)}{\sum_{j=1}^{K} \exp(z_j)}$$

Propriedades:
- $\text{softmax}(z)_i \in (0, 1)$ para todo i
- $\sum_i \text{softmax}(z)_i = 1$ (e uma distribuicao de probabilidade valida)
- Diferenciavel em todo ponto (essencial para backpropagation)
- Invariante a adicao de constante: $\text{softmax}(z + c) = \text{softmax}(z)$
  (estabilidade numerica: usamos $\text{softmax}(z - \max(z))$)

A operacao $\exp(.)$ amplifica diferencas: se $z_i \gg z_j$, entao
$\text{softmax}(z)_i \approx 1$ e $\text{softmax}(z)_j \approx 0$. Isso cria "atencao esparsa"
quando os scores tem alta variancia.

---

## 2. Teoria da Informacao

Shannon (1948) fundou a teoria da informacao para quantificar incerteza.
Os LLMs sao treinados para minimizar entropia cruzada — conexao direta.

### 2.1 Entropia de Shannon

Para uma variavel aleatoria discreta X com suporte em $\{x_1,...,x_K\}$
e distribuicao P:

$$H(X) = H(P) = -\sum_{i=1}^{K} P(x_i) \log_2 P(x_i)$$

(convencao: $0 \cdot \log 0 = 0$)

- $H(P) \geq 0$ sempre
- $H(P) = 0$ sse a distribuicao e deterministica (toda massa em um ponto)
- $H(P) = \log_2(K)$ maxima para distribuicao uniforme

**Intuicao:** $H(P)$ mede o numero medio de bits necessarios para codificar
amostras de P. Um LLM perfeito, que previsse o proximo token com certeza
absoluta, teria $H = 0$. Um LLM que nao sabe nada e usa distribuicao uniforme
sobre o vocabulario de 100K tokens tem $H = \log_2(100000) \approx 17$ bits.

### 2.2 Divergencia de Kullback-Leibler

$$D_{KL}(P \| Q) = \sum_i P(x_i) \log\left(\frac{P(x_i)}{Q(x_i)}\right)$$

- $D_{KL}(P \| Q) \geq 0$ (Desigualdade de Gibbs)
- $D_{KL}(P \| Q) = 0$ sse $P = Q$ quase certamente
- NAO e simetrica: $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$

Interpretacao: custo extra de usar o codigo otimo para Q quando a
distribuicao verdadeira e P.

### 2.3 Entropia Cruzada e Treinamento de LLMs

$$H(P, Q) = H(P) + D_{KL}(P \| Q) = -\sum_i P(x_i) \log Q(x_i)$$

A funcao de perda de um LLM e a entropia cruzada entre a distribuicao
verdadeira dos tokens P (empirica, one-hot no token correto) e a
distribuicao prevista $Q_\theta$ pelo modelo com parametros $\theta$:

$$L(\theta) = -\sum_{t=1}^{T} \log Q_\theta(x_t | x_1, \ldots, x_{t-1})$$

Minimizar $L(\theta)$ equivale a minimizar $D_{KL}(P \| Q_\theta)$, ou seja,
fazer $Q_\theta$ se aproximar de P. O gradiente e calculado via
backpropagation atraves de todas as camadas do Transformer.

**Perplexidade:** metrica derivada da entropia cruzada:

$$\text{PPL} = \exp(H(P, Q)) = \exp\left(-\frac{1}{T} \sum_{t=1}^{T} \log Q_\theta(x_t | x_{<t})\right)$$

PPL = 1 e perfeito, PPL = $|V|$ e aleatoria ($|V|$ = tamanho do vocabulario).
GPT-4 tem PPL $\approx$ 5-10 em texto geral, Claude $\approx$ similar.

---

## 3. Modelos de Linguagem

### 3.1 Modelo de Linguagem como Distribuicao Condicional

Um modelo de linguagem e uma distribuicao de probabilidade sobre sequencias
de tokens. Pela regra da cadeia:

$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} P(x_t | x_1, \ldots, x_{t-1})$$

O desafio e modelar $P(x_t | x_1, \ldots, x_{t-1})$ de forma eficiente.

**N-gramas (historico):** aproximam $P(x_t | x_{t-1}, \ldots, x_{t-n+1})$
assumindo que apenas os n-1 tokens anteriores importam. Falham em capturar
dependencias de longa distancia.

**Redes Neurais Recorrentes (RNN, LSTM):** processam sequencias com estado
oculto $h_t = f(h_{t-1}, x_t)$. O gradiente "desaparece" ou "explode" para
sequencias longas (gradiente vanishing/exploding problem).

**Transformers:** Vaswani et al. (2017) — "Attention is All You Need".
Elimina recorrencia. Calcula dependencias entre todos os pares de tokens
em paralelo via self-attention.

### 3.2 Arquitetura do Transformer

Input: sequencia de tokens $(x_1, \ldots, x_T)$, cada convertido em embedding
$e_t \in \mathbb{R}^{d_{\text{model}}}$.

**Camada do Transformer** (L camadas empilhadas):

Entrada: $X \in \mathbb{R}^{T \times d_{\text{model}}}$

**1. Multi-Head Self-Attention:**
$$X' = \text{LayerNorm}(X + \text{MultiHeadAttn}(X, X, X))$$

**2. Feed-Forward Network:**
$$X'' = \text{LayerNorm}(X' + \text{FFN}(X'))$$

onde:
$$\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 \cdot x + b_1) + b_2$$

$W_1 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$, $W_2 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$

$d_{\text{ff}} = 4 \cdot d_{\text{model}}$ tipicamente (ex: $d_{\text{model}}=768$, $d_{\text{ff}}=3072$)

**Positional Encoding** (posicoes nao tem ordem intrinseca em atencao):

$$PE(\text{pos}, 2i) = \sin\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$
$$PE(\text{pos}, 2i+1) = \cos\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

Ou aprendido como embedding de posicao. O encoding sinosoidal garante que
$PE(\text{pos}+k)$ seja funcao linear de $PE(\text{pos})$, preservando nocao de deslocamento.

---

## 4. Mecanismo de Atencao

O mecanismo de atencao e a inovacao central do Transformer e a razao pela
qual LLMs conseguem capturar dependencias de longa distancia.

### 4.1 Atencao Scaled Dot-Product

Dados:
- Query $Q \in \mathbb{R}^{n \times d_k}$ (o que estou buscando)
- Key $K \in \mathbb{R}^{m \times d_k}$ (o que esta disponivel)
- Value $V \in \mathbb{R}^{m \times d_v}$ (o conteudo a ser recuperado)

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**Derivacao passo a passo:**

**Passo 1: Scores de compatibilidade**

$$S = QK^T \in \mathbb{R}^{n \times m}$$
$$S_{ij} = q_i \cdot k_j \quad \text{(produto interno entre query i e key j)}$$

$S_{ij}$ mede o quanto o token i "presta atencao" no token j.

**Passo 2: Escalonamento**

$$S_{\text{scaled}} = \frac{S}{\sqrt{d_k}}$$

**Por que dividir por $\sqrt{d_k}$?**
Se q e k sao vetores com componentes i.i.d. com media 0 e variancia 1,
entao $q \cdot k = \sum_{i=1}^{d_k} q_i \cdot k_i$ tem variancia $d_k$ (soma de $d_k$
termos independentes de variancia 1). Logo o desvio padrao e $\sqrt{d_k}$.
Sem o escalonamento, para $d_k$ grande os scores tem variancia alta, o que
empurra o softmax para regioes de gradiente muito pequeno (saturacao),
dificultando o treinamento.

**Passo 3: Mascaramento (para decoder/causal LM)**

$$S_{\text{masked}}_{ij} = \begin{cases} S_{\text{scaled}}_{ij} & \text{se } j \leq i \\ -\infty & \text{se } j > i \end{cases}$$

Isso garante que token i so "ve" tokens anteriores (causalidade).
Na pratica: adiciona uma mascara triangular inferior antes do softmax.

**Passo 4: Distribuicao de atencao**

$$A = \text{softmax}(S_{\text{masked}}) \in \mathbb{R}^{n \times m}$$
$$A_{ij} \geq 0, \sum_j A_{ij} = 1$$

A linha i de A e uma distribuicao de probabilidade sobre as m posicoes.
$A_{ij}$ e a atencao do token i sobre o token j.

**Passo 5: Agregacao ponderada dos values**

$$\text{Output} = A \cdot V \in \mathbb{R}^{n \times d_v}$$
$$\text{Output}_i = \sum_j A_{ij} \cdot v_j$$

O output do token i e uma media ponderada dos values, com pesos dados
pela atencao. O token i "agrega" informacao de todos os outros tokens
na proporcao da atencao.

### 4.2 Multi-Head Attention

Em vez de uma unica atencao com $d_{\text{model}}$ dims, usamos h cabecas paralelas
com $d_k = d_v = d_{\text{model}} / h$ dims cada:

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

onde:
- $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \cdot W^O$$

$W^O \in \mathbb{R}^{h \cdot d_v \times d_{\text{model}}}$

**Por que multiplas cabecas?**
Cada cabeca aprende a atender a diferentes aspectos:
- Cabeca 1: relacoes sintaticas (sujeito-verbo)
- Cabeca 2: correferencia (pronome -> antecedente)
- Cabeca 3: relacoes semanticas (sinonimos, antwnimos)
- ...

A interpretabilidade mostra que cabecas distintas capturam padroes
linguisticos qualitativamente diferentes (Vig, 2019; Clark et al., 2019).

### 4.3 Complexidade Computacional

Self-attention tem complexidade $O(T^2 \cdot d)$ em tempo e espaco, onde T e
o comprimento da sequencia. Para T = 100K tokens, isso e proibitivo.

Solucoes: Flash Attention (recomputa ativacoes para economizar memoria,
$O(T^2)$ tempo mas $O(T)$ memoria), Sparse Attention (atende apenas a
subconjuntos), Linear Attention ($O(T)$ atraves de kernel trick).

**Claude (Anthropic) usa** um mecanismo de atencao com janela de contexto
de 200K tokens (Claude 3+), possibilitado por Flash Attention 2 e
arquitetura otimizada.

### 4.4 Geracao Autoregressiva e Temperatura

Em inferencia, o LLM gera tokens um por um:

$$x_{t+1} \sim P(\cdot | x_1, \ldots, x_t) = \text{softmax}(\text{logits} / \tau)$$

onde logits = $h_t \cdot W^{\text{vocab}}$ (projecao do estado oculto para o vocabulario)
e $\tau > 0$ e a temperatura.

$$\tau \to 0: \text{distribuicao colapsa no argmax (greedy decoding, deterministica)}$$
$$\tau = 1: \text{distribuicao original do modelo}$$
$$\tau \to \infty: \text{distribuicao uniforme (ruido puro)}$$

No projeto: o roteador usa temperatura baixa (quase deterministica) para
classificacao de intent — queremos reproducibilidade. O refinamento pode
usar temperatura maior para texto mais natural.

---

## 5. Embeddings

### 5.1 Hipotese Distribucional

Firth (1957): "You shall know a word by the company it keeps."

Matematicamente: palavras com distribuicoes de contexto similares
tem significados similares. O embedding e uma representacao vetorial
que captura essa distribuicao de contexto.

### 5.2 Word2Vec: Skip-Gram com Negative Sampling

Mikolov et al. (2013). Para cada palavra alvo w e palavra de contexto c
numa janela de tamanho k:

**Objetivo:** maximizar a probabilidade de observar os contextos reais
e minimizar a de contextos negativos amostrados:

$$L = \sum_{(w,c)} \left[ \log \sigma(v_w \cdot u_c) + \sum_{i=1}^{k} \mathbb{E}_{n_i \sim P_n} [\log \sigma(-v_w \cdot u_{n_i})] \right]$$

onde:
- $\sigma(x) = \frac{1}{1 + \exp(-x)}$ (funcao sigmoide)
- $P_n(w) \propto f(w)^{3/4}$ (distribuicao de noise, freq. elevada a 3/4)
- $v_w \in \mathbb{R}^d$ (embedding de output de w)
- $u_c \in \mathbb{R}^d$ (embedding de contexto de c)

Apos treinamento: $v_w$ e $u_c$ capturam propriedades semanticas/sintaticas.
A propriedade famous:

$$v(\text{rei}) - v(\text{homem}) + v(\text{mulher}) \approx v(\text{rainha})$$

decorre de regularidades lineares na estrutura de coocorrencia do corpus.

### 5.3 Embeddings Contextuais (Transformer-based)

Diferenca crucial: em Word2Vec, cada palavra tem UM vetor fixo.
Em Transformers: o embedding de uma palavra depende do contexto completo.

$$\text{embed}(\text{"banco"} | \text{"fui ao banco sacar dinheiro"}) \neq \text{embed}(\text{"banco"} | \text{"sentei no banco do parque"})$$

Os embeddings do projeto (kb/retrieval.py) sao bag-of-words deterministicos
para rodar offline. Em producao: usar text-embedding-3-large (3072 dims,
OpenAI) ou embed-multilingual-v3.0 (Cohere), que produzem vetores
contextuais de qualidade.

### 5.4 Propriedades Geometricas do Espaco de Embeddings

**Isotropia vs. Anisotropia:**
Idealmente os vetores estariam uniformemente distribuidos na esfera unitaria
(isotropia). Na pratica, embeddings de LLMs mostram forte anisotropia:
a maioria da variancia esta em poucas direcoes, correlacionadas com
frequencia de tokens (fenomeno "rogue dimensions").

**Implicacao pratica:** similaridade de cosseno e mais robusta do que
distancia euclidiana para embeddings de LLMs, pois normaliza a norma
(que varia com frequencia).

---

## 6. Recuperacao de Informacao

### 6.1 BM25: Derivacao Matematica Completa

BM25 (Best Match 25) e o estado da arte em recuperacao lexical.
Deriva de um modelo probabilistico de IR (Robertson & Sparck Jones, 1994).

**Modelo probabilistico base:**

Queremos estimar $P(\text{Relevante} | d, q)$. Pela regra de Bayes:

$$P(R | d, q) \propto \frac{P(q | d, R)}{P(q | d, NR)}$$

A razao de verossimilhancas (log-odds) para cada termo t da query e:

$$\log\left[\frac{P(t | d, R)}{P(t | d, NR)}\right] \approx \log\left[\frac{r_t / R}{(n_t - r_t) / (N - R)}\right]$$

onde N = nr de docs, R = nr de docs relevantes, $n_t$ = nr de docs com t,
$r_t$ = nr de docs relevantes com t.

A formula BM25 para um documento d e query $q = \{t_1, \ldots, t_n\}$:

$$\text{BM25}(d, q) = \sum_{t \in q} \text{IDF}(t) \cdot \text{TF}_{\text{BM25}}(t, d)$$

**IDF (Inverse Document Frequency):**

$$\text{IDF}(t) = \log\left[ \frac{N - \text{df}(t) + 0.5}{\text{df}(t) + 0.5} + 1 \right]$$

onde $\text{df}(t)$ = numero de documentos que contem o termo t.
- Termos raros: $\text{df}(t) \ll N$ => IDF alto (termo discriminativo)
- Termos comuns: $\text{df}(t) \approx N$ => IDF proximo de 0 (pouco informativo)

**TF_BM25 (Term Frequency com saturacao):**

$$\text{TF}_{\text{BM25}}(t, d) = \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot (1 - b + b \cdot |d| / \text{avgdl})}$$

onde:
- $f(t, d)$ = frequencia bruta do termo t no documento d
- $|d|$ = comprimento do documento (em tokens)
- $\text{avgdl}$ = comprimento medio dos documentos na colecao
- $k_1 \in [1.2, 2.0]$: controla a saturacao de TF ($k_1=0$ => binario, $k_1 \to \infty$ => TF puro)
- $b \in [0, 1]$: controla normalizacao por comprimento ($b=0$ => sem normaliz., $b=1$ => total)

Valores tipicos: $k_1=1.5$, $b=0.75$ (usados no projeto)

**Analise da saturacao de TF:**

$$\lim_{f(t,d) \to \infty} \text{TF}_{\text{BM25}} = k_1 + 1$$

Isso e crucial: em TF-IDF classico, um documento que menciona um termo
1000 vezes e 10x melhor ranqueado que um que menciona 100 vezes.
No BM25, ha um teto — mais mencoes alem de certo ponto nao ajudam mais.

**Implementacao no projeto (kb/retrieval.py):**

```python
def _bm25_score(self, query_tokens, chunk):
    n = len(self.chunks)
    score = 0.0
    dl = len(chunk._tokens)
    for termo in query_tokens:
        tf = chunk._vector.get(termo, 0)
        if tf == 0:
            continue
        df = self._df.get(termo, 0)
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        denom = tf + k1 * (1 - b + b * dl / avg_len)
        score += idf * (tf * (k1 + 1)) / denom
    return score
```

### 6.2 Recuperacao Vetorial: Cosine Similarity

$$\text{sim}(q, d) = \cos(e_q, e_d) = \frac{\langle e_q, e_d \rangle}{\|e_q\|_2 \cdot \|e_d\|_2}$$

Para vetores normalizados ($\|e\| = 1$):

$$\text{sim}(q, d) = \langle e_q, e_d \rangle = e_q^T \cdot e_d$$

**Busca exata:** $O(N \cdot d)$ por query, onde N = nr de documentos e d = dims.
Para N = $10^6$ e d = 1536: $\approx 1.5 \times 10^9$ operacoes. Factivel em CPU para
N moderado, requer GPU para N grande.

### 6.3 Fusao Hibrida (BM25 + Vetorial)

**Problema:** BM25 e bom para termos exatos (siglas, numeros de norma como
"CMN 4.966"), mas ruim para sinonimos. Vetorial e bom para semantica, mas
pode perder correspondencias exatas.

**Solucao: Reciprocal Rank Fusion (RRF)** — alternativa ao projeto, que usa
min-max normalizado:

**RRF (Cormack et al., 2009):**

$$\text{RRF}_{\text{score}}(d) = \sum_{r \in \text{rankings}} \frac{1}{k + \text{rank}_r(d)}$$

onde $k = 60$ (constante de suavizacao), $\text{rank}_r(d)$ e a posicao de d no
ranking r.

**Implementacao no projeto (min-max normalizado):**

```python
def _norm(d):
    lo, hi = min(d.values()), max(d.values())
    if hi - lo < 1e-9:
        return {k: 0.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}

score_hibrido = 0.5 * bm25_normalizado + 0.5 * cosine_normalizado
```

### 6.4 Re-ranking com Cross-Encoder

O pipeline de dois estagios e padrao em IR de producao:

**Estagio 1 (BI-ENCODER, barato):**
Para cada documento $d_i$: encode(q) e encode($d_i$) independentemente.
Computa similaridade de cosseno. Top-k = 20 candidatos.
Complexidade: $O(N)$ na busca vetorial.

**Estagio 2 (CROSS-ENCODER, caro):**
Para cada candidato $d_i$: encode(q, $d_i$) JUNTOS (par concatenado).
Score de relevancia $s(q, d_i)$. Top-k' = 5 resultados finais.
Complexidade: $O(20)$ passagens pelo modelo pesado.

O cross-encoder captura interacoes entre query e documento que o
bi-encoder (embeddings independentes) nao consegue. Por exemplo:
"taxa de juros alta" pode ter embedding similar a "custos financeiros
elevados" mesmo sem palavras em comum — cross-encoder detecta isso.

**Implementacao aproximada no projeto:**

```python
def _rerank_score(c):
    overlap = len(set(query_tokens) & set(c._tokens)) / len(set(query_tokens))
    return 0.6 * overlap + 0.4 * fundido[c.chunk_id]
```

Em producao: usar bge-reranker-v2-m3 (BAAI) ou Cohere Rerank v3.

---

## 7. Vector Databases e Busca Aproximada (ANN)

### 7.1 O Problema de Busca do Vizinho Mais Proximo

Dado um conjunto de N vetores em $\mathbb{R}^d$ e uma query q, encontrar:

$$\argmax_{d_i \in \text{corpus}} \text{sim}(q, d_i)$$

Busca exata e $O(N \cdot d)$ por query. Para N = $10^9$ e d = 1536: impraticavel.

Solucao: **Approximate Nearest Neighbor (ANN)** — encontra o vizinho mais
proximo com alta probabilidade (tipicamente 95-99% recall), em $O(\log N)$
ou $O(\sqrt{N})$ dependendo do algoritmo.

### 7.2 HNSW: Hierarchical Navigable Small World

Malkov & Yashunin (2018). Estrutura hierarquica de grafos de vizinhanca
proxima.

**Construcao:**
- Camada 0: todos os nos (grafo denso de vizinhos proximos)
- Camada l: subconjunto aleatorio dos nos da camada l-1 (grafo esparso)
- Probabilidade de no na camada l: $\exp(-l / m_L)$, $m_L = 1/\ln(M)$

**Busca (query q):**

```
Algoritmo HNSW-Search(q, K):
  1. Comeca no no de entrada na camada mais alta
  2. Para l = L, L-1, ..., 1:
     - Navega gananciosa na camada l ate nao melhorar mais
  3. Na camada 0: busca gulosa com fila de prioridade, retorna top-K
```

Complexidade: $O(\log N)$ esperado.

**Parametros:**
- M: numero de vizinhos por no (trade-off recall/memoria)
- ef_construction: tamanho da lista de candidatos na construcao
- ef_search: tamanho da lista de candidatos na busca

**Exemplos de uso:** Pinecone, Weaviate, pgvector, Chroma,
Qdrant, Milvus — todos implementam HNSW.

### 7.3 Product Quantization (PQ)

Para economizar memoria com N muito grande:

Divide o vetor em M subvetores: $v = [v^1, \ldots, v^M]$
Cada $v^j \in \mathbb{R}^{d/M}$ e quantizado para um dos K centroides do subespaco j.
Codigo: $c = [c^1, \ldots, c^M]$ ($M \cdot \log_2(K)$ bits)

Com M=8 e K=256: cada vetor de 1536 floats (6144 bytes) fica em 8 bytes
(78x compressao). O custo e uma pequena perda de recall (tipicamente < 2%).

---

## 8. RAG: Recuperacao Aumentada por Geracao

### 8.1 Fundamento Teorico

Lewis et al. (2020) — "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks."

O modelo generativo $P_\theta(y | x)$ e augmentado por um retriever $q_\phi(z | x)$:

$$P_{\text{RAG}}(y | x) = \sum_z P_\theta(y | x, z) \cdot q_\phi(z | x)$$

Aproximado pelo top-K documentos:

$$P_{\text{RAG-seq}}(y | x) \approx \sum_{z \in \text{top-K}} P_\theta(y | x, z) \cdot q_\phi(z | x)$$

Na pratica, para classificacao e extracao (nao geracao de texto longo):
usa-se apenas o documento com maior score.

### 8.2 Por que RAG resolve Alucinacao

Um LLM sem RAG tem o conhecimento "baked" nos parametros W durante o treino.
Para fatos que mudam (taxas de juros, normas legais, dados de clientes):

$$\text{Sem RAG: } P(\text{resposta correta}) \text{ depende de se o fato esta nos dados de treino}$$
$$\text{e se nao foi "sobrescrito" por outros fatos durante o treino.}$$

$$\text{Com RAG: } P(\text{resposta correta}) = P(\text{retriever encontra o doc correto})$$
$$\times P(\text{LLM extrai a informacao corretamente})$$

O segundo fator e muito mais alto que o primeiro quando o contexto relevante
esta explicitamente no prompt. LLMs sao muito bons em "extrair e seguir"
instrucoes quando o contexto e fornecido.

### 8.3 Chunking Semantico

**Problema com chunking por caracteres fixos:**

```
Texto: "A taxa minima e de 1,2%. O prazo maximo e de 48 meses."
Chunk de 25 chars: "A taxa minima e de 1,2%." | ". O prazo maximo e de 48"
```

A segunda chunk perdeu o contexto da primeira sentenca. Queries sobre
"prazo maximo" recuperam um chunk que comeca com ". O prazo maximo",
que pode ser ambiguo.

**Chunking semantico (implementacao no projeto):**

```python
def semantic_chunk(texto, max_chars=600):
    paragrafos = re.split(r"\n\s*\n", texto)  # split em linhas em branco
    chunks = []
    buffer = ""
    for p in paragrafos:
        if not buffer:
            buffer = p
        elif len(buffer) + len(p) + 1 <= max_chars:
            buffer = f"{buffer}\n{p}"
        else:
            chunks.append(buffer)
            buffer = p
    if buffer:
        chunks.append(buffer)
    return chunks
```

Respeita limites de paragrafos/clausulas, garantindo que cada chunk
seja uma unidade semanticamente coerente.

### 8.4 Filtro de Vigencia (Temporal Relevance)

Problema especifico de dominio juridico/financeiro: normas sao revogadas.
Usar "Resolucao CMN 3.517/2007" para CET quando a norma vigente e outra
e nao apenas errado — pode ser regulatoriamente problematico.

**Solucao: metadados de vigencia em cada chunk:**

```python
@dataclass
class ChunkMetadata:
    versao_politica: str
    vigencia_inicio: date
    vigencia_fim: date | None

def vigente_em(self, quando: date) -> bool:
    if quando < self.vigencia_inicio:
        return False
    if self.vigencia_fim is not None and quando > self.vigencia_fim:
        return False
    return True
```

**Filtro na recuperacao:**

```python
elegiveis = [c for c in self.chunks if c.metadata.vigente_em(hoje)]
```

Apenas chunks vigentes entram no pipeline de ranqueamento.
Chunks de politicas revogadas sao silenciosamente descartados.

---

## 9. Teoria dos Grafos

### 9.1 Definicao Formal

**Grafo:** $G = (V, E)$ onde:
- V e um conjunto de vertices (nos)
- $E \subseteq V \times V$ e um conjunto de arestas (edges)

**Grafo Direcionado (Digrafo):** arestas sao pares ordenados $(u, v)$.
$(u, v) \in E$ significa "ha uma aresta de u para v" (u -> v).

**Grafo Ponderado:** funcao de peso $w: E \to \mathbb{R}$ associa um valor real a
cada aresta.

### 9.2 Propriedades Relevantes

**Grau:** $\deg(v)$ = numero de arestas incidentes em v.
Em digrafos: grau de entrada $\text{in-deg}(v)$ e de saida $\text{out-deg}(v)$.

**Caminho:** sequencia de vertices $(v_0, v_1, \ldots, v_k)$ tal que
$(v_i, v_{i+1}) \in E$ para todo i.

**Alcancabilidade:** v e alcancavel de u se existe caminho de u para v.

**Componente Fortemente Conexa (SCC):** subconjunto maximal de vertices
onde todo par e mutuamente alcancavel.

**Grafo Aciclico Direcionado (DAG):** digrafo sem ciclos.
Propriedade chave: todo DAG tem uma ordenacao topologica.

### 9.3 Automato Finito Determinista (DFA)

**Definicao formal:**

$$M = (Q, \Sigma, \delta, q_0, F) \text{ onde:}$$
- $Q$ = conjunto finito de estados
- $\Sigma$ = alfabeto de entrada
- $\delta : Q \times \Sigma \to Q$ (funcao de transicao TOTAL e deterministica)
- $q_0 \in Q$ (estado inicial)
- $F \subseteq Q$ (estados finais/aceitadores)

**Computacao:** dado string $w = w_1 w_2 \ldots w_n$:

$$r_0 = q_0$$
$$r_i = \delta(r_{i-1}, w_i) \text{ para } i = 1, \ldots, n$$
$$w \text{ e aceita sse } r_n \in F$$

**Transicao como grafo:** cada estado $q \in Q$ e um vertice, cada
transicao $\delta(q, a) = p$ cria uma aresta rotulada com a de q para p.

### 9.4 Maquina de Moore vs. Maquina de Mealy

**Maquina de Moore:** saida depende APENAS do estado atual.

$$\lambda : Q \to \Gamma \quad \text{(funcao de saida)}$$

**Maquina de Mealy:** saida depende do estado E do simbolo de entrada.

$$\lambda : Q \times \Sigma \to \Gamma$$

O sistema de roteamento do projeto e uma Maquina de Moore:
dado o estado GraphState (contendo routing.intent), a proxima celula
a executar e determinada APENAS pelo estado (nao pela mensagem original).

---

## 10. O Grafo do Projeto como Automato Finito

### 10.1 Definicao Formal do Grafo de Agentes

Defino formalmente o CreditAgentGraph como:

$$G_{\text{credito}} = (Q, \Sigma, \delta, q_0, F) \text{ onde:}$$

$$Q = \{q_{\text{inicial}}, q_{\text{roteamento}}, q_{\text{clarificacao}}, q_{\text{concessao}},$$
$$q_{\text{renegociacao}}, q_{\text{bancario}}, q_{\text{refinamento}}, q_{\text{final}}, q_{\text{bloqueado}}\}$$

$$\Sigma = \{$$
- $\text{msg\_valida}$ — mensagem passou guardrails de entrada
- $\text{msg\_bloqueada}$ — guardrails bloquearam
- $\text{intent\_concessao}$ — roteador classificou como concessao
- $\text{intent\_renegociacao}$ — roteador classificou como renegociacao
- $\text{intent\_bancario}$ — roteador classificou como bancario_geral
- $\text{baixa\_confianca}$ — confidence < 0.7
- $\text{celula\_concluida}$ — celula produziu output
- $\text{refinamento\_concluido}$ — refinamento produziu final_answer
$$\}$$

**Transicoes $\delta$:**

$$q_{\text{inicial}} + \text{msg\_valida} \to q_{\text{roteamento}}$$
$$q_{\text{inicial}} + \text{msg\_bloqueada} \to q_{\text{bloqueado}}$$
$$q_{\text{roteamento}} + \text{baixa\_confianca} \to q_{\text{clarificacao}}$$
$$q_{\text{roteamento}} + \text{intent\_concessao} \to q_{\text{concessao}}$$
$$q_{\text{roteamento}} + \text{intent\_renegociacao} \to q_{\text{renegociacao}}$$
$$q_{\text{roteamento}} + \text{intent\_bancario} \to q_{\text{bancario}}$$
$$q_{\text{concessao}} + \text{celula\_concluida} \to q_{\text{refinamento}}$$
$$q_{\text{renegociacao}} + \text{celula\_concluida} \to q_{\text{refinamento}}$$
$$q_{\text{bancario}} + \text{celula\_concluida} \to q_{\text{refinamento}}$$
$$q_{\text{refinamento}} + \text{refinamento\_concluido} \to q_{\text{final}}$$
$$q_{\text{clarificacao}} \to q_{\text{final}} \quad \text{(resposta = pergunta de clarificacao)}$$
$$q_{\text{bloqueado}} \to q_{\text{final}} \quad \text{(resposta = mensagem segura)}$$

$$q_0 = q_{\text{inicial}}$$
$$F = \{q_{\text{final}}\}$$

**Propriedade de determinismo:** $\delta$ e uma FUNCAO (nao relacao). Dado
qualquer par (estado, simbolo), ha exatamente uma transicao. Isso e
a garantia arquitetural central: o comportamento e completamente previsivel
e testavel.

### 10.2 GraphState como Vetor de Estado

O GraphState e um objeto Pydantic que funciona como o "estado de memoria"
do automato. Formalmente:

$$\text{GraphState} = \{$$
- $\text{correlation\_id} : \text{str}$ — identificador do turno (auditabilidade)
- $\text{cliente\_id} : \text{str}$ — identificador do cliente
- $\text{mensagem} : \text{str}$ — input do usuario
- $\text{profile} : \text{CustomerProfile} | \text{None}$
- $\text{routing} : \text{RoutingDecision} | \text{None}$ — saida do roteador
- $\text{opinions} : \text{list}[\text{AnalystOpinion}]$ — saidas dos analistas
- $\text{credit\_proposal} : \text{CreditProposal} | \text{None}$ — saida do decisor de concessao
- $\text{renegotiation\_options} : \text{list}[\text{RenegotiationOption}]$ — saida do decisor reneg.
- $\text{raw\_answer} : \text{str}$ — resposta bruta
- $\text{final\_answer} : \text{str}$ — resposta final
- $\text{guardrail\_flags} : \text{list}[\text{GuardrailFlag}]$ — auditoria de guardrails
- $\text{escalate\_to\_human} : \text{bool}$
- $\text{timestamp} : \text{datetime}$
$$\}$$

A transicao $\delta(q, \sigma)$ em termos de codigo:

```python
# orchestrator/graph.py
def invoke(self, mensagem, *, cliente_id, profile):
    # q_inicial -> q_roteamento
    state = self.router.run(state)

    # q_roteamento -> q_bloqueado (se final_answer ja preenchido)
    if state.final_answer:
        return state  # estado final q_bloqueado

    # q_roteamento -> q_clarificacao
    if state.routing.needs_clarification:
        state.final_answer = state.routing.clarification_question
        return state  # estado final q_clarificacao

    # q_roteamento -> q_concessao | q_renegociacao | q_bancario
    intent = state.routing.intent
    if intent == Intent.CONCESSAO:
        state = self.concessao.run(state)
    elif intent == Intent.RENEGOCIACAO:
        state = self.renegociacao.run(state)
    else:
        state = self.banking.run(state)

    # q_{celula} -> q_refinamento -> q_final
    state = self.refinement.run(state)
    return state
```

### 10.3 Celulas como Sub-Grafos (Grafo Hierarquico)

Cada celula e ela mesma um sub-grafo:

$$G_{\text{concessao}} = \{$$
- nos: $\{\text{analista\_perfil, analista\_risco, analista\_politicas, decisor}\}$
- arestas: $\{$$
  - GraphState $\to$ $\{\text{analista\_perfil, analista\_risco, analista\_politicas}\}$ (paralelo)
  - $\{\text{analista\_perfil, analista\_risco, analista\_politicas}\} \to$ decisor (join)
  - decisor $\to$ GraphState_atualizado
$$\}$$
$$\}$$

A estrutura paralela (fork-join) e implementada via `ThreadPoolExecutor`:

```python
with ThreadPoolExecutor(max_workers=3) as ex:
    f1 = ex.submit(_analista_perfil, state)
    f2 = ex.submit(_analista_risco, state)
    f3 = ex.submit(_analista_politicas, state, self.kb)
    state.opinions = [f1.result(), f2.result(), f3.result()]
```

**Complexidade de latencia:**
- Serial: $O(t_p + t_r + t_{\text{pol}})$ (soma)
- Paralelo: $O(\max(t_p, t_r, t_{\text{pol}}))$ (maximo)

Para $t_p \approx 10\text{ms}$, $t_r \approx 15\text{ms}$, $t_{\text{pol}} \approx 12\text{ms}$:
- Serial: $\approx 37\text{ms}$
- Paralelo: $\approx 15\text{ms}$ (2.5x mais rapido)

### 10.4 Propriedade de Terminacao

O automato $G_{\text{credito}}$ e acíclico (DAG direcionado acíclico) em condicoes
normais: nao ha ciclos (cada turno de conversa e sempre processado de
$q_{\text{inicial}}$ ate $q_{\text{final}}$ sem loops). O grafo executa no maximo

$$\text{|arestas|} = 1 \text{ (entrada)} + 1 \text{ (roteamento)} + 1 \text{ (celula)} + 1 \text{ (refinamento)} = 4 \text{ transicoes}$$

Portanto, a terminacao e garantida em tempo $O(\text{max\_cell\_time})$.
