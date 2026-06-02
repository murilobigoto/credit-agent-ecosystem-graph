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

```
+ : V x V -> V    (adicao de vetores)
* : F x V -> V    (multiplicacao por escalar)
```

que satisfazem os 8 axiomas: comutatividade e associatividade da adicao,
existencia de elemento neutro e inverso aditivo, distributividade escalar
sobre vetor e sobre escalar, associatividade escalar, elemento neutro
multiplicativo.

No projeto: cada texto e convertido em um vetor em R^d (d tipicamente
768, 1536, ou 3072 dimensoes). Operacoes sobre esses vetores determinam
quais documentos sao "proximos" semanticamente.

### 1.2 Norma e Produto Interno

**Produto interno em R^n:**

```
<u, v> = u^T v = sum_{i=1}^{n} u_i * v_i
```

Propriedades: bilinearidade, simetria, positividade definitiva.

**Norma euclidiana (L2):**

```
||v||_2 = sqrt(<v, v>) = sqrt(sum_{i=1}^{n} v_i^2)
```

**Desigualdade de Cauchy-Schwarz** (fundamental para similaridade):

```
|<u, v>| <= ||u||_2 * ||v||_2
```

com igualdade se e somente se u e v sao linearmente dependentes (paralelos).

**Similaridade de Cosseno** emerge diretamente:

```
cos(theta) = <u, v> / (||u||_2 * ||v||_2)

cos(u, v) em [-1, 1]
cos(u, v) = 1  => vetores identicos (mesma direcao)
cos(u, v) = 0  => vetores ortogonais (semanticamente nao relacionados)
cos(u, v) = -1 => vetores antiparalelos (opostos semanticos)
```

No projeto (kb/retrieval.py, funcao _cosine): usamos um bag-of-words como
vetor de embedding. Em producao, seria um vetor denso de 1536 dims do
text-embedding-3-large da OpenAI, ou equivalente Anthropic/Cohere.

### 1.3 Transformacoes Lineares e Matrizes

Uma transformacao linear T: V -> W satisfaz:

```
T(u + v) = T(u) + T(v)
T(alpha * v) = alpha * T(v)
```

Representada por uma matriz M em R^{m x n}. A composicao de transformacoes
lineares e multiplicacao de matrizes — operacao O(n^3) naive, O(n^2.37) com
Coppersmith-Winograd.

**Por que importa no Transformer:** as matrizes de peso W^Q, W^K, W^V, W^O
sao transformacoes lineares aprendidas que projetam o espaco de entrada
em subespacos especializados para query, key e value.

### 1.4 Softmax: de Scores a Distribuicoes de Probabilidade

```
softmax(z)_i = exp(z_i) / sum_{j=1}^{K} exp(z_j)
```

Propriedades:
- softmax(z)_i em (0, 1) para todo i
- sum_i softmax(z)_i = 1  (e uma distribuicao de probabilidade valida)
- Diferenciavel em todo ponto (essencial para backpropagation)
- Invariante a adicao de constante: softmax(z + c) = softmax(z)
  (estabilidade numerica: usamos softmax(z - max(z)))

A operacao exp(.) amplifica diferencas: se z_i >> z_j, entao
softmax(z)_i ≈ 1 e softmax(z)_j ≈ 0. Isso cria "atencao esparsa"
quando os scores tem alta variancia.

---

## 2. Teoria da Informacao

Shannon (1948) fundou a teoria da informacao para quantificar incerteza.
Os LLMs sao treinados para minimizar entropia cruzada — conexao direta.

### 2.1 Entropia de Shannon

Para uma variavel aleatoria discreta X com suporte em {x_1,...,x_K}
e distribuicao P:

```
H(X) = H(P) = -sum_{i=1}^{K} P(x_i) * log_2 P(x_i)
```

(convencao: 0 * log 0 = 0)

- H(P) >= 0 sempre
- H(P) = 0 sse a distribuicao e deterministica (toda massa em um ponto)
- H(P) = log_2(K) maxima para distribuicao uniforme

**Intuicao:** H(P) mede o numero medio de bits necessarios para codificar
amostras de P. Um LLM perfeito, que previsse o proximo token com certeza
absoluta, teria H = 0. Um LLM que nao sabe nada e usa distribuicao uniforme
sobre o vocabulario de 100K tokens tem H = log_2(100000) ≈ 17 bits.

### 2.2 Divergencia de Kullback-Leibler

```
D_KL(P || Q) = sum_i P(x_i) * log(P(x_i) / Q(x_i))
```

- D_KL(P || Q) >= 0 (Desigualdade de Gibbs)
- D_KL(P || Q) = 0 sse P = Q quase certamente
- NAO e simetrica: D_KL(P || Q) != D_KL(Q || P)

Interpretacao: custo extra de usar o codigo otimo para Q quando a
distribuicao verdadeira e P.

### 2.3 Entropia Cruzada e Treinamento de LLMs

```
H(P, Q) = H(P) + D_KL(P || Q) = -sum_i P(x_i) * log Q(x_i)
```

A funcao de perda de um LLM e a entropia cruzada entre a distribuicao
verdadeira dos tokens P (empirica, one-hot no token correto) e a
distribuicao prevista Q_theta pelo modelo com parametros theta:

```
L(theta) = -sum_{t=1}^{T} log Q_theta(x_t | x_1, ..., x_{t-1})
```

Minimizar L(theta) equivale a minimizar D_KL(P || Q_theta), ou seja,
fazer Q_theta se aproximar de P. O gradiente e calculado via
backpropagation atraves de todas as camadas do Transformer.

**Perplexidade:** metrica derivada da entropia cruzada:

```
PPL = exp(H(P, Q)) = exp(-1/T * sum_{t=1}^{T} log Q_theta(x_t | x_<t))
```

PPL = 1 e perfeito, PPL = |V| e aleatoria (|V| = tamanho do vocabulario).
GPT-4 tem PPL ≈ 5-10 em texto geral, Claude ≈ similar.

---

## 3. Modelos de Linguagem

### 3.1 Modelo de Linguagem como Distribuicao Condicional

Um modelo de linguagem e uma distribuicao de probabilidade sobre sequencias
de tokens. Pela regra da cadeia:

```
P(x_1, x_2, ..., x_T) = prod_{t=1}^{T} P(x_t | x_1, ..., x_{t-1})
```

O desafio e modelar P(x_t | x_1, ..., x_{t-1}) de forma eficiente.

**N-gramas (historico):** aproximam P(x_t | x_{t-1}, ..., x_{t-n+1})
assumindo que apenas os n-1 tokens anteriores importam. Falham em capturar
dependencias de longa distancia.

**Redes Neurais Recorrentes (RNN, LSTM):** processam sequencias com estado
oculto h_t = f(h_{t-1}, x_t). O gradiente "desaparece" ou "explode" para
sequencias longas (gradiente vanishing/exploding problem).

**Transformers:** Vaswani et al. (2017) — "Attention is All You Need".
Elimina recorrencia. Calcula dependencias entre todos os pares de tokens
em paralelo via self-attention.

### 3.2 Arquitetura do Transformer

Input: sequencia de tokens (x_1, ..., x_T), cada convertido em embedding
e_t in R^{d_model}.

**Camada do Transformer** (L camadas empilhadas):

```
Entrada: X in R^{T x d_model}

1. Multi-Head Self-Attention:
   X' = LayerNorm(X + MultiHeadAttn(X, X, X))

2. Feed-Forward Network:
   X'' = LayerNorm(X' + FFN(X'))

onde:
FFN(x) = W_2 * ReLU(W_1 * x + b_1) + b_2
W_1 in R^{d_ff x d_model}, W_2 in R^{d_model x d_ff}
d_ff = 4 * d_model tipicamente (ex: d_model=768, d_ff=3072)
```

**Positional Encoding** (posicoes nao tem ordem intrinseca em atencao):

```
PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
```

Ou aprendido como embedding de posicao. O encoding sinosoidal garante que
PE(pos+k) seja funcao linear de PE(pos), preservando nocao de deslocamento.

---

## 4. Mecanismo de Atencao

O mecanismo de atencao e a inovacao central do Transformer e a razao pela
qual LLMs conseguem capturar dependencias de longa distancia.

### 4.1 Atencao Scaled Dot-Product

Dados:
- Query Q in R^{n x d_k}      (o que estou buscando)
- Key   K in R^{m x d_k}      (o que esta disponivel)
- Value V in R^{m x d_v}      (o conteudo a ser recuperado)

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) * V
```

**Derivacao passo a passo:**

**Passo 1: Scores de compatibilidade**

```
S = Q K^T in R^{n x m}
S_{ij} = q_i . k_j  (produto interno entre query i e key j)
```

S_{ij} mede o quanto o token i "presta atencao" no token j.

**Passo 2: Escalonamento**

```
S_scaled = S / sqrt(d_k)
```

**Por que dividir por sqrt(d_k)?**
Se q e k sao vetores com componentes i.i.d. com media 0 e variancia 1,
entao q.k = sum_{i=1}^{d_k} q_i * k_i tem variancia d_k (soma de d_k
termos independentes de variancia 1). Logo o desvio padrao e sqrt(d_k).
Sem o escalonamento, para d_k grande os scores tem variancia alta, o que
empurra o softmax para regioes de gradiente muito pequeno (saturacao),
dificultando o treinamento.

**Passo 3: Mascaramento (para decoder/causal LM)**

```
S_masked_{ij} = S_scaled_{ij}  se j <= i
              = -inf           se j > i
```

Isso garante que token i so "ve" tokens anteriores (causalidade).
Na pratica: adiciona uma mascara triangular inferior antes do softmax.

**Passo 4: Distribuicao de atencao**

```
A = softmax(S_masked)   in R^{n x m}
A_{ij} >= 0, sum_j A_{ij} = 1
```

A linha i de A e uma distribuicao de probabilidade sobre as m posicoes.
A_{ij} e a atencao do token i sobre o token j.

**Passo 5: Agregacao ponderada dos values**

```
Output = A * V   in R^{n x d_v}
Output_i = sum_j A_{ij} * v_j
```

O output do token i e uma media ponderada dos values, com pesos dados
pela atencao. O token i "agrega" informacao de todos os outros tokens
na proporcao da atencao.

### 4.2 Multi-Head Attention

Em vez de uma unica atencao com d_model dims, usamos h cabecas paralelas
com d_k = d_v = d_model / h dims cada:

```
head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

onde:
W_i^Q in R^{d_model x d_k}
W_i^K in R^{d_model x d_k}
W_i^V in R^{d_model x d_v}

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O

W^O in R^{h*d_v x d_model}
```

**Por que multiplas cabecas?**
Cada cabeca aprende a atender a diferentes aspectos:
- Cabeca 1: relacoes sintaticas (sujeito-verbo)
- Cabeca 2: correferencia (pronome -> antecedente)
- Cabeca 3: relacoes semanticas (sinonimos, antwnimos)
- ...

A interpretabilidade mostra que cabecas distintas capturam padroes
linguisticos qualitativamente diferentes (Vig, 2019; Clark et al., 2019).

### 4.3 Complexidade Computacional

Self-attention tem complexidade O(T^2 * d) em tempo e espaco, onde T e
o comprimento da sequencia. Para T = 100K tokens, isso e proibitivo.

Solucoes: Flash Attention (recomputa ativacoes para economizar memoria,
O(T^2) tempo mas O(T) memoria), Sparse Attention (atende apenas a
subconjuntos), Linear Attention (O(T) atraves de kernel trick).

**Claude (Anthropic) usa** um mecanismo de atencao com janela de contexto
de 200K tokens (Claude 3+), possibilitado por Flash Attention 2 e
arquitetura otimizada.

### 4.4 Geracao Autoregressiva e Temperatura

Em inferencia, o LLM gera tokens um por um:

```
x_{t+1} ~ P(. | x_1, ..., x_t) = softmax(logits / tau)
```

onde logits = h_t * W^{vocab} (projecao do estado oculto para o vocabulario)
e tau > 0 e a temperatura.

```
tau -> 0:  distribuicao colapsa no argmax (greedy decoding, deterministica)
tau = 1:   distribuicao original do modelo
tau -> inf: distribuicao uniforme (ruido puro)
```

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

```
L = sum_{(w,c)} [ log sigma(v_w . u_c)
               + sum_{i=1}^{k} E_{n_i ~ P_n} [log sigma(-v_w . u_{n_i})] ]

onde:
sigma(x) = 1/(1 + exp(-x))   (funcao sigmoide)
P_n(w) ∝ f(w)^{3/4}          (distribuicao de noise, freq. elevada a 3/4)
v_w ∈ R^d                      (embedding de output de w)
u_c ∈ R^d                      (embedding de contexto de c)
```

Apos treinamento: v_w e u_c capturam propriedades semanticas/sintaticas.
A propriedade famous:

```
v(rei) - v(homem) + v(mulher) ≈ v(rainha)
```

decorre de regularidades lineares na estrutura de coocorrencia do corpus.

### 5.3 Embeddings Contextuais (Transformer-based)

Diferenca crucial: em Word2Vec, cada palavra tem UM vetor fixo.
Em Transformers: o embedding de uma palavra depende do contexto completo.

```
embed("banco" | "fui ao banco sacar dinheiro") != embed("banco" | "sentei no banco do parque")
```

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

Queremos estimar P(Relevante | d, q). Pela regra de Bayes:

```
P(R | d, q) ∝ P(q | d, R) / P(q | d, NR)
```

A razao de verossimilhancas (log-odds) para cada termo t da query e:

```
log[P(t | d, R) / P(t | d, NR)] ≈ log[(r_t / R) / (n_t - r_t) / (N - R)]
```

onde N = nr de docs, R = nr de docs relevantes, n_t = nr de docs com t,
r_t = nr de docs relevantes com t.

A formula BM25 para um documento d e query q = {t_1, ..., t_n}:

```
BM25(d, q) = sum_{t in q} IDF(t) * TF_BM25(t, d)
```

**IDF (Inverse Document Frequency):**

```
IDF(t) = log[ (N - df(t) + 0.5) / (df(t) + 0.5) + 1 ]
```

onde df(t) = numero de documentos que contem o termo t.
- Termos raros: df(t) << N => IDF alto (termo discriminativo)
- Termos comuns: df(t) ≈ N => IDF proximo de 0 (pouco informativo)

**TF_BM25 (Term Frequency com saturacao):**

```
TF_BM25(t, d) = f(t, d) * (k1 + 1)
                ---------------------------------
                f(t, d) + k1 * (1 - b + b * |d| / avgdl)
```

onde:
- f(t, d) = frequencia bruta do termo t no documento d
- |d| = comprimento do documento (em tokens)
- avgdl = comprimento medio dos documentos na colecao
- k1 ∈ [1.2, 2.0]: controla a saturacao de TF (k1=0 => binario, k1->inf => TF puro)
- b ∈ [0, 1]: controla normalizacao por comprimento (b=0 => sem normaliz., b=1 => total)
  Valores tipicos: k1=1.5, b=0.75 (usados no projeto)

**Analise da saturacao de TF:**

```
lim_{f(t,d)->inf} TF_BM25 = k1 + 1
```

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

```
sim(q, d) = cos(e_q, e_d) = <e_q, e_d> / (||e_q||_2 * ||e_d||_2)
```

Para vetores normalizados (||e|| = 1):

```
sim(q, d) = <e_q, e_d> = e_q^T * e_d
```

**Busca exata:** O(N * d) por query, onde N = nr de documentos e d = dims.
Para N = 10^6 e d = 1536: ≈ 1.5 * 10^9 operacoes. Factivel em CPU para
N moderado, requer GPU para N grande.

### 6.3 Fusao Hibrida (BM25 + Vetorial)

**Problema:** BM25 e bom para termos exatos (siglas, numeros de norma como
"CMN 4.966"), mas ruim para sinonimos. Vetorial e bom para semantica, mas
pode perder correspondencias exatas.

**Solucao: Reciprocal Rank Fusion (RRF)** — alternativa ao projeto, que usa
min-max normalizado:

**RRF (Cormack et al., 2009):**

```
RRF_score(d) = sum_{r in rankings} 1 / (k + rank_r(d))
```

onde k = 60 (constante de suavizacao), rank_r(d) e a posicao de d no
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

```
Estagio 1 (BI-ENCODER, barato):
  Para cada documento d_i: encode(q) e encode(d_i) independentemente.
  Computa similaridade de cosseno. Top-k = 20 candidatos.
  Complexidade: O(N) na busca vetorial.

Estagio 2 (CROSS-ENCODER, caro):
  Para cada candidato d_i: encode(q, d_i) JUNTOS (par concatenado).
  Score de relevancia s(q, d_i). Top-k' = 5 resultados finais.
  Complexidade: O(20) passagens pelo modelo pesado.
```

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

Dado um conjunto de N vetores em R^d e uma query q, encontrar:

```
argmax_{d_i in corpus} sim(q, d_i)
```

Busca exata e O(N * d) por query. Para N = 10^9 e d = 1536: impraticavel.

Solucao: **Approximate Nearest Neighbor (ANN)** — encontra o vizinho mais
proximo com alta probabilidade (tipicamente 95-99% recall), em O(log N)
ou O(sqrt(N)) dependendo do algoritmo.

### 7.2 HNSW: Hierarchical Navigable Small World

Malkov & Yashunin (2018). Estrutura hierarquica de grafos de vizinhanca
proxima.

**Construcao:**
- Camada 0: todos os nos (grafo denso de vizinhos proximos)
- Camada l: subconjunto aleatorio dos nos da camada l-1 (grafo esparso)
- Probabilidade de no na camada l: exp(-l / m_L), m_L = 1/ln(M)

**Busca (query q):**
```
Algoritmo HNSW-Search(q, K):
  1. Comeca no no de entrada na camada mais alta
  2. Para l = L, L-1, ..., 1:
     - Navega gananciosa na camada l ate nao melhorar mais
  3. Na camada 0: busca gulosa com fila de prioridade, retorna top-K
```

Complexidade: O(log N) esperado.

**Parametros:**
- M: numero de vizinhos por no (trade-off recall/memoria)
- ef_construction: tamanho da lista de candidatos na construcao
- ef_search: tamanho da lista de candidatos na busca

**Exemplos de uso:** Pinecone, Weaviate, pgvector, Chroma,
Qdrant, Milvus — todos implementam HNSW.

### 7.3 Product Quantization (PQ)

Para economizar memoria com N muito grande:

```
Divide o vetor em M subvetores: v = [v^1, ..., v^M]
Cada v^j ∈ R^{d/M} e quantizado para um dos K centroides do subespaco j.
Codigo: c = [c^1, ..., c^M]  (M * log2(K) bits)
```

Com M=8 e K=256: cada vetor de 1536 floats (6144 bytes) fica em 8 bytes
(78x compressao). O custo e uma pequena perda de recall (tipicamente < 2%).

---

## 8. RAG: Recuperacao Aumentada por Geracao

### 8.1 Fundamento Teorico

Lewis et al. (2020) — "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks."

O modelo generativo P_theta(y | x) e augmentado por um retriever q_phi(z | x):

```
P_RAG(y | x) = sum_z P_theta(y | x, z) * q_phi(z | x)

Aproximado pelo top-K documentos:
P_RAG-seq(y | x) ≈ sum_{z in top-K} P_theta(y | x, z) * q_phi(z | x)
```

Na pratica, para classificacao e extracao (nao geracao de texto longo):
usa-se apenas o documento com maior score.

### 8.2 Por que RAG resolve Alucinacao

Um LLM sem RAG tem o conhecimento "baked" nos parametros W durante o treino.
Para fatos que mudam (taxas de juros, normas legais, dados de clientes):

```
Sem RAG: P(resposta correta) depende de se o fato esta nos dados de treino
         e se nao foi "sobrescrito" por outros fatos durante o treino.

Com RAG: P(resposta correta) = P(retriever encontra o doc correto)
                              * P(LLM extrai a informacao corretamente)
```

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

**Grafo:** G = (V, E) onde:
- V e um conjunto de vertices (nos)
- E ⊆ V × V e um conjunto de arestas (edges)

**Grafo Direcionado (Digrafo):** arestas sao pares ordenados (u, v).
(u, v) ∈ E significa "ha uma aresta de u para v" (u -> v).

**Grafo Ponderado:** funcao de peso w: E -> R associa um valor real a
cada aresta.

### 9.2 Propriedades Relevantes

**Grau:** deg(v) = numero de arestas incidentes em v.
Em digrafos: grau de entrada in-deg(v) e de saida out-deg(v).

**Caminho:** sequencia de vertices (v_0, v_1, ..., v_k) tal que
(v_i, v_{i+1}) ∈ E para todo i.

**Alcancabilidade:** v e alcancavel de u se existe caminho de u para v.

**Componente Fortemente Conexa (SCC):** subconjunto maximal de vertices
onde todo par e mutuamente alcancavel.

**Grafo Aciclico Direcionado (DAG):** digrafo sem ciclos.
Propriedade chave: todo DAG tem uma ordenacao topologica.

### 9.3 Automato Finito Determinista (DFA)

**Definicao formal:**

```
M = (Q, Sigma, delta, q_0, F) onde:
  Q     = conjunto finito de estados
  Sigma = alfabeto de entrada
  delta : Q x Sigma -> Q  (funcao de transicao TOTAL e deterministica)
  q_0   ∈ Q               (estado inicial)
  F ⊆ Q                   (estados finais/aceitadores)
```

**Computacao:** dado string w = w_1 w_2 ... w_n:

```
r_0 = q_0
r_i = delta(r_{i-1}, w_i)  para i = 1, ..., n
w e aceita sse r_n ∈ F
```

**Transicao como grafo:** cada estado q ∈ Q e um vertice, cada
transicao delta(q, a) = p cria uma aresta rotulada com a de q para p.

### 9.4 Maquina de Moore vs. Maquina de Mealy

**Maquina de Moore:** saida depende APENAS do estado atual.
```
lambda : Q -> Gamma   (funcao de saida)
```

**Maquina de Mealy:** saida depende do estado E do simbolo de entrada.
```
lambda : Q x Sigma -> Gamma
```

O sistema de roteamento do projeto e uma Maquina de Moore:
dado o estado GraphState (contendo routing.intent), a proxima celula
a executar e determinada APENAS pelo estado (nao pela mensagem original).

---

## 10. O Grafo do Projeto como Automato Finito

### 10.1 Definicao Formal do Grafo de Agentes

Defino formalmente o CreditAgentGraph como:

```
G_credito = (Q, Sigma, delta, q_0, F) onde:

Q = {q_inicial, q_roteamento, q_clarificacao, q_concessao,
     q_renegociacao, q_bancario, q_refinamento, q_final, q_bloqueado}

Sigma = {
  msg_valida,           -- mensagem passou guardrails de entrada
  msg_bloqueada,        -- guardrails bloquearam
  intent_concessao,     -- roteador classificou como concessao
  intent_renegociacao,  -- roteador classificou como renegociacao
  intent_bancario,      -- roteador classificou como bancario_geral
  baixa_confianca,      -- confidence < 0.7
  celula_concluida,     -- celula produziu output
  refinamento_concluido -- refinamento produziu final_answer
}

delta:
  q_inicial         x msg_valida           -> q_roteamento
  q_inicial         x msg_bloqueada        -> q_bloqueado
  q_roteamento      x baixa_confianca      -> q_clarificacao
  q_roteamento      x intent_concessao     -> q_concessao
  q_roteamento      x intent_renegociacao  -> q_renegociacao
  q_roteamento      x intent_bancario      -> q_bancario
  q_concessao       x celula_concluida     -> q_refinamento
  q_renegociacao    x celula_concluida     -> q_refinamento
  q_bancario        x celula_concluida     -> q_refinamento
  q_refinamento     x refinamento_concluido-> q_final
  q_clarificacao    -> q_final  (resposta = pergunta de clarificacao)
  q_bloqueado       -> q_final  (resposta = mensagem segura)

q_0 = q_inicial
F = {q_final}
```

**Propriedade de determinismo:** delta e uma FUNCAO (nao relacao). Dado
qualquer par (estado, simbolo), ha exatamente uma transicao. Isso e
a garantia arquitetural central: o comportamento e completamente previsivel
e testavel.

### 10.2 GraphState como Vetor de Estado

O GraphState e um objeto Pydantic que funciona como o "estado de memoria"
do automato. Formalmente:

```
GraphState = {
  correlation_id  : str,      -- identificador do turno (auditabilidade)
  cliente_id      : str,      -- identificador do cliente
  mensagem        : str,      -- input do usuario
  profile         : CustomerProfile | None,
  routing         : RoutingDecision | None,   -- saida do roteador
  opinions        : list[AnalystOpinion],     -- saidas dos analistas
  credit_proposal : CreditProposal | None,    -- saida do decisor de concessao
  renegotiation_options : list[RenegotiationOption], -- saida do decisor reneg.
  raw_answer      : str,      -- resposta bruta
  final_answer    : str,      -- resposta final
  guardrail_flags : list[GuardrailFlag],      -- auditoria de guardrails
  escalate_to_human : bool,
  timestamp       : datetime
}
```

A transicao delta(q, sigma) em termos de codigo:

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

```
G_concessao = {
  nos: {analista_perfil, analista_risco, analista_politicas, decisor}
  arestas: {
    GraphState -> {analista_perfil, analista_risco, analista_politicas}  (paralelo)
    {analista_perfil, analista_risco, analista_politicas} -> decisor     (join)
    decisor -> GraphState_atualizado
  }
}
```

A estrutura paralela (fork-join) e implementada via `ThreadPoolExecutor`:

```python
with ThreadPoolExecutor(max_workers=3) as ex:
    f1 = ex.submit(_analista_perfil, state)
    f2 = ex.submit(_analista_risco, state)
    f3 = ex.submit(_analista_politicas, state, self.kb)
    state.opinions = [f1.result(), f2.result(), f3.result()]
```

**Complexidade de latencia:**
- Serial: O(t_p + t_r + t_pol) (soma)
- Paralelo: O(max(t_p, t_r, t_pol)) (maximo)

Para t_p ≈ 10ms, t_r ≈ 15ms, t_pol ≈ 12ms:
- Serial: ≈ 37ms
- Paralelo: ≈ 15ms (2.5x mais rapido)

### 10.4 Propriedade de Terminacao

O automato G_credito e acíclico (DAG direcionado acíclico) em condicoes
normais: nao ha ciclos (cada turno de conversa e sempre processado de
q_inicial ate q_final sem loops). O grafo executa no maximo

```
|arestas| = 1 (entrada) + 1 (roteamento) + 1 (celula) + 1 (refinamento)
          = 4 transicoes
```

Portanto, a terminacao e garantida em tempo O(max_cell_time).

---

## 11. Sistemas Multi-Agente

### 11.1 Definicao Formal

**Definicao (Wooldridge & Jennings, 1995):**
Um agente e um sistema computacional situado em um ambiente e capaz de
acao autonoma para atingir objetivos de design.

**Propriedades desejadas:**
- Reatividade: responde a mudancas no ambiente
- Pro-atividade: comportamento orientado a objetivos
- Habilidade social: interacao com outros agentes

**Sistema Multi-Agente (MAS):**

```
MAS = <A, E, I, P> onde:
  A = {a_1, ..., a_n}  conjunto de agentes
  E                     ambiente compartilhado
  I : A x A -> 2^I     protocolo de interacao
  P                     conjunto de objetivos globais
```

### 11.2 Padrao Supervisor + Workers (Hierarquico)

No projeto:

```
Supervisor (CreditAgentGraph):
  - Percebe: mensagem do cliente
  - Decide: qual worker (celula) ativar
  - Nao executa o trabalho substantivo
  - Garante: fluxo deterministico

Workers (Celulas):
  - Recebem: GraphState do supervisor
  - Executam: analise especializada
  - Devolvem: GraphState atualizado
  - Sao independentes entre si (sem comunicacao direta)
```

**Vantagem teorica:** o supervisor e o ponto de controle de consistencia.
Como workers nao se comunicam diretamente, nao ha condicoes de corrida
entre celulas. O unico estado compartilhado e o GraphState, que e
passado por copia entre as transicoes.

### 11.3 Paralelismo Intra-Celula: ThreadPoolExecutor

Dentro de cada celula, os tres analistas rodam em paralelo.
Formalmente, e um padrao de computacao concorrente fork-join:

```
Fork: estado S distribuido para n workers simultaneamente
  w_i = execute(analista_i, S)  para i em {perfil, risco, politicas}

Join: aguarda todos os workers terminarem
  results = [w_i.result() for each w_i]  (barreira de sincronizacao)
```

**Condicao de corrida:** como os analistas SO LEEM o GraphState (nao
modificam), nao ha race condition. O Python GIL (Global Interpreter Lock)
nao e um problema aqui pois os analistas fazem I/O (chamadas de tools)
que libera o GIL.

### 11.4 O Decisor como Funcao de Agregacao

O decisor e uma funcao deterministica:

```
decisor : list[AnalystOpinion] -> GraphState_atualizado
```

Formalmente, implementa uma regra de votacao ponderada com veto:

```
resultado = {
  if risco.verdict == REPROVADO:        BLOQUEADO (veto duro)
  elif politicas.verdict == REPROVADO:  BLOQUEADO (veto duro)
  elif is_borderline(risco.score):      ESCALADO_HUMANO
  else:                                 APROVADO com proposta calculada
}
```

Esta e uma REGRA DE DECISAO NAO-COMPENSATORIA: um veto (REPROVADO em
risco ou politicas) nao pode ser compensado por scores altissimos nos
outros analistas. Difere de regras compensatorias como media ponderada,
onde um analista excepcional pode "salvar" o resultado.

---

## 12. Guardrails como Restricoes de Otimizacao

### 12.1 Framework Teorico

O problema de geracao de respostas pode ser formulado como:

```
max_theta  L(theta) = E[utilidade(resposta)]

sujeito a:
  g_1(resposta) = 0   (disclaimer obrigatorio)
  g_2(resposta) <= 0  (nao-alucinacao numerica)
  g_3(entrada) <= 0   (sem PII)
  g_4(saida) <= 0     (sem toxicidade)
  ...
```

onde g_i sao restricoes duras (hard constraints) que devem ser satisfeitas
independentemente do valor da funcao objetivo.

**Soft constraints vs. Hard constraints:**

- Soft: incorporadas como penalidades na funcao objetivo (Lagrangiano)
  L(theta) + lambda * g(resposta)
  Podem ser violadas se a penalidade for pequena.

- Hard: nunca podem ser violadas. Implementadas em codigo deterministico.

No projeto, TODOS os guardrails sao hard constraints implementadas em
Python puro — nunca delegadas ao LLM (que e probabilistico e pode
violer restricoes).

### 12.2 Nao-Alucinacao Numerica: Analise Formal

O guardrail `numeric_non_hallucination` implementa a seguinte invariante:

```
Para toda resposta r gerada pelo sistema:
  numeros_em_texto(r) ⊆ numeros_autorizados(output_estruturado)
```

Onde:
- numeros_em_texto(r): conjunto de strings numericas extraidas do texto via regex
- numeros_autorizados: conjunto derivado deterministicamente de CreditProposal
  ou list[RenegotiationOption]

**Por que isso e necessario:**
LLMs podem "alucinar" numeros plausíveis. Exemplo:
- Proposta real: limite R$ 15.120,00, taxa 1,89% a.m., 24 meses
- LLM alucina: "com taxa de 1,5% e parcela de R$ 700" (numeros plausíveis
  mas errados)

O guardrail bloqueia qualquer numero no texto que nao conste no conjunto
autorizado, revertendo para o texto base estruturado (que usa f-strings
com valores diretos do objeto Pydantic).

**Analise de falsos positivos:**
O guardrail usa formas canonicas (remove separadores, extrai apenas digitos):

```python
def _number_forms(valor: float) -> set[str]:
    formas = set()
    for fmt in (f"{valor:.0f}", f"{valor:.1f}", f"{valor:.2f}"):
        digitos = re.sub(r"[^\d]", "", fmt)
        if digitos:
            formas.add(digitos)
    return formas
```

Para taxa = 1.89: gera {"2", "19", "189"} (representacoes de 1.9 e 1.89).
Isso garante que "1,89%" e "1.89%" e "1.9%" sejam todos aceitos.

### 12.3 Rate Limiting: Janela Deslizante

O rate limiter implementa uma janela deslizante (sliding window):

```
allow(session_id):
  agora = time.monotonic()
  janela = [t para t em hits[session_id] se agora - t < window_s]
  if len(janela) >= max_calls:
    return BLOQUEADO
  janela.append(agora)
  hits[session_id] = janela
  return PERMITIDO
```

Complexidade: O(max_calls) por chamada (compressao da janela).

**Alternativas mais eficientes:**
- Token bucket: O(1) por chamada (acumula tokens, drena por chamada)
- Leaky bucket: O(1), taxa de saida constante
- Counter fixo: O(1) mas pode permitir burst no limite da janela

---

## 13. Amortizacao Price: Derivacao Completa

### 13.1 Valor Presente e Valor Futuro

**Valor Futuro:** um capital C aplicado a taxa i por n periodos:

```
VF = C * (1 + i)^n
```

**Valor Presente:** quanto vale hoje um valor VF disponível em n periodos:

```
VP = VF / (1 + i)^n = VF * (1 + i)^{-n}
```

O fator (1 + i)^{-n} e o fator de desconto — quanto 1 unidade monetaria
futura vale hoje.

### 13.2 Deducao da Formula Price (Tabela Franceza)

Considere um emprestimo de saldo S dividido em n parcelas iguais de
valor P, com taxa mensal i.

**O emprestimo e uma anualidade:** serie de n pagamentos iguais.
O valor presente da serie deve ser igual ao saldo S:

```
S = P/(1+i)^1 + P/(1+i)^2 + ... + P/(1+i)^n
  = P * sum_{k=1}^{n} (1+i)^{-k}
  = P * sum_{k=1}^{n} r^k    onde r = 1/(1+i)
```

A sum e uma serie geometrica finita:

```
sum_{k=1}^{n} r^k = r * (1 - r^n) / (1 - r)
                  = r * (1 - r^n) / (1 - r)
```

Substituindo r = 1/(1+i) e 1 - r = i/(1+i):

```
sum_{k=1}^{n} (1+i)^{-k} = [1/(1+i)] * [1 - (1+i)^{-n}] / [i/(1+i)]
                          = [1 - (1+i)^{-n}] / i
```

Portanto:

```
S = P * [1 - (1+i)^{-n}] / i

Resolvendo para P:

P = S * i / [1 - (1+i)^{-n}]     <- FORMULA PRICE
```

**Implementacao no projeto (agents/renegociacao/cell.py):**

```python
def _parcela_price(saldo, taxa_mensal, n):
    if taxa_mensal <= 0:
        return round(saldo / n, 2)  # sem juros: amortizacao linear
    fator = (1 + taxa_mensal) ** (-n)
    return round(saldo * taxa_mensal / (1 - fator), 2)
```

### 13.3 Propriedades da Tabela Price

**Parcela constante:** P e fixo para todo k ∈ {1,...,n}.

**Decomposicao de cada parcela:**

```
Juros do periodo k:  J_k = S_{k-1} * i
Amortizacao:         A_k = P - J_k = P - S_{k-1} * i
Saldo apos k:        S_k = S_{k-1} - A_k = S_{k-1} * (1+i) - P
```

**Saldo residual apos m pagamentos:**

```
S_m = P * [1 - (1+i)^{-(n-m)}] / i
    = S * (1+i)^m - P * [(1+i)^m - 1] / i
```

**Verificacao:** S_n = 0 (emprestimo totalmente liquidado apos n parcelas).

### 13.4 Impacto do Desconto na Renegociacao

Para desconto d aplicado ao saldo:

```
S' = S * (1 - d)     (saldo com desconto)
P' = S' * i / (1 - (1+i)^{-n})
   = S * (1-d) * i / (1 - (1+i)^{-n})
   = (1-d) * P     (parcela proporcional ao desconto)
```

O desconto reduz a parcela proporcionalmente — relacao linear simples.

**Custo Total da Operacao com Desconto:**

```
Custo_total = n * P' = n * S * (1-d) * i / (1 - (1+i)^{-n})
```

O cliente paga mais do que o saldo com desconto S' (custo dos juros).
O total pago e n * P', sempre maior que S' para i > 0.

---

## 14. Metricas de Avaliacao

### 14.1 Metricas de Roteamento

**Precisao de roteamento:**

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

Para multi-classe (3 intents): usa macro-F1 ou weighted-F1:

```
Macro-F1 = (1/3) * sum_{c in {concessao, reneg, bancario}} F1_c

Weighted-F1 = sum_c (support_c / N) * F1_c
```

**Matriz de confusao** para 3 classes:

```
           concessao  reneg  bancario
concessao    TP_c      FN_r   FN_b
reneg        FP_c      TP_r   FN_b
bancario     FP_c      FP_r   TP_b
```

O FakeLLM atinge precisao de roteamento 100% no golden set porque usa
regras deterministicas de keywords. LLMs reais ficam em 90-95% em
producao para intents bem definidos.

### 14.2 Metricas de Geracao de Texto

**ROUGE-N:** sobreposicao de N-gramas entre referencia r e hipotese h:

```
ROUGE-N = |ngrams(r) ∩ ngrams(h)| / |ngrams(r)|
```

e uma metrica orientada a recall. ROUGE-L usa a maior subsequencia comum.

**BERTScore (Zhang et al., 2019):**
Computa similaridade entre embeddings BERT de palavras de referencia e
hipotese. Mais correlacionado com julgamento humano que ROUGE.

```
BERTScore_F1(r, h) = 2 * P_bert * R_bert / (P_bert + R_bert)

R_bert = (1/|r|) * sum_{w in r} max_{w' in h} cos(e_w, e_{w'})
P_bert = (1/|h|) * sum_{w' in h} max_{w in r} cos(e_{w'}, e_w)
```

### 14.3 Metricas de Latencia

**Percentil p95:** valor abaixo do qual estao 95% das latencias.
Mais informativo que a media pois captura o comportamento da "cauda longa."

```
p95 = valor x tal que P(latencia <= x) = 0.95
```

**Calculado sobre n amostras ordenadas l_1 <= l_2 <= ... <= l_n:**

```
p95 ≈ l_{ceil(0.95 * n)}
```

**SLO tipico de producao:** p99 < 2000ms para resposta completa.
O projeto em modo offline (FakeLLM) atinge p95 ≈ 15ms.
Com LLM real (Claude Haiku): p50 ≈ 300ms, p95 ≈ 800ms (TTFT).

### 14.4 Alucinacao Numerica: Taxa de Falha

```
taxa_alucinacao = N_falha / N_total

onde:
N_falha = numero de respostas onde numeric_non_hallucination retornou blocked=True
N_total = numero de respostas com dados numericos estruturados
```

Meta do projeto: taxa_alucinacao = 0% (o fallback para texto base garante isso
em codigo — se o LLM alucinar, o output do decisor e usado diretamente).

---

## 15. Conexoes entre os Componentes

### 15.1 O Pipeline como Composicao de Funcoes

O ecossistema inteiro e uma composicao de funcoes puras (ou quase-puras):

```
f_guardrail_entrada : (mensagem, perfil) -> GraphState
f_roteador          : GraphState -> GraphState  (com RoutingDecision)
f_celula            : GraphState -> GraphState  (com proposta/opcoes)
f_refinamento       : GraphState -> GraphState  (com final_answer)
f_guardrail_saida   : GraphState -> GraphState  (ou resposta bloqueada)

Pipeline: (f_guardrail_saida ∘ f_refinamento ∘ f_celula ∘ f_roteador ∘ f_guardrail_entrada)(msg)
```

**Idempotencia:** dada a mesma mensagem e estado inicial, o sistema
com FakeLLM produz SEMPRE a mesma saida. Com RealLLM, a idempotencia
se perde para o texto (LLM e estocastico), mas a ESTRUTURA da decisao
(RoutingDecision, CreditProposal) permanece deterministica.

### 15.2 O LLM como Componente Probabilistico em um Sistema Deterministico

A tensao central do projeto: LLMs sao probabilisticos e opacos, mas
decisoes financeiras exigem determinismo e auditabilidade.

**Solucao arquitetural:** o LLM e confinado a tarefas onde erros sao
toleraveis ou verificaveis:

```
Tarefa delegada ao LLM:
  - Classificacao de intent: verificada pelo guardrail de confianca
  - Reescrita empatica: verificada pelo guardrail de nao-alucinacao numerica
  - Resposta a FAQ: ancorada em contexto RAG, verifcavel manualmente

Tarefa NUNCA delegada ao LLM:
  - Decisao de aprovacao/reprovacao: codigo + regras de guardrail
  - Calculo de taxa, parcela, limite: formula Price + hard limits
  - Deteccao de PII: regex deterministica
  - Coerencia de decisao: funcao decision_coherence em Python puro
```

Esta e a "principio da minima delegacao ao LLM" — use o LLM apenas
onde sua capacidade probabilistica adiciona valor que justifica o risco,
e implemente tudo mais em codigo determinístico.

### 15.3 Informacao no Sistema: da Mensagem a Decisao

Rastreando o fluxo de informacao:

```
1. Mensagem (texto livre) -> Tokenizacao -> Embedding (R^d)
   LLM codifica a mensagem em representacao semantica densa

2. Embedding -> Atencao -> RoutingDecision (JSON estruturado)
   LLM "comprime" a semantica em uma das 3 classes + confianca

3. cliente_id -> Tools (banco de dados mock/real) -> Dados numericos
   Fatos: saldo, score, contratos — nao depende de LLM

4. Dados numericos + Politicas (RAG) -> AnalystOpinion (Pydantic)
   Cada analista combina fatos com regras de politica

5. list[AnalystOpinion] -> decision_coherence -> CreditProposal
   Decisao por regra, nao por LLM

6. CreditProposal -> LLM (refinamento) -> Texto empatico
   LLM "humaniza" dados estruturados, sem inventar numeros

7. Texto + Guardrails -> final_answer
   Verificacao final: numeros autorizados, disclaimer, toxicidade
```

A cada etapa, a incerteza e controlada: o que entra como probabilistico
(LLM) sai como estruturado (JSON validado por Pydantic) antes de afetar
a decisao critica.

### 15.4 Hierarquia de Confiana

```
Nivel 1 - Fontes de verdade (confianca maxima):
  mock_db.py / sistemas reais (CBS, CRM, SICC, ML platform)
  Fatos numericos: saldo, limite, score, capacidade

Nivel 2 - Logica de negocios em codigo (confianca alta):
  guardrails/rules.py: decision_coherence, enforce_hard_limits
  Formula Price: calculo deterministico

Nivel 3 - RAG (confianca media-alta):
  Politicas vigentes recuperadas da KB
  Verificadas por data de vigencia e versao

Nivel 4 - LLM (confianca media):
  Classificacao de intent (verificada por limiar de confianca)
  Reescrita empatica (verificada por nao-alucinacao numerica)

Nivel 5 - LLM sem verificacao (NUNCA usado para decisoes criticas):
  Inventar numeros, aprovar/reprovar credito, decidir taxas
```

Esta hierarquia codifica explicitamente onde o sistema confia no LLM e
onde confia no codigo — base da auditabilidade regulatoria.

---

## Leitura Complementar

**Transformers:**
- Vaswani et al. (2017). Attention is All You Need.
- Alammar. The Illustrated Transformer. jalammar.github.io

**Recuperacao de Informacao:**
- Robertson & Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond.
- Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA.

**RAG:**
- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
- Gao et al. (2024). Retrieval-Augmented Generation for LLMs: A Survey.

**Sistemas Multi-Agente:**
- Wooldridge & Jennings (1995). Intelligent Agents: Theory and Practice.
- Chase (2023). LangGraph: Building Stateful, Multi-Agent Applications.

**Vector Databases:**
- Malkov & Yashunin (2018). Efficient and Robust ANN Search Using HNSW.
- Johnson et al. (2019). Billion-Scale Similarity Search with GPUs (FAISS).

**Teoria da Informacao:**
- Shannon (1948). A Mathematical Theory of Communication.
- Cover & Thomas (2006). Elements of Information Theory. 2nd ed.

**Financas e Amortizacao:**
- Assaf Neto (2014). Matematica Financeira e suas Aplicacoes. 13a ed.
