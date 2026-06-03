"""
Base de Conhecimento (KB) com RAG.
==================================================================

O QUE É RAG
----------------------------------------
RAG = Retrieval-Augmented Generation (Geração Aumentada por Recuperação).
Um LLM sozinho "sabe" apenas o que viu no treino e pode alucinar. Em RAG, antes
de responder, o sistema BUSCA trechos relevantes de uma base de documentos e os
entrega ao modelo como contexto. Assim a resposta fica ancorada em fontes reais
e versionadas — crucial num banco, onde políticas mudam e normas são revogadas.

O PIPELINE DE RAG, PASSO A PASSO
--------------------------------
1. CHUNKING (fatiamento): quebrar documentos em pedaços ("chunks"). Aqui usamos
   chunking SEMÂNTICO: respeitamos seções e cláusulas em vez de cortar a cada N
   caracteres, para não partir uma regra no meio.

2. EMBEDDINGS: transformar cada chunk em um vetor de números que captura seu
   significado. Textos parecidos viram vetores próximos. (Em produção:
   text-embedding-3-large ou equivalente. Aqui, um embedding determinístico de
   "saco de palavras" para rodar offline.)

3. ARMAZENAMENTO VETORIAL: guardar os vetores num vector database (Pinecone,
   Azure AI Search, pgvector). Aqui, um índice em memória.

4. RECUPERAÇÃO HÍBRIDA: combinar duas buscas:
     - BM25 (lexical): boa para termos exatos, números de norma, siglas.
     - Densa (vetorial): boa para significado/sinônimos.
   Juntar as duas captura o melhor dos dois mundos.

5. RE-RANKING: pegar o top-20 da recuperação e reordenar com um modelo mais
   caro (cross-encoder) para ficar com o top-5 de maior qualidade.

6. FILTRO POR VIGÊNCIA: descartar políticas revogadas. SEMPRE filtramos por
   data de vigência para nunca citar norma fora de validade.

DEFINIÇÃO — "metadados": dados sobre o dado. Cada chunk carrega versão da
política, data de vigência, produto e segmento. São obrigatórios e usados nos
filtros de recuperação.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date


# ---------------------------------------------------------------------------
# Estruturas básicas
# ---------------------------------------------------------------------------
@dataclass
class ChunkMetadata:
    """Metadados obrigatórios de cada chunk (exigência do projeto)."""

    versao_politica: str
    vigencia_inicio: date
    vigencia_fim: date | None  # None = vigente indefinidamente
    produto: str
    segmento: str

    def vigente_em(self, quando: date) -> bool:
        """Diz se o chunk está vigente na data informada."""
        if quando < self.vigencia_inicio:
            return False
        if self.vigencia_fim is not None and quando > self.vigencia_fim:
            return False
        return True


@dataclass
class Chunk:
    """Um pedaço de documento com seu texto e metadados."""

    chunk_id: str
    texto: str
    metadata: ChunkMetadata
    _tokens: list[str] = field(default_factory=list, repr=False)
    _vector: Counter = field(default_factory=Counter, repr=False)

    def __post_init__(self) -> None:
        self._tokens = _tokenize(self.texto)
        self._vector = Counter(self._tokens)


# ---------------------------------------------------------------------------
# Utilidades de texto
# ---------------------------------------------------------------------------
def _tokenize(texto: str) -> list[str]:
    """Tokenização simples: minúsculas + apenas letras/dígitos.

    DEFINIÇÃO — "token": unidade mínima de texto (aqui, uma palavra normalizada).
    """
    return re.findall(r"[a-zà-ú0-9]+", texto.lower())


def semantic_chunk(texto: str, *, max_chars: int = 600) -> list[str]:
    """Chunking semântico: quebra por seções/cláusulas, não no meio de frases.

    Estratégia:
      1. Separa por linhas em branco (parágrafos/seções).
      2. Junta parágrafos pequenos até `max_chars` para evitar fragmentos.
    Isso respeita a estrutura do documento (seções e cláusulas).
    """
    paragrafos = [p.strip() for p in re.split(r"\n\s*\n", texto) if p.strip()]
    chunks: list[str] = []
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


# ---------------------------------------------------------------------------
# Recuperação: BM25 (lexical) + densa (vetorial) + fusão + re-rank
# ---------------------------------------------------------------------------
def _cosine(a: Counter, b: Counter) -> float:
    """Similaridade do cosseno entre dois 'sacos de palavras'.

    Aproxima a busca densa de forma determinística e offline. Em produção,
    troque por embeddings reais (text-embedding-3-large) + ANN do vector DB.
    """
    if not a or not b:
        return 0.0
    comuns = set(a) & set(b)
    num = sum(a[t] * b[t] for t in comuns)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return num / (na * nb) if na and nb else 0.0


class KnowledgeBase:
    """Uma KB por domínio (FAQ, políticas de concessão, etc.).

    Mantém os chunks, faz a recuperação híbrida e aplica o filtro de vigência.
    Cada domínio tem sua própria instância — bases separadas, como exigido.
    """

    # parâmetros do BM25 (valores clássicos da literatura)
    _K1 = 1.5
    _B = 0.75

    def __init__(self, dominio: str) -> None:
        self.dominio = dominio
        self.chunks: list[Chunk] = []
        self._avg_len = 0.0
        self._df: Counter = Counter()  # document frequency por termo

    def add_document(self, texto: str, metadata: ChunkMetadata, *, prefixo: str) -> None:
        """Fatia um documento e indexa cada chunk."""
        for i, pedaco in enumerate(semantic_chunk(texto)):
            chunk = Chunk(chunk_id=f"{prefixo}-{i}", texto=pedaco, metadata=metadata)
            self.chunks.append(chunk)
        self._reindex()

    def _reindex(self) -> None:
        """Recalcula estatísticas usadas pelo BM25."""
        if not self.chunks:
            return
        self._avg_len = sum(len(c._tokens) for c in self.chunks) / len(self.chunks)
        self._df = Counter()
        for c in self.chunks:
            for termo in set(c._tokens):
                self._df[termo] += 1

    def _bm25_score(self, query_tokens: list[str], chunk: Chunk) -> float:
        """Pontuação BM25 de um chunk para a query.

        DEFINIÇÃO — "BM25": função de ranqueamento lexical que premia termos
        raros e penaliza documentos muito longos. Padrão de mercado para busca
        por palavras-chave.
        """
        n = len(self.chunks)
        score = 0.0
        dl = len(chunk._tokens)
        for termo in query_tokens:
            tf = chunk._vector.get(termo, 0)
            if tf == 0:
                continue
            df = self._df.get(termo, 0)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            denom = tf + self._K1 * (1 - self._B + self._B * dl / (self._avg_len or 1))
            score += idf * (tf * (self._K1 + 1)) / denom
        return score

    def retrieve(
        self,
        query: str,
        *,
        hoje: date,
        segmento: str | None = None,
        top_k: int = 5,
        candidatos: int = 20,
    ) -> list[Chunk]:
        """Recuperação híbrida com filtro de vigência e re-ranking.

        Fluxo:
          1. Filtra apenas chunks VIGENTES na data `hoje` (e do segmento, se dado).
          2. Calcula score BM25 (lexical) e cosseno (denso) para cada candidato.
          3. Funde os dois scores normalizados (recuperação híbrida).
          4. Pega o top-`candidatos` (top-20) e re-ranqueia (top-`top_k` = top-5).
        """
        query_tokens = _tokenize(query)
        query_vec = Counter(query_tokens)

        # 1. filtro de vigência (+ segmento opcional)
        elegiveis = [
            c
            for c in self.chunks
            if c.metadata.vigente_em(hoje)
            and (segmento is None or c.metadata.segmento in (segmento, "todos"))
        ]
        if not elegiveis:
            return []

        # 2. scores brutos
        bm25 = {c.chunk_id: self._bm25_score(query_tokens, c) for c in elegiveis}
        dense = {c.chunk_id: _cosine(query_vec, c._vector) for c in elegiveis}

        # 3. fusão por min-max normalizado (recuperação híbrida)
        def _norm(d: dict[str, float]) -> dict[str, float]:
            vals = list(d.values())
            lo, hi = min(vals), max(vals)
            if hi - lo < 1e-9:
                return {k: 0.0 for k in d}
            return {k: (v - lo) / (hi - lo) for k, v in d.items()}

        bm25_n, dense_n = _norm(bm25), _norm(dense)
        fundido = {
            c.chunk_id: 0.5 * bm25_n[c.chunk_id] + 0.5 * dense_n[c.chunk_id]
            for c in elegiveis
        }

        # top-20 candidatos
        ordenados = sorted(elegiveis, key=lambda c: fundido[c.chunk_id], reverse=True)
        candidatos_top = ordenados[:candidatos]

        # 4. re-ranking (cross-encoder simulado): combina sobreposição de termos
        #    com o score fundido para refinar o top-5. Em produção, troque por um
        #    cross-encoder real (ex.: bge-reranker) do top-20 para top-5.
        def _rerank_score(c: Chunk) -> float:
            overlap = len(set(query_tokens) & set(c._tokens)) / (len(set(query_tokens)) or 1)
            return 0.6 * overlap + 0.4 * fundido[c.chunk_id]

        rerankeados = sorted(candidatos_top, key=_rerank_score, reverse=True)
        return rerankeados[:top_k]
