"""Phase 2: hybrid retrieval over the catalogue — the Module 3 lab pipeline, scaled to 32k products.

Adapted from Module 4 · Agentic_app (relay_ops/rag.py), which packages Module 3 · RAG_Pipeline_Lab:

    product text  -> one document per product   (or 300/50 chunks with a title header, for comparison)
        -> dense (FAISS + an embedder: BLaIR for the system, bge-small in the lab baseline)  +  sparse (BM25)
        -> hybrid fusion (Reciprocal Rank Fusion)
        -> cross-encoder re-rank of the hybrid shortlist

Changes for scale: queries are embedded and re-ranked in batches, document embeddings are cached in
data/, and FAISS is used directly (inner product on normalised vectors = cosine similarity). Every
ranking is a list of catalogue row numbers, best first, so the evaluation and the search tool share one
code path. A `category` restricts a search to one catalogue category.
"""
import hashlib
import re
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config

# Query instructions from each model card; documents get none
QUERY_PREFIX = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "Qwen/Qwen3-Embedding-0.6B": "Instruct: Given a shopper's description of what they need, retrieve the "
                                 "product that satisfies it\nQuery:",
}
# Models without a sentence-transformers config, built from their model card's recipe
CLS_POOLING = {"hyp1231/blair-roberta-base"}       # BLaIR: first-token embedding, L2-normalised
METHODS = ("bm25", "dense", "hybrid", "hybrid+rerank", "dense+rerank")


def tokenize(text: str) -> list:
    return re.findall(r"[a-z0-9]+", text.lower())


def chunk_catalogue(catalogue: pd.DataFrame) -> tuple:
    """Module 3 chunking: 300/50 splits, each prefixed with its product title (contextual chunk header)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    texts, rows = [], []
    for row, (title, text) in enumerate(zip(catalogue["title"], catalogue["text"])):
        for chunk in splitter.split_text(text) or [text]:
            texts.append(chunk if chunk.startswith(title) else f"{title}\n{chunk}")
            rows.append(row)
    return texts, np.array(rows)


@lru_cache(maxsize=2)
def get_embedder(name: str = config.EMBED_MODEL):
    from sentence_transformers import SentenceTransformer
    if name in CLS_POOLING:
        from sentence_transformers.sentence_transformer.modules import Normalize, Pooling, Transformer
        encoder = Transformer(name, max_seq_length=512)
        pooling = Pooling(encoder.get_embedding_dimension(), pooling_mode="cls")
        return SentenceTransformer(modules=[encoder, pooling, Normalize()])
    model = SentenceTransformer(name)
    model.max_seq_length = min(model.max_seq_length or 512, 512)    # product text beyond 512 tokens is tail
    return model


@lru_cache(maxsize=2)
def get_reranker(name: str = config.RERANK_MODEL):
    import torch
    from sentence_transformers import CrossEncoder
    # bfloat16 weights run slowly on many CPUs; float32 there, the checkpoint's dtype on GPU
    kwargs = {} if torch.cuda.is_available() else {"model_kwargs": {"dtype": torch.float32}}
    return CrossEncoder(name, max_length=512, **kwargs)


class BM25:
    """BM25Okapi exactly as in rank_bm25 (the Module 3 lab's library), vectorised with a sparse matrix.

    rank_bm25 loops over every document in Python for each query word (~0.17 s per query on 32k
    products); one sparse matrix product gives identical scores ~100x faster. Checked in the notebook.
    """

    def __init__(self, corpus: list, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        from scipy.sparse import csr_matrix
        self.vocab = {}
        rows, cols, tf = [], [], []
        for d, tokens in enumerate(corpus):
            counts = {}
            for t in tokens:
                i = self.vocab.setdefault(t, len(self.vocab))
                counts[i] = counts.get(i, 0) + 1
            rows += [d] * len(counts)
            cols += counts.keys()
            tf += counts.values()
        n, tf = len(corpus), np.array(tf, dtype="float64")
        doc_len = np.array([len(t) for t in corpus], dtype="float64")
        df = np.bincount(cols, minlength=len(self.vocab))
        idf = np.log(n - df + 0.5) - np.log(df + 0.5)
        idf[idf < 0] = epsilon * idf.mean()                  # rank_bm25's floor for very common words
        norm = k1 * (1 - b + b * doc_len / doc_len.mean())
        weights = idf[cols] * tf * (k1 + 1) / (tf + norm[rows])
        self.matrix = csr_matrix((weights, (rows, cols)), shape=(n, len(self.vocab))).tocsc()

    def get_scores(self, tokens: list) -> np.ndarray:
        counts = {}
        for t in tokens:                                     # repeated query words count again, as in rank_bm25
            if t in self.vocab:
                counts[self.vocab[t]] = counts.get(self.vocab[t], 0) + 1
        if not counts:
            return np.zeros(self.matrix.shape[0])
        cols = list(counts)
        return self.matrix[:, cols] @ np.array([counts[c] for c in cols], dtype="float64")


def rrf(rankings: list, depth: int = config.DEPTH, k_const: int = config.RRF_K, weights: tuple = None) -> list:
    """Reciprocal Rank Fusion: combine rankings by position, not raw score (optionally weighted)."""
    weights = weights or (1.0,) * len(rankings)
    scores = {}
    for ranking, w in zip(rankings, weights):
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0) + w / (k_const + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)[:depth]


class Retriever:
    """One index over the catalogue: BM25 + FAISS, fused with RRF, optionally re-ranked."""

    def __init__(self, catalogue: pd.DataFrame, chunked: bool = False, embed_model: str = config.EMBED_MODEL,
                 query_prefix: bool = True, rerank_model: str = config.RERANK_MODEL,
                 rerank_top_n: int = config.RERANK_TOP_N):
        import faiss
        self.catalogue = catalogue.reset_index(drop=True)
        if chunked:
            self.texts, self.doc_item = chunk_catalogue(self.catalogue)
        else:
            self.texts, self.doc_item = list(self.catalogue["text"]), np.arange(len(self.catalogue))
        self.chunked = chunked
        self.embed_model = embed_model
        self.rerank_model = rerank_model
        self.rerank_top_n = rerank_top_n
        self.query_prefix = QUERY_PREFIX.get(embed_model, "") if query_prefix else ""
        doc_category = self.catalogue["category"].to_numpy()[self.doc_item]
        self.category_docs = {c: np.flatnonzero(doc_category == c) for c in np.unique(doc_category)}

        self.bm25 = BM25([tokenize(t) for t in self.texts])
        vectors = self._doc_vectors()
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self._sparse_cache = {}

    def _doc_vectors(self) -> np.ndarray:
        digest = hashlib.sha256("\x00".join(self.texts).encode()).hexdigest()[:12]
        slug = self.embed_model.split("/")[-1]
        path = config.DATA_DIR / f"emb_{slug}_{'chunks' if self.chunked else 'items'}_{digest}.npy"
        if path.exists():
            return np.load(path)
        vectors = get_embedder(self.embed_model).encode(
            self.texts, batch_size=128, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
        return vectors

    # ------------------------------------------------------------ document level -> product level
    def _doc_depth(self, depth: int) -> int:
        return depth * 4 if self.chunked else depth      # chunks of one product collapse into one result

    def _to_items(self, doc_ids, depth: int) -> list:
        """Collapse a document ranking to products, best chunk first (= max score per product)."""
        return list(dict.fromkeys(int(self.doc_item[d]) for d in doc_ids if d >= 0))[:depth]

    # ------------------------------------------------------------ the four retrievers
    def sparse(self, queries: list, depth: int = config.DEPTH, categories: list = None) -> list:
        categories = categories or [None] * len(queries)
        out = []
        for query, category in zip(queries, categories):
            key = (query, category, depth)
            if key not in self._sparse_cache:
                scores = self.bm25.get_scores(tokenize(query))
                if category is not None:
                    keep = self.category_docs.get(category, np.array([], dtype=int))
                    masked = np.full_like(scores, -np.inf)
                    masked[keep] = scores[keep]
                    scores = masked
                doc_depth = self._doc_depth(depth)
                top = np.argpartition(-scores, min(doc_depth, len(scores) - 1))[:doc_depth]
                top = top[np.argsort(-scores[top])]
                self._sparse_cache[key] = self._to_items(top[np.isfinite(scores[top])], depth)
            out.append(self._sparse_cache[key])
        return out

    def dense(self, queries: list, depth: int = config.DEPTH, categories: list = None) -> list:
        import faiss
        categories = categories or [None] * len(queries)
        vectors = get_embedder(self.embed_model).encode(
            [self.query_prefix + q for q in queries], batch_size=64, normalize_embeddings=True,
            convert_to_numpy=True).astype("float32")
        out = [None] * len(queries)
        for category in set(categories):                 # one FAISS call per category filter
            rows = [i for i, c in enumerate(categories) if c == category]
            params = None
            if category is not None:
                keep = self.category_docs.get(category, np.array([], dtype="int64"))
                params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(keep.astype("int64")))
            _, ids = self.index.search(vectors[rows], self._doc_depth(depth), params=params)
            for i, doc_ids in zip(rows, ids):
                out[i] = self._to_items(doc_ids, depth)
        return out

    def hybrid(self, queries: list, depth: int = config.DEPTH, categories: list = None,
               k_const: int = config.RRF_K, weights: tuple = config.RRF_WEIGHTS) -> list:
        """RRF of BM25 and dense; weights = (bm25, dense)."""
        return [rrf(pair, depth, k_const, weights) for pair in zip(self.sparse(queries, depth, categories),
                                                                   self.dense(queries, depth, categories))]

    def rerank(self, queries: list, rankings: list, top_n: int = None) -> list:
        """Cross-encoder re-scores each ranking's top_n; the rest keep their hybrid order behind them."""
        top_n = top_n or self.rerank_top_n
        texts = self.catalogue["text"].str.slice(0, 2000).to_numpy()     # the model truncates at 512 tokens anyway
        pairs = [(q, texts[i]) for q, ranking in zip(queries, rankings) for i in ranking[:top_n]]
        scores = get_reranker(self.rerank_model).predict(pairs, batch_size=64, show_progress_bar=False) if pairs else []
        out, start = [], 0
        for ranking in rankings:
            head = ranking[:top_n]
            s = scores[start:start + len(head)]
            start += len(head)
            out.append([head[i] for i in np.argsort(-np.asarray(s), kind="stable")] + ranking[top_n:])
        return out

    def rank(self, queries: list, method: str = "hybrid+rerank", depth: int = config.DEPTH,
             categories: list = None, rerank_queries: list = None, top_n: int = None,
             weights: tuple = config.RRF_WEIGHTS) -> list:
        """Rankings (catalogue rows, best first) for a batch of queries. rerank_queries default to queries."""
        if method == "bm25":
            return self.sparse(queries, depth, categories)
        if method == "dense":
            return self.dense(queries, depth, categories)
        if method == "dense+rerank":
            return self.rerank(rerank_queries or queries, self.dense(queries, depth, categories), top_n)
        fused = self.hybrid(queries, depth, categories, weights=weights)
        if method == "hybrid":
            return fused
        if method == "hybrid+rerank":
            return self.rerank(rerank_queries or queries, fused, top_n)
        raise ValueError(f"method must be one of {METHODS}")

    # ------------------------------------------------------------ for the search tool
    def search(self, query: str, category: str = None, k: int = 10, method: str = "hybrid+rerank") -> pd.DataFrame:
        """Top-k products for one query, with exactly the settings that are evaluated."""
        rows = self.rank([query], method, depth=max(k, config.DEPTH), categories=[category])[0][:k]
        return self.catalogue.iloc[rows].assign(rank=range(1, len(rows) + 1))

    def item_ids(self, rankings: list) -> list:
        ids = self.catalogue["item_id"].to_numpy()
        return [[ids[i] for i in ranking] for ranking in rankings]
