"""Retrieval metrics against the single gold item of each query (Module 3 lab: Recall@k, MRR; plus nDCG@k).

With one relevant item per query, nDCG@k reduces to 1 / log2(rank + 1) when the gold item is in the
top k (the ideal DCG is 1), and MRR is averaged over the full ranking depth (100 by default).
"""
import numpy as np


def gold_ranks(rankings: list, golds: list) -> np.ndarray:
    """1-based rank of the gold item in each ranking of item_ids; 0 when it was not retrieved."""
    return np.array([ranking.index(g) + 1 if g in ranking else 0 for ranking, g in zip(rankings, golds)])


def evaluate(retriever, queries: list, golds: list, method: str, return_ranks: bool = False, **rank_kwargs):
    """Rank a batch of queries with one retriever setting and score it against the gold items.
    With return_ranks=True, also return the per-query gold ranks (for paired comparisons)."""
    rankings = retriever.item_ids(retriever.rank(queries, method, **rank_kwargs))
    ranks = gold_ranks(rankings, golds)
    return (retrieval_metrics(ranks), ranks) if return_ranks else retrieval_metrics(ranks)


def retrieval_metrics(ranks: np.ndarray, k: int = 10) -> dict:
    found = ranks > 0
    reciprocal = np.where(found, 1.0 / np.maximum(ranks, 1), 0.0)
    in_top_k = found & (ranks <= k)
    return {
        f"Recall@{k}": in_top_k.mean(),
        "MRR": reciprocal.mean(),
        f"nDCG@{k}": np.where(in_top_k, 1.0 / np.log2(np.maximum(ranks, 1) + 1), 0.0).mean(),
        "Recall@50": (found & (ranks <= 50)).mean(),       # how much a re-ranker of the top 50 can recover
        "Recall@100": found.mean(),
    }


def paired_bootstrap(ranks_a: np.ndarray, ranks_b: np.ndarray, k: int = 10, n_boot: int = 2000, seed: int = 42) -> dict:
    """Difference a - b in Recall@k and MRR on the same queries, with a 95% bootstrap interval
    (queries resampled with replacement). An interval that excludes 0 is a real difference."""
    rng = np.random.default_rng(seed)
    per_query = lambda r: np.stack([(r > 0) & (r <= k), np.where(r > 0, 1.0 / np.maximum(r, 1), 0.0)])
    diff = per_query(ranks_a) - per_query(ranks_b)                      # 2 x n_queries
    boots = diff[:, rng.integers(0, diff.shape[1], (n_boot, diff.shape[1]))].mean(axis=2)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=1)
    return {f"Δ Recall@{k}": diff[0].mean(), f"Recall@{k} 95% CI": (lo[0], hi[0]),
            "Δ MRR": diff[1].mean(), "MRR 95% CI": (lo[1], hi[1])}
