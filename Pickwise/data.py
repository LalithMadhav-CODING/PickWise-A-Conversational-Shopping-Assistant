"""Phase 1: queries, catalogue and the dev/test split (brief §02).

The catalogue is streamed from Hugging Face with load_dataset on the first run, then cached to
data/catalogue.parquet so re-runs skip the 640 MB scan. The cache is keyed on the settings that
produced it, so changing the seed, sample size or distractor rate rebuilds it.
"""
import hashlib
import json
import random

import pandas as pd
from datasets import load_dataset

from . import config


def load_queries(n: int = config.N_QUERIES, seed: int = config.SEED):
    """A fixed, shuffled sample of the Amazon-C4 test queries (the notebook shows this call inline)."""
    return load_dataset(config.DATASET_ID)["test"].shuffle(seed=seed).select(range(n))


def split_title(metadata: str) -> str:
    """The catalogue has one text field, "<title>. <description>"; the title is the first sentence."""
    head, sep, _ = metadata.partition(". ")
    return head if sep else metadata


def _cache_key(gold: set, seed: int, rate: float) -> dict:
    gold_hash = hashlib.sha256("\n".join(sorted(gold)).encode()).hexdigest()[:16]
    return {"seed": seed, "rate": rate, "n_gold": len(gold), "gold_hash": gold_hash}


def build_catalogue(gold: set, seed: int = config.SEED, rate: float = config.DISTRACTOR_RATE,
                    cache=config.CATALOGUE_CACHE) -> pd.DataFrame:
    """Gold items of the sampled queries + random distractors: columns item_id, category, title, text.

    `text` is the full metadata string (title + description) and is the only field ever indexed.
    Nothing from the query file, ori_review included, enters the catalogue.
    """
    key = _cache_key(gold, seed, rate)
    meta = cache.with_suffix(".json")
    if cache.exists() and meta.exists() and json.loads(meta.read_text()) == key:
        return pd.read_parquet(cache)

    pool = load_dataset("json", data_files=config.CATALOGUE_FILE, split="train", streaming=True)
    rng = random.Random(seed)
    rows = [r for r in pool if r["item_id"] in gold or rng.random() < rate]   # same filter as the brief
    df = pd.DataFrame(rows).rename(columns={"metadata": "text"})
    df["title"] = df["text"].map(split_title)
    df = df[["item_id", "category", "title", "text"]].drop_duplicates("item_id").reset_index(drop=True)

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    meta.write_text(json.dumps(key))
    return df


def split_queries(queries, n_dev: int = config.N_DEV, quick: bool = config.QUICK_RUN):
    """First n_dev of the shuffled sample = dev (tuning only), the rest = test (reported once)."""
    dev = queries.select(range(n_dev))
    test = queries.select(range(n_dev, len(queries)))
    if quick:
        test = test.select(range(config.QUICK_N))
    return dev, test


def check_no_leakage(queries, catalogue: pd.DataFrame) -> None:
    """Fail loudly if the answer could leak into the index."""
    assert "ori_review" not in catalogue.columns
    text = dict(zip(catalogue["item_id"], catalogue["text"]))
    leaked = [q["qid"] for q in queries if q["ori_review"].strip() and q["ori_review"].strip() in text.get(q["item_id"], "")]
    assert not leaked, f"ori_review text found in the gold item's index text: qids {leaked[:10]}"
