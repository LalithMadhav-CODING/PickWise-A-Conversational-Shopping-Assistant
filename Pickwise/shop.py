"""Phase 4: the catalogue as tools. The SAME functions back the MCP server (mcp_server.py) and the local
@tools (tools.py), so the two transports cannot drift apart: same names, descriptions and results.

Each tool returns JSON text, which travels unchanged over MCP. search_products also reports how long the
search itself took, so the MCP-vs-local comparison can separate retrieval time from transport time.
"""
import json
import time
from functools import lru_cache

import pandas as pd

from . import config
from .retrieval import Retriever


@lru_cache(maxsize=1)
def get_catalogue() -> pd.DataFrame:
    if not config.CATALOGUE_CACHE.exists():
        raise FileNotFoundError(f"{config.CATALOGUE_CACHE} not found: run the notebook's Data section first")
    return pd.read_parquet(config.CATALOGUE_CACHE)


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """The system retriever (BLaIR · hybrid RRF · Qwen3-Reranker), built once per process."""
    return Retriever(get_catalogue())


def product_card(row, snippet_chars: int = config.SNIPPET_CHARS) -> dict:
    text = row["text"]
    return {"item_id": row["item_id"], "title": row["title"] or row["item_id"], "category": row["category"],
            "text": text if len(text) <= snippet_chars else text[:snippet_chars].rsplit(" ", 1)[0] + " …"}


def search_products(query: str, category: str = "", k: int = config.N_CANDIDATES) -> str:
    """Search the product catalogue and return the best-matching products, best first.

    query: what the shopper needs, in words (e.g. "bath toy set, safe for toddlers, easy to clean").
    category: optional; only use it when the shopper named a category (see list_categories).
    k: how many products to return (default 10).
    Returns JSON: {"products": [{item_id, title, category, text}], "seconds": search time}.
    """
    t = time.perf_counter()
    rows = get_retriever().search(query, category=category or None, k=int(k), method=config.RETRIEVAL_METHOD)
    products = [product_card(row) for _, row in rows.iterrows()]
    return json.dumps({"products": products, "seconds": round(time.perf_counter() - t, 4)})


def get_product(item_id: str) -> str:
    """Get one product's full details by its item_id (from search_products).
    Returns JSON: {item_id, title, category, text}, or {"error": ...} if there is no such product."""
    catalogue = get_catalogue()
    match = catalogue[catalogue["item_id"] == item_id]
    if match.empty:
        return json.dumps({"error": f"no product with item_id {item_id!r}"})
    return json.dumps(product_card(match.iloc[0], snippet_chars=10_000))


def list_categories() -> str:
    """List the catalogue's product categories with how many products each has.
    Returns JSON: {category: number of products}."""
    return json.dumps(get_catalogue()["category"].value_counts().to_dict())


TOOL_FUNCTIONS = (search_products, get_product, list_categories)


def warm_up() -> None:
    """Load the catalogue, index and both retrieval models before the first real request."""
    search_products("warm up", k=1)
