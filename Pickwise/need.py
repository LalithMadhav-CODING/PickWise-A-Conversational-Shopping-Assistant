"""Phase 3: the need analyst — a long, situational request -> a structured catalogue search.

Builds on Module 2 · Structured_Generation_Lab: a Pydantic schema is the contract, and the category
is a Literal so the allowed values are compiled into the generation grammar (Ollama's JSON-schema
`format`), not just validated afterwards.

Replies are cached in data/needs/ (one JSONL file per model + reasoning mode + prompt version), so
re-running the notebook or an evaluation never calls the model twice for the same request.
"""
import hashlib
import json
import time
from typing import Literal

from pydantic import BaseModel, Field

from . import config
from .llm import get_llm

# The catalogue's 31 categories (Amazon-C4 short names) and what each one actually contains
CATEGORY_GUIDE = {
    "Appliances": "large appliances and their parts (fridge, washer, dryer, oven parts)",
    "Arts": "arts, crafts and sewing supplies",
    "Automotive": "car, truck and motorcycle parts and accessories",
    "Baby": "baby products (nursery, feeding, strollers, baby clothing accessories)",
    "Beauty": "a few beauty items (most beauty products are under Care)",
    "Books": "printed books",
    "CDs": "music CDs and vinyl",
    "Care": "beauty and personal care: skin, hair, makeup, nails, shaving, oral care",
    "Cell": "cell phones and accessories: cases, chargers, screen protectors",
    "Clothing": "clothing, shoes, jewelry and watches",
    "Electronics": "electronics: computers, audio, TV, cameras, cables, batteries",
    "Fashion": "a few clothing items (most clothing is under Clothing)",
    "Food": "grocery and gourmet food and drinks",
    "Games": "video games, consoles and gaming accessories",
    "Garden": "patio, lawn and garden: outdoor, grills, lawn mower parts",
    "Gift": "gift cards",
    "Handmade": "handmade items",
    "Health": "a few health items (most are under Household)",
    "Home": "home and kitchen: furniture, bedding, bath, decor, kitchenware, small kitchen appliances",
    "Household": "health and household: vitamins, supplements, medical supplies, cleaning and household supplies",
    "Instruments": "musical instruments and accessories",
    "Kindle": "Kindle e-books",
    "Magazine": "magazine subscriptions",
    "Movies": "movies and TV (DVD, Blu-ray)",
    "Office": "office products: stationery, printers, ink, office furniture",
    "Pet": "pet supplies",
    "Scientific": "industrial and scientific: lab supplies, fasteners, electrical components",
    "Software": "software",
    "Sports": "sports and outdoors: fitness, camping, cycling, fan gear",
    "Tools": "tools and home improvement: power tools, hardware, lighting, plumbing, electrical",
    "Toys": "toys and games, including board games and puzzles",
}
Category = Literal[tuple(CATEGORY_GUIDE)]


class ShoppingNeed(BaseModel):
    product_type: str = Field(description="the kind of product, 1-5 words, as a shop would name it")
    must_haves: list[str] = Field(description="up to 5 short properties the product must have, from the request")
    category: Category = Field(description="the catalogue category the product belongs to")


SYSTEM_PROMPT = """You are the need analyst of a shopping assistant. A shopper describes their situation in their own words. Work out what product they want and turn the request into a catalogue search.

- product_type: the kind of product in 1-5 words, the way a shop would name it (e.g. "bath toy set", "air purifier filter", "trail running shoes").
- must_haves: up to 5 short phrases with the properties the product must have, using only what the shopper said or clearly implied. Leave out feelings and filler ("works great", "I love it").
- category: the catalogue category the product belongs to.

Catalogue categories:
""" + "\n".join(f"- {name}: {desc}" for name, desc in CATEGORY_GUIDE.items())

PROMPT_VERSION = hashlib.sha256((SYSTEM_PROMPT + json.dumps(ShoppingNeed.model_json_schema())).encode()).hexdigest()[:8]


def search_query(need: ShoppingNeed, request: str = None) -> str:
    """The query the searcher sends to the catalogue: the rewrite, optionally followed by the shopper's
    own words. On dev, rewrite + request matched or beat either one alone across the retrievers: the
    rewrite rescues BM25, the full request keeps what dense retrieval needs."""
    rewrite = ", ".join([need.product_type, *need.must_haves])
    return f"{rewrite}. {request}" if request else rewrite


def analyse(request: str, reasoning: bool = False, model: str = config.MODEL) -> ShoppingNeed:
    """One request -> ShoppingNeed. reasoning=False is the default for schema-constrained calls (decided
    2026-10-26): the schema keeps the reply clean either way, and it is ~27x faster."""
    analyst = get_llm(model, reasoning).with_structured_output(ShoppingNeed)
    return analyst.invoke([("system", SYSTEM_PROMPT), ("human", request)])


async def aanalyse(request: str, reasoning: bool = False, model: str = config.MODEL) -> ShoppingNeed:
    """Async analyse, for the agent graph."""
    analyst = get_llm(model, reasoning).with_structured_output(ShoppingNeed)
    return await analyst.ainvoke([("system", SYSTEM_PROMPT), ("human", request)])


def cache_path(reasoning: bool, model: str = config.MODEL):
    return config.DATA_DIR / "needs" / f"{model.replace(':', '-')}_reasoning-{reasoning}_{PROMPT_VERSION}.jsonl"


def analyse_many(qids: list, requests: list, reasoning: bool = False, model: str = config.MODEL,
                 retries: int = 1, progress: bool = True) -> dict:
    """Analyse a batch of requests, resuming from the cache. Returns {qid: record}, where a record is
    {"need": dict | None, "seconds": float, "error": str | None}; need is None if every attempt failed."""
    path = cache_path(reasoning, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            done[rec["qid"]] = rec
    todo = [(q, r) for q, r in zip(qids, requests) if q not in done]
    with path.open("a") as f:
        for n, (qid, request) in enumerate(todo, 1):
            rec = {"qid": qid, "need": None, "seconds": 0.0, "error": None}
            t = time.time()
            for _ in range(retries + 1):
                try:
                    rec["need"], rec["error"] = analyse(request, reasoning, model).model_dump(), None
                    break
                except Exception as e:          # invalid / truncated JSON: retry, then give up on this one
                    rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            rec["seconds"] = round(time.time() - t, 2)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[qid] = rec
            if progress and (n % 50 == 0 or n == len(todo)):
                print(f"  need analyst: {n}/{len(todo)} new requests analysed", flush=True)
    return {q: done[q] for q in qids}


def to_need(record: dict):
    return ShoppingNeed(**record["need"]) if record["need"] else None
