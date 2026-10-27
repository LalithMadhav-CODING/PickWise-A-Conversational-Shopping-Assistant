"""Phase 7: faithfulness of the recommendations — LLM-as-judge (the brief's "every claim appears in the
product text"), plus a sheet for the team's 30 human checks.

The judge reads an assistant's answer and the texts of the products it recommended, and decides for each
product whether every claim the answer makes ABOUT that product is backed by its text. The same judge and
prompt score the multi-agent system and the single-ReAct baseline. It runs with reasoning on (judging needs
it; the output is schema-constrained, so the verdict stays clean) and is cached in data/runs/judge_*.jsonl.

A 4B judge grading a 4B model can be lenient with its own family's mistakes, so 30 recommendations go to the
team for a blind human label (human_check_sheet); agreement with the judge is reported once filled in.
"""
import json
import re
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from . import config
from .llm import get_llm

VERDICTS = ("supported", "partly supported", "unsupported")


class ProductVerdict(BaseModel):
    product: Literal[1, 2, 3] = Field(description="the product's number, as in the product texts")
    verdict: Literal[VERDICTS] = Field(description="supported: every claim about this product is in its text; "
                                                    "partly supported: some claims are, some are not; "
                                                    "unsupported: the main claim is not in its text")
    unsupported_claim: str = Field(default="", description="the claim not backed by the text, if any")


class Judgement(BaseModel):
    verdicts: list[ProductVerdict] = Field(description="one verdict per recommended product", min_length=1, max_length=3)


JUDGE_PROMPT = """You check a shopping assistant's answer for faithfulness. The assistant recommended the numbered products below.
For EACH product, decide whether every claim the answer makes about THAT product (its features, materials, size, quality, safety, suitability) is supported by that product's text. Statements about what the shopper wants are not claims about the product. Quoted phrases must appear in the text.
- supported: every claim about the product is in its text
- partly supported: some claims are in the text, some are not
- unsupported: the main claim about the product is not in its text"""


def product_block(catalogue: pd.DataFrame, item_ids: list, chars: int = 1500) -> str:
    rows = catalogue.set_index("item_id").loc[item_ids]
    return "\n\n".join(f"[{i}] {r['title']}\n{r['text'][:chars]}" for i, (_, r) in enumerate(rows.iterrows(), 1))


async def judge(answer: str, item_ids: list, catalogue: pd.DataFrame) -> list:
    """Verdicts for the products an answer recommended (in the order given)."""
    if not item_ids:
        return []
    llm = get_llm(reasoning=True).with_structured_output(Judgement)
    out = await llm.ainvoke([("system", JUDGE_PROMPT),
                             ("human", f"Assistant's answer:\n{answer}\n\nProduct texts:\n\n{product_block(catalogue, item_ids)}")])
    by_number = {v.product: v for v in out.verdicts}

    def claim(v):                                       # the model writes "None", "none." or "None. All claims…"
        return "" if re.match(r"\s*(none\b|n/a\b|-|$)", v.unsupported_claim, re.I) else v.unsupported_claim
    return [{"item_id": i, "verdict": by_number[n].verdict if n in by_number else "unsupported",
             "unsupported_claim": claim(by_number[n]) if n in by_number else "(no verdict)"}
            for n, i in enumerate(item_ids, 1)]


async def judge_runs(name: str, records: dict, catalogue: pd.DataFrame, qids: list, progress: bool = True) -> dict:
    """Judge the cached answers of run `name` for the given qids; cached in data/runs/judge_<name>.jsonl."""
    path = config.DATA_DIR / "runs" / f"judge_{name}.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            done[rec["qid"]] = rec
    todo = [q for q in qids if q not in done]
    with path.open("a") as f:
        for n, qid in enumerate(todo, 1):
            rec = records[qid]
            try:
                verdicts = await judge(rec["final"], rec["picks"][:3], catalogue)
            except Exception as e:                       # an unusable judge reply is recorded, not hidden
                verdicts = [{"item_id": i, "verdict": "error", "unsupported_claim": str(e)[:200]} for i in rec["picks"][:3]]
            done[qid] = {"qid": qid, "verdicts": verdicts}
            f.write(json.dumps(done[qid]) + "\n")
            f.flush()
            if progress and (n % 10 == 0 or n == len(todo)):
                print(f"  judge {name}: {n}/{len(todo)} new answers", flush=True)
    return {q: done[q] for q in qids}


def faithfulness(judged: dict) -> dict:
    """Share of recommended products whose claims are supported (and partly supported) by their text."""
    verdicts = [v["verdict"] for rec in judged.values() for v in rec["verdicts"]]
    n = len(verdicts)
    return {"recommendations judged": n,
            "supported": verdicts.count("supported") / n if n else float("nan"),
            "supported or partly": (verdicts.count("supported") + verdicts.count("partly supported")) / n if n else float("nan"),
            "judge errors": verdicts.count("error")}


def human_check_sheet(records: dict, catalogue: pd.DataFrame, queries: dict, n: int = 30, seed: int = config.SEED) -> pd.DataFrame:
    """A blind sheet of n system recommendations (request, product text, reason, quote) for the team to label
    supported / partly supported / unsupported. The judge's verdicts are NOT on the sheet."""
    texts = catalogue.set_index("item_id")
    rows = [{"qid": qid, "item_id": d["item_id"], "request": queries[qid],
             "product": texts.loc[d["item_id"], "title"], "product text": texts.loc[d["item_id"], "text"][:1500],
             "assistant's reason": d["reason"], "assistant's quote": d["quote"]}
            for qid, rec in records.items() for d in rec.get("details", [])]
    sheet = pd.DataFrame(rows).sample(n=min(n, len(rows)), random_state=seed).reset_index(drop=True)
    sheet.insert(0, "row", range(1, len(sheet) + 1))
    sheet["human label (supported / partly supported / unsupported)"] = ""
    sheet["notes"] = ""
    return sheet
