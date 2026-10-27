"""End-to-end runs over many requests, cached to disk (data/runs/), so a re-run never repeats LLM calls.

Each line of data/runs/<name>.jsonl is one request: {qid, picks, seconds, quotes_ok, final, timings}.
Hit@3 = is the gold item among the 3 picks.
"""
import json
import uuid

import numpy as np

from . import config


def runs_path(name: str):
    return config.DATA_DIR / "runs" / f"{name}.jsonl"


async def run_cached(name: str, run_one, qids: list, requests: list, progress: bool = True) -> dict:
    """Run `await run_one(request) -> dict(picks, seconds, ...)` for every request not yet cached under `name`."""
    path = runs_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            done[rec["qid"]] = rec
    todo = [(q, r) for q, r in zip(qids, requests) if q not in done]
    with path.open("a") as f:
        for n, (qid, request) in enumerate(todo, 1):
            out = await run_one(request)
            rec = {"qid": qid, "picks": out["picks"], "seconds": round(out["seconds"], 2),
                   "quotes_ok": out.get("quotes_ok"), "final": out.get("final", ""),
                   "timings": {k: round(v, 3) for k, v in out.get("timings", {}).items()}}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[qid] = rec
            if progress and (n % 10 == 0 or n == len(todo)):
                print(f"  {name}: {n}/{len(todo)} new requests", flush=True)
    return {q: done[q] for q in qids}


def system_runner(graph):
    """Adapter: one request through the multi-agent graph -> a record for run_cached."""
    from .graph import run_turn

    async def run_one(request: str) -> dict:
        # a fresh conversation AND a fresh shopper per request: nothing remembered leaks between requests
        shopper = f"eval-{uuid.uuid4().hex[:8]}"
        state = await run_turn(graph, request, {"configurable": {"user_id": shopper}}, verbose=False)
        picks = state.get("picks", [])
        return {"picks": [p["item_id"] for p in picks], "seconds": state["seconds"],
                "quotes_ok": sum(p["quote_ok"] for p in picks), "final": state.get("final", ""),
                "timings": state.get("timings", {})}
    return run_one


def react_runner(agent):
    from .graph import run_react

    async def run_one(request: str) -> dict:
        out = await run_react(agent, request, verbose=False)
        return {"picks": out["picks"], "seconds": out["seconds"], "final": out["final"]}
    return run_one


def summarise(records: dict, golds: dict) -> dict:
    """Hit@3, how often 3 products were named, quotes verified, and seconds per request."""
    recs = list(records.values())
    hit = np.mean([golds[r["qid"]] in r["picks"][:3] for r in recs])
    quotes = [r["quotes_ok"] for r in recs if r.get("quotes_ok") is not None]
    return {"n": len(recs), "Hit@3": hit, "3 picks named": np.mean([len(r["picks"]) >= 3 for r in recs]),
            "quotes verified": (sum(quotes) / (3 * len(quotes))) if quotes else np.nan,
            "s / request": np.mean([r["seconds"] for r in recs]),
            "median s": np.median([r["seconds"] for r in recs])}
