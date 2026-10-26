# Current Project Status

_Last updated: 2026-10-26_

Live log of where the Pickwise (P14) build stands. The plan is in `P14_Project_Plan.md`; this file tracks progress against it.

## At a glance

| Phase | Status |
|---|---|
| 0 — Setup & Scaffold | ✅ done 2026-10-26 |
| 1 — Data & Catalogue | ✅ done 2026-10-26 |
| 2 — Retrieval (RAG) | ⚪ next |
| 3 — Need Analyst | ⚪ not started |
| 4 — MCP Server + Multi-Agent | ⚪ not started |
| 5 — Memory + Human-in-the-Loop | ⚪ not started |
| 6 — Demo Cell | ⚪ not started |
| 7 — Evaluation | ⚪ not started |
| 8 — Stretch Goals | ⚪ not started |

**Milestone M1 (Data + Retrieval):** data half done; retrieval is Phase 2.

## Files so far

| File | What it is |
|---|---|
| `P14_Shopping_Assistant.ipynb` | The deliverable notebook — §1 Problem, §2 Setup, §3 Data filled and executed; §4–10 placeholders |
| `Pickwise/config.py` | All settings (model, data sizes, seed, QUICK_RUN, RISKY_TOOLS) |
| `Pickwise/data.py` | `load_queries`, `build_catalogue` (cached), `split_queries`, `check_no_leakage` |
| `Pickwise/llm.py` | `get_llm(model, reasoning)`, `reasoning_trace(msg)` |
| `requirements.txt` | Package list for the notebook's setup cell (local + Colab) |
| `data/catalogue.parquet` (+ `.json` key) | Cached catalogue — gitignored, rebuilt automatically |

## Decisions made (with the team)

| Date | Decision | Why |
|---|---|---|
| 2026-10-26 | **Project name: Pickwise**; package folder `Pickwise/` (capital P, team choice) | Short, memorable |
| 2026-10-26 | **Code layout: notebook + small `Pickwise/` package** (+ a separate MCP server script later) | MCP server must run as its own process; Streamlit stretch must import the code. Same pattern as Module 4 · Agentic_app |
| 2026-10-26 | **Catalogue: spec default** — gold items of 1,000 queries + ~3% distractors, seed 42 | Matches the brief exactly |
| 2026-10-26 | **Index text = full `metadata` string**; `title` parsed as text before the first `". "` (display only) | The jsonl has one `metadata` field (`"Title. Description…"`) — resolves review issue #5 |
| 2026-10-26 | **Cache the filtered catalogue** to `data/catalogue.parquet`, keyed on seed / rate / gold set | First build 163 s; re-runs < 1 s. First run still uses `load_dataset` — resolves #14 |
| 2026-10-26 | **Dev = first 200 of the shuffled sample (tuning only); test = remaining 800 (reported once)** | Resolves #12 |
| 2026-10-26 | **Reasoning mode:** `reasoning=True` for free text + tool calls; `reasoning=False` for schema-constrained calls (need analyst, supervisor router) — confirm on dev in Phase 3 | Off leaks narration into free text/tool calls; structured output stays clean and is ~27× faster (0.6 s vs 16.4 s) |
| 2026-10-26 | **Index one document per item** (no chunking); Module 3 chunking kept as a comparison row | Median item is 32 words; avoids chunk→item merging before scoring — resolves #2 |
| 2026-10-26 | **Keep unanswerable queries** (gold item has empty text) in the test set; report them | Dropping them would flatter the metrics |

## Pending decisions (need team input)

- _None right now._

## Environment

- Dev machine: Linux, Python 3.12.3, NVIDIA RTX 4000 Ada (20 GB), 24 cores, 125 GB RAM
- Ollama 0.33.2 with `qwen3:4b` pulled
- Project venv: `.venv/` (gitignored) — torch 2.14.1+cu130 (CUDA OK), langgraph 1.2.17, langchain-core 1.7.0, langchain-ollama 1.1.0, langchain-mcp-adapters 0.3.2, mcp 1.30.0, datasets 5.1.0, sentence-transformers 6.2.0, faiss-cpu 1.15.1, rank-bm25 0.2.2
- Run the notebook: `.venv/bin/jupyter execute --inplace P14_Shopping_Assistant.ipynb` (≈ 20 s with the cache)
- The notebook must still run on Colab T4 / laptop CPU (brief) — check before submission

## Phase 0 — Setup & Scaffold ✅

- [x] Project venv + full stack installed
- [x] `Pickwise/config.py` — single CONFIG (model, paths, seed, QUICK_RUN, RISKY_TOOLS)
- [x] Verify Ollama + qwen3:4b from Python (tool call, structured output, reasoning on/off) — review issue #8 ✅ resolved
- [x] `Pickwise/llm.py` — `get_llm(model, reasoning)` + `reasoning_trace(msg)`
- [x] `requirements.txt` (setup cell does `%pip install -r requirements.txt`, same on Colab)
- [x] Notebook skeleton with all 10 template sections; §4–10 placeholders cite their phase + lab
- [x] Notebook runs top-to-bottom with no errors (≈ 20 s)

### LLM probe results (2026-10-26, qwen3:4b, RTX 4000 Ada, one request each)

| Call | `reasoning=False` | `reasoning=True` |
|---|---|---|
| Free text answer | ❌ 1,169 chars, "Okay, the user needs…" narration leaks into `content` (3.8 s) | ✅ 318 clean chars; trace (1,191 chars) in `additional_kwargs["reasoning_content"]` (3.3 s) |
| Structured output (`with_structured_output`, need-analyst schema) | ✅ valid, concise JSON in **0.6 s** | ✅ valid but verbose must-haves, **16.4 s** |
| Tool call (`bind_tools`) | ✅ correct call, but ~5k chars of narration also in `content` (11.8 s) | ✅ correct call, empty `content` (12.0 s) |

- The flag is `ChatOllama(reasoning=True)`; the trace is `msg.additional_kwargs["reasoning_content"]` — **not** `think=` / `message.thinking` (those belong to the raw `ollama` client).
- `reasoning=None` (default): model still thinks, LangChain drops the trace — clean answer but no trace to show.

## Phase 1 — Data & Catalogue ✅

- [x] `Pickwise/data.py` — `load_queries`, `build_catalogue` (cached), `split_queries`, `check_no_leakage`
- [x] First catalogue build: **163 s** streaming 1.06M items
- [x] Verified: all gold items present; no `ori_review` leakage; sizes stated in the notebook
- [x] Data section (§3) in the notebook, incl. a data-quality report

### Numbers

| | |
|---|---|
| Queries sampled | 1,000 (seed 42) — mean 43 words, median 36, max 212 |
| Unique gold items | 998 (2 queries share a gold item) |
| Distractors | 31,616 |
| **Catalogue** | **32,614 items**, 30+ categories (Home 5,251 · Clothing 4,445 · Electronics 2,420 · …) |
| Item text length | mean 86 words, **median 32**, 75th pct 96, max 12,237 |
| Dev / test | 200 / 800 (QUICK_RUN: 100) |
| Empty-text items | 99 in the catalogue; **4 gold** (all 4 in test → unanswerable); 15 gold items ≤ 5 words |

## Findings / notes

- Catalogue jsonl schema: `{"item_id", "category", "metadata"}` — `metadata` is one string, title first. Short items often look like `"Her Leaves Remain. ."`.
- Queries file (`test.csv`): `qid, query, item_id, user_id, ori_rating, ori_review`.
- **For Phase 2 (review issue #2):** median item is only 32 words, so most items fit in one 300-char chunk anyway. Indexing **one document per item** (title first, so embedder truncation at 512 tokens keeps the important part) avoids chunk→item score aggregation entirely. **Decided 2026-10-26: one doc per item.**

## Open issues

See *Plan Review — Open Issues* in `P14_Project_Plan.md`. Resolved so far: #2 (decided), #5, #8, #12, #14.

## Change log

- 2026-10-26 — Project named **Pickwise**: package `p14/` → `Pickwise/`, env vars `P14_*` → `PICKWISE_*`, titles updated. Notebook re-run clean after the rename.
- 2026-10-26 — Package folder renamed to `Pickwise/` (capital P); imports and docs updated, notebook re-run clean. README added; first push to GitHub.
