# Current Project Status

_Last updated: 2026-10-27_

Live log of where the Pickwise (P14) build stands. The plan is in `P14_Project_Plan.md`; this file tracks progress against it.

## At a glance

| Phase | Status |
|---|---|
| 0 — Setup & Scaffold | ✅ done 2026-10-26 |
| 1 — Data & Catalogue | ✅ done 2026-10-26 |
| 2 — Retrieval (RAG) | ✅ done 2026-10-26 |
| 3 — Need Analyst | ✅ done 2026-10-26 |
| 4 — MCP Server + Multi-Agent | ✅ done 2026-10-26 (branch `phase-4-mcp-multi-agent`) |
| 5 — Memory + Human-in-the-Loop | ✅ done 2026-10-27 (branch `phase-5-memory-approval`) |
| 6 — Demo Cell | ✅ done 2026-10-27 (branch `phase-6-demo`) |
| 7 — Evaluation | ⚪ next (code drafted: `Pickwise/judge.py`, eval cells) |
| 8 — Stretch Goals | 🟡 web app ✅ 2026-10-27 (branch `phase-8-app`), BLaIR ✅, LoRA ⚪ |
| 9 — Colab T4 / CPU | ⚪ after Phase 7 (agreed order) |

**Milestone M1 (Data + Retrieval): ✅** catalogue built; 4 retrieval variants scored. **M2 (Need analyst): ✅** rewriting vs raw measured on test; reasoning mode confirmed on dev. **Stretch: BLaIR embedder ✅** (adopted, with Qwen3-Reranker).

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
| 2026-10-26 | **Reasoning mode:** `reasoning=True` for free text + tool calls; `reasoning=False` for schema-constrained calls (need analyst, supervisor router) — **confirmed on dev in Phase 3** (28× faster, no retrieval loss) | Off leaks narration into free text/tool calls; structured output stays clean and is ~27× faster (0.6 s vs 16.4 s) |
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

## Phase 2 — Retrieval (RAG) ✅

- [x] `Pickwise/retrieval.py` — `Retriever`: BM25 + FAISS dense (bge-small, cosine) → RRF (optionally weighted) → cross-encoder re-rank; category filter; one doc per product (chunked mode for the comparison)
- [x] `Pickwise/metrics.py` — Recall@10, MRR (depth 100), nDCG@10, Recall@50/100
- [x] BM25 vectorised (sparse matrix) — **identical scores to `rank_bm25`** (checked), 200 queries in 0.05 s vs 33 s
- [x] Doc embeddings cached in `data/emb_*.npy` (first build ≈ 60 s on GPU)
- [x] Dev tuning sweep (raw requests)
- [x] Chunking comparison on dev: 83,021 chunks; BM25 0.110, dense 0.440, hybrid 0.340, hybrid+rerank 0.465 Recall@10 — about the same as one doc per product with 2.5× the index → keep one doc per product
- [x] Notebook §4 Baseline + §5a Retrieval, 4 variants + chunking row scored on test

### Dev results on the raw request (200 queries; ±3.5 pts standard error on Recall@10)

| Variant | Recall@10 | MRR | nDCG@10 | Recall@100 |
|---|---|---|---|---|
| BM25 | 0.070 | 0.050 | 0.052 | 0.190 |
| Dense (bge + query instruction) | **0.465** | **0.279** | **0.318** | 0.655 |
| Dense (no instruction) | 0.445 | 0.276 | 0.310 | 0.640 |
| Hybrid RRF, equal weights (lab) | 0.325 | 0.159 | 0.188 | 0.615 |
| Hybrid RRF, BM25 weight 0.25 | 0.455 | 0.224 | 0.272 | 0.655 |
| Hybrid RRF, BM25 weight 0.1 | 0.465 | 0.253 | 0.298 | 0.655 |
| Hybrid (equal) + rerank top 20 / 50 / 100 | 0.460 / 0.470 / 0.435 | 0.246 / 0.240 / 0.225 | | |
| Dense + rerank top 20 / 50 / 100 | 0.480 / 0.440 / 0.435 | 0.242 / 0.226 / 0.223 | | |
| Oracle: hybrid + gold item's category as filter | 0.400 | 0.225 | 0.257 | 0.735 |

**Takeaways:** BM25 collapses on 43-word situational requests (the brief's premise) and drags equal-weight RRF below dense. The ms-marco cross-encoder (trained on short web questions) doesn't beat dense on MRR for raw requests. A correct category filter helps. Hypothesis for Phase 3: short rewritten queries should rescue BM25 and the re-ranker. Final retrieval settings will be chosen on dev **after** Phase 3.

## Phase 3 — Need Analyst ✅

- [x] `Pickwise/need.py` — `ShoppingNeed {product_type, must_haves, category: Literal[31 categories]}` (Module 2 pattern), prompt with a guide to what each short category name holds, `analyse` / `analyse_many` with a JSONL reply cache (`data/needs/`, keyed by model + reasoning + prompt version), `search_query(need, request)`
- [x] Dev + test analysed with `reasoning=False`: **0 failures**, ~1.0 s per request
- [x] Dev experiment: query form × retriever × category filter
- [x] Multi-query fusion tried (BM25 on rewrite + dense on raw [+ dense on rewrite]) — no gain beyond noise
- [x] Reasoning on vs off on dev — **reasoning off confirmed**

| Dev, 200 requests | Reasoning OFF | Reasoning ON |
|---|---|---|
| Time per request | **1.1 s** (median 0.7) | 30.7 s (median 29, max 143) |
| Failures | 0 | 0 |
| Category accuracy | 43.0% | **52.5%** |
| System retriever Recall@10 / MRR | 0.475 / 0.260 | 0.465 / 0.276 |

Paired bootstrap, ON − OFF: Recall@10 −0.010 [−0.045, +0.025], MRR +0.016 [−0.009, +0.041] → no measurable retrieval gain for 28× the time. Note for later: reasoning ON picks categories better — relevant only if category is ever used as a soft boost.
- [x] Notebook §5b need-analyst cells; test rows in the results table; paired bootstrap vs baselines

### Test results (800 queries, scored once)

| Condition | Recall@10 | MRR | nDCG@10 | Recall@100 |
|---|---|---|---|---|
| BM25 · raw request | 0.098 | 0.062 | 0.068 | 0.200 |
| Dense · raw request | 0.409 | 0.248 | 0.279 | 0.672 |
| Hybrid (RRF) · raw request | 0.315 | 0.158 | 0.186 | 0.602 |
| Hybrid + re-rank · raw request | 0.405 | 0.232 | 0.267 | 0.602 |
| Hybrid + re-rank · raw · chunked index | 0.399 | 0.226 | 0.261 | 0.581 |
| BM25 · rewrite + request | 0.234 | 0.140 | 0.158 | 0.381 |
| Dense · rewrite + request | 0.422 | 0.247 | 0.282 | 0.664 |
| Hybrid (RRF) · rewrite + request | 0.361 | 0.217 | 0.243 | 0.636 |
| Hybrid + re-rank · rewrite only | 0.442 | 0.257 | 0.294 | 0.682 |
| **Hybrid + re-rank · rewrite + request (system)** | **0.452** | **0.276** | **0.312** | 0.636 |

Paired bootstrap (2,000 resamples), system minus baseline:
- vs **dense · raw**: Recall@10 **+0.044** [0.019, 0.069], MRR **+0.027** [0.007, 0.048]
- vs **hybrid + re-rank · raw**: Recall@10 **+0.048** [0.025, 0.072], MRR **+0.043** [0.027, 0.061]

→ **Rewriting beats the raw request for the full pipeline** (both CIs exclude 0). Test category accuracy: 52%.
- `Pickwise/metrics.py` gained `evaluate(..., return_ranks=True)` and `paired_bootstrap` (reused in Phase 7).

### Dev results (200 queries, reasoning off)

| | raw request | rewrite | rewrite + request |
|---|---|---|---|
| BM25 Recall@10 / MRR | 0.070 / 0.050 | **0.310 / 0.180** | 0.215 / 0.124 |
| Dense | **0.465 / 0.279** | 0.420 / 0.260 | 0.445 / 0.277 |
| Hybrid (RRF) | 0.325 / 0.159 | 0.385 / 0.236 | 0.395 / 0.216 |
| Hybrid + re-rank (top 50) | 0.470 / 0.240 | 0.455 / 0.243 | **0.475 / 0.260** |

- **Category accuracy 43%** → using the predicted category as a hard filter **halves recall** (hybrid+rerank Recall@10 0.47 → 0.23). Overlapping category names (Home / Tools / Household) cause much of it. → **No automatic category filter**; `category` stays an optional tool argument for when the shopper states one.
- **Rewriting rescues BM25 (×4) and hybrid**, and raises deep recall (dense Recall@100 0.655 → 0.695), but **doesn't beat dense on the raw request at the top-10**. Every configuration lands at Recall@10 0.41–0.50, within the ±3.5-pt dev noise.
- ~30% of gold items aren't in any top 100 → the bottleneck is the general-purpose embedder on review-style requests. **The BLaIR embedder (stretch goal) is the most promising upgrade.**
- **Decision (evidence-based, revisable):** system retriever = **hybrid + re-rank, lab defaults (equal-weight RRF k=60, re-rank top 50), on the query "rewrite. request"**. One query string (natural for the MCP tool), top-tier on dev, meets the brief's hybrid + re-ranker requirement.

## Model upgrade (2026-10-26) ✅ adopted: BLaIR + Qwen3-Reranker

Tried alternative open models for the two retrieval models (agent LLM unchanged). All on the 200 dev queries; test untouched.

| Setting (query = rewrite + request unless noted) | Recall@10 | MRR | nDCG@10 | Recall@100 | GPU ms/query |
|---|---|---|---|---|---|
| **Current:** bge-small · hybrid · ms-marco re-rank top 50 | 0.475 | 0.260 | 0.305 | 0.660 | ~70 |
| bge-small · hybrid · **Qwen3-Reranker-0.6B** top 50 | 0.525 | 0.349 | 0.388 | 0.660 | ~860 |
| **BLaIR** · dense only · raw request | **0.610** | 0.364 | 0.417 | **0.795** | ~5 |
| BLaIR · hybrid · ms-marco top 50 | 0.535 | 0.278 | — | 0.750 | ~70 |
| **BLaIR · hybrid · Qwen3-Reranker top 20** | 0.590 | **0.381** | **0.426** | 0.750 | 334 |
| BLaIR · hybrid · Qwen3-Reranker top 10 / 30 / 50 | 0.505 / 0.575 / 0.575 | 0.353 / 0.375 / 0.377 | | | 199 / 495 / 830 |
| Qwen3-Embedding-0.6B · dense · raw | 0.465 | 0.302 | 0.335 | 0.710 | |
| Qwen3-Embedding-0.6B · hybrid · Qwen3-Reranker top 50 | 0.550 | 0.353 | 0.396 | 0.675 | |
| any embedder · bge-reranker-v2-m3 / bge-reranker-base | ≤ 0.495 | ≤ 0.321 | | | |

- Paired bootstrap, best (BLaIR · Qwen3-Reranker) − current: **Recall@10 +0.095 [0.04, 0.15], MRR +0.118 [0.071, 0.164]**.
- BLaIR (0.1B, Amazon-specific) beats the general-purpose embedders by a wide margin; Qwen3-Reranker is the first re-ranker that improves MRR on long requests; bge re-rankers don't help.
- Qwen3-Reranker on CPU (8 threads): ~10 s per query for top 20, ~17 s for top 50.
- EmbeddingGemma-300M not tested: gated on Hugging Face (needs a login + licence acceptance).
- Caveats: dev only; BLaIR was pretrained on Amazon Reviews 2023 (possible overlap with the reviews C4 queries were written from → Limitations).
- `Pickwise/retrieval.py`: `Retriever(embed_model=..., rerank_model=...)`; BLaIR loaded with CLS pooling (model-card recipe); Qwen3-Embedding query instruction added.
- **Decision (team, 2026-10-26): adopt** BLaIR · hybrid RRF · Qwen3-Reranker top 20 as the system retriever (`config.EMBED_MODEL/RERANK_MODEL/RERANK_TOP_N`); the lab models stay as `config.LAB_*` for the Phase 2 rows and as an ablation. Each `Retriever` carries its own embedder, re-ranker and re-rank depth.

### Test results (800 queries, scored once)

| Condition | Recall@10 | MRR | nDCG@10 | Recall@100 |
|---|---|---|---|---|
| Hybrid + re-rank · rewrite + request (lab models, previous system) | 0.452 | 0.276 | 0.312 | 0.636 |
| bge-small · hybrid + Qwen3 re-rank · rewrite + request | 0.430 | 0.303 | 0.329 | 0.636 |
| BLaIR · hybrid + ms-marco re-rank · rewrite + request | 0.504 | 0.300 | 0.339 | 0.758 |
| BLaIR · dense only · raw request | 0.575 | 0.356 | 0.401 | 0.804 |
| BLaIR · hybrid + Qwen3 re-rank · raw request | 0.568 | 0.360 | 0.406 | 0.752 |
| **BLaIR · hybrid + Qwen3 re-rank · rewrite + request (system)** | **0.548** | **0.360** | **0.400** | 0.758 |

Paired bootstrap, system minus:
- dense · raw (bge-small, best brief baseline): Recall@10 **+0.139** [0.107, 0.170], MRR **+0.112** [0.090, 0.135]
- previous system (lab models): Recall@10 **+0.095** [0.066, 0.124], MRR **+0.085** [0.063, 0.106]
- BLaIR · dense · raw: Recall@10 −0.028 [−0.054, 0.000], MRR +0.004 [−0.018, 0.027] → **level**
- BLaIR · hybrid + Qwen3 · raw request: Recall@10 −0.020 [−0.044, 0.005], MRR +0.000 → **rewriting no longer helps with BLaIR**

**Takeaways:** the upgrade is the largest gain in the project. With BLaIR, the hybrid + re-ranker + rewrite pipeline is no better than BLaIR alone on the raw request, which is a ~5 ms, CPU-friendly fallback. The system keeps the brief's pipeline (chosen on dev; test isn't used for choices). Limitation: BLaIR's pretraining (Amazon Reviews 2023) may overlap with the reviews the C4 queries were written from.
- Notebook: §5b *Retrieval model upgrade* (dev comparison of 3 embedders × 4 re-rankers, test rows, paired bootstrap); full run 32 min (`run_notebook.py`, live log in `notebook_run.log`).

## Phase 4 — MCP Server + Multi-Agent ✅ (branch `phase-4-mcp-multi-agent`)

Choices were made without asking, as requested; every one is in [`DECISION_LOG.md`](DECISION_LOG.md) (4.1–4.16, R.1–R.2).

- [x] `Pickwise/shop.py`: `search_products(query, category, k)`, `get_product(item_id)`, `list_categories()`. **One set of functions for both transports**, JSON out, and search reports its own compute time.
- [x] `Pickwise/mcp_server.py`: FastMCP server over streamable HTTP (127.0.0.1:8766) or stdio, plus the `catalogue://stats` resource. **`add_to_cart` is not exposed.**
- [x] `Pickwise/tools.py`: local `@tool`s, `ensure_server()` auto-start, `get_tools("mcp" | "local")`.
- [x] `Pickwise/graph.py`: supervisor (rule + LLM router) → need analyst → **searcher (tool node)** → **comparator (structured picks + verbatim quotes, `Literal` item ids)**; no separate writer; `run_turn` prints a trace of every step and tool call; single-ReAct baseline `build_react_baseline` / `run_react`.
- [x] `Pickwise/evaluate.py`: cached end-to-end runs (`data/runs/`), Hit@3 summary.
- [x] `PICKWISE_RETRIEVAL=dense` switch (BLaIR only) for CPU runs.
- [x] Notebook §5b *MCP server, searcher and comparator*: tools/list, MCP vs local latency (search vs transport), graph (Mermaid), traced turns, ReAct baseline, smoke test.

### Smoke test (30 dev requests)
| Variant | Hit@3 | s / request | quotes verbatim |
|---|---|---|---|
| **System** (comparator reasoning off, default) | 0.533 | **4.2** | **97.8%** |
| System, comparator reasoning on | 0.567 | 58.8 | 78.9% |
| Single ReAct agent | 0.467 | 46.9 | — |

Reasoning on − off: ΔHit@3 +0.033 [−0.10, +0.17] → **reasoning off** (decision 4.15). One turn on a free GPU: need analyst ≈ 1–2 s, MCP search ≈ 0.5 s, comparator ≈ 3 s.

- **MCP vs local (20 dev searches): identical rankings.** Search compute ≈ 0.46 s either way; MCP transport ≈ 0.027 s per call (≈ 6%).
- Full notebook run: 31.7 min (26 code cells), no errors.
- **Known issue → Phase 7:** quotes often copy the title, and reasons can go beyond the quote, so the quote check is not a faithfulness score. The LLM judge must check the reasons (DECISION_LOG 4.17).

## Phase 5 — Memory + Human-in-the-Loop ✅ (branch `phase-5-memory-approval`)

Built on top of Phase 4. Every choice is in [`DECISION_LOG.md`](DECISION_LOG.md) (5.1–5.16). Agreed order: finish and verify all phases first, then optimise for Colab T4.

- [x] `Pickwise/memory.py`: Store-backed preferences `prefer / avoid / avoid_brands / budget`. A cue-gated extractor, plus deterministic guards: only the shopper's own words, a negation decides the side, a capitalised avoided name is a brand.
- [x] `Pickwise/cart.py`: `add_to_cart` (needs `approved=True`), audit log, `approval_share()`.
- [x] `Pickwise/graph.py`: `remember` node; supervisor rule for cart words; **cart → approval** with `interrupt()`; checkpointer + Store attached; `run_turn` / `resume` with thread and user ids and the recursion limit; avoided-brand filter; first-sentence reasons; follow-up questions answered by the comparator.
- [x] Notebook §5b *Memory and human approval*: two conversations for one shopper, approve / reject, an avoided-brand filter, and an adversarial cart test.

**Test results:**
- Memory across turns and conversations: avoid latex and Munchkin (Munchkin flagged as a brand), carried into a new conversation.
- The sippy-cup search **removed 1 Munchkin product**.
- Cart: approve → added; "already approved, don't ask" → still paused → rejected → not added.
- **Adversarial: 8/8 paused, 0 added.**
- **100%** of executed adds were approved.
- Extractor on 7 varied messages: 5/7 right (misses: "vegan"; a double negative).

## Phase 6 — Demo ✅ (branch `phase-6-demo`)

Choices are in [`DECISION_LOG.md`](DECISION_LOG.md) (6.1–6.12).

- [x] Notebook §8: `await ask("...")` and `await chat()` for the live talk. Every step and tool call is traced; the cart pauses for *approve / reject?*. With `PICKWISE_INTERACTIVE` unset, the demo follows a script and answers an unattended pause with **reject**.
- [x] Scripted demo: "I'm vegan, so no leather please. I need a slim wallet" → "which of these holds the most cards?" → add #1 (approve) → "now I need a dress belt to wear with my office trousers" (leather remembered; 10 leather belts removed) → add #2 (reject).
- [x] **Fixes found while making the demo:**
  - "which of these" compares only the 3 products shown;
  - unsupported "no latex" claims are removed, with an honest *"the description doesn't mention latex"* note;
  - **avoided materials and brands are filtered deterministically, from the same message** (wallets: 7 leather products removed; belts: 4);
  - the LLM router can no longer route to the cart (it once routed "Now I need a belt" there);
  - **the avoid filter fails closed**: it once switched itself off when most results were leather and recommended 2 leather wallets to the vegan shopper. Now the searcher fetches 30, keeps the 10 best that pass, and shows fewer than 3 (saying so) rather than an avoided product. 0/15 leather picks on 5 vegan follow-ups;
  - **clear follow-ups skip the LLM router** ("I need…" → search, "which of these…" → compare): routing 14/16 → **26/26** on follow-up messages, 10 of them held out;
  - reasons no longer lose their wording when an unsupported clause is cut ("This wallet is described, a…").
- [ ] **Open, for the Phase 7 error analysis** (6.12): the need analyst can carry the last product's must-haves into a new one; a weak search still shows 3 weak picks.

## Phase 8 — Web app ✅ (branch `phase-8-app`)

Choices are in [`DECISION_LOG.md`](DECISION_LOG.md) (8.1–8.8). Built while the Phase 6 verification run was going.

- [x] `app.py` (Gradio 6.30): chat; **agent trace streaming step by step**; **Approve / Reject** buttons when a cart action pauses (or type yes / no); remembered preferences, cart and approval log per shopper; New conversation / Forget my preferences; MCP ↔ local switch. Run: `.venv/bin/python app.py` → http://127.0.0.1:7860 (`--share` for a public link).
- [x] Runs the **same graph** as the notebook. `graph.py` now yields its trace lines (`stream_turn`), and the notebook prints them exactly as before.
- [x] **Fixes from your try-out** (screenshots, 2026-10-27; `DECISION_LOG.md` 8.9–8.12):
  - "I'm allergic to latex and I **never buy** anything from Munchkin. I need …" paused for the cart (the word "buy"). Negated buying and new requests are no longer cart actions; the 8 adversarial messages still all pause.
  - "more details on third item" re-ranked all three, and naming a product by its title started a new search. New **details** step: one shown product (by position or by its title) is fetched with `get_product` and described, or a question about it is answered with a quote checked against its text; if the text is silent it says so. "Add it" then adds that product.
  - "I always buy Columbia" was not remembered (the 4B extractor returned nothing 5/5 times in that session). A rule now backs up explicit brand statements: 8/8 test messages right.
  - Routing test: **34/34** in both contexts (8 new cases).
- [x] Tested through the app's own HTTP API: **21/21 checks** (16 + 5 for the reported cases); before the fixes: **16/16 checks**: streaming, vegan filter, approve button, typed "no", MCP→local mid-conversation, memory across conversations, forget, two shoppers in two sessions. First reply 4.1 s (graph built when the page opens; was 14.2 s). Browser screenshots (desktop + phone) checked the layout; fixed the trace wrapping, a duplicated cart question and squeezed table columns.

## Findings / notes

- Catalogue jsonl schema: `{"item_id", "category", "metadata"}` — `metadata` is one string, title first. Short items often look like `"Her Leaves Remain. ."`.
- Queries file (`test.csv`): `qid, query, item_id, user_id, ori_rating, ori_review`.
- **For Phase 2 (review issue #2):** median item is only 32 words, so most items fit in one 300-char chunk anyway. Indexing **one document per item** (title first, so embedder truncation at 512 tokens keeps the important part) avoids chunk→item score aggregation entirely. **Decided 2026-10-26: one doc per item.**

## Open issues

See *Plan Review — Open Issues* in `P14_Project_Plan.md`. Resolved so far: #2 (decided), #5, #8, #12, #14.

## Change log

- 2026-10-26 — Project named **Pickwise**: package `p14/` → `Pickwise/`, env vars `P14_*` → `PICKWISE_*`, titles updated. Notebook re-run clean after the rename.
- 2026-10-26 — Package folder renamed to `Pickwise/` (capital P); imports and docs updated, notebook re-run clean. README added; first push to GitHub.
- 2026-10-26 — **Phases 2 and 3 done.** Retrieval (`retrieval.py`, `metrics.py`) and need analyst (`need.py`) built; notebook §4–5b filled; dev tuning + test results + paired bootstrap; reasoning-off confirmed for structured calls. Next: Phase 4 (MCP server + multi-agent).
- 2026-10-26 — **Retrieval model upgrade adopted** (BLaIR + Qwen3-Reranker top 20): test Recall@10 0.548 / MRR 0.360 (+9.5 pts / +0.085 over the lab models). `run_notebook.py` added (live per-cell log). GPU was briefly blocked by another Ollama model (qwen3-coder:30b) — unloaded on request.
- 2026-10-26 — **Phase 4 built** on branch `phase-4-mcp-multi-agent`: MCP server + local tools, multi-agent graph, ReAct baseline, smoke test (system 0.533 Hit@3 at 4.2 s/request). `DECISION_LOG.md` started.
- 2026-10-27 — **Phase 5 built** on branch `phase-5-memory-approval`: memory (checkpointer + Store), cart → approval with interrupt(), audit log; adversarial test 8/8 paused. Agreed: finish all phases, then optimise for Colab T4.
- 2026-10-27 — **Phase 6 built** on branch `phase-6-demo`: `ask()` / `chat()` demo; fixes for compare scope, unsupported avoidance claims, deterministic avoid-filter (same turn), router can't pick cart.
- 2026-10-27 — **Phase 6 verification run** found 3 more bugs, all fixed (6.8–6.10): the avoid filter failed open, the LLM router sent a new request to compare, the clause stripper broke grammar. Demo belt request reworded (6.11). `P14_Project_Plan.md` brought fully up to date (diagram, success target, Phase 7 plan, Phase 9, milestones, risks, checklist).
- 2026-10-27 — **Phase 6 pushed** (`c1372fe`). **Phase 8 web app built** (`app.py`, branch `phase-8-app`): 16/16 end-to-end checks.
