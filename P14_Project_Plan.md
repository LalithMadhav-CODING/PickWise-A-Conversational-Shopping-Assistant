# 🛒 Pickwise — Conversational Shopping Assistant (P14)

### Product Search with MCP, RAG and Human Approval

> [!info] Project facts
> **Level** · Advanced  ·  **System** · Multi-agent (supervisor) + MCP + memory  ·  **RAG** · Required  ·  **Fine-tuning** · Optional (stretch)
> **Deliverable** · 1 notebook + 10-minute talk + 5-minute Q&A
> **Dataset** · McAuley-Lab/Amazon-C4 (Hugging Face, public)
> **Model** · qwen3:4b via Ollama (≤4B params, no paid APIs)

> [!abstract] Where we are · updated 2026-10-27 (evening)
> Phases 0–7 done (✅ below) · Phase 8 web app ✅ · **Phase 9: run on free Colab T4 / laptop CPU — next** · talk slides + outline ready (`TALK_OUTLINE.md`). Results vs the pre-registered target: Hit@3 0.380 (missed narrowly) · vs ReAct not significant · faithfulness 59% (missed) · approvals 100% (met). Live progress: `CURRENT_PROJECT_STATUS.md` · every choice made without the team: `DECISION_LOG.md`.

---

## 📌 TL;DR

Shoppers describe a **situation**, not keywords — *"something to keep my kids busy at bath time that can get messy."* Keyword search fails on these.

We build an assistant that turns a long request into catalogue searches, **recommends and compares** the best products **with reasons taken from the product text**, **remembers shopper preferences** across conversations, and **never adds to cart without confirmation**.

The system is a **LangGraph multi-agent graph** (supervisor → need analyst / searcher / comparator / cart), with the catalogue served through an **MCP server** (and as local `@tool`s for comparison), **hybrid RAG** retrieval, **checkpointer + store** memory, and a **human approval** gate in front of every cart action.

---

## 🏁 Project Goal

Build a shopping assistant that works on **situational requests** — where keyword search fails — and **prove it with numbers**.

**The assistant must:**
1. **Understand** a long (~45-word) situational request and turn it into catalogue searches
2. **Recommend the top 3 products**, each with a reason **grounded in the product text** (no invented claims)
3. **Remember each shopper's preferences** (brands, budget, dislikes) across conversations
4. **Never add to cart without explicit shopper approval** — 100%, no exceptions

**Constraints** · small open models (≤ 4B, qwen3:4b via Ollama) · free Colab T4 or laptop CPU · no paid APIs

**Proof** · one results table on a fixed test subset of Amazon-C4 (each request has a gold product): Retrieval (Recall@10, MRR, nDCG@10), End-to-end (Hit@3), Faithfulness of reasons, Approval rate (100%) — baselines vs. full system vs. ablations, plus ≥ 10 analysed failures

**Deliverable** · one re-runnable notebook + 10-minute talk with live demo + 5-minute Q&A

> [!question] Research question *(required by the notebook template — confirm with the team)*
> Does a multi-agent system with structured query rewriting recommend the gold product more often than plain hybrid retrieval on the raw request?
>
> **Success target** · *proposed 2026-10-27; written into notebook §1 **before** any Phase 7 test run (`DECISION_LOG.md` 7.1)*. On the fixed 200-request test subset:
> 1. the full system reaches **Hit@3 ≥ 0.40**;
> 2. it **beats the single-ReAct baseline** on Hit@3, with a paired 95% interval excluding zero;
> 3. **≥ 90%** of its recommendations are judged *supported* by the product text;
> 4. **100%** of executed `add_to_cart` calls were approved.
>
> Retrieval (already met, §5): the system retriever beats the best brief baseline on Recall@10 (0.548 vs 0.409, paired CI excludes 0).
>
> **Outcome (Phase 7, 200 test requests):** (1) Hit@3 0.380, 95% CI [0.31, 0.45]: missed narrowly · (2) vs ReAct +5.5 points [0.0, 10.5]: not met · (3) 59% judged supported (91% with partly): missed · (4) 100% approved: met.

---

## 🎯 Problem Statement

| | |
|---|---|
| **Input** | A long, situational shopping request (~45 words) |
| **Output** | Top-3 product recommendations, each with a reason grounded in the product text |
| **Memory** | Shopper preferences (brands, budget, dislikes) persist across conversations |
| **Hard constraint** | `add_to_cart` **must never** run without explicit human approval (100%) |
| **Forbidden** | Using `ori_review` in the index (it is the review the query was written from — answer leakage) |

---

## 🏗️ System Architecture

As built (Phases 4–6; `Pickwise/graph.py`):

```
shopper ──► remember ──► supervisor ──► need analyst ──► searcher ──► comparator ──► END   new request
            (Store)          │                            (tool node)  (top 3 + reason
                             │                                │         + verbatim quote)
                             ├──────────────────────────────────────► comparator ──► END   "which of these…?"
                             ├──► details (get_product) ─────────────────────────► END     "more about the third one"
                             └──► cart ──► approval ──► add_to_cart ─────────────► END     "add the second one"
                                           interrupt(): shopper says yes / no
                                                              │
                                                ┌─────────────┴─────────────┐
                                                ▼                           ▼
                                        MCP server (HTTP)              local tools    ← same functions,
                                        search_products ·              (in-process)     two transports
                                        get_product · list_categories

Memory:  checkpointer (this conversation, thread_id)  +  Store (shopper preferences, user_id)
```

> [!note] Node roles
> - **Remember** — keeps lasting preferences (prefer / avoid / avoided brands / budget) in the Store; a cue-gated LLM extractor plus code rules for grounding, polarity and brands
> - **Supervisor** — rules first (no products shown yet → search; a request to add a shown product → cart; a message about ONE shown product → details; a clear new request → search; a clear question about the shown products → compare), then an LLM router (structured output) for the rest, choosing *search* or *compare*. The LLM can never route to the cart
> - **Need analyst** — rewrites the long request into a structured query (product type, must-haves, category)
> - **Searcher** — a tool node (no LLM): calls `search_products` over MCP or locally (BLaIR · hybrid RRF · Qwen3 re-rank). When the shopper avoids a brand or material, it fetches 30, drops the avoided ones and keeps the best 10 (fails closed)
> - **Comparator** — picks the top 3 from the candidates with a reason and a **verbatim quote** from the product text; unsupported claims are cut. The reply is rendered from these picks (no separate writer)
> - **Details** *(Phase 8)* — one shown product, fetched with `get_product`: its description, or the answer to a question about it backed by a quote checked against its text ("doesn't say" when the text is silent)
> - **Cart → approval** — cart works out *which* product is meant; approval pauses with `interrupt()` and calls `add_to_cart` only on an explicit yes

---

## 📊 Data Plan

| | |
|---|---|
| **Queries** | `load_dataset("McAuley-Lab/Amazon-C4")["test"]` — 21,223 rows (qid, query, item_id, user_id, ori_rating, ori_review) |
| **Catalogue** | `sampled_item_metadata_1M.jsonl` — 1.06M items (item_id, category, metadata). `metadata` is **one string**, `"Title. Description…"` |
| **Sample** | 1,000 queries, `shuffle(seed=42)` → 998 unique gold items (2 queries share one) |
| **Distractors** | 31,616 random items (`rng.random() < 0.03`, seed 42) → **32,614-item catalogue** |
| **Split** | first 200 = dev (tuning only) / remaining 800 = test (reported once) (QUICK_RUN: first 100 of test) |
| **Index text** | the full `metadata` string; `title` = text before the first `". "` (display only) — **never `ori_review`** |
| **Cache** | filtered catalogue → `data/catalogue.parquet` (gitignored), keyed on seed / rate / gold set; first build ≈ 3 min |
| **Data quality** | 4 gold items have **empty text** (unanswerable), 15 have ≤ 5 words — all 4 empty ones fall in test. Kept in, reported in error analysis |

> [!warning] Rule
> The notebook must download data with `load_dataset(...)` at the **top**, so anyone can re-run it. If you use a subset, select it with a **fixed seed** and **state its size**. The RAG index must be built from this downloaded data, not pasted text.

---

## 🤖 Models

| Purpose | Model | Notes |
|---|---|---|
| Agents (tool calling, structured output) | **qwen3:4b** | primary; `qwen3:1.7b` fallback. **Reasoning off** for every schema-constrained call (need analyst, router, comparator, preference extractor, cart resolver): same quality, 14–28× faster, no leak. **Reasoning on** for free text and tool calls (ReAct baseline, faithfulness judge), where it keeps narration out of the answer (see Risks) |
| Embeddings (system) | **`hyp1231/blair-roberta-base`** (BLaIR, 0.1B) | the brief's optional product embedder; CLS pooling; **adopted 2026-10-26** — dev Recall@10 0.61 vs 0.465 (dense, raw) |
| Re-ranker (system) | **`Qwen/Qwen3-Reranker-0.6B`**, top 20 | a Qwen model ≤ 4B; first re-ranker that improves MRR on long requests; ~0.33 s/query GPU, ~10 s CPU |
| Embeddings (lab baseline) | `BAAI/bge-small-en-v1.5` | 384-dim, with the bge query instruction; Phase 2 baselines + ablation |
| Re-ranker (lab baseline) | `cross-encoder/ms-marco-MiniLM-L-6-v2`, top 50 | Phase 2 pipeline + ablation |
| Also tested (dev) | `Qwen3-Embedding-0.6B`, `bge-reranker-base`, `bge-reranker-v2-m3` | not adopted (smaller or no gains); EmbeddingGemma-300M gated on HF, not tested |
| Optional LoRA need analyst | `Qwen2.5-0.5B` | stretch; 2–5k C4 queries outside test subset |

**Compute** · free Colab T4 or laptop CPU ≤ 4B params · no paid APIs

---

## 🚧 Build Plan

Each phase lists what it **builds on** (the lab code to adapt) and its **done-when**.

### Phase 0 — Setup & Scaffold
- [x] `pip install -r requirements.txt` — full stack in `.venv/` (langgraph, langchain, langchain-ollama, langchain-mcp-adapters, mcp, faiss-cpu, rank-bm25, sentence-transformers, datasets)
- [x] `CONFIG` block — `Pickwise/config.py` (model, paths, seed, QUICK_RUN flag, RISKY_TOOLS)
- [x] Verify Ollama + qwen3:4b — reasoning on/off, structured output, tool calls probed (`Pickwise/llm.py`)
- [x] Notebook `P14_Shopping_Assistant.ipynb` with all 10 template sections
- **Builds on** · Module 4 · Agentic_app (`config.py`, `llm.py`)
- **Done when** · clean notebook skeleton runs top-to-bottom ✅ 2026-10-26

### Phase 1 — Data & Catalogue
- [x] `load_dataset` at the top (test split + metadata jsonl, streaming)
- [x] 1,000 queries (seed 42) + 998 gold items + 31,616 distractors
- [x] Build catalogue DataFrame: `item_id, category, title, text` (`Pickwise/data.py`), cached to parquet
- [x] 200 dev / 800 test split
- [x] Asserts: every gold item present; no `ori_review` in the index
- [x] Data-quality report: empty / very short gold items
- **Builds on** · spec §Data
- **Done when** · reproducible catalogue; no `ori_review` in the index

### Phase 2 — Retrieval (RAG)  →  *Module 3*
- [x] **One document per item** (full `metadata` text, title first) — median item is 32 words, so no chunk→item merging is needed _(decided 2026-10-26, review issue #2)_
- [x] Comparison row: Module 3 chunking (`RecursiveCharacterTextSplitter`, 300 / 50, contextual headers) with max-score-per-item aggregation — same quality, 2.5× the index (83k chunks)
- [x] **Dense** — FAISS + `bge-small-en-v1.5` (cosine, bge query instruction), embeddings cached
- [x] **Sparse** — BM25 (`rank_bm25` formula, vectorised; identical scores, ~600× faster)
- [x] **Hybrid** — Reciprocal Rank Fusion (k = 60, equal weights = lab default; weights tuned on dev, no reliable gain)
- [x] **Re-rank** — cross-encoder over the hybrid top 50
- [x] Evaluate: **Recall@10, MRR, nDCG@10** vs gold `item_id` (+ Recall@50/100, paired bootstrap CIs)
- [x] Compare: BM25 raw · dense raw · hybrid · hybrid+rerank — **test Recall@10 0.10 · 0.41 · 0.32 · 0.41**
- **Builds on** · Module 3 · RAG_Pipeline_Lab; Module 4 · Agentic_app (`rag.py`)
- **Done when** · results table for all 4 retrieval variants ✅ 2026-10-26

### Phase 3 — Need Analyst  →  *Module 2*
- [x] Structured query rewriting: long request → `{product_type, must_haves, category}`; category = `Literal` over the 31 catalogue categories, with a guide to what each name holds
- [x] Schema-constrained output (`with_structured_output`) with `reasoning=False`; replies cached (`data/needs/`); 0 failures on 1,000 requests, ~1 s each
- [x] Confirm on dev: `reasoning=False` vs `True` — **confirmed `False`**: 1.1 s vs 30.7 s per request (28×), no retrieval gain (paired bootstrap Recall@10 −0.010 [−0.045, +0.025], MRR +0.016 [−0.009, +0.041]); reasoning on picks the category better (52.5% vs 43%) but category isn't used as a filter
- [x] Measure: does rewriting beat searching with the raw request? **Yes for the full pipeline** — hybrid+rerank on *rewrite + request*: test Recall@10 0.45 / MRR 0.28 vs 0.41 / 0.23 raw; beats the best baseline (dense raw) with paired-bootstrap 95% CIs excluding 0
- [x] Category as a hard filter **rejected**: 43% dev category accuracy → recall halves. `category` stays an optional tool argument
- **Builds on** · Module 2 · Structured_Generation_Lab (constrained decoding, `Literal` choices)
- **Done when** · retrieval metrics with vs. without rewriting ✅ 2026-10-26

### Phase 4 — MCP Server + Multi-Agent  →  *Module 4*  ✅ (branch `phase-4-mcp-multi-agent`; choices in `DECISION_LOG.md`)
- [x] **MCP server** (`Pickwise/mcp_server.py`): `search_products(query, category, k)`, `get_product(item_id)`, `list_categories()` over streamable HTTP (or stdio); **no `add_to_cart`**
- [x] **Local `@tool`** versions: the *same* functions (`Pickwise/shop.py`), so the ablation measures transport only
- [x] **LangGraph supervisor**: rule ("no products shown yet → search") + LLM-as-router (structured output, reasoning off) for follow-ups
- [x] **Specialists**: need analyst (Phase 3) → **searcher = tool node** (no LLM; runs exactly the evaluated retriever) → **comparator = structured picks** `{item_id ∈ candidates, reason, verbatim quote}`, quotes checked against the product text
- [x] ~~Writer node~~ → **dropped**: the reply is rendered from the comparator's picks (one LLM call fewer; nothing added beyond the checked picks). Closes review issue #3
- [x] **Reasoning mode**: reasoning off for every structured call: need analyst, router, **comparator** (dev smoke test: same Hit@3, 14× faster, 98% vs 79% verbatim quotes). Reasoning on only for free text / tool calls (ReAct baseline)
- [x] ~~Tune the writer prompt for prose~~: not needed (template reply)
- [x] **Separate model time from transport time**: `search_products` reports its compute time; call − search = transport
- [x] **Single-ReAct baseline** (review issue #6): one agent, one search tool, lab guardrail
- [x] Printed trace of every agent step and tool call (`run_turn`); end-to-end runs cached (`data/runs/`)
- **Builds on** · Module 4 · Multi_Agent_System_End_to_End + MCP_Server_and_Tools_Lab + Agentic_app
- **Done when** · end-to-end run with a printed trace of every agent step + tool call ✅ 2026-10-26

### Phase 5 — Memory + Human-in-the-Loop  →  *Module 4*  ✅ (branch `phase-5-memory-approval`; choices in `DECISION_LOG.md` 5.1–5.16)
- [x] **Checkpointer** (short-term, per `thread_id`): follow-ups resolve ("the second one", "which of these is easiest to clean?")
- [x] **Store** (long-term, per `user_id`): `prefer / avoid / avoid_brands / budget`, written by a `remember` node (cue-gated LLM extractor + grounding, polarity and brand rules); used from the next turn on; avoidances applied after search (comparator + brand title filter), never in the query
- [x] `add_to_cart` in `RISKY_TOOLS`; **cart → approval** nodes, `interrupt()` before it; only an explicit yes executes; not on MCP, bound to no LLM, needs `approved=True`; audit log (`cart.AUDIT`)
- [x] Recursion limit on every turn (`run_turn`) + step tracing (remember / cart / approval / pause in the trace)
- [x] Adversarial cart test: 8/8 attempts to skip approval paused, 0 added
- **Builds on** · Module 4 · Multi_Agent_System_End_to_End (Sections D & E)
- **Done when** · multi-turn memory works; cart **always** pauses for approval ✅ 2026-10-27

### Phase 6 — Demo Cell  ✅ (branch `phase-6-demo`; choices in `DECISION_LOG.md` 6.1–6.12)
- [x] Multi-turn chat in the notebook: `await ask("...")` (the template's `ask()`), `await chat()` for typed input in the talk
- [x] Live trace of every agent step and tool call (`run_turn`)
- [x] Pause at the cart, resume **approve** / **reject** (`decision=` or typed when `PICKWISE_INTERACTIVE=1`; unattended runs default to reject)
- [x] Scripted demo: preferences ("vegan, no leather") → a follow-up question → approve → a new product with the remembered preference (a dress belt) → reject
- [x] Fixes from the Phase 5 run: "which of these" only compares the products shown; unsupported "no latex" claims removed, with an honest note instead
- [x] Avoided materials and brands filtered deterministically from the message that states them ("genuine leather" out, "faux / vegan leather" kept); the LLM router can no longer route to the cart (DECISION_LOG 6.6–6.7)
- [x] Verification-run fixes (6.8–6.10): the avoid filter **fails closed** (searcher fetches 30 and refills; fewer than 3 picks shown rather than an avoided one; 0/15 leather picks on vegan follow-ups); rules before the LLM router for clear follow-ups (routing 14/16 → 26/26, 10 held out); reasons keep their wording when a clause is cut
- [x] 6.12 reviewed in Phase 7: both are multi-turn behaviours, which the first-turn evaluation can't measure; they are listed in §9 Limitations
- **Builds on** · Module 4 · Multi_Agent_System_End_to_End (Section E) + the template's `ask()`
- **Done when** · a clean, presentable demo flow ✅ 2026-10-27

### Phase 7 — Evaluation  ✅ (branch `phase-7-evaluation`; choices in `DECISION_LOG.md` 7.1–7.12)
- [x] **Success target written into §1 before any test run** (see *Project Goal*; pushed as `b1a7130` before the runs)
- [x] **One fixed test subset**: the first 200 test requests (`QUICK_RUN`: 100), each with a fresh conversation and shopper. Retrieval metrics stay on all 800
- [x] **One results table**, columns Recall@10 · Hit@3 · faithful · seconds per turn:
  - baselines: BM25 raw · dense raw (Hit@3 = top 3 retrieved) · **single ReAct agent** (MCP)
  - **full system** (MCP)
  - ablations: rewriting off (`build_system(rewrite=False)`) · re-ranker off (hybrid only) · MCP vs local tools (same answers, latency)
- [x] Code: `Pickwise/judge.py` (judge + human sheet; calibrated on planted errors, 6/6), `evaluate.py` records each pick's reason and quote and each run's own query + 10 candidates, `rewrite` switch in `graph.py`; notebook §6 / §7 / §9 cells written
- [x] **Faithfulness**: verbatim-quote check on every pick + an **LLM judge** (qwen3:4b, reasoning on; each product *supported / partly / unsupported*) on the first 100, system vs ReAct + a **blind 30-row sheet for human labels** (`evaluation/human_check_faithfulness.csv`, *team task*) → judge-vs-human agreement
- [x] **Harness**: approval share from the audit log (must be 100%) + the adversarial cart test (Phase 5); latency MCP vs local
- [x] **Memory on/off**: 3 returning shoppers (avoid a brand: Nike · avoid a material: leather · prefer a brand: Columbia); count violations / adherence, after checking the catalogue has products that can violate or match
- [x] **Error analysis**: every miss classified (empty gold text · gold not in top 100 · ranked 11–100 · in top 10 but not picked · analyst misread) + ≥ 10 annotated examples
- [x] Paired bootstrap CIs for system vs ReAct and each ablation
- **Compute** · ≈ 5 h on the RTX 4000 (ReAct ≈ 2.6 h, judge ≈ 1.7 h, each system variant ≈ 15 min); run in the background, cached in `data/runs/`, so the notebook re-run only reads the caches
- **Builds on** · Module 3 (Recall@k, MRR) + spec §Evaluation
- **Done when** · full results table + error analysis written up; success target met or the miss explained

### Phase 8 — Stretch Goals (top marks)
- [x] **Web app (Gradio)** — `app.py`: chat, live agent trace, Approve / Reject at the cart, preferences, cart, approval log, MCP/local switch; runs the same graph as the notebook (branch `phase-8-app`, `DECISION_LOG.md` 8.1–8.8). Tested: 24/24 end-to-end checks through the app's HTTP API + browser screenshots (desktop, phone)
- [x] Fixes from trying the app (8.9–8.13): "I never buy X" no longer triggers the cart; a **details** step for one shown product, named by position, title or brand (`get_product`, quoted answers; asks which one if unclear); explicit brand statements remembered by rule
- [x] BLaIR embedder vs. general embedders — **done and adopted** (plus re-ranker comparison; notebook §5b *Retrieval model upgrade*)
- [ ] LoRA-tuned need analyst
- **Builds on** · Module 4 · Agentic_app (Streamlit reference; Gradio used, see 8.2)
- **Done when** · at least one stretch goal working ✅ (BLaIR, web app)

### Phase 9 — Run on free Colab T4 / laptop CPU  (brief constraint; after every phase is verified, as agreed 2026-10-27)
- [ ] `QUICK_RUN` also shrinks the dev experiments (model comparison, reasoning on/off check, smoke tests), which take hours on a first run today
- [ ] Recorded comparisons are loaded from saved results instead of recomputed (with a flag to recompute)
- [ ] Colab setup cell: install Ollama + pull qwen3:4b; T4 memory check (qwen3:4b + BLaIR + Qwen3-Reranker together)
- [ ] CPU path: the lab re-ranker as fallback (Qwen3-Reranker ≈ 10 s per search on CPU); state CPU timings
- [ ] Fresh run on a free Colab T4, top to bottom; time per section written in the notebook
- **Done when** · a fresh Colab T4 runs the notebook end to end in `QUICK_RUN`, and the run time is stated

---

## 📈 Evaluation Plan

| Layer | Metric | Target | Status |
|---|---|---|---|
| Retrieval | Recall@10, MRR, nDCG@10 vs. gold `item_id` | beat the best brief baseline, paired CI excludes 0 | ✅ met on test (800): 0.548 / 0.360 / 0.400 vs dense raw 0.409 / 0.248 |
| End-to-end | Hit@3 (gold among 3 recommendations) | **≥ 0.40**, and beat single ReAct (paired CI excludes 0) | ❌ 0.380 [0.31, 0.45], narrowly missed · vs ReAct +5.5 [0.0, 10.5], not met |
| Faithfulness | claims ∈ product text | **≥ 90%** judged supported; LLM judge + 30 human checks | ❌ 59% supported (91% with partly); quotes 97% verbatim; human checks ⏳ (team) |
| Harness | % cart via approval | **100%** | ✅ 100%; adversarial 8/8 paused |
| Harness | latency per turn | MCP vs. local | ✅ 4.3 s vs 4.2 s per turn (MCP transport ≈ 0.03 s); ReAct 46.3 s |

**Minimum standard** · ≥ 1 baseline + full system + ≥ 1 ablation, same fixed test subset, one results table, error analysis of ≥ 10 failures.

---

## 📓 Notebook Structure

```
Title + Team + Contributions
1.  Problem Statement
2.  Setup (pip install, imports, CONFIG, QUICK_RUN)
3.  Data (load_dataset)
4.  Baseline
5.  System
      5a. Retrieval / RAG          ← Module 3
      5b. Agents and tools        ← Module 4
      5c. Fine-tuning (optional)  ← Module 2
6.  Evaluation (results table)
7.  Error Analysis (≥ 10 failures)
8.  Demo
9.  Limitations & Ethics
10. References
```

---

## 👥 Team & Contributions

| Member | Contribution |
|---|---|
| _TBD_ | _TBD — fill in: data, RAG, agents, eval, demo, write-up_ |

> **Rule** · cite which lab each part of the notebook builds on.

---

## 🗓️ Milestones

| Milestone | Exit criteria | Status |
|---|---|---|
| M1 — Data + Retrieval | catalogue built; 4 retrieval variants scored | ✅ 2026-10-26 (`main`) |
| M2 — Need analyst | rewriting vs. raw measured | ✅ 2026-10-26 (`main`, + model upgrade) |
| M3 — Multi-agent + MCP | end-to-end run with trace | ✅ 2026-10-26 (`phase-4-mcp-multi-agent`) |
| M4 — Memory + approval | multi-turn memory; 100% approval gate | ✅ 2026-10-27 (`phase-5-memory-approval`) |
| M5 — Evaluation | results table + ≥ 10 error cases | ✅ 2026-10-27 (`phase-7-evaluation`) |
| M6 — Demo + talk | clean demo; 10-min talk rehearsed | demo ✅ · slides + outline ✅ 2026-10-27 (`TALK_OUTLINE.md`) · rehearsal ⏳ (team) |
| M7 — Colab T4 / CPU | fresh Colab run, timed | ⏳ after M5 (Phase 9) |

Phase branches are not merged into `main` yet (the team's call).

---

## ⚠️ Risks & Mitigations

| Risk | Mitigation |
|---|---|
| ~~Small model misses tool calls / structured output~~ · **VERIFIED OK 2026-10-24** | Probed against local Ollama: emits `search_products` unprompted with correct args, consumes tool results via the `tool_name` key, returns valid schema-constrained JSON for the need-analyst schema, and resolves "that one" to `add_to_cart`. Calls no tool on a bare greeting. No rule-based fallback needed |
| **Model reasoning leaks into the shopper-facing answer** | **Free text / tool calls: reasoning on** (`ChatOllama(reasoning=True)` → `additional_kwargs["reasoning_content"]`). With it off, ~1–7k chars of "Okay, let's see…" narration land in `content`, and a system prompt does **not** stop it. **Schema-constrained calls: reasoning off**: they don't leak, and they are 14–28× faster (DECISION_LOG 4.15). Thinking that still slipped into a comparator reason ("Wait, let me re-read…") is cut to the first sentence (Phase 6) |
| Comparator claims what the product text doesn't say ("no latex") | Verbatim-quote check on every pick; unsupported avoid-claims cut, with an honest "the description doesn't mention X" note (6.5); measured by the Phase 7 judge |
| Preference extractor (4B) puts items in the wrong field or invents them | Cue gating, grounding in the shopper's words, polarity and brand decided by code (5.2–5.4); avoided items filtered deterministically (6.6). Still misses double negatives (5.16) → Limitations |
| End-to-end evaluation takes hours (ReAct ≈ 2.6 h, judge ≈ 1.7 h on 200 requests) | One 200-request subset, background runs, every answer cached in `data/runs/` |
| A fresh notebook run takes hours, and Colab T4 / CPU is untested (a brief constraint) | Phase 9: `QUICK_RUN` shrinks the dev experiments, recorded comparisons loaded, timed Colab run |
| A 4B judge grading a 4B model may be lenient | 30 blind human checks by the team; judge–human agreement reported |
| The catalogue has no prices, so a budget can't be checked | Budget is remembered and the reply says it can't be checked; in Limitations |
| `ori_review` leaks into the index | Never read it into the index; assert in code |
| Retrieval metrics look bad | **Happened on raw requests** (BM25 Recall@10 0.10; ~⅓ of gold items in no top 100). Mitigated by query rewriting (Recall@10 0.45) and then the BLaIR + Qwen3-Reranker upgrade (**test Recall@10 0.548, MRR 0.360**) |
| Cart action fires without approval | `RISKY_TOOLS` + `interrupt()`; assert 100% in eval |
| Ollama tool results silently dropped | key `tool_name` (not `name`) in tool results |
| Notebook too slow to re-run | QUICK_RUN flag; cache the FAISS index |
| **BLaIR may have seen the reviews C4 queries were written from** (pretrained on Amazon Reviews 2023) | Keep the lab-model rows (bge-small) in every results table so the gain without BLaIR is visible; state it in Limitations and the talk |
| Qwen3-Reranker slow on CPU (~10 s per search, top 20) | Re-rank only the top 20; the lab re-ranker stays available as a CPU fallback (`Retriever(rerank_model=config.LAB_RERANK_MODEL)`) |
| GPU shared with other Ollama models (a 30B model filled 17 GB and caused an OOM, 2026-10-26) | Check `ollama ps` before long runs; unload other models first |

---

## 🔍 Plan Review — Open Issues (2026-10-26)

Gaps found when checking this plan against the brief (`Project/P14_Shopping_Assistant_MCP.pdf`) and the notebook template. Resolve each, then fold the fix into the relevant phase.

| # | Issue | Proposed fix | Phase |
|---|---|---|---|
| 1 | **Eval runtime not budgeted** — 800 test queries × multi-agent × `think=True` × every ablation/baseline could take days on T4/CPU | ✅ **Planned 2026-10-27** — retrieval on all 800; LLM metrics on the first 200 test requests (QUICK_RUN 100), cached, ≈ 5 h in the background (Phase 7, Risks) | 7 |
| 2 | **Chunk vs. item metrics** — chunking (300/50) returns chunks, but metrics are per `item_id`; duplicate chunks distort Recall@10 | ✅ **Decided 2026-10-26** — index one doc per item; chunking kept as a comparison row (max score per item) | 2 |
| 3 | **Writer vs. comparator overlap** — Phase 4 adds a Writer node not in the architecture diagram; the comparator already writes reasons | ✅ **Resolved 2026-10-26 (Phase 4)** — see DECISION_LOG 4.3 | 4 |
| 4 | **Tool signature** — brief specifies `search_products(query, category, k)`; plan omits `k` | ✅ **Resolved 2026-10-26 (Phase 4)** — see DECISION_LOG 4.9 | 4 |
| 5 | **Catalogue fields** — brief lists `metadata (title + description)`; plan assumes separate `title` / `description` | ✅ **Resolved 2026-10-26** — one `metadata` string; index it whole, parse title for display | 1 |
| 6 | **Single ReAct baseline not built** — listed under Baselines but no phase builds it | ✅ **Resolved 2026-10-26 (Phase 4)** — see DECISION_LOG 4.11 | 4 |
| 7 | **No preference writer** — the Store holds preferences, but no node extracts/saves them; the memory ablation depends on it | ✅ **Resolved 2026-10-27 (Phase 5)** — see DECISION_LOG 5.2–5.4 | 5 |
| 8 | **`think=True` API naming — verify** — `think` / `message.thinking` is the raw `ollama` client; `langchain_ollama.ChatOllama` likely uses `reasoning=True` → `additional_kwargs["reasoning_content"]` | ✅ **Resolved 2026-10-26** — confirmed `reasoning=True` / `reasoning_content`; structured output works both ways (see Phase 4 note) | 0 / 4 |
| 9 | **Approval % not instrumented** | ✅ **Resolved 2026-10-27 (Phase 5)** — see DECISION_LOG 5.10 (the brief's approval % is now measured from the audit log; adversarial prompts in the notebook) | 5 / 7 |
| 10 | **Milestones undated; team TBD** | Milestones now dated with status; **owners and the team table still TBD (team)** | — |
| 11 | **No research question / success target** (required by the notebook template) | ✅ **Proposed 2026-10-27** — see *Project Goal*; fixed in notebook §1 before the Phase 7 test runs (7.1) | 7 |
| 12 | **Dev split purpose unstated** | ✅ **Resolved 2026-10-26** — dev = tuning (RRF k, rerank shortlist size, prompts); test touched once | 1 / 7 |
| 13 | **Faithfulness judging** — 30 human checks need an owner; qwen3:4b judging its own output may be biased | Judge built (`Pickwise/judge.py`); Phase 7 writes a blind 30-row sheet. **Owner for the human labels: team, TBD**; bias goes in Limitations | 7 |
| 14 | **Catalogue build is slow** — streaming the 640 MB jsonl isn't reduced by QUICK_RUN | ✅ **Resolved 2026-10-26** — `data/catalogue.parquet`; re-runs load in < 1 s | 1 |
| 15 | **Expectations** — Recall@10 on a ~30k-item catalogue will be well below 1.0 | Frame expected ranges in the talk so a modest Hit@3 isn't read as failure | talk |
| 16 | **"Runs on free Colab T4 or laptop CPU" unverified** (audit 2026-10-27) — a fresh first run takes hours (reasoning check ≈ 100 min, model comparison, smoke tests) | Phase 9, after every phase is verified (agreed with the user) | 9 |
| 17 | **Re-ranker differs from the brief's example** — Qwen3-Reranker-0.6B is an LLM-based re-ranker, not ms-marco MiniLM | Justified on dev (DECISION_LOG R.1); the lab re-ranker rows stay in every table; **say it in the talk** | talk |

---

## ✅ Mark-Losing Checklist

- [x] Data downloaded with `load_dataset` at the top
- [x] Fixed seed + subset size stated
- [x] `ori_review` never in the index (asserted in `check_no_leakage`)
- [x] All required metrics reported (Recall@10, MRR, nDCG@10, Hit@3, faithfulness, approval %, latency MCP vs local)
- [ ] All notebook sections present: ✅ all filled; **team table still empty (team)**
- [x] ≥ 10 failure cases in error analysis (all 124 misses categorised; annotated examples per category)
- [x] `add_to_cart` 100% through approval (Phase 5: 100% approval share, adversarial 8/8 paused; re-checked in Phase 7)
- [x] ≤ 4B params, no paid APIs (qwen3:4b · BLaIR 0.1B · Qwen3-Reranker 0.6B · all local)
- [x] Each part cites the lab it builds on ("Builds on" lines in §2–§5; §8 cites Module 4 Section E); §6–§7 to add in Phase 7
- [ ] Runs on free Colab T4 / laptop CPU (Phase 9)

---

## 📚 References — which lab each part builds on

| Project part | Lab |
|---|---|
| Need analyst (query rewriting) | Module 2 · Structured_Generation_Lab |
| Retrieval / RAG | Module 3 · RAG_Pipeline_Lab |
| Multi-agent supervisor + specialists | Module 4 · Multi_Agent_System_End_to_End |
| MCP server + tools | Module 4 · MCP_Server_and_Tools_Lab |
| Memory + approval | Module 4 · Multi_Agent_System_End_to_End (D & E) |
| Web app (stretch; Gradio) | Module 4 · Agentic_app |
| Spec | P14_Shopping_Assistant_MCP.pdf |

---

> [!tip] How to use this file
> Copy everything above into a new Notion page (or import the `.md`). Each `[ ]` becomes a To-do toggle, each table stays a table, and the callouts render as Notion callouts.
