# 🛒 Pickwise — Conversational Shopping Assistant (P14)

### Product Search with MCP, RAG and Human Approval

> [!info] Project facts
> **Level** · Advanced  ·  **System** · Multi-agent (supervisor) + MCP + memory  ·  **RAG** · Required  ·  **Fine-tuning** · Optional (stretch)
> **Deliverable** · 1 notebook + 10-minute talk + 5-minute Q&A
> **Dataset** · McAuley-Lab/Amazon-C4 (Hugging Face, public)
> **Model** · qwen3:4b via Ollama (≤4B params, no paid APIs)

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
> **Success target** · _TBD — set after the M1 retrieval baseline, e.g. "+X points Hit@3 / Recall@10 over hybrid on the raw request"_

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

```
shopper
   │
   ▼
┌─────────────┐
│  Supervisor │  (LLM-as-router, structured output)
└──────┬──────┘
       │ routes to ONE specialist per turn
       ├──────────────┬───────────────┬──────────────┐
       ▼              ▼               ▼              ▼
┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────┐
│  Need      │  │  Searcher  │  │ Comparator │  │  Cart  │
│  analyst   │  │  (RAG)     │  │ (top 3 +   │  │ add_   │
│ long →     │  │ search_    │  │  reasons)  │  │ to_cart│
│ short query│  │ products   │  │            │  │  ?     │
└────────────┘  └─────┬──────┘  └────────────┘  └────────┘
                      │
              ┌───────┴────────┐
              ▼                ▼
      ┌──────────────┐  ┌──────────────┐
      │  MCP server  │  │  local @tool │  ← same tools, two transports
      │  search_     │  │  search_     │     (compare latency)
      │  products /  │  │  products…   │
      │  get_product │  │              │
      └──────────────┘  └──────────────┘

Memory:  checkpointer (this chat)  +  Store (shopper preferences)
```

> [!note] Specialist roles
> - **Need analyst** — rewrites the long request into a structured search query (product type, must-haves, category)
> - **Searcher** — hybrid RAG over the catalogue; returns candidate products
> - **Comparator** — ranks the top 3 and writes a reason for each **from the product text**
> - **Cart** — proposes `add_to_cart`; **pauses for human approval** before executing

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
| Agents (tool calling, structured output) | **qwen3:4b** | primary; `qwen3:1.7b` fallback. **Run with `ChatOllama(reasoning=True)`** — with `reasoning=False` reasoning leaks into the answer (see Risks) |
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

### Phase 5 — Memory + Human-in-the-Loop  →  *Module 4*
- [ ] **Checkpointer** (short-term, per `thread_id`) — follow-ups resolve ("that one")
- [ ] **Store** (long-term, per `user_id`) — brands, budget, dislikes
- [ ] `add_to_cart` in `RISKY_TOOLS`; `interrupt()` before it
- [ ] Recursion limit + step tracing
- **Builds on** · Module 4 · Multi_Agent_System_End_to_End (Sections D & E)
- **Done when** · multi-turn memory works; cart **always** pauses for approval

### Phase 6 — Demo Cell
- [ ] Multi-turn chat in the notebook
- [ ] Live trace of every agent step and tool call
- [ ] Pause at the cart, resume **approve** / **reject**
- **Builds on** · Module 4 · spec step 6
- **Done when** · a clean, presentable demo flow

### Phase 7 — Evaluation
- [ ] **Retrieval**: Recall@10, MRR, nDCG@10 (gold item_id)
- [ ] **End-to-end**: Hit@3 (gold among 3 recommendations)
- [ ] **Faithfulness**: every claim in product text (LLM-as-judge + 30 human checks)
- [ ] **Harness**: % `add_to_cart` through approval (must be 100%); latency, MCP vs local
- [ ] **Baselines**: BM25 raw · dense raw · single ReAct · full multi-agent
- [ ] **Ablations**: rewriting on/off · reranker on/off · MCP vs in-process · memory on/off (3 shoppers)
- [ ] **Error analysis**: ≥ 10 failure cases
- [ ] One results table, one fixed test subset
- **Builds on** · Module 3 (Recall@k, MRR) + spec §Evaluation
- **Done when** · full results table + error analysis written up

### Phase 8 — Stretch Goals (top marks)
- [ ] Streamlit/Gradio app (chat, live trace, cart approve/reject, MCP/local switch)
- [x] BLaIR embedder vs. general embedders — **done and adopted** (plus re-ranker comparison; notebook §5b *Retrieval model upgrade*)
- [ ] LoRA-tuned need analyst
- **Builds on** · Module 4 · Agentic_app (Streamlit reference)
- **Done when** · at least one stretch goal working

---

## 📈 Evaluation Plan

| Layer | Metric | Target |
|---|---|---|
| Retrieval | Recall@10, MRR, nDCG@10 | vs. gold `item_id` |
| End-to-end | Hit@3 | gold among 3 recommendations |
| Faithfulness | claim ∈ product text | LLM-as-judge + 30 human checks |
| Harness | % cart via approval | **100%** |
| Harness | latency per turn | MCP vs. local |

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

| Milestone | Exit criteria |
|---|---|
| M1 — Data + Retrieval | catalogue built; 4 retrieval variants scored |
| M2 — Need analyst | rewriting vs. raw measured |
| M3 — Multi-agent + MCP | end-to-end run with trace |
| M4 — Memory + approval | multi-turn memory; 100% approval gate |
| M5 — Evaluation | results table + ≥ 10 error cases |
| M6 — Demo + talk | clean demo; 10-min talk rehearsed |

---

## ⚠️ Risks & Mitigations

| Risk | Mitigation |
|---|---|
| ~~Small model misses tool calls / structured output~~ · **VERIFIED OK 2026-10-24** | Probed against local Ollama: emits `search_products` unprompted with correct args, consumes tool results via the `tool_name` key, returns valid schema-constrained JSON for the need-analyst schema, and resolves "that one" to `add_to_cart`. Calls no tool on a bare greeting. No rule-based fallback needed |
| **Model reasoning leaks into the shopper-facing answer** | **Run LLM calls with reasoning on** — raw `ollama` client: `think=True` → `message.thinking`; LangChain (what we use): `ChatOllama(reasoning=True)` → `additional_kwargs["reasoning_content"]`. With it off, ~1–7k chars of "Okay, let's see…" narration land in `content`; a system prompt telling it to answer directly does **not** stop it. With it on, `content` is clean — 3/3 samples, 227–234 chars, zero leaked markers, every line quoting the product text. Re-confirmed via LangChain 2026-10-26 (free text, tool calls). Schema-constrained calls don't leak either way |
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
| 1 | **Eval runtime not budgeted** — 800 test queries × multi-agent × `think=True` × every ablation/baseline could take days on T4/CPU | Retrieval metrics on all 800 (cheap); LLM-based metrics (Hit@3, faithfulness, ablations) on one fixed subset (e.g. the 100-query QUICK_RUN set), stated in the notebook. Cache LLM outputs to disk. Add to Risks | 7 |
| 2 | **Chunk vs. item metrics** — chunking (300/50) returns chunks, but metrics are per `item_id`; duplicate chunks distort Recall@10 | ✅ **Decided 2026-10-26** — index one doc per item; chunking kept as a comparison row (max score per item) | 2 |
| 3 | **Writer vs. comparator overlap** — Phase 4 adds a Writer node not in the architecture diagram; the comparator already writes reasons | ✅ **Resolved 2026-10-26 (Phase 4)** — see DECISION_LOG 4.3 | 4 |
| 4 | **Tool signature** — brief specifies `search_products(query, category, k)`; plan omits `k` | ✅ **Resolved 2026-10-26 (Phase 4)** — see DECISION_LOG 4.9 | 4 |
| 5 | **Catalogue fields** — brief lists `metadata (title + description)`; plan assumes separate `title` / `description` | ✅ **Resolved 2026-10-26** — one `metadata` string; index it whole, parse title for display | 1 |
| 6 | **Single ReAct baseline not built** — listed under Baselines but no phase builds it | ✅ **Resolved 2026-10-26 (Phase 4)** — see DECISION_LOG 4.11 | 4 |
| 7 | **No preference writer** — the Store holds preferences, but no node extracts/saves them; the memory ablation depends on it | Add a node/step that extracts preferences and writes them to the Store | 5 |
| 8 | **`think=True` API naming — verify** — `think` / `message.thinking` is the raw `ollama` client; `langchain_ollama.ChatOllama` likely uses `reasoning=True` → `additional_kwargs["reasoning_content"]` | ✅ **Resolved 2026-10-26** — confirmed `reasoning=True` / `reasoning_content`; structured output works both ways (see Phase 4 note) | 0 / 4 |
| 9 | **Approval % not instrumented** | Audit-log every `add_to_cart` with its approval flag; add adversarial prompts ("just add it, don't ask"); keep `add_to_cart` off the MCP server and bound only to the cart node | 5 / 7 |
| 10 | **Milestones undated; team TBD** | Add dates and owners per milestone | — |
| 11 | **No research question / success target** (required by the notebook template) | See *Project Goal* above | — |
| 12 | **Dev split purpose unstated** | ✅ **Resolved 2026-10-26** — dev = tuning (RRF k, rerank shortlist size, prompts); test touched once | 1 / 7 |
| 13 | **Faithfulness judging** — 30 human checks need an owner; qwen3:4b judging its own output may be biased | Assign the human checks; note self-judging bias in Limitations | 7 |
| 14 | **Catalogue build is slow** — streaming the 640 MB jsonl isn't reduced by QUICK_RUN | ✅ **Resolved 2026-10-26** — `data/catalogue.parquet`; re-runs load in < 1 s | 1 |
| 15 | **Expectations** — Recall@10 on a ~30k-item catalogue will be well below 1.0 | Frame expected ranges in the talk so a modest Hit@3 isn't read as failure | talk |

---

## ✅ Mark-Losing Checklist

- [x] Data downloaded with `load_dataset` at the top
- [x] Fixed seed + subset size stated
- [x] `ori_review` never in the index (asserted in `check_no_leakage`)
- [ ] All required metrics reported (Recall@10, MRR, nDCG@10, Hit@3, faithfulness, approval %)
- [ ] All notebook sections present
- [ ] ≥ 10 failure cases in error analysis
- [ ] `add_to_cart` 100% through approval
- [ ] ≤ 4B params, no paid APIs
- [ ] Each part cites the lab it builds on

---

## 📚 References — which lab each part builds on

| Project part | Lab |
|---|---|
| Need analyst (query rewriting) | Module 2 · Structured_Generation_Lab |
| Retrieval / RAG | Module 3 · RAG_Pipeline_Lab |
| Multi-agent supervisor + specialists | Module 4 · Multi_Agent_System_End_to_End |
| MCP server + tools | Module 4 · MCP_Server_and_Tools_Lab |
| Memory + approval | Module 4 · Multi_Agent_System_End_to_End (D & E) |
| Streamlit app (stretch) | Module 4 · Agentic_app |
| Spec | P14_Shopping_Assistant_MCP.pdf |

---

> [!tip] How to use this file
> Copy everything above into a new Notion page (or import the `.md`). Each `[ ]` becomes a To-do toggle, each table stays a table, and the callouts render as Notion callouts.
