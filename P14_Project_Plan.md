# 🛒 P14 — Conversational Shopping Assistant

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
| **Catalogue** | `sampled_item_metadata_1M.jsonl` — 1.06M items (item_id, category, title, description) |
| **Sample** | 1,000 queries, `shuffle(seed=42)` — state subset size in the notebook |
| **Distractors** | ~30k random items (`rng.random() < 0.03`) |
| **Split** | 200 dev / 800 test (QUICK_RUN: 100) |
| **Index text** | `title + description` only — **never `ori_review`** |

> [!warning] Rule
> The notebook must download data with `load_dataset(...)` at the **top**, so anyone can re-run it. If you use a subset, select it with a **fixed seed** and **state its size**. The RAG index must be built from this downloaded data, not pasted text.

---

## 🤖 Models

| Purpose | Model | Notes |
|---|---|---|
| Agents (tool calling, structured output) | **qwen3:4b** | primary; `qwen3:1.7b` fallback. **Must run with `think=True`** — with `think=False` reasoning leaks into the answer (see Risks) |
| Embeddings | `BAAI/bge-small-en-v1.5` or `all-MiniLM-L6-v2` | 384-dim, CPU-friendly |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | cross-encoder, top of the hybrid pipeline |
| Optional product embedder | `hyp1231/blair-roberta-base` | stretch comparison |
| Optional LoRA need analyst | `Qwen2.5-0.5B` | stretch; 2–5k C4 queries outside test subset |

**Compute** · free Colab T4 or laptop CPU ≤ 4B params · no paid APIs

---

## 🚧 Build Plan

Each phase lists what it **builds on** (the lab code to adapt) and its **done-when**.

### Phase 0 — Setup & Scaffold
- [ ] `pip install` the full stack (langgraph, langchain, langchain-ollama, langchain-mcp-adapters, mcp, faiss-cpu, rank-bm25, sentence-transformers, datasets)
- [ ] `CONFIG` block (model, paths, seed, QUICK_RUN flag)
- [ ] Verify Ollama + qwen3:4b
- **Builds on** · —
- **Done when** · clean notebook skeleton runs top-to-bottom

### Phase 1 — Data & Catalogue
- [ ] `load_dataset` at the top (test split + metadata jsonl, streaming)
- [ ] 1,000 queries (seed 42) + gold items + ~30k distractors
- [ ] Build `catalogue` dict: `item_id → {category, title, description}`
- [ ] 200 dev / 800 test split
- **Builds on** · spec §Data
- **Done when** · reproducible catalogue; no `ori_review` in the index

### Phase 2 — Retrieval (RAG)  →  *Module 3*
- [ ] Chunking (`RecursiveCharacterTextSplitter`, size 300 / overlap 50, contextual headers)
- [ ] **Dense** — FAISS + embeddings
- [ ] **Sparse** — BM25
- [ ] **Hybrid** — Reciprocal Rank Fusion
- [ ] **Re-rank** — cross-encoder over hybrid shortlist
- [ ] Evaluate: **Recall@10, MRR, nDCG@10** vs gold `item_id`
- [ ] Compare: BM25 raw · dense raw · hybrid · hybrid+rerank
- **Builds on** · Module 3 · RAG_Pipeline_Lab
- **Done when** · results table for all 4 retrieval variants

### Phase 3 — Need Analyst  →  *Module 2*
- [ ] Structured query rewriting: long request → `{product_type, must_haves, category}`
- [ ] Schema-constrained output (`with_structured_output`)
- [ ] Measure: does rewriting beat searching with the raw request?
- **Builds on** · Module 2 · Structured_Generation_Lab (constrained decoding, tool calling)
- **Done when** · retrieval metrics with vs. without rewriting

### Phase 4 — MCP Server + Multi-Agent  →  *Module 4*
- [ ] **MCP server**: `search_products(query, category)`, `get_product(id)`, `list_categories()`
- [ ] **Local `@tool`** versions of the same (for latency comparison)
- [ ] **LangGraph supervisor** (LLM-as-router, structured output)
- [ ] **Specialists**: searcher, comparator (each a ReAct sub-graph with its own context)
- [ ] **Writer** node: final answer, reasons from product text only
- [ ] **Every LLM call with `think=True`** — read `message.content` as the shopper-facing answer and `message.thinking` as the reasoning trace; `think=False` leaks reasoning into the answer (see Risks)
- [ ] **Tune the writer prompt for prose** — with `think=True` the clean answers came back as terse `Title: "quote"` lists, dropping connective sentences
- [ ] **Separate model time from transport time** in the latency comparison, or the MCP-vs-local ablation measures the model, not the transport
- **Builds on** · Module 4 · Multi_Agent_System_End_to_End + MCP_Server_and_Tools_Lab
- **Done when** · end-to-end run with a printed trace of every agent step + tool call

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
- [ ] BLaIR embedder vs. general embedders
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
| **Model reasoning leaks into the shopper-facing answer** | **Run every LLM call with `think=True`.** With `think=False` Ollama returns no `thinking` field and dumps ~5–7k chars of "Okay, let's see…" narration into `content`; a system prompt telling it to answer directly does **not** stop it. With `think=True` reasoning moves to `message.thinking` and `content` is clean — 3/3 samples, 227–234 chars, zero leaked markers, every line quoting the product text. Read `content` as the answer, `thinking` as the trace |
| `ori_review` leaks into the index | Never read it into the index; assert in code |
| Retrieval metrics look bad | Start with hybrid+rerank (Module 3 shows it beats raw BM25/dense) |
| Cart action fires without approval | `RISKY_TOOLS` + `interrupt()`; assert 100% in eval |
| Ollama tool results silently dropped | key `tool_name` (not `name`) in tool results |
| Notebook too slow to re-run | QUICK_RUN flag; cache the FAISS index |

---

## ✅ Mark-Losing Checklist

- [ ] Data downloaded with `load_dataset` at the top
- [ ] Fixed seed + subset size stated
- [ ] `ori_review` never in the index
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
