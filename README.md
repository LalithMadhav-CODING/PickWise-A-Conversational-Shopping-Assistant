# 🛒 Pickwise

**A conversational shopping assistant that understands situations, not keywords.** Pickwise turns a long, indirect request into catalogue searches, recommends the top 3 products with reasons quoted from the product text, remembers each shopper's preferences, and **never adds anything to the cart without asking**.

> *"Something to keep my kids busy at bath time that can get messy."*: keyword search fails on requests like this. Pickwise is built to handle them.

Built with **LangGraph** (multi-agent supervisor), **MCP** (Model Context Protocol), **hybrid RAG** (BM25 + dense + re-ranker) and **human-in-the-loop approval**. It runs entirely on small open models via **Ollama**, with no paid APIs.

Course group project **P14** (UG3/UG4, Modules 1–4).

---

## How it works

```
shopper
   │
   ▼
┌─────────────┐
│  Supervisor │  routes each turn to ONE specialist
└──────┬──────┘
       ├──────────────┬───────────────┬──────────────┐
       ▼              ▼               ▼              ▼
  Need analyst     Searcher       Comparator       Cart
  long request →   hybrid RAG     top 3 + reasons  add_to_cart?
  structured       over the       quoted from the  ─► interrupt()
  query            catalogue      product text     ─► shopper approves
                      │
              ┌───────┴────────┐
              ▼                ▼
         MCP server       local @tool      ← same tools, two transports
                                              (compared for latency)

Memory: checkpointer (this chat) + Store (shopper preferences: brands, budget, dislikes)
```

| Component | Choice |
|---|---|
| Agents | `qwen3:4b` via Ollama (`reasoning=True` for free text and tool calls; `reasoning=False` for structured output) |
| Embeddings | `BAAI/bge-small-en-v1.5` (with the bge query instruction) |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Need analyst | long request → `{product_type, must_haves, category}` via schema-constrained output |
| Retrieval | BM25 + FAISS dense → Reciprocal Rank Fusion → cross-encoder (top 50); one document per product; query = *rewrite + the shopper's own words* |
| Orchestration | LangGraph supervisor + specialists, checkpointer + Store, `interrupt()` before `add_to_cart` |
| Tools | MCP server (`search_products`, `get_product`, `list_categories`) and the same functions as local `@tool`s |

## Data

[McAuley-Lab/Amazon-C4](https://huggingface.co/datasets/McAuley-Lab/Amazon-C4) on Hugging Face (public):

- **Queries**: a fixed sample of 1,000 long shopping requests from the `test` split (seed 42), each with its gold product → **200 dev** (tuning) / **800 test** (reported once)
- **Catalogue**: the 998 gold products + 31,616 random distractors from the 1.06M-item metadata file → **32,614 products**
- The index uses product text only. `ori_review` (the review each query was written from) is **never** indexed, and the notebook asserts this.

## Results so far

Retrieval on the fixed **800-query test subset** (one gold product per request, catalogue of 32,614 products):

| Condition | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|
| BM25 · raw request (baseline) | 0.098 | 0.062 | 0.068 |
| Dense · raw request (baseline) | 0.409 | 0.248 | 0.279 |
| Hybrid (RRF) · raw request | 0.315 | 0.158 | 0.186 |
| Hybrid + re-rank · raw request | 0.405 | 0.232 | 0.267 |
| Hybrid + re-rank · raw request · chunked index | 0.399 | 0.226 | 0.261 |
| BM25 · rewrite + request | 0.234 | 0.140 | 0.158 |
| Hybrid + re-rank · rewrite only | 0.442 | 0.257 | 0.294 |
| **Hybrid + re-rank · rewrite + request (system retriever)** | **0.452** | **0.276** | **0.312** |

- **Situational requests defeat keyword search**: BM25 finds the gold product in the top 10 for about 1 request in 10.
- **The need analyst's rewrite helps the full pipeline.** Against the best baseline (dense on the raw request), the system retriever adds +4.4 points Recall@10 (95% CI [1.9, 6.9]) and +0.027 MRR ([0.007, 0.048]) in a paired bootstrap.
- Using the predicted category as a filter halves recall (category accuracy is about 43–52%), so it isn't applied automatically.
- The need analyst runs with Qwen3's reasoning **off**. On dev it's 28× faster than with reasoning on (1.1 s vs 30.7 s per request), with no measurable retrieval loss.

All choices were made on the 200 dev queries; test was scored once. Details: notebook §4–5 and [`CURRENT_PROJECT_STATUS.md`](CURRENT_PROJECT_STATUS.md).

## Evaluation plan

| Layer | Metric |
|---|---|
| Retrieval | Recall@10, MRR, nDCG@10 vs. the gold item |
| End-to-end | Hit@3: is the gold item among the 3 recommendations? |
| Faithfulness | every reason supported by the product text (LLM-as-judge + 30 human checks) |
| Harness | % of `add_to_cart` through approval (**must be 100%**); latency per turn, MCP vs. local |

Baselines: BM25 and dense retrieval on the raw request, and a single ReAct agent. Ablations: query rewriting, re-ranker, MCP vs. in-process tools, and memory on/off.

## Status

| Phase | |
|---|---|
| 0 — Setup & scaffold | ✅ |
| 1 — Data & catalogue | ✅ |
| 2 — Retrieval (RAG) | ✅ |
| 3 — Need analyst | ✅ |
| 4 — MCP server + multi-agent | ⏳ next |
| 5 — Memory + human approval | ⚪ |
| 6 — Demo | ⚪ |
| 7 — Evaluation | ⚪ |
| 8 — Stretch (Streamlit app, BLaIR embedder, LoRA need analyst) | ⚪ |

Detailed progress, decisions and numbers: [`CURRENT_PROJECT_STATUS.md`](CURRENT_PROJECT_STATUS.md) · Full plan: [`P14_Project_Plan.md`](P14_Project_Plan.md)

## Quick start

Requirements: Python ≥ 3.10, [Ollama](https://ollama.com), about 4 GB of disk, and internet on the first run.

```bash
# 1. Model
ollama pull qwen3:4b

# 2. Python environment
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 3. Run the notebook end to end
.venv/bin/jupyter execute --inplace P14_Shopping_Assistant.ipynb
#    or open it in Jupyter / VS Code with the .venv kernel
```

The first run streams the 640 MB catalogue file (≈ 3 min), embeds the catalogue and analyses 1,000 requests with the LLM (≈ 15 min on a GPU, plus about an hour for the reasoning-on comparison). Everything is cached in `data/`, so later runs only re-score (a few minutes). Set `PICKWISE_QUICK_RUN=1` for a 100-query test subset.

It also runs on a free Colab T4: clone the repo, then install and start Ollama in the runtime (see the notebook's Setup section).

### Settings

All settings live in [`Pickwise/config.py`](Pickwise/config.py). Overrides:

| Variable | Default |
|---|---|
| `PICKWISE_MODEL` | `qwen3:4b` |
| `PICKWISE_QUICK_RUN` | `0` |
| `PICKWISE_DATA_DIR` | `./data` |
| `OLLAMA_HOST` | `http://localhost:11434` |

## Repository layout

```
P14_Shopping_Assistant.ipynb   the deliverable notebook (problem → data → system → evaluation → demo)
Pickwise/                      reusable code imported by the notebook, MCP server and app
  config.py                    all settings
  data.py                      queries, cached catalogue, dev/test split, leakage checks
  llm.py                       Ollama chat model + reasoning trace
  retrieval.py                 BM25 + FAISS → RRF → cross-encoder; category filter; chunked mode
  metrics.py                   Recall@k, MRR, nDCG@k, paired bootstrap
  need.py                      need analyst: schema, prompt, cached batch analysis, search query
requirements.txt
P14_Project_Plan.md            build plan, risks, open issues
CURRENT_PROJECT_STATUS.md      live progress log
```

The course brief (`Project/`) and labs (`modules/`) are kept locally and are not part of this repository.

## Built on the course labs

| Part | Lab |
|---|---|
| Need analyst (query rewriting) | Module 2 · Structured_Generation_Lab |
| Retrieval / RAG | Module 3 · RAG_Pipeline_Lab |
| Supervisor + specialists, memory, approval | Module 4 · Multi_Agent_System_End_to_End |
| MCP server + tools | Module 4 · MCP_Server_and_Tools_Lab |
| Config / LLM wrapper / app structure | Module 4 · Agentic_app |

## Team

| Member | Contribution |
|---|---|
| _TBD_ | _TBD_ |

## References

- Hou et al., 2024. *Bridging Language and Items for Retrieval and Recommendation* (Amazon-C4, BLaIR).
- LangGraph, LangChain, Model Context Protocol, FAISS, rank-bm25, sentence-transformers, Hugging Face Datasets.
