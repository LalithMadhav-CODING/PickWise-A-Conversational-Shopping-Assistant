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
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Retrieval | BM25 + FAISS dense → Reciprocal Rank Fusion → cross-encoder; one document per product |
| Orchestration | LangGraph supervisor + specialists, checkpointer + Store, `interrupt()` before `add_to_cart` |
| Tools | MCP server (`search_products`, `get_product`, `list_categories`) and the same functions as local `@tool`s |

## Data

[McAuley-Lab/Amazon-C4](https://huggingface.co/datasets/McAuley-Lab/Amazon-C4) on Hugging Face (public):

- **Queries**: a fixed sample of 1,000 long shopping requests from the `test` split (seed 42), each with its gold product → **200 dev** (tuning) / **800 test** (reported once)
- **Catalogue**: the 998 gold products + 31,616 random distractors from the 1.06M-item metadata file → **32,614 products**
- The index uses product text only. `ori_review` (the review each query was written from) is **never** indexed, and the notebook asserts this.

## Evaluation (planned)

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
| 2 — Retrieval (RAG) | ⏳ next |
| 3 — Need analyst | ⚪ |
| 4 — MCP server + multi-agent | ⚪ |
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

The first run streams the 640 MB catalogue file (≈ 3 min) and caches the filtered catalogue in `data/`. Later runs take seconds. Set `PICKWISE_QUICK_RUN=1` for a 100-query test subset.

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
