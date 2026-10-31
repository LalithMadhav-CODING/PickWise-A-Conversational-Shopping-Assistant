# 🛒 Pickwise

**A conversational shopping assistant that understands situations, not keywords.** Pickwise turns a long, indirect request into catalogue searches, recommends the top 3 products with reasons quoted from the product text, remembers each shopper's preferences, and **never adds anything to the cart without asking**.

> *"Something to keep my kids busy at bath time that can get messy."*: keyword search fails on requests like this. Pickwise is built to handle them.

Built with **LangGraph** (multi-agent supervisor), **MCP** (Model Context Protocol), **hybrid RAG** (BM25 + dense + re-ranker) and **human-in-the-loop approval**. It runs entirely on small open models via **Ollama**, with no paid APIs.

Course group project **P14** (UG3/UG4, Modules 1–4).

---

## How it works

```
shopper ──► remember ──► supervisor ──► need analyst ──► searcher ──► comparator ──► END   a new request
            (Store)          │                           (search_products) (top 3 + reasons
                             │                                │             quoted from the text)
                             ├─────────────────────────────────────────► comparator ──► END   "which of these…?"
                             ├──► details (get_product) ─────────────────────────────► END   "more about the third one"
                             └──► cart ──► approval ──► add_to_cart ─────────────────► END   "add the second one"
                                           interrupt(): the shopper says yes / no
                                                                │
                                                  ┌─────────────┴─────────────┐
                                                  ▼                           ▼
                                           MCP server (HTTP)            local tools      ← same functions,
                                                                                            two transports
Memory: checkpointer (this conversation) + Store (shopper preferences: brands, materials, budget)
```

| Component | Choice |
|---|---|
| Agents | `qwen3:4b` via Ollama (`reasoning=True` for free text and tool calls; `reasoning=False` for structured output) |
| Embeddings | **`hyp1231/blair-roberta-base`** (BLaIR, 0.1B, product-specific); lab baseline: `BAAI/bge-small-en-v1.5` |
| Re-ranker | **`Qwen/Qwen3-Reranker-0.6B`** on the top 20; lab baseline: `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Need analyst | long request → `{product_type, must_haves, category}` via schema-constrained output |
| Retrieval | BM25 + FAISS dense → Reciprocal Rank Fusion → re-ranker; one document per product; query = *rewrite + the shopper's own words* |
| Orchestration | LangGraph supervisor + specialists, checkpointer + Store, `interrupt()` before `add_to_cart` |
| Tools | MCP server (`search_products`, `get_product`, `list_categories`) and the same functions as local `@tool`s |

## Data

[McAuley-Lab/Amazon-C4](https://huggingface.co/datasets/McAuley-Lab/Amazon-C4) on Hugging Face (public):

- **Queries**: a fixed sample of 1,000 long shopping requests from the `test` split (seed 42), each with its gold product → **200 dev** (tuning) / **800 test** (reported once)
- **Catalogue**: the 998 gold products + 31,616 random distractors from the 1.06M-item metadata file → **32,614 products**
- The index uses product text only. `ori_review` (the review each query was written from) is **never** indexed, and the notebook asserts this.

## Results

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
| Hybrid + re-rank · rewrite + request (lab models) | 0.452 | 0.276 | 0.312 |
| *Model upgrade (stretch goal):* | | | |
| bge-small · hybrid + **Qwen3** re-rank · rewrite + request | 0.430 | 0.303 | 0.329 |
| **BLaIR** · hybrid + ms-marco re-rank · rewrite + request | 0.504 | 0.300 | 0.339 |
| BLaIR · dense only · raw request | 0.575 | 0.356 | 0.401 |
| BLaIR · hybrid + Qwen3 re-rank · raw request | 0.568 | 0.360 | 0.406 |
| **BLaIR · hybrid + Qwen3 re-rank · rewrite + request (system retriever)** | **0.548** | **0.360** | **0.400** |

- **Situational requests defeat keyword search**: BM25 finds the gold product in the top 10 for about 1 request in 10.
- **With the lab models, the need analyst's rewrite helps the full pipeline:** +4.4 points Recall@10 (95% CI [1.9, 6.9]) and +0.027 MRR ([0.007, 0.048]) over the best baseline (dense on the raw request), in a paired bootstrap.
- **The model upgrade is the biggest gain.** BLaIR (a product-specific embedder) plus Qwen3-Reranker add **+9.5 points Recall@10 [6.6, 12.4] and +0.085 MRR [0.063, 0.106]** over the same pipeline with the lab models, and **+14 points Recall@10** over the best baseline.
- **With BLaIR, the rest of the pipeline adds nothing measurable on test.** BLaIR on the raw request alone is level with the system (MRR +0.004 [−0.018, 0.027]). The system keeps the brief's hybrid + re-ranker, which was chosen on dev, and BLaIR-only is a fast fallback for CPU. BLaIR's pretraining data may overlap with the reviews the queries came from (see Limitations).
- Using the predicted category as a filter halves recall (category accuracy is about 43–52%), so it isn't applied automatically.
- The need analyst runs with Qwen3's reasoning **off**. On dev it's 28× faster than with reasoning on (1.1 s vs 30.7 s per request), with no measurable retrieval loss.

**End to end (Phase 7, the first 200 test requests, each answered from scratch with a fresh shopper):**

| Condition | Recall@10 | Hit@3 | Faithful (judge) | s / turn |
|---|---|---|---|---|
| BM25 · raw request | 0.070 | 0.040 | — | — |
| Dense (bge-small) · raw request | 0.360 | 0.235 | — | — |
| Dense (BLaIR) · raw request | 0.540 | 0.350 | — | — |
| Single ReAct agent (MCP) | 0.460 | 0.325 | 0.561 | 46.3 |
| **Full system (MCP)** | **0.495** | **0.380** | **0.590** | **4.3** |
| − rewriting | 0.535 | 0.370 | — | 3.6 |
| − re-ranker | 0.390 | 0.285 | — | 3.8 |
| − comparator (the search's own top 3) | 0.495 | 0.385 | — | — |
| in-process tools instead of MCP | 0.490 | 0.380 | — | 4.2 |

Against the success target written down before these runs (`DECISION_LOG.md` 7.1):
- **Hit@3 ≥ 0.40: missed narrowly**, 0.380 with a 95% CI of [0.31, 0.45].
- **Beat the single ReAct agent: not met.** It is ahead by 5.5 points, but the paired CI is [0.0, 10.5] (McNemar p = 0.07). It is also 11× faster.
- **≥ 90% judged supported: missed.** 59% of recommendations are fully supported, and 91% are supported or partly. The quotes are verbatim (97%), but the sentence around a quote often states the shopper's wish as a product fact. **30 blind human labels do not confirm the judge:** they agree on 43% (κ −0.15). The human rated 80% fully supported, the judge 57% of the same 30, so read 59% as a strict, rough estimate (notebook §6, `DECISION_LOG.md` 9.16).
- **100% of executed adds approved: met.**

Ablations: the re-ranker is the only part whose removal clearly hurts (−9.5 points Hit@3, CI [4.5, 14.5]). Rewriting, the comparator and MCP make no measurable difference to Hit@3. MCP adds about 0.03 s per turn. Memory on/off with 3 returning shoppers: avoided-brand and avoided-material picks fall from 2 and 3 to 0, and preferred-brand picks rise from 2 to 3.

Error analysis: 100 of the 124 misses happen before the comparator. 57 were never retrieved and 43 were ranked 11–100. Two thirds of the gold products have only a title (43% of the catalogue).

**Memory and approval (Phase 5):** Pickwise remembers each shopper's lasting preferences across conversations (e.g. *avoid latex, never Munchkin*). Products naming an avoided brand or material are filtered out of the results (never handed back, even when few remain), and **every add-to-cart pauses for the shopper's yes**: 8 out of 8 attempts to skip approval paused, and nothing was added without approval.

**Try it (Phase 6):** in the notebook's §8, `await ask("I'm vegan, so no leather please. I need a slim wallet.")` shows every agent step and tool call. At the cart it pauses and asks *approve / reject?*. Set `PICKWISE_INTERACTIVE=1` to type answers, or use `await chat()`.

**Web app (Phase 8):** `.venv/bin/python app.py` opens Pickwise at http://127.0.0.1:7860 (`--share` adds a temporary public link). It runs the same graph as the notebook: chat on the left, the agent trace streaming step by step on the right, **Approve / Reject** buttons when a cart action pauses, what Pickwise remembers about you, your cart, the approval log, and a switch between MCP and local tools.

All choices were made on the 200 dev queries; test was scored once. Details: notebook §4–5b and [`CURRENT_PROJECT_STATUS.md`](CURRENT_PROJECT_STATUS.md).

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
| 4 — MCP server + multi-agent | ✅ (branch `phase-4-mcp-multi-agent`) |
| 5 — Memory + human approval | ✅ (branch `phase-5-memory-approval`) |
| 6 — Demo | ✅ (branch `phase-6-demo`) |
| 7 — Evaluation | ✅ (branch `phase-7-evaluation`) |
| 8 — Stretch: web app ✅ (branch `phase-8-app`), BLaIR embedder ✅, LoRA need analyst ⚪ | 🟡 |
| 9 — Colab T4 / CPU | ✅ (branch `phase-9-colab`): fresh clone verified on GPU and CPU-only; a quick run completed on a free Colab T4 (2026-10-30; time not recorded) |

Detailed progress, decisions and numbers: [`CURRENT_PROJECT_STATUS.md`](CURRENT_PROJECT_STATUS.md) · Full plan: [`P14_Project_Plan.md`](P14_Project_Plan.md)

## Quick start

| Where | How | Time (quick run) |
|---|---|---|
| **Free Colab T4** | Upload `P14_Shopping_Assistant_Colab.ipynb` (*File → Upload notebook*), Runtime → T4 GPU, *Run all*. It clones this repository (`main`), installs the packages and Ollama, and pulls qwen3:4b itself. `P14_Shopping_Assistant_Colab_Final.ipynb` is the same notebook, the one we submit after a Colab run. | ≈ 40 min (estimated) |
| Laptop / workstation (GPU) | open `P14_Shopping_Assistant_Jupyter.ipynb` from the project folder on the project's kernel, or see below | 14.5 min cold, 9.4 min warm (RTX 4000 Ada) |
| Laptop CPU only | see below; `PICKWISE_RETRIEVAL=dense` keeps searches fast | 2 h 29 min cold on a 24-core desktop CPU (mostly the Qwen3 re-ranker); longer on a typical laptop |

```bash
ollama pull qwen3:4b                                   # https://ollama.com
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
PICKWISE_QUICK_RUN=1 .venv/bin/python run_notebook.py  # live log: tail -f notebook_run.log
```

**On a machine with 16 GB of RAM or less**, start Ollama with `LLAMA_ARG_CACHE_RAM=0` (e.g. `LLAMA_ARG_CACHE_RAM=0 ollama serve`). Its llama.cpp runner otherwise keeps a prompt cache that grows to ~8 GB of RAM. The Colab cell does this already.

**Recorded LLM outputs.** The LLM batch work would take ~10 h on a T4: 1,000 need analyses, the reasoning check, the smoke tests, the end-to-end runs and the judge. Its outputs are in `recorded/` and are replayed, so every reported number comes out the same. Everything else runs live: the catalogue, embeddings, searches, metrics, demo, memory test and cart attacks. `PICKWISE_RECOMPUTE=1` recomputes everything (about 5 h on our GPU). Without `PICKWISE_QUICK_RUN` the notebook uses all 800 test requests and recomputes the model comparison: ~35 min on our GPU.

### Settings

All settings live in [`Pickwise/config.py`](Pickwise/config.py). Overrides:

| Variable | Default |
|---|---|
| `PICKWISE_MODEL` | `qwen3:4b` |
| `PICKWISE_QUICK_RUN` | `0` (`1`: 100 test requests, recorded model comparison) |
| `PICKWISE_RECOMPUTE` | `0` (`1`: recompute the recorded LLM outputs instead of replaying them) |
| `PICKWISE_DATA_DIR` | `./data` |
| `PICKWISE_RETRIEVAL` | `hybrid+rerank` (or `dense`: BLaIR only, ~5 ms per search, same test MRR; good for CPU) |
| `PICKWISE_MCP_PORT` | `8766` |
| `PICKWISE_INTERACTIVE` | `0` (set `1` for the live demo: type messages and approve/reject at the cart) |
| `OLLAMA_HOST` | `http://localhost:11434` |

## Repository layout

```
P14_Shopping_Assistant.ipynb   the deliverable notebook (problem → data → system → evaluation → demo), executed
P14_Shopping_Assistant_Jupyter.ipynb      the same, restructured for evaluation: a local Jupyter kernel (full run)
P14_Shopping_Assistant_Colab.ipynb        the same for a free Colab T4 (quick run)
P14_Shopping_Assistant_Colab_Final.ipynb  the notebook we submit: the Colab one, run on a free T4
Pickwise/                      reusable code imported by the notebook, MCP server and app
  config.py                    all settings
  data.py                      queries, cached catalogue, dev/test split, leakage checks
  llm.py                       Ollama chat model + reasoning trace
  retrieval.py                 BM25 + FAISS → RRF → cross-encoder; category filter; chunked mode
  metrics.py                   Recall@k, MRR, nDCG@k, paired bootstrap
  need.py                      need analyst: schema, prompt, cached batch analysis, search query
  shop.py                      the catalogue tools (search_products, get_product, list_categories)
  mcp_server.py                MCP server for those tools (python -m Pickwise.mcp_server)
  tools.py                     the tools via MCP or in-process; starts the server when needed
  graph.py                     LangGraph supervisor → need analyst → searcher → comparator; ReAct baseline
  evaluate.py                  cached end-to-end runs, Hit@3
  judge.py                     faithfulness judge + the sheet for human checks (Phase 7)
  memory.py                    long-term memory: shopper preferences in the LangGraph Store
  cart.py                      add_to_cart (only after approval) + audit log
app.py                         the web app (Gradio): chat, live trace, approve / reject at the cart
recorded/                      recorded LLM outputs, replayed so the notebook runs on a T4 / CPU
requirements.txt
run_notebook.py                runs the notebook top to bottom with a live per-cell log
P14_Project_Plan.md            build plan, risks, open issues
CURRENT_PROJECT_STATUS.md      live progress log
DECISION_LOG.md                design choices made without a team vote, with evidence and how to revert
```

The course brief (`Project/`) and labs (`modules/`) are kept locally and are not part of this repository.

## Built on the course labs

| Part | Lab |
|---|---|
| Need analyst (query rewriting) | Module 2 · Structured_Generation_Lab |
| Retrieval / RAG | Module 3 · RAG_Pipeline_Lab |
| Supervisor + specialists, memory, approval | Module 4 · Multi_Agent_System_End_to_End |
| MCP server + tools | Module 4 · MCP_Server_and_Tools_Lab |
| Config / LLM wrapper / app structure, web app | Module 4 · Agentic_app |

## Team

| Member | Contribution |
|---|---|
| A Lalith Madhav | _TBD_ |
| Anusha T | _TBD_ |
| Dhanvanth | _TBD_ |

## References

The full list (models, papers, course labs) is in the notebook's §10.

- Hou et al., 2024. *Bridging Language and Items for Retrieval and Recommendation* (Amazon-C4, BLaIR).
- Models: qwen3:4b (Ollama), `hyp1231/blair-roberta-base`, `Qwen/Qwen3-Reranker-0.6B`; lab models `BAAI/bge-small-en-v1.5`, `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Cormack et al., 2009 (reciprocal rank fusion); Robertson & Zaragoza, 2009 (BM25); Yao et al., 2023 (ReAct); Zheng et al., 2023 (LLM-as-a-judge).
- LangGraph, LangChain, Model Context Protocol, FAISS, rank-bm25, sentence-transformers, Hugging Face Datasets, Gradio.
