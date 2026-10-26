# Decision Log

Design choices made **without asking first**, for later review: what was decided, what else was considered, the evidence, and how to undo it. Earlier, team-approved decisions are in the *Decisions made* table of [`CURRENT_PROJECT_STATUS.md`](CURRENT_PROJECT_STATUS.md).

Status: ✅ in effect · 🔁 revisit · ↩️ reverted

---

## Retrieval, after the model upgrade (2026-10-26)

| # | Decision | Options considered | Evidence / reason | How to revert | Status |
|---|---|---|---|---|---|
| R.1 | **Keep the full pipeline** (BM25 + BLaIR → RRF → Qwen3-Reranker) as the system retriever, although BLaIR alone on the raw request scored level on test | Switch to BLaIR-only (5 ms, CPU-friendly) | (1) The brief's RAG requirement is hybrid BM25 + dense with RRF and a re-ranker, served over MCP. (2) On **dev** the full pipeline led on MRR (0.381 vs 0.364); switching on test results would tune on test. (3) C4 queries are all vague stories; exact terms (brands, model numbers) in real chats are where BM25 and the re-ranker help, and C4 doesn't measure them. | Use `PICKWISE_RETRIEVAL=dense` (R.2) | ✅ |
| R.2 | **BLaIR-only as a switchable mode** (`PICKWISE_RETRIEVAL=dense`) for laptop-CPU runs and quick demos; default stays `hybrid+rerank` | No switch | Same test MRR (0.356 vs 0.360); ~5 ms vs ~0.3 s GPU / ~10 s CPU per search. | Unset the variable | ✅ |

## Phase 4: MCP server + multi-agent (2026-10-26, branch `phase-4-mcp-multi-agent`)

| # | Decision | Options considered | Evidence / reason | How to revert | Status |
|---|---|---|---|---|---|
| 4.1 | **Searcher is a tool node with no LLM.** It calls `search_products` with the need analyst's query (*rewrite + request*). | A ReAct searcher agent (an LLM writes the tool call) | Runs exactly the retriever evaluated in Phases 2–3 (dev Recall@10 0.59); ~0.3 s vs ~10–15 s for an LLM tool call, and a 4B model may rewrite the query. The ReAct pattern is still shown by the baseline (4.11). *You approved this one.* | `graph.py · searcher`: replace with `build_react_baseline`-style sub-graph | ✅ |
| 4.2 | **Comparator returns structured picks:** 3 × `{item_id, reason, quote}`; `item_id` is a `Literal` of that turn's candidate ids. | Free-text answer | Hit@3 is read straight from `item_id`s; the comparator *cannot* name a product it was not shown (enforced by the generation grammar); every quote is checked against the product text, giving a faithfulness check for free. *You approved this one.* | `graph.py · recommendation_schema`, `comparator` | ✅ |
| 4.3 | **No separate Writer node:** the reply is rendered from the comparator's picks with a fixed template. | An LLM Writer (the lab pattern) | Saves one LLM call per turn (~5–20 s with reasoning on); a template can't add claims that aren't in the picks, so faithfulness stays where it's checked (4.2). Closes review issue #3. | Add a writer node between comparator and END | ✅ |
| 4.4 | **Supervisor = rule + LLM router.** No products shown yet → always `search` (rule, no LLM call); otherwise a structured-output router (reasoning off) chooses `search` / `compare` / `cart`. | LLM router on every turn; rules only | Single-shot requests (all of the evaluation) skip a useless LLM call; follow-ups ("the second one", "cheaper") need the LLM. Same hybrid pattern as the lab's supervisor. | `graph.py · supervisor`: drop the `if not state.get("candidates")` rule | ✅ |
| 4.5 | **One set of tool functions for both transports** (`shop.py`), registered on the MCP server and wrapped as local `StructuredTool`s; all return JSON text. | Separate implementations (the lab duplicates them) | MCP and local can't drift apart (same names, docstrings, results); the ablation then measures transport only. | — | ✅ |
| 4.6 | **`search_products` reports its own compute time** in its JSON result. | Time only the client call | Lets the MCP-vs-local study split a call into *search* (retrieval) and *transport* (call − search), as the plan required ("separate model time from transport time"). | Remove `"seconds"` from the JSON | ✅ |
| 4.7 | **`add_to_cart` is not on the MCP server.** | Expose it over MCP like the other tools | A high-risk action must only run from the cart node after approval (Phase 5); any MCP client could call a tool on the server. Closes part of review issue #9. | — | ✅ |
| 4.8 | **MCP over streamable HTTP on 127.0.0.1:8766**, started automatically by `tools.ensure_server()`; `--transport stdio` for Claude Desktop / MCP Inspector. | stdio only; port 8765 | HTTP keeps one long-lived server with the models loaded once (lab pattern); 8765 is the lab app's port, so 8766 avoids a clash. | `config.MCP_PORT` / `PICKWISE_MCP_PORT` | ✅ |
| 4.9 | **`search_products(query, category, k)`** with `k` defaulting to 10; `category` optional (empty = no filter). | No `k` (old plan) | Matches the brief's signature; closes review issue #4. Category stays optional because a predicted-category filter halves recall (Phase 3). | — | ✅ |
| 4.10 | **Comparator falls back to search order** when its reply is invalid or has < 3 distinct picks; fallback picks carry a generic reason and no quote. | Fail the turn; retry the LLM | The shopper always gets 3 options; fallbacks are visible in the data (`quote_ok = False`), so they count against faithfulness, not hidden. | `graph.py · comparator` | ✅ |
| 4.11 | **Single-ReAct baseline:** one agent, one tool (`search_products`), reasoning on, the lab's "must call a tool first" guardrail; its picks = the `[item_id]`s it names that came from its own search results. | `create_react_agent` from LangGraph | The lab's `build_react_agent` pattern, so the baseline differs from the system only in architecture. Parsing only ids it actually saw stops it getting credit for invented ids. | `graph.py · build_react_baseline` | ✅ |
| 4.12 | **Graph drawn as Mermaid text** (`get_graph().draw_mermaid()`), not a PNG. | `draw_mermaid_png()` | The PNG renderer calls an external web service; the notebook should run offline after the first download. | — | ✅ |
| 4.13 | **End-to-end runs cached** in `data/runs/<name>.jsonl` (`Pickwise/evaluate.py`). | Re-run every time | A full turn costs tens of seconds of LLM time; caching makes the notebook re-runnable and is what Phase 7 needs (review issue #1). | Delete the files to force a re-run | ✅ |
| 4.14 | **Comparator sees 10 candidates, ≤ 700 characters of text each** (title first). | 5 / 20 candidates; full text | 10 = the retriever's Recall@10 cut-off; 700 chars covers the median product (32 words) in full and keeps the prompt ~2k tokens for a 4B model. With reasoning off the comparator takes ~3 s, so no reason to shrink it. | `config.N_CANDIDATES`, `config.SNIPPET_CHARS` | ✅ |
| 4.15 | **Comparator reasoning OFF** (structured output keeps the reply clean; same rule as the need analyst and router). | on / off | **30-request dev smoke test:** off = Hit@3 0.533, 4.2 s/request, 97.8% of quotes found verbatim in the product text; on = Hit@3 0.567, 58.8 s/request, only 78.9% of quotes verbatim (thinking first makes it paraphrase). Paired ΔHit@3 (on − off) +0.033, 95% CI [−0.10, +0.17], i.e. no evidence of a gain for 14× the time and worse faithfulness. One-turn timing: the comparator's thinking was ~95% of a turn. Re-check on a larger sample in Phase 7. | `build_system(comparator_reasoning=True)` | ✅ (🔁 Phase 7) |
| 4.17 | **Known issue, left for Phase 7:** quotes often copy the product *title*, and reasons sometimes claim more than the quote shows (e.g. "uses regular batteries"). Not fixed in Phase 4. | Change the comparator prompt now | The quote check is an automatic proxy; whether reasons are faithful is what the Phase 7 LLM judge + 30 human checks measure. Changing the prompt now would invalidate the smoke results the choices above rest on. **Candidate fix:** "quote from the description, not the title; the reason may only claim what the quote or product text says". | — | 🔁 Phase 7 |
| 4.16 | **Phase 4 lives on branch `phase-4-mcp-multi-agent`**; `main` gets the retrieval model upgrade (Phases 2–3 follow-up) first. | Everything on `main` | You asked for Phase 4 on a separate branch. | Merge or delete the branch | ✅ |

### Phase 4 smoke test (30 dev requests): reference numbers
| Variant | Hit@3 | s / request | quotes verbatim | named 3 products |
|---|---|---|---|---|
| System · comparator reasoning off (**default**) | 0.533 | 4.2 | 97.8% | 100% |
| System · comparator reasoning on | 0.567 | 58.8 | 78.9% | 100% |
| Single ReAct agent (baseline) | 0.467 | 46.9 | — | 93% |

System (off) − ReAct: ΔHit@3 +0.067, 95% CI [−0.10, +0.23]. Too few requests to call it; Phase 7 measures it on the test subset.
