"""Phase 4: the multi-agent system. Adapted from Module 4 · Agentic_app (graph.py) and
Multi_Agent_System_End_to_End (Sections B, C, F).

    shopper ──► Supervisor ──► Need analyst ──► Searcher ──► Comparator ──► END
                    │             (LLM)          (tool)        (LLM)
                    ├──────────────────────────────────────►  Comparator   a question about products already shown
                    └──► Cart                                               Phase 5: approval before add_to_cart

- Supervisor: LLM-as-router with structured output (reasoning off), backed by a rule: with no products
  shown yet, a turn is always a search (as in the lab, rules back up the LLM where a small model is unreliable).
- Need analyst: need.aanalyse, structured output, reasoning off (Phase 3).
- Searcher: a tool node, no LLM. It calls search_products over MCP or in-process with the need analyst's
  query (rewrite + request): exactly the retriever that was evaluated.
- Comparator: structured output, reasoning off (dev smoke test: same Hit@3, 14x faster, more faithful
  quotes; DECISION_LOG 4.15). It picks 3 products by item_id (a Literal of the candidates'
  ids, so it cannot name a product it was not shown), each with a reason and a quote copied from that
  product's text. Every quote is checked against the text.

build_react_baseline() is the brief's single-ReAct-agent baseline: one agent, one search tool.
"""
import json
import re
import time
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
from pydantic import BaseModel, Field, create_model

from . import config
from .llm import ainvoke_requiring_tool, get_llm
from .need import aanalyse, search_query
from .tools import get_tools, tool_text


# ================================================================== shared state
class ShopState(MessagesState):
    route: str            # the supervisor's choice for this turn
    need: dict            # need analyst output
    query: str            # what the searcher sent to the catalogue
    candidates: list      # products the searcher found: [{item_id, title, category, text}]
    picks: list           # comparator's 3 picks: [{item_id, title, reason, quote, quote_ok}]
    final: str            # the reply to the shopper
    timings: dict         # seconds per step, for the trace and the latency study


def new_turn(text: str) -> dict:
    """Input for one shopper turn: the new message, with the per-turn fields reset."""
    return {"messages": [HumanMessage(text)], "route": "", "final": "", "timings": {}}


def last_user_message(state) -> str:
    return next(m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage))


def conversation(state, last_n: int = 6) -> str:
    """Recent turns as plain text, so agents can resolve 'that one' or 'something cheaper'."""
    return "\n".join(f"{'shopper' if isinstance(m, HumanMessage) else 'assistant'}: {m.content}"
                     for m in state["messages"][-last_n:] if isinstance(m, (HumanMessage, AIMessage)) and m.content)


def normalise(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def quote_in_text(quote: str, text: str) -> bool:
    """Is the comparator's quote really in the product text? (punctuation and case ignored)"""
    q = normalise(quote)
    return len(q) >= 3 and q in normalise(text)


# ================================================================== the comparator's output schema
def recommendation_schema(candidate_ids: list):
    """Built per turn: item_id is a Literal of the candidates' ids, compiled into the generation grammar."""
    pick = create_model(
        "Pick",
        item_id=(Literal[tuple(candidate_ids)], Field(description="the item_id of one candidate product")),
        reason=(str, Field(description="one sentence: why this product fits the shopper's need")),
        quote=(str, Field(description="a short phrase copied word for word from this product's text that supports the reason")),
    )
    return create_model("Recommendation", picks=(list[pick], Field(
        description="the 3 best candidates for the shopper, best first", min_length=3, max_length=3)))


COMPARATOR_PROMPT = """You are the COMPARATOR of a shopping assistant. From the candidate products, choose the 3 that best satisfy the shopper's request, best first.
For each pick give:
- reason: one sentence on why it fits the request,
- quote: a short phrase copied WORD FOR WORD from that product's text that supports the reason.
Use only the product text. Never invent features, prices, ratings or reviews."""


def format_candidates(candidates: list) -> str:
    return "\n\n".join(f"[{c['item_id']}] {c['title']} (category: {c['category']})\n{c['text']}" for c in candidates)


def render(picks: list, need: dict) -> str:
    what = need.get("product_type") if need else None
    lines = [f"Here are my top {len(picks)} picks" + (f" (searched for: {what}):" if what else ":")]
    for i, p in enumerate(picks, 1):
        title = p["title"] if len(p["title"]) <= 100 else p["title"][:97] + "…"
        lines.append(f"{i}. {title}\n   {p['reason']}")
        if p.get("quote") and p.get("quote_ok"):
            lines.append(f'   From the product description: "{p["quote"]}"')
    lines.append("Would you like more details on one of them, or shall I add one to your cart?")
    return "\n".join(lines)


# ================================================================== the system
class Route(BaseModel):
    next: Literal["search", "compare", "cart"] = Field(description=(
        "search: the shopper describes what they need, or changes it (cheaper, another colour, a different product). "
        "compare: a question about the products already shown (which is better, more details). "
        "cart: the shopper wants to buy or add a shown product to the cart."))


async def build_system(via: str = "mcp", comparator_reasoning: bool = False, checkpointer=None, store=None):
    """Assemble the multi-agent graph with its tools taken from the MCP server (via='mcp') or in-process."""
    tools = await get_tools(via)
    search_tool = tools["search_products"]
    router = get_llm(reasoning=False).with_structured_output(Route)
    comparator_llm = get_llm(reasoning=comparator_reasoning)

    async def supervisor(state: ShopState) -> Command:
        t = time.perf_counter()
        if not state.get("candidates"):
            route, why = "search", "rule: no products shown yet"
        else:
            decision = await router.ainvoke(
                "You route a shopper's message to the right step of a shopping assistant.\n"
                f"{Route.model_fields['next'].description}\n\n"
                f"Conversation so far:\n{conversation(state)}\n\n"
                f"LAST shopper message: {last_user_message(state)}\n\nChoose the step for the LAST message only.")
            route, why = decision.next, "LLM router"
        node = {"search": "need_analyst", "compare": "comparator", "cart": "cart"}[route]
        timings = {**state.get("timings", {}), "supervisor": time.perf_counter() - t}
        return Command(goto=node, update={"route": f"{route} ({why})", "timings": timings})

    async def need_analyst(state: ShopState):
        t = time.perf_counter()
        text = last_user_message(state)
        earlier = conversation(state, last_n=6) if len(state["messages"]) > 1 else ""
        request = f"Earlier conversation:\n{earlier}\n\nLatest request: {text}" if earlier else text
        try:
            analysed = await aanalyse(request)
            need, query = analysed.model_dump(), search_query(analysed, text)
        except Exception:                                   # fall back to the shopper's own words
            need, query = {}, text
        return {"need": need, "query": query,
                "timings": {**state["timings"], "need_analyst": time.perf_counter() - t}}

    async def searcher(state: ShopState):
        t = time.perf_counter()
        result = json.loads(tool_text(await search_tool.ainvoke({"query": state["query"], "k": config.N_CANDIDATES})))
        return {"candidates": result["products"],
                "timings": {**state["timings"], "searcher": time.perf_counter() - t,
                            "search_compute": result["seconds"]}}

    async def comparator(state: ShopState):
        t = time.perf_counter()
        candidates = state["candidates"]
        by_id = {c["item_id"]: c for c in candidates}
        prompt = [SystemMessage(COMPARATOR_PROMPT),
                  HumanMessage(f"Shopper's request: {last_user_message(state)}\n\nCandidate products:\n\n"
                               f"{format_candidates(candidates)}")]
        picks = []
        try:
            out = await comparator_llm.with_structured_output(
                recommendation_schema(list(by_id))).ainvoke(prompt)
            for p in out.picks:
                if p.item_id in by_id and p.item_id not in {x["item_id"] for x in picks}:
                    picks.append({"item_id": p.item_id, "title": by_id[p.item_id]["title"], "reason": p.reason,
                                  "quote": p.quote, "quote_ok": quote_in_text(p.quote, by_id[p.item_id]["text"])})
        except Exception:
            pass                                            # an invalid reply falls back to the search order
        for c in candidates:                                # fewer than 3 valid picks: fill from the search order
            if len(picks) >= 3:
                break
            if c["item_id"] not in {x["item_id"] for x in picks}:
                picks.append({"item_id": c["item_id"], "title": c["title"], "quote": "", "quote_ok": False,
                              "reason": "One of the closest matches to your request in the catalogue."})
        final = render(picks, state.get("need"))
        return {"picks": picks, "final": final, "messages": [AIMessage(final)],
                "timings": {**state["timings"], "comparator": time.perf_counter() - t}}

    async def cart(state: ShopState):
        final = "Adding to the cart needs your approval first; that step is built in Phase 5."
        return {"final": final, "messages": [AIMessage(final)]}

    g = StateGraph(ShopState)
    g.add_node("supervisor", supervisor, destinations=("need_analyst", "comparator", "cart"))
    g.add_node("need_analyst", need_analyst)
    g.add_node("searcher", searcher)
    g.add_node("comparator", comparator)
    g.add_node("cart", cart)
    g.add_edge(START, "supervisor")
    g.add_edge("need_analyst", "searcher")
    g.add_edge("searcher", "comparator")
    g.add_edge("comparator", END)
    g.add_edge("cart", END)
    graph = g.compile(checkpointer=checkpointer, store=store)
    graph.via, graph.comparator_reasoning = via, comparator_reasoning
    return graph


# ================================================================== running a turn, with a trace
def _short(text: str, n: int = 110) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n - 1] + "…"


async def run_turn(graph, text: str, cfg: dict = None, verbose: bool = True) -> dict:
    """Run one shopper turn and print a trace of every agent step and tool call as it happens."""
    state, t0 = dict(new_turn(text)), time.perf_counter()
    if verbose:
        print(f"🛒 shopper: {_short(text, 200)}")
    async for update in graph.astream(new_turn(text), cfg or {}, stream_mode="updates"):
        for node, delta in update.items():
            if not isinstance(delta, dict):
                continue
            state.update(delta)
            s = state.get("timings", {}).get(node, 0.0)
            if not verbose:
                continue
            if node == "supervisor":
                print(f"  [supervisor]   → {delta['route']}  ({s:.1f}s)")
            elif node == "need_analyst":
                n = delta["need"]
                print(f"  [need_analyst] {n.get('product_type')!r} · must-haves {n.get('must_haves')} · "
                      f"category {n.get('category')}  ({s:.1f}s)" if n else f"  [need_analyst] failed → raw request ({s:.1f}s)")
            elif node == "searcher":
                ids = [c["item_id"] for c in delta["candidates"]]
                print(f"  [searcher]     search_products(query={_short(state['query'], 80)!r}, k={config.N_CANDIDATES}) "
                      f"via {graph.via.upper()} → {len(ids)} products  "
                      f"(call {s:.2f}s, of which search {state['timings']['search_compute']:.2f}s)")
            elif node == "comparator":
                ok = sum(p["quote_ok"] for p in delta["picks"])
                print(f"  [comparator]   picks {[p['item_id'] for p in delta['picks']]} · "
                      f"quotes found in product text {ok}/{len(delta['picks'])}  ({s:.1f}s)")
            else:
                print(f"  [{node}]")
    state["seconds"] = time.perf_counter() - t0
    if verbose:
        print(f"\n🤖 Pickwise ({state['seconds']:.1f}s):\n{state.get('final', '')}\n")
    return state


# ================================================================== baseline: a single ReAct agent
REACT_PROMPT = """You are a shopping assistant with one tool, search_products.
1. Call search_products ONCE with a short description of what the shopper needs (k=10).
2. From the results, recommend the 3 best products, best first. For each, write its title, its item_id in square brackets like [B0XXXXXXXX], and one reason taken from the product text.
Use only the product text; never invent features, prices or ratings."""


async def build_react_baseline(via: str = "mcp"):
    """The brief's baseline: one ReAct agent (model -> tools -> model) with one search tool.
    Adapted from Module 4 · Agentic_app (graph.build_react_agent)."""
    tool = (await get_tools(via))["search_products"]
    model = get_llm(reasoning=True).bind_tools([tool])

    async def call_model(state: MessagesState):
        msgs = [SystemMessage(REACT_PROMPT)] + state["messages"]
        if not any(isinstance(m, ToolMessage) for m in state["messages"]):
            return {"messages": [await ainvoke_requiring_tool(model, msgs)]}   # guardrail: search first
        return {"messages": [await model.ainvoke(msgs)]}

    g = StateGraph(MessagesState)
    g.add_node("model", call_model)
    g.add_node("tools", ToolNode([tool], handle_tool_errors=True))
    g.add_edge(START, "model")
    g.add_conditional_edges("model", tools_condition)
    g.add_edge("tools", "model")
    agent = g.compile(name="react_baseline")
    agent.via = via
    return agent


async def run_react(agent, text: str, verbose: bool = True) -> dict:
    """Run the baseline on one request; picks = item_ids it names that came from its search results."""
    t0 = time.perf_counter()
    out = await agent.ainvoke({"messages": [HumanMessage(text)]}, {"recursion_limit": config.RECURSION_LIMIT})
    seen = []
    for m in out["messages"]:
        if isinstance(m, ToolMessage):
            try:
                seen += [p["item_id"] for p in json.loads(tool_text(m.content))["products"]]
            except (ValueError, KeyError, TypeError):
                pass
    final = out["messages"][-1].content or ""
    named = [i for i in dict.fromkeys(re.findall(r"\b([A-Z0-9]{10})\b", final)) if i in seen]
    calls = [c for m in out["messages"] if isinstance(m, AIMessage) for c in m.tool_calls]
    result = {"final": final, "picks": named[:3], "tool_calls": calls, "seconds": time.perf_counter() - t0}
    if verbose:
        print(f"🛒 shopper: {_short(text, 200)}")
        for c in calls:
            print(f"  [react]  {c['name']}({_short(json.dumps(c['args']), 100)}) via {agent.via.upper()}")
        print(f"  [react]  named {len(named)} products from its results: {named[:3]}")
        print(f"\n🤖 ReAct baseline ({result['seconds']:.1f}s):\n{final}\n")
    return result
