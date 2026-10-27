"""Phases 4-5: the multi-agent system, with memory and human approval. Adapted from Module 4 ·
Agentic_app (graph.py) and Multi_Agent_System_End_to_End (Sections B-F).

    shopper ──► remember ──► Supervisor ──► Need analyst ──► Searcher ──► Comparator ──► END
                (Store)          │             (LLM)          (tool)        (LLM)
                                 ├──────────────────────────────────────►  Comparator   a question about the products shown
                                 └──► Cart ──► Approval ──► END                         interrupt(): the shopper says yes or no
    Memory: checkpointer = this conversation (thread_id) · Store = shopper preferences (user_id)

- remember: extracts lasting preferences from each message into the Store (memory.py); every later turn
  and conversation of the same shopper reads them.
- Supervisor: no products shown yet -> search (rule); a cart word in the LAST message -> cart (rule: as in
  the lab, the decision that leads to a high-risk action is a rule, so the LLM router cannot choose cart);
  otherwise an LLM router (structured output, reasoning off) chooses search / compare.
- Need analyst: need.aanalyse, structured output, reasoning off, with the remembered preferences.
- Searcher: a tool node, no LLM: search_products over MCP or in-process with the need analyst's query.
- Comparator: structured output, reasoning off (DECISION_LOG 4.15): 3 picks by item_id (a Literal of the
  candidates' ids), each with a reason and a quote that is checked against the product text.
- Cart -> Approval: the cart node only works out WHICH product; the approval node pauses with interrupt()
  and runs add_to_cart only on an explicit yes. Two nodes, because a node re-runs from its start when the
  graph resumes: the LLM's choice of product must not be re-made after the shopper has seen it.

build_react_baseline() is the brief's single-ReAct-agent baseline: one agent, one search tool.
"""
import json
import re
import time
import uuid
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field, create_model

from . import cart as shop_cart
from . import config, memory
from .llm import ainvoke_requiring_tool, get_llm
from .need import aanalyse, search_query
from .tools import get_tools, tool_text

config_module = config     # LangGraph passes nodes a `config` argument, which hides the settings module


# ================================================================== shared state
class ShopState(MessagesState):
    route: str            # the supervisor's choice for this turn
    preferences: dict     # what the Store remembered about this shopper BEFORE this message (used this turn)
    learned: dict         # preferences learned from this message (stored for later turns)
    need: dict            # need analyst output
    query: str            # what the searcher sent to the catalogue
    candidates: list      # products the searcher found: [{item_id, title, category, text}]
    picks: list           # comparator's 3 picks: [{item_id, title, reason, quote, quote_ok}]
    searched: int         # how many results the searcher asked for (more when the shopper avoids something)
    dropped: int          # products removed because they name a brand or material the shopper avoids
    pending: dict         # the cart action waiting for the shopper's approval
    cart_status: str      # "", "added", "rejected", "unclear"
    final: str            # the reply to the shopper
    timings: dict         # seconds per step, for the trace and the latency study


def new_turn(text: str) -> dict:
    """Input for one shopper turn: the new message, with the per-turn fields reset. Products shown and
    picks stay in the conversation's checkpoint, so 'the second one' still means something next turn."""
    return {"messages": [HumanMessage(text)], "route": "", "final": "", "timings": {}, "pending": {},
            "cart_status": ""}


def with_learned(state) -> dict:
    """The remembered preferences plus the avoidances learned from this message: those apply at once."""
    prefs = dict(state.get("preferences") or {})
    learned = state.get("learned") or {}
    for field in ("avoid", "avoid_brands"):
        prefs[field] = list(dict.fromkeys([*prefs.get(field, []), *learned.get(field, [])]))
    return prefs


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


def user_of(cfg) -> str:
    return cfg.get("configurable", {}).get("user_id", "anonymous")


# ================================================================== the comparator's output schema
def recommendation_schema(candidate_ids: list, n: int = 3):
    """Built per turn: item_id is a Literal of the candidates' ids, compiled into the generation grammar;
    exactly n picks (3, or fewer when the avoid filter leaves fewer candidates)."""
    pick = create_model(
        "Pick",
        item_id=(Literal[tuple(candidate_ids)], Field(description="the item_id of one candidate product")),
        reason=(str, Field(description="one sentence: why this product fits the shopper's need")),
        quote=(str, Field(description="a short phrase copied word for word from this product's text that supports the reason")),
    )
    return create_model("Recommendation", picks=(list[pick], Field(
        description=f"the {n} best candidates for the shopper, best first", min_length=n, max_length=n)))


COMPARATOR_PROMPT = """You are the COMPARATOR of a shopping assistant. From the candidate products, choose the 3 that best satisfy the shopper's request, best first.
For each pick give:
- reason: one sentence on why it fits the request,
- quote: a short phrase copied WORD FOR WORD from that product's text that supports the reason.
Use only the product text. Never invent features, prices, ratings or reviews."""


def format_candidates(candidates: list) -> str:
    return "\n\n".join(f"[{c['item_id']}] {c['title']} (category: {c['category']})\n{c['text']}" for c in candidates)


def first_sentence(text: str, limit: int = 300) -> str:
    """The comparator is asked for ONE sentence; a 4B model sometimes keeps 'thinking aloud' inside the JSON
    string ('Wait, let me re-read...'). Keep the first sentence only."""
    sentence = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
    return sentence if len(sentence) <= limit else sentence[:limit - 1].rsplit(" ", 1)[0] + "…"


def strip_unsupported(reason: str, avoid: list, text: str) -> tuple:
    """A 4B model claims 'no latex' / 'avoids Munchkin' when the product text never mentions them, and an
    instruction alone doesn't stop it. Drop every clause of the reason that names an avoided term the text
    doesn't mention; return the reason and those terms, so the reply can say the description is silent."""
    unknown = [a for a in avoid if normalise(a) and normalise(a) not in normalise(text)]
    if not unknown:
        return reason, []
    # the avoided terms themselves, and talk about the shopper's allergy / preferences that depends on them
    mentions = re.compile(r"\b(" + "|".join(re.escape(a) for a in unknown) +
                          r"|allerg\w*|(brand )?preferences?)\b", re.I)
    if mentions.search(reason):
        # split keeping the separators, so the clauses that stay keep their own "as" / "and" / ", "
        parts = re.split(r"(,\s*|;\s*|\s+(?:and|which|making|so|that|while|as)\s+)", reason.rstrip(". "))
        kept = ""
        for i in range(0, len(parts), 2):
            if parts[i] and not mentions.search(parts[i]):
                kept += (parts[i - 1] if kept else "") + parts[i]
        kept = kept.strip()
        reason = (kept[0].upper() + kept[1:] + ".") if kept else "It matches your request."
    return reason, unknown


NEGATED_BEFORE = re.compile(r"\b(faux|vegan|pu|synthetic|imitation|artificial|fake|non|no|without|free of)[\s-]*$", re.I)
NEGATED_AFTER = re.compile(r"^[\s-]*(free)\b", re.I)


def violates(term: str, text: str) -> bool:
    """Does the product text name an avoided term in a positive way? 'genuine leather' yes; 'faux leather',
    'vegan leather', 'latex-free', 'non-latex' no."""
    for match in re.finditer(r"\b" + re.escape(term) + r"\b", text, re.I):
        if not NEGATED_BEFORE.search(text[max(0, match.start() - 20):match.start()]) \
                and not NEGATED_AFTER.search(text[match.end():match.end() + 8]):
            return True
    return False


def drop_avoided(candidates: list, avoid: list, brands: list) -> tuple:
    """Remove products whose title names an avoided brand, or whose text names an avoided material or
    ingredient in a positive way (a 4B comparator does not reliably honour 'must avoid'). Fails closed: an
    avoided product is never handed back, even if fewer than 3 remain (the searcher fetches extra to refill)."""
    if not avoid and not brands:
        return candidates, 0
    brand_pattern = re.compile(r"\b(" + "|".join(re.escape(b) for b in brands) + r")\b", re.I) if brands else None
    materials = [a for a in avoid if a.lower() not in {b.lower() for b in brands}]
    kept = [c for c in candidates
            if not (brand_pattern and brand_pattern.search(c["title"]))
            and not any(violates(m, c["title"] + " " + c["text"]) for m in materials)]
    return kept, len(candidates) - len(kept)


def render(picks: list, need: dict, question: str = None) -> str:
    what = need.get("product_type") if need else None
    if question:
        lines = [f'On your question "{question.strip()}", here is how the products compare, best first:']
    else:
        top = f"are my top {len(picks)} picks" if len(picks) > 1 else "is my top pick"
        lines = [f"Here {top}" + (f" (searched for: {what}):" if what else ":")]
    for i, p in enumerate(picks, 1):
        title = p["title"] if len(p["title"]) <= 100 else p["title"][:97] + "…"
        lines.append(f"{i}. {title}\n   {p['reason']}")
        if p.get("quote") and p.get("quote_ok"):
            lines.append(f'   From the product description: "{p["quote"]}"')
        if p.get("unknown"):
            lines.append(f"   Note: the description doesn't mention {' or '.join(p['unknown'])}, so please check before buying.")
    lines.append("Would you like more details on one of them, or shall I add one to your cart?")
    return "\n".join(lines)


# ================================================================== the system
class Route(BaseModel):
    # no "cart" option: the step that leads to the high-risk action is chosen by a rule only (CART_WORDS)
    next: Literal["search", "compare"] = Field(description=(
        "search: the shopper describes what they need, or changes it (cheaper, another colour, a different product). "
        "compare: a question about the products already shown (which is better, more details)."))


CART_WORDS = re.compile(r"\b(add|cart|basket|buy|order|purchase|checkout|get me)\b", re.I)
# Clear cases skip the LLM router, which sent "Now I need a dress belt…" to compare after a vegan-wallet turn
# and "Tell me more about the first one" to search. A message with both signals, or neither, goes to the LLM.
NEW_NEED = re.compile(r"\b(need|needs|looking for|want|show me|find me|search for|do you (have|sell)|instead|"
                      r"another|cheaper|different)\b", re.I)
ABOUT_SHOWN = re.compile(r"\b(these|those|them|which (one|of)|one of|the (first|second|third|last|other)|"
                         r"(first|second|third) one|this one|that one|compare|difference|between)\b", re.I)
ORDINALS = {1: r"\b(first|1st|number one|#1|option 1|no\.? ?1)\b", 2: r"\b(second|2nd|number two|#2|option 2|no\.? ?2)\b",
            3: r"\b(third|3rd|number three|#3|option 3|no\.? ?3)\b"}


class CartChoice(BaseModel):
    position: Literal[0, 1, 2, 3] = Field(description="which recommended product (1, 2 or 3) the shopper wants added; 0 if unclear")
    quantity: int = Field(default=1, ge=1, le=20, description="how many; 1 unless the shopper says otherwise")


async def build_system(via: str = "mcp", comparator_reasoning: bool = False, memory_on: bool = True,
                       rewrite: bool = True, checkpointer=None, store=None):
    """Assemble the multi-agent graph. via: tools from the MCP server ('mcp') or in-process ('local').
    Ablations: memory_on=False skips the Store; rewrite=False searches with the shopper's own words
    (the need analyst is skipped)."""
    tools = await get_tools(via)
    search_tool = tools["search_products"]
    router = get_llm(reasoning=False).with_structured_output(Route)
    comparator_llm = get_llm(reasoning=comparator_reasoning)
    cart_resolver = get_llm(reasoning=False).with_structured_output(CartChoice)

    async def remember(state: ShopState, config: RunnableConfig, *, store: BaseStore):
        """Load what is remembered (used this turn) and store what this message adds (used from the next turn
        on: the current message is already in front of the agents, and a one-message request then behaves
        exactly like the evaluated pipeline)."""
        t = time.perf_counter()
        prefs, learned = {}, {}
        if memory_on:
            user, text = user_of(config), last_user_message(state)
            prefs = memory.load(store, user)
            if memory.has_cue(text):
                try:
                    updated = memory.merge(prefs, await memory.extract(text), text)
                    learned = {k: v for k, v in updated.items() if v and v != prefs.get(k)}
                    if learned:
                        memory.save(store, user, updated)
                except Exception:                           # a failed extraction never blocks the turn
                    pass
        return {"preferences": prefs, "learned": learned,
                "timings": {**state["timings"], "remember": time.perf_counter() - t}}

    async def supervisor(state: ShopState) -> Command:
        t = time.perf_counter()
        text = last_user_message(state)
        if not state.get("candidates"):
            route, why = "search", "rule: no products shown yet"
        elif state.get("picks") and CART_WORDS.search(text):
            route, why = "cart", "rule: cart word in the last message"
        elif NEW_NEED.search(text) and not ABOUT_SHOWN.search(text):
            route, why = "search", "rule: asks for something new"
        elif ABOUT_SHOWN.search(text) and not NEW_NEED.search(text):
            route, why = "compare", "rule: about the products shown"
        else:
            decision = await router.ainvoke(
                "You route a shopper's message to the right step of a shopping assistant.\n"
                f"{Route.model_fields['next'].description}\n\n"
                f"Conversation so far:\n{conversation(state)}\n\n"
                f"LAST shopper message: {text}\n\nChoose the step for the LAST message only.")
            route, why = decision.next, "LLM router"
        node = {"search": "need_analyst", "compare": "comparator", "cart": "cart"}[route]
        timings = {**state["timings"], "supervisor": time.perf_counter() - t}
        return Command(goto=node, update={"route": f"{route} ({why})", "timings": timings})

    async def need_analyst(state: ShopState):
        t = time.perf_counter()
        text = last_user_message(state)
        if not rewrite:                                     # ablation: search with the raw request
            return {"need": {}, "query": text, "timings": {**state["timings"], "need_analyst": 0.0}}
        earlier = conversation(state, last_n=6) if len(state["messages"]) > 1 else ""
        request = f"Earlier conversation:\n{earlier}\n\nLatest request: {text}" if earlier else text
        remembered = memory.describe(state.get("preferences") or {}, negatives=False)   # avoidances: comparator
        if remembered:
            request += f"\n\nRemembered shopper preferences: {remembered}"
        try:
            analysed = await aanalyse(request)
            need, query = analysed.model_dump(), search_query(analysed, text)
        except Exception:                                   # fall back to the shopper's own words
            need, query = {}, text
        return {"need": need, "query": query,
                "timings": {**state["timings"], "need_analyst": time.perf_counter() - t}}

    async def searcher(state: ShopState):
        t = time.perf_counter()
        prefs = with_learned(state)
        avoid, brands = prefs.get("avoid", []), prefs.get("avoid_brands", [])
        k = config.N_CANDIDATES * 3 if avoid or brands else config.N_CANDIDATES   # room to refill after the filter
        result = json.loads(tool_text(await search_tool.ainvoke({"query": state["query"], "k": k})))
        candidates, dropped = drop_avoided(result["products"], avoid, brands)
        return {"candidates": candidates[:config.N_CANDIDATES], "searched": k, "dropped": dropped,
                "timings": {**state["timings"], "searcher": time.perf_counter() - t,
                            "search_compute": result["seconds"]}}

    async def comparator(state: ShopState):
        t = time.perf_counter()
        prefs = with_learned(state)
        pool = state["candidates"]
        if state.get("route", "").startswith("compare") and state.get("picks"):
            shown = [p["item_id"] for p in state["picks"]]  # "which of these…": only the products just shown
            pool = sorted((c for c in pool if c["item_id"] in shown), key=lambda c: shown.index(c["item_id"]))
        candidates, dropped = drop_avoided(pool, prefs.get("avoid", []), prefs.get("avoid_brands", []))
        by_id = {c["item_id"]: c for c in candidates}
        n = min(3, len(candidates))
        if not n:                                           # everything found names something the shopper avoids
            final = (f"Every product I found mentions {' or '.join(prefs['avoid'] + prefs['avoid_brands'])}, "
                     "which you avoid. Could you describe what you need in another way?")
            return {"picks": [], "final": final, "messages": [AIMessage(final)], "dropped": dropped,
                    "timings": {**state["timings"], "comparator": time.perf_counter() - t}}
        request = f"Shopper's request: {last_user_message(state)}"
        if len(state["messages"]) > 1:                      # a follow-up: give the conversation as context
            request += f"\n\nConversation so far:\n{conversation(state)}"
        remembered = memory.describe(prefs)
        if remembered:
            request += (f"\n\nRemembered shopper preferences: {remembered}. Respect them: prefer liked brands "
                        "and avoid products whose text shows something the shopper must avoid. A product's text "
                        "not mentioning something does NOT show that the product avoids it: never claim that.")
        if n < 3:
            request += f"\n\nOnly {n} candidate{'s' if n > 1 else ''} fit{'' if n > 1 else 's'} the shopper's avoidances: rank {'them' if n > 1 else 'it'}."
        prompt = [SystemMessage(COMPARATOR_PROMPT),
                  HumanMessage(f"{request}\n\nCandidate products:\n\n{format_candidates(candidates)}")]
        picks = []
        try:
            out = await comparator_llm.with_structured_output(
                recommendation_schema(list(by_id), n)).ainvoke(prompt)
            for p in out.picks:
                if p.item_id in by_id and p.item_id not in {x["item_id"] for x in picks}:
                    text = by_id[p.item_id]["text"]
                    reason, unknown = strip_unsupported(first_sentence(p.reason), prefs.get("avoid", []), text)
                    picks.append({"item_id": p.item_id, "title": by_id[p.item_id]["title"], "reason": reason,
                                  "unknown": [u for u in unknown if u not in prefs.get("avoid_brands", [])],
                                  "quote": p.quote, "quote_ok": quote_in_text(p.quote, text)})
        except Exception:
            pass                                            # an invalid reply falls back to the search order
        for c in candidates:                                # fewer than n valid picks: fill from the search order
            if len(picks) >= n:
                break
            if c["item_id"] not in {x["item_id"] for x in picks}:
                picks.append({"item_id": c["item_id"], "title": c["title"], "quote": "", "quote_ok": False,
                              "reason": "One of the closest matches to your request in the catalogue."})
        question = last_user_message(state) if state.get("route", "").startswith("compare") else None
        final = render(picks, state.get("need"), question)
        if n < 3 and not question and (prefs["avoid"] or prefs["avoid_brands"]):
            final = (f"Only {n} of the products I found avoid {' and '.join(prefs['avoid'] + prefs['avoid_brands'])}.\n"
                     + final)
        return {"picks": picks, "final": final, "messages": [AIMessage(final)], "dropped": dropped,
                "timings": {**state["timings"], "comparator": time.perf_counter() - t}}

    async def cart(state: ShopState, config: RunnableConfig) -> Command:
        """Work out WHICH shown product the shopper wants; never touches the cart itself."""
        t = time.perf_counter()
        text, picks = last_user_message(state), state.get("picks") or []
        position = next((n for n, pattern in ORDINALS.items() if re.search(pattern, text, re.I)), 0)
        quantity = 1
        if not position and picks:
            numbered = "\n".join(f"{i}. {p['title']}" for i, p in enumerate(picks, 1))
            try:
                choice = await cart_resolver.ainvoke(
                    f"The assistant recommended:\n{numbered}\n\nConversation so far:\n{conversation(state)}\n\n"
                    f"LAST shopper message: {text}\n\nWhich recommended product (1-{len(picks)}) does the shopper "
                    "want added to the cart, and how many? Answer 0 if it is not clear.")
                position, quantity = choice.position, choice.quantity
            except Exception:
                position = 0
        timings = {**state["timings"], "cart": time.perf_counter() - t}
        if not picks or not 1 <= position <= len(picks):
            final = ("Which one should I add: 1, 2 or 3?" if picks
                     else "There is nothing to add yet. Tell me what you are looking for first.")
            return Command(goto=END, update={"final": final, "messages": [AIMessage(final)],
                                             "cart_status": "unclear", "timings": timings})
        pick = picks[position - 1]
        draft = {"tool": "add_to_cart", "user_id": user_of(config), "item_id": pick["item_id"],
                 "title": pick["title"], "quantity": quantity}
        return Command(goto="approval", update={"pending": draft, "timings": timings})

    async def approval(state: ShopState, config: RunnableConfig):
        """Human-in-the-loop: pause before the high-risk tool and wait for the shopper's decision."""
        draft = state["pending"]
        assert draft["tool"] in config_module.RISKY_TOOLS
        decision = interrupt(draft)                         # <-- the graph pauses here until resumed
        approved = shop_cart.is_approval(decision)
        if approved:
            final = shop_cart.add_to_cart(draft["user_id"], draft["item_id"], draft["title"], draft["quantity"],
                                          approved=True)
        else:
            final = f"OK, I did not add {draft['title']} to your cart."
        shop_cart.record(draft, decision, config.get("configurable", {}).get("thread_id", ""), executed=approved)
        return {"final": final, "messages": [AIMessage(final)], "pending": {},
                "cart_status": "added" if approved else "rejected"}

    g = StateGraph(ShopState)
    g.add_node("remember", remember)
    g.add_node("supervisor", supervisor, destinations=("need_analyst", "comparator", "cart"))
    g.add_node("need_analyst", need_analyst)
    g.add_node("searcher", searcher)
    g.add_node("comparator", comparator)
    g.add_node("cart", cart, destinations=("approval", END))
    g.add_node("approval", approval)
    g.add_edge(START, "remember")
    g.add_edge("remember", "supervisor")
    g.add_edge("need_analyst", "searcher")
    g.add_edge("searcher", "comparator")
    g.add_edge("comparator", END)
    g.add_edge("approval", END)
    graph = g.compile(checkpointer=checkpointer or InMemorySaver(), store=store or InMemoryStore())
    graph.via, graph.comparator_reasoning, graph.memory_on, graph.rewrite = via, comparator_reasoning, memory_on, rewrite
    return graph



# ================================================================== running a turn, with a trace
def _short(text: str, n: int = 110) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n - 1] + "…"


def _print_step(graph, node: str, delta: dict, state: dict) -> None:
    s = state.get("timings", {}).get(node, 0.0)
    if node == "remember":
        remembered = memory.describe(delta.get("preferences") or {})
        learned = memory.describe({**memory.EMPTY, **(delta.get("learned") or {})})
        print(f"  [remember]     using: {remembered or 'nothing on record'}"
              f"{' · learned: ' + learned if learned else ''}  ({s:.1f}s)")
    elif node == "supervisor":
        print(f"  [supervisor]   → {delta['route']}  ({s:.1f}s)")
    elif node == "need_analyst":
        n = delta["need"]
        print(f"  [need_analyst] {n.get('product_type')!r} · must-haves {n.get('must_haves')} · "
              f"category {n.get('category')}  ({s:.1f}s)" if n else f"  [need_analyst] failed → raw request ({s:.1f}s)")
    elif node == "searcher":
        removed = f", {delta['dropped']} removed that the shopper avoids" if delta.get("dropped") else ""
        print(f"  [searcher]     search_products(query={_short(state['query'], 80)!r}, k={delta['searched']}) "
              f"via {graph.via.upper()} → {len(delta['candidates'])} products{removed}  "
              f"(call {s:.2f}s, of which search {state['timings']['search_compute']:.2f}s)")
    elif node == "comparator":
        ok = sum(p["quote_ok"] for p in delta["picks"])
        dropped = f" · removed {delta['dropped']} that the shopper avoids" if delta.get("dropped") else ""
        print(f"  [comparator]   picks {[p['item_id'] for p in delta['picks']]} · "
              f"quotes found in product text {ok}/{len(delta['picks'])}{dropped}  ({s:.1f}s)")
    elif node == "cart":
        d = delta.get("pending")
        print(f"  [cart]         {'wants ' + str(d['quantity']) + ' × ' + _short(d['title'], 60) if d else 'unclear which product'}  ({s:.1f}s)")
    elif node == "approval":
        print(f"  [approval]     {'✅ approved → add_to_cart executed' if delta.get('cart_status') == 'added' else '❌ not approved → cart unchanged'}")
    else:
        print(f"  [{node}]")


async def _stream(graph, graph_input, cfg: dict, verbose: bool, t0: float) -> dict:
    """Stream one turn (or a resume), printing each step; return the conversation state afterwards."""
    waiting, running = None, dict((await graph.aget_state(cfg)).values)
    async for update in graph.astream(graph_input, cfg, stream_mode="updates"):
        for node, delta in update.items():
            if node == "__interrupt__":
                waiting = delta[0].value
                if verbose:
                    print(f"  ⏸  waiting for the shopper: add {waiting['quantity']} × {_short(waiting['title'], 70)} "
                          "to the cart? (approve / reject)")
            elif isinstance(delta, dict):
                running.update(delta)
                if verbose:
                    _print_step(graph, node, delta, running)
    state = dict((await graph.aget_state(cfg)).values)
    state.update(waiting_for_approval=waiting, cfg=cfg, seconds=time.perf_counter() - t0)
    if verbose and not waiting:
        print(f"\n🤖 Pickwise ({state['seconds']:.1f}s):\n{state.get('final', '')}\n")
    return state


async def run_turn(graph, text: str, cfg: dict = None, verbose: bool = True) -> dict:
    """Run one shopper turn and print a trace of every agent step and tool call as it happens.
    cfg = {"configurable": {"thread_id": conversation, "user_id": shopper}}; a missing thread_id starts a
    new conversation. If the turn pauses for approval, state['waiting_for_approval'] holds the action:
    continue with resume(graph, "approve" | "reject", state['cfg'])."""
    cfg = cfg or {}
    configurable = {"thread_id": f"t-{uuid.uuid4().hex[:8]}", "user_id": "anonymous", **cfg.get("configurable", {})}
    cfg = {**cfg, "configurable": configurable, "recursion_limit": config.RECURSION_LIMIT}
    if verbose:
        print(f"🛒 shopper ({configurable['user_id']}): {_short(text, 200)}")
    return await _stream(graph, new_turn(text), cfg, verbose, time.perf_counter())


async def resume(graph, decision: str, cfg: dict, verbose: bool = True) -> dict:
    """Give the shopper's decision on a paused cart action and finish the turn."""
    if verbose:
        print(f"🛒 shopper decides: {decision}")
    return await _stream(graph, Command(resume=decision), cfg, verbose, time.perf_counter())


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
