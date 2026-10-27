"""Pickwise web app (Phase 8 stretch goal): chat with the multi-agent system, watch every agent step and
tool call as it happens, and approve or reject each cart action with a button.

    .venv/bin/python app.py              # http://127.0.0.1:7860
    .venv/bin/python app.py --share      # also a temporary public link (e.g. from Colab)

Builds on: Module 4 · Agentic_app (the lab's Streamlit app). Gradio is used here because it streams an async
generator straight into the page, so the trace appears step by step. The app runs the SAME graph as the
notebook (Pickwise.graph.build_system): the same agents, MCP tools, memory and approval gate that are
evaluated. Conversations and shopper preferences are kept in memory for as long as the app runs.
"""
import argparse
import uuid

import gradio as gr
import pandas as pd
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from Pickwise import cart, memory, shop
from Pickwise.graph import build_system, new_turn, stream_turn, turn_config

SAVER, STORE = InMemorySaver(), InMemoryStore()     # shared by both transports: switching keeps the memory
GRAPHS = {}                                          # "mcp" / "local" -> compiled graph, built on first use

EXAMPLES = [
    "I'm vegan, so no leather please. I need a slim wallet that fits in my front pocket.",
    "Which of these holds the most cards?",
    "Add the first one to my cart.",
    "Now I need a dress belt to wear with my office trousers.",
    "I'm allergic to latex and I never buy anything from Munchkin. I need something to keep my toddler busy at bath time.",
]


async def graph_for(via: str):
    if via not in GRAPHS:
        GRAPHS[via] = await build_system(via, checkpointer=SAVER, store=STORE)
    return GRAPHS[via]


def new_session(shopper: str = "guest") -> dict:
    return {"thread": f"app-{uuid.uuid4().hex[:8]}", "shopper": shopper, "waiting": False}


def as_markdown(text: str) -> str:
    """The reply is plain text with one item per line; keep its line breaks in the chat bubble."""
    return "  \n".join(text.splitlines())


def panels(shopper: str) -> tuple:
    """What the app remembers about the shopper, their cart, and the approval log."""
    prefs = memory.describe(memory.load(STORE, shopper)) or "nothing yet"
    items = pd.DataFrame(cart.CARTS.get(shopper, []), columns=["quantity", "title"]).rename(columns={"quantity": "qty"})
    audit = pd.DataFrame([a for a in cart.AUDIT if a["user"] == shopper],
                         columns=["time", "item_id", "decision", "executed", "approved"])
    return f"**Remembered about {shopper}:** {prefs}", items, audit


async def turn(text: str, shown: str, history: list, session: dict, shopper: str, via: str):
    """One step of the conversation, streamed: a new message, or the answer to a paused cart action."""
    shopper = shopper.strip() or "guest"
    if shopper != session["shopper"]:                # another shopper: their own conversation and memory
        session, history = new_session(shopper), []
    graph = await graph_for(via)
    cfg = turn_config({"configurable": {"thread_id": session["thread"], "user_id": shopper}})
    graph_input = Command(resume=text) if session["waiting"] else new_turn(text)
    history = history + [{"role": "user", "content": shown}, {"role": "assistant", "content": "…"}]
    trace = [f"🛒 {shopper}: {text}" if not session["waiting"] else f"🛒 {shopper} decides: {text}"]

    def view(waiting: bool, pending: str = ""):
        return (history, "\n".join(trace), gr.update(visible=waiting), pending, session, *panels(shopper), "")

    yield view(False)
    state = None
    try:
        async for item in stream_turn(graph, graph_input, cfg):
            if isinstance(item, dict):
                state = item
            else:
                trace.append(item)
                yield view(False)
    except Exception as e:                           # e.g. Ollama not running: say so instead of hanging
        trace.append(f"  ⚠️ {type(e).__name__}: {e}")
        history[-1]["content"] = f"Sorry, something went wrong: {type(e).__name__}. See the trace."
        session["waiting"] = False
        yield view(False)
        return
    waiting = state["waiting_for_approval"]
    session["waiting"] = bool(waiting)
    if waiting:
        pending = f"Add **{waiting['quantity']} × {waiting['title']}** to your cart?"
        history[-1]["content"] = "⏸ This needs your approval: press **Approve** or **Reject** below (or type yes / no)."
        yield view(True, pending)
    else:
        trace.append(f"🤖 reply in {state['seconds']:.1f}s")
        history[-1]["content"] = as_markdown(state.get("final") or "")
        yield view(False)


async def send(message: str, history: list, session: dict, shopper: str, via: str):
    if not message.strip():
        return
    async for out in turn(message.strip(), message.strip(), history, session, shopper, via):
        yield out


async def approve(history: list, session: dict, shopper: str, via: str):
    if session["waiting"]:
        async for out in turn("approve", "✅ Approve", history, session, shopper, via):
            yield out


async def reject(history: list, session: dict, shopper: str, via: str):
    if session["waiting"]:
        async for out in turn("reject", "❌ Reject", history, session, shopper, via):
            yield out


def restart(shopper: str):
    """A new conversation with the same shopper: their remembered preferences and cart stay."""
    shopper = shopper.strip() or "guest"
    return ([], "", gr.update(visible=False), "", new_session(shopper), *panels(shopper), "")


def forget(shopper: str):
    shopper = shopper.strip() or "guest"
    STORE.delete((memory.NAMESPACE, shopper), memory.KEY)
    return panels(shopper)


def build_ui(default_via: str = "mcp") -> gr.Blocks:
    with gr.Blocks(title="Pickwise") as demo:
        gr.Markdown("# 🛒 Pickwise\nDescribe what you need in your own words. Pickwise searches the catalogue, "
                    "recommends 3 products with a reason quoted from each product's text, remembers your "
                    "preferences, and **never adds to your cart without your approval**.")
        session = gr.State(new_session())
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(label="Conversation", height=520)
                with gr.Group(visible=False) as approval:
                    pending = gr.Markdown()
                    with gr.Row():
                        yes = gr.Button("✅ Approve", variant="primary")
                        no = gr.Button("❌ Reject", variant="stop")
                with gr.Row():
                    box = gr.Textbox(placeholder="e.g. I need a slim wallet that fits in my front pocket",
                                     show_label=False, scale=5, autofocus=True)
                    go = gr.Button("Send", variant="primary", scale=1)
                gr.Examples(EXAMPLES, inputs=box, label="Try")
            with gr.Column(scale=2):
                trace = gr.Code(label="Agent trace (this turn)", language=None, interactive=False, lines=16,
                                wrap_lines=True)
                with gr.Row():
                    shopper = gr.Textbox(value="guest", label="Shopper", scale=2)
                    via = gr.Radio(["mcp", "local"], value=default_via, label="Tools via", scale=2)
                first_prefs, first_cart, first_audit = panels("guest")
                prefs = gr.Markdown(first_prefs)
                with gr.Row():
                    new_chat = gr.Button("New conversation", size="sm")
                    forget_me = gr.Button("Forget my preferences", size="sm")
                basket = gr.Dataframe(first_cart, label="Cart", interactive=False, wrap=True,
                                      column_widths=["15%", "85%"])
                audit = gr.Dataframe(first_audit, label="Approval log (every cart request)", interactive=False)

        outputs = [chat, trace, approval, pending, session, prefs, basket, audit, box]
        for event in (box.submit, go.click):
            event(send, [box, chat, session, shopper, via], outputs)
        yes.click(approve, [chat, session, shopper, via], outputs)
        no.click(reject, [chat, session, shopper, via], outputs)
        new_chat.click(restart, [shopper], outputs)
        forget_me.click(forget, [shopper], [prefs, basket, audit])

        async def warm():                    # build the graph (and MCP connection) when the page opens,
            await graph_for(default_via)     # so the first message isn't ~10 s slower than the rest
        demo.load(warm)
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pickwise web app")
    parser.add_argument("--share", action="store_true", help="also create a temporary public link")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--via", choices=["mcp", "local"], default="mcp", help="default tool transport")
    args = parser.parse_args()
    print("Loading the catalogue and retrieval models…", flush=True)
    shop.warm_up()
    build_ui(args.via).queue().launch(server_name="127.0.0.1", server_port=args.port, share=args.share)
