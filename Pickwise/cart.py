"""Phase 5: the cart, the one HIGH-RISK action. Adapted from Module 4 · Multi_Agent_System_End_to_End
(Section E: send_report behind interrupt()).

add_to_cart is deliberately out of every agent's reach: it is not on the MCP server, no LLM has it bound
as a tool, and it refuses to run without approved=True, which only the approval node passes, after the
shopper said yes. Every cart request is audited (AUDIT), so the harness metric "share of add_to_cart
calls that went through approval" (must be 100%) is measured from records, not assumed.
"""
from datetime import datetime

CARTS: dict = {}       # user_id -> [{item_id, title, quantity}]
AUDIT: list = []       # one record per cart request that reached the approval step


def add_to_cart(user_id: str, item_id: str, title: str, quantity: int = 1, approved: bool = False) -> str:
    """Add a product to the shopper's cart. HIGH RISK: only after the shopper approved it."""
    if not approved:
        raise PermissionError("add_to_cart needs the shopper's approval")
    CARTS.setdefault(user_id, []).append({"item_id": item_id, "title": title, "quantity": quantity})
    return f"Added {quantity} × {title} to the cart."


def record(draft: dict, decision, thread_id: str, executed: bool) -> None:
    AUDIT.append({"time": datetime.now().strftime("%H:%M:%S"), "thread": thread_id, "user": draft["user_id"],
                  "item_id": draft["item_id"], "decision": str(decision), "executed": executed,
                  "approved": is_approval(decision)})


def is_approval(decision) -> bool:
    """Only an explicit yes counts; anything else (reject, silence, a typo) leaves the cart alone."""
    if isinstance(decision, dict):
        decision = decision.get("decision", "")
    return str(decision).strip().lower() in {"approve", "approved", "yes", "y"}


def approval_share() -> float:
    """Share of executed add_to_cart calls that the shopper approved (the brief's harness metric)."""
    executed = [a for a in AUDIT if a["executed"]]
    return sum(a["approved"] for a in executed) / len(executed) if executed else float("nan")
