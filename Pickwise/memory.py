"""Phase 5: long-term memory, shopper preferences kept in the LangGraph Store across conversations.
Adapted from Module 4 · Multi_Agent_System_End_to_End (Section D: store.put / store.get per user).

A message that contains a preference cue ("allergic", "never", "always", "budget", ...) goes to an extractor
(structured output, reasoning off) that lists what the shopper wants in general (prefer), what they never want
(avoid) and a budget. A small model is good at WHAT was said and unreliable at WHICH field it belongs in, so the
code decides the rest deterministically:
  - grounded: every remembered item must be the shopper's own words (no invented preferences);
  - polarity: an item preceded by a negation in its clause ("never buy ... Munchkin", "no leather") is an
    avoidance, otherwise a preference;
  - brands: an avoided item written with a capital mid-sentence is a brand, filtered out of results by title.
Everything is stored under ("shoppers", user_id) and read on every later turn and conversation. The catalogue
has no prices, so a budget is remembered but cannot be checked.
"""
import re

from pydantic import BaseModel, Field

from .llm import get_llm

NAMESPACE = "shoppers"
KEY = "preferences"
EMPTY = {"prefer": [], "avoid": [], "avoid_brands": [], "budget": ""}


class PreferenceUpdate(BaseModel):
    prefer: list[str] = Field(default=[], description="brands or general qualities the shopper wants in all their shopping, e.g. a brand they always buy, 'vegan'")
    avoid: list[str] = Field(default=[], description="brands, materials or ingredients the shopper never wants, e.g. an allergy or a brand they won't buy")
    budget: str = Field(default="", description="a spending limit the shopper states, e.g. 'under $30'; empty if none")


EXTRACT_PROMPT = """You keep a shopping assistant's memory of a shopper. From the shopper's message, extract only LASTING preferences that should be remembered for future shopping: brands or qualities they always want (prefer), brands, materials or ingredients they never want (avoid), and a budget.
Do NOT include what they are asking for right now, product features, or instructions to the assistant. Use the shopper's own words. Most messages contain no lasting preference: then return empty fields.

Examples (copy words from the message itself, never from these examples):
"My son is allergic to peanuts, and I don't buy Acme products anymore. Looking for snacks." -> avoid ["peanuts", "Acme"]
"I always go for Zenith, and I can only spend under $50." -> prefer ["Zenith"], budget "under $50"
"I need a thermometer with auto off that uses regular batteries." -> nothing
"Add the second one to my cart." -> nothing"""

# Only messages with a cue like these can hold a lasting preference; the rest skip the extractor (saves an LLM
# call and stops the model 'remembering' the current request). "love" is left out on purpose: in shopping
# requests it is nearly always about the product ("I love this protein").
CUES = re.compile(r"\b(allerg\w*|never|always|hate|dislike|don'?t (like|buy|want|use|mind)|do not (like|buy|want|use)|"
                  r"can'?t stand|avoid\w*|prefer\w*|favou?rite|only (buy|use|wear|eat)|budget|under \$|less than \$|"
                  r"vegan|vegetarian|gluten|sensitive|intoleran\w*|brand)\b", re.I)
NEGATION = re.compile(r"\b(no|not|never|don'?t|do not|won'?t|avoid\w*|allerg\w*|hate|dislike|without|"
                      r"can'?t stand|intoleran\w*|free of)\b", re.I)
NOT_A_PREFERENCE = re.compile(r"\b(first|second|third|one|cart|add|approve\w*|this|these|that|prefer\w*|"
                              r"like|love|want|need|please|anything|something)\b", re.I)
CONNECTIVES = {"and", "the", "for", "with", "any", "all"}


def has_cue(message: str) -> bool:
    return bool(CUES.search(message))


# An explicit brand statement ("I always buy Columbia", "I never buy anything from Munchkin", "I only wear Nike").
# The LLM extractor is borderline on these: at temperature 0 it returned nothing for "I always buy Columbia…"
# 5 times in a row in one session after getting it right in two others, so a rule backs it up.
BRAND_STATEMENT = re.compile(r"(?i:\b(?:always|only|never|don'?t|do not|won'?t)\s+(?:buy|use|wear|get|go for|choose|pick|order)"
                             r"\s+(?:anything\s+|any\s+|products?\s+)?(?:from\s+|by\s+)?)([A-Z][\w&'-]+(?:\s+[A-Z][\w&'-]+)*)")


async def extract(message: str) -> PreferenceUpdate:
    """The LLM's reading of the message, plus any explicit brand statement; merge() then decides the side
    (prefer / avoid) of every item from the negation in its clause."""
    extractor = get_llm(reasoning=False).with_structured_output(PreferenceUpdate)
    update = await extractor.ainvoke([("system", EXTRACT_PROMPT), ("human", message)])
    found = {v.lower() for v in [*update.prefer, *update.avoid]}
    update.prefer += [b for b in BRAND_STATEMENT.findall(message) if b.lower() not in found]
    return update


def _words(text: str) -> list:
    return re.findall(r"[a-z0-9$]+", text.lower())


def grounded(value: str, message: str) -> bool:
    """Every word of the item (3+ letters, ignoring connectives) appears in the shopper's message."""
    said = set(_words(message))
    words = [w for w in _words(value) if len(w) >= 3 and w not in CONNECTIVES]
    return bool(words) and all(w in said or w.rstrip("s") in said or w + "s" in said for w in words)


def _position(value: str, message: str) -> int:
    first = next((w for w in _words(value) if len(w) >= 3), "")
    match = re.search(r"\b" + re.escape(first), message, re.I) if first else None
    return match.start() if match else -1


def polarity(value: str, message: str) -> str:
    """'avoid' if a negation comes before the item within its clause, else 'prefer'."""
    at = _position(value, message)
    clause = re.split(r"[.;!?,]", message[:at])[-1] if at >= 0 else ""
    return "avoid" if NEGATION.search(clause) else "prefer"


def is_brand(value: str, message: str) -> bool:
    """Written with a capital in the message, and not just because it starts a sentence."""
    at = _position(value, message)
    if at < 0 or not message[at:at + 1].isupper():
        return False
    before = message[:at].rstrip()
    return bool(before) and before[-1] not in ".!?"


def load(store, user_id: str) -> dict:
    item = store.get((NAMESPACE, user_id), KEY)
    return {**EMPTY, **(item.value if item else {})}


def merge(old: dict, update: PreferenceUpdate, message: str) -> dict:
    """Add the grounded items of `update`, each on the side its own clause puts it."""
    new = {k: list(v) if isinstance(v, list) else v for k, v in {**EMPTY, **old}.items()}
    for value in [*update.prefer, *update.avoid]:
        value = value.strip()
        if not value or len(value.split()) > 4 or NOT_A_PREFERENCE.search(value) or not grounded(value, message):
            continue
        side, other = ("avoid", "prefer") if polarity(value, message) == "avoid" else ("prefer", "avoid")
        if value.lower() not in (v.lower() for v in new[side]):
            new[side].append(value)
        new[other] = [v for v in new[other] if v.lower() != value.lower()]          # a changed mind replaces
        new["avoid_brands"] = [b for b in new["avoid_brands"] if b.lower() != value.lower()]
        if side == "avoid" and is_brand(value, message):
            new["avoid_brands"].append(value)
    if update.budget and any(d in message for d in re.findall(r"\d+", update.budget)):
        new["budget"] = update.budget
    return new


def save(store, user_id: str, prefs: dict) -> None:
    store.put((NAMESPACE, user_id), KEY, prefs)


def describe(prefs: dict, negatives: bool = True) -> str:
    """Preferences as one line for a prompt; '' when there are none. negatives=False leaves out what to
    avoid: a search query that names 'Munchkin' or 'latex' finds MORE of them, so avoidances are applied
    after the search (comparator + brand filter), never in the query."""
    parts = []
    if prefs.get("prefer"):
        parts.append("prefers " + ", ".join(prefs["prefer"]))
    if negatives and prefs.get("avoid"):
        parts.append("must avoid " + ", ".join(prefs["avoid"]))
    if prefs.get("budget"):
        parts.append(f"budget {prefs['budget']} (prices are not in the catalogue, so this cannot be checked)")
    return "; ".join(parts)
