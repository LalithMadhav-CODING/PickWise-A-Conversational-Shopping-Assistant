"""All settings in one place. Each can be overridden with an environment variable.

Adapted from Module 4 · Agentic_app (relay_ops/config.py).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("PICKWISE_DATA_DIR", ROOT / "data"))   # gitignored: rebuilt from load_dataset
SEED = 42

# --- Quick run: smaller test subset so the whole notebook re-runs fast ----------------------------
QUICK_RUN = os.getenv("PICKWISE_QUICK_RUN", "0") == "1"
QUICK_N = 100

# --- Data (brief §02) -----------------------------------------------------------------------------
DATASET_ID = "McAuley-Lab/Amazon-C4"
CATALOGUE_FILE = f"hf://datasets/{DATASET_ID}/sampled_item_metadata_1M.jsonl"
N_QUERIES = 1000          # fixed sample of the 21,223 test queries
DISTRACTOR_RATE = 0.03    # ~32k random distractors from the 1.06M-item pool
N_DEV = 200               # 200 dev (tuning) / 800 test (reported once)
CATALOGUE_CACHE = DATA_DIR / "catalogue.parquet"

# --- LLM (Ollama) ---------------------------------------------------------------------------------
MODEL = os.getenv("PICKWISE_MODEL", "qwen3:4b")              # fallback: "qwen3:1.7b"
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
REASONING = True          # keeps Qwen3's thinking out of the answer (see plan: Risks)

# --- RAG (same models as the Module 3 lab) --------------------------------------------------------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEPTH = 100               # every ranking is cut at 100 products (MRR is computed over this depth)
RRF_K = 60                # Reciprocal Rank Fusion constant (Module 3 lab default)
RRF_WEIGHTS = (1.0, 1.0)  # (BM25, dense) weights in the fusion; equal = the lab's plain RRF (tuned on dev)
RERANK_TOP_N = 50         # hybrid shortlist the cross-encoder re-scores (tuned on dev)
CHUNK_SIZE = 300          # only for the chunking comparison; the main index is one document per product
CHUNK_OVERLAP = 50

# --- Harness --------------------------------------------------------------------------------------
RECURSION_LIMIT = 25
RISKY_TOOLS = {"add_to_cart"}   # the harness asks the shopper before running any of these
