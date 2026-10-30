"""Ollama on Colab. Standard library only: the Colab setup cell imports it before `pip install`.

`ollama serve` runs in its own session, so restarting the notebook's Python session (Colab asks for one after some
installs) no longer stops it, and its output goes to ollama.log instead of nowhere (DECISION_LOG 9.11).
"""
import os
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen

# Memory on a free Colab (15 GB GPU, 12.7 GB RAM; DECISION_LOG 9.8-9.9): keep qwen3:4b loaded for the whole run
# with a 4,096-token context and one request at a time (3.2 GB of GPU; peak with the notebook and the MCP server:
# 11.8 GB), and switch off llama.cpp's RAM prompt cache, which otherwise grows to ~8 GB of system RAM.
OLLAMA_ENV = {"OLLAMA_KEEP_ALIVE": "-1", "OLLAMA_CONTEXT_LENGTH": "4096", "OLLAMA_NUM_PARALLEL": "1",
              "LLAMA_ARG_CACHE_RAM": "0"}
LOG = Path(__file__).resolve().parent.parent / "ollama.log"


def ollama_is_up() -> bool:
    host = os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
    try:
        with urlopen((host if "://" in host else f"http://{host}") + "/api/version", timeout=2):
            return True
    except OSError:
        return False


def start_ollama(timeout: float = 60) -> str:
    """Start `ollama serve` unless it already answers, and wait until it does."""
    if ollama_is_up():
        return "already running"
    with open(LOG, "a") as log:
        server = subprocess.Popen(["ollama", "serve"], env={**os.environ, **OLLAMA_ENV}, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ollama_is_up():
            return "started"
        if server.poll() is not None:
            break
        time.sleep(0.5)
    raise RuntimeError(f"Ollama did not start. Its log ({LOG}), last lines:\n{LOG.read_text(errors='replace')[-3000:]}")
