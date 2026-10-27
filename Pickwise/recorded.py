"""Phase 9: replay the recorded LLM outputs so the notebook runs on a free Colab T4 or a laptop CPU.

recorded/needs/*.jsonl and recorded/runs/*.jsonl are the same cache files the notebook writes while it works
(need analyses, smoke tests, end-to-end runs, judge verdicts). restore() copies any that are missing into
config.DATA_DIR, where analyse_many / run_cached / judge_runs find them and skip the LLM calls. Nothing is
overwritten, and nothing is copied when PICKWISE_RECOMPUTE=1. The live parts of the notebook (Ollama check,
demo, memory test, cart attacks) still call the LLM.
"""
import shutil

import pandas as pd

from . import config


def restore() -> list:
    """Copy the recorded caches that DATA_DIR doesn't have yet; return their names."""
    if config.RECOMPUTE or not config.RECORDED_DIR.exists():
        return []
    copied = []
    for folder in ("needs", "runs"):
        target = config.DATA_DIR / folder
        target.mkdir(parents=True, exist_ok=True)
        for src in sorted((config.RECORDED_DIR / folder).glob("*.jsonl")):
            if not (target / src.name).exists():
                shutil.copy2(src, target / src.name)
                copied.append(f"{folder}/{src.name}")
    return copied


def table(name: str) -> pd.DataFrame:
    """A recorded result table (recorded/<name>.csv), indexed by its first column."""
    return pd.read_csv(config.RECORDED_DIR / f"{name}.csv", index_col=0)
