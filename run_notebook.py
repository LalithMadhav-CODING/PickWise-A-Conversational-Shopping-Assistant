"""Run the notebook top to bottom with a live log: each cell's start, outputs and time, as they happen.

    .venv/bin/python run_notebook.py                 # runs P14_Shopping_Assistant.ipynb in place
    tail -f notebook_run.log                         # in another terminal, to watch it

The executed notebook is saved at the end (and also if a cell fails, up to that cell).
"""
import sys
import time
from datetime import datetime

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NOTEBOOK = sys.argv[1] if len(sys.argv) > 1 else "P14_Shopping_Assistant.ipynb"
LOG = sys.argv[2] if len(sys.argv) > 2 else "notebook_run.log"

nb = nbformat.read(NOTEBOOK, as_version=4)
n_code = sum(c.cell_type == "code" for c in nb.cells)
log = open(LOG, "w", buffering=1)
started = {}


def say(text: str) -> None:
    log.write(text + "\n")
    print(text, flush=True)


def output_text(output) -> str:
    if output.output_type == "stream":
        return output.text
    if output.output_type == "error":
        return f"{output.ename}: {output.evalue}"
    return output.get("data", {}).get("text/plain", "")


def on_cell_start(cell, cell_index):
    if cell.cell_type != "code":
        return
    started[cell_index] = time.time()
    number = sum(c.cell_type == "code" for c in nb.cells[:cell_index + 1])
    first_line = cell.source.strip().splitlines()[0] if cell.source.strip() else ""
    say(f"\n[{datetime.now():%H:%M:%S}] ▶ cell {number}/{n_code}: {first_line[:90]}")


def on_cell_executed(cell, cell_index, execute_reply):
    if cell.cell_type != "code":
        return
    for output in cell.get("outputs", []):
        text = "\n".join(line for line in output_text(output).splitlines()
                         if "it/s]" not in line and "Loading weights" not in line)
        if text.strip():
            say(text.rstrip())
    say(f"  ✓ {time.time() - started.get(cell_index, time.time()):.0f}s")


client = NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": "."}},
                        on_cell_start=on_cell_start, on_cell_executed=on_cell_executed)
t0 = time.time()
say(f"[{datetime.now():%H:%M:%S}] running {NOTEBOOK} ({n_code} code cells)")
try:
    client.execute()
    say(f"\n[{datetime.now():%H:%M:%S}] ✅ finished in {(time.time() - t0) / 60:.1f} min")
except CellExecutionError as e:
    say(f"\n[{datetime.now():%H:%M:%S}] ❌ failed after {(time.time() - t0) / 60:.1f} min:\n{str(e)[-2000:]}")
    raise
finally:
    nbformat.write(nb, NOTEBOOK)
