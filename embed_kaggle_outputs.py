"""Embed a completed Kaggle run's outputs back into the notebook.

Kaggle's CLI does NOT return the executed notebook (kernels pull = source only,
kernels output = /kaggle/working artifacts + a flat stdout/stderr log). So to get
an openable notebook WITH cell outputs, this script reattaches the real run
outputs to the clean generated notebook:

  - per-cell stdout, split on the deterministic markers train.py/build_notebook.py
    emit via log_line() (=== TRAIN ... ===, OUT-OF-SAMPLE, Saved equity_curves)
  - the equity_curves.png figure as an image output on the final plot cell

Inputs live in notebooks/out/ (populated by `./kaggle_sync.sh pull`):
  drl-trading-pipeline.log   the kernel execution log (JSON list of stream chunks)
  equity_curves.png          the plot artifact

Usage:  python embed_kaggle_outputs.py
Writes: notebooks/drl_trading_pipeline.ipynb (in place, now with outputs)
"""
import base64
import json
import os
import sys

import nbformat as nbf

ROOT = os.path.dirname(os.path.abspath(__file__))
NB_PATH = os.path.join(ROOT, "notebooks", "drl_trading_pipeline.ipynb")
LOG_PATH = os.path.join(ROOT, "notebooks", "out", "drl-trading-pipeline.log")
PNG_PATH = os.path.join(ROOT, "notebooks", "out", "equity_curves.png")


def _streams(log, name):
    return "".join(e["data"] for e in log if e.get("stream_name") == name)


def stdout_stream(text):
    return nbf.v4.new_output("stream", name="stdout", text=text)


def main():
    if not os.path.exists(LOG_PATH):
        print(f"[embed] no run log at {LOG_PATH} — run ./kaggle_sync.sh pull first")
        return 1

    nb = nbf.read(NB_PATH, as_version=4)
    log = json.load(open(LOG_PATH))
    out = _streams(log, "stdout")

    # ── Deterministic split points (anchors emitted by the notebook itself) ──
    def cut(a, b=None):
        i = out.find(a)
        if i < 0:
            return ""
        j = len(out) if b is None else out.find(b)
        j = len(out) if j < 0 else j
        return out[i:j]

    pos_cfg = out.find("data=")
    pos_plot = out.find("\nSaved equity_curves")
    if pos_plot < 0:
        pos_plot = out.find("Saved equity_curves")

    # The eval table is 3 border lines (open, under-title, close). Bound the
    # eval cell to exactly that table so trailing env-construction chatter from
    # the plot cell isn't misattributed to it.
    border = "=" * 98
    pos_out = out.find("OUT-OF-SAMPLE")
    eval_start = out.rfind(border, 0, pos_out) if pos_out > 0 else -1
    eval_end = -1
    if eval_start > 0:
        b2 = out.find(border, pos_out)               # border under the title
        b3 = out.find(border, b2 + 1) if b2 > 0 else -1  # closing border
        eval_end = (b3 + len(border)) if b3 > 0 else pos_plot

    regions = {
        "gpu": out[:pos_cfg] if pos_cfg > 0 else "",
        "config": out[pos_cfg:out.find("=== TRAIN ppo_cvml |")] if pos_cfg > 0 else "",
        "train_cvml": cut("=== TRAIN ppo_cvml |", "=== TRAIN ppo_cvml_nomacro"),
        "train_nomacro": cut("=== TRAIN ppo_cvml_nomacro", "=== TRAIN ppo_flat"),
        "train_flat": out[out.find("=== TRAIN ppo_flat"):eval_start] if eval_start > 0 else "",
        "eval": out[eval_start:eval_end] if eval_start > 0 and eval_end > 0 else "",
        "plot": out[eval_end:] if eval_end > 0 else (out[pos_plot:] if pos_plot > 0 else ""),
    }

    # ── Map regions onto code cells in order ─────────────────────────────────
    # Code-cell order (see build_notebook.py): 0 install+GPU, 1 config,
    # 2-4 module sources, 5 training harness, 6-8 train the three variants,
    # 9 baseline defs, 10 eval table, 11 plot.
    order = ["gpu", "config", None, None, None, None,
             "train_cvml", "train_nomacro", "train_flat", None, "eval", "plot"]

    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    if len(code_cells) != len(order):
        print(f"[embed] WARNING: notebook has {len(code_cells)} code cells, "
              f"expected {len(order)} — regenerate with build_notebook.py")

    exec_count = 0
    for cell, key in zip(code_cells, order):
        exec_count += 1
        cell.execution_count = exec_count
        cell.outputs = []
        text = regions.get(key, "") if key else ""
        if text.strip():
            cell.outputs.append(stdout_stream(text if text.endswith("\n") else text + "\n"))

    # ── Attach the figure to the plot cell (last code cell) ──────────────────
    if os.path.exists(PNG_PATH):
        png_b64 = base64.b64encode(open(PNG_PATH, "rb").read()).decode("ascii")
        code_cells[-1].outputs.append(nbf.v4.new_output(
            "display_data",
            data={"image/png": png_b64, "text/plain": "<Figure size 1200x500>"},
            metadata={},
        ))
    else:
        print(f"[embed] note: {PNG_PATH} missing — plot image not embedded")

    nbf.write(nb, NB_PATH)
    embedded = sum(1 for c in code_cells if c.outputs)
    print(f"[embed] wrote {NB_PATH} with outputs on {embedded}/{len(code_cells)} code cells"
          f"{' + equity_curves.png' if os.path.exists(PNG_PATH) else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
