#!/usr/bin/env python3
"""Aggregate codec_paper.csv into the per-model x arm accuracy table and pooled means, and emit
LaTeX-ready rows. Reads only the real CSV; computes nothing that isn't in it."""
from __future__ import annotations

import csv
import statistics
import sys

ARMS = ["full", "recency", "digest", "reactive_afm", "foveance", "codec", "foveance+codec"]


def main(path: str) -> int:
    rows = list(csv.DictReader(open(path)))
    models = sorted({r["model"] for r in rows})

    def cell(model, arm):
        vals = [int(r["acc"]) for r in rows if r["model"] == model and r["arm"] == arm]
        return statistics.mean(vals) if vals else float("nan")

    def toks(arm):
        vals = [int(r["in_tokens"]) for r in rows if r["arm"] == arm]
        return statistics.mean(vals) if vals else float("nan")

    def vis(arm):
        vals = [int(r["gold_visible"]) for r in rows if r["arm"] == arm]
        return statistics.mean(vals) if vals else float("nan")

    print("=== accuracy: model x arm ===")
    print("model         " + "".join(f"{a[:12]:>14s}" for a in ARMS))
    for m in models:
        print(f"{m:14s}" + "".join(f"{cell(m, a):14.2f}" for a in ARMS))
    print("mean          " + "".join(
        f"{statistics.mean([cell(m, a) for m in models]):14.2f}" for a in ARMS))

    print("\n=== tokens / visibility (pooled) ===")
    for a in ARMS:
        print(f"  {a:15s} tokens={toks(a):7.0f}  gold_visible={vis(a):.2f}")

    print("\n=== LaTeX accuracy rows (paste into Result 3) ===")
    for m in models:
        cells = " & ".join(f"{cell(m, a):.2f}" for a in ARMS)
        print(f"{m:14s}& {cells} \\\\")
    means = " & ".join(f"{statistics.mean([cell(m, a) for m in models]):.2f}" for a in ARMS)
    print(f"\\textbf{{mean}} & {means} \\\\")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "bench/results_replay/codec_paper.csv"))
