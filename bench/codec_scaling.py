#!/usr/bin/env python3
"""Scaling, separation, and overhead measurements for the codec (pure-codec, no model calls).

Three real experiments feeding Paper 3's expanded theory:
  * SCALING   -- codec compression ratio vs number of turns M. Agents re-observe state, so
                 cross-item redundancy (hence the codec's saving) grows with trajectory length.
  * SEPARATION-- joint (cross-item) coding vs per-item coding. The gap is the empirical
                 total-correlation / multi-information: what NO per-item compressor can remove.
  * OVERHEAD  -- codec wall-clock vs context size, to show it is negligible beside an LLM call.

Writes codec_scaling.csv, codec_separation.csv, codec_overhead.csv. Nothing is fabricated.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from foveance.codec import RedundancyCodec  # noqa: E402

_LISTING = "\n".join(f"src/service/module_{i:02d}.py" for i in range(20))
_STACK = ("Traceback (most recent call last):\n"
          '  File "tests/test_checkout.py", line 91, in test_flow\n'
          "    resp = client.post('/checkout', json=payload)\n"
          '  File "app/api.py", line 142, in post\n'
          "    return self.handler(req)\n"
          "AttributeError: 'NoneType' object has no attribute 'total'")
_BOILER = "[cwd=/srv/app CI=true LANG=C TERM=dumb]"


def _tool(cmd: str, body: str) -> str:
    return f"$ {cmd}\n{_BOILER}\n{body}\n(exit 0)"


def trace(turns: int) -> list:
    """A redundant multi-turn agent transcript with `turns` tool outputs. Each turn re-emits one of
    a small set of recurring blocks (listing / stack) plus a unique line, exactly the pattern that
    makes real agent context redundant."""
    items = []
    blocks = [("ls -R src/service", _LISTING), ("pytest -x tests/test_checkout.py", _STACK),
              ("cat app/api.py", _STACK), ("ls -R src/service", _LISTING)]
    for k in range(turns):
        cmd, body = blocks[k % len(blocks)]
        body = body + f"\n# unique probe line for turn {k}: token-{k*7+3}"
        items.append((f"t{k}", _tool(cmd, body)))
    return items


def _count(s: str) -> int:
    return max(1, len(s) // 4)


def run_scaling(turn_grid: list) -> list:
    codec = RedundancyCodec(min_run=2, token_counter=_count)
    rows = []
    for M in turn_grid:
        items = trace(M)
        raw = sum(_count(t) for _, t in items)
        rep = codec.analyze(items)
        rows.append({"turns": M, "raw_tokens": raw, "codec_tokens": rep.tokens_out,
                     "saved_pct": round(rep.saved_pct, 1), "factor": round(rep.factor, 2),
                     "lossless": rep.lossless})
    return rows


def run_separation(turn_grid: list) -> list:
    """Joint (cross-item) vs per-item coding. per_item = sum of packing each item in isolation
    (only intra-item repeats); joint = one shared history (also cross-item). The difference, in
    tokens, is the empirical total correlation a per-item method must leave on the table."""
    codec = RedundancyCodec(min_run=2, token_counter=_count)
    rows = []
    for M in turn_grid:
        items = trace(M)
        raw = sum(_count(t) for _, t in items)
        joint = codec.analyze(items).tokens_out
        per_item = sum(codec.analyze([it]).tokens_out for it in items)
        rows.append({"turns": M, "raw_tokens": raw, "per_item_tokens": per_item,
                     "joint_tokens": joint,
                     "per_item_saved_pct": round(100 * (1 - per_item / raw), 1) if raw else 0.0,
                     "joint_saved_pct": round(100 * (1 - joint / raw), 1) if raw else 0.0,
                     "cross_item_gap_tokens": per_item - joint,
                     "cross_item_gap_pct": round(100 * (per_item - joint) / raw, 1) if raw else 0.0})
    return rows


def run_overhead(turn_grid: list, reps: int = 5) -> list:
    codec = RedundancyCodec(min_run=2, token_counter=_count)
    rows = []
    for M in turn_grid:
        items = trace(M)
        raw = sum(_count(t) for _, t in items)
        best = min(_time_once(codec, items) for _ in range(reps))
        rows.append({"turns": M, "raw_tokens": raw, "codec_ms": round(best * 1000, 3),
                     "us_per_token": round(best * 1e6 / max(raw, 1), 2)})
    return rows


def _time_once(codec, items) -> float:
    t0 = time.perf_counter()
    codec.render(items)
    return time.perf_counter() - t0


def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", default="2,4,8,16,32,64,128")
    ap.add_argument("--outdir", default="bench/results_replay")
    args = ap.parse_args(argv)
    grid = [int(x) for x in args.turns.split(",")]

    sc = run_scaling(grid)
    _write(os.path.join(args.outdir, "codec_scaling.csv"), sc)
    for r in sc:
        print(f"  M={r['turns']:4d}  {r['saved_pct']:5.1f}% saved  {r['factor']:.2f}x")

    sep = run_separation(grid)
    _write(os.path.join(args.outdir, "codec_separation.csv"), sep)
    for r in sep:
        print(f"  M={r['turns']:4d}  per-item {r['per_item_saved_pct']:5.1f}%  "
              f"joint {r['joint_saved_pct']:5.1f}%  cross-item gap {r['cross_item_gap_pct']:5.1f}%")

    ov = run_overhead(grid)
    _write(os.path.join(args.outdir, "codec_overhead.csv"), ov)
    for r in ov:
        print(f"  M={r['turns']:4d}  {r['codec_ms']:8.3f} ms  {r['us_per_token']:.2f} us/token")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
