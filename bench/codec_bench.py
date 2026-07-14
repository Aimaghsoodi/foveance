#!/usr/bin/env python3
"""Codec benchmark: measure the cross-item redundancy codec on REAL recorded traces.

Reports, for each trace, the honest compression ratio of:
  raw            -> nothing (baseline; what you pay today)
  digest         -> per-item structural salience digestion (v0.2 behaviour, lossy)
  codec          -> cross-item redundancy codec alone (LOSSLESS, reversible)
  digest+codec   -> codec on top of digested items (the composed Foveance stack)

Every number is measured from the trace; nothing is assumed. Token accounting is chars/4 to match
the store default (pass --exact-tokens for tiktoken). Writes a CSV row per (trace, method).

Usage:
  python bench/codec_bench.py bench/traces/sre_debug_trace.jsonl --out bench/results_replay/codec_ratio.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from foveance.audit import load_conversations  # noqa: E402
from foveance.codec import RedundancyCodec  # noqa: E402
from foveance.store import Fidelity, Item, default_renderer  # noqa: E402


def _block_text(block) -> str:
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        if isinstance(block.get("text"), str):
            return block["text"]
        c = block.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return "\n".join(_block_text(b) for b in c)
        if block.get("input") is not None:
            return str(block["input"])
    return ""


def items_from_conversation(conv: dict) -> list:
    """One codec item per message, text flattened (tool outputs included)."""
    out = []
    for k, msg in enumerate(conv.get("messages", [])):
        content = msg.get("content")
        if isinstance(content, list):
            text = "\n".join(_block_text(b) for b in content)
        else:
            text = _block_text(content)
        if text.strip():
            out.append((f"m{k}", text))
    return out


def digested(items: list, counter) -> list:
    """Apply the per-item structural digest (the v0.2 lossy tier) to each item."""
    out = []
    for iid, text in items:
        it = Item(iid, "tool_output", text, 0)
        out.append((iid, default_renderer(it, Fidelity.DIGEST)))
    return out


def run(path: str, counter) -> list:
    convs = load_conversations(path)
    items: list = []
    for conv in convs:
        items.extend(items_from_conversation(conv))
    codec = RedundancyCodec(min_run=2, token_counter=counter)

    raw_tokens = sum(counter(t) for _, t in items)
    dig = digested(items, counter)
    dig_tokens = sum(counter(t) for _, t in dig)

    codec_rep = codec.analyze(items)
    combo_rep = codec.analyze(dig)

    rows = [
        {"method": "raw", "tokens": raw_tokens, "saved_pct": 0.0, "factor": 1.0,
         "lossless": True},
        {"method": "digest", "tokens": dig_tokens,
         "saved_pct": round(100 * (1 - dig_tokens / raw_tokens), 1) if raw_tokens else 0.0,
         "factor": round(raw_tokens / dig_tokens, 2) if dig_tokens else 0.0, "lossless": False},
        {"method": "codec", "tokens": codec_rep.tokens_out,
         "saved_pct": round(codec_rep.saved_pct, 1), "factor": round(codec_rep.factor, 2),
         "lossless": codec_rep.lossless},
        {"method": "digest+codec", "tokens": combo_rep.tokens_out,
         "saved_pct": round(100 * (1 - combo_rep.tokens_out / raw_tokens), 1) if raw_tokens else 0.0,
         "factor": round(raw_tokens / combo_rep.tokens_out, 2) if combo_rep.tokens_out else 0.0,
         "lossless": False},
    ]
    for r in rows:
        r["trace"] = os.path.basename(path)
        r["items"] = len(items)
    return rows


def fullstack_sweep(path: str, budgets: list, counter) -> list:
    """Measure the *composed* Foveance stack: the anticipatory allocator grading fidelity under a
    token budget (lossy-but-recoverable via foveance_expand), then the lossless codec on top. The
    ratio climbs as the budget tightens; the high end is the aggressive operating point, loss-free
    in effect because every down-rendered item stays addressable."""
    from foveance.baselines import foveance as foveance_policy
    from foveance.embedders import HashingEmbedder
    from foveance.predictor import AnticipatoryPredictor, PredictorConfig
    from foveance.store import MultiFidelityStore

    convs = load_conversations(path)
    pairs: list = []
    last_query = ""
    for conv in convs:
        for iid, text in items_from_conversation(conv):
            pairs.append((iid, text))
        for msg in reversed(conv.get("messages", [])):
            if msg.get("role") in (None, "user"):
                c = msg.get("content")
                last_query = c if isinstance(c, str) else "\n".join(_block_text(b) for b in c) \
                    if isinstance(c, list) else ""
                break
    store = MultiFidelityStore(token_counter=counter)
    for iid, text in pairs:
        store.add(Item(iid, "tool_output", text, 0))
    raw = sum(counter(t) for _, t in pairs)

    pred = AnticipatoryPredictor(store, HashingEmbedder(), config=PredictorConfig(drift=0.6))
    pred.observe_query(last_query)
    codec = RedundancyCodec(min_run=2, token_counter=counter)

    rows = []
    for b in budgets:
        levels = foveance_policy(store, pred, b, 1)
        _, ntok = store.assemble(levels)
        rendered = [(iid, store.render(iid, levels.get(iid, Fidelity.POINTER))) for iid, _ in pairs]
        combo = codec.analyze(rendered)
        rows.append({"trace": os.path.basename(path), "budget": b, "raw_tokens": raw,
                     "allocated_tokens": ntok,
                     "alloc_saved_pct": round(100 * (1 - ntok / raw), 1) if raw else 0.0,
                     "alloc+codec_tokens": combo.tokens_out,
                     "alloc+codec_saved_pct": round(100 * (1 - combo.tokens_out / raw), 1)
                     if raw else 0.0})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--out", default="bench/results_replay/codec_ratio.csv")
    ap.add_argument("--sweep-out", default="bench/results_replay/codec_fullstack.csv")
    ap.add_argument("--budgets", default="1200,600,300,150,80")
    ap.add_argument("--exact-tokens", action="store_true")
    args = ap.parse_args(argv)

    counter = None
    if args.exact_tokens:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            counter = lambda s: len(enc.encode(s))  # noqa: E731
        except Exception:
            print("tiktoken unavailable; falling back to chars/4")
    if counter is None:
        counter = lambda s: max(1, len(s) // 4)  # noqa: E731

    all_rows: list = []
    for t in args.traces:
        all_rows.extend(run(t, counter))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trace", "items", "method", "tokens", "saved_pct",
                                          "factor", "lossless"])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    print(f"wrote {args.out}")
    for r in all_rows:
        print(f"  {r['trace']:24s} {r['method']:14s} {r['tokens']:7d} tok "
              f"{r['saved_pct']:5.1f}% saved  {r['factor']:.2f}x  lossless={r['lossless']}")

    # full-stack budget sweep (allocator + codec) -> shows the ratio climbing with budget
    budgets = [int(x) for x in args.budgets.split(",")]
    sweep_rows: list = []
    for t in args.traces:
        sweep_rows.extend(fullstack_sweep(t, budgets, counter))
    with open(args.sweep_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trace", "budget", "raw_tokens", "allocated_tokens",
                                          "alloc_saved_pct", "alloc+codec_tokens",
                                          "alloc+codec_saved_pct"])
        w.writeheader()
        for r in sweep_rows:
            w.writerow(r)
    print(f"wrote {args.sweep_out}")
    for r in sweep_rows:
        print(f"  {r['trace']:24s} budget={r['budget']:5d}  alloc {r['alloc_saved_pct']:5.1f}%  "
              f"alloc+codec {r['alloc+codec_saved_pct']:5.1f}% saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
