#!/usr/bin/env python3
"""Head-to-head compression comparison: the lossless codec vs. the real competing methods.

For each redundant agent document (with a known buried fact) we run every method and record, on
identical inputs:
  * tokens / % saved,
  * lossless?  -- can the exact original be recovered (round-trip)?  Only the codec can.
  * fact preserved?  -- does the buried fact survive the compression?

Methods:
  raw          no compression (baseline)
  recency      keep the last K items (lossy)
  digest       AFM-style per-item salience digest (lossy)
  llmlingua2   REAL LLMLingua-2, the leading prompt compressor (lossy, per-item), matched to the
               codec's output size so the comparison is at equal compression
  codec        our lossless cross-item redundancy codec

The point: the codec is the only lossless, model-readable method, and at matched compression the
lossy competitors drop the buried fact while the codec never does. Every number is measured.

Usage: python bench/codec_compare.py --docs 12 --out bench/results_replay/codec_compare.csv
       (add --no-llmlingua to skip the heavy arm)
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from foveance.codec import RedundancyCodec  # noqa: E402
from foveance.store import Fidelity, Item, default_renderer  # noqa: E402

# reuse the buried-fact-under-redundancy task builder
sys.path.insert(0, os.path.dirname(__file__))
from codec_paper_bench import make_task  # noqa: E402


def _count(s: str) -> int:
    return max(1, len(s) // 4)


_LL = {}


def llmlingua2_compress(text: str, target: int) -> str:
    if "c" not in _LL:
        from llmlingua import PromptCompressor
        _LL["c"] = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True, device_map="cpu")
    return _LL["c"].compress_prompt(text, target_token=max(20, target))["compressed_prompt"]


def methods(items, gold, use_ll):
    """Return {name: (text, lossless_bool)} for one document."""
    codec = RedundancyCodec(min_run=2, token_counter=_count)
    raw = "\n".join(t for _, t in items)
    out = {}
    out["raw"] = (raw, True)
    # recency: last 4 items verbatim, older elided
    keep = 4
    rec = [t if k >= len(items) - keep else f"[{i}: older elided]"
           for k, (i, t) in enumerate(items)]
    out["recency"] = ("\n".join(rec), False)
    # digest: per-item salience
    dig = [default_renderer(Item(i, "tool_output", t, 0), Fidelity.DIGEST) for i, t in items]
    out["digest"] = ("\n".join(dig), False)
    # codec: lossless cross-item
    rendered, rep = codec.render(items)
    coded = "\n".join(t for _, t in rendered)
    # verify losslessness by structural round-trip
    lossless = codec.unpack(codec.pack(items)) == list(items)
    out["codec"] = (coded, lossless)
    # llmlingua2 at two operating points: matched to the codec's size (equal-compression), and
    # aggressive (3x tighter) to expose that its fact-preservation is probabilistic, not guaranteed.
    if use_ll:
        try:
            out["llmlingua2"] = (llmlingua2_compress(raw, rep.tokens_out), False)
            out["llmlingua2_aggr"] = (llmlingua2_compress(raw, max(20, rep.tokens_out // 3)), False)
        except Exception as e:  # keep the run alive if the heavy dep misbehaves
            print(f"  llmlingua2 skipped: {type(e).__name__}: {str(e)[:70]}", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=12)
    ap.add_argument("--no-llmlingua", action="store_true")
    ap.add_argument("--out", default="bench/results_replay/codec_compare.csv")
    args = ap.parse_args(argv)
    use_ll = not args.no_llmlingua

    rows = []
    for d in range(args.docs):
        task = make_task(d)
        items = task["items"]
        gold = task["gold"]
        raw_tok = sum(_count(t) for _, t in items)
        for name, (text, lossless) in methods(items, gold, use_ll).items():
            tok = _count(text)
            g = gold.lower().replace(" ", "")
            rows.append({"doc": d, "method": name, "tokens": tok,
                         "saved_pct": round(100 * (1 - tok / raw_tok), 1) if raw_tok else 0.0,
                         "lossless": lossless,
                         "fact_preserved": int(g in text.lower().replace(" ", ""))})
        print(f"  doc {d}: done", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["doc", "method", "tokens", "saved_pct", "lossless",
                                          "fact_preserved"])
        w.writeheader()
        w.writerows(rows)

    order = ["raw", "recency", "digest", "llmlingua2", "llmlingua2_aggr", "codec"]
    print(f"\nwrote {args.out}\n\n{'method':12s} {'mean saved':>11s} {'lossless':>9s} "
          f"{'fact kept':>10s}")
    agg = os.path.join(os.path.dirname(args.out), "codec_compare_summary.csv")
    with open(agg, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "n", "mean_saved_pct", "pct_lossless",
                                          "pct_fact_preserved"])
        w.writeheader()
        for name in order:
            rs = [r for r in rows if r["method"] == name]
            if not rs:
                continue
            rec = {"method": name, "n": len(rs),
                   "mean_saved_pct": round(statistics.mean(r["saved_pct"] for r in rs), 1),
                   "pct_lossless": round(100 * statistics.mean(r["lossless"] for r in rs)),
                   "pct_fact_preserved": round(100 * statistics.mean(r["fact_preserved"]
                                                                     for r in rs))}
            w.writerow(rec)
            print(f"{name:12s} {rec['mean_saved_pct']:10.1f}% {rec['pct_lossless']:8d}% "
                  f"{rec['pct_fact_preserved']:9d}%")
    print(f"wrote {agg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
