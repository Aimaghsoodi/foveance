#!/usr/bin/env python3
"""Codec compression ratio on the REAL LongBench-v2 public benchmark (THUDM/LongBench-v2), by
domain. This is the public-benchmark number for the lossless codec: it measures how much
cross-chunk/line redundancy the codec removes from real long contexts, with a round-trip
losslessness check on every example. No model is needed for a compression-ratio measurement.

Prereq: bench/fetch step already saved data/longbench/longbench_v2.jsonl (context + domain).
Usage:  python bench/codec_longbench.py --in data/longbench/longbench_v2.jsonl \
                                        --out bench/results_replay/codec_longbench.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from foveance.codec import RedundancyCodec  # noqa: E402


def _count(s: str) -> int:
    return max(1, len(s) // 4)


def chunk(context: str, window: int = 40) -> list:
    """Split a long context into items the way an agent accumulates it: prefer paragraph
    boundaries; fall back to fixed line windows so the codec can dedup across chunks."""
    paras = [p for p in context.split("\n\n") if p.strip()]
    if len(paras) >= 4:
        return [(f"p{i}", p) for i, p in enumerate(paras)]
    lines = context.split("\n")
    return [(f"w{i}", "\n".join(lines[i:i + window]))
            for i in range(0, len(lines), window)] or [("c0", context)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/longbench/longbench_v2.jsonl")
    ap.add_argument("--out", default="bench/results_replay/codec_longbench.csv")
    ap.add_argument("--window", type=int, default=40)
    args = ap.parse_args(argv)

    if not os.path.exists(args.inp):
        print(f"missing {args.inp}; run the LongBench-v2 fetch first", file=sys.stderr)
        return 2

    codec = RedundancyCodec(min_run=2, token_counter=_count)
    rows = []
    with open(args.inp, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            items = chunk(r["context"], args.window)
            rep = codec.analyze(items)
            rows.append({"domain": r.get("domain", ""), "items": len(items),
                         "raw_tokens": rep.tokens_in, "codec_tokens": rep.tokens_out,
                         "saved_pct": round(rep.saved_pct, 1), "factor": round(rep.factor, 3),
                         "lossless": rep.lossless})

    # per-example CSV
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["domain", "items", "raw_tokens", "codec_tokens",
                                          "saved_pct", "factor", "lossless"])
        w.writeheader()
        w.writerows(rows)

    # aggregate by domain
    by = {}
    for r in rows:
        by.setdefault(r["domain"], []).append(r)
    print(f"wrote {args.out} ({len(rows)} examples)  |  all lossless: "
          f"{all(r['lossless'] for r in rows)}")
    print(f"\n{'domain':38s} {'n':>3s} {'mean saved':>10s} {'median':>8s} {'mean x':>7s}")
    agg_out = os.path.join(os.path.dirname(args.out), "codec_longbench_bydomain.csv")
    with open(agg_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["domain", "n", "mean_saved_pct", "median_saved_pct",
                                          "mean_factor", "all_lossless"])
        w.writeheader()
        allrows = []
        for dom, rs in sorted(by.items()):
            sv = [r["saved_pct"] for r in rs]
            rec = {"domain": dom, "n": len(rs),
                   "mean_saved_pct": round(statistics.mean(sv), 1),
                   "median_saved_pct": round(statistics.median(sv), 1),
                   "mean_factor": round(statistics.mean(r["factor"] for r in rs), 2),
                   "all_lossless": all(r["lossless"] for r in rs)}
            w.writerow(rec)
            allrows.append(rec)
            print(f"{dom:38s} {rec['n']:3d} {rec['mean_saved_pct']:9.1f}% "
                  f"{rec['median_saved_pct']:7.1f}% {rec['mean_factor']:6.2f}x")
        overall = [r["saved_pct"] for r in rows]
        print(f"\n{'OVERALL':38s} {len(rows):3d} {statistics.mean(overall):9.1f}% "
              f"{statistics.median(overall):7.1f}% "
              f"{statistics.mean(r['factor'] for r in rows):6.2f}x")
    print(f"wrote {agg_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
