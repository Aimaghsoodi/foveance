#!/usr/bin/env python3
"""Codec compression ratio on the REAL RULER long-context benchmark (simonjegou/ruler), by task
family and context length. Companion to ``codec_longbench.py``: a second *public* benchmark number
for the lossless codec, with a round-trip losslessness check on every example.

RULER is the standard synthetic long-context suite, and it is a deliberately *adversarial* case for
a cross-item codec: most of its tasks plant distinct needles/keys so a model cannot shortcut them,
leaving nothing to dedup. That makes it a useful **negative control**, and the measurement confirms
it: `qa` 0.1%, `niah_multikey` 0.0%, overall median 0.0% saved -- the codec correctly finds almost
nothing and, crucially, *never inflates*. Where RULER genuinely does repeat (`vt`, variable tracking:
91.4%) the codec captures it. This is the same law as the LongBench-v2 arm from the other direction:
the saving tracks the redundancy actually present, which is large on real agent transcripts and
~zero on text engineered to have none.

NOTE for anyone re-running: sample *across* the split (``fetch_ruler.py`` strides). Reading
sequentially from offset 0 returns only ``niah_single_1``, whose haystack is one filler sentence
repeated ~150x; that yields a headline ~97.7% / 51x which is real but measures deliberate padding,
not information, and would badly misrepresent the benchmark.

Prereq: python bench/fetch_ruler.py
Usage:  python bench/codec_ruler.py --in data/ruler/ruler.jsonl \
                                    --out bench/results_replay/codec_ruler.csv
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

sys.path.insert(0, os.path.dirname(__file__))
from codec_longbench import chunk  # noqa: E402  (same agent-style chunking)


def _count(s: str) -> int:
    return max(1, len(s) // 4)


def _family(task: str) -> str:
    """Collapse RULER's task ids into their families (niah_single_1 -> niah_single)."""
    t = task.rsplit("_", 1)
    return t[0] if len(t) == 2 and t[1].isdigit() else task


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/ruler/ruler.jsonl")
    ap.add_argument("--out", default="bench/results_replay/codec_ruler.csv")
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--template", action="store_true",
                    help="also measure the opt-in shared-prefix template pass")
    args = ap.parse_args(argv)

    if not os.path.exists(args.inp):
        print(f"missing {args.inp}; run: python bench/fetch_ruler.py", file=sys.stderr)
        return 2

    codec = RedundancyCodec(min_run=1, token_counter=_count)
    tpl = RedundancyCodec(min_run=1, token_counter=_count, template=True)
    rows = []
    with open(args.inp, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            items = chunk(r["context"], args.window)
            rep = codec.analyze(items)
            rec = {"task": _family(r.get("task", "")), "length": r.get("length", ""),
                   "items": len(items), "raw_tokens": rep.tokens_in,
                   "codec_tokens": rep.tokens_out, "saved_pct": round(rep.saved_pct, 1),
                   "factor": round(rep.factor, 3), "lossless": rep.lossless}
            if args.template:
                t = tpl.analyze(items)
                rec["tpl_saved_pct"] = round(t.saved_pct, 1)
                rec["tpl_lossless"] = t.lossless
            rows.append(rec)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fields = list(rows[0].keys())
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {args.out} ({len(rows)} examples)  |  all lossless: "
          f"{all(r['lossless'] for r in rows)}  |  never inflates: "
          f"{all(r['codec_tokens'] <= r['raw_tokens'] for r in rows)}")

    by = {}
    for r in rows:
        by.setdefault(r["task"], []).append(r)
    agg_out = os.path.join(os.path.dirname(args.out), "codec_ruler_bytask.csv")
    print(f"\n{'task family':22s} {'n':>3s} {'mean saved':>10s} {'median':>8s} {'mean x':>7s}")
    with open(agg_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "n", "mean_saved_pct", "median_saved_pct",
                                          "mean_factor", "all_lossless"])
        w.writeheader()
        for task, rs in sorted(by.items()):
            sv = [r["saved_pct"] for r in rs]
            rec = {"task": task, "n": len(rs), "mean_saved_pct": round(statistics.mean(sv), 1),
                   "median_saved_pct": round(statistics.median(sv), 1),
                   "mean_factor": round(statistics.mean(r["factor"] for r in rs), 2),
                   "all_lossless": all(r["lossless"] for r in rs)}
            w.writerow(rec)
            print(f"{task:22s} {rec['n']:3d} {rec['mean_saved_pct']:9.1f}% "
                  f"{rec['median_saved_pct']:7.1f}% {rec['mean_factor']:6.2f}x")
    ov = [r["saved_pct"] for r in rows]
    print(f"\n{'OVERALL':22s} {len(rows):3d} {statistics.mean(ov):9.1f}% "
          f"{statistics.median(ov):7.1f}% {statistics.mean(r['factor'] for r in rows):6.2f}x")
    # by context length: does the saving grow with the horizon (Thm scale)?
    print(f"\n{'length':22s} {'n':>3s} {'mean saved':>10s}")
    bylen = {}
    for r in rows:
        bylen.setdefault(r["length"], []).append(r["saved_pct"])
    for ln, sv in sorted(bylen.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0):
        print(f"{str(ln):22s} {len(sv):3d} {statistics.mean(sv):9.1f}%")
    print(f"\nwrote {agg_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
