#!/usr/bin/env python3
"""Fetch the REAL RULER long-context benchmark (simonjegou/ruler on the HF datasets server) to
``data/ruler/ruler.jsonl``, so the codec arm measures a public benchmark rather than a synthetic one.

RULER is the standard synthetic long-context suite (needle-in-a-haystack, multi-hop tracing,
aggregation, QA) at controlled context lengths. We keep only what a compression-ratio measurement
needs: the ``context``, its ``task`` family, and the config's nominal length. No model is called.

Usage: python bench/fetch_ruler.py --configs 4096,8192 --limit 200
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request

ROWS = "https://datasets-server.huggingface.co/rows"
DATASET = "simonjegou/ruler"


def _num_rows(config: str) -> int:
    q = urllib.parse.urlencode({"dataset": DATASET, "config": config, "split": "test",
                                "offset": 0, "length": 1})
    with urllib.request.urlopen(f"{ROWS}?{q}", timeout=60) as fh:
        return int(json.load(fh).get("num_rows_total", 0))


def fetch(config: str, limit: int) -> list:
    """Sample *across* the split, not just the head.

    The split is ordered by task, so reading sequentially from offset 0 returns a single task
    family and would misrepresent the benchmark. We stride over the whole split so every RULER task
    (niah variants, multi-hop tracing, aggregation, QA) is represented.
    """
    total = _num_rows(config)
    if not total:
        return []
    page = 20
    starts = [int(i * total / max(1, limit // page)) for i in range(max(1, limit // page))]
    out = []
    for off in starts:
        if len(out) >= limit:
            break
        q = urllib.parse.urlencode({"dataset": DATASET, "config": config, "split": "test",
                                    "offset": off, "length": min(page, limit - len(out))})
        with urllib.request.urlopen(f"{ROWS}?{q}", timeout=60) as fh:
            data = json.load(fh)
        for r in data.get("rows", []):
            row = r["row"]
            out.append({"context": row["context"], "task": row.get("task", ""),
                        "length": config})
        print(f"  {config}: {len(out)}/{limit} (offset {off}/{total})", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="4096,8192,16384")
    ap.add_argument("--limit", type=int, default=200, help="examples per config")
    ap.add_argument("--out", default="data/ruler/ruler.jsonl")
    args = ap.parse_args(argv)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for cfg in [c.strip() for c in args.configs.split(",") if c.strip()]:
            try:
                for row in fetch(cfg, args.limit):
                    f.write(json.dumps(row) + "\n")
                    n += 1
            except Exception as e:  # a config may be unavailable; keep the rest
                print(f"  {cfg}: skipped ({type(e).__name__}: {str(e)[:60]})")
    print(f"wrote {args.out} ({n} examples)")
    return 0 if n else 2


if __name__ == "__main__":
    raise SystemExit(main())
