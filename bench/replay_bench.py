#!/usr/bin/env python3
"""Replay benchmark: compare compression arms on RECORDED conversations (the audit format),
answering the "does this hold beyond synthetic suites?" objection with the user's own traffic.

Arms:
  raw        no compression (what you pay today)
  digest     v0.2 behaviour: structural salience digestion of old payloads
  allocator  v0.3 behaviour: anticipatory graded fidelities (agentic allocator + vault)

Usage:
  python bench/replay_bench.py LOGFILE [--budget 2000] [--out bench/results_replay/replay.csv]

Token accounting is chars/4 (or tiktoken via --exact-tokens). This measures TOKENS faithfully on
real traces; accuracy claims still require probes with gold answers (see --probe-note). Nothing
is fabricated: every row is written from an actual replay.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from foveance.audit import load_conversations  # noqa: E402
from foveance.proxy import FoveanceProxy, _payload_text  # noqa: E402
from foveance.vault import ItemVault  # noqa: E402


def replay_arm(convs: list, arm: str, budget: int, count) -> dict:
    before = after = requests = 0
    with tempfile.TemporaryDirectory() as td:
        for ci, conv in enumerate(convs):
            kwargs: dict = {"budget": budget}
            if arm == "allocator":
                kwargs.update(agentic_allocator=True,
                              vault=ItemVault(path=os.path.join(td, f"v{ci}.db")))
            px = FoveanceProxy(**kwargs)
            msgs = conv.get("messages") or []
            for k in range(1, len(msgs) + 1):
                if msgs[k - 1].get("role") not in (None, "user"):
                    continue
                req: dict = {"messages": msgs[:k], "user": f"replay-{ci}"}
                if conv.get("system") is not None:
                    req["system"] = conv["system"]
                if conv.get("tools"):
                    req["tools"] = conv["tools"]
                if arm == "raw":
                    fwd = req
                else:
                    fwd, _ = (px.prepare_anthropic(dict(req)) if "system" in req
                              else px.prepare(dict(req)))
                before += count(_payload_text(req))
                after += count(_payload_text(fwd))
                requests += 1
    saved = max(before - after, 0)
    return {"arm": arm, "requests": requests, "tokens_before": before, "tokens_after": after,
            "tokens_saved": saved,
            "saved_pct": round(100.0 * saved / before, 1) if before else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile")
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "results_replay", "replay.csv"))
    ap.add_argument("--exact-tokens", action="store_true")
    args = ap.parse_args()

    convs = load_conversations(args.logfile)
    if not convs:
        print("no conversations found", file=sys.stderr)
        return 2
    if args.exact_tokens:
        from foveance.metrics import make_token_counter
        counter = make_token_counter()
        count = counter
    else:
        def count(text: str) -> int:
            return len(text) // 4

    rows = [replay_arm(convs, arm, args.budget, count)
            for arm in ("raw", "digest", "allocator")]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{'arm':<10} {'requests':>8} {'before':>12} {'after':>12} {'saved%':>7}")
    for r in rows:
        print(f"{r['arm']:<10} {r['requests']:>8} {r['tokens_before']:>12,} "
              f"{r['tokens_after']:>12,} {r['saved_pct']:>6}%")
    print(f"\nwrote {args.out}")
    print("--probe-note: token savings above are measured on your real traces; pair with gold-"
          "answer probes (bench/compare_baselines.py style) before making accuracy claims.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
