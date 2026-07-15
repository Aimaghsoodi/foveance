#!/usr/bin/env python3
"""Head-to-head compression comparison against the well-known frameworks, on identical inputs.

The comparison is split along the axis that actually decides whether a method is usable as an
LLM-agent context compressor, because mixing the two classes hides the real result:

TABLE A -- in-context compressors (output must be valid text the model can still read).
  recency         keep the last K items, elide the rest              (heuristic; lossy)
  digest          AFM-style per-item salience digest                 (Cruz 2025; lossy)
  llmlingua2      REAL LLMLingua-2, the leading learned prompt        (Microsoft; lossy)
                  compressor, matched to the codec's output size
  llmlingua2_aggr LLMLingua-2 pushed 3x tighter                       (Microsoft; lossy)
  codec           our lossless cross-item redundancy codec           (ours; LOSSLESS)
  Reported in TOKENS. Columns: saved%, legible?, lossless?, fact-preserved?.

TABLE B -- transport / storage codecs (maximal ratio, but output is opaque bytes: it CANNOT be
placed in a prompt, so it is disqualified as an in-context method and only usable for the vault).
  gzip (RFC1952), zlib (RFC1950), bz2, lzma/xz, zstd (Meta), brotli (Google), and the codec's
  own byte form. Reported in BYTES. Columns: saved%, legible?=No for byte codecs, lossless?=Yes.

The thesis this makes measurable: among methods whose output an LLM can actually read, the codec
is the only lossless one and the only one that preserves buried facts by construction; among raw
byte codecs the general-purpose ones win on ratio but produce non-legible output, so they belong
to a different problem (storage), where we adopt the best of them for the vault.

Usage: python bench/codec_compare.py --docs 12 --out bench/results_replay/codec_compare.csv
       (add --no-llmlingua to skip the heavy learned arm)
"""
from __future__ import annotations

import argparse
import bz2
import csv
import gzip
import lzma
import os
import statistics
import sys
import zlib

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


# -- TABLE A: in-context compressors (token cost; output must stay legible) -------------------
def incontext_methods(items, use_ll):
    """Return {name: (text, legible_bool, lossless_bool)} for one document."""
    codec = RedundancyCodec(min_run=1, token_counter=_count)
    raw = "\n".join(t for _, t in items)
    out = {"raw": (raw, True, True)}
    keep = 4
    rec = [t if k >= len(items) - keep else f"[{i}: older elided]"
           for k, (i, t) in enumerate(items)]
    out["recency"] = ("\n".join(rec), True, False)
    dig = [default_renderer(Item(i, "tool_output", t, 0), Fidelity.DIGEST) for i, t in items]
    out["digest"] = ("\n".join(dig), True, False)
    rendered, rep = codec.render(items)
    coded = "\n".join(t for _, t in rendered)
    lossless = codec.unpack(codec.pack(items)) == list(items)
    out["codec"] = (coded, True, lossless)
    if use_ll:
        try:
            out["llmlingua2"] = (llmlingua2_compress(raw, rep.tokens_out), True, False)
            out["llmlingua2_aggr"] = (llmlingua2_compress(raw, max(20, rep.tokens_out // 3)),
                                      True, False)
        except Exception as e:  # keep the run alive if the heavy dep misbehaves
            print(f"  llmlingua2 skipped: {type(e).__name__}: {str(e)[:70]}", flush=True)
    return out


# -- TABLE B: transport/storage byte codecs (byte cost; output is NOT legible) ----------------
def _zstd(b: bytes) -> bytes:
    import zstandard
    return zstandard.ZstdCompressor(level=19).compress(b)


def _brotli(b: bytes) -> bytes:
    import brotli
    return brotli.compress(b, quality=11)


def byte_codecs(raw_bytes: bytes, codec_legible_bytes: int):
    """Return {name: (out_bytes:int, legible_bool)} for one concatenated document."""
    out = {}
    for name, fn in [("gzip", lambda b: gzip.compress(b, 9)),
                     ("zlib", lambda b: zlib.compress(b, 9)),
                     ("bz2", lambda b: bz2.compress(b, 9)),
                     ("lzma", lambda b: lzma.compress(b))]:
        out[name] = (len(fn(raw_bytes)), False)
    try:
        out["zstd"] = (len(_zstd(raw_bytes)), False)
    except Exception:
        pass
    try:
        out["brotli"] = (len(_brotli(raw_bytes)), False)
    except Exception:
        pass
    # the codec's own byte form is legible text (it can go back in a prompt) AND lossless
    out["codec"] = (codec_legible_bytes, True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=12)
    ap.add_argument("--no-llmlingua", action="store_true")
    ap.add_argument("--out", default="bench/results_replay/codec_compare.csv")
    args = ap.parse_args(argv)
    use_ll = not args.no_llmlingua

    a_rows, b_rows = [], []
    codec = RedundancyCodec(min_run=1, token_counter=_count)
    for d in range(args.docs):
        task = make_task(d)
        items = task["items"]
        gold = task["gold"].lower().replace(" ", "")
        raw_tok = sum(_count(t) for _, t in items)
        # Table A
        for name, (text, legible, lossless) in incontext_methods(items, use_ll).items():
            tok = _count(text)
            a_rows.append({"doc": d, "method": name, "tokens": tok,
                           "saved_pct": round(100 * (1 - tok / raw_tok), 1) if raw_tok else 0.0,
                           "legible": legible, "lossless": lossless,
                           "fact_preserved": int(gold in text.lower().replace(" ", ""))})
        # Table B
        raw_bytes = "\n".join(t for _, t in items).encode("utf-8")
        rendered, _ = codec.render(items)
        codec_bytes = len("\n".join(t for _, t in rendered).encode("utf-8"))
        for name, (nb, legible) in byte_codecs(raw_bytes, codec_bytes).items():
            b_rows.append({"doc": d, "method": name, "bytes": nb,
                           "saved_pct": round(100 * (1 - nb / len(raw_bytes)), 1),
                           "legible": legible, "lossless": True})
        print(f"  doc {d}: done", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["doc", "method", "tokens", "saved_pct", "legible",
                                          "lossless", "fact_preserved"])
        w.writeheader()
        w.writerows(a_rows)
    b_out = args.out.replace(".csv", "_bytes.csv")
    with open(b_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["doc", "method", "bytes", "saved_pct", "legible",
                                          "lossless"])
        w.writeheader()
        w.writerows(b_rows)

    def _summarise(rows, order, path, extra):
        with open(path, "w", newline="") as f:
            cols = ["method", "n", "mean_saved_pct", "pct_legible", "pct_lossless"] + extra
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for name in order:
                rs = [r for r in rows if r["method"] == name]
                if not rs:
                    continue
                rec = {"method": name, "n": len(rs),
                       "mean_saved_pct": round(statistics.mean(r["saved_pct"] for r in rs), 1),
                       "pct_legible": round(100 * statistics.mean(int(r["legible"]) for r in rs)),
                       "pct_lossless": round(100 * statistics.mean(int(r["lossless"]) for r in rs))}
                for e in extra:
                    rec[e] = round(100 * statistics.mean(int(r[e.replace("pct_", "")])
                                                         for r in rs))
                w.writerow(rec)

    a_order = ["raw", "recency", "digest", "llmlingua2", "llmlingua2_aggr", "codec"]
    b_order = ["gzip", "zlib", "bz2", "lzma", "zstd", "brotli", "codec"]
    _summarise(a_rows, a_order, args.out.replace(".csv", "_summary.csv"), ["pct_fact_preserved"])
    _summarise(b_rows, b_order, b_out.replace(".csv", "_summary.csv"), [])

    print("\nTABLE A -- in-context compressors (tokens)")
    print(f"{'method':16s} {'saved':>7s} {'legible':>8s} {'lossless':>9s} {'fact kept':>10s}")
    for name in a_order:
        rs = [r for r in a_rows if r["method"] == name]
        if not rs:
            continue
        print(f"{name:16s} {statistics.mean(r['saved_pct'] for r in rs):6.1f}% "
              f"{round(100*statistics.mean(int(r['legible']) for r in rs)):7d}% "
              f"{round(100*statistics.mean(int(r['lossless']) for r in rs)):8d}% "
              f"{round(100*statistics.mean(r['fact_preserved'] for r in rs)):9d}%")
    print("\nTABLE B -- transport/storage byte codecs (bytes)")
    print(f"{'method':16s} {'saved':>7s} {'legible':>8s} {'lossless':>9s}")
    for name in b_order:
        rs = [r for r in b_rows if r["method"] == name]
        if not rs:
            continue
        print(f"{name:16s} {statistics.mean(r['saved_pct'] for r in rs):6.1f}% "
              f"{round(100*statistics.mean(int(r['legible']) for r in rs)):7d}% "
              f"{round(100*statistics.mean(int(r['lossless']) for r in rs)):8d}%")
    print(f"\nwrote {args.out}, {b_out} (+ summaries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
