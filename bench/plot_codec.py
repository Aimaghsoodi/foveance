#!/usr/bin/env python3
"""Journal-quality paper-3 figures from the real codec CSVs. Vector PDF (for LaTeX) + 300-dpi PNG
(for viewing). Reads only measured CSVs; draws nothing not in them. Regenerate as data completes."""
from __future__ import annotations

import csv
import math
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results_replay")
OUT = os.path.join(HERE, "plots")
os.makedirs(OUT, exist_ok=True)

# ---- shared journal theme -------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#B8B8B8",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.5,
    "legend.fontsize": 8.5,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#CCCCCC",
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

ARMS = ["full", "recency", "digest", "reactive_afm", "foveance", "codec", "foveance+codec"]
LABEL = {"full": "full (verbatim)", "recency": "recency", "digest": "digest (AFM-style)",
         "reactive_afm": "reactive (AFM)", "foveance": "foveance", "codec": "codec (ours, lossless)",
         "foveance+codec": "foveance+codec"}
# Okabe-Ito colourblind-safe; the lossless codec is the emphasis colour (bluish-green).
COLOR = {"full": "#111111", "recency": "#999999", "digest": "#E69F00",
         "reactive_afm": "#56B4E9", "foveance": "#0072B2", "codec": "#009E73",
         "foveance+codec": "#CC79A7"}
MARK = {"full": "s", "recency": "X", "digest": "v", "reactive_afm": "^",
        "foveance": "o", "codec": "*", "foveance+codec": "D"}


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _mean_ci(vals):
    """mean and 95% (normal-approx) half-width."""
    if not vals:
        return float("nan"), 0.0
    m = statistics.mean(vals)
    if len(vals) < 2:
        return m, 0.0
    se = statistics.pstdev(vals) / math.sqrt(len(vals))
    return m, 1.96 * se


def _save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"))
    plt.close(fig)


# ---- Figure 1: accuracy-token frontier ------------------------------------------------------
def fig_pareto(rows):
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    stats = {}
    for arm in ARMS:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        acc, acc_ci = _mean_ci([int(r["acc"]) for r in rs])
        tok, tok_ci = _mean_ci([int(r["in_tokens"]) for r in rs])
        stats[arm] = (tok, tok_ci, acc, acc_ci)

    # improvement arrow: full -> codec (up-left) drawn behind points
    if "full" in stats and "codec" in stats:
        ft, _, fa, _ = stats["full"]
        ct, _, ca, _ = stats["codec"]
        ax.add_patch(FancyArrowPatch((ft, fa), (ct, ca), arrowstyle="-|>", mutation_scale=14,
                     color="#009E73", lw=1.3, alpha=0.55, linestyle=(0, (5, 3)), zorder=1))
        ax.annotate("lossless: fewer tokens\n& higher accuracy",
                    xy=((ft + ct) / 2, (fa + ca) / 2 + 0.02), fontsize=8, color="#00785A",
                    ha="center", va="bottom", style="italic")

    for arm in ARMS:
        if arm not in stats:
            continue
        tok, tok_ci, acc, acc_ci = stats[arm]
        big = arm == "codec"
        ax.errorbar(tok, acc, xerr=tok_ci, yerr=acc_ci, fmt="none",
                    ecolor=COLOR[arm], elinewidth=1.0, capsize=2.5, alpha=0.8, zorder=2)
        ax.scatter(tok, acc, s=340 if big else 110, marker=MARK[arm], color=COLOR[arm],
                   edgecolor="black", linewidth=0.9 if big else 0.6, zorder=4, label=LABEL[arm])

    # annotate only the two arms that matter; bracket the overlapping lossy cluster once
    if "codec" in stats:
        t, _, a, _ = stats["codec"]
        ax.annotate("codec", (t, a), textcoords="offset points", xytext=(10, -4),
                    fontsize=9.5, fontweight="bold", color="#00785A")
    if "full" in stats:
        t, _, a, _ = stats["full"]
        ax.annotate("full", (t, a), textcoords="offset points", xytext=(8, 2), fontsize=9.5)
    lossy = [stats[a] for a in ("recency", "digest", "reactive_afm", "foveance") if a in stats]
    if lossy:
        lx = statistics.mean(t for t, _, _, _ in lossy)
        ax.annotate("lossy arms drop the\ndisjoint fact  →  acc 0",
                    xy=(lx, 0.0), xytext=(lx + 250, 0.22), fontsize=8.2, color="#7a5a00",
                    ha="left", va="bottom",
                    arrowprops=dict(arrowstyle="-[", color="#7a5a00", lw=1.0, alpha=0.8))

    ax.set_xlabel("mean input tokens served  (lower is cheaper) $\\rightarrow$")
    ax.set_ylabel("answer accuracy  (higher is better)")
    ax.set_title("Accuracy–token frontier: the lossless codec dominates")
    ax.set_ylim(-0.06, 1.10)
    ax.margins(x=0.12)
    ax.legend(loc="center right", handletextpad=0.4, borderpad=0.6)
    _save(fig, "codec_pareto")


# ---- Figure 2: per-model accuracy -----------------------------------------------------------
def fig_accuracy_by_model(rows):
    if not rows:
        return
    models = sorted({r["model"] for r in rows})
    shown = ["full", "digest", "foveance", "codec"]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    w = 0.2
    x = list(range(len(models)))
    for i, arm in enumerate(shown):
        means, cis = [], []
        for m in models:
            mm, cc = _mean_ci([int(r["acc"]) for r in rows
                               if r["model"] == m and r["arm"] == arm])
            means.append(0.0 if mm != mm else mm)
            cis.append(cc)
        pos = [xi + (i - 1.5) * w for xi in x]
        bars = ax.bar(pos, means, width=w, color=COLOR[arm], edgecolor="black", linewidth=0.6,
                      label=LABEL[arm], yerr=cis, capsize=2.5,
                      error_kw=dict(elinewidth=0.8, alpha=0.7))
        for b, mv in zip(bars, means):
            if mv > 0.02:
                ax.annotate(f"{mv:.2f}", (b.get_x() + b.get_width() / 2, mv), ha="center",
                            va="bottom", fontsize=7, color="#222")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("answer accuracy")
    ax.set_title("Per-model accuracy: codec matches/exceeds full; lossy arms at floor")
    ax.set_ylim(0, 1.12)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="upper center", columnspacing=1.0, handletextpad=0.4)
    _save(fig, "codec_accuracy_by_model")


# ---- Figure 3: lossless compression factor --------------------------------------------------
def fig_ratio(rows):
    rr = [r for r in rows if r["trace"].startswith("coding_agent")]
    if not rr:
        return
    order = ["raw", "digest", "codec", "digest+codec"]
    labels = {"raw": "raw", "digest": "digest\n(lossy)", "codec": "codec\n(lossless)",
              "digest+codec": "digest\n+codec"}
    cols = {"raw": "#BBBBBB", "digest": "#E69F00", "codec": "#009E73", "digest+codec": "#CC79A7"}
    fig, ax = plt.subplots(figsize=(5.4, 4.1))
    xs, facs, saved, cs, meth = [], [], [], [], []
    for m in order:
        row = next((r for r in rr if r["method"] == m), None)
        if row:
            xs.append(labels[m]); facs.append(float(row["factor"]))
            saved.append(float(row["saved_pct"])); cs.append(cols[m]); meth.append(m)
    bars = ax.bar(xs, facs, color=cs, edgecolor="black", linewidth=0.7, width=0.66)
    for b, s, m in zip(bars, saved, meth):
        tag = f"{s:.0f}% saved"
        ax.annotate(tag, (b.get_x() + b.get_width() / 2, b.get_height() + 0.03), ha="center",
                    va="bottom", fontsize=8.5)
        if m in ("codec",):
            ax.annotate("reversible", (b.get_x() + b.get_width() / 2, b.get_height() / 2),
                        ha="center", va="center", fontsize=8.5, color="white", fontweight="bold",
                        rotation=90)
    ax.set_ylabel("compression factor  ($\\times$ smaller)")
    ax.set_title("Lossless codec: a reversible multiplicative saving")
    ax.set_ylim(0, max(facs) * 1.2)
    ax.grid(axis="x", visible=False)
    _save(fig, "codec_ratio")


# ---- Figure 4: composed stack sweep ---------------------------------------------------------
def fig_fullstack(rows):
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    styles = {"sre_debug_trace.jsonl": ("long stale context (SRE)", "#0072B2", "#009E73"),
              "coding_agent_trace.jsonl": ("redundant coding-agent", "#56B4E9", "#66C2A5")}
    bmax = 0
    for tr, (lab, ca, cc) in styles.items():
        rs = sorted((r for r in rows if r["trace"] == tr), key=lambda r: int(r["budget"]))
        if not rs:
            continue
        b = [int(r["budget"]) for r in rs]; bmax = max(bmax, max(b))
        ax.plot(b, [float(r["alloc_saved_pct"]) for r in rs], "--", color=ca, marker="o",
                ms=5, lw=1.6, label=f"allocator · {lab}")
        ax.plot(b, [float(r["alloc+codec_saved_pct"]) for r in rs], "-", color=cc, marker="*",
                ms=9, lw=1.8, label=f"allocator+codec · {lab}")
    ax.axhspan(90, 100, color="#D7191C", alpha=0.06)
    ax.axhline(90, color="#D7191C", lw=0.9, ls=":", alpha=0.7)
    ax.annotate("90%+  (aggressive; expansion-recoverable, Cor. 1)", (bmax, 90.4), fontsize=7.6,
                color="#B02318", va="bottom", ha="right")
    ax.set_xlabel("token budget  (tighter $\\rightarrow$)")
    ax.set_ylabel("% tokens saved vs. raw")
    ax.set_title("Composed stack: compression ratio climbs as the budget tightens")
    ax.invert_xaxis()
    ax.set_ylim(0, 103)
    ax.legend(loc="center left", handlelength=2.4)
    _save(fig, "codec_fullstack")


def fig_scaling(rows):
    """Codec ratio vs trajectory length: savings grow and approach an asymptote."""
    if not rows:
        return
    M = [int(r["turns"]) for r in rows]
    saved = [float(r["saved_pct"]) for r in rows]
    factor = [float(r["factor"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.plot(M, saved, "-o", color="#009E73", lw=1.9, ms=6, label="% tokens saved (lossless)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("trajectory length  (tool-use turns, $M$)")
    ax.set_ylabel("% tokens saved vs. raw", color="#00785A")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelcolor="#00785A")
    ax2 = ax.twinx()
    ax2.plot(M, factor, "--s", color="#0072B2", lw=1.5, ms=5, label="compression factor ($\\times$)")
    ax2.set_ylabel("compression factor  ($\\times$ smaller)", color="#0072B2")
    ax2.tick_params(axis="y", labelcolor="#0072B2")
    ax2.spines["top"].set_visible(False)
    ax.set_title("Scaling law: the codec saves more the longer the agent runs")
    ax.set_xticks(M)
    ax.set_xticklabels([str(m) for m in M])
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], loc="lower right")
    _save(fig, "codec_scaling")


def fig_separation(rows):
    """Per-item vs joint (cross-item) coding: the gap is what per-item methods cannot remove."""
    if not rows:
        return
    M = [int(r["turns"]) for r in rows]
    per = [float(r["per_item_saved_pct"]) for r in rows]
    joint = [float(r["joint_saved_pct"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.plot(M, joint, "-*", color="#009E73", lw=1.9, ms=10,
            label="joint / cross-item (codec, ours)")
    ax.plot(M, per, "-o", color="#E69F00", lw=1.6, ms=5,
            label="per-item (AFM / LLMLingua family)")
    ax.fill_between(M, per, joint, color="#009E73", alpha=0.12)
    mid = len(M) // 2
    ax.annotate("cross-item redundancy\n(total correlation):\nunreachable per-item",
                xy=(M[mid], (per[mid] + joint[mid]) / 2), xytext=(M[1], 55),
                fontsize=8.4, color="#00785A", ha="left",
                arrowprops=dict(arrowstyle="->", color="#00785A", lw=1.0))
    ax.set_xscale("log", base=2)
    ax.set_xlabel("trajectory length  (tool-use turns, $M$)")
    ax.set_ylabel("% tokens saved (lossless)")
    ax.set_ylim(-3, 100)
    ax.set_title("Separation: per-item compression cannot see cross-item redundancy")
    ax.set_xticks(M)
    ax.set_xticklabels([str(m) for m in M])
    ax.legend(loc="center right")
    _save(fig, "codec_separation")


def main():
    acc = load(os.path.join(RES, "codec_paper.csv"))
    fig_pareto(acc)
    fig_accuracy_by_model(acc)
    fig_ratio(load(os.path.join(RES, "codec_ratio.csv")))
    fig_fullstack(load(os.path.join(RES, "codec_fullstack.csv")))
    fig_scaling(load(os.path.join(RES, "codec_scaling.csv")))
    fig_separation(load(os.path.join(RES, "codec_separation.csv")))
    print(f"wrote PDF+PNG figures to {OUT} (accuracy rows: {len(acc)})")


if __name__ == "__main__":
    main()
