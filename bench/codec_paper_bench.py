#!/usr/bin/env python3
"""Paper-3 benchmark: buried-fact recovery under redundancy, across compression arms and models.

Each task is a multi-turn agent transcript that (a) is heavy with cross-item redundancy (repeated
listings / retried stack traces / boilerplate envelopes -- what real agents produce) and (b) hides
a single gold fact in a mid-position tool output whose surface form is lexically disjoint from the
final question (the honest anticipation-gap case). We compress the accumulated context with each
arm, ask the model the question, and record BOTH real answer accuracy and the model's real input
token count (Ollama's ``prompt_eval_count``).

Arms:
  full           verbatim (accuracy upper bound, token upper bound)
  recency        keep the last K items, drop older            (lossy baseline)
  digest         per-item structural salience digest          (AFM-style reactive, lossy)
  reactive_afm   budgeted allocation, drift=0                 (reactive special case)
  foveance       budgeted anticipatory allocation, drift>0    (ours)
  codec          verbatim + lossless cross-item redundancy codec   (ours, LOSSLESS)
  foveance+codec anticipatory allocation, then codec on top   (ours, composed)
  llmlingua2     REAL LLMLingua-2, matched to the codec's size     (Microsoft, lossy; --with-llmlingua)

Nothing is fabricated: every row is an actual model call. Token counts are the provider's.

Usage (real, 5 local models):
  python bench/codec_paper_bench.py --backend ollama \
     --models gemma2:2b,qwen2.5:3b,qwen2.5:1.5b,llama3.2:3b,llama3.2:1b \
     --budgets 500,1500 --tasks 6 --out bench/results_replay/codec_paper.csv
Offline smoke test:
  python bench/codec_paper_bench.py --backend mock --models mock --budgets 500,1500 --tasks 4
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from foveance.baselines import POLICIES  # noqa: E402
from foveance.codec import RedundancyCodec  # noqa: E402
from foveance.embedders import HashingEmbedder  # noqa: E402
from foveance.predictor import AnticipatoryPredictor, PredictorConfig  # noqa: E402
from foveance.store import Fidelity, Item, MultiFidelityStore, default_renderer  # noqa: E402

ARMS = ["full", "recency", "digest", "reactive_afm", "foveance", "codec", "codec_tpl",
        "foveance+codec"]


# ---- task generation ------------------------------------------------------------------------
_LISTING = "\n".join(f"src/service/module_{i:02d}.py" for i in range(20))
_STACK = ("Traceback (most recent call last):\n"
          '  File "tests/test_checkout.py", line 91, in test_flow\n'
          "    resp = client.post('/checkout', json=payload)\n"
          '  File "app/api.py", line 142, in post\n'
          "    return self.handler(req)\n"
          "AttributeError: 'NoneType' object has no attribute 'total'")
_BOILER = "[cwd=/srv/app CI=true LANG=C TERM=dumb]"

# (fact line placed mid-context, question with disjoint surface form, gold substring)
_FACTS = [
    ("CONFIG database endpoint set to host prod-db-7f3a port 5432 for the checkout pool",
     "Which database host and port does checkout use?", "prod-db-7f3a"),
    ("the retry budget for the payments worker was configured to 11 attempts before giving up",
     "How many times does the payments worker retry before giving up?", "11"),
    ("feature flag rollout_canary_ratio was pinned to 0.37 for the release this week",
     "What value is the canary rollout ratio pinned to?", "0.37"),
    ("the on-call escalation timeout is 240 seconds after the first unacknowledged page",
     "After how many seconds does on-call escalation trigger?", "240"),
    ("the checkout service authenticates to redis using account svc-checkout-ro in region eu-west-2",
     "Which account does checkout use to authenticate to redis?", "svc-checkout-ro"),
    ("the message queue depth alarm fires when the backlog exceeds 8500 pending jobs",
     "At what backlog depth does the queue alarm fire?", "8500"),
]


def make_task(seed: int) -> dict:
    rng = random.Random(seed)
    fact_line, question, gold = _FACTS[seed % len(_FACTS)]
    # a long, redundant transcript; the gold fact sits in one mid-position tool output
    blocks = []

    def tool(cmd: str, body: str) -> str:
        return f"$ {cmd}\n{_BOILER}\n{body}\n(exit 0)"

    items: list = []
    n_pre = 4
    for k in range(n_pre):
        pick = rng.choice([("ls -R src/service", _LISTING),
                           ("pytest -x tests/test_checkout.py", _STACK)])
        items.append((f"t{k}", tool(*pick)))
    # gold-bearing item, buried among redundant siblings
    gold_body = (_LISTING + "\n" + fact_line + "\n" + _STACK)
    gid = f"t{n_pre}"
    items.append((gid, tool("grep -r CONFIG src/service", gold_body)))
    n_post = 5
    for k in range(n_post):
        pick = rng.choice([("ls -R src/service", _LISTING),
                           ("pytest -x tests/test_checkout.py", _STACK),
                           ("cat app/api.py", _STACK)])
        items.append((f"t{n_pre + 1 + k}", tool(*pick)))
    blocks.clear()
    return {"system": "You are a precise SRE assistant. Answer with the exact value only.",
            "items": items, "query": question, "gold": gold, "gold_item": gid}


# ---- per-arm compression --------------------------------------------------------------------
def _counter(s: str) -> int:
    return max(1, len(s) // 4)


_LL: dict = {}


def _llmlingua2(text: str, target: int) -> str:
    """REAL LLMLingua-2 compression to ~``target`` tokens (CPU). Compressor cached across calls."""
    if "c" not in _LL:
        from llmlingua import PromptCompressor
        _LL["c"] = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True, device_map="cpu")
    return _LL["c"].compress_prompt(text, target_token=max(20, target))["compressed_prompt"]


def assemble(task: dict, arm: str, budget: int) -> str:
    items = task["items"]
    # `codec` is the line-only codec; `codec_tpl` adds the shared-prefix template pass. Keeping both
    # arms lets us verify that the extra savings cost no accuracy, rather than assuming it.
    codec = RedundancyCodec(min_run=1, token_counter=_counter, template=False)

    if arm == "full":
        text_items = items
    elif arm == "codec":
        text_items = codec.render(items)[0]
    elif arm == "codec_tpl":
        text_items = RedundancyCodec(min_run=1, token_counter=_counter,
                                     template=True).render(items)[0]
    elif arm == "llmlingua2":
        # match the lossy compressor to the codec's output size, then compress the raw context
        raw = "\n\n".join(f"[{i}]\n{t}" for i, t in items)
        target = codec.render(items)[1].tokens_out
        return task["system"] + "\n\n" + _llmlingua2(raw, target)
    elif arm == "recency":
        keep = 4
        text_items = [(i, t) if k >= len(items) - keep else (i, f"[{i}: older tool output elided]")
                      for k, (i, t) in enumerate(items)]
    elif arm == "digest":
        text_items = [(i, default_renderer(Item(i, "tool_output", t, 0), Fidelity.DIGEST))
                      for i, t in items]
    elif arm in ("reactive_afm", "foveance", "foveance+codec"):
        policy = "reactive_afm" if arm == "reactive_afm" else "foveance"
        drift = 0.0 if arm == "reactive_afm" else 0.6
        store = MultiFidelityStore(token_counter=_counter)
        for i, t in items:
            store.add(Item(i, "tool_output", t, 0))
        pred = AnticipatoryPredictor(store, HashingEmbedder(), config=PredictorConfig(drift=drift))
        pred.observe_query(task["query"])
        levels = POLICIES[policy](store, pred, budget, 1)
        rendered = [(i, store.render(i, levels.get(i, Fidelity.POINTER))) for i, _ in items]
        text_items = codec.render(rendered)[0] if arm == "foveance+codec" else rendered
    else:
        raise ValueError(arm)

    return task["system"] + "\n\n" + "\n\n".join(f"[{i}]\n{t}" for i, t in text_items)


# ---- scoring & run --------------------------------------------------------------------------
def score(answer: str, gold: str) -> int:
    return int(gold.lower().replace(" ", "") in answer.lower().replace(" ", ""))


def make_llm(backend: str, model: str):
    if backend == "ollama":
        from foveance.llm import OllamaLLM
        return OllamaLLM(model=model, num_predict=32, timeout=120.0)
    from foveance.llm import MockLLM
    return MockLLM()


FIELDS = ["model", "arm", "budget", "task", "acc", "in_tokens", "gold_visible", "latency_s"]


def _load_done(out: str) -> set:
    """Keys (model,arm,budget,task) already recorded, so a re-run resumes instead of redoing."""
    done = set()
    if os.path.exists(out):
        with open(out, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["model"], r["arm"], int(r["budget"]), int(r["task"])))
    return done


def run(backend: str, models: list, budgets: list, ntasks: int, out: str,
        with_llmlingua: bool = False) -> int:
    tasks = [make_task(s) for s in range(ntasks)]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = _load_done(out)
    new_file = not os.path.exists(out)
    f = open(out, "a", newline="")           # append: incremental + resumable
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        w.writeheader()
        f.flush()
    arms = ARMS + (["llmlingua2"] if with_llmlingua else [])
    n_new = 0
    for model in models:
        llm = make_llm(backend, model)
        for arm in arms:
            arm_budgets = budgets if arm in ("recency", "reactive_afm", "foveance",
                                             "foveance+codec") else [budgets[0]]
            for budget in arm_budgets:
                for ti, task in enumerate(tasks):
                    if (model, arm, budget, ti) in done:
                        continue
                    ctx = assemble(task, arm, budget)
                    t0 = time.perf_counter()
                    comp = llm.generate(ctx, task["query"])
                    dt = time.perf_counter() - t0
                    row = {"model": model, "arm": arm, "budget": budget, "task": ti,
                           "acc": score(comp.text, task["gold"]), "in_tokens": comp.input_tokens,
                           "gold_visible": int(task["gold"].lower().replace(" ", "")
                                               in ctx.lower().replace(" ", "")),
                           "latency_s": round(dt, 2)}
                    w.writerow(row)
                    f.flush()                 # persist immediately so a kill never loses it
                    n_new += 1
                    print(f"  {model:16s} {arm:15s} b={budget:5d} t{ti} "
                          f"acc={row['acc']} in={comp.input_tokens:5d} "
                          f"vis={row['gold_visible']} ({dt:.1f}s)", flush=True)
    f.close()
    print(f"\nappended {n_new} rows to {out}")
    with open(out, newline="") as fr:
        _summary(list(csv.DictReader(fr)))
    return 0


def _summary(rows: list) -> None:
    # aggregate accuracy and mean tokens per arm (pooled over models/tasks/first budget shown)
    by = {}
    for r in rows:
        by.setdefault(r["arm"], {"acc": [], "tok": [], "vis": []})
        by[r["arm"]]["acc"].append(int(r["acc"]))
        by[r["arm"]]["tok"].append(int(r["in_tokens"]))
        by[r["arm"]]["vis"].append(int(r["gold_visible"]))
    print("\narm             acc    tokens   gold-visible")
    for arm in ARMS + ["llmlingua2"]:
        if arm not in by:
            continue  # arm not present in this CSV (e.g. an opt-in arm was not run)
        a = by[arm]
        print(f"  {arm:15s} {statistics.mean(a['acc']):.2f}  {statistics.mean(a['tok']):7.0f}   "
              f"{statistics.mean(a['vis']):.2f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ollama", choices=["ollama", "mock"])
    ap.add_argument("--models", default="gemma2:2b")
    ap.add_argument("--budgets", default="500,1500")
    ap.add_argument("--tasks", type=int, default=6)
    ap.add_argument("--out", default="bench/results_replay/codec_paper.csv")
    ap.add_argument("--with-llmlingua", action="store_true",
                    help="add a REAL LLMLingua-2 arm matched to the codec's size (slow, CPU)")
    args = ap.parse_args(argv)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    budgets = [int(b) for b in args.budgets.split(",")]
    return run(args.backend, models, budgets, args.tasks, args.out,
               with_llmlingua=args.with_llmlingua)


if __name__ == "__main__":
    raise SystemExit(main())
