#!/usr/bin/env python3
"""Paper-2 multi-model benchmark: buried-fact accuracy under interactive anticipatory compression.

For each model x arm x budget x query-mode it plants a fact in a long tool-use transcript, runs
the request through Foveance's compression, sends it to the model over OpenRouter, and scores
whether the model recovered the fact. The headline arm, ``allocator+expand``, additionally gives
tool-capable models the ``foveance_expand`` action and runs the real re-inflation loop.

Arms:
  raw               no compression (accuracy ceiling, token floor-cost)
  digest            v0.2 salience digestion of old payloads
  allocator         v0.3 anticipatory graded fidelities (agentic allocator + vault)
  allocator+expand  allocator + the model may call foveance_expand to restore any item (tools only)

Query modes:
  overlap   the query lexically cues the fact (salience can keep it)
  disjoint  the query is semantically related but lexically disjoint (the anticipation gap)

Budget safety: a shared CostAccountant guards every call against a hard --budget-usd cap (default
$20). If the next call would exceed it, the run stops and writes whatever it has. Real dollars come
from OpenRouter's usage.cost; the model price table is only for the pre-send estimate.

Offline: point --base-url at the mock in tests/mock_openrouter.py to validate the whole pipeline
with no key and no spend (this is what CI does).

Usage:
  OPENROUTER_API_KEY=sk-or-... python bench/paper2_bench.py \
      --budgets 400,800 --tasks 8 --seeds 3 --budget-usd 20 \
      --out bench/results_replay/paper2_accuracy.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from foveance.llm import CostAccountant, BudgetExceeded  # noqa: E402
from foveance.proxy import FoveanceProxy, _payload_text  # noqa: E402
from foveance.vault import ItemVault  # noqa: E402
from models import DEFAULT_ROSTER, resolve  # noqa: E402

_EXPAND_TOOL = {"type": "function", "function": {
    "name": "foveance_expand",
    "description": "Restore a compressed context item to full fidelity. Pass the item_id shown "
                   "in a '[compressed item <id>]' marker to get its full original text back.",
    "parameters": {"type": "object", "properties": {"item_id": {"type": "string"}},
                   "required": ["item_id"]}}}


def _noise(n: int, tag: str, seed: int) -> list[str]:
    return [f"{tag} {i}: status=ok code=200 region us-{(i + seed) % 5} latency={(i * 7) % 400}ms"
            for i in range(n)]


def make_task(seed: int, mode: str) -> dict:
    """A long tool-transcript with a fact buried mid-payload; two query phrasings."""
    secret = f"pg://svc:PW-{4000 + seed}@10.2.0.{seed % 200}/orders"
    fact_line = f"conn string: {secret}"
    lines = _noise(150, "cfg", seed)
    lines[75] = fact_line
    payload1 = "\n".join(lines)
    payload2 = "\n".join(_noise(200, "log", seed + 1))
    query = ("what is the exact conn string in the config?" if mode == "overlap"
             else "which upstream datastore does the service authenticate against? "
                  "give the exact value.")
    tools = [{"name": "bash", "description": "run a shell command",
              "input_schema": {"type": "object", "properties": {"cmd": {"type": "string"}}}}]
    messages = [
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                                      "content": payload1}]},
        {"role": "assistant", "content": [{"type": "text", "text": "read config"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2",
                                      "content": payload2}]},
        {"role": "assistant", "content": [{"type": "text", "text": "checked logs"}]},
        {"role": "user", "content": [{"type": "text", "text": query}]},
    ]
    return {"secret": secret, "system": "You are a precise SRE. Answer with only the exact value.",
            "messages": messages, "tools": tools}


def _anthropic_to_openai(system: str, messages: list) -> list:
    """Flatten Anthropic-shaped (compressed) messages to OpenAI chat messages for OpenRouter.
    Text and tool_result blocks become plain text; the structure is already compressed by now."""
    out = [{"role": "system", "content": system}]
    for m in messages:
        c = m["content"]
        if isinstance(c, str):
            out.append({"role": m["role"], "content": c})
            continue
        parts = []
        for blk in c:
            if blk.get("type") == "text":
                parts.append(blk["text"])
            elif blk.get("type") == "tool_result":
                parts.append(str(blk.get("content", "")))
            elif blk.get("type") == "tool_use":
                parts.append(f"[called {blk.get('name')}]")
        role = "assistant" if m["role"] == "assistant" else "user"
        out.append({"role": role, "content": "\n".join(parts)})
    return out


def _chat(base_url, api_key, model, messages, accountant, prices, tools=None,
          num_predict=64, timeout=90.0):  # pragma: no cover - network in real runs
    """One OpenRouter chat call, cost-guarded. Returns (text, tool_calls, cost, in_tok)."""
    est = sum(len(str(m.get("content", ""))) for m in messages) // 4 / 1e6 * prices[0]
    accountant.guard(est)
    body = {"model": model, "messages": messages, "temperature": 0.0,
            "max_tokens": num_predict, "usage": {"include": True}}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}",
                 "HTTP-Referer": "https://github.com/Aimaghsoodi/foveance",
                 "X-Title": "Foveance paper2"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    msg = data["choices"][0]["message"]
    usage = data.get("usage", {}) or {}
    it = usage.get("prompt_tokens", sum(len(str(m.get("content", ""))) for m in messages) // 4)
    cost = usage.get("cost")
    if cost is None:
        ot = usage.get("completion_tokens", num_predict)
        cost = it / 1e6 * prices[0] + ot / 1e6 * prices[1]
    accountant.record(model, float(cost))
    return msg.get("content") or "", msg.get("tool_calls") or [], float(cost), it


def compress(task: dict, budget: int, arm: str, vault):
    """Return (openai_messages, prompt_tokens_est) for the arm. raw = uncompressed."""
    req = {"system": task["system"], "messages": [json.loads(json.dumps(m))
                                                   for m in task["messages"]],
           "tools": task["tools"], "user": "p2"}
    if arm == "raw":
        fwd = req
    else:
        agentic = arm in ("allocator", "allocator+expand")
        px = FoveanceProxy(budget=budget, agentic_protect_last=1, agentic_allocator=agentic,
                           expand_tool=(arm == "allocator+expand"), vault=vault)
        fwd, _ = px.prepare_anthropic(req)
    msgs = _anthropic_to_openai(fwd.get("system", task["system"]), fwd["messages"])
    return msgs, len(_payload_text(fwd)) // 4


def run_cell(base_url, api_key, model_spec, arm, budget, mode, seeds, accountant,
             out_of_budget):  # pragma: no cover - exercised via mock in tests
    """One (model, arm, budget, mode) cell across seeds -> a result row."""
    correct = n = expands = 0
    ptoks = 0
    for seed in range(seeds):
        if out_of_budget["hit"]:
            break
        task = make_task(seed, mode)
        with_expand = arm == "allocator+expand"
        if with_expand and not model_spec.tools:
            continue  # expand arm needs tool calling
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            vault = ItemVault(path=os.path.join(td, "v.db"))
            msgs, pt = compress(task, budget, arm, vault)
            ptoks += pt
            prices = (model_spec.price_in, model_spec.price_out)
            try:
                tools = [_EXPAND_TOOL] if with_expand else None
                text, calls, _, _ = _chat(base_url, api_key, model_spec.slug, msgs,
                                          accountant, prices, tools=tools)
                # bounded expand loop: resolve foveance_expand from the vault, feed it back
                loops = 0
                while calls and with_expand and loops < 3:
                    tool_msgs = []
                    for call in calls:
                        fn = call.get("function", {})
                        if fn.get("name") != "foveance_expand":
                            continue
                        args = json.loads(fn.get("arguments") or "{}")
                        full = vault.get_any(args.get("item_id", "")) or "(not found)"
                        expands += 1
                        tool_msgs.append({"role": "tool", "tool_call_id": call.get("id", "x"),
                                          "content": full})
                    if not tool_msgs:
                        break
                    msgs = msgs + [{"role": "assistant", "content": None, "tool_calls": calls}]
                    msgs = msgs + tool_msgs
                    text, calls, _, _ = _chat(base_url, api_key, model_spec.slug, msgs,
                                              accountant, prices, tools=tools)
                    loops += 1
            except BudgetExceeded:
                out_of_budget["hit"] = True
                break
            n += 1
            correct += int(task["secret"] in (text or ""))
            # post-call stop: the pre-send guard uses an estimate, so also halt the instant
            # ACTUAL recorded spend reaches the cap -- overshoot is then at most one call's cost.
            if accountant.spent_usd >= accountant.budget_usd:
                out_of_budget["hit"] = True
                break
    if n == 0:
        return None
    return {"model": model_spec.label, "slug": model_spec.slug, "arm": arm, "budget": budget,
            "query_mode": mode, "n": n, "accuracy": round(correct / n, 4),
            "mean_prompt_tokens": round(ptoks / max(n, 1), 1),
            "expand_calls": expands, "cost_usd": round(accountant.by_model.get(model_spec.slug,
                                                                              0.0), 5)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="comma list of OpenRouter slugs (default roster)")
    ap.add_argument("--arms", default="raw,digest,allocator,allocator+expand")
    ap.add_argument("--budgets", default="400,800")
    ap.add_argument("--modes", default="overlap,disjoint")
    ap.add_argument("--tasks", type=int, default=8, help="seeds per cell")
    ap.add_argument("--seeds", type=int, default=None, help="alias for --tasks")
    ap.add_argument("--budget-usd", type=float, default=20.0)
    ap.add_argument("--base-url", default=os.environ.get("OPENROUTER_BASE_URL",
                                                         "https://openrouter.ai/api/v1"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "results_replay", "paper2_accuracy.csv"))
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and "openrouter.ai" in args.base_url:
        print("set OPENROUTER_API_KEY (or point --base-url at the offline mock)", file=sys.stderr)
        return 2
    specs = resolve(args.models.split(",")) if args.models else DEFAULT_ROSTER
    seeds = args.seeds or args.tasks
    budgets = [int(b) for b in args.budgets.split(",")]
    modes = args.modes.split(",")
    arms = args.arms.split(",")
    accountant = CostAccountant(budget_usd=args.budget_usd)
    out_of_budget = {"hit": False}

    rows = []
    for spec in specs:
        for mode in modes:
            for budget in budgets:
                for arm in arms:
                    if out_of_budget["hit"]:
                        break
                    row = run_cell(args.base_url, api_key, spec, arm, budget, mode, seeds,
                                   accountant, out_of_budget)
                    if row:
                        rows.append(row)
                        print(f"{row['model']:<14} {arm:<17} b={budget:<5} {mode:<8} "
                              f"acc={row['accuracy']:<6} ptok={row['mean_prompt_tokens']:<7} "
                              f"exp={row['expand_calls']:<3} ${accountant.spent_usd:.4f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if rows:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} rows)")
    print(f"total spend: ${accountant.spent_usd:.4f} / ${args.budget_usd:.2f} cap"
          f"{'  [STOPPED: budget hit]' if out_of_budget['hit'] else ''}")
    print("by model:", accountant.by_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
