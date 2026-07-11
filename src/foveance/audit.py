"""``foveance audit`` -- replay your own conversation logs offline and report what Foveance
would have saved. No API key, no network, no risk: the compressor runs locally on the log and
only token accounting comes out.

Accepted input (auto-detected):
  * JSONL: one conversation per line, each either ``{"messages": [...], "system"?: ..., "tools"?: ...}``
    or a bare ``[...]`` messages list.
  * JSON: a single conversation object/list, or a list of conversation objects.

The replay is *per turn*: agents re-send the whole history on every user turn, so the audit
simulates exactly that -- each user turn costs its full prefix, compressed vs raw. That is the
honest model of how the bill actually accrues.
"""
from __future__ import annotations

import json
from typing import Callable, Optional

from .proxy import FoveanceProxy, _payload_text


def _normalize(entry) -> Optional[dict]:
    """One log entry -> {"messages": [...], ...} or None if unrecognized."""
    if isinstance(entry, dict) and isinstance(entry.get("messages"), list):
        return entry
    if isinstance(entry, list) and (not entry or isinstance(entry[0], dict)):
        return {"messages": entry}
    return None


def load_conversations(path: str) -> list[dict]:
    """Parse a log file into a list of {"messages": [...], "system": ..., "tools": ...} dicts."""
    with open(path, encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []
    try:  # whole-file JSON first
        data = json.loads(raw)
        if isinstance(data, list) and data and isinstance(data[0], dict) and "role" in data[0]:
            entries: list = [data]            # a single bare messages list
        elif isinstance(data, list):
            entries = data                    # a list of conversations
        else:
            entries = [data]                  # a single conversation object
    except json.JSONDecodeError:              # JSONL: one conversation per line
        entries = [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
    return [c for c in (_normalize(e) for e in entries) if c is not None]


def audit_conversations(convs: list[dict], budget: int = 2000, drift: float = 0.6,
                        token_counter: Optional[Callable[[str], int]] = None) -> dict:
    """Replay each conversation turn-by-turn through a fresh proxy; return the savings report."""

    def count(text: str) -> int:
        return token_counter(text) if token_counter else len(text) // 4

    total_before = total_after = requests = 0
    for ci, conv in enumerate(convs):
        msgs = conv.get("messages") or []
        px = FoveanceProxy(budget=budget, drift=drift, agentic_allocator=False)
        for k in range(1, len(msgs) + 1):
            if msgs[k - 1].get("role") not in (None, "user"):
                continue  # requests happen on user turns
            req: dict = {"messages": msgs[:k], "user": f"audit-{ci}"}
            if conv.get("system") is not None:
                req["system"] = conv["system"]
            if conv.get("tools"):
                req["tools"] = conv["tools"]
            fwd, _ = (px.prepare_anthropic(req) if "system" in req else px.prepare(req))
            total_before += count(_payload_text(req))
            total_after += count(_payload_text(fwd))
            requests += 1
    saved = max(total_before - total_after, 0)
    return {
        "conversations": len(convs),
        "requests": requests,
        "tokens_before": total_before,
        "tokens_after": total_after,
        "tokens_saved": saved,
        "saved_pct": round(100.0 * saved / total_before, 1) if total_before else 0.0,
        "avg_saved_per_request": round(saved / requests, 1) if requests else 0.0,
    }


def format_report(r: dict, price_per_mtok: float = 3.0,
                  monthly_requests: Optional[int] = None) -> str:
    usd = r["tokens_saved"] * price_per_mtok / 1e6
    lines = [
        "Foveance audit — offline replay of your own logs (nothing was sent anywhere)",
        f"  conversations : {r['conversations']}   simulated requests: {r['requests']}",
        f"  input tokens  : {r['tokens_before']:,} -> {r['tokens_after']:,}",
        f"  saved         : {r['tokens_saved']:,} tokens ({r['saved_pct']}%)"
        f"  ~ ${usd:,.2f} at ${price_per_mtok}/Mtok input",
    ]
    if monthly_requests:
        monthly = r["avg_saved_per_request"] * monthly_requests * price_per_mtok / 1e6
        lines.append(f"  extrapolated  : ~${monthly:,.2f}/month at {monthly_requests:,} "
                     "requests/month (your own traffic shape)")
    lines.append("  (token counts are chars/4 estimates unless --exact-tokens; "
                 "savings depend on history length)")
    return "\n".join(lines)
