"""R3: the learning loop. Opt-in, local-only trace logging of which past items each new query
actually needed, plus ``foveance train`` which fits the learned future-relevance model on those
traces. The proxy then loads the trained model, so the allocator gets measurably better on YOUR
workload the longer you run it. Nothing ever leaves the machine.

Trace rows (JSONL, ``~/.foveance/traces.jsonl``): one per query event::

    {"conv": ..., "turn": N, "query": "...", "items": [{"id", "kind", "text", "created_turn"}],
     "referenced": ["item ids whose salient tokens the query hit"]}
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional


def _home_dir() -> str:
    d = os.path.join(os.path.expanduser("~"), ".foveance")
    os.makedirs(d, exist_ok=True)
    return d


def default_traces_path() -> str:
    return os.path.join(_home_dir(), "traces.jsonl")


def default_model_path() -> str:
    return os.path.join(_home_dir(), "model.json")


def _tokens(text: str) -> set:
    return {w.lower() for w in re.findall(r"[A-Za-z0-9_\-\.]{4,}", text or "")}


class TraceLogger:
    """Append-only local logger of (query -> referenced items) events."""

    def __init__(self, path: Optional[str] = None, text_head: int = 400):
        self.path = path or default_traces_path()
        self.text_head = text_head

    def log_event(self, conv_id: str, turn: int, query: str, items) -> list:
        """items: iterable of objects with .item_id/.kind/.full_text/.created_turn (or dicts).
        Returns the referenced item ids (ground truth by salient-token overlap)."""
        q_toks = _tokens(query)
        rows, referenced = [], []
        for it in items:
            iid = getattr(it, "item_id", None) or (it.get("id") if isinstance(it, dict) else None)
            text = getattr(it, "full_text", None) or (it.get("text", "") if isinstance(it, dict)
                                                      else "")
            kind = getattr(it, "kind", None) or (it.get("kind", "item") if isinstance(it, dict)
                                                 else "item")
            created = getattr(it, "created_turn", 0) if not isinstance(it, dict) \
                else it.get("created_turn", 0)
            if iid is None:
                continue
            if q_toks & _tokens(text):
                referenced.append(iid)
            rows.append({"id": iid, "kind": kind, "text": text[:self.text_head],
                         "created_turn": created})
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"conv": conv_id, "turn": turn, "query": query[:400],
                                "items": rows, "referenced": referenced},
                               ensure_ascii=False) + "\n")
        return referenced


def build_training_traces(path: Optional[str] = None) -> list:
    """Reconstruct fit_traces-format traces from the JSONL log: one trace per conversation,
    with Item objects, the ordered query list, and per-turn referenced sets."""
    from .store import Item

    path = path or default_traces_path()
    if not os.path.exists(path):
        return []
    convs: dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            c = convs.setdefault(row["conv"], {"items": {}, "queries": [], "referenced": {}})
            t = len(c["queries"])
            c["queries"].append(row.get("query", ""))
            for it in row.get("items", []):
                if it["id"] not in c["items"]:
                    c["items"][it["id"]] = Item(item_id=it["id"], kind=it.get("kind", "item"),
                                                full_text=it.get("text", ""),
                                                created_turn=it.get("created_turn", 0))
            if row.get("referenced"):
                c["referenced"][t] = set(row["referenced"])
    return [{"items": list(c["items"].values()), "queries": c["queries"],
             "referenced": c["referenced"]} for c in convs.values() if c["queries"]]


def train(traces_path: Optional[str] = None, model_path: Optional[str] = None,
          horizon: int = 5) -> dict:
    """Fit the learned future-relevance model on the local traces; save and report."""
    from .learned import LogisticFutureRelevance

    traces = build_training_traces(traces_path)
    n_events = sum(len(t["queries"]) for t in traces)
    if not traces:
        return {"trained": False, "conversations": 0, "events": 0,
                "reason": "no traces logged yet (run the proxy with --learn first)"}
    model = LogisticFutureRelevance()
    model.fit_traces(traces, horizon=horizon)
    out = model_path or default_model_path()
    model.save(out)
    return {"trained": True, "conversations": len(traces), "events": n_events,
            "model_path": out, "weights": model.weights}


def load_model(model_path: Optional[str] = None):
    """The trained model if one exists, else None (heuristic posterior is used)."""
    from .learned import LogisticFutureRelevance

    path = model_path or default_model_path()
    if not os.path.exists(path):
        return None
    try:
        return LogisticFutureRelevance.load(path)
    except Exception:
        return None
