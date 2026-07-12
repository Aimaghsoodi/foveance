"""A tiny in-process OpenRouter-shaped chat server for offline tests of paper2_bench.

It behaves like a model that can only answer when the evidence is present -- and that USES the
foveance_expand tool to recover it when the tool is offered and a compressed-item marker is
visible. That makes the four arms produce the paper's story deterministically, with no key and no
network egress:
  raw               -> secret visible -> answered correctly
  digest/allocator  -> secret elided (disjoint query) -> UNKNOWN
  allocator+expand  -> model calls foveance_expand, gets the item back, then answers correctly

It also returns a usage.cost so the CostAccountant and --budget-usd guard are exercised for real.
"""
from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

_SECRET_RE = re.compile(r"pg://[^\s\"']+")
_ID_RE = re.compile(r"Foveance (?:compressed )?item ([0-9a-f]{12})")


def _all_text(messages) -> str:
    return "\n".join(str(m.get("content", "")) for m in messages)


class _Handler(BaseHTTPRequestHandler):
    cost_per_call = 0.001   # $0.001/call so a budget cap is reachable in a small test

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n) or b"{}")
        messages = payload.get("messages", [])
        has_tools = bool(payload.get("tools"))
        text = _all_text(messages)
        secret = _SECRET_RE.search(text)
        already_expanded = any(m.get("role") == "tool" for m in messages)

        if secret:
            msg = {"role": "assistant", "content": secret.group(0)}
        elif has_tools and not already_expanded and _ID_RE.search(text):
            iid = _ID_RE.search(text).group(1)
            msg = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "foveance_expand",
                              "arguments": json.dumps({"item_id": iid})}}]}
        else:
            msg = {"role": "assistant", "content": "UNKNOWN"}

        body = {"choices": [{"message": msg}],
                "usage": {"prompt_tokens": max(1, len(text) // 4), "completion_tokens": 8,
                          "cost": self.cost_per_call}}
        out = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):  # silence
        pass


class MockOpenRouter:
    """Context manager: `with MockOpenRouter() as base_url: ...`."""
    def __init__(self, cost_per_call: float = 0.001):
        _Handler.cost_per_call = cost_per_call
        self.server = HTTPServer(("127.0.0.1", 0), _Handler)
        self.port = self.server.server_address[1]

    def __enter__(self) -> str:
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self.port}/v1"

    def __exit__(self, *a):
        self.server.shutdown()
