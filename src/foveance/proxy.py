"""
Drop-in OpenAI-compatible reverse proxy that applies Foveance transparently.

A client points its OpenAI base URL at this proxy; the proxy keeps a multi-fidelity store per
conversation, anticipatorily allocates fidelity across the prior messages under a token budget,
rewrites the request, forwards it upstream, and returns the upstream response unchanged. Zero
client code change (mirrors Headroom's drop-in UX).

The core (``FoveanceProxy``) is pure and unit-testable with any ``upstream`` callable -- no server
or network needed -- so CI can prove "transparently compresses an OpenAI-compatible request"
against a local echo upstream. ``build_app`` wires the same core into FastAPI (``[proxy]`` extra).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Optional

from .store import MultiFidelityStore, Item, default_renderer, Renderer
from .predictor import AnticipatoryPredictor, PredictorConfig
from .embedders import HashingEmbedder
from . import baselines


@dataclass
class _ConvState:
    store: MultiFidelityStore
    pred: AnticipatoryPredictor
    turn: int = 0
    seen: int = 0          # number of prior messages already ingested as items
    reinflations: int = 0


def _is_structured(messages: list[dict]) -> bool:
    """True if the conversation uses tool calling, where collapsing it would sever the
    tool_use<->tool_result pairing that providers validate (Anthropic 400s, OpenAI rejects). The
    proxy then passes the request through unchanged (correctness over savings). Plain text content,
    including the list-of-text-blocks form, is *not* structured and is still compressed. Compressing
    tool transcripts in place is future work; see docs/limitations.md."""
    for m in messages:
        if m.get("tool_calls") or m.get("tool_call_id") or m.get("role") == "tool":
            return True  # OpenAI tool-call plumbing
        c = m.get("content")
        if isinstance(c, list):
            for blk in c:
                if isinstance(blk, dict) and blk.get("type") in ("tool_use", "tool_result"):
                    return True  # Anthropic tool blocks
    return False


def _is_agentic(request: dict, messages: list[dict]) -> bool:
    """True if the request comes from a tool-using agent rather than a plain chat client. Agents
    (Claude Code, Codex, ...) declare a ``tools`` array (and/or ``tool_choice``) on every call and
    rely on strict tool_use<->tool_result pairing, so rewriting their history can make the provider
    reject the request. The proxy forwards these verbatim (correctness first) and reserves
    compression for plain chat, where the savings are largest and the transform is always valid."""
    return bool(request.get("tools") or request.get("tool_choice")) or _is_structured(messages)


def _extract_text(content) -> str:
    """Flatten OpenAI/Anthropic message content (string, or a list of text/blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for blk in content:
            if isinstance(blk, dict):
                parts.append(str(blk.get("text") or blk.get("content") or ""))
            else:
                parts.append(str(blk))
        return "\n".join(p for p in parts if p)
    return str(content or "")


def _digest_text(text: str, head: int = 12, tail: int = 6, max_chars: int = 700) -> str:
    """Shrink an oversized text payload (e.g. a long tool output) while keeping its head and tail and
    marking what was removed. Returns the input unchanged if it is already small or cannot shrink.
    This is lossy but reversible in spirit: the elision marker tells the model context was trimmed."""
    if not isinstance(text, str) or len(text) <= max_chars:
        return text
    lines = text.splitlines()
    if len(lines) >= head + tail + 4:
        elided = len(lines) - head - tail
        out = "\n".join(lines[:head] + [f"[... {elided} lines elided by Foveance ...]"] + lines[-tail:])
        return out if len(out) < len(text) else text
    keep = max_chars // 2
    h, t = text[:keep], text[-(max_chars // 4):]
    out = f"{h}\n[... {len(text) - len(h) - len(t)} chars elided by Foveance ...]\n{t}"
    return out if len(out) < len(text) else text


def _payload_text(request: dict) -> str:
    """Serialize the model-visible payload (messages / system / input / instructions) to a
    string, for both the chars/4 heuristic and any configured exact tokenizer."""
    import json as _json

    parts = [request.get(k) for k in ("messages", "system", "input", "instructions")
             if request.get(k)]
    try:
        return _json.dumps(parts, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(parts)


def _payload_chars(request: dict) -> int:
    """Rough size in characters of the model-visible payload. Used for the running
    tokens-saved estimate (chars/4 ~= tokens) when no exact ``token_counter`` is configured; the
    estimate is labelled as such everywhere it is shown and is never used in benchmark numbers."""
    return len(_payload_text(request))


def _digest_block(blk):
    """Digest the text payload of a content block in place, preserving its type and ids (so
    Anthropic tool_use<->tool_result pairing stays valid). Blocks carrying a ``cache_control``
    breakpoint are never modified (touching one invalidates the provider's prompt cache).
    Unknown/small blocks pass through."""
    if not isinstance(blk, dict):
        return blk
    if blk.get("cache_control"):
        return blk
    t = blk.get("type")
    if t == "text" and isinstance(blk.get("text"), str):
        nd = _digest_text(blk["text"])
        return blk if nd == blk["text"] else {**blk, "text": nd}
    if t == "tool_result":
        c = blk.get("content")
        if isinstance(c, str):
            nd = _digest_text(c)
            return blk if nd == c else {**blk, "content": nd}
        if isinstance(c, list):
            nc = [_digest_block(b) for b in c]
            return blk if nc == c else {**blk, "content": nc}
    return blk


def _digest_responses_item(item):
    """Digest the large text payload of an OpenAI Responses-API input item in place, preserving its
    type and ``call_id`` so function_call<->function_call_output pairing stays valid. Targets big
    tool outputs (``function_call_output.output``) and oversized message text."""
    if not isinstance(item, dict):
        return item
    t = item.get("type")
    if t == "function_call_output" and isinstance(item.get("output"), str):
        nd = _digest_text(item["output"])
        return item if nd == item["output"] else {**item, "output": nd}
    if t == "message" or "content" in item:
        c = item.get("content")
        if isinstance(c, str):
            nd = _digest_text(c)
            return item if nd == c else {**item, "content": nd}
        if isinstance(c, list):
            changed = False
            nc = []
            for part in c:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    nt = _digest_text(part["text"])
                    if nt != part["text"]:
                        changed = True
                        nc.append({**part, "text": nt})
                        continue
                nc.append(part)
            return {**item, "content": nc} if changed else item
    return item


@dataclass
class FoveanceProxy:
    """Per-conversation anticipatory compression of OpenAI/Anthropic chat requests."""

    budget: int = 2000
    drift: float = 0.6
    policy: str = "foveance"
    renderer: Renderer = default_renderer
    system_prefix: str = "Context (compressed by Foveance):"
    token_counter: Optional[Callable[[str], int]] = None
    convs: dict[str, _ConvState] = field(default_factory=dict)
    requests: int = 0
    # Agentic (tool-using) requests are compressed in place: recent turns are protected and only
    # large old content blocks (tool outputs) are digested, so tool_use<->tool_result pairing stays
    # intact and the provider still accepts the request.
    agentic_protect_last: int = 3
    agentic_min_chars: int = 1000
    # Cache-aware mode: never modify content at or before the last explicit Anthropic
    # ``cache_control`` breakpoint, so the provider's prompt cache is never invalidated by the
    # proxy. Off by default: with cached input billed at a discount, busting the cache to cut raw
    # tokens is usually still cheaper, but flip this on when the cache discount dominates (see
    # docs/limitations.md for the arithmetic).
    cache_aware: bool = False
    # Assumed input price for the running $-saved estimate shown by /admin and `foveance wrap`
    # (USD per million input tokens; configure per your provider/model).
    price_per_mtok: float = 3.0
    compressed_requests: int = 0
    est_chars_before: int = 0
    est_chars_after: int = 0
    est_tokens_before: int = 0
    est_tokens_after: int = 0

    def _state(self, conv_id: str) -> _ConvState:
        if conv_id not in self.convs:
            store = MultiFidelityStore(self.renderer, self.token_counter)
            cfg = PredictorConfig(drift=(0.0 if self.policy in ("reactive", "reactive_afm")
                                         else self.drift))
            pred = AnticipatoryPredictor(store, HashingEmbedder(), config=cfg)
            self.convs[conv_id] = _ConvState(store=store, pred=pred)
        return self.convs[conv_id]

    def _ingest_and_allocate(self, history: list[dict], last_text: str, conv_id: str):
        """Shared core: ingest newly-seen history, score the next need, allocate, assemble."""
        st = self._state(conv_id)
        for m in history[st.seen:]:
            iid = f"m{len(st.store.order)}"
            st.store.add(Item(item_id=iid, kind=m.get("role", "user"),
                              full_text=_extract_text(m.get("content", "")), created_turn=st.turn))
        st.seen = len(history)
        st.pred.observe_query(last_text)
        fn = baselines.POLICIES.get(self.policy, baselines.foveance)
        levels = fn(st.store, st.pred, self.budget, st.turn)
        ctx, ntok = st.store.assemble(levels, system=self.system_prefix)
        st.turn += 1
        return ctx, ntok, st

    def transform(self, messages: list[dict], conv_id: str = "default") -> tuple[list[dict], dict]:
        """Compress the prior messages of an OpenAI chat request; keep system + last turn verbatim."""
        system_msgs = [m for m in messages if m.get("role") == "system"]
        convo = [m for m in messages if m.get("role") != "system"]
        if not convo:
            return messages, {"compressed": False, "items": 0}
        if _is_structured(convo):
            return messages, {"compressed": False, "items": 0, "reason": "structured-passthrough"}
        last = convo[-1]
        ctx, ntok, st = self._ingest_and_allocate(convo[:-1], _extract_text(last.get("content", "")),
                                                   conv_id)
        new_messages = list(system_msgs)
        if st.store.order:
            new_messages.append({"role": "system", "content": ctx})
        new_messages.append(last)
        stats = {"compressed": True, "conv_id": conv_id, "items": len(st.store.order),
                 "context_tokens": ntok, "budget": self.budget, "turn": st.turn}
        return new_messages, stats

    def transform_anthropic(self, system, messages: list[dict],
                            conv_id: str = "default") -> tuple[str, list[dict], dict]:
        """Compress an Anthropic Messages request. The compressed context is folded into the
        ``system`` string (Anthropic has no system-role messages), and only the final turn is
        kept in ``messages``."""
        if not messages:
            return system, messages, {"compressed": False, "items": 0}
        if _is_structured(messages):
            # Tool-use / structured conversation (e.g. Claude Code): pass through untouched so the
            # tool_use<->tool_result pairing and cache_control stay valid for the upstream.
            return system, messages, {"compressed": False, "items": 0,
                                      "reason": "structured-passthrough"}
        last = messages[-1]
        ctx, ntok, st = self._ingest_and_allocate(messages[:-1], _extract_text(last.get("content", "")),
                                                   conv_id)
        if not st.store.order:  # nothing to compress yet -- keep the request (and its system) intact
            return system, messages, {"compressed": False, "items": 0}
        base_system = _extract_text(system)
        new_system = (base_system + "\n\n" + ctx).strip()
        stats = {"compressed": True, "conv_id": conv_id, "items": len(st.store.order),
                 "context_tokens": ntok, "budget": self.budget, "turn": st.turn}
        return new_system, [last], stats

    def _compress_agentic_anthropic(self, messages: list[dict]) -> list[dict]:
        """Structure-preserving compression for Anthropic tool-use requests: keep every message,
        role, and tool_use/tool_result id intact; protect the last ``agentic_protect_last`` turns;
        digest only large content blocks in older turns. The result is always a valid Messages
        request, so agents like Claude Code keep working while large stale tool output is trimmed.

        With ``cache_aware=True``, messages at or before the last explicit ``cache_control``
        breakpoint are additionally left byte-identical, so the provider's prompt-cache prefix is
        never invalidated (individual blocks carrying ``cache_control`` are always preserved
        regardless; see :func:`_digest_block`)."""
        cut = max(0, len(messages) - self.agentic_protect_last)
        start = 0
        if self.cache_aware:
            for i, m in enumerate(messages):
                c = m.get("content")
                if isinstance(c, list) and any(isinstance(b, dict) and b.get("cache_control")
                                               for b in c):
                    start = i + 1
        out = []
        for i, m in enumerate(messages):
            c = m.get("content")
            if i >= cut or i < start:
                out.append(m)
            elif isinstance(c, str):
                nd = _digest_text(c)
                out.append(m if nd == c else {**m, "content": nd})
            elif isinstance(c, list):
                nc = [_digest_block(b) for b in c]
                out.append(m if nc == c else {**m, "content": nc})
            else:
                out.append(m)
        return out

    def _compress_agentic_openai(self, messages: list[dict]) -> list[dict]:
        """Structure-preserving compression for OpenAI tool-use requests: protect recent turns and
        system, keep tool_calls/tool_call_id pairing intact, and digest large old tool outputs
        (role=tool) and oversized text content/blocks in place."""
        cut = max(0, len(messages) - self.agentic_protect_last)
        out = []
        for i, m in enumerate(messages):
            c = m.get("content")
            if i >= cut or m.get("role") == "system":
                out.append(m)
            elif isinstance(c, str) and (m.get("role") == "tool" or len(c) > self.agentic_min_chars):
                nd = _digest_text(c)
                out.append(m if nd == c else {**m, "content": nd})
            elif isinstance(c, list):
                nc = [_digest_block(b) for b in c]
                out.append(m if nc == c else {**m, "content": nc})
            else:
                out.append(m)
        return out

    def _compress_agentic_responses(self, items: list[dict]) -> list[dict]:
        """In-place compression for the OpenAI Responses API ``input`` list: protect the most recent
        items and digest large old tool outputs / message text, keeping every item, type, role, and
        ``call_id`` intact so the request stays valid (used by Codex and the OpenAI Agents SDK)."""
        cut = max(0, len(items) - self.agentic_protect_last)
        return [it if i >= cut else _digest_responses_item(it) for i, it in enumerate(items)]

    def prepare_responses(self, request: dict) -> tuple[dict, dict]:
        """Compress an OpenAI Responses request (``input`` is a string or a list of items) and return
        the forward-ready body plus stats. A string ``input`` is passed through unchanged."""
        self.requests += 1
        inp = request.get("input")
        if not isinstance(inp, list):
            fwd = dict(request)
            return fwd, self._account(request, fwd,
                                      {"compressed": False, "items": 0,
                                       "reason": "responses-passthrough"})
        new = self._compress_agentic_responses(inp)
        fwd = dict(request)
        fwd["input"] = new
        return fwd, self._account(request, fwd, {"compressed": new != inp, "items": len(inp),
                                                 "reason": "responses-inplace"})

    @staticmethod
    def _conv_id(request: dict, messages: list[dict]) -> str:
        """Pick a stable per-conversation key. Prefer an explicit id the client supplies
        (``user``/``conversation_id``/Anthropic ``metadata.user_id``); otherwise derive one from
        the first non-system message so distinct conversations get distinct stores even when the
        client (e.g. a CLI agent) sends no id at all."""
        meta_raw = request.get("metadata")
        meta: dict = meta_raw if isinstance(meta_raw, dict) else {}
        explicit = request.get("user") or request.get("conversation_id") or meta.get("user_id")
        if explicit:
            return str(explicit)
        for m in messages:
            if m.get("role") != "system":
                seed = _extract_text(m.get("content", "")).encode("utf-8", "ignore")
                return "auto-" + hashlib.sha1(seed).hexdigest()[:12]
        return "default"

    def prepare(self, request: dict) -> tuple[dict, dict]:
        """Compress an OpenAI chat request and return the forward-ready body plus stats.
        Separated from :meth:`handle` so a streaming server can forward ``fwd`` itself."""
        self.requests += 1
        msgs = list(request.get("messages", []))
        if _is_agentic(request, msgs):
            new_messages = self._compress_agentic_openai(msgs)
            fwd = dict(request)
            fwd["messages"] = new_messages
            fwd.pop("conversation_id", None)
            return fwd, self._account(request, fwd,
                                      {"compressed": new_messages != msgs, "items": len(msgs),
                                       "reason": "agentic-inplace"})
        conv_id = self._conv_id(request, msgs)
        new_messages, stats = self.transform(msgs, conv_id)
        fwd = dict(request)
        fwd["messages"] = new_messages
        fwd.pop("conversation_id", None)
        return fwd, self._account(request, fwd, stats)

    def prepare_anthropic(self, request: dict) -> tuple[dict, dict]:
        """Compress an Anthropic Messages request and return the forward-ready body plus stats."""
        self.requests += 1
        msgs = list(request.get("messages", []))
        if _is_agentic(request, msgs):
            new_messages = self._compress_agentic_anthropic(msgs)
            fwd = dict(request)
            fwd["messages"] = new_messages  # system left intact (preserves its cache_control)
            return fwd, self._account(request, fwd,
                                      {"compressed": new_messages != msgs, "items": len(msgs),
                                       "reason": "agentic-inplace"})
        conv_id = self._conv_id(request, msgs)
        new_system, new_messages, stats = self.transform_anthropic(
            request.get("system", ""), msgs, conv_id)
        fwd = dict(request)
        if new_system:
            fwd["system"] = new_system
        fwd["messages"] = new_messages
        return fwd, self._account(request, fwd, stats)

    def handle(self, request: dict, upstream: Callable[[dict], dict]) -> dict:
        """Rewrite ``request['messages']`` (OpenAI) then forward to ``upstream`` and return it."""
        fwd, stats = self.prepare(request)
        resp = upstream(fwd)
        if isinstance(resp, dict):
            resp["foveance"] = stats
        return resp

    def handle_anthropic(self, request: dict, upstream: Callable[[dict], dict]) -> dict:
        """Rewrite an Anthropic Messages request (``system`` + ``messages``) and forward it."""
        fwd, stats = self.prepare_anthropic(request)
        resp = upstream(fwd)
        if isinstance(resp, dict):
            resp["foveance"] = stats
        return resp

    def _account(self, request: dict, fwd: dict, stats: dict) -> dict:
        """Record the payload size before/after compression on the running totals and annotate
        ``stats`` with per-request counts. Uses ``self.token_counter`` (an exact tokenizer, e.g.
        tiktoken) when configured; otherwise falls back to the chars/4 heuristic (an estimate,
        not billing -- see ``stats()``)."""
        before_text, after_text = _payload_text(request), _payload_text(fwd)
        self.est_chars_before += len(before_text)
        self.est_chars_after += len(after_text)
        if stats.get("compressed"):
            self.compressed_requests += 1
        if self.token_counter is not None:
            tb, ta = self.token_counter(before_text), self.token_counter(after_text)
            self.est_tokens_before += tb
            self.est_tokens_after += ta
        else:
            tb, ta = len(before_text) // 4, len(after_text) // 4
        stats["est_tokens_before"] = tb
        stats["est_tokens_after"] = ta
        stats["est_tokens_exact"] = self.token_counter is not None
        return stats

    def stats(self) -> dict:
        exact = self.token_counter is not None
        tb, ta = ((self.est_tokens_before, self.est_tokens_after) if exact
                  else (self.est_chars_before // 4, self.est_chars_after // 4))
        saved = max(tb - ta, 0)
        return {
            "requests": self.requests,
            "compressed_requests": self.compressed_requests,
            "conversations": len(self.convs),
            "est_tokens_before": tb,
            "est_tokens_after": ta,
            "est_tokens_saved": saved,
            "est_tokens_exact": exact,
            "est_saved_pct": round(100.0 * saved / tb, 1) if tb else 0.0,
            "price_per_mtok": self.price_per_mtok,
            "est_usd_saved": round(saved * self.price_per_mtok / 1e6, 4),
            "per_conv": {cid: {"items": len(s.store.order), "turns": s.turn}
                         for cid, s in self.convs.items()},
        }


def build_app(proxy: Optional[FoveanceProxy] = None,
              upstream_url: str = "http://localhost:11434/v1"):
    """Build a FastAPI app that is a transparent, streaming drop-in for both the OpenAI and the
    Anthropic wire protocols, so *any* client or agent that speaks either one works unchanged:

    * ``POST /v1/chat/completions`` -- OpenAI Chat Completions (OpenAI SDK, Ollama, Codex, LangChain, ...)
    * ``POST /v1/messages``         -- Anthropic Messages (Anthropic SDK, Claude Code, ...)
    * ``GET  /v1/models``           -- passthrough so clients that probe the model list succeed
    * ``GET  /health`` and ``/``    -- liveness for orchestrators
    * ``GET  /admin/stats``         -- per-conversation compression stats

    Both chat routes honour ``"stream": true`` and stream the upstream bytes back verbatim (the
    compression is applied to the *request*, so the response is a pure passthrough). *Every* client
    header is forwarded upstream untouched except hop-by-hop ones, so auth of any kind
    (``x-api-key``, ``Authorization`` bearer/OAuth), ``anthropic-beta`` feature flags, and
    tool-specific headers all pass through and no secret is stored by the proxy."""
    from fastapi import FastAPI, Request  # type: ignore
    from fastapi.responses import StreamingResponse, Response  # type: ignore
    import json
    import urllib.error
    import urllib.request

    # ``from __future__ import annotations`` stringizes the route annotations below; FastAPI
    # resolves them with get_type_hints against THIS module's globals, but ``Request`` is imported
    # locally here, so expose it at module scope so ``request: Request`` is recognized (not 422'd).
    globals()["Request"] = Request

    px = proxy or FoveanceProxy()
    app = FastAPI(title="Foveance proxy", version="0.1.0")
    base = upstream_url.rstrip("/")

    # Hop-by-hop / length headers must be recomputed by urllib, not forwarded verbatim.
    _SKIP = {"host", "content-length", "connection", "accept-encoding",
             "transfer-encoding", "content-encoding"}

    def _client_headers(request: "Request") -> dict:  # pragma: no cover - needs server
        h = {k: v for k, v in request.headers.items() if k.lower() not in _SKIP}
        h.setdefault("Content-Type", "application/json")
        return h

    def _open(req: dict, path: str, headers: dict):  # pragma: no cover - needs upstream
        body = json.dumps(req).encode()
        r = urllib.request.Request(base + path, data=body, headers=headers)
        return urllib.request.urlopen(r, timeout=600)

    def _respond(fwd: dict, stats: dict, path: str, headers: dict):  # pragma: no cover
        try:
            resp = _open(fwd, path, headers)
        except urllib.error.HTTPError as e:
            # Surface the upstream's status and error body verbatim instead of failing internally,
            # so clients see the real error (and don't retry-storm) and operators can diagnose.
            return Response(content=e.read(), status_code=e.code,
                            media_type=e.headers.get("Content-Type", "application/json"))
        if fwd.get("stream"):
            def _gen():
                try:
                    while True:
                        chunk = resp.read(2048)
                        if not chunk:
                            break
                        yield chunk
                finally:
                    resp.close()
            return StreamingResponse(_gen(),
                                     media_type=resp.headers.get("Content-Type", "text/event-stream"))
        data = json.loads(resp.read())
        resp.close()
        if isinstance(data, dict):
            data["foveance"] = stats
        return data

    @app.post("/v1/chat/completions")
    async def chat(request: Request):  # pragma: no cover - needs server
        fwd, stats = px.prepare(await request.json())
        return _respond(fwd, stats, "/chat/completions", _client_headers(request))

    @app.post("/v1/messages")
    async def messages(request: Request):  # pragma: no cover - needs server
        fwd, stats = px.prepare_anthropic(await request.json())
        return _respond(fwd, stats, "/messages", _client_headers(request))

    @app.post("/responses")
    @app.post("/v1/responses")
    async def responses(request: Request):  # pragma: no cover - needs server
        fwd, stats = px.prepare_responses(await request.json())
        return _respond(fwd, stats, "/responses", _client_headers(request))

    @app.get("/v1/models")
    async def models(request: Request):  # pragma: no cover - needs upstream
        try:
            r = urllib.request.Request(base + "/models", headers=_client_headers(request))
            return Response(content=urllib.request.urlopen(r, timeout=60).read(),
                            media_type="application/json")
        except urllib.error.HTTPError as e:
            return Response(content=e.read(), status_code=e.code,
                            media_type=e.headers.get("Content-Type", "application/json"))

    @app.get("/health")
    async def health():  # pragma: no cover - trivial
        return {"status": "ok", "service": "foveance-proxy", "upstream": base,
                "budget": px.budget, "policy": px.policy}

    @app.get("/admin/stats")
    async def admin_stats():  # pragma: no cover - needs server
        return px.stats()

    @app.get("/")
    @app.get("/admin")
    async def dashboard():  # pragma: no cover - needs server
        return Response(content=_DASHBOARD_HTML, media_type="text/html")

    return app


# Self-contained live dashboard served at / and /admin: polls /admin/stats and shows the running
# tokens-saved estimate (chars/4, or exact via --exact-tokens) and its $-equivalent at the
# configured input price. No external assets, no build step, no tracking -- one HTML string.
_DASHBOARD_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Foveance proxy</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --bg:#0b0e1a; --card:#141830; --ink:#e6e8f2; --dim:#8b90ad;
          --amber:#F59E0B; --indigo:#6366f1; }
  * { box-sizing:border-box; margin:0 }
  body { background:var(--bg); color:var(--ink); min-height:100vh; display:flex;
         align-items:center; justify-content:center;
         font:16px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace }
  main { width:min(680px,94vw); padding:32px 0 }
  h1 { font-size:20px; font-weight:600; letter-spacing:.04em; margin-bottom:4px }
  h1 b { color:var(--amber) }
  .sub { color:var(--dim); font-size:13px; margin-bottom:24px }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px }
  .card { background:var(--card); border:1px solid #232849; border-radius:12px; padding:16px }
  .card .v { font-size:26px; font-weight:700; margin-top:2px }
  .card .k { color:var(--dim); font-size:12px; text-transform:uppercase; letter-spacing:.08em }
  .hero { grid-column:1/-1; text-align:center; padding:28px 16px;
          border-color:var(--amber) }
  .hero .v { font-size:44px; color:var(--amber) }
  .usd { color:var(--indigo); font-size:15px; margin-top:6px }
  .foot { color:var(--dim); font-size:12px; margin-top:20px }
</style></head><body><main>
<h1>fove<b>a</b>nce proxy</h1>
<div class="sub">anticipatory context allocation &middot; live stats (refreshes every 2s)</div>
<div class="grid">
  <div class="card hero"><div class="k">estimated tokens saved</div>
    <div class="v" id="saved">&ndash;</div><div class="usd" id="usd"></div></div>
  <div class="card"><div class="k">requests</div><div class="v" id="req">&ndash;</div></div>
  <div class="card"><div class="k">compressed</div><div class="v" id="cmp">&ndash;</div></div>
  <div class="card"><div class="k">tokens in &rarr; out</div><div class="v" id="io">&ndash;</div></div>
  <div class="card"><div class="k">saved</div><div class="v" id="pct">&ndash;</div></div>
</div>
<div class="foot"><span id="footNote">Estimates use chars/4 &asymp; tokens on the request payload</span>;
the $ figure uses the configured <code>--price-per-mtok</code>. JSON at
<a href="/admin/stats" style="color:var(--indigo)">/admin/stats</a>.</div>
</main><script>
const f = n => n >= 1e6 ? (n/1e6).toFixed(2)+"M" : n >= 1e3 ? (n/1e3).toFixed(1)+"k" : String(n);
async function tick(){
  try {
    const s = await (await fetch("/admin/stats")).json();
    document.getElementById("saved").textContent = f(s.est_tokens_saved);
    document.getElementById("usd").textContent =
      "\\u2248 $" + s.est_usd_saved.toFixed(4) + " at $" + s.price_per_mtok + "/Mtok input";
    document.getElementById("req").textContent = s.requests;
    document.getElementById("cmp").textContent = s.compressed_requests;
    document.getElementById("io").textContent = f(s.est_tokens_before)+" \\u2192 "+f(s.est_tokens_after);
    document.getElementById("pct").textContent = s.est_saved_pct + "%";
    document.getElementById("footNote").textContent = s.est_tokens_exact
      ? "Token counts use a real tokenizer (tiktoken) on the request payload"
      : "Estimates use chars/4 \\u2248 tokens on the request payload";
  } catch (e) {}
}
tick(); setInterval(tick, 2000);
</script></body></html>
"""
