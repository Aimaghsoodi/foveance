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
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .store import MultiFidelityStore, Item, Fidelity, default_renderer, Renderer
from .predictor import AnticipatoryPredictor, FutureRelevancePredictor, PredictorConfig
from .embedders import HashingEmbedder
from . import baselines


@dataclass
class _ConvState:
    store: MultiFidelityStore
    pred: AnticipatoryPredictor
    turn: int = 0
    seen: int = 0          # number of prior messages already ingested as items
    reinflations: int = 0
    last_used: float = field(default_factory=time.time)


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


def _terms(text: str) -> set:
    """Salient lowercase tokens of a query (>=3 chars), for salience-aware digestion."""
    import re

    return {w.lower() for w in re.findall(r"[A-Za-z0-9_\-\.]{3,}", text or "")[:64]}


def _digest_text(text: str, head: int = 12, tail: int = 6, max_chars: int = 700,
                 query_terms: Optional[set] = None) -> str:
    """Shrink an oversized text payload (e.g. a long tool output), keeping its head and tail and
    marking what was removed. When ``query_terms`` is given, lines overlapping the current query
    are additionally kept in place (salience-aware digestion), so the one line that matters is
    not blindly elided. Returns the input unchanged if already small or if shrinking would not
    help. Lossy but reversible in spirit: elision markers tell the model context was trimmed."""
    if not isinstance(text, str) or len(text) <= max_chars:
        return text
    lines = text.splitlines()
    if len(lines) >= head + tail + 4:
        if query_terms:
            scored = []
            for i in range(head, len(lines) - tail):
                low = lines[i].lower()
                overlap = sum(1 for t in query_terms if t in low)
                if overlap:
                    scored.append((overlap, -i))
            keep = {-neg_i for _, neg_i in sorted(scored, reverse=True)[:8]}
            out_lines: list[str] = []
            elided = 0
            for i, ln in enumerate(lines):
                if i < head or i >= len(lines) - tail or i in keep:
                    if elided:
                        out_lines.append(f"[... {elided} lines elided by Foveance ...]")
                        elided = 0
                    out_lines.append(ln)
                else:
                    elided += 1
            if elided:
                out_lines.append(f"[... {elided} lines elided by Foveance ...]")
            out = "\n".join(out_lines)
            if len(out) < len(text):
                return out
            # salience kept too much -- fall through to the plain head/tail digest
        elided = len(lines) - head - tail
        out = "\n".join(lines[:head] + [f"[... {elided} lines elided by Foveance ...]"] + lines[-tail:])
        return out if len(out) < len(text) else text
    keep_c = max_chars // 2
    h, t = text[:keep_c], text[-(max_chars // 4):]
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


def _digest_block(blk, query_terms: Optional[set] = None):
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
        nd = _digest_text(blk["text"], query_terms=query_terms)
        return blk if nd == blk["text"] else {**blk, "text": nd}
    if t == "tool_result":
        c = blk.get("content")
        if isinstance(c, str):
            nd = _digest_text(c, query_terms=query_terms)
            return blk if nd == c else {**blk, "content": nd}
        if isinstance(c, list):
            nc = [_digest_block(b, query_terms) for b in c]
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


_EXPAND_DESC = ("Retrieve the full original content of a context item that Foveance compressed. "
                "Use this when a marker like [Foveance compressed item <id> ...] hides "
                "information you need to answer correctly.")
_EXPAND_SCHEMA = {"type": "object",
                  "properties": {"item_id": {"type": "string",
                                             "description": "the item id from the marker"}},
                  "required": ["item_id"]}
_EXPAND_TOOL_ANTHROPIC = {"name": "foveance_expand", "description": _EXPAND_DESC,
                          "input_schema": _EXPAND_SCHEMA}
_EXPAND_TOOL_OPENAI = {"type": "function",
                       "function": {"name": "foveance_expand", "description": _EXPAND_DESC,
                                    "parameters": _EXPAND_SCHEMA}}


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
    # Pro feature: when set (a foveance.license.SavingsLog), per-request savings are persisted so
    # totals survive restarts; the dashboard then shows all-time and per-day history.
    savings_log: Optional[object] = None
    # Durable full-text spill (foveance.vault.ItemVault): re-inflation survives restarts and the
    # foveance_expand tool can retrieve any compressed item. Set by the CLI; None disables.
    vault: Optional[object] = None
    # Conversation-state eviction: without this a long-running proxy grows without bound. LRU
    # beyond max_convs, plus a TTL for idle conversations. Evicted state can still be re-inflated
    # via the vault; in-memory eviction only forgets the working set.
    max_convs: int = 256
    conv_ttl_s: float = 6 * 3600.0
    evictions: int = 0
    # R1: anticipatory agentic compression -- old tool-transcript payloads get graded fidelities
    # from the same allocator plain chat uses (instead of blind digestion). Opt-in while it
    # matures; digestion remains the default behaviour.
    agentic_allocator: bool = False
    # R1: expose a foveance_expand tool so the MODEL can re-inflate any compressed item; the
    # server resolves those calls transparently (non-streaming requests only). Requires vault.
    expand_tool: bool = False
    expansions: int = 0
    # R3: the learning loop. trace_log (foveance.traces.TraceLogger) records which items each
    # query referenced; future_model (a trained LogisticFutureRelevance) replaces the heuristic
    # posterior when present, so allocation improves on the user's own workload over time.
    trace_log: Optional[object] = None
    future_model: Optional[FutureRelevancePredictor] = None
    # 0.5: lossless cross-item redundancy codec on the assembled plain-chat context. After
    # allocation renders each item, repeated line-runs across items are reference-encoded
    # losslessly (the first occurrence stays verbatim, so no fact is lost). Off by default;
    # `--codec` / FOVEANCE_CODEC=1 enables it. Reduces tokens with zero accuracy risk.
    apply_codec: bool = False
    codec_saved_tokens: int = 0
    # 0.5: the same lossless codec, but on the AGENTIC in-place paths. Instead of (lossily) digesting
    # each large old tool payload, run the codec *across* the eligible free-text payloads so
    # cross-message repeats (re-listed dirs, retried stack traces, boilerplate) collapse to legible
    # pointers with the first occurrence kept verbatim -- no fact lost. It rewrites only the free-text
    # payload strings; message count/order, roles, tool_use<->tool_result ids, cache_control blocks,
    # and the last ``agentic_protect_last`` turns are all left byte-identical, so the provider still
    # validates the request and the prompt cache is never invalidated. Off by default; opt-in.
    agentic_codec: bool = False

    def _codec_stream(self, payloads: list):
        """Run the lossless cross-item codec over ``payloads`` (ordered eligible free-text strings)
        and return an iterator of their coded forms in the same order. Each coded string is <= its
        input in tokens (the codec never inflates) and the set is exactly reversible (first
        occurrence verbatim). Updates ``codec_saved_tokens``. Structure/ids are never seen here, so
        tool pairing cannot break."""
        from .codec import RedundancyCodec
        codec = RedundancyCodec(min_run=1, token_counter=self.token_counter)
        rendered, rep = codec.render([(str(k), p) for k, p in enumerate(payloads)])
        self.codec_saved_tokens += max(0, rep.tokens_in - rep.tokens_out)
        return iter([t for _, t in rendered])

    def _assemble(self, store, levels):
        """Assemble the rendered context, optionally running the lossless codec across items."""
        if not self.apply_codec:
            return store.assemble(levels, system=self.system_prefix)
        from .codec import RedundancyCodec
        items = [(iid, store.render(iid, levels.get(iid, Fidelity.POINTER)))
                 for iid in store.order]
        rendered, rep = RedundancyCodec(min_run=1, token_counter=store._count).render(items)
        parts = ([self.system_prefix] if self.system_prefix else []) + [t for _, t in rendered]
        ctx = "\n".join(p for p in parts if p)
        self.codec_saved_tokens += max(0, rep.tokens_in - rep.tokens_out)
        return ctx, store._count(ctx)

    def _evict(self) -> None:
        now = time.time()
        expired = [cid for cid, st in self.convs.items()
                   if now - st.last_used > self.conv_ttl_s]
        for cid in expired:
            del self.convs[cid]
        self.evictions += len(expired)
        # called before inserting a new conversation: make room so the cap holds post-insert
        while len(self.convs) >= self.max_convs:
            oldest = min(self.convs, key=lambda cid: self.convs[cid].last_used)
            del self.convs[oldest]
            self.evictions += 1

    def _state(self, conv_id: str) -> _ConvState:
        if conv_id not in self.convs:
            self._evict()
            store = MultiFidelityStore(self.renderer, self.token_counter)
            cfg = PredictorConfig(drift=(0.0 if self.policy in ("reactive", "reactive_afm")
                                         else self.drift))
            pred = AnticipatoryPredictor(store, HashingEmbedder(), config=cfg,
                                         future_model=self.future_model)
            self.convs[conv_id] = _ConvState(store=store, pred=pred)
        st = self.convs[conv_id]
        st.last_used = time.time()
        return st

    def _ingest_and_allocate(self, history: list[dict], last_text: str, conv_id: str):
        """Shared core: ingest newly-seen history, score the next need, allocate, assemble."""
        st = self._state(conv_id)
        for m in history[st.seen:]:
            iid = f"m{len(st.store.order)}"
            st.store.add(Item(item_id=iid, kind=m.get("role", "user"),
                              full_text=_extract_text(m.get("content", "")), created_turn=st.turn))
        st.seen = len(history)
        st.pred.observe_query(last_text)
        if self.trace_log is not None:
            try:
                self.trace_log.log_event(  # type: ignore[attr-defined]
                    conv_id, st.turn, last_text, st.store.items.values())
            except Exception:
                pass
        fn = baselines.POLICIES.get(self.policy, baselines.foveance)
        levels = fn(st.store, st.pred, self.budget, st.turn)
        ctx, ntok = self._assemble(st.store, levels)
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

    # ------------------------------------------------------------------ agentic compression
    def _agentic_levels(self, conv_id: str, candidates: list, last_text: str) -> dict:
        """R1 core: score every compressible old item by *predicted future relevance* and allocate
        graded fidelities under the budget (the same anticipatory machinery plain chat uses).
        ``candidates`` is a list of (item_id, kind, full_text). Full texts are spilled to the
        vault so the foveance_expand tool (and future sessions) can re-inflate any of them."""
        from .store import Item

        st = self._state(conv_id)
        for iid, kind, text in candidates:
            if iid not in st.store.order:
                st.store.add(Item(item_id=iid, kind=kind, full_text=text, created_turn=st.turn))
            if self.vault is not None:
                try:
                    self.vault.put(conv_id, iid, kind, text)  # type: ignore[attr-defined]
                except Exception:
                    pass
        st.pred.observe_query(last_text)
        if self.trace_log is not None:
            try:
                self.trace_log.log_event(  # type: ignore[attr-defined]
                    conv_id, st.turn, last_text, st.store.items.values())
            except Exception:
                pass  # learning must never break request handling
        fn = baselines.POLICIES.get(self.policy, baselines.foveance)
        levels = fn(st.store, st.pred, self.budget, st.turn)
        st.turn += 1
        return levels

    def _render_level(self, text: str, kind: str, iid: str, level, qt: Optional[set]) -> str:
        """Render an item at its allocated fidelity, with a marker that names the item so the
        model can ask for it back (via foveance_expand when enabled)."""
        from .store import Fidelity

        hint = (" Use the foveance_expand tool to retrieve it." if self.expand_tool else "")
        note = f"\n[Foveance compressed item {iid} ({kind}, {len(text)} chars).{hint}]"
        if level == Fidelity.FULL:
            return text
        if level == Fidelity.DIGEST:
            return _digest_text(text, query_terms=qt) + note
        if level == Fidelity.GIST:
            head = "\n".join(text.splitlines()[:3])[:240]
            return head + "\n[...]" + note
        return f"[Foveance item {iid} ({kind}, {len(text)} chars) elided.{hint}]"

    def _alloc_or_digest(self, text: str, kind: str, conv_id: str, qt: Optional[set],
                         levels: Optional[dict]) -> str:
        """Compress one payload: by allocated fidelity when the allocator ran, else salience
        digestion (the pre-R1 behaviour)."""
        if levels is not None:
            from .vault import item_id_for
            iid = item_id_for(conv_id, text)
            if iid in levels:
                return self._render_level(text, kind, iid, levels[iid], qt)
        return _digest_text(text, query_terms=qt)

    def _collect_candidates(self, conv_id: str, payloads: list) -> Optional[dict]:
        """When the anticipatory agentic allocator is on, register (item_id, kind, text) payloads
        and return their fidelity allocation; otherwise None (digestion path)."""
        if not self.agentic_allocator:
            return None
        found, last_text = payloads
        if not found:
            return None
        from .vault import item_id_for
        cands = [(item_id_for(conv_id, text), kind, text) for kind, text in found]
        return self._agentic_levels(conv_id, cands, last_text)

    def _compress_agentic_anthropic(self, messages: list[dict],
                                    conv_id: str = "agentic") -> list[dict]:
        """Structure-preserving compression for Anthropic tool-use requests: keep every message,
        role, and tool_use/tool_result id intact; protect the last ``agentic_protect_last`` turns;
        compress only large content payloads in older turns. With ``agentic_allocator=True`` the
        payloads get graded fidelities from the anticipatory allocator (and are vaulted for
        re-inflation); otherwise they are salience-digested in place.

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
        qt = _terms(_extract_text(messages[-1].get("content", ""))) if messages else None

        def _payloads():
            found = []
            for i in range(start, cut):
                c = messages[i].get("content")
                if isinstance(c, str) and len(c) > self.agentic_min_chars:
                    found.append((messages[i].get("role", "message"), c))
                elif isinstance(c, list):
                    for b in c:
                        if not isinstance(b, dict) or b.get("cache_control"):
                            continue
                        if b.get("type") == "text" and isinstance(b.get("text"), str) \
                                and len(b["text"]) > self.agentic_min_chars:
                            found.append(("text", b["text"]))
                        elif b.get("type") == "tool_result" and isinstance(b.get("content"), str) \
                                and len(b["content"]) > self.agentic_min_chars:
                            found.append(("tool_output", b["content"]))
            return found

        levels = self._collect_candidates(
            conv_id, [_payloads(), _extract_text(messages[-1].get("content", ""))]
        ) if messages else None
        # When the agentic codec is on, replace the lossy per-payload digest with a single lossless
        # codec pass across those same payloads (consumed below in the identical traversal order).
        cod = self._codec_stream([p for _, p in _payloads()]) if self.agentic_codec else None

        def _rewrite(text: str, kind: str) -> str:
            return next(cod) if cod is not None else self._alloc_or_digest(
                text, kind, conv_id, qt, levels)

        def _blk(b):
            if not isinstance(b, dict) or b.get("cache_control"):
                return b
            if b.get("type") == "text" and isinstance(b.get("text"), str) \
                    and len(b["text"]) > self.agentic_min_chars:
                nd = _rewrite(b["text"], "text")
                return b if nd == b["text"] else {**b, "text": nd}
            if b.get("type") == "tool_result" and isinstance(b.get("content"), str) \
                    and len(b["content"]) > self.agentic_min_chars:
                nd = _rewrite(b["content"], "tool_output")
                return b if nd == b["content"] else {**b, "content": nd}
            return _digest_block(b, qt)

        out = []
        for i, m in enumerate(messages):
            c = m.get("content")
            if i >= cut or i < start:
                out.append(m)
            elif isinstance(c, str):
                nd = (_rewrite(c, m.get("role", "message"))
                      if len(c) > self.agentic_min_chars else c)
                out.append(m if nd == c else {**m, "content": nd})
            elif isinstance(c, list):
                nc = [_blk(b) for b in c]
                out.append(m if nc == c else {**m, "content": nc})
            else:
                out.append(m)
        return out

    def _compress_agentic_openai(self, messages: list[dict],
                                 conv_id: str = "agentic") -> list[dict]:
        """Structure-preserving compression for OpenAI tool-use requests: protect recent turns and
        system, keep tool_calls/tool_call_id pairing intact, and compress large old payloads
        (allocator-graded when ``agentic_allocator=True``, salience-digested otherwise)."""
        cut = max(0, len(messages) - self.agentic_protect_last)
        qt = _terms(_extract_text(messages[-1].get("content", ""))) if messages else None

        def _payloads():
            found = []
            for i in range(cut):
                m = messages[i]
                c = m.get("content")
                if m.get("role") == "system":
                    continue
                if isinstance(c, str) and (m.get("role") == "tool"
                                           or len(c) > self.agentic_min_chars):
                    kind = "tool_output" if m.get("role") == "tool" else m.get("role", "message")
                    found.append((kind, c))
            return found

        levels = self._collect_candidates(
            conv_id, [_payloads(), _extract_text(messages[-1].get("content", ""))]
        ) if messages else None
        cod = self._codec_stream([p for _, p in _payloads()]) if self.agentic_codec else None

        out = []
        for i, m in enumerate(messages):
            c = m.get("content")
            if i >= cut or m.get("role") == "system":
                out.append(m)
            elif isinstance(c, str) and (m.get("role") == "tool" or len(c) > self.agentic_min_chars):
                kind = "tool_output" if m.get("role") == "tool" else m.get("role", "message")
                nd = next(cod) if cod is not None else self._alloc_or_digest(c, kind, conv_id, qt,
                                                                             levels)
                out.append(m if nd == c else {**m, "content": nd})
            elif isinstance(c, list):
                nc = [_digest_block(b, qt) for b in c]
                out.append(m if nc == c else {**m, "content": nc})
            else:
                out.append(m)
        return out

    def _compress_agentic_responses(self, items: list[dict],
                                    conv_id: str = "agentic") -> list[dict]:
        """In-place compression for the OpenAI Responses API ``input`` list: protect the most recent
        items and compress large old tool outputs / message text, keeping every item, type, role,
        and ``call_id`` intact so the request stays valid (used by Codex and the Agents SDK)."""
        cut = max(0, len(items) - self.agentic_protect_last)
        last_text = ""
        for it in reversed(items):
            if isinstance(it, dict) and it.get("type") in (None, "message"):
                last_text = _extract_text(it.get("content", ""))
                if last_text:
                    break
        qt = _terms(last_text) if last_text else None

        def _payloads():
            found = []
            for it in items[:cut]:
                if isinstance(it, dict) and it.get("type") == "function_call_output" \
                        and isinstance(it.get("output"), str) \
                        and len(it["output"]) > self.agentic_min_chars:
                    found.append(("tool_output", it["output"]))
            return found

        levels = self._collect_candidates(conv_id, [_payloads(), last_text]) if items else None
        cod = self._codec_stream([p for _, p in _payloads()]) if self.agentic_codec else None

        def _item(it):
            if isinstance(it, dict) and it.get("type") == "function_call_output" \
                    and isinstance(it.get("output"), str) \
                    and len(it["output"]) > self.agentic_min_chars:
                nd = next(cod) if cod is not None else self._alloc_or_digest(
                    it["output"], "tool_output", conv_id, qt, levels)
                return it if nd == it["output"] else {**it, "output": nd}
            return _digest_responses_item(it)

        return [it if i >= cut else _item(it) for i, it in enumerate(items)]

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
        new = self._compress_agentic_responses(inp, conv_id="resp-agentic")
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
            conv_id = "ag-" + self._conv_id(request, msgs)
            new_messages = self._compress_agentic_openai(msgs, conv_id=conv_id)
            fwd = dict(request)
            fwd["messages"] = new_messages
            fwd.pop("conversation_id", None)
            if self.expand_tool and not request.get("stream"):
                fwd["tools"] = list(request.get("tools") or []) + [_EXPAND_TOOL_OPENAI]
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
            conv_id = "ag-" + self._conv_id(request, msgs)
            new_messages = self._compress_agentic_anthropic(msgs, conv_id=conv_id)
            fwd = dict(request)
            fwd["messages"] = new_messages  # system left intact (preserves its cache_control)
            if self.expand_tool and not request.get("stream"):
                fwd["tools"] = list(request.get("tools") or []) + [_EXPAND_TOOL_ANTHROPIC]
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

    # ------------------------------------------------------------ foveance_expand resolution
    def _vault_lookup(self, item_id: str) -> str:
        full = None
        if self.vault is not None:
            try:
                full = self.vault.get_any(item_id)  # type: ignore[attr-defined]
            except Exception:
                full = None
        if full is None:
            return (f"[foveance: item {item_id} not found. It may have been pruned; "
                    "answer from the visible context.]")
        self.expansions += 1
        return full

    def expand_requested_anthropic(self, data: dict):
        """(tool_use_id, item_id) when an Anthropic response calls foveance_expand, else None."""
        if not isinstance(data, dict) or data.get("stop_reason") != "tool_use":
            return None
        for blk in data.get("content") or []:
            if isinstance(blk, dict) and blk.get("type") == "tool_use" \
                    and blk.get("name") == "foveance_expand":
                return blk.get("id", ""), str((blk.get("input") or {}).get("item_id", ""))
        return None

    def expand_followup_anthropic(self, fwd: dict, data: dict,
                                  tool_use_id: str, item_id: str) -> dict:
        """Build the follow-up request answering a foveance_expand call with vault content."""
        nxt = dict(fwd)
        nxt["messages"] = list(fwd.get("messages") or []) + [
            {"role": "assistant", "content": data.get("content")},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use_id,
                                          "content": self._vault_lookup(item_id)}]},
        ]
        return nxt

    def expand_requested_openai(self, data: dict):
        """(tool_call_id, item_id, message) when an OpenAI chat response calls foveance_expand."""
        if not isinstance(data, dict):
            return None
        msg = ((data.get("choices") or [{}])[0] or {}).get("message") or {}
        for tc in msg.get("tool_calls") or []:
            fn = (tc or {}).get("function") or {}
            if fn.get("name") == "foveance_expand":
                import json as _json
                try:
                    item_id = str(_json.loads(fn.get("arguments") or "{}").get("item_id", ""))
                except Exception:
                    item_id = ""
                return tc.get("id", ""), item_id, msg
        return None

    def expand_followup_openai(self, fwd: dict, msg: dict,
                               tool_call_id: str, item_id: str) -> dict:
        nxt = dict(fwd)
        nxt["messages"] = list(fwd.get("messages") or []) + [
            msg,
            {"role": "tool", "tool_call_id": tool_call_id,
             "content": self._vault_lookup(item_id)},
        ]
        return nxt

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
        if self.savings_log is not None:
            try:
                self.savings_log.record(tb, ta)  # type: ignore[attr-defined]
            except Exception:
                pass  # persistence must never break request handling
        return stats

    def stats(self) -> dict:
        exact = self.token_counter is not None
        tb, ta = ((self.est_tokens_before, self.est_tokens_after) if exact
                  else (self.est_chars_before // 4, self.est_chars_after // 4))
        saved = max(tb - ta, 0)
        out: dict = {
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
            "evictions": self.evictions,
            "expansions": self.expansions,
            "codec": self.apply_codec,
            "agentic_codec": self.agentic_codec,
            "codec_saved_tokens": self.codec_saved_tokens,
            "per_conv": {cid: {"items": len(s.store.order), "turns": s.turn}
                         for cid, s in self.convs.items()},
        }
        if self.vault is not None:
            try:
                out["vault_items"] = self.vault.count()  # type: ignore[attr-defined]
            except Exception:
                pass
        if self.savings_log is not None:
            try:
                t = self.savings_log.totals()  # type: ignore[attr-defined]
                out["alltime"] = {**t, "usd_saved": round(
                    t["tokens_saved"] * self.price_per_mtok / 1e6, 4)}
            except Exception:
                pass
        return out


def build_app(proxy: Optional[FoveanceProxy] = None,
              upstream_url: str = "http://localhost:11434/v1",
              admin_token: Optional[str] = None):
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

    def _json_call(fwd: dict, path: str, headers: dict):  # pragma: no cover - needs upstream
        resp = _open(fwd, path, headers)
        data = json.loads(resp.read())
        resp.close()
        return data

    def _respond_expanding(fwd: dict, stats: dict, path: str, headers: dict,
                           protocol: str):  # pragma: no cover - needs upstream
        """Like _respond, but transparently resolves foveance_expand tool calls: when the model
        asks for a compressed item back, the proxy fetches it from the vault and re-issues the
        request, so the CLIENT never sees a tool it doesn't know. Non-streaming only."""
        if fwd.get("stream") or not px.expand_tool:
            return _respond(fwd, stats, path, headers)
        try:
            data = _json_call(fwd, path, headers)
            for _ in range(3):  # bounded expansion loop
                if protocol == "anthropic":
                    hit = px.expand_requested_anthropic(data)
                    if not hit:
                        break
                    fwd = px.expand_followup_anthropic(fwd, data, *hit)
                else:
                    hit = px.expand_requested_openai(data)
                    if not hit:
                        break
                    tc_id, item_id, msg = hit
                    fwd = px.expand_followup_openai(fwd, msg, tc_id, item_id)
                data = _json_call(fwd, path, headers)
        except urllib.error.HTTPError as e:
            return Response(content=e.read(), status_code=e.code,
                            media_type=e.headers.get("Content-Type", "application/json"))
        if isinstance(data, dict):
            stats["expansions"] = px.expansions
            data["foveance"] = stats
        return data

    @app.post("/v1/chat/completions")
    async def chat(request: Request):  # pragma: no cover - needs server
        fwd, stats = px.prepare(await request.json())
        return _respond_expanding(fwd, stats, "/chat/completions", _client_headers(request),
                                  protocol="openai")

    @app.post("/v1/messages")
    async def messages(request: Request):  # pragma: no cover - needs server
        fwd, stats = px.prepare_anthropic(await request.json())
        return _respond_expanding(fwd, stats, "/messages", _client_headers(request),
                                  protocol="anthropic")

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

    def _authed(request: "Request") -> bool:  # pragma: no cover - needs server
        """Admin auth: open by default; when --admin-token is set, require it via
        ``?token=...`` or ``Authorization: Bearer ...``."""
        if not admin_token:
            return True
        supplied = request.query_params.get("token") or \
            request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        return supplied == admin_token

    @app.get("/admin/stats")
    async def admin_stats(request: Request):  # pragma: no cover - needs server
        if not _authed(request):
            return Response(content="unauthorized", status_code=401)
        return px.stats()

    @app.get("/admin/export.csv")
    async def export_csv(request: Request):  # pragma: no cover - needs server
        if not _authed(request):
            return Response(content="unauthorized", status_code=401)
        if px.savings_log is None:
            return Response(content="Foveance Pro feature: persistent history export requires an "
                                    "active license (foveance license activate <key>).\n",
                            status_code=402, media_type="text/plain")
        return Response(content=px.savings_log.export_csv(),  # type: ignore[attr-defined]
                        media_type="text/csv",
                        headers={"Content-Disposition":
                                 "attachment; filename=foveance-savings.csv"})

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
  <div class="card" id="proCard" style="display:none;grid-column:1/-1">
    <div class="k">all-time saved (pro) &middot; <a href="/admin/export.csv"
      style="color:var(--indigo)">export csv</a></div>
    <div class="v" id="alltime">&ndash;</div></div>
</div>
<div class="foot"><span id="footNote">Estimates use chars/4 &asymp; tokens on the request payload</span>;
the $ figure uses the configured <code>--price-per-mtok</code>. JSON at
<a href="/admin/stats" style="color:var(--indigo)">/admin/stats</a>.</div>
</main><script>
const f = n => n >= 1e6 ? (n/1e6).toFixed(2)+"M" : n >= 1e3 ? (n/1e3).toFixed(1)+"k" : String(n);
async function tick(){
  try {
    const s = await (await fetch("/admin/stats" + location.search)).json();
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
    if (s.alltime) {
      document.getElementById("proCard").style.display = "block";
      document.getElementById("alltime").textContent =
        f(s.alltime.tokens_saved) + " tokens  \\u2248 $" + s.alltime.usd_saved.toFixed(2);
    }
  } catch (e) {}
}
tick(); setInterval(tick, 2000);
</script></body></html>
"""
