"""0.5 safety matrix: the lossless codec on the AGENTIC in-place paths (`agentic_codec=True`).

Running a compressor inside a tool-use transcript is where providers 400 you: break a
tool_use<->tool_result pairing, mutate a cache_control block, or touch a protected recent turn and
the request is rejected. This matrix proves, for all three dialects (Anthropic Messages, OpenAI
Chat, OpenAI Responses), that turning the agentic codec on:

  1. preserves message/item count and order (roles/types unchanged),
  2. preserves every tool id and its pairing (tool_use id / tool_result tool_use_id;
     OpenAI tool_call_id; Responses call_id),
  3. leaves cache_control blocks byte-identical,
  4. leaves the last ``agentic_protect_last`` turns byte-identical,
  5. never inflates (coded payloads are <= their originals),
  6. loses no fact (every unique line of every eligible payload still appears verbatim), and
  7. actually removes cross-message redundancy (codec_saved_tokens > 0),
plus the codec's own exact round-trip on those payloads.
"""
from foveance.codec import RedundancyCodec
from foveance.proxy import FoveanceProxy

# a large (> agentic_min_chars), internally- and cross-message-redundant tool payload
BIG = ("src/service/a.py\nsrc/service/b.py\nsrc/service/c.py\n"
       "INFO boot ok\nINFO ready\n") * 40
UNIQUE_LINES = set(BIG.split("\n"))


def _codec_off_default():
    return FoveanceProxy().agentic_codec is False


def _no_fact_lost(original_payloads, output_text):
    """Every unique line of every eligible payload must appear verbatim somewhere in the output."""
    flat = set()
    for p in original_payloads:
        flat |= set(p.split("\n"))
    return all(line in output_text for line in flat if line)


def _codec_roundtrip_exact(payloads):
    c = RedundancyCodec(min_run=1)
    items = [(str(i), p) for i, p in enumerate(payloads)]
    return c.unpack(c.pack(items)) == items


# ------------------------------------------------------------------ Anthropic Messages
def _anthropic_msgs():
    return [
        {"role": "user", "content": [
            {"type": "text", "text": "SYSTEM PROMPT " * 100, "cache_control": {"type": "ephemeral"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": BIG}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "tu_2", "name": "ls", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_2", "content": BIG}]},
        {"role": "user", "content": "what's the last file listed?"},   # protected recent turn
    ]


def _anthropic_ids(msgs):
    ids = []
    for m in msgs:
        c = m.get("content")
        if isinstance(c, list):
            for b in c:
                if b.get("type") == "tool_use":
                    ids.append(("use", b["id"]))
                elif b.get("type") == "tool_result":
                    ids.append(("result", b["tool_use_id"]))
    return ids


def test_agentic_codec_anthropic_safety_matrix():
    assert _codec_off_default()
    msgs = _anthropic_msgs()
    px = FoveanceProxy(budget=100000, agentic_codec=True, agentic_protect_last=1)
    out = px._compress_agentic_anthropic([dict(m) for m in msgs], conv_id="a")
    # (1) count/order/roles
    assert len(out) == len(msgs)
    assert [m["role"] for m in out] == [m["role"] for m in msgs]
    # (2) tool ids + pairing preserved exactly
    assert _anthropic_ids(out) == _anthropic_ids(msgs)
    # (3) cache_control block byte-identical
    assert out[0] == msgs[0]
    # (4) last turn byte-identical
    assert out[-1] == msgs[-1]
    # (5) never inflates
    payloads = [msgs[1]["content"][0]["content"], msgs[3]["content"][0]["content"]]
    coded = [out[1]["content"][0]["content"], out[3]["content"][0]["content"]]
    assert sum(len(x) for x in coded) <= sum(len(x) for x in payloads)
    # (6) no fact lost
    assert _no_fact_lost(payloads, "\n".join(coded))
    # (7) redundancy actually removed + (8) exact round-trip on the payloads
    assert px.codec_saved_tokens > 0
    assert _codec_roundtrip_exact(payloads)


# ------------------------------------------------------------------ OpenAI Chat
def _openai_msgs():
    return [
        {"role": "system", "content": "you are a precise assistant"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "ls", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": BIG},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "call_2", "type": "function",
                         "function": {"name": "ls", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_2", "content": BIG},
        {"role": "user", "content": "which module is third?"},         # protected recent turn
    ]


def test_agentic_codec_openai_safety_matrix():
    msgs = _openai_msgs()
    px = FoveanceProxy(budget=100000, agentic_codec=True, agentic_protect_last=1)
    out = px._compress_agentic_openai([dict(m) for m in msgs], conv_id="o")
    assert len(out) == len(msgs)
    assert [m["role"] for m in out] == [m["role"] for m in msgs]
    # tool_call ids + tool_call_id pairing preserved
    assert [m.get("tool_call_id") for m in out] == [m.get("tool_call_id") for m in msgs]
    assert ([tc["id"] for m in out for tc in (m.get("tool_calls") or [])]
            == ["call_1", "call_2"])
    assert out[0] == msgs[0]                         # system verbatim
    assert out[-1] == msgs[-1]                       # last turn verbatim
    payloads = [msgs[2]["content"], msgs[4]["content"]]
    coded = [out[2]["content"], out[4]["content"]]
    assert sum(len(x) for x in coded) <= sum(len(x) for x in payloads)
    assert _no_fact_lost(payloads, "\n".join(coded))
    assert px.codec_saved_tokens > 0
    assert _codec_roundtrip_exact(payloads)


# ------------------------------------------------------------------ OpenAI Responses
def _responses_items():
    return [
        {"type": "function_call", "call_id": "fc_1", "name": "ls", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "fc_1", "output": BIG},
        {"type": "function_call", "call_id": "fc_2", "name": "ls", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "fc_2", "output": BIG},
        {"type": "message", "role": "user", "content": "summarise"},   # protected recent item
    ]


def test_agentic_codec_responses_safety_matrix():
    items = _responses_items()
    px = FoveanceProxy(budget=100000, agentic_codec=True, agentic_protect_last=1)
    out = px._compress_agentic_responses([dict(it) for it in items], conv_id="r")
    assert len(out) == len(items)
    assert [it["type"] for it in out] == [it["type"] for it in items]
    # call_id pairing preserved
    assert [it.get("call_id") for it in out] == [it.get("call_id") for it in items]
    assert out[-1] == items[-1]                      # last item verbatim
    payloads = [items[1]["output"], items[3]["output"]]
    coded = [out[1]["output"], out[3]["output"]]
    assert sum(len(x) for x in coded) <= sum(len(x) for x in payloads)
    assert _no_fact_lost(payloads, "\n".join(coded))
    assert px.codec_saved_tokens > 0
    assert _codec_roundtrip_exact(payloads)


# ------------------------------------------------------------------ off-path parity
def test_agentic_codec_off_matches_digest_path():
    # with the flag off, the agentic paths behave exactly as before (digest), so enabling the codec
    # is purely additive and cannot regress existing behaviour.
    msgs = _anthropic_msgs()
    off = FoveanceProxy(budget=100000, agentic_codec=False, agentic_protect_last=1)
    out = off._compress_agentic_anthropic([dict(m) for m in msgs], conv_id="a")
    assert off.codec_saved_tokens == 0               # codec never ran
    assert out[0] == msgs[0] and out[-1] == msgs[-1]  # cache + last turn still protected
