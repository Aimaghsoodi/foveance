"""v0.3 features: vault durability, conversation eviction, salience digests, the anticipatory
agentic allocator, and the foveance_expand re-inflation flow."""
import time

from foveance.proxy import (FoveanceProxy, _digest_text, _EXPAND_TOOL_ANTHROPIC,
                            _EXPAND_TOOL_OPENAI)
from foveance.vault import ItemVault, item_id_for


# ------------------------------------------------------------------------------- vault (R4)
def test_vault_roundtrip_idempotent_and_prune(tmp_path):
    v = ItemVault(path=str(tmp_path / "v.db"))
    iid = item_id_for("conv1", "hello world")
    v.put("conv1", iid, "tool_output", "hello world")
    v.put("conv1", iid, "tool_output", "hello world")          # idempotent
    assert v.count() == 1
    assert v.get("conv1", iid) == "hello world"
    assert v.get_any(iid) == "hello world"
    assert v.get("conv1", "nope") is None
    assert v.prune(older_than_days=0) == 1                     # everything is older than "now"
    assert v.count() == 0


def test_item_ids_are_stable_and_conversation_scoped():
    assert item_id_for("c", "text") == item_id_for("c", "text")
    assert item_id_for("c1", "text") != item_id_for("c2", "text")


def test_vault_leaves_no_open_handles(tmp_path):
    # Regression: sqlite3's own ``with conn`` commits but does not close the handle; on Windows a
    # lingering handle blocks the .db file from being unlinked. Every op must close its connection.
    import os
    sub = tmp_path / "vaultdir"
    sub.mkdir()
    p = str(sub / "v.db")
    v = ItemVault(path=p)
    iid = item_id_for("c", "payload")
    v.put("c", iid, "tool_output", "payload")
    assert v.get_any(iid) == "payload"
    v.count()
    v.prune(older_than_days=999)
    os.remove(p)                       # would raise PermissionError on Windows if a handle leaked
    assert not os.path.exists(p)


# ---------------------------------------------------------------------------- eviction (R4)
def test_conv_eviction_lru_and_ttl():
    px = FoveanceProxy(budget=120, max_convs=3, conv_ttl_s=9999)
    for i in range(5):
        px._state(f"c{i}")
    assert len(px.convs) <= 3 and px.evictions >= 2
    assert "c4" in px.convs                                     # newest survives
    # TTL: age one conversation artificially
    px.convs["c4"].last_used = time.time() - 10 ** 6
    px.conv_ttl_s = 10.0
    px._state("fresh")                                          # triggers eviction sweep
    assert "c4" not in px.convs


# ---------------------------------------------------------------------- salience digest (R6)
def test_salience_digest_keeps_query_relevant_line():
    lines = [f"log {i}: status=ok" for i in range(200)]
    lines[120] = "API_KEY=SECRET-9944 recorded here"
    text = "\n".join(lines)
    plain = _digest_text(text)
    assert "SECRET-9944" not in plain                           # blind head/tail loses it
    salient = _digest_text(text, query_terms={"api_key", "secret-9944"})
    assert "SECRET-9944" in salient                             # salience keeps it
    assert "elided by Foveance" in salient
    assert len(salient) < len(text)


# ------------------------------------------------------- anticipatory agentic allocator (R1)
def _agentic_request(big_relevant: str, big_noise: str) -> dict:
    tools = [{"name": "bash", "description": "x",
              "input_schema": {"type": "object", "properties": {}}}]
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "start"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "bash",
                                           "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                                      "content": big_relevant}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t2", "name": "bash",
                                           "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2",
                                      "content": big_noise}]},
        {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        {"role": "user", "content": [{"type": "text",
                                      "text": "what was the API_KEY from earlier?"}]},
    ]
    return {"system": "s", "messages": msgs, "tools": tools, "user": "alloc-test"}


def test_agentic_allocator_grades_by_relevance_and_vaults(tmp_path):
    big_relevant = "API_KEY=SECRET-7777\n" + "\n".join(f"cfg {i}: ok" for i in range(150))
    big_noise = "\n".join(f"noise {i}: zzz" for i in range(150))
    vault = ItemVault(path=str(tmp_path / "v.db"))
    px = FoveanceProxy(budget=400, agentic_protect_last=1, agentic_allocator=True, vault=vault)
    fwd, stats = px.prepare_anthropic(_agentic_request(big_relevant, big_noise))
    assert stats["reason"] == "agentic-inplace" and stats["compressed"] is True
    # structure preserved
    assert [m["role"] for m in fwd["messages"]] == \
        [m["role"] for m in _agentic_request("a", "b")["messages"]]
    assert fwd["messages"][1]["content"][0]["id"] == "t1"
    out_rel = fwd["messages"][2]["content"][0]["content"]
    out_noise = fwd["messages"][4]["content"][0]["content"]
    # the query-relevant item keeps its key content; the noise item is compressed harder
    assert "SECRET-7777" in out_rel
    assert len(out_noise) < len(big_noise)
    # both full texts are recoverable from the vault
    assert vault.get_any(item_id_for("ag-alloc-test", big_relevant)) == big_relevant
    assert vault.get_any(item_id_for("ag-alloc-test", big_noise)) == big_noise


def test_agentic_allocator_markers_reference_expand_tool(tmp_path):
    big_noise = "\n".join(f"noise {i}: zzz" for i in range(150))
    px = FoveanceProxy(budget=60, agentic_protect_last=1, agentic_allocator=True,
                       expand_tool=True, vault=ItemVault(path=str(tmp_path / "v.db")))
    fwd, _ = px.prepare_anthropic(_agentic_request(big_noise + "A", big_noise + "B"))
    flat = str(fwd["messages"])
    assert "foveance_expand" in flat                            # markers tell the model how

def test_expand_tool_injected_only_for_nonstreaming(tmp_path):
    vault = ItemVault(path=str(tmp_path / "v.db"))
    px = FoveanceProxy(budget=200, expand_tool=True, vault=vault)
    req = _agentic_request("x" * 2000, "y" * 2000)
    fwd, _ = px.prepare_anthropic(dict(req))
    assert any(t.get("name") == "foveance_expand" for t in fwd["tools"])
    fwd2, _ = px.prepare_anthropic({**_agentic_request("x" * 2000, "y" * 2000), "stream": True})
    assert all(t.get("name") != "foveance_expand" for t in fwd2["tools"])


# ------------------------------------------------------------- expand resolution flow (R1)
def test_expand_flow_anthropic_roundtrip(tmp_path):
    vault = ItemVault(path=str(tmp_path / "v.db"))
    vault.put("c", "abc123", "tool_output", "THE FULL SECRET CONTENT")
    px = FoveanceProxy(vault=vault, expand_tool=True)
    data = {"stop_reason": "tool_use",
            "content": [{"type": "text", "text": "let me check"},
                        {"type": "tool_use", "id": "tu9", "name": "foveance_expand",
                         "input": {"item_id": "abc123"}}]}
    hit = px.expand_requested_anthropic(data)
    assert hit == ("tu9", "abc123")
    fwd = px.expand_followup_anthropic({"messages": [{"role": "user", "content": "q"}]},
                                       data, *hit)
    assert fwd["messages"][-1]["content"][0]["tool_use_id"] == "tu9"
    assert fwd["messages"][-1]["content"][0]["content"] == "THE FULL SECRET CONTENT"
    assert px.expansions == 1
    # a normal end-turn response is not treated as an expansion request
    assert px.expand_requested_anthropic({"stop_reason": "end_turn", "content": []}) is None


def test_expand_flow_openai_roundtrip_and_missing_item(tmp_path):
    vault = ItemVault(path=str(tmp_path / "v.db"))
    px = FoveanceProxy(vault=vault, expand_tool=True)
    msg = {"role": "assistant", "content": None,
           "tool_calls": [{"id": "call9", "type": "function",
                           "function": {"name": "foveance_expand",
                                        "arguments": '{"item_id": "missing1"}'}}]}
    hit = px.expand_requested_openai({"choices": [{"message": msg}]})
    assert hit is not None and hit[0] == "call9" and hit[1] == "missing1"
    fwd = px.expand_followup_openai({"messages": []}, msg, hit[0], hit[1])
    assert fwd["messages"][-1]["role"] == "tool"
    assert "not found" in fwd["messages"][-1]["content"]        # graceful missing-item message
    assert _EXPAND_TOOL_ANTHROPIC["name"] == _EXPAND_TOOL_OPENAI["function"]["name"]
