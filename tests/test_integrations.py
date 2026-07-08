"""Tests for the optional foveance.integrations.* modules (litellm, langchain)."""
import asyncio
import json
import sys
import types

import pytest


# ------------------------------------------------------------------------------------ litellm
def test_litellm_shrink_completion_shrinks_messages_before_call(monkeypatch):
    captured = {}
    fake_litellm = types.ModuleType("litellm")

    def fake_completion(**kwargs):
        captured["kwargs"] = kwargs
        return {"choices": [{"message": {"content": "ok"}}]}

    fake_litellm.completion = fake_completion
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    from foveance.integrations.litellm import shrink_completion

    long_ctx = "FACT secret=42\n" + "\n".join(f"log {i} ok" for i in range(200))
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": long_ctx},
        {"role": "assistant", "content": "noted"},
        {"role": "user", "content": "recall secret"},
    ]
    resp = shrink_completion(model="gpt-4o-mini", messages=messages, budget=120)
    assert resp["choices"][0]["message"]["content"] == "ok"
    fwd = captured["kwargs"]
    assert fwd["model"] == "gpt-4o-mini"
    assert fwd["messages"][-1]["content"] == "recall secret"
    assert len(json.dumps(fwd["messages"])) < len(long_ctx)


def test_litellm_ashrink_completion_shrinks_messages_before_call(monkeypatch):
    captured = {}
    fake_litellm = types.ModuleType("litellm")

    async def fake_acompletion(**kwargs):
        captured["kwargs"] = kwargs
        return {"choices": [{"message": {"content": "ok"}}]}

    fake_litellm.acompletion = fake_acompletion
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    from foveance.integrations.litellm import ashrink_completion

    messages = [{"role": "user", "content": "hi"}, {"role": "user", "content": "recall x"}]
    resp = asyncio.run(ashrink_completion(model="gpt-4o-mini", messages=messages, budget=500))
    assert resp["choices"][0]["message"]["content"] == "ok"
    assert captured["kwargs"]["messages"][-1]["content"] == "recall x"


def test_litellm_wrapper_passes_through_when_no_messages(monkeypatch):
    captured = {}
    fake_litellm = types.ModuleType("litellm")

    def fake_completion(**kwargs):
        captured["kwargs"] = kwargs
        return {"ok": True}

    fake_litellm.completion = fake_completion
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    from foveance.integrations.litellm import shrink_completion

    shrink_completion(model="gpt-4o-mini")
    assert "messages" not in captured["kwargs"]


# ----------------------------------------------------------------------------------- langchain
def test_langchain_shrink_messages_compresses_and_preserves_last_turn():
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    from foveance.integrations.langchain import shrink_messages

    long_ctx = "FACT secret=42\n" + "\n".join(f"log {i} ok" for i in range(200))
    messages = [
        SystemMessage(content="You are helpful."),
        HumanMessage(content=long_ctx),
        AIMessage(content="noted"),
        HumanMessage(content="recall secret"),
    ]
    chain = shrink_messages(budget=120)
    out = chain.invoke(messages)
    assert isinstance(out[-1], HumanMessage)
    assert out[-1].content == "recall secret"
    assert sum(len(m.content) for m in out) < sum(len(m.content) for m in messages)


def test_langchain_shrink_messages_passes_through_unsupported_message_types():
    pytest.importorskip("langchain_core")
    from langchain_core.messages import HumanMessage, ToolMessage

    from foveance.integrations.langchain import shrink_messages

    messages = [HumanMessage(content="hi"), ToolMessage(content="42", tool_call_id="t1")]
    chain = shrink_messages(budget=50)
    assert chain.invoke(messages) == messages


# --------------------------------------------------------------------------------- llamaindex
def test_llamaindex_shrink_chat_messages_compresses_and_preserves_last_turn():
    pytest.importorskip("llama_index.core")
    from llama_index.core.llms import ChatMessage, MessageRole

    from foveance.integrations.llamaindex import shrink_chat_messages

    long_ctx = "FACT secret=42\n" + "\n".join(f"log {i} ok" for i in range(200))
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are helpful."),
        ChatMessage(role=MessageRole.USER, content=long_ctx),
        ChatMessage(role=MessageRole.ASSISTANT, content="noted"),
        ChatMessage(role=MessageRole.USER, content="recall secret"),
    ]
    out = shrink_chat_messages(messages, budget=120)
    assert out[-1].role == MessageRole.USER
    assert out[-1].content == "recall secret"
    assert sum(len(m.content or "") for m in out) < sum(len(m.content or "") for m in messages)


def test_llamaindex_shrink_chat_messages_passes_through_unsupported_roles():
    pytest.importorskip("llama_index.core")
    from llama_index.core.llms import ChatMessage, MessageRole

    from foveance.integrations.llamaindex import shrink_chat_messages

    messages = [
        ChatMessage(role=MessageRole.USER, content="hi"),
        ChatMessage(role=MessageRole.TOOL, content="42"),
    ]
    assert shrink_chat_messages(messages, budget=50) == messages
