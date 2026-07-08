#!/usr/bin/env python3
"""
Shrink a LlamaIndex chat history with Foveance before sending it to a chat engine.

Requires the `llamaindex` extra: pip install "foveance[llamaindex]"

Run: python examples/llamaindex_integration_demo.py
"""
from __future__ import annotations

import sys


def main() -> None:
    try:
        from llama_index.core.llms import ChatMessage, MessageRole
    except ImportError:
        print("This example needs llama-index-core: pip install \"foveance[llamaindex]\"")
        sys.exit(0)

    from foveance.integrations.llamaindex import shrink_chat_messages

    long_ctx = "FACT invoice_total=4821\n" + "\n".join(f"log line {i} status=ok" for i in range(200))
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content=long_ctx),
        ChatMessage(role=MessageRole.ASSISTANT, content="noted"),
        ChatMessage(role=MessageRole.USER, content="recall invoice_total"),
    ]
    before = sum(len(m.content or "") for m in messages)

    shrunk = shrink_chat_messages(messages, budget=200)  # feed `shrunk` to your chat engine
    after = sum(len(m.content or "") for m in shrunk)

    print(f"messages: {len(messages)} -> {len(shrunk)}   chars: {before} -> {after}")
    print("last turn preserved verbatim:", shrunk[-1].content == messages[-1].content)


if __name__ == "__main__":
    main()
