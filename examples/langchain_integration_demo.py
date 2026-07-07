#!/usr/bin/env python3
"""
Slot Foveance into an LCEL chain as a message-shrinking step.

Requires the `langchain` extra: pip install "foveance[langchain]"

Run: python examples/langchain_integration_demo.py
"""
from __future__ import annotations

import sys


def main() -> None:
    try:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    except ImportError:
        print("This example needs langchain-core: pip install \"foveance[langchain]\"")
        sys.exit(0)

    from foveance.integrations.langchain import shrink_messages

    long_ctx = "FACT invoice_total=4821\n" + "\n".join(f"log line {i} status=ok" for i in range(200))
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=long_ctx),
        AIMessage(content="noted"),
        HumanMessage(content="recall invoice_total"),
    ]
    before = sum(len(m.content) for m in messages)

    chain = shrink_messages(budget=200)   # slot this anywhere in an LCEL chain: chain | prompt | model
    shrunk = chain.invoke(messages)
    after = sum(len(m.content) for m in shrunk)

    print(f"messages: {len(messages)} -> {len(shrunk)}   chars: {before} -> {after}")
    print("last turn preserved verbatim:", shrunk[-1].content == messages[-1].content)


if __name__ == "__main__":
    main()
