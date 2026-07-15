# Using Foveance: the LLM context codec (drop-in)

Foveance is a **token-compression codec** with a measured ratio, not a message trimmer. It has a
*lossless* core (a cross-item redundancy codec — `foveance compress` / `foveance.compress()`), an
*anticipatory* allocation layer that down-renders whole stale items under a budget, and a
re-inflation tool that makes that layer loss-free in effect. The theory and the measured ratios
are in [`compression.md`](compression.md). The one-liner:

```bash
foveance compress my_trace.jsonl     # lossless: e.g. "861 -> 376 tokens (56.3% saved, 2.29x)"
```


## The one-command way: `foveance wrap`
```bash
foveance wrap claude                       # Claude Code, routed through Foveance
foveance wrap -- codex "fix the tests"     # any other CLI (flags before the --)
foveance wrap --upstream http://localhost:11434/v1 -- aider   # local models too
```
`wrap` starts the proxy on localhost, sets **exactly the base-URL env vars the target tool reads**
(Anthropic tools get `ANTHROPIC_BASE_URL` at the proxy root; OpenAI tools get `OPENAI_BASE_URL` /
`OPENAI_API_BASE` at root `/v1`) **for the child process only**, launches the tool, and prints a
tokens-saved summary when it exits. The upstream and env vars come from a registry of known
agents, overridable with `--upstream` / `FOVEANCE_UPSTREAM`. All the proxy flags below
(`--budget`, `--drift`, `--cache-aware`, `--price-per-mtok`, ...) work on `wrap` too. While it
runs, a live dashboard serves at `http://localhost:8799/`.

### Supported agents (`foveance adapters`)
Foveance ships first-class adapters for the popular agent CLIs and SDKs, so `wrap` targets them by
name with the right dialect and env vars:

| adapter | dialect | launch |
|---|---|---|
| Claude Code | Anthropic | `foveance wrap claude` |
| Codex | OpenAI (Chat + Responses) | `foveance wrap -- codex "..."` |
| Aider | OpenAI | `foveance wrap -- aider` |
| Cline / opencode / Goose / Continue | OpenAI | `foveance wrap <tool>` |
| OpenAI / Anthropic SDKs, LiteLLM | either | `foveance wrap <tool>` |

Run `foveance adapters` to list them. For a **long-running** proxy (started separately with
`foveance proxy`), print the exports for your tool instead of wrapping it:
```bash
eval "$(foveance env codex --port 8799)"     # bash: point Codex at a running proxy
foveance env claude-code                      # or copy the PowerShell lines it also prints
```
Unknown tool? `foveance wrap -- <cmd>` still sets the common base-URL vars, and `--upstream` picks
the API it forwards to. (The benchmark's OpenRouter client is separate and internal — not a
user-facing adapter.)

## The manual way: run the proxy yourself

Foveance ships an OpenAI- and Anthropic-compatible reverse proxy. You start it, point any client's
base URL at it, and it transparently compresses the accumulated context under a token budget
before forwarding to the real model. Your existing API key is forwarded unchanged. No client code
changes.

```bash
pip install foveance
# forward to OpenAI:
foveance proxy --upstream https://api.openai.com/v1
# or to Anthropic:
foveance proxy --upstream https://api.anthropic.com/v1
# or to a local Ollama / vLLM / TGI / LM Studio server:
foveance proxy --upstream http://localhost:11434/v1
```

It listens on `http://localhost:8799` by default (`--host`, `--port` to change).

The proxy exposes:
- `POST /v1/chat/completions` (OpenAI format, honours `"stream": true`)
- `POST /v1/messages` (Anthropic format, honours streaming)
- `GET  /v1/models` (passed through so clients that probe the model list succeed)
- `GET  /health` (JSON liveness for orchestrators)
- `GET  /` and `/admin` (live dashboard: tokens saved, % and ≈$ at `--price-per-mtok`)
- `GET  /admin/stats` (the same numbers as JSON, plus per-conversation detail)

### Configuration (flags or environment)
Every flag has an env equivalent, so you can configure it without editing code (all of these work
on `foveance wrap` as well):

| Flag | Env var | Default | Meaning |
|---|---|---|---|
| `--upstream` | `FOVEANCE_UPSTREAM` | `http://localhost:11434/v1` | upstream base URL |
| `--budget` | `FOVEANCE_BUDGET` | `2000` | per-turn token budget for the rendered context |
| `--drift` | `FOVEANCE_DRIFT` | `0.6` | anticipation strength (0 = reactive AFM behaviour) |
| `--policy` | `FOVEANCE_POLICY` | `foveance` | `foveance`/`reactive_afm`/`recency`/`full` |
| `--agentic-protect-last` | `FOVEANCE_AGENTIC_PROTECT_LAST` | `3` | recent tool-use turns kept full |
| `--cache-aware` | — | off | never modify content at/before the last Anthropic `cache_control` breakpoint (see [`limitations.md`](limitations.md) for when to enable) |
| `--price-per-mtok` | — | `3.0` | assumed $/M input tokens for the $-saved estimate |
| `--codec` | `FOVEANCE_CODEC` | off | apply the **lossless** cross-item redundancy codec to the assembled context (removes repeated line-runs with zero accuracy risk; stacks on any policy) |
| `--exact-tokens` | — | off | count with a real tokenizer (tiktoken) instead of the chars/4 heuristic |
| `--token-encoding` | `FOVEANCE_TOKEN_ENCODING` | `cl100k_base` | tiktoken encoding for `--exact-tokens`; use `o200k_base` for gpt-4o and newer |
| `--host` / `--port` | — | `0.0.0.0` / `8799` | bind address |

## v0.3: interactive compression, audit, and the learning loop

**Anticipatory agentic compression + re-inflation** (the flagship pair):
```bash
foveance proxy --agentic-allocator --expand-tool --upstream https://api.anthropic.com/v1
```
With `--agentic-allocator`, old tool outputs get graded fidelities from the anticipatory
allocator (not blind digestion). With `--expand-tool`, every compressed item carries an
addressable marker and the model can call `foveance_expand` to get the full content back; the
proxy resolves those calls itself against a durable vault (`~/.foveance/vault.db`), so the
client never sees the tool. Expansion runs on non-streaming requests; streaming requests fall
back to plain compression.

**Audit your own logs (no API key, nothing sent anywhere):**
```bash
foveance audit my_conversations.jsonl --monthly-requests 50000
```
Accepts JSON/JSONL of `messages` lists and prints the tokens/$ Foveance would have saved,
replayed per user turn exactly as agents pay.

**The learning loop (allocation that improves on your workload):**
```bash
foveance proxy --learn ...     # logs local traces of what each query referenced
foveance train                 # fits the relevance model on those traces
# subsequent --learn runs load the trained model automatically
```
Traces and the model live in `~/.foveance/`; nothing leaves the machine.

**Ops:** `--admin-token SECRET` protects `/admin*` (pass `?token=SECRET`); conversation state is
LRU+TTL evicted so long-running proxies stay bounded, with full texts recoverable via the vault.

## OpenAI SDK (Python)
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8799/v1", api_key="sk-...")  # your real key
client.chat.completions.create(model="gpt-4o-mini",
    messages=[...long history...])  # Foveance compresses the history before it leaves your machine
```

## Anthropic SDK (Python)
```python
import anthropic
client = anthropic.Anthropic(base_url="http://localhost:8799", api_key="sk-ant-...")
client.messages.create(model="claude-3-5-sonnet", max_tokens=512,
    system="...", messages=[...long history...])
```

## Coding agents and CLIs (copy-paste recipes)
Any agent that lets you set an OpenAI- or Anthropic-compatible base URL can route through Foveance.
Start the proxy once with your real provider as `--upstream`, then point the tool at it.

**Claude Code** (Anthropic protocol):
```bash
foveance proxy --upstream https://api.anthropic.com/v1 &
ANTHROPIC_BASE_URL=http://localhost:8799 claude
```
> Note: Claude Code is an agentic client &mdash; it declares a `tools` array on every request and
> relies on strict tool_use/tool_result pairing. The proxy detects this and applies
> **structure-preserving compression**: it keeps every message, role, and tool_use/tool_result pair
> intact, protects the most recent turns (`--agentic-protect-last`, default 3), and digests only
> large stale tool outputs in older turns. The request stays valid, so Claude Code keeps working
> (verified live) while large old tool output is trimmed. See [`limitations.md`](limitations.md).

**OpenAI Codex CLI** (Responses API). Codex speaks the OpenAI **Responses** protocol, which the
proxy supports at `/v1/responses` with in-place agentic compression. Two important notes: Codex will
not let you override its built-in `openai` provider, and **ChatGPT-subscription (OAuth) Codex cannot
be proxied** (the OAuth is pinned to the ChatGPT backend). To route Codex through Foveance, use an
**API key** and a *custom* provider in `~/.codex/config.toml`:
```toml
model_provider = "foveance"
[model_providers.foveance]
name = "foveance"
base_url = "http://localhost:8799/v1"
wire_api = "responses"
env_key = "OPENAI_API_KEY"   # your OpenAI API key; the proxy forwards it to api.openai.com
```
```bash
foveance proxy --upstream https://api.openai.com/v1 &
OPENAI_API_KEY=sk-... codex exec "say hello"
```

**Ollama** (local models, OpenAI-compatible endpoint):
```bash
foveance proxy --upstream http://localhost:11434/v1 &
export OPENAI_BASE_URL=http://localhost:8799/v1     # any OpenAI client now talks to Ollama, compressed
```

**Google Antigravity / Cursor / Continue / opencode / Crush / aider** — in the tool's settings set
the custom OpenAI base URL (sometimes called "OpenAI API base" or `--openai-api-base`) to
`http://localhost:8799/v1` and use your normal key. Generic env that many tools read:
```bash
export OPENAI_BASE_URL=http://localhost:8799/v1      # OpenAI-style tools and LiteLLM
export OPENAI_API_KEY=sk-...
export ANTHROPIC_BASE_URL=http://localhost:8799      # Anthropic-style tools
export ANTHROPIC_API_KEY=sk-ant-...
```

**Node / npm world** — a thin launcher wraps the Python proxy so JS-tool users need no separate
Python step (requires Python + `pip install foveance` once):
```bash
npx foveance-proxy --upstream https://api.openai.com/v1     # see the npm/ directory
```

## curl
```bash
curl http://localhost:8799/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","user":"conv-1","messages":[{"role":"user","content":"...big context..."},{"role":"user","content":"recall X"}]}'
```

## Notes
- The proxy keeps one multi-fidelity store per conversation. It uses an explicit id when the client
  sends one (OpenAI `user` field, or Anthropic `metadata.user_id`); otherwise it auto-derives a
  stable id from the first message so separate conversations stay separate without any client change.
  Passing an explicit `user` is still the most robust option for long-lived multi-session clients.
- `--budget` is the per-turn token budget for the rendered context; lower it to save more, raise it
  to keep more verbatim. The response carries an `foveance` field with what was compressed.
- The store is in-memory per process; restarting the proxy resets it. For production, run it behind
  your own auth and do not expose `/admin/stats` publicly (see `SECURITY.md`).
- Honest expectation: budgeted allocation reaches full-replay accuracy at roughly 60% fewer tokens
  on our benchmark; the savings depend on how much history you accumulate and how tight the budget
  is. It will not change answers when the budget already fits your context.

## Library (no proxy)
If you control the agent loop, use the library directly:
```python
from foveance import Controller, Item
from foveance.llm import OpenAICompatLLM      # or OllamaLLM, or your own LLM adapter
ctrl = Controller(OpenAICompatLLM(model="gpt-4o-mini", base_url="https://api.openai.com/v1"),
                  budget=2000, policy="foveance")
ctrl.add_item(Item("obs0", "tool_output", "FACT api_key=...\n...logs...", created_turn=0))
print(ctrl.step("recall api_key", turn=0).answer)
```
