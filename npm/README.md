# foveance-proxy (npm launcher)

<p align="center">
  <a href="https://github.com/aimaghsoodi/foveance">
    <img alt="Foveance — the lossless token codec for LLM context, up to 82% fewer tokens" src="https://raw.githubusercontent.com/aimaghsoodi/foveance/main/assets/social-card.png" width="640">
  </a>
</p>

**Cut LLM input tokens up to 82% — losslessly.** Not one line of your app changed.

Your AI agent re-sends its whole history on every turn — the same listings, tool outputs and stack
traces, over and over. [Foveance](https://github.com/aimaghsoodi/foveance) is a real **lossless
codec** for that context: like `gzip` finds repeated bytes, it replaces text that already appeared
with a compact back-reference, removing tokens without removing information. Nothing is dropped,
every fact survives.

This is a tiny Node launcher for the Foveance proxy, so users of Node-based AI tools (Claude Code,
OpenAI Codex, opencode, Crush, Continue, ...) can start it with one command, without a manual
Python step. The proxy is **OpenAI- and Anthropic-compatible**, streams, and forwards your API key
untouched — you only point a client's base URL at it.

## Prerequisites

- Node.js >= 16
- Python >= 3.10 with Foveance installed once:

```bash
pip install foveance
```

## Use

```bash
# OpenAI upstream
npx foveance-proxy --upstream https://api.openai.com/v1

# Anthropic upstream (Claude Code, Anthropic SDK)
npx foveance-proxy --upstream https://api.anthropic.com/v1

# Local Ollama / vLLM / TGI / LM Studio
npx foveance-proxy --upstream http://localhost:11434/v1
```

All flags are forwarded to `foveance proxy` (`--host`, `--port`, `--budget`, `--drift`, `--policy`,
`--upstream`, `--cache-aware`, `--price-per-mtok`), and the `FOVEANCE_UPSTREAM` / `FOVEANCE_BUDGET`
/ `FOVEANCE_DRIFT` / `FOVEANCE_POLICY` environment variables are honoured too. A live tokens-saved
dashboard serves at `http://localhost:8799/` while the proxy runs.

> Even simpler: the Python CLI has `foveance wrap claude` (or `foveance wrap -- <any command>`),
> which starts the proxy, routes the tool through it, and prints a tokens-saved summary on exit.

Then point your tool at it (one variable):

```bash
# OpenAI-style tools, Codex, Ollama-backed clients
export OPENAI_BASE_URL=http://localhost:8799/v1

# Anthropic-style tools, Claude Code
export ANTHROPIC_BASE_URL=http://localhost:8799
```

See the [main README](https://github.com/aimaghsoodi/foveance#cut-your-token-usage-everywhere-drop-in-proxy-zero-client-changes)
and [`docs/usage.md`](https://github.com/aimaghsoodi/foveance/blob/main/docs/usage.md) for per-tool
recipes. This launcher is a convenience wrapper; the proxy itself lives in the Python package.

Apache-2.0.
