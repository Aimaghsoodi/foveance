<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/aimaghsoodi/foveance/main/assets/logo-dark.png">
    <img alt="Foveance" src="https://raw.githubusercontent.com/aimaghsoodi/foveance/main/assets/logo.png" width="440">
  </picture>
</p>

<p align="center"><b>Cut your LLM token bill by 60%+ — without changing your code or your answers.</b></p>

<p align="center">
  <a href="https://pypi.org/project/foveance/"><img alt="PyPI" src="https://img.shields.io/pypi/v/foveance?color=blue"></a>
  <a href="https://pypi.org/project/foveance/"><img alt="Downloads" src="https://img.shields.io/pypi/dm/foveance?color=blue"></a>
  <a href="https://github.com/aimaghsoodi/foveance/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/aimaghsoodi/foveance/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-green"></a>
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/python-3.10%E2%80%933.13-blue"></a>
  <a href="https://aimaghsoodi.github.io/foveance/"><img alt="Docs" src="https://img.shields.io/badge/docs-online-6366f1"></a>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/AbteeXAILabs/foveance"><img alt="Live demo" src="https://img.shields.io/badge/%F0%9F%A4%97%20live%20demo-try%20in%20browser-yellow"></a>
  <a href="https://colab.research.google.com/github/aimaghsoodi/foveance/blob/main/examples/foveance_quickstart.ipynb"><img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>
<p align="center">
  <a href="https://huggingface.co/spaces/AbteeXAILabs/foveance"><b>Try the live demo</b></a>
  &nbsp;<b>·</b>&nbsp; <a href="#get-started-in-30-seconds">30-second start</a>
  &nbsp;<b>·</b>&nbsp; <a href="https://aimaghsoodi.github.io/foveance/">Docs</a>
</p>

---

## What is this?

When you chat with an AI agent for a while, the conversation history keeps piling up. You pay
for **every** old message on **every** new turn, and past a point the model actually gets
*worse* because the important facts are buried under clutter.

**Foveance fixes that automatically.** It keeps the parts of the history that still matter,
trims the parts that don't, and hands the model a shorter context — so you get the **same
answers for a fraction of the tokens**. Nothing is deleted forever, and you don't change a
single line of your app.

> In real tests it kept full accuracy while using **60–64% fewer tokens**, and it *correctly*
> recalled a buried fact that the full, uncompressed history got **wrong**.

---

## Get started in 30 seconds

### Option A — you use a coding agent (Claude Code, Codex, aider, …)
One command. It runs your tool exactly as before, just cheaper, and prints how much you saved:

```bash
pip install foveance
foveance wrap claude          # or:  foveance wrap -- codex "fix the tests"
```

<p align="center"><img alt="foveance wrap demo — 3,590 to 1,677 input tokens, -53%, same answer" src="https://raw.githubusercontent.com/aimaghsoodi/foveance/main/assets/demo.gif" width="640"></p>

That's the whole thing. Your API key is untouched, nothing is stored, and a live
"tokens saved ≈ $" dashboard runs at <http://localhost:8799/> while you work.

### Option B — you write Python
One install, one function. No server, no config, nothing to run:

```bash
pip install foveance
```
```python
from foveance import shrink

smaller = shrink(messages, budget=2000)   # messages = your OpenAI-style list
# ...now send `smaller` to your model instead of `messages`. Same answers, fewer tokens.
```

`shrink` keeps your system prompt and your latest message exactly as-is and intelligently
compresses the older turns. That's all you need to start.

<p align="center"><img alt="foveance.shrink() collapses older turns and keeps the system prompt + last turn; input tokens drop 53%" src="https://raw.githubusercontent.com/aimaghsoodi/foveance/main/assets/shrink-demo.gif" width="660"></p>

### Option C — try it right now, no API key, no GPU
```bash
pip install foveance
foveance demo
```
Prints a side-by-side table showing the token savings on a built-in example.

---

## Does it actually work? (real numbers, nothing invented)

Measured on **Gemma 2 (2B), Llama 3.2 (1B), and Qwen 2.5 (1.5B)** via Ollama, 5 seeds each.
At a tight token budget, Foveance matched the full, uncompressed accuracy using **~⅓ of the
tokens**, while the naive shortcuts (keep-recent, truncate, spread-evenly) failed:

<p align="center"><img alt="with vs without Foveance: same accuracy, 64% fewer tokens" src="https://raw.githubusercontent.com/aimaghsoodi/foveance/main/assets/demo_comparison.gif" width="640"></p>

| Model | full (no compression) | keep-recent | truncate | spread-evenly | **Foveance** |
|---|---|---|---|---|---|
| gemma2:2b   | 1.00 (10.2k tok) | 0.67 | 0.00 | 0.00 | **1.00 (3.7k tok)** |
| llama3.2:1b | 1.00 (8.3k tok)  | 0.67 | 0.00 | 1.00 | **1.00 (3.1k tok)** |
| qwen2.5:1.5b| 1.00 (9.9k tok)  | 0.67 | 0.00 | 0.00 | **1.00 (3.6k tok)** |

Accuracy is "did it recall the buried fact." Foveance holds 1.00 at ~⅓ the tokens on every
model; the shortcuts drop the fact. Every number traces to a CSV in
[`bench/results/`](bench/results/) — nothing is hand-entered.

<sub>Full benchmark, head-to-head vs LLMLingua-2, and the theory are further down and in
[`bench/report.md`](bench/report.md) / [`docs/`](docs/).</sub>

---

<details>
<summary><b>Install options</b> (click to expand)</summary>

```bash
pip install foveance          # everything you normally need: shrink(), foveance wrap, the proxy, and the demo
pip install "foveance[all]"   # the above plus the ML embedder and benchmark tooling (numpy, torch, matplotlib, …)
```
The allocator/predictor core imports no heavy libraries; the base install adds only the small
web-server packages that power `foveance wrap` and the proxy.
</details>

---

# Under the hood (the technical part)

Everything above is all most people need. The rest of this document is for people who want the
proxy details, the full benchmark, and the theory.

## The drop-in proxy — cut tokens for *any* tool, zero code changes

`foveance wrap <tool>` is a convenience wrapper around a small reverse proxy you can also run
yourself. It speaks the OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages wire
protocols, streams, and forwards your credentials untouched. It keeps a per-conversation
multi-fidelity store and spends a token budget on the context most likely to matter next, before
forwarding upstream.

### Run with Docker (Recommended)
You can run the Foveance proxy with zero Python setup using Docker:
```bash
docker run -p 8799:8799 ghcr.io/aimaghsoodi/foveance --upstream https://api.openai.com/v1
```

### Run with Python / CLI

```bash
foveance proxy --upstream https://api.openai.com/v1      # OpenAI
foveance proxy --upstream https://api.anthropic.com/v1   # Anthropic / Claude
foveance proxy --upstream http://localhost:11434/v1      # Ollama (local), vLLM, TGI, LM Studio
```
```bash
# then point any client at it with one variable (your API key still goes straight upstream):
export OPENAI_BASE_URL=http://localhost:8799/v1          # OpenAI SDK, Codex, Ollama-backed apps
export ANTHROPIC_BASE_URL=http://localhost:8799          # Anthropic SDK, Claude Code
```

> **Works with anything that lets you set its base URL** — the OpenAI and Anthropic SDKs,
> **Claude Code**, **Codex** (with an API key), aider, Continue, Cursor, LangChain, LiteLLM,
> and local runtimes like **Ollama / vLLM / LM Studio**. Foveance is **auth-free**: it adds no
> login of its own and stores no key. The only thing it can't intercept is a client that
> cryptographically hard-pins its endpoint (e.g. ChatGPT-**subscription** Codex); give such a
> tool an API key and it works like everything else.

It listens on `http://localhost:8799` and exposes `POST /v1/chat/completions`,
`POST /v1/messages`, `POST /v1/responses`, `GET /v1/models`, `GET /health`, `GET /admin/stats`
(JSON), and a **live dashboard** at `GET /` (tokens saved and ≈$ at `--price-per-mtok`).
`"stream": true` is passed through verbatim. Plain chat is compressed by the anticipatory
allocator; tool-using (agentic) requests are compressed *structurally* in place, preserving
every message and tool-call pairing.

**Prompt-cache aware:** blocks carrying an Anthropic `cache_control` breakpoint are never
modified, and with `--cache-aware` the proxy never touches anything at or before the last
breakpoint — so it never invalidates the provider's prompt cache. See
[`docs/limitations.md`](docs/limitations.md) for the cost arithmetic.

<p align="center"><img alt="foveance works with Claude Code, Codex, Ollama, and any OpenAI/Anthropic-compatible tool" src="https://raw.githubusercontent.com/aimaghsoodi/foveance/main/assets/demo_anytool.gif" width="640"></p>

| Client / agent | How to route it through Foveance |
|---|---|
| **OpenAI SDK** (Python/JS) | `base_url="http://localhost:8799/v1"` (or `OPENAI_BASE_URL`) |
| **Anthropic SDK** / **Claude Code** | `ANTHROPIC_BASE_URL=http://localhost:8799` |
| **Ollama** | `foveance proxy --upstream http://localhost:11434/v1`; point your app at `:8799/v1` |
| **OpenAI Codex CLI** | API-key custom provider in `~/.codex/config.toml`: `base_url="http://localhost:8799/v1"`, `wire_api="responses"` (subscription Codex can't be proxied — use an API key) |
| **Cursor / Continue / Antigravity** | set the custom OpenAI base URL to `http://localhost:8799/v1` |
| **aider / opencode / Crush** | set the OpenAI-compatible base URL to `http://localhost:8799/v1` |
| **LangChain / LlamaIndex / LiteLLM** | pass `base_url=`/`api_base="http://localhost:8799/v1"` |
| **Node / npm tools** | `npx foveance-proxy --upstream https://api.openai.com/v1` |

### Measured real-world results
| Setting | Tokens | Outcome |
|---|---|---|
| **`foveance wrap`** (live) — llama3.2:1b via Ollama, buried-fact recall | **2,127 → 186 est. tokens (−91%)** | fact recalled **correctly** through the compressed context |
| **Local model** — llama3.2:1b, long chat with a buried fact | **3,590 → 1,677 tokens (−53%)** | Foveance **correct**; full replay **hallucinated** the value |
| **Claude Code** (live, Anthropic OAuth) — agentic in-place compression | ~71% fewer tokens on an 8-tool-call transcript | works end-to-end, tool pairing preserved |
| **Benchmark** — Gemma/Llama/Qwen, 5 seeds | 62–64% fewer at iso-accuracy | matches full-replay accuracy |

## Head-to-head vs other methods (real model + real LLMLingua-2)
A long trajectory hides one load-bearing fact early amid filler; each method compresses to a
budget, then the real model (llama3.2:1b) is asked to recall it. Only the query-aware allocators
recall it at every budget, at **5–10× fewer tokens than full replay**:

| recall @ budget | full | keep-recent | truncate | spread-evenly | **LLMLingua-2** | reactive (AFM) | **Foveance** |
|---|---|---|---|---|---|---|---|
| **200** (tight) | 1.00 | 0.00 | 0.00 | 0.00 | **0.00** | 1.00 | **1.00** |
| 300 | 1.00 | 0.00 | 1.00 | 1.00 | **0.00** | 1.00 | **1.00** |
| 500 | 1.00 | 0.00 | 0.00 | 1.00 | **0.33** | 1.00 | **1.00** |

![baseline comparison](https://raw.githubusercontent.com/aimaghsoodi/foveance/main/assets/baseline_comparison.png)

The same ordering holds in the full multi-turn agent loop, ruling out a one-shot artifact:

![full agent-loop comparison](https://raw.githubusercontent.com/aimaghsoodi/foveance/main/assets/baseline_trajectory.png)

Reproduce: `python bench/compare_baselines.py --with-llmlingua && python bench/plot_baselines.py`.
LLMLingua-2 is a real run via the `llmlingua` package (CPU).

## Library usage (beyond `shrink`)
```python
from foveance import Controller, Item
from foveance.llm import MockLLM   # or OllamaLLM("gemma2:9b"), OpenAICompatLLM(...)

ctrl = Controller(MockLLM(), budget=2000, policy="foveance", drift=0.7)
ctrl.add_item(Item("obs0", "tool_output", "FACT api_key=sk-123\n...lots of logs...", created_turn=0))
rec = ctrl.step("recall api_key", turn=0)
print(rec.answer, rec.input_tokens, rec.peak_tokens)
```
Swap `policy="reactive_afm"` (the AFM baseline), `"recency"`, `"full"`, or `"oracle"` to compare.

**The public API at a glance** (`from foveance import ...`):

| Name | What it is |
|---|---|
| `shrink(messages, budget=2000)` | the one-liner — compress a messages list, no setup |
| `Controller`, `Item` | the full stepping loop (add items, `step(query, turn)`) |
| `index_allocate`, `dp_allocate`, `lp_bound` | the index policy, exact DP optimum, and LP bound (`index ≤ OPT ≤ LP`) |
| `AnticipatoryPredictor`, `PredictorConfig` | the anticipatory future-relevance scorer (`drift` knob) |
| `MultiFidelityStore`, `Fidelity` | the reversible multi-fidelity store |
| `HashingEmbedder`, `cosine` | the offline embedder + similarity |
| `baselines`, `metrics` | policy arms (`full`/`recency`/`reactive_afm`/`oracle`/…) and scoring helpers |
| `foveance.proxy.FoveanceProxy` | the proxy core, if you want to embed it |

## Honest positioning
As of mid-2026 this space is crowded. **Per-message multi-fidelity tiering under a token budget
already exists** — see AFM (Cruz 2025), ContextBudget, ACON, MemAct. That mechanism is
*substrate, not the contribution here.* Foveance ships a faithful AFM-style reactive policy as a
first-class baseline — it is literally the `drift = 0` special case of the predictor. The
defensible novelty is narrow and specific:

1. an **anticipatory** allocation criterion (expected *future* relevance) — the reactive
   AFM-style criterion is the `drift = 0` special case;
2. a **fundamental-limits theory** for the black-box, multi-turn, *task-success* setting;
3. a **near-optimal index policy** with a measured greedy gap, plus a theorem for **when**
   anticipation beats the reactive heuristics everyone ships;
4. **successive-refinability** conditions making reversible re-inflation "free";
5. an **open benchmark** placing all methods on one accuracy–token frontier vs the bound.

The deployable index allocator stays within ~1.8% of the exact DP optimum and below the LP
bound (`index ≤ OPT ≤ LP`). Full claim boundaries and the prior-art table are in
[`docs/NOVELTY.md`](docs/NOVELTY.md).

## What's in the package
```
src/foveance/   store.py · predictor.py (anticipatory future-relevance) · allocator.py
              (index + exact DP + LP bound) · controller.py · compressors.py · embedders.py ·
              baselines.py · metrics.py · learned.py · proxy.py · cli.py · llm.py
tests/        store/predictor/allocator/controller (100% covered) + integration
bench/        run_bench.py · analyze.py · plots.py · report.md · results/ (real CSVs)
docs/         architecture.md · theory.md · baselines.md · limitations.md · NOVELTY.md
```

## Reproduce the benchmark
```bash
bash scripts/run_everything.sh       # real models via Ollama (installs + pulls + runs + plots)
bash scripts/run_offline_demo.sh     # no GPU: identical chain with a deterministic mock model
```
Outputs land in `bench/report.md`, `bench/results/`, and `bench/plots/`. No number is
hand-entered; every figure traces to a CSV.

## License
Apache-2.0. See [`LICENSE`](LICENSE).
