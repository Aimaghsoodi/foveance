# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.2] - 2026-07-16 (branding)
Documentation and packaging metadata only; no code change. The social card ("Up to 82% fewer
tokens. Losslessly.") is now the hero banner across the PyPI README, the npm README, and the docs
home, and the docs carry Open Graph / Twitter `og:image` meta so shared links preview the card.
Remaining "60%+" copy in the npm README and docs home was corrected to the lossless-codec story.

## [0.5.1] - 2026-07-16 (positioning)
Documentation and packaging metadata only; no code behaviour change. Foveance is repositioned around
its headline feature: it is now described everywhere as **the lossless token-optimization codec**
(up to 82% fewer input tokens, nothing dropped), with the anticipatory allocator presented as the
optional lossy-but-recoverable layer on top. README rebuilt to lead with `compress()` and the
codec benchmark (scorecard + 8-framework head-to-head), PyPI/npm summaries updated, and the social
card / share square / Twitter header re-cut to "Up to 82% fewer tokens. Losslessly."

## [0.5.0] - 2026-07-16 (the codec-everywhere release)

The first release since 0.2.0, and it ships everything from the 0.3 and 0.4 lines as well. Those two
versions were developed and are documented in full below, but were never published: 0.3.0 predates
its own vault handle-leak fix, and both predate the codec work that supersedes them. Rather than
put superseded builds on PyPI permanently, 0.5.0 folds them in. If you are coming from 0.2.0 you get
the 0.3 interactive-compression work, the 0.4 lossless codec, and the 0.5 codec-everywhere work in
one step.

### Added
- **Shared-prefix template pass** (`RedundancyCodec(template=True)`, `compress(..., template=True)`):
  factors the prefix shared by a run of near-identical lines out once
  (`[fov:tpl 20 "src/service/module_"]` + the suffixes), recovering the intra-line redundancy that
  line-level dedup structurally cannot see. Exactly invertible via the new
  `foveance.expand_templates()`, never inflates, and takes the head-to-head from 75.4% to **82.4%**
  saved — past LLMLingua-2 matched (79.8%) while remaining lossless with 100% of facts preserved.
  **Opt-in by design**: it measured 0.90 vs 0.95 mean accuracy at 14% fewer tokens on the five-model
  benchmark (one task in twenty, within noise, but not measurably free), and the default must stay
  safe to enable unconditionally.
- **RULER public benchmark arm** (`bench/fetch_ruler.py`, `bench/codec_ruler.py`): 480 real examples
  across all task families at 4k/8k/16k. Serves as a negative control — RULER plants distinct
  needles, so the codec correctly saves ~0% on `qa`/`niah_multikey` and never inflates, while
  removing 91.4% on the one family that genuinely repeats. All 480 round-trip exactly.
- `.foveance.toml` now also carries the `codec` and `agentic_codec` switches, so the codec can be
  turned on once per machine/project instead of per invocation.
### Fixed
- `foveance --version` / `-V` now print the version. Previously only the `foveance version`
  subcommand worked and the flags silently printed help and exited 0, which reads like a broken
  install. Regression-tested.
- The test suite no longer hard-fails without the optional ML extras. Four tests imported `numpy`
  (via `foveance.learned`) without a guard, so `pip install "foveance[dev]" && pytest` reported
  four `ModuleNotFoundError` failures on an otherwise healthy install. They now
  `pytest.importorskip("numpy")` and skip cleanly, matching how the integration tests already
  handle optional dependencies.
### Changed (packaging)
- The source distribution no longer ships `assets/` (README images are served from GitHub by
  absolute URL) or the manuscript's vector PDFs, halving it from 4.4 MB to 2.2 MB. The benchmark
  figures (`bench/plots/*.png`) and every results CSV are still included — they are the evidence
  behind the README's numbers. The installed wheel is unchanged at ~82 KB.
- **Lossless codec on the agentic (tool-use) paths** (`foveance proxy --agentic-codec` /
  `FOVEANCE_AGENTIC_CODEC=1`): on tool-using requests the codec now runs *across* the old tool
  payloads in place, collapsing cross-message repeats (re-listed dirs, retried stack traces,
  boilerplate) to legible pointers with the first occurrence kept verbatim — instead of lossily
  digesting each payload. Only free-text payload strings are rewritten; message count/order, roles,
  `tool_use`↔`tool_result` ids, `cache_control` blocks, and the last `agentic_protect_last` turns
  stay byte-identical, so the provider still validates the request and the prompt cache is never
  invalidated. Wired through all three dialects (Anthropic Messages, OpenAI Chat, OpenAI Responses)
  and covered by a full tool-pairing safety matrix. `agentic_codec` surfaced in `/admin/stats`.
- **New agent adapters**: `foveance wrap`/`env` now know Cursor, Windsurf, Roo Code, Zed, and the
  Gemini CLI (15 total), each with an honest note on GUI-vs-env configuration.
- **Lossless codec in the proxy** (`foveance proxy --codec` / `FOVEANCE_CODEC=1`): the 0.4
  cross-item redundancy codec now runs on the assembled plain-chat context, reference-encoding
  repeated line-runs across items **losslessly** (the first occurrence stays verbatim, so no fact
  is dropped). Off by default; a `codec_saved_tokens` counter reports the additional lossless
  saving. Zero accuracy risk, so it is safe to layer on top of any budget/policy.
### Changed
- **Codec default `min_run=1`** (was 2): because every reference is already gated by the token
  guard, dereferencing a single repeated line only when the pointer is strictly cheaper is
  cost-optimal — measured 76.1% vs 75.6% saved on the redundancy suite, never inflates, losslessness
  unchanged. Applied to `compress`/`compress_anthropic`/proxy.
- **Vault storage codec** now picks the strongest installed backend — **Brotli (~50×) or Zstandard
  (~44×)** when available, falling back to stdlib zlib — via a self-describing 3-byte header.
  Backward-compatible: legacy headerless-zlib blobs still decode.
- **`foveance.compress_anthropic(system, messages)`** — the Anthropic-shaped lossless codec
  one-liner, returning `(new_system, new_messages, report)` with the `system` string participating
  in the cross-message dedup.
- **Entropy-coded vault storage** (`ItemVault(compress=True)`, default on): stored full texts are
  now zlib-compressed (the transport-codec storage saving — up to ~50–100× on redundant content),
  realising the paper's storage-side result. Reads transparently handle both compressed and legacy
  plaintext rows, so existing vaults keep working (a `blob` column is migrated in automatically).

## [0.4.0] - developed 2026-07-14, released as part of 0.5.0 (the codec release)
Adds the lossless cross-item codec as the headline feature, on top of the 0.3 interactive-
compression work. Never published on its own; superseded by and included in 0.5.0.
### Added
- **Cross-item redundancy codec** (`foveance.codec.RedundancyCodec`, `foveance.compress`,
  `foveance compress FILE`): an LZ-family dictionary coder specialised to the line granularity of
  tool transcripts. It removes redundancy that lives *across* items (re-printed listings, retried
  stack traces, repeated boilerplate envelopes) which per-item compressors cannot see, and it is
  **exactly reversible** — `unpack(pack(items)) == items` byte-for-byte, verified by a 200-example
  property test. Measured **2.3× (56% saved), lossless** on representative redundant agent traffic;
  **3.10× composed with digestion**; correctly ~0% on low-redundancy traces (it never inflates —
  a reference is emitted only when it provably costs fewer tokens than the run it replaces).
- **`CompressionReport`** with a first-class, measured `ratio` / `saved_pct` / `factor`, so
  Foveance reports a real compression ratio the way a codec should.
- **`bench/codec_bench.py`** + `bench/traces/make_redundant_trace.py`: measure the codec and the
  composed allocator+codec ratio on real traces; writes `codec_ratio.csv` and
  `codec_fullstack.csv`. The full-stack sweep documents the honest high-ratio operating point
  (up to ~90%+ token reduction on long stale-heavy context, **loss-free via `foveance_expand`**,
  not for free) alongside the guaranteed-lossless 2.3× floor. See `docs/compression.md`.

## [0.3.0] - developed 2026-07-12, released as part of 0.5.0 (the interactive-compression release)
Never published on its own; superseded by and included in 0.5.0.
### Added
- **Anticipatory agentic compression** (`--agentic-allocator`): old tool-transcript payloads get
  graded fidelities (full/digest/gist/pointer) from the same anticipatory allocator plain chat
  uses, instead of blind digestion. Structure (tool pairing, cache breakpoints, protected recent
  turns) is preserved by construction.
- **Model-driven re-inflation** (`--expand-tool`): compressed items carry addressable markers and
  the model can retrieve any of them via a `foveance_expand` tool the proxy resolves
  transparently against a durable vault -- compression becomes lossless in effect
  (non-streaming requests; bounded loop).
- **Durable item vault** (`~/.foveance/vault.db`): full texts survive restarts, so
  "nothing is deleted forever" now holds across sessions.
- **`foveance audit LOGFILE`**: replay your own conversation logs offline and get a
  tokens/$-saved report with monthly extrapolation. Nothing leaves the machine.
- **The learning loop** (`--learn` + `foveance train`): local trace logging of which items each
  query referenced; training calibrates the future-relevance model on YOUR workload, and the
  proxy uses it automatically.
- **Salience-aware digestion**: digests keep query-relevant lines instead of blind head/tail.
- **Agent-CLI adapters** (`foveance.adapters`, `foveance adapters`, `foveance env`): a registry of
  popular agent CLIs/SDKs (Claude Code, Codex, Aider, Cline, opencode, Goose, Continue, LiteLLM,
  and the raw OpenAI/Anthropic SDKs) with the correct API dialect and base-URL env vars for each.
  `foveance wrap <tool>` now uses it to set exactly the vars that tool reads (Anthropic vars point
  at the proxy root, OpenAI vars at root/v1); `foveance adapters` lists them and `foveance env
  <tool>` prints the exports (bash + PowerShell) for pointing a tool at a long-running proxy.
- **Replay benchmark** (`bench/replay_bench.py`): raw vs digest vs allocator on recorded traces.
- **Multi-model accuracy benchmark** (`bench/paper2_bench.py` + `bench/models.py`): buried-fact
  recovery across a roster of 8 models via OpenRouter (one OpenAI-compatible endpoint), scoring
  the four arms raw / digest / allocator / allocator+expand -- the last running the real
  `foveance_expand` re-inflation loop for tool-capable models. A shared `CostAccountant`
  (`foveance.llm`) with a hard `--budget-usd` cap keeps a full sweep inside a fixed dollar budget;
  `OpenRouterLLM` records the provider's exact per-call cost. The whole pipeline (cost accounting,
  budget guard, expand loop, arm-wise accuracy) is validated offline against an in-process mock,
  so it is provably ready before spending a cent.
- `--admin-token` auth for /admin endpoints; stats now report evictions/expansions/vault size.

### Fixed
- Long-running proxies no longer grow without bound: conversation state is LRU+TTL evicted
  (`max_convs`, `conv_ttl_s`), with full texts still recoverable via the vault.
- **Vault file-handle leak**: `ItemVault` now closes every SQLite connection (via
  `contextlib.closing`). Previously `with sqlite3.connect(...)` committed but never closed the
  handle, leaking one per operation and, on Windows, blocking the `.db` file from being deleted
  (broke the replay benchmark under a temp dir). Regression test added.

## [0.2.0] - 2026-07-12
### Added
- **Foveance Pro** (optional, offline-verified): a Pro license unlocks *persistent* savings
  accounting — the proxy's token/$ totals survive restarts (SQLite in `~/.foveance/`), the
  dashboard shows all-time and per-day history, and `/admin/export.csv` exports it. The
  open-source package remains fully functional without a license.
  - `foveance license activate <key>` / `status` / `deactivate`. Keys are RSA-2048 signatures
    over a small JSON payload, verified in pure stdlib against a bundled public modulus — no
    network call, no phoning home, no new dependencies. The gate is a courtesy to honest users,
    not DRM.
  - `FoveanceProxy(savings_log=...)` persists per-request savings; `/admin/export.csv` returns
    HTTP 402 with an activation hint when no license is active.
- `foveance.integrations.llamaindex` (a `shrink_chat_messages` wrapper), completing the framework
  integration matrix started in 0.1.3 (#5).
- `--token-encoding` / `FOVEANCE_TOKEN_ENCODING` for `foveance proxy`/`wrap`: picks the tiktoken
  encoding used by `--exact-tokens` (default stays `cl100k_base` for back-compat; pass
  `o200k_base` for gpt-4o and newer so exact-token accounting matches the model you're actually
  routing to).

## [0.1.3] - 2026-07-07
### Added
- `foveance.shrink_anthropic(system, messages, budget)` — the Anthropic-shaped one-liner,
  returning `(new_system, new_messages)` (#9).
- Config file support: `~/.foveance.toml` and `./.foveance.toml` set defaults for budget, drift,
  policy, upstream, and agentic-protect-last, with precedence flag > env > cwd config > home
  config > default (#2, #12).
- Exact token counting: `--exact-tokens` uses a real tokenizer (tiktoken, if installed) instead of
  the chars/4 heuristic for accounting and the dashboard; the counter is pluggable via
  `FoveanceProxy(token_counter=...)` (#7, #12).
- Framework integrations behind extras: `foveance.integrations.litellm` (a `shrink_completion`
  wrapper) and `foveance.integrations.langchain` (a `shrink_messages` Runnable) (#4, #5, #13).
- Docker image + a GHCR publish workflow, so the proxy runs with `docker run` and no Python setup
  (#8, #10, contributed by @Hayathorium).

### Fixed
- `LogisticFutureRelevance.fit()` now raises a clear ImportError when numpy is missing rather than
  failing obscurely (#14).
- Cross-platform config-file test now mocks the home directory on Windows (`USERPROFILE`) as well
  as POSIX (`HOME`).

## [0.1.2] - 2026-07-04
### Changed
- `pip install foveance` now includes everything a normal user needs — the `shrink()` one-liner,
  `foveance wrap`/`proxy`, and the demo. The proxy web-server packages moved from the `[proxy]`
  extra into the base dependencies; `[proxy]` is kept as a no-op alias for back-compat. Only the
  ML embedder and benchmark tooling remain behind `[all]`/`[ml]`/`[bench]`.
- README/docs simplified: install is just `pip install foveance` everywhere, and the full public
  API (including `shrink`) is listed in one table.
- `CITATION.cff` trimmed to a software-only citation (dropped the premature reference to the
  unpublished manuscript).

## [0.1.1] - 2026-07-04
### Added
- `foveance.shrink(messages, budget=2000)` — the dead-simple one-liner: compress an
  OpenAI-style messages list from Python with no proxy, no server, no config. Works on the
  plain `pip install foveance` (no extras).

### Changed
- README rewritten to lead with the plain-English pitch and the two 30-second paths
  (`foveance wrap` and `shrink`), with the proxy/theory details moved below. Logo and figures
  switched to PNG with a `<picture>` fallback so they render on PyPI and npm.

### Fixed
- CI type-check step: dropped the `mypy` `python_version` pin (newer `numpy` stubs use PEP 695
  syntax that the pinned parser rejected) and untangled a variable-shadowing type error in the
  pure-Python bootstrap fallback.

## [0.1.0] - 2026-06-22
### Added
- `foveance wrap` — run any CLI/agent through the proxy with one command: starts the proxy,
  sets `ANTHROPIC_BASE_URL`/`OPENAI_BASE_URL` for the child process only, launches the tool,
  and prints a tokens-saved summary (with a ≈$ estimate at `--price-per-mtok`) on exit.
- Live dashboard at `GET /` and `/admin`: running tokens-saved counter, %, and $-equivalent,
  polling `/admin/stats` (which now reports `est_tokens_before/after/saved`, `est_saved_pct`,
  `est_usd_saved`, and `compressed_requests`).
- Prompt-cache-aware compression: blocks carrying Anthropic `cache_control` are never modified,
  and `--cache-aware` additionally freezes everything at or before the last breakpoint so the
  provider's prompt cache is never invalidated (cost arithmetic in `docs/limitations.md`).
- Structure-preserving in-place compression for agentic requests (Anthropic tool_use/tool_result,
  OpenAI tool_calls, and the OpenAI Responses API used by Codex), protecting the most recent
  `--agentic-protect-last` turns; verified live with Claude Code.
- Additional baseline arms `truncate` and `uniform` in the package and benchmark; single-shot
  head-to-head probe (`bench/compare_baselines.py`) including real LLMLingua-2.
- Anticipatory predictor with a forward-drifting future-query posterior; `drift=0` recovers the
  reactive (AFM) criterion.
- Multi-fidelity reversible store (POINTER/GIST/DIGEST/FULL) with content-hash-cached renders.
- Index allocator (Lagrangian/Whittle greedy on concave envelopes), exact DP oracle, and LP
  bound, giving the `index <= OPT <= LP` sandwich.
- Compressors (heuristic + LLM), embedders (hashing/sentence-transformers/API), metrics, a
  learned logistic future-relevance predictor, and an OpenAI-compatible reverse proxy.
- Baselines as first-class policy arms: `full`, `recency`, `reactive_afm`, `oracle`, optional
  `llmlingua2`; a drift-twin audit proves `reactive_afm` and `foveance` differ only in drift.
- Benchmark harness (suites, budget sweep, bootstrap CIs, paired Wilcoxon, greedy-gap, drift /
  predictor / retrieve / fidelity-cost ablations) with synthetic and LongBench/RULER/AppWorld/
  OfficeBench adapters that skip gracefully when data is absent.
- Theory summary (`docs/theory.md`) backed by five theorems with full proofs in the accompanying manuscript.
- CLI (`foveance demo|proxy|bench|version`), examples, docs, and CI across Python 3.10-3.13.

### Performance
- `index_allocate` reimplemented with a binary heap, attaining the documented
  `O(sum_i L_i log sum_i L_i)` time (previously re-sorted the queue each iteration); allocates
  4000 items in tens of milliseconds. A reproducible overhead benchmark is in `bench/overhead.py`.

### Fixed
- Proxy `/v1/chat/completions` route returned HTTP 422 because a stringized `Request`
  annotation (from `from __future__ import annotations`) was not recognized by FastAPI; the
  route now takes the JSON body as a dict and is covered by a real end-to-end HTTP test.

### Testing & tooling
- Property-based tests (Hypothesis) for the allocator invariants: budget feasibility,
  monotonicity, the one-item greedy bound, and the `IDX <= OPT <= LP` sandwich.
- End-to-end proxy integration test over real HTTP (threaded upstream + FastAPI TestClient).
- `py.typed` marker (PEP 561), `mkdocs.yml`, `Dockerfile`/`.dockerignore`, `.pre-commit-config.yaml`,
  `CITATION.cff`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, PR template, and a PyPI release workflow.

### Notes
- Real-model results (Gemma/Qwen/Llama via Ollama) show budgeted policies reaching full-replay
  accuracy at roughly 61-62% fewer tokens; see `bench/report.md`. No numbers are hand-entered.
