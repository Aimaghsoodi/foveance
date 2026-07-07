# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
