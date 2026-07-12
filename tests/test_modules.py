"""Tests for embedders, compressors, baselines, metrics, learned, proxy, and cli."""
import json
import sys

import pytest

from foveance.store import MultiFidelityStore, Item, Fidelity
from foveance.embedders import HashingEmbedder, cosine, as_embed_fn
from foveance.compressors import HeuristicCompressor, LLMCompressor, make_renderer
from foveance import baselines, metrics
from foveance.predictor import AnticipatoryPredictor, PredictorConfig
from foveance.llm import Completion, LLM


# ----------------------------------------------------------------------------- embedders
def test_hashing_embedder_deterministic_cross_instance():
    a = HashingEmbedder().embed("the quick brown fox")
    b = HashingEmbedder().embed("the quick brown fox")
    assert a == b
    assert abs(cosine(a, a) - 1.0) < 1e-9


def test_as_embed_fn_accepts_callable_and_object():
    fn = as_embed_fn(lambda s: [1.0, 0.0])
    assert fn("x") == [1.0, 0.0]
    assert callable(as_embed_fn(HashingEmbedder()))
    with pytest.raises(TypeError):
        as_embed_fn(123)


# ---------------------------------------------------------------------------- compressors
def test_heuristic_compressor_matches_default():
    it = Item("i", "tool_output", "FACT a=1\n" + "noise\n" * 20, 0)
    hc = HeuristicCompressor()
    assert "FACT a=1" in hc(it, Fidelity.DIGEST)


class _EchoLLM(LLM):
    def generate(self, prompt, query):
        # a faithful "summary": echo the first salient token
        return Completion("GIST: " + prompt.split("\n")[-1][:40], 1, 1, 0.0)


def test_llm_compressor_caches_and_falls_back():
    it = Item("i", "tool_output", "FACT a=1\nbody body body", 0)
    comp = LLMCompressor(_EchoLLM())
    first = comp(it, Fidelity.GIST)
    assert comp(it, Fidelity.GIST) == first  # cached
    # FULL is not in level_prompts -> falls back to heuristic verbatim
    assert comp(it, Fidelity.FULL) == it.full_text


def test_llm_compressor_handles_model_failure():
    class _BoomLLM(LLM):
        def generate(self, prompt, query):
            raise RuntimeError("upstream down")
    comp = LLMCompressor(_BoomLLM())
    it = Item("i", "tool_output", "FACT a=1\n" + "x\n" * 10, 0)
    out = comp(it, Fidelity.DIGEST)  # must not raise; heuristic fallback
    assert "FACT a=1" in out


def test_llm_compressor_callable_model_and_empty_render():
    it = Item("i", "tool_output", "FACT a=1\nbody", 0)
    comp = LLMCompressor(lambda prompt: "")  # empty -> fallback
    assert comp(it, Fidelity.GIST)  # non-empty (fallback used)


def test_make_renderer_offline_and_online():
    assert isinstance(make_renderer(None), HeuristicCompressor)
    assert isinstance(make_renderer(_EchoLLM()), LLMCompressor)


# ------------------------------------------------------------------------------ baselines
def test_reactive_afm_runs_as_policy():
    store = MultiFidelityStore()
    for t in range(4):
        store.add(Item(f"i{t}", "tool_output", f"FACT k{t}=v{t}\n" + "n\n" * 15, t))
    pred = AnticipatoryPredictor(store, config=PredictorConfig(drift=0.0))
    pred.observe_query("recall k0")
    levels = baselines.reactive_afm(store, pred, budget=800, turn=3)
    assert set(levels) == set(store.order)
    assert all(isinstance(v, Fidelity) for v in levels.values())


def test_reactive_and_foveance_share_code_path():
    # The only intended difference is the predictor's drift (GATE 2 invariant).
    store = MultiFidelityStore()
    for t in range(5):
        store.add(Item(f"i{t}", "tool_output", f"FACT k{t}=v{t}\n" + "n\n" * 12, t))
    pred = AnticipatoryPredictor(store, config=PredictorConfig(drift=0.0))
    pred.observe_query("recall k0")
    assert baselines.reactive_afm(store, pred, 700, 4) == baselines.foveance(store, pred, 700, 4)


def test_full_recency_noop_oracle_lp():
    store = MultiFidelityStore()
    for t in range(4):
        store.add(Item(f"i{t}", "tool_output", "x" * 100, t))
    pred = AnticipatoryPredictor(store)
    pred.observe_query("q")
    assert all(v == Fidelity.FULL for v in baselines.full(store, pred, 100, 3).values())
    assert all(v == Fidelity.FULL for v in baselines.noop(store, pred, 100, 3).values())
    rec = baselines.recency(store, pred, 100, 3, k=1)
    assert rec["i3"] == Fidelity.FULL and rec["i0"] == Fidelity.POINTER
    assert isinstance(baselines.oracle(store, pred, 400, 3), dict)
    assert baselines.lp_value(store, pred, 400, 3) >= 0.0


def test_baselines_empty_store():
    store = MultiFidelityStore()
    pred = AnticipatoryPredictor(store)
    assert baselines.reactive_afm(store, pred, 100, 0) == {}
    assert baselines.oracle(store, pred, 100, 0) == {}
    assert baselines.lp_value(store, pred, 100, 0) == 0.0


def test_llmlingua_skips_when_absent():
    if not baselines.llmlingua2_available():
        store = MultiFidelityStore()
        pred = AnticipatoryPredictor(store)
        with pytest.raises(RuntimeError):
            baselines.llmlingua2(store, pred, 100, 0)


# -------------------------------------------------------------------------------- metrics
def test_token_counter_and_cost():
    count = metrics.make_token_counter()
    assert count("hello world this is a test") > 0
    assert metrics.whitespace_counter("a" * 8) == 2
    cm = metrics.CostModel(input_per_m=1.0, output_per_m=2.0)
    assert cm.cost(1_000_000, 1_000_000) == pytest.approx(3.0)


def test_tokens_per_correct_and_bootstrap():
    assert metrics.tokens_per_correct(100, 4) == 25.0
    assert metrics.tokens_per_correct(100, 0) == float("inf")
    m, lo, hi = metrics.bootstrap_ci([1.0, 1.0, 1.0, 1.0])
    assert lo <= m <= hi
    assert metrics.bootstrap_ci([])[0] != metrics.bootstrap_ci([])[0]  # nan
    ls = metrics.latency_stats([0.1, 0.2, 0.3])
    assert ls.ms_per_turn == pytest.approx(200.0)


# -------------------------------------------------------------------------------- learned
def test_learned_predictor_fit_predict_save_load(tmp_path):
    from foveance.learned import LogisticFutureRelevance
    # Build a trivially separable trace: item referenced soon -> positive.
    items = [Item(f"i{j}", "tool_output", f"FACT k{j}=v{j}", 0) for j in range(6)]
    queries = ["recall k0", "recall k1", "recall k2", "recall k3"]
    referenced = {1: {"i1"}, 2: {"i2"}, 3: {"i3"}}
    model = LogisticFutureRelevance()
    model.fit_traces([{"items": items, "queries": queries, "referenced": referenced}], horizon=3)
    from foveance.predictor import PredictorContext
    ctx = PredictorContext(turn=0, future_posterior=HashingEmbedder().embed("recall k1"),
                           query_history=[HashingEmbedder().embed("recall k1")],
                           cfg=PredictorConfig())
    p = model.predict(items[1], ctx)
    assert 0.0 <= p <= 1.0
    path = tmp_path / "m.json"
    model.save(str(path))
    loaded = LogisticFutureRelevance.load(str(path))
    assert loaded.weights == model.weights


def test_learned_fit_empty_is_noop():
    from foveance.learned import LogisticFutureRelevance
    m = LogisticFutureRelevance()
    before = list(m.weights)
    m.fit([], [])
    assert m.weights == before


def test_learned_fit_without_numpy_raises_helpful_error(monkeypatch):
    from foveance.learned import LogisticFutureRelevance
    monkeypatch.setitem(sys.modules, "numpy", None)
    m = LogisticFutureRelevance()
    with pytest.raises(ImportError, match=r'foveance\[ml\]'):
        m.fit([[1.0]], [1.0])


# ---------------------------------------------------------------------------------- proxy
def test_proxy_transparently_compresses_request():
    from foveance.proxy import FoveanceProxy
    captured = {}

    def echo_upstream(req):
        captured["messages"] = req["messages"]
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    px = FoveanceProxy(budget=120)
    long_ctx = "FACT secret=42\n" + "\n".join(f"log line {i} status=ok" for i in range(200))
    request = {
        "model": "gpt-x",
        "user": "conv-1",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": long_ctx},
            {"role": "assistant", "content": "noted"},
            {"role": "user", "content": "recall secret"},
        ],
    }
    resp = px.handle(request, echo_upstream)
    # Upstream saw a rewritten, shorter message list; last user query preserved verbatim.
    fwd = captured["messages"]
    assert fwd[-1]["content"] == "recall secret"
    forwarded_text = json.dumps(fwd)
    assert len(forwarded_text) < len(long_ctx)            # actually compressed
    assert resp["foveance"]["compressed"] is True
    assert px.stats()["requests"] == 1
    assert px.stats()["conversations"] == 1


def test_proxy_empty_conversation_passthrough():
    from foveance.proxy import FoveanceProxy
    px = FoveanceProxy()
    msgs, stats = px.transform([{"role": "system", "content": "s"}], "c")
    assert stats["compressed"] is False


def test_proxy_extract_text_handles_blocks():
    from foveance.proxy import _extract_text
    assert _extract_text("plain") == "plain"
    assert _extract_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"
    assert _extract_text(None) == ""


def test_proxy_stats_use_chars_heuristic_without_token_counter():
    from foveance.proxy import FoveanceProxy

    def echo_upstream(req):
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    px = FoveanceProxy(budget=120)
    request = {"messages": [{"role": "user", "content": "recall secret"}]}
    resp = px.handle(request, echo_upstream)
    assert resp["foveance"]["est_tokens_exact"] is False
    assert px.stats()["est_tokens_exact"] is False


def test_proxy_stats_use_exact_counter_when_configured():
    from foveance.proxy import FoveanceProxy, _payload_text

    def counter(text):
        return len(text.split())

    def echo_upstream(req):
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    px = FoveanceProxy(budget=120, token_counter=counter)
    request = {
        "model": "gpt-x",
        "messages": [
            {"role": "user", "content": "FACT secret=42 " + "log line ok " * 50},
            {"role": "user", "content": "recall secret"},
        ],
    }
    resp = px.handle(request, echo_upstream)
    assert resp["foveance"]["est_tokens_exact"] is True
    assert resp["foveance"]["est_tokens_before"] == counter(_payload_text(request))
    assert px.stats()["est_tokens_exact"] is True


def test_proxy_anthropic_transform_folds_context_into_system():
    from foveance.proxy import FoveanceProxy
    px = FoveanceProxy(budget=120)
    big = "FACT secret=42\n" + "\n".join(f"log {i} ok" for i in range(200))
    system, messages, stats = px.transform_anthropic(
        system="You are helpful.",
        messages=[{"role": "user", "content": big},
                  {"role": "assistant", "content": "noted"},
                  {"role": "user", "content": [{"type": "text", "text": "recall secret"}]}],
        conv_id="c1")
    assert stats["compressed"] is True
    assert "You are helpful." in system          # original system preserved
    assert len(system) < len(big)                # context compressed into system
    assert messages[-1]["content"][0]["text"] == "recall secret"  # final turn kept verbatim


# ------------------------------------------------------------------------------------ cli
def test_cli_demo_runs(capsys):
    from foveance.cli import main
    rc = main(["demo", "--budgets", "800,1600", "--turns", "18"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "foveance" in out and "budget" in out


def test_cli_version(capsys):
    import foveance
    from foveance.cli import main
    assert main(["version"]) == 0
    assert foveance.__version__ in capsys.readouterr().out


def test_cli_no_command_prints_help(capsys):
    from foveance.cli import main
    assert main([]) == 0
    assert "usage" in capsys.readouterr().out.lower()


def _proxy_args(**overrides):
    import argparse
    base = dict(upstream=None, budget=None, drift=None, policy=None,
                agentic_protect_last=None, cache_aware=False, price_per_mtok=3.0,
                exact_tokens=False, token_encoding=None)
    base.update(overrides)
    return argparse.Namespace(**base)


requires_tomllib = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="tomllib is stdlib only on Python 3.11+")


def test_config_file_absent_falls_back_to_builtin_defaults(tmp_path, monkeypatch):
    from foveance.cli import _proxy_from_args
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FOVEANCE_BUDGET", raising=False)
    proxy, upstream = _proxy_from_args(_proxy_args())
    assert proxy.budget == 2000
    assert upstream == "http://localhost:11434/v1"


@requires_tomllib
def test_config_file_sets_defaults(tmp_path, monkeypatch):
    from foveance.cli import _proxy_from_args
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".foveance.toml").write_text(
        'budget = 555\ndrift = 0.25\npolicy = "recency"\nupstream = "http://x/v1"\n')
    for var in ("FOVEANCE_BUDGET", "FOVEANCE_DRIFT", "FOVEANCE_POLICY", "FOVEANCE_UPSTREAM"):
        monkeypatch.delenv(var, raising=False)
    proxy, upstream = _proxy_from_args(_proxy_args())
    assert (proxy.budget, proxy.drift, proxy.policy, upstream) == (555, 0.25, "recency", "http://x/v1")


@requires_tomllib
def test_env_var_overrides_config_file(tmp_path, monkeypatch):
    from foveance.cli import _proxy_from_args
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".foveance.toml").write_text("budget = 555\n")
    monkeypatch.setenv("FOVEANCE_BUDGET", "999")
    proxy, _ = _proxy_from_args(_proxy_args())
    assert proxy.budget == 999


@requires_tomllib
def test_cli_flag_overrides_env_and_config_file(tmp_path, monkeypatch):
    from foveance.cli import _proxy_from_args
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".foveance.toml").write_text("budget = 555\n")
    monkeypatch.setenv("FOVEANCE_BUDGET", "999")
    proxy, _ = _proxy_from_args(_proxy_args(budget=42))
    assert proxy.budget == 42


@requires_tomllib
def test_home_config_used_and_cwd_config_takes_precedence(tmp_path, monkeypatch):
    from foveance.cli import _proxy_from_args
    home_dir, cwd_dir = tmp_path / "home", tmp_path / "cwd"
    home_dir.mkdir()
    cwd_dir.mkdir()
    (home_dir / ".foveance.toml").write_text("budget = 111\ndrift = 0.11\n")
    (cwd_dir / ".foveance.toml").write_text("budget = 222\n")
    # os.path.expanduser("~") reads HOME on POSIX but USERPROFILE on Windows, so set both.
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("USERPROFILE", str(home_dir))
    monkeypatch.chdir(cwd_dir)
    monkeypatch.delenv("FOVEANCE_BUDGET", raising=False)
    monkeypatch.delenv("FOVEANCE_DRIFT", raising=False)
    proxy, _ = _proxy_from_args(_proxy_args())
    assert proxy.budget == 222   # cwd overrides home
    assert proxy.drift == 0.11   # home-only key still picked up


def test_cli_no_exact_tokens_flag_leaves_counter_unset():
    from foveance.cli import _proxy_from_args
    proxy, _ = _proxy_from_args(_proxy_args())
    assert proxy.token_counter is None


def test_cli_exact_tokens_flag_wires_a_token_counter():
    from foveance.cli import _proxy_from_args
    proxy, _ = _proxy_from_args(_proxy_args(exact_tokens=True))
    assert proxy.token_counter is not None
    assert proxy.token_counter("hello world") > 0


def test_cli_exact_tokens_defaults_to_cl100k_base(monkeypatch):
    import foveance.cli as cli
    import foveance.metrics as metrics_mod

    seen = {}
    real = metrics_mod.make_token_counter

    def spy(encoding="cl100k_base"):
        seen["encoding"] = encoding
        return real(encoding)

    monkeypatch.setattr(metrics_mod, "make_token_counter", spy)
    proxy, _ = cli._proxy_from_args(_proxy_args(exact_tokens=True))
    assert seen["encoding"] == "cl100k_base"
    assert proxy.token_counter is not None


def test_cli_token_encoding_flag_overrides_default(monkeypatch):
    import foveance.cli as cli
    import foveance.metrics as metrics_mod

    seen = {}
    real = metrics_mod.make_token_counter

    def spy(encoding="cl100k_base"):
        seen["encoding"] = encoding
        return real(encoding)

    monkeypatch.setattr(metrics_mod, "make_token_counter", spy)
    cli._proxy_from_args(_proxy_args(exact_tokens=True, token_encoding="o200k_base"))
    assert seen["encoding"] == "o200k_base"


def test_cli_token_encoding_env_var(monkeypatch):
    from foveance.cli import _proxy_from_args
    monkeypatch.setenv("FOVEANCE_TOKEN_ENCODING", "o200k_base")
    proxy, _ = _proxy_from_args(_proxy_args(exact_tokens=True))
    assert proxy.token_counter is not None
    assert proxy.token_counter("hello world") > 0
