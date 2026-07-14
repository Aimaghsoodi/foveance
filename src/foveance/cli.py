"""
Foveance command-line entry points:

    foveance demo                     offline Pareto demo (MockLLM, no GPU/network)
    foveance proxy --port 8799        start the OpenAI-compatible reverse proxy
    foveance wrap -- <command>        run any CLI/agent through the proxy, one command
    foveance bench [args...]          delegate to the benchmark harness (bench/run_bench.py)
    foveance version

``demo`` is fully self-contained so a fresh clone can show the headline behaviour --
budgeted policies match full-replay accuracy at a fraction of the tokens and dominate recency.
"""
from __future__ import annotations

import argparse
import random
import sys
from typing import Optional

from .store import Item
from .controller import Controller
from .llm import MockLLM
from . import __version__


# ----------------------------------------------------------------- compact demo task
def _demo_turns(seed: int = 0, n_turns: int = 36, n_facts: int = 10,
                block_lines: int = 30, drift: float = 0.7):
    """Tiny needle-reuse-with-drift trajectory (a self-contained slice of bench/tasks.py)."""
    rng = random.Random(seed)
    keys = [f"k{i}" for i in range(n_facts)]
    vals = {k: f"v{rng.randint(1000, 9999)}" for k in keys}
    turns = []
    plant = max(n_facts, n_turns // 3)

    def block(t, needle=""):
        lines = [f"log {t}.{j} status=ok latency={rng.randint(1, 400)}ms path=/srv/{rng.randint(1, 999)}"
                 for j in range(block_lines)]
        if needle:
            lines.insert(rng.randint(0, len(lines)), needle)
        return "\n".join(lines)

    for t in range(plant):
        k = keys[t % n_facts]
        turns.append(("note " + k, "",
                      Item(f"obs{t}", "tool_output", block(t, f"FACT {k}={vals[k]}"), t)))
    idx = 0
    for t in range(plant, n_turns):
        idx = (idx + rng.choice([0, 1, 1, 2])) % n_facts if rng.random() < drift else rng.randrange(n_facts)
        k = keys[idx]
        turns.append(("recall " + k, vals[k], Item(f"obs{t}", "tool_output", block(t), t)))
    return turns


def _score(answer: str, gold: str) -> float:
    if not gold:
        return 1.0
    return 1.0 if gold in (answer or "") else 0.0


def cmd_demo(args: argparse.Namespace) -> int:
    budgets = [int(b) for b in args.budgets.split(",")]
    arms = ["full", "recency", "reactive_afm", "foveance", "oracle"]
    turns = _demo_turns(seed=args.seed, n_turns=args.turns, drift=args.drift)

    print(f"\nFoveance offline demo  (MockLLM, {args.turns} turns, drift={args.drift})")
    print("Per-arm accuracy and total input tokens across budgets:\n")
    header = "budget   " + "  ".join(f"{a:>13}" for a in arms)
    print(header)
    for b in budgets:
        cells = []
        for arm in arms:
            ctrl = Controller(MockLLM(), budget=b, policy=arm, drift=args.drift)
            n_recall = in_tok = 0
            n_correct = 0.0
            for t, (q, gold, item) in enumerate(turns):
                ctrl.add_item(item)
                rec = ctrl.step(q, t)
                in_tok += rec.input_tokens
                if gold:
                    n_recall += 1
                    n_correct += _score(rec.answer, gold)
            acc = n_correct / n_recall if n_recall else 0.0
            cells.append(f"{acc:4.2f}/{in_tok//1000:>4}k")
        print(f"{b:>6}   " + "  ".join(f"{c:>13}" for c in cells))
    print("\nReading: cell = accuracy / total input tokens. Budgeted arms (reactive_afm, foveance,")
    print("oracle) match 'full' accuracy at a fraction of its tokens and dominate 'recency'.")
    print("reactive_afm and foveance differ ONLY in predictor drift (anticipation). See docs/NOVELTY.md.\n")
    return 0


def _load_config_file() -> dict:
    """Load default settings from ``~/.foveance.toml`` then ``./.foveance.toml`` (the latter
    overrides the former key-by-key). Uses the stdlib ``tomllib`` (Python 3.11+); on 3.10,
    where it isn't available, config files are silently skipped and flags/env vars still work."""
    try:
        import tomllib
    except ModuleNotFoundError:
        return {}

    import os

    config: dict = {}
    for path in (os.path.expanduser("~/.foveance.toml"), ".foveance.toml"):
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as fh:
            config.update(tomllib.load(fh))
    return config


def _proxy_from_args(args: argparse.Namespace):
    """Build a configured FoveanceProxy from CLI flags, with fallback to env vars, then to
    ``~/.foveance.toml``/``./.foveance.toml``, then built-in defaults (shared by ``proxy``
    and ``wrap``). Returns (proxy, upstream)."""
    import os

    from .proxy import FoveanceProxy

    config = _load_config_file()

    def setting(arg_val, env_name, key, default, cast):
        if arg_val is not None:
            return arg_val
        if env_name in os.environ:
            return cast(os.environ[env_name])
        if key in config:
            return cast(config[key])
        return default

    upstream = setting(args.upstream, "FOVEANCE_UPSTREAM", "upstream",
                       "http://localhost:11434/v1", str)
    budget = setting(args.budget, "FOVEANCE_BUDGET", "budget", 2000, int)
    drift = setting(args.drift, "FOVEANCE_DRIFT", "drift", 0.6, float)
    policy = setting(args.policy, "FOVEANCE_POLICY", "policy", "foveance", str)
    protect = setting(args.agentic_protect_last, "FOVEANCE_AGENTIC_PROTECT_LAST",
                      "agentic_protect_last", 3, int)
    token_counter = None
    if args.exact_tokens:
        from .metrics import make_token_counter
        encoding = setting(args.token_encoding, "FOVEANCE_TOKEN_ENCODING", "token_encoding",
                           "cl100k_base", str)
        token_counter = make_token_counter(encoding)
    # Pro: with an active license, savings persist across restarts (all-time dashboard + CSV).
    savings_log = None
    from . import license as _license
    if _license.current() is not None:
        savings_log = _license.SavingsLog()
    # R1: anticipatory agentic compression + model-driven re-inflation. Both imply the vault
    # (durable full texts) so nothing is ever truly lost.
    agentic_allocator = bool(getattr(args, "agentic_allocator", False)
                             or os.environ.get("FOVEANCE_AGENTIC_ALLOCATOR")
                             or config.get("agentic_allocator", False))
    expand_tool = bool(getattr(args, "expand_tool", False)
                       or os.environ.get("FOVEANCE_EXPAND_TOOL")
                       or config.get("expand_tool", False))
    vault = None
    if agentic_allocator or expand_tool:
        from .vault import ItemVault
        vault = ItemVault()
    # R3: --learn logs local traces of what each query referenced, and loads the trained model
    # (foveance train) when one exists, so allocation improves on your own workload.
    trace_log = future_model = None
    if bool(getattr(args, "learn", False) or os.environ.get("FOVEANCE_LEARN")
            or config.get("learn", False)):
        from .traces import TraceLogger, load_model
        trace_log = TraceLogger()
        future_model = load_model()
    proxy = FoveanceProxy(budget=budget, drift=drift, policy=policy, agentic_protect_last=protect,
                          cache_aware=args.cache_aware, price_per_mtok=args.price_per_mtok,
                          token_counter=token_counter, savings_log=savings_log,
                          agentic_allocator=agentic_allocator, expand_tool=expand_tool,
                          vault=vault, trace_log=trace_log, future_model=future_model)
    return proxy, upstream


def _admin_token(args) -> "Optional[str]":
    import os

    return getattr(args, "admin_token", None) or os.environ.get("FOVEANCE_ADMIN_TOKEN") or None


def cmd_train(args: argparse.Namespace) -> int:
    """Fit the learned future-relevance model on locally logged traces (see --learn)."""
    from .traces import train

    r = train(horizon=args.horizon)
    if not r["trained"]:
        print(r["reason"], file=sys.stderr)
        return 1
    print(f"Trained on {r['events']} events across {r['conversations']} conversations.")
    print(f"Model saved to {r['model_path']}; the proxy will use it automatically with --learn.")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Replay a conversation log offline and report what Foveance would have saved."""
    from .audit import audit_conversations, format_report, load_conversations

    convs = load_conversations(args.logfile)
    if not convs:
        print(f"no conversations found in {args.logfile} (expected JSON/JSONL with "
              "'messages' lists)", file=sys.stderr)
        return 2
    token_counter = None
    if args.exact_tokens:
        from .metrics import make_token_counter
        token_counter = make_token_counter(args.token_encoding or "cl100k_base")
    report = audit_conversations(convs, budget=args.budget or 2000,
                                 token_counter=token_counter)
    print(format_report(report, price_per_mtok=args.price_per_mtok,
                        monthly_requests=args.monthly_requests))
    return 0


def cmd_compress(args: argparse.Namespace) -> int:
    """Losslessly compress a conversation log or text file with the redundancy codec and report
    the measured ratio. This is the codec surface: reversible, no facts dropped."""
    from .codec import RedundancyCodec

    token_counter = None
    if args.exact_tokens:
        from .metrics import make_token_counter
        token_counter = make_token_counter(args.token_encoding or "cl100k_base")
    codec = RedundancyCodec(min_run=args.min_run, token_counter=token_counter)

    # Try the conversation-log shape first (one item per message); fall back to plain text.
    items: list = []
    try:
        from .audit import load_conversations
        convs = load_conversations(args.path)
    except Exception:
        convs = []
    if convs:
        for ci, conv in enumerate(convs):
            for mi, msg in enumerate(conv.get("messages", [])):
                c = msg.get("content")
                text = c if isinstance(c, str) else "\n".join(
                    str(b.get("text", b.get("content", "")) if isinstance(b, dict) else b)
                    for b in c) if isinstance(c, list) else str(c)
                if text.strip():
                    items.append((f"c{ci}m{mi}", text))
    if not items:
        with open(args.path, encoding="utf-8", errors="replace") as f:
            items = [("file", f.read())]

    rendered, report = codec.render(items)
    print(str(report))
    if not report.lossless:  # should never happen; the codec guarantees it
        print("WARNING: round-trip check failed", file=sys.stderr)
        return 1
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n\n".join(t for _, t in rendered))
        print(f"wrote compressed text to {args.out}")
    return 0


def cmd_license(args: argparse.Namespace) -> int:
    """Activate, inspect, or remove a Foveance Pro license (verified offline)."""
    from . import license as _license

    if args.action == "activate":
        if not args.key:
            print("usage: foveance license activate FOV1-...", file=sys.stderr)
            return 2
        data = _license.activate(args.key)
        if data is None:
            print("Invalid license key.", file=sys.stderr)
            return 1
        print(f"Foveance Pro activated for {data.get('email')} (plan: {data.get('plan')}).")
        print("Persistent savings history is now on for `foveance proxy` and `foveance wrap`.")
        return 0
    if args.action == "deactivate":
        print("License removed." if _license.deactivate() else "No license was active.")
        return 0
    data = _license.current()
    if data is None:
        print("No active license. The package is fully functional; Pro adds persistent "
              "savings history and CSV export.")
    else:
        print(f"Foveance Pro active: {data.get('email')} (plan: {data.get('plan')}).")
        totals = _license.SavingsLog().totals()
        print(f"All-time recorded: {totals['tokens_saved']:,} tokens saved "
              f"across {totals['requests']:,} requests.")
    return 0


def cmd_proxy(args: argparse.Namespace) -> int:
    from .proxy import build_app

    try:
        import uvicorn  # type: ignore
    except Exception:
        print("foveance proxy needs uvicorn: pip install foveance",
              file=sys.stderr)
        return 2

    proxy, upstream = _proxy_from_args(args)
    base = f"http://{args.host}:{args.port}/v1"
    app = build_app(proxy, upstream_url=upstream, admin_token=_admin_token(args))
    print(f"Foveance proxy  http://{args.host}:{args.port}  ->  upstream {upstream}")
    print(f"  policy={proxy.policy} budget={proxy.budget} tokens/turn drift={proxy.drift}."
          " Point any client here:")
    print(f"    OpenAI / Ollama / Codex:   OPENAI_BASE_URL={base}   (POST /chat/completions)")
    print(f"    Anthropic / Claude Code:   ANTHROPIC_BASE_URL=http://{args.host}:{args.port}"
          "   (POST /v1/messages)")
    print("  Streaming and /v1/models are passed through; your API key is forwarded unchanged.")
    print(f"  Live tokens-saved dashboard: http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port)  # pragma: no cover
    return 0


def cmd_wrap(args: argparse.Namespace) -> int:
    """Run any CLI/agent through the proxy with a single command:

        foveance wrap claude
        foveance wrap --upstream https://api.openai.com/v1 -- codex "fix the tests"

    Starts the proxy on localhost, points ``ANTHROPIC_BASE_URL``/``OPENAI_BASE_URL`` at it for
    the child process only, runs the tool, and prints a tokens-saved summary on exit. Your API
    key/OAuth is untouched: the proxy forwards credentials and stores nothing."""
    import os
    import shutil
    import subprocess
    import threading
    import time

    from .proxy import build_app

    try:
        import uvicorn  # type: ignore
    except Exception:
        print("foveance wrap needs uvicorn: pip install foveance",
              file=sys.stderr)
        return 2

    cmd = list(args.command)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("usage: foveance wrap [options] -- <command> [args...]"
              "   e.g.: foveance wrap claude", file=sys.stderr)
        return 2

    # Resolve the tool to a known adapter (claude-code, codex, aider, ...) for a precise upstream
    # and the exact env vars that tool reads. Unknown tools fall back to the broad default env.
    from .adapters import resolve_adapter, default_env
    adapter = resolve_adapter(cmd[0])
    if not args.upstream and not os.environ.get("FOVEANCE_UPSTREAM"):
        args.upstream = adapter.upstream if adapter else "https://api.openai.com/v1"
    proxy, upstream = _proxy_from_args(args)

    app = build_app(proxy, upstream_url=upstream, admin_token=_admin_token(args))
    config = uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 15
    while not server.started and time.time() < deadline:  # pragma: no cover - timing
        time.sleep(0.05)
    if not server.started:  # pragma: no cover - port conflict
        print(f"foveance wrap: proxy failed to start on port {args.port} (already in use? "
              "pass --port)", file=sys.stderr)
        return 2

    root = f"http://127.0.0.1:{args.port}"
    env = dict(os.environ)
    # Set the tool's specific base-URL vars if we know it; else the broad default set so whatever
    # the child reads, it finds Foveance. A known adapter also gets the broad set as a safety net.
    env.update(default_env(root))
    if adapter:
        env.update(adapter.env(root))

    exe = shutil.which(cmd[0]) or cmd[0]
    tag = f" [{adapter.name}]" if adapter else ""
    print(f"foveance wrap{tag}: proxy {root} -> {upstream}  (dashboard: {root}/)")
    print(f"foveance wrap: launching {' '.join(cmd)}\n")
    try:
        rc = subprocess.call([exe, *cmd[1:]], env=env)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        rc = 130
    except OSError as e:
        print(f"foveance wrap: could not launch {cmd[0]!r}: {e}", file=sys.stderr)
        rc = 2
    finally:
        server.should_exit = True
        s = proxy.stats()
        print("\n" + "-" * 62)
        print("Foveance session summary")
        print(f"  requests proxied : {s['requests']}  ({s['compressed_requests']} compressed)")
        print(f"  est. input tokens: {s['est_tokens_before']:,} -> {s['est_tokens_after']:,}")
        print(f"  est. saved       : {s['est_tokens_saved']:,} tokens ({s['est_saved_pct']}%)"
              f"  ~ ${s['est_usd_saved']:.4f} at ${s['price_per_mtok']}/Mtok input")
        print("  (chars/4 estimate on request payloads; exact counts come from your provider)")
    return rc


def cmd_adapters(args: argparse.Namespace) -> int:
    """List the agent CLIs/SDKs Foveance knows how to sit in front of."""
    from .adapters import list_adapters
    print("Foveance adapters -- run any of these through the proxy:\n")
    print(f"  {'adapter':<14} {'dialect':<10} {'launch':<26} env vars set")
    print(f"  {'-'*14} {'-'*10} {'-'*26} {'-'*24}")
    for a in list_adapters():
        launch = f"foveance wrap {a.aliases[0] if a.aliases else a.name}"
        evs = ", ".join((*a.env_root, *a.env_v1)) or "(SDK base_url)"
        print(f"  {a.name:<14} {a.dialect:<10} {launch:<26} {evs}")
    print("\nUnknown tool? `foveance wrap -- <cmd>` still sets the common base-URL vars, or use")
    print("`foveance env <adapter>` to print exports for a long-running `foveance proxy`.")
    return 0


def cmd_env(args: argparse.Namespace) -> int:
    """Print shell exports that point a tool at an already-running proxy (bash + PowerShell)."""
    from .adapters import resolve_adapter, default_env
    root = f"http://{args.host}:{args.port}".rstrip("/")
    adapter = resolve_adapter(args.adapter)
    env = default_env(root)
    if adapter:
        env.update(adapter.env(root))
    elif args.adapter not in (None, "", "all"):
        print(f"# unknown adapter {args.adapter!r}; printing the broad default set", file=sys.stderr)
    tag = adapter.name if adapter else "default"
    print(f"# Foveance env for {tag}  (proxy at {root}; start it with `foveance proxy`)")
    for k, v in env.items():
        print(f"export {k}={v}")
    print("# PowerShell:")
    for k, v in env.items():
        print(f"#   $env:{k} = \"{v}\"")
    return 0


def cmd_bench(args: argparse.Namespace, extra: list[str]) -> int:
    import os
    import subprocess

    here = os.path.dirname(__file__)
    runner = os.path.normpath(os.path.join(here, "..", "..", "bench", "run_bench.py"))
    if not os.path.exists(runner):
        print(f"benchmark runner not found at {runner}", file=sys.stderr)
        return 2
    return subprocess.call([sys.executable, runner, *extra])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="foveance", description="Anticipatory context allocation.")
    sub = p.add_subparsers(dest="cmd")

    d = sub.add_parser("demo", help="offline Pareto demo (MockLLM)")
    d.add_argument("--budgets", default="800,1600,2500,4000")
    d.add_argument("--turns", type=int, default=36)
    d.add_argument("--drift", type=float, default=0.7)
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(func=cmd_demo)

    pr = sub.add_parser("proxy", help="OpenAI- and Anthropic-compatible reverse proxy")
    pr.add_argument("--host", default="0.0.0.0")
    pr.add_argument("--port", type=int, default=8799)
    pr.add_argument("--budget", type=int, default=None, help="tokens/turn (env: FOVEANCE_BUDGET)")
    pr.add_argument("--drift", type=float, default=None, help="anticipation drift (env: FOVEANCE_DRIFT)")
    pr.add_argument("--policy", default=None,
                    help="foveance|reactive_afm|recency|full (env: FOVEANCE_POLICY)")
    pr.add_argument("--upstream", default=None,
                    help="upstream base URL (env: FOVEANCE_UPSTREAM)")
    pr.add_argument("--agentic-protect-last", type=int, default=None,
                    help="recent tool-use turns kept full (env: FOVEANCE_AGENTIC_PROTECT_LAST)")
    pr.add_argument("--cache-aware", action="store_true",
                    help="never modify content at/before the last Anthropic cache_control "
                         "breakpoint (preserves the provider's prompt cache)")
    pr.add_argument("--price-per-mtok", type=float, default=3.0,
                    help="assumed $/M input tokens for the dashboard's $-saved estimate")

    pr.add_argument("--agentic-allocator", action="store_true",
                    help="R1: allocate graded fidelities to old tool-transcript payloads with the "
                         "anticipatory allocator (instead of blind digestion); implies the vault")
    pr.add_argument("--expand-tool", action="store_true",
                    help="R1: let the model re-inflate any compressed item via a foveance_expand "
                         "tool the proxy resolves transparently (non-streaming requests)")
    pr.add_argument("--admin-token", default=None,
                    help="require this token (?token=... or Bearer) on /admin endpoints "
                         "(env: FOVEANCE_ADMIN_TOKEN)")
    pr.add_argument("--learn", action="store_true",
                    help="R3: log local traces of what each query referenced and use the "
                         "trained model from `foveance train` when present (env: FOVEANCE_LEARN)")
    pr.add_argument("--exact-tokens", action="store_true",
                    help="count tokens with a real tokenizer (tiktoken, if installed) instead "
                         "of the chars/4 heuristic, for accounting and the dashboard")
    pr.add_argument("--token-encoding", default=None,
                    help="tiktoken encoding for --exact-tokens, e.g. o200k_base for gpt-4o+ "
                         "(env: FOVEANCE_TOKEN_ENCODING, default: cl100k_base)")
    pr.set_defaults(func=cmd_proxy)

    w = sub.add_parser("wrap", help="run any CLI/agent through the proxy (one command); "
                                    "flags go BEFORE the wrapped command")
    w.add_argument("--port", type=int, default=8799)
    w.add_argument("--budget", type=int, default=None, help="tokens/turn (env: FOVEANCE_BUDGET)")
    w.add_argument("--drift", type=float, default=None, help="anticipation drift (env: FOVEANCE_DRIFT)")
    w.add_argument("--policy", default=None,
                   help="foveance|reactive_afm|recency|full (env: FOVEANCE_POLICY)")
    w.add_argument("--upstream", default=None,
                   help="upstream base URL; inferred from the tool if omitted "
                        "(claude* -> Anthropic, otherwise OpenAI; env: FOVEANCE_UPSTREAM)")
    w.add_argument("--agentic-protect-last", type=int, default=None,
                   help="recent tool-use turns kept full (env: FOVEANCE_AGENTIC_PROTECT_LAST)")
    w.add_argument("--cache-aware", action="store_true",
                   help="never modify content at/before the last Anthropic cache_control breakpoint")
    w.add_argument("--price-per-mtok", type=float, default=3.0,
                   help="assumed $/M input tokens for the exit summary's $-saved estimate")

    w.add_argument("--agentic-allocator", action="store_true",
                    help="R1: allocate graded fidelities to old tool-transcript payloads with the "
                         "anticipatory allocator (instead of blind digestion); implies the vault")
    w.add_argument("--expand-tool", action="store_true",
                    help="R1: let the model re-inflate any compressed item via a foveance_expand "
                         "tool the proxy resolves transparently (non-streaming requests)")
    w.add_argument("--admin-token", default=None,
                    help="require this token (?token=... or Bearer) on /admin endpoints "
                         "(env: FOVEANCE_ADMIN_TOKEN)")
    w.add_argument("--learn", action="store_true",
                    help="R3: log local traces of what each query referenced and use the "
                         "trained model from `foveance train` when present (env: FOVEANCE_LEARN)")
    w.add_argument("--exact-tokens", action="store_true",
                   help="count tokens with a real tokenizer (tiktoken, if installed) instead "
                        "of the chars/4 heuristic, for accounting and the exit summary")
    w.add_argument("--token-encoding", default=None,
                   help="tiktoken encoding for --exact-tokens, e.g. o200k_base for gpt-4o+ "
                        "(env: FOVEANCE_TOKEN_ENCODING, default: cl100k_base)")
    w.add_argument("command", nargs=argparse.REMAINDER,
                   help="the tool to launch, e.g.: claude   or:  -- codex 'fix the tests'")
    w.set_defaults(func=cmd_wrap)

    tr = sub.add_parser("train", help="fit the learned predictor on locally logged traces")
    tr.add_argument("--horizon", type=int, default=5)
    tr.set_defaults(func=cmd_train)

    au = sub.add_parser("audit", help="replay a conversation log offline and report savings")
    au.add_argument("logfile", help="JSON/JSONL file of conversations (messages lists)")
    au.add_argument("--budget", type=int, default=None)
    au.add_argument("--price-per-mtok", type=float, default=3.0)
    au.add_argument("--monthly-requests", type=int, default=None,
                    help="extrapolate savings to this many requests per month")
    au.add_argument("--exact-tokens", action="store_true",
                    help="count with tiktoken instead of chars/4")
    au.add_argument("--token-encoding", default=None)
    au.set_defaults(func=cmd_audit)

    cp = sub.add_parser("compress", help="losslessly compress a log/text file with the "
                        "redundancy codec and report the ratio")
    cp.add_argument("path", help="conversation log (JSON/JSONL) or any text file")
    cp.add_argument("--min-run", type=int, default=2,
                    help="shortest run of repeated lines worth referencing")
    cp.add_argument("--out", default=None, help="write the compressed text here")
    cp.add_argument("--exact-tokens", action="store_true",
                    help="count with tiktoken instead of chars/4")
    cp.add_argument("--token-encoding", default=None)
    cp.set_defaults(func=cmd_compress)

    lic = sub.add_parser("license", help="activate/status/deactivate a Foveance Pro license")
    lic.add_argument("action", nargs="?", default="status",
                     choices=["activate", "status", "deactivate"])
    lic.add_argument("key", nargs="?", default=None, help="license key (for activate)")
    lic.set_defaults(func=cmd_license)

    ad = sub.add_parser("adapters", help="list the agent CLIs/SDKs Foveance can wrap")
    ad.set_defaults(func=cmd_adapters)

    ev = sub.add_parser("env", help="print shell exports to point a tool at a running proxy")
    ev.add_argument("adapter", nargs="?", default="all",
                    help="claude-code | codex | aider | ... (see `foveance adapters`)")
    ev.add_argument("--host", default="127.0.0.1")
    ev.add_argument("--port", type=int, default=8799)
    ev.set_defaults(func=cmd_env)

    b = sub.add_parser("bench", help="run the benchmark harness (forwards extra args)")
    b.set_defaults(func=None)

    sub.add_parser("version", help="print version").set_defaults(func=lambda a: print(__version__) or 0)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if args.cmd is None:
        parser.print_help()
        return 0
    if args.cmd == "bench":
        return cmd_bench(args, extra)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
