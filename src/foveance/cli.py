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
    proxy = FoveanceProxy(budget=budget, drift=drift, policy=policy, agentic_protect_last=protect,
                          cache_aware=args.cache_aware, price_per_mtok=args.price_per_mtok)
    return proxy, upstream


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
    app = build_app(proxy, upstream_url=upstream)
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

    # Infer the upstream from the tool when not given: Claude-family tools speak Anthropic,
    # everything else defaults to the OpenAI protocol (override with --upstream/FOVEANCE_UPSTREAM).
    if not args.upstream and not os.environ.get("FOVEANCE_UPSTREAM"):
        name = os.path.basename(cmd[0]).lower()
        args.upstream = ("https://api.anthropic.com/v1" if "claude" in name
                         else "https://api.openai.com/v1")
    proxy, upstream = _proxy_from_args(args)

    app = build_app(proxy, upstream_url=upstream)
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
    env["ANTHROPIC_BASE_URL"] = root          # Anthropic SDK / Claude Code
    env["OPENAI_BASE_URL"] = root + "/v1"     # OpenAI SDK / most agents
    env["OPENAI_API_BASE"] = root + "/v1"     # older OpenAI-compatible clients

    exe = shutil.which(cmd[0]) or cmd[0]
    print(f"foveance wrap: proxy {root} -> {upstream}  (dashboard: {root}/)")
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
    w.add_argument("command", nargs=argparse.REMAINDER,
                   help="the tool to launch, e.g.: claude   or:  -- codex 'fix the tests'")
    w.set_defaults(func=cmd_wrap)

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
