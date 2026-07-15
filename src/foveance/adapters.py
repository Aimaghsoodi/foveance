"""First-class adapters for agent CLIs and SDKs.

Foveance is a black-box reverse proxy: it speaks the OpenAI Chat/Responses and Anthropic Messages
dialects, forwards your real credentials untouched, and compresses the request in between. To put
it in front of a given tool you only need two things: the tool's *upstream* (which real API it
talks to) and the *environment variable(s)* that tool reads to discover a base URL. This module is
the registry of those facts for the popular agent CLIs, so ``foveance wrap`` / ``foveance env``
can target a tool by name instead of guessing.

Honesty note: env-var names and support vary by tool and version. The entries below are the
well-established ones; each carries a ``note`` and there are generic escape hatches
(``openai`` / ``anthropic``) plus a ``--set-env`` override on the CLI for anything not listed.
This registry is user-facing routing only; it is unrelated to the benchmark's OpenRouter client,
which is internal.
"""
from __future__ import annotations

from dataclasses import dataclass

ANTHROPIC = "https://api.anthropic.com/v1"
OPENAI = "https://api.openai.com/v1"


@dataclass(frozen=True)
class Adapter:
    name: str                       # canonical id
    dialect: str                    # "anthropic" | "openai" (Responses is served on the OpenAI app)
    upstream: str                   # real API the tool ultimately calls
    env_root: tuple = ()            # env vars set to the proxy ROOT (http://host:port)
    env_v1: tuple = ()              # env vars set to the proxy root + "/v1"
    aliases: tuple = ()             # command names / synonyms that map here
    note: str = ""

    def env(self, root: str) -> dict:
        """The environment overrides that point this tool at a running proxy at ``root``."""
        out = {k: root for k in self.env_root}
        out.update({k: root.rstrip("/") + "/v1" for k in self.env_v1})
        return out


# The registry. Ordered roughly by popularity for the `adapters` listing.
_ADAPTERS: list = [
    Adapter("claude-code", "anthropic", ANTHROPIC,
            env_root=("ANTHROPIC_BASE_URL",), aliases=("claude", "claude-code", "cc"),
            note="Anthropic's CLI. Reads ANTHROPIC_BASE_URL; OAuth/API key forwarded untouched."),
    Adapter("codex", "openai", OPENAI,
            env_v1=("OPENAI_BASE_URL", "OPENAI_API_BASE"),
            aliases=("codex", "codex-cli"),
            note="OpenAI Codex CLI. Speaks Chat + Responses; both are served by the proxy."),
    Adapter("aider", "openai", OPENAI,
            env_v1=("OPENAI_API_BASE", "OPENAI_BASE_URL"), aliases=("aider",),
            note="Aider. Set the model to an openai/ model so it uses OPENAI_API_BASE."),
    Adapter("cline", "openai", OPENAI,
            env_v1=("OPENAI_BASE_URL",), aliases=("cline",),
            note="Cline (VS Code). Set the OpenAI-compatible Base URL to the proxy in its "
                 "settings; the env var helps only if launched from a terminal."),
    Adapter("opencode", "openai", OPENAI,
            env_v1=("OPENAI_BASE_URL",), aliases=("opencode",),
            note="opencode agent. OpenAI-compatible base URL."),
    Adapter("goose", "openai", OPENAI,
            env_v1=("OPENAI_BASE_URL", "OPENAI_HOST"), aliases=("goose",),
            note="Block's Goose. Point its OpenAI provider host at the proxy."),
    Adapter("continue", "openai", OPENAI,
            env_v1=("OPENAI_BASE_URL",), aliases=("continue", "continuedev"),
            note="Continue.dev. Set apiBase to the proxy in config for an openai provider."),
    Adapter("openai", "openai", OPENAI,
            env_v1=("OPENAI_BASE_URL", "OPENAI_API_BASE"),
            aliases=("openai", "openai-sdk", "python"),
            note="Generic OpenAI SDK / any OpenAI-compatible client (base_url=...)."),
    Adapter("anthropic", "anthropic", ANTHROPIC,
            env_root=("ANTHROPIC_BASE_URL",), aliases=("anthropic", "anthropic-sdk"),
            note="Generic Anthropic SDK (base_url=...)."),
    Adapter("litellm", "openai", OPENAI,
            env_v1=("OPENAI_API_BASE", "OPENAI_BASE_URL"), aliases=("litellm",),
            note="LiteLLM proxy/SDK for openai-family models."),
    Adapter("cursor", "openai", OPENAI,
            env_v1=("OPENAI_BASE_URL",), aliases=("cursor", "cursor-ide"),
            note="Cursor (IDE). Set Settings > Models > 'Override OpenAI Base URL' to the proxy "
                 "and add an OpenAI-compatible key; the env var applies only when launched from a "
                 "terminal."),
    Adapter("windsurf", "openai", OPENAI,
            env_v1=("OPENAI_BASE_URL",), aliases=("windsurf",),
            note="Windsurf (Codeium IDE). Point its OpenAI-compatible base URL at the proxy in "
                 "settings."),
    Adapter("roo", "openai", OPENAI,
            env_v1=("OPENAI_BASE_URL",), aliases=("roo", "roo-code", "roocode"),
            note="Roo Code (VS Code). Set the OpenAI-Compatible provider's Base URL to the proxy."),
    Adapter("zed", "openai", OPENAI,
            env_v1=("OPENAI_BASE_URL",), aliases=("zed",),
            note="Zed editor. Configure an openai provider with api_url pointing at the proxy."),
    Adapter("gemini", "openai", OPENAI,
            env_v1=("OPENAI_BASE_URL", "OPENAI_API_BASE"), aliases=("gemini", "gemini-cli"),
            note="Gemini CLI via its OpenAI-compatible endpoint (base_url=...)."),
]

_BY_KEY = {}
for _a in _ADAPTERS:
    for _k in (_a.name, *_a.aliases):
        _BY_KEY[_k.lower()] = _a


def list_adapters() -> list:
    """All registered adapters, in listing order."""
    return list(_ADAPTERS)


def resolve_adapter(name: str):
    """Look up an adapter by name/alias, or by the basename of a launch command (so
    ``resolve_adapter('/usr/bin/claude')`` -> claude-code). Returns None if unknown."""
    if not name:
        return None
    key = name.lower()
    if key in _BY_KEY:
        return _BY_KEY[key]
    base = key.replace("\\", "/").rsplit("/", 1)[-1]
    base = base[:-4] if base.endswith(".exe") else base
    if base in _BY_KEY:
        return _BY_KEY[base]
    # heuristic fallback: anything mentioning claude speaks Anthropic, else OpenAI
    if "claude" in base:
        return _BY_KEY["anthropic"]
    return None


def default_env(root: str, all_dialects: bool = True) -> dict:
    """The broad env set ``wrap`` uses when the tool is unknown: point every common base-URL var
    at the proxy so whatever the child reads, it finds Foveance."""
    root = root.rstrip("/")
    env = {"ANTHROPIC_BASE_URL": root, "OPENAI_BASE_URL": root + "/v1",
           "OPENAI_API_BASE": root + "/v1"}
    return env
