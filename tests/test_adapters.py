"""User-facing CLI adapters: the registry that lets `foveance wrap`/`env` target Claude Code,
Codex, Aider and friends by name with the correct dialect and base-URL env vars."""
from foveance.adapters import (Adapter, default_env, list_adapters, resolve_adapter)


def test_known_tools_resolve_with_right_dialect():
    assert resolve_adapter("claude").name == "claude-code"
    assert resolve_adapter("claude-code").dialect == "anthropic"
    assert resolve_adapter("codex").dialect == "openai"
    assert resolve_adapter("aider").dialect == "openai"
    assert resolve_adapter("litellm").name == "litellm"


def test_resolve_by_command_path_and_exe():
    assert resolve_adapter("/usr/local/bin/claude").name == "claude-code"
    assert resolve_adapter(r"C:\\tools\\codex.exe").name == "codex"


def test_unknown_claude_like_falls_back_to_anthropic():
    a = resolve_adapter("claude-flavoured-thing")
    assert a is not None and a.dialect == "anthropic"


def test_unknown_tool_is_none():
    assert resolve_adapter("totally-unknown-xyz") is None
    assert resolve_adapter("") is None


def test_anthropic_env_is_root_openai_env_is_v1():
    root = "http://127.0.0.1:8799"
    cc = resolve_adapter("claude-code").env(root)
    assert cc["ANTHROPIC_BASE_URL"] == root           # SDK appends /v1/messages itself
    cx = resolve_adapter("codex").env(root)
    assert cx["OPENAI_BASE_URL"] == root + "/v1"       # SDK appends /chat/completions
    assert cx["OPENAI_API_BASE"] == root + "/v1"


def test_default_env_covers_both_dialects():
    env = default_env("http://h:1/")
    assert env["ANTHROPIC_BASE_URL"] == "http://h:1"
    assert env["OPENAI_BASE_URL"] == "http://h:1/v1"
    assert env["OPENAI_API_BASE"] == "http://h:1/v1"


def test_registry_is_nonempty_and_well_formed():
    ads = list_adapters()
    assert len(ads) >= 8                               # claude-code..codex..and more
    for a in ads:
        assert isinstance(a, Adapter)
        assert a.dialect in ("anthropic", "openai")
        assert a.upstream.startswith("https://")
        assert a.env("http://x:1")                     # produces at least one var


def test_cli_adapters_and_env_commands(capsys):
    from foveance.cli import main
    assert main(["adapters"]) == 0
    out = capsys.readouterr().out
    assert "claude-code" in out and "codex" in out and "aider" in out

    assert main(["env", "codex", "--port", "9001"]) == 0
    out = capsys.readouterr().out
    assert "OPENAI_BASE_URL=http://127.0.0.1:9001/v1" in out
    assert "PowerShell" in out                          # both shells printed
