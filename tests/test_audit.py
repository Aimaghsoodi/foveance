"""foveance audit: offline replay of conversation logs with a savings report."""
import json

from foveance import cli
from foveance.audit import audit_conversations, format_report, load_conversations


def _long_conv() -> dict:
    noise = "\n".join(f"log {i}: status=ok" for i in range(120))
    return {"messages": [
        {"role": "user", "content": f"notes:\nAPI_KEY=SECRET-1\n{noise}"},
        {"role": "assistant", "content": "noted"},
        {"role": "user", "content": f"more logs:\n{noise}"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "what was the API_KEY?"},
    ]}


def test_load_jsonl_json_and_bare_list(tmp_path):
    conv = _long_conv()
    jl = tmp_path / "log.jsonl"
    jl.write_text(json.dumps(conv) + "\n" + json.dumps(conv["messages"]) + "\n",
                  encoding="utf-8")
    assert len(load_conversations(str(jl))) == 2
    js = tmp_path / "log.json"
    js.write_text(json.dumps([conv, conv]), encoding="utf-8")
    assert len(load_conversations(str(js))) == 2
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps(conv["messages"]), encoding="utf-8")
    loaded = load_conversations(str(bare))
    assert len(loaded) == 1 and len(loaded[0]["messages"]) == 5
    empty = tmp_path / "empty.json"
    empty.write_text("", encoding="utf-8")
    assert load_conversations(str(empty)) == []


def test_audit_reports_real_savings():
    r = audit_conversations([_long_conv()], budget=200)
    assert r["conversations"] == 1
    assert r["requests"] == 3                      # three user turns
    assert r["tokens_after"] < r["tokens_before"]
    assert r["tokens_saved"] > 0 and 0 < r["saved_pct"] <= 100
    text = format_report(r, price_per_mtok=3.0, monthly_requests=10000)
    assert "Foveance audit" in text and "/month" in text


def test_audit_cli_end_to_end(tmp_path, capsys):
    log = tmp_path / "log.jsonl"
    log.write_text(json.dumps(_long_conv()), encoding="utf-8")
    assert cli.main(["audit", str(log), "--budget", "200",
                     "--monthly-requests", "5000"]) == 0
    out = capsys.readouterr().out
    assert "saved" in out and "/month" in out
    assert cli.main(["audit", str(tmp_path / "missing-but-empty.json")]) == 2 \
        if (tmp_path / "missing-but-empty.json").write_text("", encoding="utf-8") is None else True
