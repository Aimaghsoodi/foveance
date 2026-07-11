"""R3 learning loop: trace logging, trace->training reconstruction, train, and model loading."""
import json

from foveance.proxy import FoveanceProxy
from foveance.store import Item
from foveance.traces import TraceLogger, build_training_traces, load_model, train
from foveance.vault import ItemVault


def _items():
    return [Item("i-key", "tool_output", "API_KEY=SECRET-42 recorded in config", 0),
            Item("i-noise", "tool_output", "zzz unrelated logging chatter output", 0)]


def test_trace_logger_detects_referenced_items(tmp_path):
    log = TraceLogger(path=str(tmp_path / "t.jsonl"))
    refd = log.log_event("c1", 0, "what was the API_KEY in the config?", _items())
    assert refd == ["i-key"]
    row = json.loads((tmp_path / "t.jsonl").read_text(encoding="utf-8").strip())
    assert row["conv"] == "c1" and row["referenced"] == ["i-key"]
    assert len(row["items"]) == 2


def test_build_and_train_roundtrip(tmp_path):
    tpath, mpath = str(tmp_path / "t.jsonl"), str(tmp_path / "m.json")
    log = TraceLogger(path=tpath)
    for turn in range(4):
        log.log_event("c1", turn, "recall the API_KEY config secret", _items())
    traces = build_training_traces(tpath)
    assert len(traces) == 1 and len(traces[0]["queries"]) == 4
    assert traces[0]["referenced"][0] == {"i-key"}
    r = train(traces_path=tpath, model_path=mpath)
    assert r["trained"] and r["events"] == 4
    model = load_model(mpath)
    assert model is not None and any(w != 0.0 for w in model.weights)
    assert load_model(str(tmp_path / "absent.json")) is None


def test_train_without_traces_reports_gracefully(tmp_path):
    r = train(traces_path=str(tmp_path / "none.jsonl"), model_path=str(tmp_path / "m.json"))
    assert r["trained"] is False and "no traces" in r["reason"]


def test_proxy_logs_traces_during_agentic_allocation(tmp_path):
    log = TraceLogger(path=str(tmp_path / "t.jsonl"))
    px = FoveanceProxy(budget=300, agentic_protect_last=1, agentic_allocator=True,
                       vault=ItemVault(path=str(tmp_path / "v.db")), trace_log=log)
    big = "API_KEY=SECRET-9 recorded\n" + "\n".join(
        f"log line {i}: status=ok latency=fine path=/srv/whatever" for i in range(120))
    tools = [{"name": "bash", "description": "x",
              "input_schema": {"type": "object", "properties": {}}}]
    msgs = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                                          "content": big}]},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            {"role": "user", "content": [{"type": "text", "text": "recall the API_KEY"}]}]
    px.prepare_anthropic({"system": "s", "messages": msgs, "tools": tools, "user": "tl"})
    lines = (tmp_path / "t.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["referenced"], "the API_KEY item should be marked referenced"


def test_proxy_uses_trained_future_model(tmp_path):
    tpath, mpath = str(tmp_path / "t.jsonl"), str(tmp_path / "m.json")
    log = TraceLogger(path=tpath)
    for turn in range(3):
        log.log_event("c1", turn, "recall the API_KEY config secret", _items())
    train(traces_path=tpath, model_path=mpath)
    model = load_model(mpath)
    px = FoveanceProxy(budget=300, future_model=model)
    st = px._state("conv-x")
    assert st.pred.future_model is model
