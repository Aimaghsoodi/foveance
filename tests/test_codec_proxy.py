"""0.5: the lossless codec wired into the proxy (`--codec`) and the `compress_anthropic` helper."""
from foveance import compress_anthropic
from foveance.proxy import FoveanceProxy


def _redundant_messages(n=6):
    block = "def handler(req):\n    log.info('start')\n    validate(req)\n    return do(req)"
    msgs = []
    for k in range(n):
        msgs.append({"role": "user", "content": f"trace {k}\n{block}"})
        msgs.append({"role": "assistant", "content": "ok"})
    msgs.append({"role": "user", "content": "recall the handler"})
    return msgs


def test_proxy_codec_reduces_tokens_vs_plain():
    msgs = _redundant_messages()
    plain = FoveanceProxy(budget=100000, apply_codec=False)
    coded = FoveanceProxy(budget=100000, apply_codec=True)
    out_plain, s_plain = plain.transform(list(msgs), conv_id="a")
    out_coded, s_coded = coded.transform(list(msgs), conv_id="b")
    # both keep the request valid (same number of messages back)
    assert len(out_coded) == len(out_plain)
    # the codec run recorded a positive lossless saving on this redundant history
    assert coded.codec_saved_tokens > 0
    # and the compressed system context is smaller than without the codec
    plain_ctx = next((m["content"] for m in out_plain if m.get("role") == "system"), "")
    coded_ctx = next((m["content"] for m in out_coded if m.get("role") == "system"), "")
    assert len(coded_ctx) < len(plain_ctx)


def test_proxy_codec_off_by_default():
    assert FoveanceProxy().apply_codec is False


def test_proxy_codec_keeps_last_turn_verbatim():
    msgs = _redundant_messages()
    coded = FoveanceProxy(budget=100000, apply_codec=True)
    out, _ = coded.transform(list(msgs), conv_id="c")
    assert out[-1]["content"] == "recall the handler"   # last turn never touched


def test_compress_anthropic_lossless_and_shrinks():
    block = "SELECT *\nFROM orders\nWHERE id = 7\nLIMIT 1"
    system = "You are precise."
    messages = [{"role": "user", "content": f"first\n{block}"},
                {"role": "user", "content": f"again\n{block}"}]
    new_system, new_msgs, report = compress_anthropic(system, messages)
    assert report.lossless is True
    assert len(new_msgs) == len(messages)
    assert new_system == system                          # unique system text unchanged
    assert report.saved_pct > 0                           # the repeated block was de-duplicated


def test_cli_codec_flag_parses():
    from foveance.cli import build_parser
    args = build_parser().parse_args(["proxy", "--codec", "--budget", "1500"])
    assert args.codec is True
