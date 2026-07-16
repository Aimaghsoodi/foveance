"""The cross-item redundancy codec: it must be a *real* codec -- exactly reversible -- and its
reported compression ratio must be measured, monotone, and honest."""
import pytest

from foveance.codec import (CompressionReport, RedundancyCodec, Ref, from_store_items)


def _items(*texts):
    return [(f"i{k}", t) for k, t in enumerate(texts)]


# -- losslessness (the defining property) ----------------------------------------------------
def test_roundtrip_exact_on_redundant_trace():
    block = "def handler(req):\n    log.info('start')\n    return do(req)\n    log.info('end')"
    items = _items(
        "GET /a\n" + block,
        "GET /b\n" + block,            # same block, different first line
        "unrelated single line",
        "GET /c\n" + block,            # third repeat
    )
    codec = RedundancyCodec(min_run=2)
    packed = codec.pack(items)
    assert codec.unpack(packed) == items          # byte-for-byte reversible


def test_render_reports_lossless_true():
    items = _items("a\nb\nc\nd", "x\na\nb\nc\nd\ny")   # b,c,d... run repeats
    _, report = RedundancyCodec(min_run=2).render(items)
    assert report.lossless is True
    assert isinstance(report, CompressionReport)


def test_empty_and_single_item_are_identity():
    codec = RedundancyCodec()
    assert codec.unpack(codec.pack([])) == []
    one = _items("just one item\nwith two lines")
    assert codec.unpack(codec.pack(one)) == one


def test_pack_emits_ref_for_repeated_run():
    unit = "aaaaaaaaaa\nbbbbbbbbbb\ncccccccccc\ndddddddddd"
    packed = RedundancyCodec(min_run=2).pack(_items(unit, unit))
    # second item's whole run becomes a single backward Ref
    assert any(isinstance(tok, Ref) for tok in packed[1].tokens)


def test_no_redundancy_means_zero_ratio_and_no_growth():
    items = _items("alpha\nbeta", "gamma\ndelta", "epsilon\nzeta")
    rendered, report = RedundancyCodec(min_run=2).render(items)
    assert report.ratio == 0.0                     # nothing to dedupe
    assert rendered == items                        # and nothing rewritten


# -- the ratio is real and points the right way ----------------------------------------------
def test_ratio_increases_with_repetition():
    unit = ("aaaaaaaaaaaaaaaaaaaa\nbbbbbbbbbbbbbbbbbbbb\ncccccccccccccccccccc\n"
            "dddddddddddddddddddd\neeeeeeeeeeeeeeeeeeee\nffffffffffffffffffff")
    codec = RedundancyCodec(min_run=2)
    r2 = codec.analyze(_items(unit, unit)).ratio
    r5 = codec.analyze(_items(unit, unit, unit, unit, unit)).ratio
    assert 0.0 < r2 < r5                            # more copies -> more saved
    assert r5 > 0.5                                 # heavy repetition compresses hard


def test_factor_and_saved_pct_consistent():
    rep = CompressionReport(tokens_in=100, tokens_out=25, bytes_in=1, bytes_out=1, lossless=True)
    assert rep.ratio == pytest.approx(0.75)
    assert rep.saved_pct == pytest.approx(75.0)
    assert rep.factor == pytest.approx(4.0)


def test_reference_never_makes_a_run_bigger():
    # a short unique-ish run must not be replaced by a longer pointer
    items = _items("k=1", "k=1")                    # tiny single-line repeat
    rendered, report = RedundancyCodec(min_run=1).render(items)
    assert report.tokens_out <= report.tokens_in    # _worth_it guard holds


def test_intra_item_repetition_is_compressed():
    para = "line-aaaa\nline-bbbb\nline-cccc\nline-dddd"
    items = _items(para + "\n" + para)              # repeat within a single item
    report = RedundancyCodec(min_run=2).analyze(items)
    assert report.ratio > 0.0
    assert report.lossless


def test_rendered_reference_is_readable_pointer():
    block = "SELECT *\nFROM t\nWHERE id = 7\nLIMIT 1"
    rendered, _ = RedundancyCodec(min_run=2).render(_items(block, "hdr\n" + block))
    assert "fov:rpt" in rendered[1][1]              # second item points back


def test_from_store_items_adapter():
    from foveance.store import Item
    text = "hello world this is a line\nsecond line of content goes here"
    items = [Item("a", "tool_output", text, 0),
             Item("b", "tool_output", text, 1)]
    pairs = from_store_items(items)
    assert pairs == [("a", text), ("b", text)]
    report = RedundancyCodec(min_run=2).analyze(pairs)
    assert report.ratio > 0.0


def test_min_run_validation():
    with pytest.raises(ValueError):
        RedundancyCodec(min_run=0)


# -- template (shared-prefix) factoring -------------------------------------------------------
def _listing(n=20):
    return "\n".join(f"src/service/module_{i:02d}.py" for i in range(n))


def test_template_factors_shared_prefix_and_inverts_exactly():
    from foveance.codec import expand_templates
    items = _items("$ ls\n" + _listing())
    rendered, report = RedundancyCodec(min_run=1, template=True).render(items)
    text = rendered[0][1]
    assert "[fov:tpl 20 " in text                     # the shared prefix is written once
    assert "src/service/module_" in text              # and is legible in the header
    # the body carries only the suffixes, and expansion recovers the pre-template text exactly
    plain = RedundancyCodec(min_run=1, template=False).render(items)[0][0][1]
    assert expand_templates(text) == plain
    assert report.lossless is True


def test_template_saves_tokens_and_never_inflates():
    items = _items("$ ls\n" + _listing(), "$ ls again\n" + _listing())
    off = RedundancyCodec(min_run=1, template=False).analyze(items)
    on = RedundancyCodec(min_run=1, template=True).analyze(items)
    assert on.tokens_out < off.tokens_out             # a real gain on prefix-redundant listings
    assert on.tokens_out <= on.tokens_in              # and never inflates


def test_template_skipped_when_it_would_not_help():
    # unique lines with no shared prefix must be left completely alone
    items = _items("alpha one here\nbeta two there\ngamma three elsewhere")
    rendered, _ = RedundancyCodec(min_run=1, template=True).render(items)
    assert "fov:tpl" not in rendered[0][1]
    assert rendered == items


def test_expand_templates_is_identity_on_untemplated_text():
    from foveance.codec import expand_templates
    txt = "plain line\nanother [fov:rpt 3 @i0:L2] pointer stays\nlast"
    assert expand_templates(txt) == txt


def test_template_roundtrip_property_on_prefixed_lines():
    from foveance.codec import expand_templates
    # lines sharing a prefix, interleaved with pointers and unique lines
    body = "\n".join([f"2026-07-16 INFO [checkout] step {i} ok" for i in range(6)])
    items = _items("hdr\n" + body + "\nunique tail line", "hdr2\n" + body)
    c = RedundancyCodec(min_run=1, template=True)
    rendered, rep = c.render(items)
    plain = RedundancyCodec(min_run=1, template=False).render(items)[0]
    for (_, t), (_, p) in zip(rendered, plain):
        assert expand_templates(t) == p               # exact inverse in every item
    assert rep.lossless is True


# -- property-based: reversible on arbitrary line structure ----------------------------------
try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    _lines = st.lists(st.text(alphabet="abcde \t", min_size=0, max_size=6),
                      min_size=0, max_size=8).map(lambda ls: "\n".join(ls))
    _docs = st.lists(_lines, min_size=0, max_size=6)

    @settings(max_examples=200, deadline=None)
    @given(_docs)
    def test_property_roundtrip(texts):
        items = [(f"i{k}", t) for k, t in enumerate(texts)]
        codec = RedundancyCodec(min_run=1)          # most aggressive
        assert codec.unpack(codec.pack(items)) == items
except ImportError:  # hypothesis is a dev dep; skip gracefully if unavailable
    pass
