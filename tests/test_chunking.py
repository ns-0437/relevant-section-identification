"""
Fast unit tests for the deterministic parts of the pipeline.

Deliberately model-free so CI can run them in seconds without torch, the
embedding weights or a GPU. They cover the logic where a silent regression would
be hardest to notice downstream: table rendering, split overlap, boilerplate
detection and keyword extraction.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdf_chunker import (  # noqa: E402
    HARD_LIMIT,
    OVERLAP,
    Chunk,
    _bp_key,
    apply_cross_chunk_overlap,
    extract_keywords,
    first_sentence,
    html_table_to_markdown,
    rows_to_markdown,
    split_markdown_table,
)


# --------------------------------------------------------------- markdown ---
def test_html_table_to_markdown_escapes_pipes_and_pads_rows():
    html = ("<table><tr><th>Region</th><th>Q1</th></tr>"
            "<tr><td>North | inc</td><td>10</td></tr>"
            "<tr><td>South</td></tr></table>")
    md = html_table_to_markdown(html)
    lines = md.split("\n")
    assert lines[1] == "| --- | --- |"
    assert r"North \| inc" in lines[2]
    # short row padded to full width, so every row has the same column count
    assert lines[3].count("|") == lines[0].count("|")


def test_rows_to_markdown_handles_none_cells():
    md = rows_to_markdown([["A", None, "C"], [None, "b", None]])
    assert md.split("\n")[0] == "| A |  | C |"
    assert md.split("\n")[1] == "| --- | --- | --- |"


def test_rows_to_markdown_empty():
    assert rows_to_markdown([]) == ""
    assert rows_to_markdown([[None, ""]]) == ""


# ------------------------------------------------------------ table split ---
def _big_table(n=200):
    html = "<table><tr><th>Name</th><th>Description</th></tr>" + "".join(
        f"<tr><td>item-{i:03d}</td><td>{'x' * 120}</td></tr>" for i in range(n)
    ) + "</table>"
    return html_table_to_markdown(html)


def test_oversized_table_splits_within_limit_and_repeats_header():
    parts = split_markdown_table(_big_table(), caption="Table 3: Inventory")
    assert len(parts) > 1
    for p in parts:
        assert len(p) <= HARD_LIMIT
        lines = p.split("\n")
        assert p.startswith("Table 3: Inventory (part")
        assert lines[2] == "| Name | Description |"      # header repeated
        assert lines[3].startswith("| --- |")


def test_table_splits_share_rows_as_overlap():
    parts = split_markdown_table(_big_table(), caption="T")
    rows = lambda p: [l for l in p.split("\n")[4:] if l.strip()]
    for a, b in zip(parts, parts[1:]):
        shared = [r for r in rows(a) if r in rows(b)]
        assert shared, "consecutive splits must share trailing rows"
        # overlap rounds DOWN to whole rows; splitting mid-row would corrupt it
        assert sum(len(r) + 1 for r in shared) <= OVERLAP


def test_small_table_is_not_split_and_caption_leads():
    md = html_table_to_markdown("<table><tr><th>A</th></tr><tr><td>1</td></tr></table>")
    parts = split_markdown_table(md, caption="Table 1: Small")
    assert len(parts) == 1
    assert parts[0].startswith("Table 1: Small\n\n| A |")


def test_table_without_caption_starts_with_the_header():
    parts = split_markdown_table(_big_table())
    assert parts[0].startswith("| Name |")


# ------------------------------------------------------------ boilerplate ---
def test_bp_key_unifies_page_numbers():
    assert _bp_key("Maxim Integrated | 4") == _bp_key("Maxim Integrated | 37")
    assert _bp_key("  Spaced   Out  ") == "spaced out"


def test_bp_key_distinguishes_real_content():
    assert _bp_key("Absolute Maximum Ratings") != _bp_key("Electrical Characteristics")


# --------------------------------------------------------------- keywords ---
def test_extract_keywords_drops_stopwords_and_short_tokens():
    kw = extract_keywords("The Revenue for the FY2024 quarter was 12.5% above it",
                          {"the", "for", "was", "it", "above"})
    assert "revenue" in kw and "fy2024" in kw
    assert "the" not in kw and "was" not in kw


def test_extract_keywords_preserves_first_appearance_order():
    kw = extract_keywords("beta alpha beta gamma", set())
    assert kw == ["beta", "alpha", "gamma"]


# -------------------------------------------------------- caption trimming ---
def test_first_sentence_trims_a_long_caption():
    long = "This is the opening claim. " + "And then much more text. " * 40
    out = first_sentence(long)
    assert out == "This is the opening claim."
    assert len(out) < len(long)


def test_first_sentence_on_empty_input():
    assert first_sentence("") == ""


# ---------------------------------------------------------------- overlap ---
def test_cross_chunk_overlap_only_touches_image_and_first_table_part():
    chunks = [
        Chunk("a", "text", "T" * 2000, {"page_number": 1}),
        Chunk("b", "image", "A figure.", {"page_number": 1}),
        Chunk("c", "table", "T2\n\n| A |\n| --- |\n| 1 |",
              {"page_number": 2, "table_part": 1}),
        Chunk("d", "table", "T2\n\n| A |\n| --- |\n| 2 |",
              {"page_number": 2, "table_part": 2}),
    ]
    apply_cross_chunk_overlap(chunks)

    assert chunks[0].text == "T" * 2000                       # first chunk untouched
    assert chunks[1].metadata["overlap_prefix_chars"] == OVERLAP
    assert chunks[1].text.startswith("T" * OVERLAP)
    assert chunks[1].metadata["core_text"] == "A figure."     # original recoverable
    # part 2 of a table already carries row-level overlap of its own
    assert "overlap_prefix_chars" not in chunks[3].metadata


def test_cross_chunk_overlap_prefix_matches_previous_tail_exactly():
    chunks = [
        Chunk("a", "text", "abcdefghij" * 200, {"page_number": 1}),
        Chunk("b", "image", "Figure.", {"page_number": 1}),
    ]
    apply_cross_chunk_overlap(chunks)
    k = chunks[1].metadata["overlap_prefix_chars"]
    assert chunks[1].text[:k] == chunks[0].text[-OVERLAP:].strip()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
