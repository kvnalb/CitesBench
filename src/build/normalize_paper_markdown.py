"""
Make ReviewArena text look like the text the 2018-2020 pipeline was fed.

The archive ran on plain extracted text with no markdown headings. The traces prove it:
production intro/method prompts contain the literal line `## 1. Full Document [other]`,
meaning `_parse_sections_from_markdown` found no `#` headings, returned the document as
one untyped section, and every review stage worked from that blob truncated to the
per-stage character budgets. Meanwhile the structural inventory — which matches bare
lines like `4 EXPERIMENTS` — read correctly.

ReviewArena text is nearly the same shape, with one difference that matters. About 30%
of 2025 papers contain at least one stray `#` line: OCR of table headers such as
`# layers`, `# queries`, `# Parameters` (i.e. "number of layers"), or fragments of
prompt templates like `# Instruction:`. Those are not section headings, but the parser
cannot tell. A single one flips a paper from "one Full Document section" to "one section
starting at that line" — and text before the first heading belongs to no section at all.
For papers with exactly one stray marker the median position is 77% of the way through,
so roughly three quarters of the paper would be dropped, silently, with a normal-looking
review produced from the remainder.

So: strip the marker, keep the words. Every paper then parses to a single Full Document
section, exactly as in 2018-2020, and the inventory still sees the bare heading lines.

An earlier version of this module did the opposite — it promoted detected headings TO
`## `, so the parser would split papers into sections. That produced better-structured
prompts than production ever had, on top of blinding the inventory, and would have made
2025 a different instrument. Inverted deliberately; see the comments in
src/probes/slim_pipeline.py.

ponytail: one regex over line starts. If ReviewArena ever ships real markdown, this
becomes wrong rather than unnecessary — check the parsed section count, not this file.
"""
import re

# a leading markdown heading marker, and only that: the line's text is preserved
HEADING_MARKER = re.compile(r"^\s{0,3}#{1,6}\s+", re.M)


def to_archive_text(md):
    """ReviewArena markdown -> plain text shaped like the archive's extracted text."""
    return HEADING_MARKER.sub("", md)


def demo():
    raw = "\n".join([
        "Published as a conference paper at ICLR 2025",
        "ABSTRACT",
        "We study dropout in graph convolutional networks.",
        "1 INTRODUCTION",
        "Dropout is poorly understood in this setting.",
        "# graphs",                 # OCR of a table header: "number of graphs"
        "Table 2 reports results.",
        "## Instruction:",          # fragment of a quoted prompt template
        "Answer the question.",
    ])
    out = to_archive_text(raw)

    # markers gone, words kept
    assert "#" not in out, out
    assert "graphs" in out and "Instruction:" in out
    # bare heading lines untouched, so the structural inventory still sees them
    assert "1 INTRODUCTION" in out and "ABSTRACT" in out
    # nothing else moved: same line count, same order
    assert len(out.split("\n")) == len(raw.split("\n"))
    assert out.split("\n")[0] == raw.split("\n")[0]

    # the property that matters: no line can start a new section
    import re as _re
    assert not _re.search(r"^(#{1,4})\s+(.+)$", out, _re.M)
    print("ok")


if __name__ == "__main__":
    demo()
