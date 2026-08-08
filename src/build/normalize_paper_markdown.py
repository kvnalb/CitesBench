"""
Turn ReviewArena's `markdown` column into text that actually has markdown headings.

The column is misnamed: it is OCR'd PDF text, not markdown. Section titles arrive as
bare uppercase lines ("INTRODUCTION") or numbered ones ("4 EXPERIMENTS"), and across
200 sampled 2025 papers the median count of real `#` headings is ZERO.

That matters because the slim pipeline's section parser splits on `^#{1,4}\\s+`. Fed
the raw column it returns a single untyped "Full Document" section, which makes the
methodology stage silently skip (no method section found) and the introduction stage
fall back to the abstract. The pipeline would still emit reviews — it would just be
reviewing an abstract and a blob, and nothing in the output would say so.

So: promote detected headings to `## `, leave every other line alone, and let the
existing parser work unmodified. Detection is deliberately conservative — OCR noise
puts plenty of short uppercase junk on its own line (table cells like "GRIT", "SANE",
garbled fragments like "CIFARIO"), and a false heading fragments a section, which is
worse than missing one.

Two rules, both required to fire on a line on its own:
  numbered   "3 THEORETICAL FRAMEWORK", "4.1 DATASETS AND SETUP"  -> any numbered title
  keyword    "INTRODUCTION", "RELATED WORK", "CONCLUSIONS"        -> known section names

Measured on 400 papers: median 18 headings each, zero papers with none.

ponytail: regex over the known ICLR section vocabulary, not a layout model. If a
paper's sections stop being found, widen KEYWORDS before reaching for anything bigger.
"""
import re

# a numbered heading: "3 FOO", "4.1 FOO BAR", "2. Foo" — title must start with a capital
NUMBERED = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z][A-Za-z].{2,60}$")

# unnumbered headings we trust by name; trailing words allowed ("EXPERIMENTAL SETUP")
KEYWORDS = re.compile(
    r"^(ABSTRACT|INTRODUCTION|RELATED WORKS?|BACKGROUND|PRELIMINARIES|"
    r"METHOD(S|OLOGY)?|APPROACH|EXPERIMENTS?|EXPERIMENTAL SETUP|RESULTS|"
    r"EVALUATION|ANALYSIS|ABLATIONS?|DISCUSSIONS?|LIMITATIONS|CONCLUSIONS?|"
    r"REFERENCES|ACKNOWLEDGE?MENTS?|APPENDI(X|CES))\b.{0,40}$"
)


def is_heading(line):
    s = line.strip()
    return bool(NUMBERED.match(s) or KEYWORDS.match(s))


def normalize(md):
    """OCR'd paper text -> same text with section titles promoted to '## ' headings."""
    out = []
    for line in md.split("\n"):
        out.append(f"## {line.strip()}" if is_heading(line) else line)
    return "\n".join(out)


def demo():
    raw = "\n".join([
        "Published as a conference paper at ICLR 2025",
        "ABSTRACT",
        "Graph Convolutional Networks have emerged as powerful tools.",
        "1 INTRODUCTION",
        "Dropout is poorly understood in this setting.",
        "3.2 DIMENSION-SPECIFIC SUBGRAPHS",
        "We define the sampling procedure.",
        "4 EXPERIMENTS",
        "GRIT",          # OCR noise: a table cell, must NOT become a heading
        "SANE",
        "5 CONCLUSIONS",
        "We conclude.",
    ])
    got = [l for l in normalize(raw).split("\n") if l.startswith("## ")]
    assert got == ["## ABSTRACT", "## 1 INTRODUCTION", "## 3.2 DIMENSION-SPECIFIC SUBGRAPHS",
                   "## 4 EXPERIMENTS", "## 5 CONCLUSIONS"], got
    # noise stayed put, and the paper's own prose was not touched
    assert "GRIT" in normalize(raw).split("\n")
    assert "## GRIT" not in normalize(raw)
    assert "Published as a conference paper at ICLR 2025" in normalize(raw)
    print("ok")


if __name__ == "__main__":
    demo()
