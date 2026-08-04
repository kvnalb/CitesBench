"""
Load prompt templates from prompts/ so no prompt text lives inside a script.

    from prompts import load
    load("recall/lap_cot", title=t, year=2019)

File convention: the template is the file's contents with exactly one trailing
newline stripped (every prompt we send ends mid-line, not with a blank line).
Placeholders are `{name}`; substitution is a plain string replace, NOT
str.format — abstracts and titles contain braces and dollar signs (LaTeX), and
.format would raise or mangle them. That also means a template can contain a
literal `{"score": ...}` JSON example without escaping.

Self-check: python src/prompts.py
"""
import os
import glob

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")


def load(name, **kw):
    """Render prompts/<name>.txt. Every {placeholder} must be supplied."""
    path = os.path.join(ROOT, name if name.endswith(".txt") else name + ".txt")
    with open(path) as f:
        text = f.read()
    if text.endswith("\n"):
        text = text[:-1]
    for k, v in kw.items():
        text = text.replace("{" + k + "}", str(v))
    return text


def demo():
    import re
    filled = {
        "recall/lap_oneword": dict(title="T", year=2019),
        "recall/fame_oneword": dict(title="T", year=2019),
        "recall/lap_cot": dict(title="T", year=2019),
        "recall/fame_cot": dict(title="T", year=2019),
        "recall/abstract_completion": dict(title="T", year=2019, first_sentence="S."),
        "review/iclr_review_calibrated": dict(year=2019),
        "review/paraphrase_abstract": dict(abstract="A"),
        "review/title_abstract_body": dict(title="T", abstract="A"),
        "review/abstract_only_body": dict(abstract="A"),
    }
    for name, kw in filled.items():
        out = load(name, **kw)
        # a leftover {word} means a placeholder the caller never supplied
        left = [m for m in re.findall(r"\{([a-z_]+)\}", out)]
        assert not left, f"{name}: unfilled placeholders {left}"
        assert out and not out.endswith("\n"), f"{name}: bad trailing newline"

    # braces and $ in interpolated values must survive untouched
    tricky = "we set $\\alpha$ and {beta} to 0.1"
    assert tricky in load("review/abstract_only_body", abstract=tricky)

    # anything not listed above is a system prompt taking no arguments — it must still load
    files = sorted(glob.glob(os.path.join(ROOT, "**", "*.txt"), recursive=True))
    for p in files:
        load(os.path.relpath(p, ROOT)[:-4])
    print(f"ok — {len(files)} templates load, placeholders resolve, braces/$ pass through")


if __name__ == "__main__":
    demo()
