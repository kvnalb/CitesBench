"""
Can ReviewArena's markdown stand in for the archive's PDF extraction? (issue #9)

The 2018-2020 reviews were generated from Docling-extracted PDF text. A 2025 run would
use ReviewArena's `markdown` column, which is OCR'd text of visibly different character.
If the text source alone moves the score, a 2025 number is not comparable to a
2018-2020 number, and no amount of pipeline fidelity fixes that.

The archive's own extracted text is NOT in this repo — all_paper_results.csv points at
an external share and none of those files exist locally — so a same-paper text diff is
impossible. What IS possible, and is what this script does:

  1. Apply the archive's own acceptance criteria to ReviewArena text. The archive
     shipped both: garble_ratio() (docstring: >~0.005 means OCR trouble) and
     _check_extraction_quality() (fails on no sections, or <500 chars of section text).
     Text that fails the gate the archive applied to its own inputs is disqualified.

  2. Compare ReviewArena 2020 against ReviewArena 2025. 2020 is the year the archive
     also processed, so it separates "ReviewArena is noisy" from "2025 differs from
     2020" — the latter would be a confound even if the source were perfect.

No LLM calls. Every measure here is deterministic and reproducible.

Section parsing deliberately imports the SAME functions the pipeline uses, so what is
measured is what the pipeline would actually see — not a reimplementation that could
agree with the pipeline by accident or disagree with it silently.

Outputs:
  outputs/reviewarena_text_audit.csv   one row per paper, both years
  outputs/reviewarena_text_audit.md    summary tables for the docs/ report

Run: python src/audit/audit_reviewarena_text.py
     python src/audit/audit_reviewarena_text.py --sample 400   # faster pass
"""
import os
import re
import sys
import argparse

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from build.build_slim_2025_papers import load_year
from build.normalize_paper_markdown import normalize
from probes.slim_pipeline import (
    PaperText, SectionType, _heuristic_structure, _parse_sections_from_markdown,
    _method_sections, _intro_sections, _combine_sections,
)

OUT_CSV = "outputs/reviewarena_text_audit.csv"
OUT_MD = "outputs/reviewarena_text_audit.md"
YEARS = (2020, 2025)

# Verbatim from Archive/CompletePipeline/llm/coarse/garble.py — the archive's own OCR
# detector. Copied rather than imported because importing coarse/ drags in litellm.
GARBLE_CHARS = re.compile(
    r"[®õÈÀÁ�￾￿]"
    r"|/C[0-9]{2}"
    r"|glyph\[\w+\]"
    r"|/lscript"
)
GARBLE_THRESHOLD = 0.005          # garble.py docstring
MIN_SECTION_CHARS = 500           # pipeline.py:_check_extraction_quality

# sections the slim pipeline actually consumes; a missing methodology section silently
# costs a call, which is the failure mode this audit exists to make visible
NEEDED = (SectionType.INTRODUCTION, SectionType.METHODOLOGY, SectionType.CONCLUSION)


def garble_ratio(text):
    if not text:
        return 0.0
    return sum(len(m) for m in GARBLE_CHARS.findall(text)) / max(len(text), 1)


def measure(md):
    """Everything we can say about one paper's text without calling a model."""
    raw_sections = _parse_sections_from_markdown(md)
    norm = normalize(md)
    structure = _heuristic_structure(PaperText(full_markdown=norm,
                                               token_estimate=len(norm) // 4))
    sections = structure.sections
    typed = [s for s in sections if s.section_type != SectionType.OTHER]
    section_chars = sum(len(s.text) for s in sections)
    kinds = {s.section_type for s in sections}

    non_ascii = sum(1 for c in md if ord(c) > 127)
    return {
        "chars": len(md),
        "garble_ratio": round(garble_ratio(md), 6),
        "garble_over_threshold": garble_ratio(md) > GARBLE_THRESHOLD,
        "non_ascii_rate": round(non_ascii / max(len(md), 1), 5),
        "sections_raw": len(raw_sections),          # before normalize(): what the
        "sections_normalized": len(sections),       # pipeline would have seen unaided
        "sections_typed": len(typed),
        "section_chars": section_chars,
        # the archive's gate, applied exactly as pipeline.py applies it
        "passes_extraction_gate": bool(sections) and section_chars >= MIN_SECTION_CHARS,
        "has_intro": SectionType.INTRODUCTION in kinds,
        "has_method": SectionType.METHODOLOGY in kinds,
        "has_conclusion": SectionType.CONCLUSION in kinds,
        "has_all_needed": all(k in kinds for k in NEEDED),
        # What the pipeline ACTUALLY does, which is not the same as section typing:
        # _method_sections falls back from typed METHODOLOGY -> title-keyword match ->
        # any non-boilerplate section. The method call is skipped only when that whole
        # chain yields nothing, which is why all 4,497 archived papers show 8 calls.
        # `method_call_fires` is the metric that governs comparability; `has_method`
        # only describes how cleanly the section was labelled.
        "method_call_fires": bool(_combine_sections(_method_sections(structure), 6000).strip()),
        "method_from_typed": SectionType.METHODOLOGY in kinds,
        "intro_call_fires": bool(_combine_sections(_intro_sections(structure), 3500).strip()),
        "has_references": bool(re.search(r"^\s*#*\s*REFERENCES\b", md,
                                         re.M | re.I)),
        "hyphen_breaks": len(re.findall(r"\w-\s\w", md)),   # "regular- ization"
        "title_len": len(structure.title or ""),
        "abstract_len": len(structure.abstract or ""),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0,
                    help="papers per year (0 = all); seeded, for a faster pass")
    args = ap.parse_args()
    os.makedirs("outputs", exist_ok=True)

    rows = []
    for year in YEARS:
        d = load_year(year)
        if args.sample:
            d = d.sample(min(args.sample, len(d)), random_state=42)
        print(f"{year}: {len(d)} papers", flush=True)
        for r in d.itertuples(index=False):
            rec = {"paper_id": r.forum_id, "year": year, "decision": r.decision}
            rec.update(measure(r.markdown))
            rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    lines = ["# ReviewArena text audit (issue #9)", "",
             f"{len(out)} papers, no LLM calls. "
             f"Archive thresholds: garble > {GARBLE_THRESHOLD}, "
             f"section text >= {MIN_SECTION_CHARS} chars.", ""]

    g = out.groupby("year")
    summary = pd.DataFrame({
        "papers": g.size(),
        "median_chars": g.chars.median().astype(int),
        "median_garble": g.garble_ratio.median(),
        "pct_garble_over_thresh": (g.garble_over_threshold.mean() * 100).round(1),
        "median_sections_raw": g.sections_raw.median(),
        "median_sections_norm": g.sections_normalized.median(),
        "pct_zero_sections_raw": (out.assign(z=out.sections_raw == 0)
                                  .groupby("year").z.mean() * 100).round(1),
        "pct_passes_gate": (g.passes_extraction_gate.mean() * 100).round(1),
        "pct_has_intro": (g.has_intro.mean() * 100).round(1),
        "pct_has_method": (g.has_method.mean() * 100).round(1),
        "pct_has_conclusion": (g.has_conclusion.mean() * 100).round(1),
        "pct_has_all_needed": (g.has_all_needed.mean() * 100).round(1),
        "pct_has_references": (g.has_references.mean() * 100).round(1),
        # the two that decide whether every paper gets the same number of LLM calls
        "pct_method_call_fires": (g.method_call_fires.mean() * 100).round(1),
        "pct_intro_call_fires": (g.intro_call_fires.mean() * 100).round(1),
    }).T
    lines += ["## 2020 vs 2025", "", summary.to_markdown(), ""]

    print(summary.to_string())
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {OUT_CSV} and {OUT_MD}")


if __name__ == "__main__":
    main()
