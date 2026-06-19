# Presentation

Quarto slide draft for the RDD + LLM review pipeline talk.

## Files

- `slides.qmd` — revealjs slide deck, outline:
  1. RDD sample motivation + description + selection
  2. The new LLM pipeline
  3. Prompts and outputs at key stages
  4. Overall performance (3-model confusion matrix)
  5. Breakdown vs. human eval (similarity + stage-level)
  6. Back to the RDD and implications

## Render

HTML (revealjs):

```bash
quarto render Report/RDD_Coarse/Presentation/slides.qmd
```

Output → `slides.html` next to `slides.qmd`.

PDF (Beamer, via XeLaTeX):

```bash
bash Report/RDD_Coarse/Presentation/build-pdf.sh
```

Output → `Latex/slides.pdf` (and `slides.tex` alongside it).
The script renders with `--to beamer`, then moves the `.pdf`, `.tex`,
and LaTeX intermediates into the `Latex/` subfolder.

Requires XeLaTeX (TeX Live) for Unicode glyphs and Helvetica / Menlo fonts.

## Assets

Plots and tables are referenced by relative path from the sibling
`../plots/` and `../tables/` directories; the tiebreaker experiment
figures come from `../../../Playground/fuzzy_rdd_llm_tiebreaker/`. The
deck does not copy them — it links in place, so re-running the scripts
in `../Code/` refreshes the slides without any edits here.
