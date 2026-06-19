#!/usr/bin/env bash
# Render the Beamer PDF and move all LaTeX intermediates into Latex/.
# Leaves the revealjs HTML next to slides.qmd untouched.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

mkdir -p Latex

quarto render slides.qmd --to beamer

# quarto emits slides.pdf, slides.tex, and intermediate .aux/.log/.out
for f in slides.pdf slides.tex slides.aux slides.log slides.out slides.snm slides.nav slides.toc; do
  [[ -f "$f" ]] && mv -f "$f" Latex/ || true
done

echo
echo "Built Latex/slides.pdf"
