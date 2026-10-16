#!/usr/bin/env bash
# Render the design memos to PDF.
#
# Quarto is not installed here, and the .qmd files are plain markdown with YAML
# front matter, so pandoc renders them directly. TinyTeX is installed but not on
# PATH, which is why it is added below rather than assumed.
set -euo pipefail
export PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH"
cd "$(dirname "$0")"
for f in *.qmd notes/*.qmd; do
  [ -e "$f" ] || continue
  pandoc "$f" -o "${f%.qmd}.pdf" --pdf-engine=pdflatex \
    -V geometry:margin=1in -V fontsize=11pt --resource-path=.:..
  echo "-> docs/${f%.qmd}.pdf"
done
