#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

for tex in report.tex imputation_methods_note.tex; do
  pdflatex -interaction=nonstopmode -halt-on-error "$tex"
  pdflatex -interaction=nonstopmode -halt-on-error "$tex"
done
