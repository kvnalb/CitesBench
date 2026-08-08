#!/usr/bin/env python3
"""
Barebones pipeline:
- Read PDFs from a path (single file or a directory)
- Extract text using PyMuPDF
- Load and format prompt
- Call selected OpenAI model
- Save results as JSON next to processed/evaluation/

Usage examples:
  python barebones.py --api-key $OPENAI_API_KEY --pdf rawdata/Bloomfield.pdf \
    --model o1-preview --prompt evaluation_prompt

  python barebones.py --api-key $OPENAI_API_KEY --pdf-dir rawdata \
    --model o1-preview --prompt evaluation_prompt
"""

import os
import sys
import json
from pathlib import Path
from typing import Iterable, Optional

from simple_extractor import PDFExtractor, PaperInfo
from simple_evaluator import PaperEvaluator, PaperEvaluation
from prompt_loader import PromptLoader


def iter_pdfs(pdf: Optional[str], pdf_dir: Optional[str]) -> Iterable[Path]:
    if pdf:
        p = Path(pdf)
        if not p.exists():
            raise FileNotFoundError(f"PDF not found: {p}")
        if p.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {p}")
        yield p
        return
    if pdf_dir:
        d = Path(pdf_dir)
        if not d.exists() or not d.is_dir():
            raise FileNotFoundError(f"Directory not found: {d}")
        for file in sorted(d.glob("*.pdf")):
            yield file
        return
    raise ValueError("Provide either --pdf or --pdf-dir")


def evaluate_one(evaluator: PaperEvaluator, extractor: PDFExtractor, prompt_loader: PromptLoader,
                 pdf_path: Path, model: str, prompt_name: str, out_dir: Path) -> Path:
    paper: PaperInfo = extractor.extract_paper(str(pdf_path))
    formatted_prompt = prompt_loader.format_prompt(
        prompt_name,
        paper_title=paper.title,
        paper_content=paper.text[:8000]
    )
    evaluation: PaperEvaluation = evaluator.evaluate_paper(
        paper_text=paper.text,
        paper_title=paper.title,
        model=model,
        custom_prompt=formatted_prompt
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return evaluator.save_evaluation(evaluation, str(pdf_path), str(out_dir))


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Barebones PDF evaluator")
    parser.add_argument("--api-key", dest="api_key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--pdf", dest="pdf", default=None, help="Path to a single PDF")
    parser.add_argument("--pdf-dir", dest="pdf_dir", default=None, help="Directory with PDFs")
    parser.add_argument("--model", dest="model", default="o1-preview")
    parser.add_argument("--prompt", dest="prompt", default="evaluation_prompt")
    parser.add_argument("--out", dest="out_dir", default="processed/evaluation")
    args = parser.parse_args(argv)

    if not args.api_key:
        print("OPENAI_API_KEY not provided. Use --api-key or set env var.")
        return 2

    extractor = PDFExtractor()
    evaluator = PaperEvaluator(args.api_key)
    prompt_loader = PromptLoader("prompts")
    out_dir = Path(args.out_dir)

    errors = 0
    for pdf_path in iter_pdfs(args.pdf, args.pdf_dir):
        try:
            print(f"Processing: {pdf_path}")
            out_file = evaluate_one(evaluator, extractor, prompt_loader, pdf_path, args.model, args.prompt, out_dir)
            print(f"Saved: {out_file}")
        except Exception as e:
            errors += 1
            print(f"Error processing {pdf_path}: {e}")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


