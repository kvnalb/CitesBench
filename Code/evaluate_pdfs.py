#!/usr/bin/env python3
"""
Single-file minimal evaluator:
- Read PDFs
- Extract text
- Load & format prompt
- Call model (gpt-5 or other)
- Save JSON results

Usage: import this file and run the step-by-step examples at bottom (commented).
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF
from openai import OpenAI


# =============================================================================
# CORE FUNCTIONS (keep minimal)
# =============================================================================

def list_pdfs(directory: str = "rawdata") -> List[Path]:
    pdf_dir = Path(directory)
    return sorted(pdf_dir.glob("*.pdf"))


def extract_pdf_text(pdf_path: str) -> Dict[str, Any]:
    doc = fitz.open(pdf_path)

    # --- Pass 1: structured extraction with font sizes to find headings ---
    structured_lines: List[Dict[str, Any]] = []
    font_sizes: List[float] = []
    for i in range(doc.page_count):
        d = doc[i].get_text("dict")
        for block in d.get("blocks", []):
            if block.get("type", 0) != 0:
                continue  # non-text
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(span.get("text", "") for span in spans).strip()
                if not text:
                    continue
                # average font size across spans
                sizes = [float(span.get("size", 0)) for span in spans if span.get("size")]
                size = sum(sizes) / len(sizes) if sizes else 0.0
                font_sizes.append(size)
                structured_lines.append({"text": text, "size": size})

    # fallback if no structured lines
    if not structured_lines:
        page_texts: List[str] = []
        for i in range(doc.page_count):
            page_texts.append(doc[i].get_text())
        doc.close()
        fallback = "\n".join(page_texts).strip()
        return {"title": fallback.splitlines()[0] if fallback else "", "abstract": "", "text": fallback}

    # compute size threshold ~90th percentile without external deps
    def percentile(values: List[float], p: float) -> float:
        if not values:
            return 0.0
        vs = sorted(values)
        k = max(0, min(len(vs) - 1, int(round((p / 100.0) * (len(vs) - 1)))))
        return vs[k]

    size_thresh = percentile(font_sizes, 90.0)

    def is_heading(text: str, size: float) -> bool:
        t = text.strip()
        low = t.lower()
        if size >= size_thresh:
            return True
        if len(t) <= 40 and (t.isupper() or t.endswith(":")):
            return True
        canonical = low.replace(":", "").strip()
        if canonical in {"abstract", "introduction", "references", "bibliography", "works cited", "appendix", "appendices"}:
            return True
        return False

    def looks_like_caption(s: str) -> bool:
        low = s.lower().strip()
        if low.startswith(("figure", "fig.", "table", "exhibit")):
            return True
        if "|" in s:
            return True
        digits = sum(ch.isdigit() for ch in s)
        if len(s) > 0 and digits / max(len(s), 1) > 0.5:
            return True
        return False

    # Identify reference boundary (first heading matching references/bibliography/etc.)
    ref_idx: Optional[int] = None
    abstract_idx: Optional[int] = None
    for idx, item in enumerate(structured_lines):
        t = item["text"].strip()
        low = t.lower().strip().replace(":", "")
        if abstract_idx is None and (low.startswith("abstract") or low == "abstract") and is_heading(t, item["size"]):
            abstract_idx = idx
        if ref_idx is None and low in {"references", "bibliography", "works cited", "appendix", "appendices"} and is_heading(t, item["size"]):
            ref_idx = idx
            break

    # Title = first non-empty non-caption line near top
    title = ""
    for item in structured_lines[:30]:
        t = item["text"].strip()
        if t and len(t) > 10 and not looks_like_caption(t):
            title = t
            break

    # Build abstract
    abstract_parts: List[str] = []
    if abstract_idx is not None:
        # collect until next heading or reference boundary
        for j in range(abstract_idx + 1, len(structured_lines)):
            if ref_idx is not None and j >= ref_idx:
                break
            t = structured_lines[j]["text"].strip()
            if is_heading(t, structured_lines[j]["size"]):
                break
            if looks_like_caption(t):
                continue
            abstract_parts.append(t)

    # Build main text up to references
    main_parts: List[str] = []
    start_idx = 0
    if abstract_idx is not None:
        # start after abstract section end
        k = abstract_idx + 1
        while k < len(structured_lines):
            t = structured_lines[k]["text"].strip()
            if is_heading(t, structured_lines[k]["size"]):
                start_idx = k + 1
                break
            k += 1
    for j in range(start_idx, len(structured_lines)):
        if ref_idx is not None and j >= ref_idx:
            break
        t = structured_lines[j]["text"].strip()
        if looks_like_caption(t):
            continue
        main_parts.append(t)

    abstract = " ".join(abstract_parts).strip()
    main_text = "\n".join(main_parts).strip()

    doc.close()
    return {"title": title, "abstract": abstract, "text": main_text}


def load_prompt(prompt_file: str) -> str:
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()


def format_prompt(template: str, paper_title: str, paper_content: str) -> str:
    return template.format(
        paper_title=paper_title,
        paper_content=paper_content
    )


def call_model(api_key: str, model: str, prompt: str, base_url: str = None) -> str:
    """
    Call an LLM model for inference.

    Args:
        api_key: API key for authentication
        model: Model name to use
        prompt: The prompt text
        base_url: Optional base URL for D-LLM or other OpenAI-compatible APIs
    """
    if base_url:
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=api_key)

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content


def parse_json_output(output: str) -> Dict[str, Any]:
    s = output.strip()
    if s.startswith("```json"):
        s = s[7:]
    if s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    s = s.strip()
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        data = {
            "overall_summary": s,
            "overall_score": 0,
            "criteria_scores": {},
            "strengths": [],
            "weaknesses": [],
            "recommendations": []
        }
    return data


def save_evaluation(evaluation: Dict[str, Any], pdf_path: str, prompt_file: str, out_dir: str = "processed/evaluation") -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    pdf_stem = Path(pdf_path).stem
    prompt_stem = Path(prompt_file).stem
    out_file = out_path / f"{pdf_stem}_{prompt_stem}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)
    return out_file


# =============================================================================
# STEP-BY-STEP (examples; run these lines manually as needed)
# =============================================================================

if __name__ == "__main__":
    print("Loaded minimal evaluator. Set OPENAI_API_KEY and use the examples below.")

    # Examples (copy/paste lines into your REPL):
    #
 