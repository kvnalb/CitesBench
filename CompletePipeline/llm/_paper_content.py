#!/usr/bin/env python3
"""
Shared paper-content selection utilities for pointwise and pairwise runs.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


SECTION_ALIAS_GROUPS = {
    "introduction": ("introduction", "overview"),
    "methods": (
        "method",
        "methods",
        "approach",
        "approaches",
        "methodology",
        "framework",
        "frameworks",
        "algorithm",
        "algorithms",
        "implementation",
        "setup",
    ),
    "results": (
        "experiments",
        "experiment",
        "evaluation",
        "evaluations",
        "results",
        "analysis",
        "ablation",
        "ablations",
        "findings",
    ),
    "discussion": (
        "discussion",
        "limitations",
        "limitation",
        "broader impact",
    ),
    "conclusion": (
        "conclusion",
        "conclusions",
        "future work",
    ),
}
IGNORE_SECTION_ALIASES = {
    "abstract",
    "references",
    "appendix",
    "appendices",
    "acknowledgements",
    "acknowledgments",
}
CORE_SECTION_ORDER = ("introduction", "methods", "results", "discussion", "conclusion")
SECTION_MATCH_PRIORITY = ("introduction", "methods", "results", "conclusion", "discussion")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def infer_fulltext_dir(input_path: Path) -> Path | None:
    candidate = input_path.resolve().parent / "fulltext"
    return candidate if candidate.is_dir() else None


def get_fulltext_path(fulltext_dir: Path | None, paper_id: str) -> Path | None:
    if fulltext_dir is None:
        return None
    path = fulltext_dir / f"{paper_id}.txt"
    return path if path.exists() else None


def normalise_heading(line: str) -> str:
    candidate = line.replace("\u00a0", " ").strip()
    candidate = re.sub(r"^[0-9]+\s*$", "", candidate)
    candidate = re.sub(r"^(?:section\s+)?(?:[0-9]+|[ivxlcdm]+)[.)]?\s+", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"[^a-zA-Z0-9/& -]+", " ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip().lower()
    return candidate


def is_likely_top_level_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) > 90:
        return False
    if len(stripped.split()) > 10:
        return False
    if re.fullmatch(r"[0-9]+", stripped):
        return False
    if re.match(r"^(?:[0-9]+|[IVXLCDM]+)[.)]?\s+[A-Za-z]", stripped):
        return True
    letters = [ch for ch in stripped if ch.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
    return upper_ratio >= 0.75


def canonical_section_kind(line: str) -> str | None:
    if not is_likely_top_level_heading(line):
        return None
    normalized = normalise_heading(line)
    if not normalized:
        return None
    if normalized in IGNORE_SECTION_ALIASES:
        return normalized
    for kind in SECTION_MATCH_PRIORITY:
        aliases = SECTION_ALIAS_GROUPS[kind]
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", normalized):
                return kind
    return None


def extract_named_sections(full_text: str) -> list[dict[str, str]]:
    lines = full_text.splitlines()
    heading_indices: list[tuple[int, str, str]] = []
    for idx, line in enumerate(lines):
        kind = canonical_section_kind(line)
        if kind is not None:
            heading_indices.append((idx, kind, line.strip()))

    sections: list[dict[str, str]] = []
    for pos, (start_idx, kind, heading_text) in enumerate(heading_indices):
        end_idx = heading_indices[pos + 1][0] if pos + 1 < len(heading_indices) else len(lines)
        body_lines = []
        for line in lines[start_idx + 1 : end_idx]:
            stripped = line.strip()
            if not stripped:
                body_lines.append("")
                continue
            if stripped.startswith("Published as a conference paper at ICLR"):
                continue
            if stripped.startswith("Under review as a conference paper at ICLR"):
                continue
            if re.fullmatch(r"[0-9]+", stripped):
                continue
            body_lines.append(stripped)

        body = "\n".join(body_lines).strip()
        sections.append(
            {
                "kind": kind,
                "heading": heading_text,
                "normalized_heading": normalise_heading(heading_text),
                "text": body,
            }
        )
    return sections


def build_core_section_excerpt(
    abstract: str,
    full_text: str,
    max_content_chars: int,
    section_char_limit: int,
) -> dict | None:
    sections = extract_named_sections(full_text)
    selected_chunks: list[str] = []
    selected_sections: list[dict[str, int | str]] = []
    first_intro_idx = next((idx for idx, section in enumerate(sections) if section["kind"] == "introduction"), None)

    abstract_block = abstract.strip()
    if abstract_block:
        clipped_abstract = abstract_block[: min(section_char_limit, max_content_chars)]
        selected_chunks.append(f"ABSTRACT\n{clipped_abstract}")
        selected_sections.append(
            {
                "kind": "abstract",
                "heading": "ABSTRACT",
                "char_count_total": len(abstract_block),
                "char_count_used": len(clipped_abstract),
            }
        )

    used_kinds: set[str] = set()
    total_chars = sum(len(chunk) for chunk in selected_chunks)
    min_section_idx = first_intro_idx if first_intro_idx is not None else 0
    for desired_kind in CORE_SECTION_ORDER:
        match = None
        match_idx = None
        for idx, section in enumerate(sections[min_section_idx:], start=min_section_idx):
            if section["kind"] != desired_kind:
                continue
            if desired_kind in used_kinds:
                continue
            text = section["text"].strip()
            if not text:
                continue
            match = section
            match_idx = idx
            break
        if match is None:
            continue

        remaining_budget = max_content_chars - total_chars
        if remaining_budget <= 0:
            break

        body_budget = min(section_char_limit, remaining_budget)
        body = match["text"][:body_budget].strip()
        if not body:
            continue
        heading = match["heading"].upper()
        chunk = f"{heading}\n{body}"
        if len(chunk) > remaining_budget:
            chunk = chunk[:remaining_budget].rstrip()
        if not chunk:
            continue

        selected_chunks.append(chunk)
        selected_sections.append(
            {
                "kind": desired_kind,
                "heading": match["heading"],
                "char_count_total": len(match["text"]),
                "char_count_used": len(body),
            }
        )
        total_chars += len(chunk)
        used_kinds.add(desired_kind)
        if match_idx is not None:
            min_section_idx = match_idx + 1

    excerpt = "\n\n".join(chunk for chunk in selected_chunks if chunk).strip()
    if not excerpt:
        return None

    return {
        "content": excerpt[:max_content_chars].rstrip(),
        "selected_sections": selected_sections,
        "all_detected_sections": [
            {"kind": section["kind"], "heading": section["heading"]}
            for section in sections
        ],
    }


def resolve_paper_content(
    paper: dict,
    content_mode: str,
    fulltext_dir: Path | None,
    max_content_chars: int,
    fulltext_selection: str,
    section_char_limit: int,
) -> dict:
    abstract = (paper.get("abstract") or "").strip()
    paper_id = str(paper["paper_id"])

    if content_mode == "fulltext":
        fulltext_path = get_fulltext_path(fulltext_dir, paper_id)
        if fulltext_path is not None:
            full_text = fulltext_path.read_text(encoding="utf-8").strip()
            if full_text:
                if fulltext_selection == "core-sections":
                    excerpt_meta = build_core_section_excerpt(
                        abstract=abstract,
                        full_text=full_text,
                        max_content_chars=max_content_chars,
                        section_char_limit=section_char_limit,
                    )
                    if excerpt_meta is not None:
                        excerpt = excerpt_meta["content"]
                        return {
                            "requested_mode": "fulltext",
                            "used_source": "fulltext_core_sections",
                            "path": str(fulltext_path),
                            "content": excerpt,
                            "content_label": "Selected Full Paper Sections",
                            "evidence_description": (
                                "title, author-provided keywords, the clean abstract, and selected"
                                " full-paper sections (introduction, methods/approach, results/evaluation,"
                                " discussion/limitations, conclusion when available)"
                            ),
                            "char_count_total": len(full_text),
                            "char_count_used": len(excerpt),
                            "word_count_total": len(full_text.split()),
                            "word_count_used": len(excerpt.split()),
                            "content_sha256": sha256_text(excerpt),
                            "selected_sections": excerpt_meta["selected_sections"],
                            "all_detected_sections": excerpt_meta["all_detected_sections"],
                        }
                used_text = full_text[:max_content_chars]
                return {
                    "requested_mode": "fulltext",
                    "used_source": "fulltext_raw_truncated",
                    "path": str(fulltext_path),
                    "content": used_text,
                    "content_label": "Extracted Full Paper Text",
                    "evidence_description": "title, author-provided keywords, and extracted full paper text",
                    "char_count_total": len(full_text),
                    "char_count_used": len(used_text),
                    "word_count_total": len(full_text.split()),
                    "word_count_used": len(used_text.split()),
                    "content_sha256": sha256_text(used_text),
                    "selected_sections": [],
                    "all_detected_sections": [],
                }

    used_text = abstract[:max_content_chars]
    evidence_description = "title, abstract, and author-provided keywords only"
    if content_mode == "fulltext":
        evidence_description += " (full text unavailable for this paper, so the run fell back to abstract)"
    return {
        "requested_mode": content_mode,
        "used_source": "abstract",
        "path": None,
        "content": used_text,
        "content_label": "Abstract",
        "evidence_description": evidence_description,
        "char_count_total": len(abstract),
        "char_count_used": len(used_text),
        "word_count_total": len(abstract.split()),
        "word_count_used": len(used_text.split()),
        "content_sha256": sha256_text(used_text),
        "selected_sections": [
            {
                "kind": "abstract",
                "heading": "Abstract",
                "char_count_total": len(abstract),
                "char_count_used": len(used_text),
            }
        ] if used_text else [],
        "all_detected_sections": [],
    }
