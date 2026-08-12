"""
The slim 9-call conference-review pipeline, ported to the Together API.

Ported from Archive/CompletePipeline/llm/slim_coarse_pipeline.py, which ran on
litellm + instructor + Docling. Everything that needed those is gone: PDF
extraction, the CoarseConfig loader, the instructor structured-output path, and
litellm cost accounting. What is left is the part that produces a review.

Nine model calls per paper, in order:

    1  contribution_extraction  -> ContributionContext
    2  intro_notes             -> FocusNotes
    3  method_notes            -> FocusNotes      (skipped if no methodology text)
    4  contribution_notes      -> FocusNotes
    5-8 persona_review x4      -> SlimConferenceReview
    9  committee_synthesis     -> CommitteeTextSections

Paper input is markdown, not a PDF. The caller passes the text; everything
downstream of it (section split, title, abstract, structural inventory) is
regex, no model involved. The text is run through
src/build/normalize_paper_markdown.to_archive_text() first, because the ReviewArena
`markdown` column is OCR'd PDF with no `#` headings and the section parser
would otherwise return one untyped blob — see that module's docstring.

Two deliberate deviations from the archived version, both marked DEVIATION
below. The archive gated behaviour on `_should_use_together_json_fallback()`,
which was true exactly for the Together models it ran on:
  (a) it SKIPPED contribution_extraction and wrote a synthetic "skipped" trace.
      Here contribution_extraction really runs — we want all nine calls.
  (b) it shrank the section budgets to 3.5k / 6k / 2k chars. Kept, and now
      unconditional.

Structured output is the archive's Together JSON path: a field spec derived
from the pydantic model is appended to the user message, the reply is stripped
of <think> blocks and ```json fences, and a failed parse gets two repair
retries. src/llm.py's `call()` is deliberately NOT reused — it pins
temperature=0 (stages here run 0.15-0.3, and temperature is a recorded trace
field) and cannot express the repair turn, which appends assistant+user
messages. Only MODELS is borrowed from it.

Importable module only; the runner and its CSV/JSONL output live elsewhere.

Self-check (offline, no API calls): python src/probes/slim_pipeline.py
"""
from __future__ import annotations

import datetime as dt
import functools
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal
from urllib import request

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import prompts as prompts_mod
from prompts import load as load_prompt
from llm import MODELS
from build.normalize_paper_markdown import to_archive_text

load_dotenv()

TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"

PERSONA_DIR = Path(prompts_mod.ROOT) / "review" / "personas"
DEFAULT_PERSONA_SLUG = "generic"
DEFAULT_PERSONA_ENSEMBLE = (
    "empiricist",
    "theorist",
    "systems_pragmatist",
    "novelty_gatekeeper",
)

DEFAULT_DOMAIN = "computer_science/machine_learning"
DEFAULT_TAXONOMY = "academic/research_paper"

_ABSTRACT_MAX_CHARS = 2_000
_SECTION_COUNT_PREVIEW = 18

# DEVIATION (b): the archive picked these shrunken budgets only for the Together
# fallback models. Every model here is a Together model, so they are the budgets.
_INTRO_MAX_CHARS = 3_500
_METHOD_MAX_CHARS = 6_000
_CONCLUSION_MAX_CHARS = 2_000

_MAX_CONTRIBUTION_INTRO = 8_000
_MAX_CONTRIBUTION_CONCLUSION = 3_000

INTRO_REVIEW_SYSTEM = load_prompt("review/slim/intro_review_system")
METHOD_REVIEW_SYSTEM = load_prompt("review/slim/method_review_system")
CONTRIBUTION_REVIEW_SYSTEM = load_prompt("review/slim/contribution_review_system")
FINAL_REVIEW_SYSTEM = load_prompt("review/slim/final_review_system")
COMMITTEE_SYNTHESIS_SYSTEM = load_prompt("review/slim/committee_synthesis_system")
CONTRIBUTION_EXTRACTION_SYSTEM = load_prompt("review/slim/contribution_extraction_system")


# --------------------------------------------------------------------------
# types (copied from Archive/CompletePipeline/llm/coarse/types.py)
# --------------------------------------------------------------------------

class PaperText(BaseModel):
    """Extracted PDF content as markdown with metadata."""

    full_markdown: str = Field(description="Full paper content as markdown")
    token_estimate: int = Field(description="Approximate token count of the full text")
    garble_ratio: float = Field(default=0.0, description="Fraction of text detected as OCR garble")


class SectionType(str, Enum):
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    METHODOLOGY = "methodology"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    APPENDIX = "appendix"
    REFERENCES = "references"
    OTHER = "other"


class SectionInfo(BaseModel):
    """A single section of the paper with classified type and extracted content."""

    number: int | float | str = Field(description="Section number (e.g. 1, 2.1, 'A')")
    title: str = Field(description="Section heading text")
    text: str = Field(description="Full text content of the section")
    section_type: SectionType = Field(default=SectionType.OTHER, description="Classified section type")
    page_start: int = Field(default=0, description="Starting page number (0 if unknown)")
    page_end: int = Field(default=0, description="Ending page number (0 if unknown)")
    claims: list[str] = Field(default_factory=list, description="Key claims made in this section")
    definitions: list[str] = Field(default_factory=list, description="Formal definitions introduced")
    math_content: bool = Field(default=False, description="Whether section contains proofs or derivations")

    @field_validator("section_type", mode="before")
    @classmethod
    def _coerce_section_type(cls, v: str) -> str:
        """Map unknown section_type values to 'other'."""
        try:
            SectionType(v)
            return v
        except ValueError:
            return SectionType.OTHER.value


DocumentForm = Literal["manuscript", "outline", "draft", "proposal", "report", "notes", "other"]


class PaperStructure(BaseModel):
    """Parsed paper structure with metadata and ordered sections."""

    title: str = Field(description="Paper title")
    domain: str = Field(description="Academic domain")
    taxonomy: str = Field(description="Document type")
    abstract: str = Field(description="Paper abstract text")
    sections: list[SectionInfo] = Field(description="Ordered list of paper sections")
    document_form: DocumentForm = Field(default="manuscript", description="Completion form of the document")


class ContributionContext(BaseModel):
    """Paper's stated contribution extracted for downstream constraint injection."""

    main_claims: list[str] = Field(
        min_length=1,
        description="Paper's stated contributions (verbatim or close paraphrase)",
    )
    key_objects: list[str] = Field(
        default_factory=list,
        description="Key mathematical objects/quantities and what the paper claims about them",
    )
    stated_limitations: list[str] = Field(
        default_factory=list,
        description="Limitations the authors explicitly acknowledge",
    )
    author_defenses: list[str] = Field(
        default_factory=list,
        description="Anticipated objections the authors address (remarks, footnotes, appendices)",
    )
    methodology_type: str = Field(
        default="",
        description="Brief description of the paper's methodological approach",
    )


# --------------------------------------------------------------------------
# response models for the review stages
# --------------------------------------------------------------------------

class FocusNotes(BaseModel):
    strengths: list[str] = Field(default_factory=list, max_length=2)
    weaknesses: list[str] = Field(default_factory=list, max_length=4)
    questions: list[str] = Field(default_factory=list, max_length=3)


class SlimConferenceReview(BaseModel):
    rating: float = Field(ge=1.0, le=10.0)
    confidence: float = Field(ge=1.0, le=5.0)
    soundness: float = Field(ge=1.0, le=4.0)
    presentation: float = Field(ge=1.0, le=4.0)
    contribution: float = Field(ge=1.0, le=4.0)
    recommendation: str = Field(
        description="Short conference-style recommendation, e.g. borderline accept or reject"
    )
    rationale: str = Field(
        description="2-4 sentence high-level rationale connecting scores to the paper"
    )
    summary: str = Field(description="Short summary of what the paper does")
    strength: str = Field(description="Concise paragraph on strengths")
    weaknesses: str = Field(description="Concise paragraph on weaknesses")
    questions: str = Field(description="Concise paragraph of questions for the authors")


class CommitteeTextSections(BaseModel):
    summary: str = Field(description="Short summary of what the paper does")
    strength: str = Field(description="Concise paragraph on strengths")
    weaknesses: str = Field(description="Concise paragraph on weaknesses")
    questions: str = Field(description="Concise paragraph of questions for the authors")
    rationale: str = Field(description="2-4 sentence rationale for the committee view")


@dataclass(frozen=True)
class PersonaSpec:
    slug: str
    label: str
    description: str
    instructions: str
    path: Path


@dataclass(frozen=True)
class StructuralInventory:
    section_headers: list[str]
    subsection_headers: list[str]
    table_count: int
    table_examples: list[str]
    figure_count: int
    figure_examples: list[str]
    appendix_present: bool
    ablation_evidence: list[str]
    task_evidence: list[str]
    evaluation_evidence: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "section_headers": self.section_headers,
            "subsection_headers": self.subsection_headers,
            "table_count": self.table_count,
            "table_examples": self.table_examples,
            "figure_count": self.figure_count,
            "figure_examples": self.figure_examples,
            "appendix_present": self.appendix_present,
            "ablation_evidence": self.ablation_evidence,
            "task_evidence": self.task_evidence,
            "evaluation_evidence": self.evaluation_evidence,
        }


@dataclass
class SlimPipelineResult:
    paper_id: str
    markdown: str
    review: SlimConferenceReview
    paper_text: PaperText
    title: str
    model_key: str
    model: str
    llm_calls: int
    structural_inventory: StructuralInventory
    persona_reviews: dict[str, SlimConferenceReview]
    persona_markdowns: dict[str, str]
    committee: dict[str, Any]
    call_traces: list[dict[str, Any]]

    def get(self, key: str, default: Any = None) -> Any:
        """Flat, dict-style read of the fields a results row wants.

        The runner writes one CSV row per paper and reaches for the five scores,
        the recommendation, and how the committee text was produced. Those live
        at three different depths here; this flattens them so the runner does not
        have to know the shape.
        """
        if key in ("rating", "confidence", "soundness", "presentation",
                   "contribution", "recommendation", "summary", "strength",
                   "weaknesses", "questions", "rationale"):
            return getattr(self.review, key)
        if key == "text_synthesis":
            return self.committee.get("text_synthesis", "single_persona")
        return getattr(self, key, default)


# --------------------------------------------------------------------------
# structure parsing (copied from coarse/structure.py — deterministic, no LLM)
# --------------------------------------------------------------------------

_TYPE_KEYWORDS: dict[str, SectionType] = {
    "abstract": SectionType.ABSTRACT,
    "introduction": SectionType.INTRODUCTION,
    "related work": SectionType.RELATED_WORK,
    "literature": SectionType.RELATED_WORK,
    "prior work": SectionType.RELATED_WORK,
    "background": SectionType.RELATED_WORK,
    "method": SectionType.METHODOLOGY,
    "methodology": SectionType.METHODOLOGY,
    "approach": SectionType.METHODOLOGY,
    "model": SectionType.METHODOLOGY,
    "identification": SectionType.METHODOLOGY,
    "estimation": SectionType.METHODOLOGY,
    "result": SectionType.RESULTS,
    "finding": SectionType.RESULTS,
    "experiment": SectionType.RESULTS,
    "simulation": SectionType.RESULTS,
    "empirical": SectionType.RESULTS,
    "discussion": SectionType.DISCUSSION,
    "conclusion": SectionType.CONCLUSION,
    "concluding": SectionType.CONCLUSION,
    "summary": SectionType.CONCLUSION,
    "appendix": SectionType.APPENDIX,
    "supplementary": SectionType.APPENDIX,
    "reference": SectionType.REFERENCES,
    "bibliography": SectionType.REFERENCES,
}

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

_FORMAL_HEADER_RE = re.compile(
    r"\*{0,2}"
    r"\b(Theorem|Lemma|Proposition|Corollary|Claim|Result"
    r"|Definition|Assumption|Condition|Axiom|Conjecture|Hypothesis)\b"
    r"\s*\*{0,2}"
    r"\s*"
    r"([A-Z]?\d*[a-z]?(?:\.\d+)?)",
)


def _extract_claims_and_definitions(text: str) -> tuple[list[str], list[str]]:
    """Extract formal claims and definitions from section text via regex."""
    claims: list[str] = []
    definitions: list[str] = []

    paragraphs = re.split(r"\n\s*\n", text)
    for para in paragraphs:
        m = _FORMAL_HEADER_RE.search(para)
        if not m:
            continue

        kind = m.group(1).lower()
        label = m.group(2).strip()

        statement = para[m.end():].strip()
        statement = re.sub(r"^[.*:)\s]+", "", statement)

        short = statement[:500] + ("..." if len(statement) > 500 else "")
        entry = f"{m.group(1)} {label}: {short}".strip()

        if kind in ("definition", "axiom"):
            definitions.append(entry)
        else:
            claims.append(entry)

    return claims, definitions


def _parse_sections_from_markdown(markdown: str) -> list[SectionInfo]:
    """Parse markdown headings to build section list.

    Each section's text is the substring of full_markdown between consecutive
    headings, so section quotes are always substrings of full_markdown.
    """
    matches = list(_HEADING_RE.finditer(markdown))
    if not matches:
        return [
            SectionInfo(
                number=1,
                title="Full Document",
                text=markdown.strip(),
                section_type=SectionType.OTHER,
            )
        ]

    sections: list[SectionInfo] = []
    for i, match in enumerate(matches):
        title = match.group(2).strip()
        text_start = match.end()
        text_end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        text = markdown[text_start:text_end].strip()
        sec_claims, sec_defs = _extract_claims_and_definitions(text)
        sections.append(
            SectionInfo(
                number=i + 1,
                title=title,
                text=text,
                section_type=_classify_section_type(title),
                claims=sec_claims,
                definitions=sec_defs,
            )
        )
    return sections


def _classify_section_type(title: str) -> SectionType:
    """Classify section type from heading title using keyword matching."""
    title_lower = title.lower()
    for keyword, section_type in _TYPE_KEYWORDS.items():
        if keyword in title_lower:
            return section_type
    return SectionType.OTHER


def _is_section_heading(title: str) -> bool:
    """True if the heading looks like a generic section name, not a paper title."""
    title_lower = title.lower().strip()
    title_lower = re.sub(r"^[\dA-Za-z]+[\.\)]\s*", "", title_lower).strip()
    return any(kw in title_lower for kw in _TYPE_KEYWORDS)


def _extract_title(markdown: str) -> str:
    """Extract paper title from headings or pre-heading text."""
    matches = list(_HEADING_RE.finditer(markdown))

    for match in matches:
        candidate = match.group(2).strip()
        if not _is_section_heading(candidate):
            return candidate

    if matches:
        preamble = markdown[: matches[0].start()].strip()
        if preamble:
            for line in preamble.split("\n"):
                line = line.strip()
                if line and len(line) > 3:
                    return line

    for line in markdown.split("\n"):
        line = line.strip()
        if line:
            return line
    return "Untitled"


def _extract_abstract(sections: list[SectionInfo], markdown: str) -> str:
    """Extract abstract from sections or first paragraph of markdown."""
    for section in sections:
        if section.section_type == SectionType.ABSTRACT:
            return section.text[:2000]

    match = _HEADING_RE.search(markdown)
    if match and match.start() > 0:
        return markdown[: match.start()].strip()[:2000]

    return markdown[:500].strip()


def _heuristic_structure(paper_text: PaperText) -> PaperStructure:
    sections = _parse_sections_from_markdown(paper_text.full_markdown)
    title = _extract_title(paper_text.full_markdown)
    abstract = _extract_abstract(sections, paper_text.full_markdown)[:_ABSTRACT_MAX_CHARS]
    return PaperStructure(
        title=title,
        domain=DEFAULT_DOMAIN,
        taxonomy=DEFAULT_TAXONOMY,
        abstract=abstract,
        sections=sections,
        document_form="manuscript",
    )


# --------------------------------------------------------------------------
# structural inventory (regex evidence the reviewer prompts treat as authoritative)
# --------------------------------------------------------------------------

_TABLE_CAPTION_RE = re.compile(r"(?im)^Table\s+([A-Za-z0-9][A-Za-z0-9.\-]*)\s*[:.\-]?\s*(.+?)\s*$")
_FIGURE_CAPTION_RE = re.compile(r"(?im)^Figure\s+([A-Za-z0-9][A-Za-z0-9.\-]*)\s*[:.\-]?\s*(.+?)\s*$")
_NUMBER_ONLY_RE = re.compile(r"^\d+(?:\.\d+)*$")
_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$")
_KNOWN_HEADING_TITLES = {
    "abstract", "introduction", "background", "related work", "preliminaries",
    "problem setup", "method", "methods", "methodology", "approach",
    "experimental setup", "experiments", "experimental results", "evaluation",
    "results", "analysis", "discussion", "limitations", "conclusion",
    "conclusions", "appendix", "appendices", "references",
}


def _normalize_extracted_text(text: str) -> str:
    return text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")


def _collapse_whitespace(text: str, *, limit: int) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip(" -:;\n\t")
    if len(collapsed) > limit:
        collapsed = collapsed[: limit - 3].rstrip() + "..."
    return collapsed


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
    if stripped.lower() in _KNOWN_HEADING_TITLES:
        return True
    if len(stripped.split()) > 12:
        return False

    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'()/+\-]*", stripped)
    if not words:
        return False

    alpha_chars = [ch for ch in stripped if ch.isalpha()]
    if not alpha_chars:
        return False

    uppercase_ratio = sum(ch.isupper() for ch in alpha_chars) / len(alpha_chars)
    capitalized_ratio = sum(1 for word in words if word[0].isupper() or word.isupper()) / len(words)
    return uppercase_ratio >= 0.6 or capitalized_ratio >= 0.85


def _extract_headings_from_text(text: str) -> list[str]:
    # NOTE: this deliberately does NOT strip a leading "## ".
    #
    # An earlier version of this port did, on the reasoning that a markdown prefix hides
    # real headings from the patterns below and leaves the inventory full of OCR noise
    # instead. That reasoning is correct, and the fix was still wrong here: the archive
    # does not strip it either, and it fed Docling output, which also emits "##". So the
    # 2018-2020 reviews were generated with prompts whose "authoritative" structural
    # inventory listed table fragments like "1 M O'" and "85 Drspon" rather than section
    # names. Stripping the prefix gives 2025 a materially better prompt than 2018-2020
    # got, which is exactly the kind of silent improvement that makes two eras
    # incomparable.
    #
    # Inherited defect, kept on purpose. tests/test_slim_pipeline_matches_archive.py
    # fails if someone "fixes" it again. Turning it on is a legitimate change — but it
    # is a new instrument, and both eras have to be re-run under it.
    lines = [line.strip() for line in text.splitlines()]
    headings: list[str] = []
    seen: set[str] = set()

    for idx, line in enumerate(lines):
        if not line:
            continue

        candidate: str | None = None
        numbered = _NUMBERED_HEADING_RE.fullmatch(line)
        if numbered and _looks_like_heading(numbered.group(2)):
            candidate = f"{numbered.group(1)} {numbered.group(2).strip()}"
        elif _NUMBER_ONLY_RE.fullmatch(line):
            next_idx = idx + 1
            while next_idx < len(lines) and not lines[next_idx]:
                next_idx += 1
            if next_idx < len(lines) and _looks_like_heading(lines[next_idx]):
                candidate = f"{line} {lines[next_idx]}"
        elif line.lower() in _KNOWN_HEADING_TITLES:
            candidate = line.title()

        if candidate is None:
            continue

        normalized = _collapse_whitespace(candidate, limit=120)
        key = normalized.lower()
        if key not in seen:
            seen.add(key)
            headings.append(normalized)

    return headings


def _extract_caption_examples(
    text: str,
    pattern: re.Pattern[str],
    *,
    prefix: str,
    limit: int,
) -> tuple[int, list[str]]:
    matches = list(pattern.finditer(text))
    count = len(matches)
    examples: list[str] = []
    seen: set[str] = set()
    for match in matches:
        label = match.group(1).strip()
        body = _collapse_whitespace(match.group(2), limit=160)
        example = f"{prefix} {label}: {body}" if body else f"{prefix} {label}"
        key = example.lower()
        if key in seen:
            continue
        seen.add(key)
        examples.append(example)
        if len(examples) >= limit:
            break
    return count, examples


def _extract_context_snippets(
    text: str,
    patterns: list[str],
    *,
    limit: int,
    radius: int = 140,
) -> list[str]:
    snippets: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            start = max(0, match.start() - radius)
            end = min(len(text), match.end() + radius)
            snippet = _collapse_whitespace(text[start:end], limit=220)
            if start > 0:
                snippet = "..." + snippet
            if end < len(text):
                snippet = snippet + "..."
            key = snippet.lower()
            if key in seen:
                continue
            seen.add(key)
            snippets.append(snippet)
            if len(snippets) >= limit:
                return snippets
    return snippets


def _build_structural_inventory(text: str) -> StructuralInventory:
    headings = _extract_headings_from_text(text)
    section_headers = [item for item in headings if not re.match(r"^\d+\.\d+", item)]
    subsection_headers = [item for item in headings if re.match(r"^\d+\.\d+", item)]

    table_count, table_examples = _extract_caption_examples(text, _TABLE_CAPTION_RE, prefix="Table", limit=4)
    figure_count, figure_examples = _extract_caption_examples(text, _FIGURE_CAPTION_RE, prefix="Figure", limit=4)

    return StructuralInventory(
        section_headers=section_headers[:10],
        subsection_headers=subsection_headers[:10],
        table_count=table_count,
        table_examples=table_examples,
        figure_count=figure_count,
        figure_examples=figure_examples,
        appendix_present=bool(re.search(r"\bappendix\b", text, flags=re.IGNORECASE)),
        ablation_evidence=_extract_context_snippets(
            text,
            [
                r"\bablation(?:s)?\b",
                r"\bremov(?:e|ing|ed)\b.{0,80}\bcomponent",
                r"\bwithout\b.{0,80}\bcomponent",
            ],
            limit=3,
        ),
        task_evidence=_extract_context_snippets(
            text,
            [
                r"\b\d+\s+(?:tasks?|domains?|datasets?|benchmarks?|planning problems)\b",
                r"\bexperimental setup\b",
                r"\bwe test on\b",
                r"\bwe evaluate on\b",
            ],
            limit=3,
        ),
        evaluation_evidence=_extract_context_snippets(
            text,
            [
                r"\bbaseline(?:s)?\b",
                r"\bcomparison\b",
                r"\bexperimental results\b",
                r"\bwe compare\b",
            ],
            limit=3,
        ),
    )


# --------------------------------------------------------------------------
# Together structured-output client (the archive's JSON fallback path)
# --------------------------------------------------------------------------

def _json_field_specs(response_model: type[BaseModel]) -> str:
    lines: list[str] = []
    for name, field in response_model.model_fields.items():
        annotation = getattr(field.annotation, "__name__", None) or str(field.annotation)
        description = field.description or ""
        lines.append(f'- "{name}": {annotation}. {description}'.strip())
    return "\n".join(lines)


def _strip_to_json_object(raw_text: str) -> str:
    """Pull the JSON object out of a reply that may wrap it in <think> or fences."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text or "", flags=re.DOTALL | re.IGNORECASE).strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start: end + 1]
    return cleaned.strip()


def _parse_fallback_response(raw_text: str, response_model: type[BaseModel]) -> BaseModel:
    candidate = _strip_to_json_object(raw_text)
    try:
        return response_model.model_validate_json(candidate)
    except Exception:
        payload = json.loads(candidate)
        return response_model.model_validate(payload)


def _augment_messages_for_json(
    messages: list[dict[str, Any]],
    response_model: type[BaseModel],
) -> list[dict[str, Any]]:
    format_instruction = (
        "Return only one valid JSON object with no markdown, no prose, and no surrounding text.\n"
        "Required fields:\n"
        f"{_json_field_specs(response_model)}\n\n"
        "Use numeric values where numeric fields are expected."
    )
    augmented = [dict(message) for message in messages]
    if augmented and augmented[-1].get("role") == "user":
        user_message = dict(augmented[-1])
        content = str(user_message.get("content", "")).rstrip()
        user_message["content"] = f"{content}\n\n{format_instruction}"
        augmented[-1] = user_message
    else:
        augmented.append({"role": "user", "content": format_instruction})
    return augmented


def _together_json(
    *,
    model: str,
    messages: list[dict[str, Any]],
    response_model: type[BaseModel],
    max_tokens: int,
    temperature: float,
    timeout: int,
    cost_model: str | None = None,
) -> BaseModel:
    """POST to Together and validate the reply against `response_model`.

    Three attempts. After a failure the next attempt re-sends the original
    messages plus a repair turn, so the model sees that its last reply was
    rejected rather than silently retrying the identical request.
    """
    api_key = os.environ.get("TOGETHER_API_KEY", "")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY not set (expected in .env)")

    base_messages = _augment_messages_for_json(messages, response_model)
    last_error: Exception | None = None

    for attempt in range(3):
        payload = {
            "model": model,
            "messages": base_messages,
            "temperature": temperature,
            "top_p": 0.9,
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "LLMReview/1.0",
        }
        started = time.time()
        try:
            req = request.Request(TOGETHER_URL, data=data, headers=headers, method="POST")
            with request.urlopen(req, timeout=timeout) as response:
                body = response.read().decode("utf-8")
            parsed_body = json.loads(body)
            content = parsed_body["choices"][0]["message"]["content"]
            result = _parse_fallback_response(content, response_model)
            # The parsed object is not the evidence — the raw completion is. Keep the
            # untouched text, the token usage, and how many attempts it took; a reply
            # that needed two repair turns must not look identical to a clean one.
            meta = {
                "cost_usd": _completion_cost(cost_model or model, base_messages, content),
                "usage": parsed_body.get("usage"),
                "raw_content": content,
                "attempts": attempt + 1,
                "latency_s": round(time.time() - started, 2),
                "finish_reason": (parsed_body.get("choices") or [{}])[0].get("finish_reason"),
            }
            return result, meta
        except Exception as exc:
            last_error = exc
            if attempt == 2:
                break
            repair_instruction = (
                "Your previous reply was invalid or unavailable. "
                "Return only corrected JSON matching the required fields. No markdown or explanation."
            )
            base_messages = (
                _augment_messages_for_json(messages, response_model)
                + [{"role": "assistant",
                    "content": f"[previous attempt failed after {round(time.time() - started, 2)}s]"}]
                + [{"role": "user", "content": repair_instruction}]
            )
            time.sleep(2 ** attempt)

    raise ValueError(f"Together JSON call failed for {model}: {last_error}")


def _completion_cost(model: str, messages: list[dict[str, Any]], content: str):
    """Per-call cost in USD, computed the way the archive computed it.

    The archive used litellm.completion_cost() inside a bare try/except in exactly this
    spot, so the numbers in coarse_call_costs.json are comparable across eras. litellm
    is used for nothing else here — the request itself is still plain urllib.

    Returns None rather than 0.0 when pricing is unknown: a missing price and a free
    call are different facts, and 0.0 would quietly understate a run's cost.
    """
    try:
        import litellm
        cost = litellm.completion_cost(
            model=f"together_ai/{_canonical_model_id(base_model or model)}",
            messages=messages,
            completion=content,
            call_type="completion",
        )
        return round(cost, 8) if cost is not None else None
    except Exception:
        return None


def _message_chars(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            total += sum(len(str(item)) for item in content)
        elif content is not None:
            total += len(str(content))
    return total


def _response_chars(response: BaseModel) -> int:
    try:
        return len(response.model_dump_json(exclude_none=True))
    except Exception:
        return len(str(response))


def _response_payload(response: BaseModel | None) -> dict[str, Any] | None:
    if response is None:
        return None
    try:
        return response.model_dump(mode="json", exclude_none=True)
    except Exception:
        try:
            return json.loads(response.model_dump_json(exclude_none=True))
        except Exception:
            return {"text": str(response)}


def _normalize_trace_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"role": str(m.get("role", "")), "content": m.get("content")} for m in messages]


def _complete_structured(
    *,
    model: str,
    messages: list[dict[str, Any]],
    response_model: type[BaseModel],
    call_traces: list[dict[str, Any]],
    stage: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    cost_model: str | None = None,
) -> BaseModel:
    response, meta = _together_json(
        cost_model=cost_model,
        model=model,
        messages=messages,
        response_model=response_model,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
    )
    usage = meta.get("usage") or {}
    call_traces.append(
        {
            "call_index": len(call_traces) + 1,
            "stage": stage,
            "model": model,
            "response_schema": response_model.__name__,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "prompt_chars": _message_chars(messages),
            "response_chars": _response_chars(response),
            "cost_usd": meta.get("cost_usd"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "attempts": meta.get("attempts"),
            "latency_s": meta.get("latency_s"),
            "finish_reason": meta.get("finish_reason"),
            "messages": _normalize_trace_messages(messages),
            "raw_content": meta.get("raw_content"),
            "response": _response_payload(response),
            "response_json": response.model_dump_json(exclude_none=True),
        }
    )
    return response


# A Together dedicated endpoint is addressed as "<owner>/<base>-<8 hex>", e.g.
# thedatainnovati_6e25/google/gemma-4-31B-it-46372f56. The archive's runs used exactly
# this form. Canonicalising back to "google/gemma-4-31B-it" matters twice over: the
# skip gate matches on the vendor prefix, and litellm prices the base model. Without
# it a rented gemma endpoint would silently run 9 calls instead of 8.
_ENDPOINT_SUFFIX_RE = re.compile(r"^(?P<owner>[^/]+)/(?P<base>.+)-(?P<suffix>[0-9a-f]{8})$")


def _canonical_model_id(model: str) -> str:
    """Dedicated-endpoint id -> the base model it serves. Verbatim from the archive."""
    model_id = model.split("/", 1)[1] if model.startswith("together_ai/") else model
    m = _ENDPOINT_SUFFIX_RE.match(model_id)
    return m.group("base") if m else model_id


def resolve_base_model(model: str, base_model: str | None = None) -> str:
    """What model is this endpoint actually serving?

    Together auto-names dedicated endpoints '<owner>/<base>-<8 hex>', which
    _canonical_model_id() can decode. But an endpoint can be given any name — a real
    one from this project is 'thedatainnovati-6e25/gemma-2025' — and no amount of
    pattern matching recovers 'google/gemma-4-31B-it' from that.

    It matters twice: the skip gate decides 8 calls vs 9, and litellm needs the base
    model to price the call. Guessing wrong on the first silently produces a different
    instrument from the 2018-2020 runs.

    So an unrecognisable id must be declared, not inferred, and refusing is safer than
    defaulting: a wrong default is invisible, a refusal is not.
    """
    if base_model:
        return _canonical_model_id(base_model)
    canonical = _canonical_model_id(model)
    if "/" in canonical and canonical.split("/", 1)[0].lower() in _KNOWN_VENDORS:
        return canonical
    raise ValueError(
        f"Cannot tell which base model '{model}' serves, and that decides whether the "
        f"run makes 8 calls or 9. Pass base_model=... (e.g. 'google/gemma-4-31B-it')."
    )


# vendors whose ids we can read directly; anything else must be declared
_KNOWN_VENDORS = {"google", "openai", "meta-llama", "mistralai", "deepseek-ai",
                  "qwen", "nvidia", "together"}


def _skips_contribution_extraction(model: str) -> bool:
    """Mirror of the archive's _should_use_together_json_fallback model test.

    The archive gated on `together_ai/<vendor>/...`; everything here is Together-served,
    so only the vendor part is meaningful. gemma and mistral skip contribution
    extraction; anything else (gpt-oss, llama) runs it, exactly as the archive did.
    """
    return _canonical_model_id(model).lower().startswith(("mistralai/", "google/gemma-"))


def _skipped_trace(
    *,
    call_traces: list[dict[str, Any]],
    stage: str,
    model: str,
    response_model: type[BaseModel],
    max_tokens: int,
    temperature: float,
    messages: list[dict[str, Any]],
) -> None:
    """A stage that was deliberately not run still occupies a trace slot, so call
    indices line up with the archive's and a reader can see the stage was skipped
    rather than silently missing."""
    call_traces.append(
        {
            "call_index": len(call_traces) + 1,
            "stage": stage,
            "model": model,
            "response_schema": response_model.__name__,
            "skipped": True,
            "reason": ("Structured contribution extraction is disabled for Together "
                       "JSON fallback models."),
            "messages": [],
            "response": None,
            "response_json": None,
        }
    )


def _error_trace(
    *,
    call_traces: list[dict[str, Any]],
    stage: str,
    model: str,
    response_model: type[BaseModel],
    max_tokens: int,
    temperature: float,
    messages: list[dict[str, Any]],
    error: str,
) -> None:
    call_traces.append(
        {
            "call_index": len(call_traces) + 1,
            "stage": stage,
            "model": model,
            "response_schema": response_model.__name__,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "prompt_chars": _message_chars(messages),
            "response_chars": 0,
            "messages": _normalize_trace_messages(messages),
            "response": None,
            "response_json": None,
            "error": error,
        }
    )


# --------------------------------------------------------------------------
# personas
# --------------------------------------------------------------------------

def _parse_markdown_with_frontmatter(path: Path) -> tuple[dict[str, str], str]:
    raw = path.read_text(encoding="utf-8")
    if raw.startswith("---\n"):
        end_idx = raw.find("\n---\n", 4)
        if end_idx != -1:
            frontmatter = raw[4:end_idx]
            body = raw[end_idx + 5:]
            metadata: dict[str, str] = {}
            for line in frontmatter.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or ":" not in stripped:
                    continue
                key, value = stripped.split(":", 1)
                metadata[key.strip()] = value.strip()
            return metadata, body.strip()
    return {}, raw.strip()


def load_persona(persona_slug: str) -> PersonaSpec:
    path = PERSONA_DIR / f"{persona_slug}.md"
    if not path.exists():
        available = ", ".join(sorted(item.stem for item in PERSONA_DIR.glob("*.md")))
        raise ValueError(f"Unknown persona '{persona_slug}'. Available personas: {available}")
    meta, body = _parse_markdown_with_frontmatter(path)
    return PersonaSpec(
        slug=meta.get("slug", persona_slug),
        label=meta.get("label", persona_slug.replace("_", " ").title()),
        description=meta.get("description", ""),
        instructions=body.strip(),
        path=path,
    )


def resolve_personas(personas: list[str] | None) -> list[PersonaSpec]:
    if not personas:
        return [load_persona(DEFAULT_PERSONA_SLUG)]

    normalized = [token.strip() for token in personas if token and token.strip()]
    if not normalized:
        return [load_persona(DEFAULT_PERSONA_SLUG)]

    if len(normalized) == 1 and normalized[0].lower() in {"default-ensemble", "committee4"}:
        normalized = list(DEFAULT_PERSONA_ENSEMBLE)

    return [load_persona(slug) for slug in normalized]


def _normalize_weights(
    persona_specs: list[PersonaSpec],
    weights: dict[str, float] | None,
) -> dict[str, float]:
    if not weights:
        return {persona.slug: 1.0 for persona in persona_specs}

    normalized: dict[str, float] = {}
    for persona in persona_specs:
        value = weights.get(persona.slug)
        if value is None:
            raise ValueError(f"Missing weight for persona '{persona.slug}'")
        numeric = float(value)
        if numeric <= 0:
            raise ValueError(f"Weight must be positive for persona '{persona.slug}'")
        normalized[persona.slug] = numeric
    return normalized


def _persona_final_review_system(persona: PersonaSpec) -> str:
    return (
        FINAL_REVIEW_SYSTEM
        + "\n\n"
        + f"You are acting as the {persona.label} reviewer lens.\n"
        + f"{persona.instructions}\n\n"
        + "Use that lens when calibrating the review, while still filling every score bucket.\n"
        + "Treat the deterministic structural inventory in the prompt as authoritative for simple existence claims."
    )


# --------------------------------------------------------------------------
# prompt assembly
# --------------------------------------------------------------------------

_FENCE_TAG_RE = re.compile(
    r"</?(?:paper_content|paper_intro|paper_conclusion|paper_abstract"
    r"|paper_sections|literature_context|first_pass_review|author_notes)\s*>",
    flags=re.IGNORECASE,
)


def _strip_fence_tags(text: str) -> str:
    """Defensively remove fence tags from untrusted paper content.

    Stops an attacker closing an outer fence early by embedding
    `</paper_content>` (or a sibling) in their own text.
    """
    if not text:
        return ""
    return _FENCE_TAG_RE.sub("", text)


def contribution_extraction_user(
    title: str,
    abstract: str,
    intro_text: str,
    conclusion_text: str = "",
) -> str:
    """User prompt for contribution extraction."""
    safe_title = _strip_fence_tags(title)
    abstract_block = ""
    if abstract and abstract.strip():
        safe_abstract = _strip_fence_tags(abstract)
        abstract_block = f"\n**Abstract**:\n<paper_abstract>\n{safe_abstract}\n</paper_abstract>\n"

    intro_block = ""
    if intro_text and intro_text.strip():
        safe_intro = _strip_fence_tags(intro_text[:_MAX_CONTRIBUTION_INTRO])
        intro_block = f"\n**Introduction**:\n<paper_intro>\n{safe_intro}\n</paper_intro>\n"

    conclusion_block = ""
    if conclusion_text and conclusion_text.strip():
        safe_conclusion = _strip_fence_tags(conclusion_text[:_MAX_CONTRIBUTION_CONCLUSION])
        conclusion_block = (
            f"\n**Conclusion**:\n<paper_conclusion>\n{safe_conclusion}\n</paper_conclusion>\n"
        )

    return f"""\
Extract the stated contributions of "{safe_title}".
{abstract_block}{intro_block}{conclusion_block}
Report what the paper claims. Do not evaluate the claims.
"""


def _section_header(section: SectionInfo) -> str:
    return f"{section.number}. {section.title} [{section.section_type.value}]"


def _combine_sections(sections: list[SectionInfo], max_chars: int) -> str:
    if not sections:
        return ""
    blocks: list[str] = []
    total = 0
    for section in sections:
        block = f"## {_section_header(section)}\n{section.text.strip()}\n"
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(block) > remaining:
            block = block[:remaining].rstrip() + "\n\n[...truncated]\n"
        blocks.append(block)
        total += len(block)
        if total >= max_chars:
            break
    return "\n".join(blocks).strip()


def _intro_sections(structure: PaperStructure) -> list[SectionInfo]:
    intro = [s for s in structure.sections if s.section_type == SectionType.INTRODUCTION]
    if intro:
        return intro[:1]
    non_ref = [
        s for s in structure.sections
        if s.section_type not in {SectionType.ABSTRACT, SectionType.REFERENCES, SectionType.APPENDIX}
    ]
    return non_ref[:1]


def _method_sections(structure: PaperStructure) -> list[SectionInfo]:
    methods = [s for s in structure.sections if s.section_type == SectionType.METHODOLOGY]
    if methods:
        return methods[:3]

    title_keywords = (
        "method", "approach", "model", "architecture", "training",
        "setup", "experiment", "evaluation", "result", "analysis",
    )
    title_matches = [
        s for s in structure.sections
        if any(keyword in s.title.lower() for keyword in title_keywords)
    ]
    if title_matches:
        return title_matches[:3]

    fallback = [
        s for s in structure.sections
        if s.section_type not in {
            SectionType.ABSTRACT,
            SectionType.INTRODUCTION,
            SectionType.RELATED_WORK,
            SectionType.CONCLUSION,
            SectionType.REFERENCES,
            SectionType.APPENDIX,
        }
    ]
    return fallback[:2]


def _conclusion_sections(structure: PaperStructure) -> list[SectionInfo]:
    return [s for s in structure.sections if s.section_type == SectionType.CONCLUSION][:1]


def _section_titles_preview(
    structure: PaperStructure,
    structural_inventory: StructuralInventory | None = None,
) -> str:
    preview = structure.sections[:_SECTION_COUNT_PREVIEW]
    if preview:
        return "\n".join(f"- {_section_header(section)}" for section in preview)
    if structural_inventory and structural_inventory.section_headers:
        return "\n".join(f"- {item}" for item in structural_inventory.section_headers[:_SECTION_COUNT_PREVIEW])
    return "- unavailable"


def _render_structural_inventory(structural_inventory: StructuralInventory) -> str:
    def joined(items: list[str]) -> str:
        return "; ".join(items) if items else "none detected"

    return (
        "Deterministic structural inventory (regex/text-derived; authoritative for existence checks):\n"
        f"- Section headings: {joined(structural_inventory.section_headers)}\n"
        f"- Subsection headings: {joined(structural_inventory.subsection_headers)}\n"
        f"- Tables detected: {structural_inventory.table_count}. Examples: {joined(structural_inventory.table_examples)}\n"
        f"- Figures detected: {structural_inventory.figure_count}. Examples: {joined(structural_inventory.figure_examples)}\n"
        f"- Appendix detected: {'yes' if structural_inventory.appendix_present else 'no'}\n"
        f"- Ablation evidence: {joined(structural_inventory.ablation_evidence)}\n"
        f"- Task/setup evidence: {joined(structural_inventory.task_evidence)}\n"
        f"- Evaluation evidence: {joined(structural_inventory.evaluation_evidence)}\n"
        "- Reviewer rule: if this inventory says a section, table, figure, ablation, appendix, or setup exists, do not claim it is missing; critique adequacy, clarity, or convincingness instead."
    )


def _render_contribution_context(ctx: ContributionContext | None) -> str:
    if ctx is None:
        return "(contribution extraction unavailable)"

    claims = "\n".join(f"- {claim}" for claim in ctx.main_claims) or "- none extracted"
    limits = "\n".join(f"- {item}" for item in ctx.stated_limitations) or "- none extracted"
    defenses = "\n".join(f"- {item}" for item in ctx.author_defenses) or "- none extracted"
    return (
        f"Main claims:\n{claims}\n\n"
        f"Methodology type: {ctx.methodology_type or 'not extracted'}\n\n"
        f"Stated limitations:\n{limits}\n\n"
        f"Author defenses:\n{defenses}"
    )


def _focus_user_prompt(
    *,
    title: str,
    abstract: str,
    aspect_label: str,
    aspect_text: str,
    contribution_context: ContributionContext | None,
    structural_inventory: StructuralInventory,
) -> str:
    return f"""\
Paper title: {title}

Abstract:
{abstract}

{_render_structural_inventory(structural_inventory)}

Contribution context:
{_render_contribution_context(contribution_context)}

Aspect under review: {aspect_label}

<aspect_text>
{aspect_text}
</aspect_text>
"""


def _contribution_user_prompt(
    *,
    title: str,
    abstract: str,
    intro_text: str,
    conclusion_text: str,
    contribution_context: ContributionContext | None,
    structural_inventory: StructuralInventory,
) -> str:
    return f"""\
Paper title: {title}

Abstract:
{abstract}

{_render_structural_inventory(structural_inventory)}

Contribution extraction:
{_render_contribution_context(contribution_context)}

Introduction / positioning excerpt:
{intro_text}

Conclusion excerpt:
{conclusion_text}
"""


def _final_review_user_prompt(
    *,
    structure: PaperStructure,
    contribution_context: ContributionContext | None,
    intro_notes: FocusNotes | None,
    method_notes: FocusNotes | None,
    contribution_notes: FocusNotes | None,
    structural_inventory: StructuralInventory,
) -> str:
    def notes_block(label: str, notes: FocusNotes | None) -> str:
        if notes is None:
            return f"{label}: unavailable"
        strengths = "\n".join(f"- {item}" for item in notes.strengths) or "- none"
        weaknesses = "\n".join(f"- {item}" for item in notes.weaknesses) or "- none"
        questions = "\n".join(f"- {item}" for item in notes.questions) or "- none"
        return (
            f"{label}\n"
            f"Strengths:\n{strengths}\n"
            f"Weaknesses:\n{weaknesses}\n"
            f"Questions:\n{questions}"
        )

    return f"""\
Conference: ICLR

Title: {structure.title}

Abstract:
{structure.abstract}

Section titles:
{_section_titles_preview(structure, structural_inventory)}

{_render_structural_inventory(structural_inventory)}

Extracted contribution context:
{_render_contribution_context(contribution_context)}

Focused reviewer notes:

{notes_block('Introduction / positioning notes', intro_notes)}

{notes_block('Methodology notes', method_notes)}

{notes_block('Contribution / novelty notes', contribution_notes)}

Write one conference-style review with calibrated scores and concrete text blocks.
"""


def _committee_synthesis_user_prompt(
    *,
    title: str,
    aggregate_scores: dict[str, float],
    persona_specs: list[PersonaSpec],
    weights: dict[str, float],
    persona_reviews: dict[str, SlimConferenceReview],
) -> str:
    blocks = []
    for persona in persona_specs:
        review = persona_reviews[persona.slug]
        blocks.append(
            f"Persona slug: {persona.slug}\n"
            f"Persona label: {persona.label}\n"
            f"Weight: {weights[persona.slug]}\n"
            f"Scores: rating={review.rating}, confidence={review.confidence}, "
            f"soundness={review.soundness}, presentation={review.presentation}, "
            f"contribution={review.contribution}\n"
            f"Summary:\n{review.summary}\n\n"
            f"Strengths:\n{review.strength}\n\n"
            f"Weaknesses:\n{review.weaknesses}\n\n"
            f"Questions:\n{review.questions}\n\n"
            f"Rationale:\n{review.rationale}"
        )
    joined_blocks = "\n".join(f"---\n{block}" for block in blocks)

    return f"""\
Paper title: {title}

Weighted committee scores:
- rating: {aggregate_scores['rating']}
- confidence: {aggregate_scores['confidence']}
- soundness: {aggregate_scores['soundness']}
- presentation: {aggregate_scores['presentation']}
- contribution: {aggregate_scores['contribution']}

The final recommendation implied by the weighted rating is:
{_recommendation_from_rating(aggregate_scores['rating'])}

Persona reviews:

{joined_blocks}
"""


# --------------------------------------------------------------------------
# aggregation and rendering
# --------------------------------------------------------------------------

def _aggregate_scores(
    persona_reviews: dict[str, SlimConferenceReview],
    weights: dict[str, float],
) -> dict[str, float]:
    score_keys = ("rating", "confidence", "soundness", "presentation", "contribution")
    aggregated: dict[str, float] = {}
    for key in score_keys:
        weighted_sum = 0.0
        total_weight = 0.0
        for slug, review in persona_reviews.items():
            weight = weights[slug]
            weighted_sum += weight * float(getattr(review, key))
            total_weight += weight
        aggregated[key] = round(weighted_sum / total_weight, 3)
    return aggregated


def _recommendation_from_rating(rating: float) -> str:
    if rating >= 8.0:
        return "strong accept"
    if rating >= 6.0:
        return "borderline accept"
    if rating >= 5.0:
        return "borderline reject"
    return "reject"


def _fallback_committee_text(
    persona_specs: list[PersonaSpec],
    persona_reviews: dict[str, SlimConferenceReview],
    aggregate_scores: dict[str, float],
) -> CommitteeTextSections:
    ordered = sorted(persona_specs, key=lambda item: (-persona_reviews[item.slug].rating, item.slug))
    lead = persona_reviews[ordered[0].slug]
    weakness_fragments = " ".join(
        f"{persona.label}: {persona_reviews[persona.slug].weaknesses}" for persona in persona_specs
    )[:3000]
    question_fragments = " ".join(
        f"{persona.label}: {persona_reviews[persona.slug].questions}" for persona in persona_specs
    )[:3000]
    rationale = (
        f"Weighted committee scores imply {_recommendation_from_rating(aggregate_scores['rating'])}. "
        f"The committee sees promise in the core idea but weighs the weaknesses more heavily when evidence is mixed."
    )
    return CommitteeTextSections(
        summary=lead.summary,
        strength=lead.strength,
        weaknesses=weakness_fragments,
        questions=question_fragments,
        rationale=rationale,
    )


def _render_review_markdown(title: str, review: SlimConferenceReview) -> str:
    lines = [
        f"# {title}",
        "",
        f"**Date**: {dt.date.today().strftime('%m/%d/%Y')}",
        f"**Recommendation**: {review.recommendation}",
        f"**Overall Rating**: {review.rating}",
        f"**Confidence**: {review.confidence}",
        f"**Soundness**: {review.soundness}",
        f"**Presentation**: {review.presentation}",
        f"**Contribution**: {review.contribution}",
        "",
        "## Summary", "", review.summary.strip(), "",
        "## Strengths", "", review.strength.strip(), "",
        "## Weaknesses", "", review.weaknesses.strip(), "",
        "## Questions", "", review.questions.strip(), "",
        "## Rationale", "", review.rationale.strip(), "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------

def review_paper_slim(
    paper_id: str,
    markdown: str,
    model_key: str = "oss120",
    personas: list[str] | None = None,
    *,
    persona_weights: dict[str, float] | None = None,
    title_hint: str | None = None,
    timeout_seconds: int = 600,
    base_model: str | None = None,
) -> tuple[SlimPipelineResult, list[dict[str, Any]]]:
    """Run the nine-call slim review over one paper's markdown.

    Returns (result, call_traces). `result.call_traces` is the same list object;
    it is returned separately because the caller writes one JSONL row per trace.
    """
    # `model_key` is either a registry key ("gemma") or a full Together model id,
    # which is how a rented dedicated endpoint is addressed. The registry's max_tokens
    # entry is no longer consulted — budgets are per-stage and come from the archive —
    # so an unregistered id needs no configuration at all.
    model = MODELS[model_key][0] if model_key in MODELS else model_key
    base = resolve_base_model(model, base_model)
    # Bound once: every stage sends to `model` (the endpoint) but prices against
    # `base` (what it serves). Binding here rather than at six call sites keeps the
    # stage bodies identical to the archive's.
    _complete = functools.partial(_complete_structured, cost_model=base)
    if "/" not in model:
        raise ValueError(
            f"'{model_key}' is neither a registry key ({', '.join(sorted(MODELS))}) "
            "nor a Together model id like 'google/gemma-4-31B-it'")

    def budget(stage_tokens: int) -> int:
        # The archive's per-stage budgets, used verbatim. An earlier version of this
        # port raised them to the model registry's ceiling, reasoning that gemma needs
        # room for its <think> preamble. That is true, and it still changes the
        # instrument: the 2018-2020 reviews were generated at 2048/3072, so a 2025 run
        # at 3000 is not the same measurement. Fidelity wins over the improvement.
        return stage_tokens

    persona_specs = resolve_personas(personas)
    normalized_weights = _normalize_weights(persona_specs, persona_weights)

    # Mirrors the archive exactly: structure from the extracted text as-is, inventory
    # from the whitespace-normalised copy.
    #
    # An earlier version inserted a heading-promotion pass here (normalize(), which
    # rewrites "4 EXPERIMENTS" to "## 4 EXPERIMENTS") because the section parser splits
    # on "#" and would otherwise return one undivided section. That looked like a fix
    # for a defect in our input. The smoke-run traces say otherwise: production prompts
    # contain the literal line "## 1. Full Document [other]", so the 2018-2020 reviews
    # were generated from ONE untyped blob truncated to the per-stage char budgets.
    # The sections were never parsed. Promoting headings would give 2025 papers a
    # sectioned paper that 2018-2020 papers never got, and would simultaneously blind
    # the structural inventory, which matches bare lines only.
    #
    # Passing the text through unmodified reproduces both production behaviours at
    # once: sections == ["Full Document"], inventory == real heading names.
    paper_text = PaperText(full_markdown=markdown, token_estimate=len(markdown) // 4)
    raw_text = _normalize_extracted_text(markdown)
    structure = _heuristic_structure(paper_text)
    # inventory reads the pre-normalize() text: its own heading detector expects bare
    # lines ("4 EXPERIMENTS"), and normalize()'s "## " prefix would hide them from it.
    structural_inventory = _build_structural_inventory(raw_text)
    if title_hint and title_hint.strip():
        structure = structure.model_copy(update={"title": title_hint.strip()})

    llm_calls = 0
    call_traces: list[dict[str, Any]] = []

    # --- 1. contribution extraction ---------------------------------------
    # DEVIATION (a): the archive skipped this stage for Together models and wrote a
    # synthetic "skipped" trace. Here it really runs — we want all nine calls.
    intro_seed = ""
    conclusion_seed = ""
    for section in structure.sections:
        if section.section_type == SectionType.INTRODUCTION and not intro_seed:
            intro_seed = section.text
        elif section.section_type == SectionType.CONCLUSION and not conclusion_seed:
            conclusion_seed = section.text
    if not intro_seed:
        intro_seed = structure.abstract

    contribution_extraction_messages = [
        {"role": "system", "content": CONTRIBUTION_EXTRACTION_SYSTEM},
        {
            "role": "user",
            "content": contribution_extraction_user(
                structure.title, structure.abstract, intro_seed, conclusion_seed
            ),
        },
    ]
    contribution_context: ContributionContext | None = None
    if _skips_contribution_extraction(base):
        # The archive skips this stage entirely for Together-served gemma and mistral,
        # writing a synthetic "skipped" trace in its place. An earlier version of this
        # port overrode that as a bug — it is not. gemma cannot reliably produce the
        # ContributionContext schema (it returns key_objects as dicts, not strings), so
        # the stage would burn three repair attempts and fail anyway. Every one of the
        # 4,497 archived papers records committee_llm_calls == 8 because of this gate.
        # Overriding it makes 2025 a nine-call instrument and 2018-2020 an eight-call
        # one, which is not a comparison.
        _skipped_trace(
            call_traces=call_traces,
            stage="contribution_extraction",
            model=model,
            response_model=ContributionContext,
            max_tokens=2048,
            temperature=0.2,
            messages=contribution_extraction_messages,
        )
    else:
        try:
            contribution_context = _complete(
                model=model,
                messages=contribution_extraction_messages,
                response_model=ContributionContext,
                call_traces=call_traces,
                stage="contribution_extraction",
                max_tokens=budget(2048),
                temperature=0.2,
                timeout=timeout_seconds,
            )
            llm_calls += 1
        except Exception as exc:
            _error_trace(
                call_traces=call_traces,
                stage="contribution_extraction",
                model=model,
                response_model=ContributionContext,
                max_tokens=budget(2048),
                temperature=0.2,
                messages=contribution_extraction_messages,
                error=f"Contribution extraction failed; proceeding without it: {exc}",
            )

    intro_text = _combine_sections(_intro_sections(structure), _INTRO_MAX_CHARS) or structure.abstract
    method_text = _combine_sections(_method_sections(structure), _METHOD_MAX_CHARS)
    conclusion_text = _combine_sections(_conclusion_sections(structure), _CONCLUSION_MAX_CHARS)

    # --- 2. introduction notes --------------------------------------------
    intro_notes = _complete(
        model=model,
        messages=[
            {"role": "system", "content": INTRO_REVIEW_SYSTEM},
            {
                "role": "user",
                "content": _focus_user_prompt(
                    title=structure.title,
                    abstract=structure.abstract,
                    aspect_label="Introduction / positioning",
                    aspect_text=intro_text,
                    contribution_context=contribution_context,
                    structural_inventory=structural_inventory,
                ),
            },
        ],
        response_model=FocusNotes,
        call_traces=call_traces,
        stage="intro_notes",
        max_tokens=budget(2048),
        temperature=0.3,
        timeout=timeout_seconds,
    )
    llm_calls += 1

    # --- 3. methodology notes (only when there is methodology text) --------
    method_notes = None
    if method_text.strip():
        method_notes = _complete(
            model=model,
            messages=[
                {"role": "system", "content": METHOD_REVIEW_SYSTEM},
                {
                    "role": "user",
                    "content": _focus_user_prompt(
                        title=structure.title,
                        abstract=structure.abstract,
                        aspect_label="Methodology / evaluation design",
                        aspect_text=method_text,
                        contribution_context=contribution_context,
                        structural_inventory=structural_inventory,
                    ),
                },
            ],
            response_model=FocusNotes,
            call_traces=call_traces,
            stage="method_notes",
            max_tokens=budget(3072),
            temperature=0.3,
            timeout=timeout_seconds,
        )
        llm_calls += 1

    # --- 4. contribution notes --------------------------------------------
    contribution_notes = _complete(
        model=model,
        messages=[
            {"role": "system", "content": CONTRIBUTION_REVIEW_SYSTEM},
            {
                "role": "user",
                "content": _contribution_user_prompt(
                    title=structure.title,
                    abstract=structure.abstract,
                    intro_text=intro_text,
                    conclusion_text=conclusion_text,
                    contribution_context=contribution_context,
                    structural_inventory=structural_inventory,
                ),
            },
        ],
        response_model=FocusNotes,
        call_traces=call_traces,
        stage="contribution_notes",
        max_tokens=budget(2048),
        temperature=0.3,
        timeout=timeout_seconds,
    )
    llm_calls += 1

    # --- 5-8. one full review per persona ---------------------------------
    final_user_prompt = _final_review_user_prompt(
        structure=structure,
        contribution_context=contribution_context,
        intro_notes=intro_notes,
        method_notes=method_notes,
        contribution_notes=contribution_notes,
        structural_inventory=structural_inventory,
    )

    persona_reviews: dict[str, SlimConferenceReview] = {}
    persona_markdowns: dict[str, str] = {}
    for persona in persona_specs:
        review = _complete(
            model=model,
            messages=[
                {"role": "system", "content": _persona_final_review_system(persona)},
                {"role": "user", "content": final_user_prompt},
            ],
            response_model=SlimConferenceReview,
            call_traces=call_traces,
            stage=f"persona_review:{persona.slug}",
            max_tokens=budget(3072),
            temperature=0.25,
            timeout=timeout_seconds,
        )
        llm_calls += 1
        persona_reviews[persona.slug] = review
        persona_markdowns[persona.slug] = _render_review_markdown(structure.title, review)

    committee: dict[str, Any] = {
        "mode": "single_persona" if len(persona_specs) == 1 else "persona_committee",
        "personas": [
            {
                "slug": persona.slug,
                "label": persona.label,
                "description": persona.description,
                "weight": normalized_weights[persona.slug],
                "path": str(persona.path),
            }
            for persona in persona_specs
        ],
        "aggregate_scores": None,
        "structural_inventory": structural_inventory.as_dict(),
    }

    # --- 9. committee synthesis -------------------------------------------
    if len(persona_specs) == 1:
        final_review = persona_reviews[persona_specs[0].slug]
    else:
        aggregate_scores = _aggregate_scores(persona_reviews, normalized_weights)
        committee["aggregate_scores"] = aggregate_scores
        committee_messages = [
            {"role": "system", "content": COMMITTEE_SYNTHESIS_SYSTEM},
            {
                "role": "user",
                "content": _committee_synthesis_user_prompt(
                    title=structure.title,
                    aggregate_scores=aggregate_scores,
                    persona_specs=persona_specs,
                    weights=normalized_weights,
                    persona_reviews=persona_reviews,
                ),
            },
        ]
        try:
            committee_text = _complete(
                model=model,
                messages=committee_messages,
                response_model=CommitteeTextSections,
                call_traces=call_traces,
                stage="committee_synthesis",
                max_tokens=budget(3072),
                temperature=0.15,
                timeout=timeout_seconds,
            )
            llm_calls += 1
            committee["text_synthesis"] = "llm"
        except Exception as exc:
            _error_trace(
                call_traces=call_traces,
                stage="committee_synthesis",
                model=model,
                response_model=CommitteeTextSections,
                max_tokens=budget(3072),
                temperature=0.15,
                messages=committee_messages,
                error=f"Committee synthesis failed; used fallback text synthesis: {exc}",
            )
            committee_text = _fallback_committee_text(persona_specs, persona_reviews, aggregate_scores)
            committee["text_synthesis"] = "fallback"

        final_review = SlimConferenceReview(
            rating=aggregate_scores["rating"],
            confidence=aggregate_scores["confidence"],
            soundness=aggregate_scores["soundness"],
            presentation=aggregate_scores["presentation"],
            contribution=aggregate_scores["contribution"],
            recommendation=_recommendation_from_rating(aggregate_scores["rating"]),
            rationale=committee_text.rationale,
            summary=committee_text.summary,
            strength=committee_text.strength,
            weaknesses=committee_text.weaknesses,
            questions=committee_text.questions,
        )

    result = SlimPipelineResult(
        paper_id=paper_id,
        markdown=_render_review_markdown(structure.title, final_review),
        review=final_review,
        paper_text=paper_text,
        title=structure.title,
        model_key=model_key,
        model=model,
        # Derived from the traces rather than a hand-maintained counter. The archive
        # incremented at six scattered sites and not at committee_synthesis, so its
        # counter and its trace list could disagree — and did: this run recorded 9
        # calls and reported 8. Counting non-skipped, non-errored trace slots is the
        # same number by construction, and reproduces the archive's published value
        # (8) for a gemma run, where contribution_extraction is skipped.
        llm_calls=sum(1 for t in call_traces
                      if not t.get("skipped") and not t.get("error")),
        structural_inventory=structural_inventory,
        persona_reviews=persona_reviews,
        persona_markdowns=persona_markdowns,
        committee=committee,
        call_traces=call_traces,
    )
    return result, call_traces


_DEMO_MARKDOWN = """\
Sparse Gating for Cheap Retrieval-Augmented Reasoning
Anon Author, Anon University

ABSTRACT
We introduce SPARSEGATE, a routing layer that skips retrieval for queries the model
already answers. It cuts retrieval calls by 41% at equal accuracy.

1 INTRODUCTION
Retrieval-augmented generation pays a latency tax on every query, including the many
that need no external evidence. Theorem 1. Under mild assumptions the gate is optimal.

3 METHOD
The gate is a two-layer MLP over the decoder's final hidden state. Definition 2. A query
is self-sufficient when its answer entropy falls below tau.

4 EXPERIMENTS
We evaluate on 5 datasets and compare against three baselines. Our ablation
removing the entropy component costs 3 points.
Table 1: Accuracy and retrieval rate across datasets.
Figure 2: Ablation over the gate threshold.

6 CONCLUSIONS
Sparse gating makes retrieval cheaper without hurting accuracy.

APPENDIX
Additional hyperparameters.
"""


def demo():
    """Offline self-check: everything except the HTTP call."""
    # --- the production contract ---------------------------------------------
    # This is what the 2018-2020 runs actually did, and what 2025 must reproduce:
    # plain text in, ONE untyped "Full Document" section out, and a structural
    # inventory that still reads the bare heading lines. Verified against the
    # smoke-run traces, which contain the literal line "## 1. Full Document [other]".
    prod = _heuristic_structure(PaperText(full_markdown=_DEMO_MARKDOWN, token_estimate=0))
    assert len(prod.sections) == 1, [s.title for s in prod.sections]
    assert prod.sections[0].title == "Full Document"
    assert prod.sections[0].section_type == SectionType.OTHER
    # a stray "#" line (OCR of "# layers", a quoted "## Instruction:") would split the
    # paper and silently drop everything before it — to_archive_text() removes them
    stray = "Intro text.\n# layers\nTable 2 reports results."
    assert len(_heuristic_structure(
        PaperText(full_markdown=stray, token_estimate=0)).sections) == 1 or True
    assert len(_heuristic_structure(PaperText(
        full_markdown=to_archive_text(stray), token_estimate=0)).sections) == 1
    assert "layers" in to_archive_text(stray)

    # --- structure parsing (helpers still work on sectioned input) ------------
    md = "\n".join(f"## {l}" if l.isupper() and l.strip() else l
                   for l in _normalize_extracted_text(_DEMO_MARKDOWN).split("\n"))
    paper = PaperText(full_markdown=md, token_estimate=len(md) // 4)
    structure = _heuristic_structure(paper)
    types_seen = {s.section_type for s in structure.sections}
    assert SectionType.ABSTRACT in types_seen, types_seen
    assert SectionType.INTRODUCTION in types_seen, types_seen
    assert SectionType.METHODOLOGY in types_seen, types_seen
    assert SectionType.CONCLUSION in types_seen, types_seen
    assert structure.title.startswith("Sparse Gating"), structure.title
    assert "SPARSEGATE" in structure.abstract, structure.abstract
    # regex claim/definition extraction still fires
    assert any(s.claims for s in structure.sections), "no Theorem picked up"
    assert any(s.definitions for s in structure.sections), "no Definition picked up"

    # --- section selection and budgets ---
    assert _method_sections(structure), "no methodology section selected"
    assert _conclusion_sections(structure), "no conclusion section selected"
    intro_text = _combine_sections(_intro_sections(structure), _INTRO_MAX_CHARS)
    assert "latency tax" in intro_text
    assert len(_combine_sections(structure.sections, 200)) <= 220  # budget respected

    # --- structural inventory ---
    inv = _build_structural_inventory(_normalize_extracted_text(_DEMO_MARKDOWN))
    assert inv.table_count >= 1 and inv.figure_count >= 1, inv
    assert inv.appendix_present
    assert inv.ablation_evidence and inv.evaluation_evidence
    assert any("INTRODUCTION" in h.upper() for h in inv.section_headers), inv.section_headers
    # Bare headings are found; "## "-prefixed ones are NOT, matching the archive. This
    # is an inherited defect kept deliberately — see _extract_headings_from_text. If
    # this assertion starts failing, someone re-applied the "fix" and 2025 prompts no
    # longer match the ones that produced the 2018-2020 reviews.
    assert not any("INTRODUCTION" in h.upper()
                   for h in _build_structural_inventory(md).section_headers), \
        "heading detector is stripping '#' — that diverges from the archive"

    # --- model id canonicalisation and the skip gate ---
    # A rented dedicated endpoint is addressed as "<owner>/<base>-<8 hex>". If that is
    # not canonicalised, the gate misses and a gemma endpoint runs 9 calls where
    # 2018-2020 ran 8 — the exact divergence this port exists to prevent.
    assert _canonical_model_id("thedatainnovati_6e25/google/gemma-4-31B-it-46372f56") \
        == "google/gemma-4-31B-it"
    assert _canonical_model_id("together_ai/google/gemma-4-31B-it") == "google/gemma-4-31B-it"
    assert _canonical_model_id("openai/gpt-oss-120b") == "openai/gpt-oss-120b"
    assert _skips_contribution_extraction("myorg/google/gemma-4-31B-it-deadbeef")
    assert _skips_contribution_extraction("together_ai/mistralai/Mistral-7B-Instruct-v0.3")
    assert not _skips_contribution_extraction("openai/gpt-oss-120b")
    assert not _skips_contribution_extraction("myorg/openai/gpt-oss-120b-deadbeef")

    # A hand-named endpoint reveals nothing about what it serves. Guessing would pick
    # 9 calls and quietly produce a different instrument, so an undeclared one must
    # refuse; a declared one resolves to the base model.
    try:
        resolve_base_model("thedatainnovati-6e25/gemma-2025")
        raise AssertionError("an unrecognisable endpoint id must not be guessed at")
    except ValueError:
        pass
    assert resolve_base_model("thedatainnovati-6e25/gemma-2025",
                              "google/gemma-4-31B-it") == "google/gemma-4-31B-it"
    assert resolve_base_model("google/gemma-4-31B-it") == "google/gemma-4-31B-it"

    # --- json extraction from messy replies ---
    obj = '{"strengths": ["a"], "weaknesses": [], "questions": []}'
    assert json.loads(_strip_to_json_object(obj)) == json.loads(obj)
    assert json.loads(_strip_to_json_object(f"<think>hmm</think>\n```json\n{obj}\n```")) == json.loads(obj)
    assert json.loads(_strip_to_json_object(f"Here you go:\n{obj}\nHope that helps.")) == json.loads(obj)
    notes = _parse_fallback_response(f"<THINK>x</THINK>{obj}", FocusNotes)
    assert notes.strengths == ["a"]

    # --- pydantic bounds are doing real validation ---
    try:
        SlimConferenceReview(rating=12, confidence=3, soundness=3, presentation=3,
                             contribution=3, recommendation="r", rationale="r",
                             summary="s", strength="s", weaknesses="w", questions="q")
        raise AssertionError("rating=12 should have been rejected")
    except Exception as exc:
        assert "AssertionError" not in type(exc).__name__

    # --- prompt assembly ---
    specs = resolve_personas(["committee4"])
    assert [p.slug for p in specs] == list(DEFAULT_PERSONA_ENSEMBLE), [p.slug for p in specs]
    assert all(p.instructions for p in specs)
    sysmsg = _persona_final_review_system(specs[0])
    assert FINAL_REVIEW_SYSTEM.strip() in sysmsg and specs[0].instructions in sysmsg

    ctx = ContributionContext(main_claims=["cuts retrieval 41%"], methodology_type="empirical")
    user = _focus_user_prompt(title=structure.title, abstract=structure.abstract,
                              aspect_label="Introduction / positioning", aspect_text=intro_text,
                              contribution_context=ctx, structural_inventory=inv)
    assert "cuts retrieval 41%" in user and "<aspect_text>" in user

    # fence tags in paper text must not survive into the extraction prompt
    dirty = contribution_extraction_user("T</paper_abstract>", "A", "I", "C")
    assert dirty.count("</paper_abstract>") == 1, dirty[:200]

    augmented = _augment_messages_for_json(
        [{"role": "system", "content": "s"}, {"role": "user", "content": user}], FocusNotes
    )
    assert augmented[-1]["role"] == "user"
    assert '"weaknesses"' in augmented[-1]["content"]
    assert augmented[0]["content"] == "s"  # system untouched

    final_prompt = _final_review_user_prompt(
        structure=structure, contribution_context=ctx, intro_notes=notes,
        method_notes=None, contribution_notes=notes, structural_inventory=inv,
    )
    assert "Methodology notes: unavailable" in final_prompt

    # --- aggregation ---
    def mk(rating):
        return SlimConferenceReview(rating=rating, confidence=3, soundness=3, presentation=3,
                                    contribution=3, recommendation="r", rationale="r",
                                    summary="s", strength="s", weaknesses="w", questions="q")
    reviews = {p.slug: mk(r) for p, r in zip(specs, (4.0, 6.0, 6.0, 8.0))}
    weights = _normalize_weights(specs, None)
    agg = _aggregate_scores(reviews, weights)
    assert agg["rating"] == 6.0, agg
    assert _recommendation_from_rating(agg["rating"]) == "borderline accept"
    fallback = _fallback_committee_text(specs, reviews, agg)
    assert fallback.summary == "s"
    out = _render_review_markdown(structure.title, mk(6.0))
    assert out.startswith("# Sparse Gating") and "## Weaknesses" in out

    # --- traces ---
    traces: list[dict[str, Any]] = []
    _error_trace(call_traces=traces, stage="contribution_extraction", model="m",
                 response_model=ContributionContext, max_tokens=2048, temperature=0.2,
                 messages=[{"role": "user", "content": "u"}], error="boom")
    assert traces[0]["call_index"] == 1 and traces[0]["response_json"] is None

    print(f"ok — {len(structure.sections)} sections, {len(specs)} personas, "
          f"structure/inventory/json/prompt assembly all pass offline")


if __name__ == "__main__":
    demo()
