#!/usr/bin/env python3
"""
Slimmed-down local review pipeline built on top of coarse primitives.

This variant is meant to behave more like a conference reviewer and less like a
full audit pipeline. It deliberately removes:
    - literature search
    - domain calibration
    - completeness pass
    - proof verification
    - cross-section synthesis
    - quote verification / repair
    - per-section review over the full paper

Instead it runs a small set of focused passes:
    1. extract stated contributions
    2. review introduction / positioning
    3. review methodology / experimental setup
    4. review claimed contributions / novelty framing
    5. synthesize one human-style conference review
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
import json
from pathlib import Path
import re
import time
from typing import Any
from urllib import error, request

import litellm
from pydantic import BaseModel, Field

from coarse.config import CoarseConfig, load_config
from coarse.extraction import extract_file
from coarse.llm import LLMClient
from coarse.review_stages import extract_contribution
from coarse.structure import _extract_abstract, _extract_title, _parse_sections_from_markdown
from coarse.types import ContributionContext, PaperStructure, PaperText, SectionInfo, SectionType


ROOT = Path(__file__).resolve().parents[2]
PERSONA_DIR = ROOT / "Code" / "paper_review" / "prompts" / "personas"
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
_INTRO_MAX_CHARS = 14_000
_METHOD_MAX_CHARS = 32_000
_CONCLUSION_MAX_CHARS = 6_000
_SECTION_COUNT_PREVIEW = 18
_MISTRAL_INTRO_MAX_CHARS = 3_500
_MISTRAL_METHOD_MAX_CHARS = 6_000
_MISTRAL_CONCLUSION_MAX_CHARS = 2_000
_TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

_TABLE_CAPTION_RE = re.compile(
    r"(?im)^Table\s+([A-Za-z0-9][A-Za-z0-9.\-]*)\s*[:.\-]?\s*(.+?)\s*$"
)
_FIGURE_CAPTION_RE = re.compile(
    r"(?im)^Figure\s+([A-Za-z0-9][A-Za-z0-9.\-]*)\s*[:.\-]?\s*(.+?)\s*$"
)
_NUMBER_ONLY_RE = re.compile(r"^\d+(?:\.\d+)*$")
_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$")
_KNOWN_HEADING_TITLES = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "preliminaries",
    "problem setup",
    "method",
    "methods",
    "methodology",
    "approach",
    "experimental setup",
    "experiments",
    "experimental results",
    "evaluation",
    "results",
    "analysis",
    "discussion",
    "limitations",
    "conclusion",
    "conclusions",
    "appendix",
    "appendices",
    "references",
}


INTRO_REVIEW_SYSTEM = """\
You are an ICLR reviewer writing notes on the introduction and positioning of a paper.

Focus on:
- problem framing
- why the question matters
- whether the contribution is clearly stated
- whether the scope and claims are calibrated
- whether the positioning feels conference-ready

Write like a human conference reviewer, not a line-by-line auditor.
Prefer paper-specific points over generic requests. It is fine to acknowledge strengths.
Do not ask for extra citations unless the omission materially changes the novelty claim.
Do not produce quote-by-quote comments.

Return:
- up to 2 strengths
- up to 4 weaknesses
- up to 3 concrete questions for the authors
"""


METHOD_REVIEW_SYSTEM = """\
You are an ICLR reviewer writing notes on the methodology and empirical design of a paper.

Focus on:
- whether the method is understandable and well-scoped
- whether the evidence seems adequate for the claims
- baseline and comparison logic when relevant
- ablations, setup, and evaluation choices when they are discussed
- practical concerns that would matter to program committee reviewers

Write like a human conference reviewer, not a proof checker.
Do not demand every possible experiment. Only flag gaps that matter for acceptance.
Do not produce quote-by-quote comments.

Return:
- up to 2 strengths
- up to 4 weaknesses
- up to 3 concrete questions for the authors
"""


CONTRIBUTION_REVIEW_SYSTEM = """\
You are an ICLR reviewer writing notes on a paper's claimed contributions.

Focus on:
- novelty beyond packaging or scale
- whether the main claim is legible
- whether the contribution sounds substantial enough for ICLR
- whether the scope of the claims matches the evidence described

This is not a literature survey and not a correctness audit. Think like a reviewer
deciding whether the contribution is clear and meaningful.

Return:
- up to 2 strengths
- up to 4 weaknesses
- up to 3 concrete questions for the authors
"""


FINAL_REVIEW_SYSTEM = """\
You are writing a single human-style ICLR review from structured reviewer notes.

Write like an experienced but concise conference reviewer:
- balanced, not performatively harsh
- specific, not generic
- willing to credit strengths
- willing to say when evidence is not yet convincing

Use the native ICLR-style score buckets:
- rating: 1 to 10
- confidence: 1 to 5
- soundness: 1 to 4
- presentation: 1 to 4
- contribution: 1 to 4

Calibrate conservatively:
- interesting ideas with incomplete evidence are often 5 or 6
- reserve 8+ for unusually compelling papers
- if evidence is mixed, the weakness text should say why

Return a review that sounds like one reviewer report, not a meta-analysis.
Do not mention hidden pipeline stages or say that some paper sections were omitted.
"""


COMMITTEE_SYNTHESIS_SYSTEM = """\
You are the senior area chair consolidating a committee of persona reviewers into a
single coherent ICLR-style review.

Your job is to:
- merge overlapping strengths into a concise strengths section
- merge overlapping weaknesses into a concise weaknesses section
- preserve important disagreements when they matter for the final judgment
- avoid inventing new critiques that no persona raised
- keep the tone like a human conference review, not a meta-analysis memo

When personas disagree on severity, do not force false consensus. It is acceptable
to note that evidence seems mixed or that one risk drives the decision.

Return only the text sections of the final committee review. Numeric scores are
handled separately and are not part of your task.
"""


@dataclass(frozen=True)
class PersonaSpec:
    slug: str
    label: str
    description: str
    instructions: str
    path: Path


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


@dataclass
class SlimPipelineResult:
    markdown: str
    review: SlimConferenceReview
    paper_text: PaperText
    title: str
    llm_calls: int
    cost_usd: float
    call_costs: list[dict[str, Any]]
    structural_inventory: "StructuralInventory"
    persona_reviews: dict[str, SlimConferenceReview]
    persona_markdowns: dict[str, str]
    committee: dict[str, Any]


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


def _now_date() -> str:
    return dt.date.today().strftime("%m/%d/%Y")


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


def _should_use_together_json_fallback(model: str) -> bool:
    lower = model.lower()
    return lower.startswith("together_ai/mistralai/")


def _direct_together_model_id(model: str) -> str:
    if model.startswith("together_ai/"):
        return model.split("/", 1)[1]
    return model


def _json_field_specs(response_model: type[BaseModel]) -> str:
    lines: list[str] = []
    for name, field in response_model.model_fields.items():
        annotation = getattr(field.annotation, "__name__", None) or str(field.annotation)
        description = field.description or ""
        lines.append(f'- "{name}": {annotation}. {description}'.strip())
    return "\n".join(lines)


def _strip_to_json_object(raw_text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text or "", flags=re.DOTALL | re.IGNORECASE).strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
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


def _complete_via_together_json_fallback(
    *,
    client: LLMClient,
    messages: list[dict[str, Any]],
    response_model: type[BaseModel],
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> BaseModel:
    api_key = getattr(client, "_api_key", None)
    if not api_key:
        raise ValueError(f"Missing Together API key for fallback model {client.model}")

    model_id = _direct_together_model_id(client.model)
    base_messages = _augment_messages_for_json(messages, response_model)
    last_error: Exception | None = None

    for attempt in range(3):
        payload = {
            "model": model_id,
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
            req = request.Request(_TOGETHER_API_URL, data=data, headers=headers, method="POST")
            with request.urlopen(req, timeout=timeout) as response:
                body = response.read().decode("utf-8")
            response_payload = json.loads(body)
            content = response_payload["choices"][0]["message"]["content"]
            parsed = _parse_fallback_response(content, response_model)
            try:
                cost = litellm.completion_cost(
                    model=client.model,
                    messages=base_messages,
                    completion=content,
                    call_type="completion",
                )
                if cost is not None:
                    client.add_cost(cost)
            except Exception:
                pass
            return parsed
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
                + [{"role": "assistant", "content": f"[previous attempt failed after {round(time.time() - started, 2)}s]"}]
                + [{"role": "user", "content": repair_instruction}]
            )
            time.sleep(2**attempt)

    raise ValueError(f"Together JSON fallback failed for {client.model}: {last_error}")


def _complete_structured(
    *,
    client: LLMClient,
    messages: list[dict[str, Any]],
    response_model: type[BaseModel],
    call_costs: list[dict[str, Any]],
    stage: str,
    max_tokens: int,
    temperature: float,
    timeout: int = 600,
) -> BaseModel:
    before_cost = client.cost_usd
    if _should_use_together_json_fallback(client.model):
        response = _complete_via_together_json_fallback(
            client=client,
            messages=messages,
            response_model=response_model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
    else:
        response = client.complete(
            messages,
            response_model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
    after_cost = client.cost_usd
    call_costs.append(
        {
            "stage": stage,
            "model": client.model,
            "prompt_chars": _message_chars(messages),
            "response_chars": _response_chars(response),
            "cost_usd": round(max(after_cost - before_cost, 0.0), 6),
        }
    )
    return response


def _section_char_limits(model: str) -> tuple[int, int, int]:
    if _should_use_together_json_fallback(model):
        return (
            _MISTRAL_INTRO_MAX_CHARS,
            _MISTRAL_METHOD_MAX_CHARS,
            _MISTRAL_CONCLUSION_MAX_CHARS,
        )
    return (_INTRO_MAX_CHARS, _METHOD_MAX_CHARS, _CONCLUSION_MAX_CHARS)


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

    table_count, table_examples = _extract_caption_examples(
        text,
        _TABLE_CAPTION_RE,
        prefix="Table",
        limit=4,
    )
    figure_count, figure_examples = _extract_caption_examples(
        text,
        _FIGURE_CAPTION_RE,
        prefix="Figure",
        limit=4,
    )

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


def _intro_sections(structure: PaperStructure) -> list[SectionInfo]:
    intro = [section for section in structure.sections if section.section_type == SectionType.INTRODUCTION]
    if intro:
        return intro[:1]
    non_ref = [
        section
        for section in structure.sections
        if section.section_type not in {SectionType.ABSTRACT, SectionType.REFERENCES, SectionType.APPENDIX}
    ]
    return non_ref[:1]


def _method_sections(structure: PaperStructure) -> list[SectionInfo]:
    methods = [
        section for section in structure.sections if section.section_type == SectionType.METHODOLOGY
    ]
    if methods:
        return methods[:3]

    title_keywords = (
        "method",
        "approach",
        "model",
        "architecture",
        "training",
        "setup",
        "experiment",
        "evaluation",
        "result",
        "analysis",
    )
    title_matches = [
        section
        for section in structure.sections
        if any(keyword in section.title.lower() for keyword in title_keywords)
    ]
    if title_matches:
        return title_matches[:3]

    fallback = [
        section
        for section in structure.sections
        if section.section_type
        not in {
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
    return [section for section in structure.sections if section.section_type == SectionType.CONCLUSION][:1]


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


def _parse_markdown_with_frontmatter(path: Path) -> tuple[dict[str, str], str]:
    raw = path.read_text(encoding="utf-8")
    if raw.startswith("---\n"):
        end_idx = raw.find("\n---\n", 4)
        if end_idx != -1:
            frontmatter = raw[4:end_idx]
            body = raw[end_idx + 5 :]
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
            (
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


def _fallback_committee_text(
    persona_specs: list[PersonaSpec],
    persona_reviews: dict[str, SlimConferenceReview],
    aggregate_scores: dict[str, float],
) -> CommitteeTextSections:
    ordered = sorted(persona_specs, key=lambda item: (-persona_reviews[item.slug].rating, item.slug))
    lead = persona_reviews[ordered[0].slug]
    weakness_fragments = " ".join(
        f"{persona.label}: {persona_reviews[persona.slug].weaknesses}"
        for persona in persona_specs
    )[:3000]
    question_fragments = " ".join(
        f"{persona.label}: {persona_reviews[persona.slug].questions}"
        for persona in persona_specs
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


def _focus_user_prompt(
    *,
    title: str,
    abstract: str,
    aspect_label: str,
    aspect_text: str,
    contribution_context: ContributionContext | None,
    structural_inventory: StructuralInventory,
) -> str:
    contribution_block = _render_contribution_context(contribution_context)
    return f"""\
Paper title: {title}

Abstract:
{abstract}

{_render_structural_inventory(structural_inventory)}

Contribution context:
{contribution_block}

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


def _render_review_markdown(title: str, review: SlimConferenceReview) -> str:
    lines = [
        f"# {title}",
        "",
        f"**Date**: {_now_date()}",
        f"**Recommendation**: {review.recommendation}",
        f"**Overall Rating**: {review.rating}",
        f"**Confidence**: {review.confidence}",
        f"**Soundness**: {review.soundness}",
        f"**Presentation**: {review.presentation}",
        f"**Contribution**: {review.contribution}",
        "",
        "## Summary",
        "",
        review.summary.strip(),
        "",
        "## Strengths",
        "",
        review.strength.strip(),
        "",
        "## Weaknesses",
        "",
        review.weaknesses.strip(),
        "",
        "## Questions",
        "",
        review.questions.strip(),
        "",
        "## Rationale",
        "",
        review.rationale.strip(),
        "",
    ]
    return "\n".join(lines)


def review_paper_slim(
    pdf_path: str | Path,
    *,
    model: str | None = None,
    config: CoarseConfig | None = None,
    title_hint: str | None = None,
    personas: list[str] | None = None,
    persona_weights: dict[str, float] | None = None,
) -> SlimPipelineResult:
    if config is None:
        config = load_config()

    resolved_model = model or config.default_model
    client = LLMClient(model=resolved_model, config=config)
    persona_specs = resolve_personas(personas)
    normalized_weights = _normalize_weights(persona_specs, persona_weights)

    paper_text = extract_file(pdf_path)
    normalized_text = _normalize_extracted_text(paper_text.full_markdown)
    structure = _heuristic_structure(paper_text)
    structural_inventory = _build_structural_inventory(normalized_text)
    if title_hint and title_hint.strip():
        structure = structure.model_copy(update={"title": title_hint.strip()})

    llm_calls = 0
    call_costs: list[dict[str, Any]] = []

    contribution_context = None
    if _should_use_together_json_fallback(client.model):
        call_costs.append(
            {
                "stage": "contribution_extraction",
                "model": client.model,
                "prompt_chars": len(structure.abstract),
                "response_chars": len(_render_contribution_context(None)),
                "cost_usd": 0.0,
            }
        )
    else:
        contribution_before = client.cost_usd
        contribution_context = extract_contribution(structure, client)
        contribution_after = client.cost_usd
        call_costs.append(
            {
                "stage": "contribution_extraction",
                "model": client.model,
                "prompt_chars": len(structure.abstract)
                + sum(len(section.text) for section in structure.sections[:3]),
                "response_chars": len(_render_contribution_context(contribution_context)),
                "cost_usd": round(max(contribution_after - contribution_before, 0.0), 6),
            }
        )
        llm_calls += 1

    intro_limit, method_limit, conclusion_limit = _section_char_limits(client.model)
    intro_text = _combine_sections(_intro_sections(structure), intro_limit) or structure.abstract
    method_text = _combine_sections(_method_sections(structure), method_limit)
    conclusion_text = _combine_sections(_conclusion_sections(structure), conclusion_limit)

    intro_messages = [
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
    ]
    intro_notes = _complete_structured(
        client=client,
        messages=intro_messages,
        response_model=FocusNotes,
        call_costs=call_costs,
        stage="intro_notes",
        max_tokens=2048,
        temperature=0.3,
    )
    llm_calls += 1

    method_notes = None
    if method_text.strip():
        method_messages = [
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
        ]
        method_notes = _complete_structured(
            client=client,
            messages=method_messages,
            response_model=FocusNotes,
            call_costs=call_costs,
            stage="method_notes",
            max_tokens=3072,
            temperature=0.3,
        )
        llm_calls += 1

    contribution_messages = [
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
    ]
    contribution_notes = _complete_structured(
        client=client,
        messages=contribution_messages,
        response_model=FocusNotes,
        call_costs=call_costs,
        stage="contribution_notes",
        max_tokens=2048,
        temperature=0.3,
    )
    llm_calls += 1

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
        persona_messages = [
            {"role": "system", "content": _persona_final_review_system(persona)},
            {"role": "user", "content": final_user_prompt},
        ]
        review = _complete_structured(
            client=client,
            messages=persona_messages,
            response_model=SlimConferenceReview,
            call_costs=call_costs,
            stage=f"persona_review:{persona.slug}",
            max_tokens=3072,
            temperature=0.25,
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

    if len(persona_specs) == 1:
        final_review = persona_reviews[persona_specs[0].slug]
    else:
        aggregate_scores = _aggregate_scores(persona_reviews, normalized_weights)
        committee["aggregate_scores"] = aggregate_scores
        llm_calls += 1
        try:
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
            committee_text = _complete_structured(
                client=client,
                messages=committee_messages,
                response_model=CommitteeTextSections,
                call_costs=call_costs,
                stage="committee_synthesis",
                max_tokens=3072,
                temperature=0.15,
            )
            committee["text_synthesis"] = "llm"
        except Exception:
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

    return SlimPipelineResult(
        markdown=_render_review_markdown(structure.title, final_review),
        review=final_review,
        paper_text=paper_text,
        title=structure.title,
        llm_calls=llm_calls,
        cost_usd=client.cost_usd,
        call_costs=call_costs,
        structural_inventory=structural_inventory,
        persona_reviews=persona_reviews,
        persona_markdowns=persona_markdowns,
        committee=committee,
    )
