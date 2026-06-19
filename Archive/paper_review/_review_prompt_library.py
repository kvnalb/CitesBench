#!/usr/bin/env python3
"""
File-backed review prompt and persona definitions.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPT_ROOT = CODE_ROOT / "prompts"
DEFAULT_REVIEW_PROMPT_DIR = DEFAULT_PROMPT_ROOT / "review"
DEFAULT_PERSONA_DIR = DEFAULT_PROMPT_ROOT / "personas"
DEFAULT_PERSONA_SLUG = "generic"
DEFAULT_PERSONA_ENSEMBLE = (
    "empiricist",
    "theorist",
    "systems_pragmatist",
    "novelty_gatekeeper",
)


@dataclass(frozen=True)
class PersonaSpec:
    slug: str
    label: str
    description: str
    path: Path
    instructions: str


@dataclass(frozen=True)
class ReviewPromptBundle:
    prompt_name: str
    prompt_source: str
    content_mode: str
    persona: PersonaSpec
    system_template_path: Path
    user_template_path: Path
    system_prompt: str
    user_template: str
    system_prompt_sha256: str
    user_prompt_template_sha256: str


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def render_prompt_template(
    template: str,
    variables: dict[str, str],
    allowed_unresolved: set[str] | None = None,
) -> str:
    rendered = template
    for key, value in variables.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    unresolved = sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", rendered)))
    if allowed_unresolved:
        unresolved = [token for token in unresolved if token not in allowed_unresolved]
    if unresolved:
        raise ValueError(f"Unresolved prompt variables: {', '.join(unresolved)}")
    return rendered


def available_persona_slugs(prompt_root: Path = DEFAULT_PROMPT_ROOT) -> list[str]:
    persona_dir = prompt_root / "personas"
    return sorted(path.stem for path in persona_dir.glob("*.md"))


def load_persona(persona_slug: str, prompt_root: Path = DEFAULT_PROMPT_ROOT) -> PersonaSpec:
    prompt_root = prompt_root.resolve()
    path = prompt_root / "personas" / f"{persona_slug}.md"
    if not path.exists():
        available = ", ".join(available_persona_slugs(prompt_root))
        raise ValueError(f"Unknown persona '{persona_slug}'. Available personas: {available}")
    meta, body = _parse_markdown_with_frontmatter(path)
    label = meta.get("label", persona_slug.replace("_", " ").title())
    description = meta.get("description", "")
    return PersonaSpec(
        slug=meta.get("slug", persona_slug),
        label=label,
        description=description,
        path=path,
        instructions=body.strip(),
    )


def resolve_personas(personas_arg: str, prompt_root: Path = DEFAULT_PROMPT_ROOT) -> list[PersonaSpec]:
    token = personas_arg.strip().lower()
    if token == "all":
        persona_slugs = [slug for slug in available_persona_slugs(prompt_root) if slug != DEFAULT_PERSONA_SLUG]
    elif token in {"default-ensemble", "committee4"}:
        persona_slugs = list(DEFAULT_PERSONA_ENSEMBLE)
    else:
        persona_slugs = [part.strip() for part in personas_arg.split(",") if part.strip()]
    if not persona_slugs:
        raise ValueError("At least one persona must be specified.")
    return [load_persona(slug, prompt_root=prompt_root) for slug in persona_slugs]


def build_review_prompt_bundle(
    content_mode: str,
    persona_slug: str,
    primary_area: str,
    prompt_root: Path = DEFAULT_PROMPT_ROOT,
) -> ReviewPromptBundle:
    prompt_root = prompt_root.resolve()
    review_dir = prompt_root / "review"
    system_template_path = review_dir / "system.md"
    user_template_name = "user_abstract.md" if content_mode == "abstract" else "user_fulltext.md"
    user_template_path = review_dir / user_template_name
    if not system_template_path.exists():
        raise ValueError(f"Missing system prompt file: {system_template_path}")
    if not user_template_path.exists():
        raise ValueError(f"Missing user prompt file: {user_template_path}")

    system_meta, system_template = _parse_markdown_with_frontmatter(system_template_path)
    _user_meta, user_template = _parse_markdown_with_frontmatter(user_template_path)
    persona = load_persona(persona_slug, prompt_root=prompt_root)

    system_prompt = render_prompt_template(
        system_template,
        {
            "PRIMARY_AREA": primary_area,
            "PERSONA_LABEL": persona.label,
            "PERSONA_INSTRUCTIONS": persona.instructions,
        },
    )
    user_prompt_template = render_prompt_template(
        user_template,
        {
            "PRIMARY_AREA": primary_area,
        },
        allowed_unresolved={"{{TITLE}}", "{{KEYWORDS}}", "{{EVIDENCE_DESCRIPTION}}", "{{CONTENT_LABEL}}", "{{CONTENT}}"},
    )

    prompt_name_root = system_meta.get("name", "review_prompt")
    prompt_name = f"{prompt_name_root}__{content_mode}__{persona.slug}"
    prompt_source = " + ".join(
        (
            str(system_template_path),
            str(user_template_path),
            str(persona.path),
        )
    )
    return ReviewPromptBundle(
        prompt_name=prompt_name,
        prompt_source=prompt_source,
        content_mode=content_mode,
        persona=persona,
        system_template_path=system_template_path,
        user_template_path=user_template_path,
        system_prompt=system_prompt,
        user_template=user_prompt_template,
        system_prompt_sha256=_sha256_text(system_prompt),
        user_prompt_template_sha256=_sha256_text(user_prompt_template),
    )


def render_review_user_prompt(
    user_template: str,
    paper: dict,
    content_meta: dict,
    primary_area: str,
) -> str:
    keywords = paper.get("keywords", "") or ""
    return render_prompt_template(
        user_template,
        {
            "PRIMARY_AREA": primary_area,
            "TITLE": str(paper.get("title", "")),
            "KEYWORDS": keywords if keywords else "Not provided",
            "EVIDENCE_DESCRIPTION": str(content_meta.get("evidence_description", "")),
            "CONTENT_LABEL": str(content_meta.get("content_label", "Content")),
            "CONTENT": str(content_meta.get("content", "")),
        },
    )
