# Prompt Library

This directory stores file-backed prompt definitions for the root `code/` review pipeline.

Layout:

- `review/`
  - shared review-system prompt
  - user-message templates for `abstract` and `fulltext`
- `personas/`
  - reviewer-lens definitions used to build persona-specific review prompts

The runner composes:

1. `review/system.md`
2. `review/user_abstract.md` or `review/user_fulltext.md`
3. one file from `personas/*.md`

The persona files define the reviewer lens. The shared review files define the rubric, calibration, and output format.
