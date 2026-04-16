# Prompt Library

This directory stores file-backed prompt definitions for the pairwise review pipeline.

Layout:

- `pairwise/`
  - shared pairwise system prompt
  - user-message templates for `abstract` and `fulltext`
  - schema fragments for `simple` and `detailed`
  - strength fragments for `standard`, `strong`, and `anti-hype`
- `personas/`
  - reviewer-lens definitions used to build persona-specific pairwise prompts
- `review/`
  - copied paper-level review prompts so the pairwise codebase remains self-contained

The pairwise runner composes:

1. `pairwise/system.md`
2. `pairwise/user_abstract.md` or `pairwise/user_fulltext.md`
3. one schema file from `pairwise/schema_*.md`
4. one strength file from `pairwise/strength_*.md`
5. one file from `personas/*.md`

The persona files define the lens. The pairwise prompt files define the comparison task, calibration, and output schema.
