# Code Layout

The root `code/` directory now separates the two active workstreams:

- `paper_review/`
  - paper-level / abstract-level review generation and comparison
  - persona prompts and committee aggregation
- `pairwise_review/`
  - pairwise ranking, Swiss/anchor scheduling, and model-sweep utilities

The numbered scripts kept at the top level are compatibility launchers so existing commands do not break.

The root `code/` directory should stay minimal:

- top-level numbered wrappers only
- this README
- no canonical shared modules
- no canonical prompt files

Canonical code for new work should go into the appropriate subfolder.
