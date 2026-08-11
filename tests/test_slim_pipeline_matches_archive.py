"""
Differential test: does src/probes/slim_pipeline.py send the same prompts as the
archived slim_coarse_pipeline.py?

The port exists so the 2025 run is the same instrument as the 2018-2020 runs. Nothing
else in this repo checks that, and the port has already drifted twice — once by forcing
a 9th call the archive deliberately skips, once by raising every stage's max_tokens to
a model-wide floor. Both looked harmless and both would have made the two eras
incomparable. This test is the thing that catches that class of bug.

What it checks, by stubbing the HTTP call in both modules and capturing what each one
would have sent:
  - the stage sequence and count
  - per-stage max_tokens and temperature
  - the exact assembled message bytes, system and user, per stage

What it cannot check: model output (temperature is 0.15-0.3, so nothing is
reproducible) and extraction fidelity (the archive's primary extractor is Mistral OCR
via OpenRouter; ours is Docling — different tools, no test makes them agree).

The archive module raises at import time: find_repo_root() wants sibling Code/ and
Report/ directories. Rather than edit Archive/ (append-only by convention), this
builds a throwaway tree in tmp and imports from there. resolve() follows symlinks, so
the copy has to be real.

Run: python tests/test_slim_pipeline_matches_archive.py
"""
import os
import sys
import json
import shutil
import difflib
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

PAPER_ID = "zzR1Uskhj0"          # any 2025 paper; the text is what matters, not which
PERSONAS = ["empiricist", "theorist", "systems_pragmatist", "novelty_gatekeeper"]


def load_archive_module(tmpdir):
    """Import the archived pipeline from a tree that satisfies find_repo_root()."""
    root = os.path.join(tmpdir, "archroot")
    os.makedirs(os.path.join(root, "Code"), exist_ok=True)
    os.makedirs(os.path.join(root, "Report"), exist_ok=True)
    shutil.copytree(os.path.join(REPO, "Archive", "CompletePipeline"),
                    os.path.join(root, "Code", "CompletePipeline"))
    sys.path.insert(0, os.path.join(root, "Code", "CompletePipeline", "llm"))
    import slim_coarse_pipeline
    return slim_coarse_pipeline


def canned(response_model):
    """A schema-valid reply, so both pipelines proceed through every stage instead of
    dying at the first parse. Fields are filled from the model's own definition."""
    vals = {}
    for name, f in response_model.model_fields.items():
        ann = str(f.annotation)
        if "list" in ann:
            vals[name] = ["x"]
        elif "float" in ann or "int" in ann:
            lo = next((getattr(m, "ge", None) for m in f.metadata
                       if getattr(m, "ge", None) is not None), None)
            vals[name] = float(lo if lo is not None else 1.0)
        else:
            vals[name] = "x"
    return response_model(**vals)


def capture_archive(mod, markdown):
    """Run the archive pipeline with its network call stubbed; return per-stage calls."""
    calls = []

    def fake(*, client, messages, response_model, max_tokens, temperature, timeout):
        calls.append({"messages": messages, "max_tokens": max_tokens,
                      "temperature": temperature, "model": response_model.__name__})
        return canned(response_model)

    mod._complete_via_together_json_fallback = fake
    # force the Together branch: that is the path the 2018-2020 runs actually took
    mod._should_use_together_json_fallback = lambda model: True

    with tempfile.NamedTemporaryFile("w", suffix=".pdf", delete=False) as fh:
        fh.write("stub")
        pdf = fh.name
    mod.extract_file = lambda path: mod.PaperText(full_markdown=markdown,
                                                  token_estimate=len(markdown) // 4)
    try:
        mod.review_paper_slim(pdf, model="together_ai/google/gemma-4-31B-it",
                              personas=PERSONAS)
    except Exception as e:                       # stages past the diff point may fail
        print(f"  (archive stopped after {len(calls)} calls: {type(e).__name__})")
    finally:
        os.unlink(pdf)
    return calls


def capture_port(markdown):
    from probes import slim_pipeline as P
    calls = []

    def fake(*, model, messages, response_model, max_tokens, temperature, timeout):
        calls.append({"messages": messages, "max_tokens": max_tokens,
                      "temperature": temperature, "model": response_model.__name__})
        return canned(response_model), {"usage": {}, "raw_content": "{}", "attempts": 1,
                                        "latency_s": 0.0, "finish_reason": "stop"}

    P._together_json = fake
    try:
        P.review_paper_slim(paper_id=PAPER_ID, markdown=markdown,
                            model_key="gemma", personas=PERSONAS)
    except Exception as e:
        print(f"  (port stopped after {len(calls)} calls: {type(e).__name__})")
    return calls


def compare(a_calls, p_calls):
    problems = []
    if len(a_calls) != len(p_calls):
        problems.append(f"CALL COUNT: archive={len(a_calls)} port={len(p_calls)}")

    for i, (a, p) in enumerate(zip(a_calls, p_calls), 1):
        if a["max_tokens"] != p["max_tokens"]:
            problems.append(f"call {i} ({a['model']}): max_tokens "
                            f"archive={a['max_tokens']} port={p['max_tokens']}")
        if abs(a["temperature"] - p["temperature"]) > 1e-9:
            problems.append(f"call {i} ({a['model']}): temperature "
                            f"archive={a['temperature']} port={p['temperature']}")
        if a["model"] != p["model"]:
            problems.append(f"call {i}: schema archive={a['model']} port={p['model']}")

        at = "\n".join(f"[{m['role']}]\n{m['content']}" for m in a["messages"])
        pt = "\n".join(f"[{m['role']}]\n{m['content']}" for m in p["messages"])
        if at != pt:
            d = list(difflib.unified_diff(at.splitlines(), pt.splitlines(),
                                          "archive", "port", lineterm="", n=1))
            problems.append(f"call {i} ({a['model']}): PROMPT DIFFERS "
                            f"({len(at)} vs {len(pt)} chars)\n    " +
                            "\n    ".join(d[:12]))
    return problems


def main():
    import pandas as pd
    sys.path.insert(0, os.path.join(REPO, "src"))
    from build.build_slim_2025_papers import load_year
    from build.normalize_paper_markdown import normalize

    row = load_year(2025).iloc[0]
    markdown = normalize(row.markdown)
    print(f"paper {row.forum_id}: {len(markdown):,} chars\n")

    with tempfile.TemporaryDirectory() as tmp:
        arch = load_archive_module(tmp)
        print("archive:")
        a_calls = capture_archive(arch, markdown)
        print("port:")
        p_calls = capture_port(markdown)

    print(f"\narchive stages: {[c['model'] for c in a_calls]}")
    print(f"port stages:    {[c['model'] for c in p_calls]}")

    problems = compare(a_calls, p_calls)
    if problems:
        print(f"\n{len(problems)} DIFFERENCE(S):\n")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print(f"\nOK — {len(a_calls)} calls identical in sequence, parameters and prompt bytes")


if __name__ == "__main__":
    main()
