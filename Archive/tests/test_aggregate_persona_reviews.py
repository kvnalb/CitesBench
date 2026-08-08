from __future__ import annotations

from conftest import load_aggregate_module

agg = load_aggregate_module()


def _persona_row(slug: str, label: str, rationale: str, scores: dict | None = None) -> dict:
    return {
        "paper_id": "P1",
        "model": {"id": "modelX", "label": "Model X"},
        "prompt": {"persona_slug": slug, "persona_label": label},
        "llm_review": {
            "rationale": rationale,
            "scores": scores or {"rating": 5.0, "soundness": 3.0, "confidence": 4.0,
                                 "presentation": 3.0, "contribution": 3.0},
            "parsed_ok": True,
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "elapsed_seconds": 1.0,
        },
    }


def test_parse_editorial_response_clean_json():
    raw = '{"summary": "x", "themes": [{"title":"t","personas":["a"],"consolidated":"c"}], "contradictions": []}'
    r = agg.parse_editorial_response(raw)
    assert r["summary"] == "x"
    assert r["themes"][0]["title"] == "t"
    assert r["themes"][0]["personas"] == ["a"]
    assert r["contradictions"] == []
    assert r["parsed_ok"] is True


def test_parse_editorial_response_handles_chatter_around_json():
    raw = 'Sure, here it is:\n{"summary":"y","themes":[],"contradictions":[]}\nDone.'
    r = agg.parse_editorial_response(raw)
    assert r["summary"] == "y"
    assert r["parsed_ok"] is True


def test_parse_editorial_response_garbage_returns_safe_default():
    r = agg.parse_editorial_response("totally not json")
    assert r["parsed_ok"] is False
    assert r["summary"] == ""
    assert r["themes"] == []
    assert r["contradictions"] == []


def test_parse_editorial_response_handles_string_personas():
    raw = '{"summary":"s","themes":[{"title":"t","personas":"only_one","consolidated":"c"}],"contradictions":[]}'
    r = agg.parse_editorial_response(raw)
    assert r["themes"][0]["personas"] == ["only_one"]


def test_build_editorial_user_message_contains_all_personas():
    rows = [
        _persona_row("theorist", "Theorist", "too few proofs."),
        _persona_row("empiricist", "Empiricist", "good ablations."),
    ]
    msg = agg.build_editorial_user_message(rows)
    assert "theorist" in msg
    assert "empiricist" in msg
    assert "too few proofs" in msg
    assert "good ablations" in msg


def test_build_editorial_user_message_skips_empty_rationales():
    rows = [
        _persona_row("a", "A", ""),
        _persona_row("b", "B", "kept"),
    ]
    msg = agg.build_editorial_user_message(rows)
    assert "kept" in msg
    assert "Persona slug: a" not in msg


def test_resolve_editorial_model_aliases():
    spec = agg.resolve_editorial_model("gpt-oss-120b")
    assert spec.label == "GPT-OSS-120B"
    assert spec.model_id == "openai/gpt-oss-120b"


def test_resolve_editorial_model_bare_model_id_passes_through():
    spec = agg.resolve_editorial_model("some-custom-model/v1")
    assert spec.model_id == "some-custom-model/v1"
    assert spec.label == "some-custom-model/v1"


def test_build_committee_row_no_editorial_preserves_legacy_behavior():
    rows = [
        _persona_row("theorist", "Theorist", "proof unclear",
                     {"rating": 5.0, "soundness": 3.0, "confidence": 4.0, "presentation": 3.0, "contribution": 3.0}),
        _persona_row("empiricist", "Empiricist", "great ablations",
                     {"rating": 7.0, "soundness": 4.0, "confidence": 4.0, "presentation": 4.0, "contribution": 3.0}),
    ]
    out = agg.build_committee_row(
        model_id="modelX",
        model_label="Model X",
        paper_id="P1",
        member_rows=rows,
        weights={"theorist": 1.0, "empiricist": 1.0},
        editorial_config=None,
    )
    # Scores: equal-weighted average
    assert out["llm_review"]["scores"]["rating"] == 6.0
    # Parser tag: legacy (no editorial)
    assert out["llm_review"]["parser"] == "committee_weighted_average"
    # Rationale: joined raw concatenation
    assert "Theorist" in out["llm_review"]["rationale"]
    assert "Empiricist" in out["llm_review"]["rationale"]
    # Editorial bookkeeping: preserved as None
    assert out["committee"]["editorial"] is None
    assert out["committee"]["raw_concatenated_rationale"] == out["llm_review"]["rationale"]
    assert out["prompt"]["editorial_dedup"] is False
