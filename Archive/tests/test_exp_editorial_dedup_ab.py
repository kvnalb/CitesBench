import json

import pytest

from conftest import load_editorial_ab_module

h = load_editorial_ab_module()


def test_token_count_basic():
    assert h.token_count("hello world foo") == 3


def test_flesch_kincaid_grade_returns_a_grade_for_a_sentence():
    fk = h.flesch_kincaid_grade("The quick brown fox jumps over the lazy dog. Then it sleeps.")
    assert fk is not None
    assert 0 < fk < 20


def test_flesch_kincaid_grade_returns_none_for_empty_text():
    assert h.flesch_kincaid_grade("") is None


def test_count_syllables_min_one():
    assert h.count_syllables("a") == 1
    assert h.count_syllables("rhythm") >= 1


def test_normalise_winner_recognizes_aliases():
    assert h.normalise_winner("a") == "A"
    assert h.normalise_winner("FIRST") == "A"
    assert h.normalise_winner("right") == "B"
    assert h.normalise_winner("anything else") == "Tie"


def test_invert_winner():
    assert h.invert_winner("A") == "B"
    assert h.invert_winner("B") == "A"
    assert h.invert_winner("Tie") == "Tie"


def test_parse_judge_response_clean_json():
    raw = '{"clarity":"A","redundancy_freedom":"B","contradiction_freedom":"Tie","reasoning":"r"}'
    r = h.parse_judge_response(raw)
    assert r["clarity"] == "A"
    assert r["redundancy_freedom"] == "B"
    assert r["contradiction_freedom"] == "Tie"
    assert r["parsed_ok"] is True
    assert r["reasoning"] == "r"


def test_parse_judge_response_garbage():
    r = h.parse_judge_response("no json here")
    assert r["parsed_ok"] is False
    for dim in h.DIMENSIONS:
        assert r[dim] == "Tie"


def test_aggregate_calls_consensus_per_dimension():
    calls = [
        {"clarity": "A", "redundancy_freedom": "A", "contradiction_freedom": "Tie", "parsed_ok": True},
        {"clarity": "A", "redundancy_freedom": "B", "contradiction_freedom": "Tie", "parsed_ok": True},
    ]
    out = h.aggregate_calls(calls)
    assert out["clarity"] == "A"
    assert out["redundancy_freedom"] == "Tie"  # disagreement -> Tie
    assert out["contradiction_freedom"] == "Tie"


def test_aggregate_calls_ignores_unparsed_calls():
    calls = [
        {"clarity": "A", "redundancy_freedom": "A", "contradiction_freedom": "Tie", "parsed_ok": False},
        {"clarity": "B", "redundancy_freedom": "B", "contradiction_freedom": "B", "parsed_ok": True},
    ]
    out = h.aggregate_calls(calls)
    assert out["clarity"] == "B"


def _baseline_row(rationale: str = "X | Y | Z") -> dict:
    return {
        "committee": {"raw_concatenated_rationale": rationale},
        "llm_review": {"rationale": rationale, "parser": "committee_weighted_average"},
    }


def _editorial_row(summary: str = "XY", parsed_ok: bool = True) -> dict:
    return {
        "committee": {
            "raw_concatenated_rationale": "X | Y | Z",
            "editorial": {"parsed_ok": parsed_ok, "summary": summary},
        },
        "llm_review": {
            "rationale": summary if parsed_ok else "X | Y | Z",
            "parser": "committee_weighted_average__editorial_dedup",
        },
    }


def test_extract_texts_returns_both_when_editorial_succeeded():
    raw, ed, dbg = h.extract_texts(_baseline_row(), _editorial_row())
    assert raw == "X | Y | Z"
    assert ed == "XY"
    assert dbg["editorial_parsed_ok"] is True


def test_extract_texts_returns_none_for_editorial_when_parse_failed():
    raw, ed, dbg = h.extract_texts(_baseline_row(), _editorial_row(summary="", parsed_ok=False))
    assert raw == "X | Y | Z"
    assert ed is None
    assert dbg["editorial_parsed_ok"] is False


def test_summarise_winrates_and_proxy_means():
    per_paper = [
        {"judgment": {"clarity": "A", "redundancy_freedom": "B", "contradiction_freedom": "Tie"},
         "proxies": {"raw": {"token_count": 100, "fk_grade": 10.0},
                     "editorial": {"token_count": 50, "fk_grade": 9.0}}},
        {"judgment": {"clarity": "B", "redundancy_freedom": "B", "contradiction_freedom": "B"},
         "proxies": {"raw": {"token_count": 120, "fk_grade": 12.0},
                     "editorial": {"token_count": 60, "fk_grade": 10.0}}},
    ]
    s = h.summarise(per_paper)
    assert s["n_papers"] == 2
    assert s["win_rates"]["redundancy_freedom"]["editorial_pct"] == 100.0
    assert s["proxies"]["raw_tokens"]["mean"] == 110
    assert s["proxies"]["editorial_tokens"]["mean"] == 55


def test_read_editor_model_id_missing_manifest(tmp_path):
    assert h.read_editor_model_id(tmp_path) is None


def test_read_editor_model_id_returns_model_when_enabled(tmp_path):
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"editorial_dedup": {"enabled": True, "model_id": "openai/gpt-oss-120b"}})
    )
    assert h.read_editor_model_id(tmp_path) == "openai/gpt-oss-120b"


def test_read_editor_model_id_returns_none_when_disabled(tmp_path):
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"editorial_dedup": {"enabled": False, "model_id": None}})
    )
    assert h.read_editor_model_id(tmp_path) is None


def test_resolve_model_alias_and_bare():
    spec = h.resolve_model("deepseek-r1")
    assert spec.label == "DeepSeek-R1"
    bare = h.resolve_model("acme/v9")
    assert bare.model_id == "acme/v9"
