"""
Flatten every leakage/recall probe run in the repo into one auditable ledger.

One row per (source file, model, probe, paper). Joins the parsed results to their reasoning
traces and, where possible, to the exact prompt that produced them.

Prompt provenance is recorded explicitly, because it differs by run and that matters:
  logged        the prompt string was stored at call time (runs after this ledger existed)
  reconstructed rebuilt now by calling the producing script's prompt function — valid only
                if that function has not changed since the run; `prompt_sha1` lets a future
                run detect drift
  unavailable   neither logged nor reconstructable (e.g. outputs/leakage_lap_traces.jsonl is
                0 bytes — the July LAP run captured no traces)

Placebo rows carry `plan_version`, because outputs/samples/oos_probe_plan.csv was rewritten
when the placebo design changed and the append-only trace file therefore holds both
generations (896 placebo traces for 448 plan rows). Rows whose probe_title is not in the
current plan are marked `v1_superseded` rather than silently mixed in.

Output: outputs/leakage_ledger.csv
Run:    python src/build/build_leakage_ledger.py [--selfcheck]
"""
import os
import re
import json
import glob
import hashlib
import argparse
from datetime import datetime, timezone

import pandas as pd

OUT = "outputs/leakage_ledger.csv"
PLAN = "outputs/samples/oos_probe_plan.csv"

COLS = ["source_file", "run_family", "model", "endpoint", "run_date", "arm", "probe",
        "paper_id", "sample_id", "probe_title", "probe_year", "true_year",
        "decision", "citations", "answer", "p_pos", "p_neg", "p_unk",
        "commitment", "direction", "logprob_read", "prompt_provenance", "prompt_sha1",
        "prompt_text", "completion", "completion_chars", "plan_version", "notes"]


def sha(s):
    return hashlib.sha1(str(s).encode()).hexdigest()[:12] if s else ""


def mtime(p):
    return datetime.fromtimestamp(os.path.getmtime(p), timezone.utc).date().isoformat() \
        if os.path.exists(p) else ""


def oos_rows():
    """Current out-of-sample runs: results CSV + trace JSONL + reconstructable prompt."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from run_oos_probes import prompt_for, CHAT_ONLY
        have_prompt = True
    except Exception as e:
        print(f"  ! cannot import run_oos_probes ({e}); prompts unavailable")
        have_prompt, CHAT_ONLY = False, ()

    plan_titles = set()
    if os.path.exists(PLAN):
        plan_titles = set(pd.read_csv(PLAN)["probe_title"])

    rows = []
    for f in sorted(glob.glob("outputs/oos_probes_*.csv")):
        d = pd.read_csv(f)
        slug = os.path.basename(f)[len("oos_probes_"):-4]
        tf = f"outputs/oos_traces_{slug}.jsonl"
        traces, model_seen = {}, ""
        if os.path.exists(tf):
            for line in open(tf):
                try:
                    j = json.loads(line)
                except json.JSONDecodeError:
                    continue
                model_seen = j.get("model", model_seen)
                # last write wins per (sample_id, probe, title) — keeps v1 and v2 distinct
                traces[(j["sample_id"], j["probe"], j.get("probe_title"))] = j
        chat = any(k in model_seen.lower() for k in CHAT_ONLY) if model_seen else False
        for r in d.itertuples():
            key = None
            for (sid, pr, title), j in traces.items():
                if sid == r.sample_id and pr == r.probe:
                    if title in plan_titles or r.probe != "placebo":
                        key = (sid, pr, title)
                        break
                    key = key or (sid, pr, title)
            t = traces.get(key, {}) if key else {}
            title = t.get("probe_title", "")
            prompt, prov = "", "unavailable"
            if have_prompt and title:
                prompt = prompt_for(r.probe, title, int(r.probe_year), raw=chat)
                prov = "reconstructed"
            rows.append({
                "source_file": os.path.basename(f), "run_family": "oos_probes",
                "model": model_seen or slug,
                "endpoint": "chat/completions" if chat else "completions",
                "run_date": mtime(f), "arm": r.arm, "probe": r.probe,
                "paper_id": r.paper_id, "sample_id": r.sample_id,
                "probe_title": title, "probe_year": r.probe_year, "true_year": r.true_year,
                "decision": r.decision_class, "citations": r.s2_citations,
                "answer": r.answer, "p_pos": r.p_pos, "p_neg": r.p_neg, "p_unk": r.p_unk,
                "commitment": r.commitment, "direction": r.direction,
                "logprob_read": r.logprob_read,
                "prompt_provenance": prov, "prompt_sha1": sha(prompt), "prompt_text": prompt,
                "completion": (t.get("completion") or "").strip(),
                "completion_chars": len((t.get("completion") or "")),
                "plan_version": ("current" if title in plan_titles else
                                 ("v1_superseded" if title else "")),
                "notes": "" if key else "no trace matched",
            })
    return rows


def legacy_rows():
    """July probe runs. Traces were not captured (leakage_lap_traces.jsonl is empty), so
    completions are unavailable; prompts are reconstructed from the scripts' own functions."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    specs = [
        ("outputs/leakage_lap_v1.csv", "leakage_lap_v1", "leakage_lap_v1", "recall_prompt",
         {"answer": "answer", "p_pos": "p_accept", "p_neg": "p_reject", "p_unk": "p_unknown",
          "commitment": "lap", "direction": "ud"}),
        ("outputs/leakage_fame_v1.csv", "leakage_fame_v1", "leakage_fame_v1", "fame_prompt",
         {"answer": "answer", "p_pos": "p_high", "p_neg": "p_low", "p_unk": "p_unknown",
          "commitment": "fame", "direction": "fame_ud"}),
        ("outputs/leakage_controls.csv", "leakage_controls", "leakage_controls", None,
         {"answer": "answer", "p_pos": "p_accept", "p_neg": "p_reject", "p_unk": "p_unknown",
          "commitment": "lap", "direction": "ud"}),
    ]
    titles = {}
    if os.path.exists("outputs/eval_table.csv"):
        ev = pd.read_csv("outputs/eval_table.csv")
        titles = dict(zip(ev["paper_id"], ev["title"]))

    rows = []
    for path, family, mod_name, fn_name, m in specs:
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        model, fn = "", None
        try:
            mod = __import__(mod_name)
            model = getattr(mod, "MODEL", "")
            fn = getattr(mod, fn_name, None) if fn_name else None
        except Exception as e:
            print(f"  ! {mod_name}: {e}")
        for r in d.itertuples():
            pid = getattr(r, "paper_id", getattr(r, "probe_id", ""))
            title = titles.get(pid, "")
            yr = getattr(r, "year", getattr(r, "year_asked", ""))
            prompt, prov = "", "unavailable"
            if fn and title:
                try:
                    prompt, prov = fn(title, yr), "reconstructed"
                except Exception:
                    pass
            rows.append({
                "source_file": os.path.basename(path), "run_family": family,
                "model": model, "endpoint": "chat/completions",
                "run_date": mtime(path), "arm": "contaminated",
                "probe": getattr(r, "probe", family.replace("leakage_", "").replace("_v1", "")),
                "paper_id": pid, "sample_id": "", "probe_title": title,
                "probe_year": yr, "true_year": getattr(r, "year", ""),
                "decision": getattr(r, "decision", ""),
                "citations": getattr(r, "citation_pct_rank", ""),
                "answer": getattr(r, m["answer"], ""),
                "p_pos": getattr(r, m["p_pos"], ""), "p_neg": getattr(r, m["p_neg"], ""),
                "p_unk": getattr(r, m["p_unk"], ""),
                "commitment": getattr(r, m["commitment"], ""),
                "direction": getattr(r, m["direction"], ""), "logprob_read": "",
                "prompt_provenance": prov, "prompt_sha1": sha(prompt), "prompt_text": prompt,
                "completion": "", "completion_chars": 0, "plan_version": "",
                "notes": "traces not captured by this run (leakage_lap_traces.jsonl is 0 bytes)",
            })
    return rows


def build():
    rows = oos_rows() + legacy_rows()
    df = pd.DataFrame(rows, columns=COLS)
    df.to_csv(OUT, index=False)
    print(f"\nWrote {OUT} — {len(df):,} rows")
    print("\nby run_family x model:")
    print(df.groupby(["run_family", "model"]).size().to_string())
    print("\nprompt provenance:")
    print(df["prompt_provenance"].value_counts().to_string())
    print("\ntrace coverage (completion present):")
    print(df.assign(has=df["completion_chars"].gt(0)).groupby("run_family")["has"]
          .agg(["sum", "size"]).to_string())
    if (df["plan_version"] == "v1_superseded").any():
        print(f"\nsuperseded placebo rows flagged: "
              f"{int((df['plan_version'] == 'v1_superseded').sum()):,}")
    return df


def selfcheck(df):
    """Invariants that must hold, or the ledger is not trustworthy."""
    bad = []
    dup = df.duplicated(subset=["source_file", "model", "probe", "paper_id", "plan_version"])
    if dup.any():
        bad.append(f"{int(dup.sum())} duplicate (source,model,probe,paper,plan_version) keys")
    m = df["prompt_text"].astype(str).str.len().gt(0) & df["prompt_sha1"].eq("")
    if m.any():
        bad.append(f"{int(m.sum())} rows with a prompt but no sha1")
    m = df["prompt_provenance"].eq("reconstructed") & df["prompt_text"].astype(str).str.len().eq(0)
    if m.any():
        bad.append(f"{int(m.sum())} rows claim reconstructed prompt but store none")
    # a committed answer must not have all-zero probabilities where logprobs were read
    m = (df["logprob_read"].astype(str).eq("1")
         & df["answer"].isin(["accepted", "rejected", "high", "low"])
         & df[["p_pos", "p_neg"]].apply(pd.to_numeric, errors="coerce").sum(axis=1).le(0.01))
    if m.any():
        bad.append(f"{int(m.sum())} committed rows with zero positive+negative mass")
    # and an 'unknown' answer must not carry directional mass
    m = (df["logprob_read"].astype(str).eq("1") & df["answer"].eq("unknown")
         & df[["p_pos", "p_neg"]].apply(pd.to_numeric, errors="coerce").sum(axis=1).ge(0.5))
    if m.any():
        bad.append(f"{int(m.sum())} 'unknown' rows carrying >=0.5 directional mass "
                   "(logprob extraction is misassigning — known open bug on the chat path)")
    print("\n=== selfcheck ===")
    for b in bad:
        print(f"  FAIL {b}")
    if not bad:
        print("  all invariants hold")
    return not bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    df = build()
    if a.selfcheck:
        selfcheck(df)
