"""
Run the frozen out-of-sample probe plan against a model, keeping full reasoning traces.

This is the indicative version: ~450 papers x 4 probe variants per model, powered to detect
a ~16pp rate difference between arms. The point is not a precise estimate — it is (a) does
outcome recall drop off across the contamination gradient, and (b) what does the model
actually say when it claims to recall something. So every call's full completion is written
to a JSONL alongside the parsed answer, because the traces are the evidence, not the rates.

Prompting asks for brief recollection reasoning and then a one-word verdict on its own line,
so the answer token's logprobs are readable while the reasoning stays inspectable:

    ...reasoning...
    ANSWER: accepted

Probabilities are read from the top-5 logprobs at the answer-token position and renormalized
over the three allowed answers. Two derived measures, following the existing LAP probe:
    commitment = p(positive) + p(negative)   how sure the model is that it knows
    direction  = p(positive) - p(negative)   which way it leans

Together serves logprobs on /v1/completions but not on /v1/chat/completions, hence the
completions endpoint and explicit chat templating.

Output: outputs/oos_probes_<model>.csv    one row per call
        outputs/oos_traces_<model>.jsonl  full prompt + completion, for reading
Report: python src/run_oos_probes.py --report

Run: python src/run_oos_probes.py --model meta-llama/Llama-3.3-70B-Instruct-Turbo
     python src/run_oos_probes.py --model google/gemma-4-31B-it        # contaminated comparator
"""
import os
import re
import csv
import sys
import json
import math
import time
import hashlib
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import load

os.makedirs("outputs", exist_ok=True)
PLAN = "outputs/samples/oos_probe_plan.csv"
URL = "https://api.together.xyz/v1/completions"
CHAT_URL = "https://api.together.xyz/v1/chat/completions"
CHAT_ONLY = ("gemma",)   # models Together serves only via the chat endpoint
THINKING = ("gemma-4",)  # emit a long reasoning trace before the answer
KEY = next((l.split("=", 1)[1].strip() for l in open(".env")
            if l.startswith("TOGETHER_API_KEY")), "")
HEADERS = {"Authorization": f"Bearer {KEY}"}

ANSWERS = {                      # probe -> (positive, negative, unknown)
    "lap": ("accepted", "rejected", "unknown"),
    "placebo": ("accepted", "rejected", "unknown"),
    "wrongyear": ("accepted", "rejected", "unknown"),
    "fame": ("high", "low", "unknown"),
}
COLS = ["sample_id", "arm", "probe", "paper_id", "probe_year", "true_year", "decision_class",
        "s2_citations", "stratum", "answer", "p_pos", "p_neg", "p_unk", "commitment",
        "direction", "logprob_read", "n_completion_tokens", "prompt_sha1", "endpoint"]


def slug(m):
    return m.split("/")[-1].replace(".", "-")


def prompt_for(probe, title, year, raw=False):
    # lap / placebo / wrongyear all ask the accept-reject question; only fame differs
    q = load(f"recall/{'fame' if probe == 'fame' else 'lap'}_cot", title=title, year=year)
    if raw:
        return q
    return f"<|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|>" \
           f"<|start_header_id|>assistant<|end_header_id|>\n\n"


def read_answer(text, tokens, top_logprobs, probe):
    """Parse the word after ANSWER: and, if the token stream lines up, its probabilities."""
    pos, neg, unk = ANSWERS[probe]
    m = re.search(r"ANSWER:\s*([A-Za-z]+)", text or "")
    answer = m.group(1).lower() if m else ""
    answer = answer if answer in (pos, neg, unk) else ("other" if answer else "")

    probs, read = {}, False
    if tokens and top_logprobs:
        # "ANSWER:" spans several tokens ('ANS','WER',':'), so locate it by character
        # offset in the reconstructed text and map back to a token index
        anchor, run = None, ""
        for i, t in enumerate(tokens):
            run += str(t)
            if anchor is None and re.search(r"ANSWER\s*:\s*$", run):
                anchor = i
                break
        if anchor is not None:
            for j in range(anchor + 1, min(anchor + 5, len(tokens))):
                cand = top_logprobs[j] if j < len(top_logprobs) else None
                if not isinstance(cand, dict):
                    continue
                acc = {}
                for tok, lp in cand.items():
                    w = str(tok).strip().lower().rstrip(".,:!?")
                    for target in (pos, neg, unk):
                        if w and (target.startswith(w) or w.startswith(target[:4])):
                            acc[target] = acc.get(target, 0.0) + math.exp(lp)
                if acc:
                    probs, read = acc, True
                    break
    tot = sum(probs.values()) or 1.0
    p_pos, p_neg, p_unk = (probs.get(pos, 0) / tot, probs.get(neg, 0) / tot,
                           probs.get(unk, 0) / tot)
    return answer, p_pos, p_neg, p_unk, read


def call(model, probe, title, year, max_tokens, retries=4):
    chat = any(k in model.lower() for k in CHAT_ONLY)
    if chat:
        url = CHAT_URL
        body = {"model": model, "messages": [{"role": "user",
                "content": prompt_for(probe, title, year, raw=True)}],
                "max_tokens": max_tokens, "temperature": 0,
                "logprobs": True, "top_logprobs": 5}
    else:
        url = URL
        body = {"model": model, "prompt": prompt_for(probe, title, year),
                "max_tokens": max_tokens, "temperature": 0, "logprobs": 5,
                "stop": ["<|eot_id|>", "\n\n\n"]}
    for a in range(retries):
        try:
            r = requests.post(url, headers=HEADERS, json=body, timeout=120)
        except requests.RequestException:
            time.sleep(2 * (a + 1)); continue
        if r.status_code == 200:
            ch = (r.json().get("choices") or [{}])[0]
            if chat:
                text = ((ch.get("message") or {}).get("content")) or ""
                lp = ch.get("logprobs") or {}
                content = lp.get("content") or []
                toks = [e.get("token") for e in content]
                tops = [{t["token"]: t["logprob"] for t in (e.get("top_logprobs") or [])}
                        for e in content]
                return text, toks, tops
            lp = ch.get("logprobs") or {}
            return ch.get("text", ""), lp.get("tokens"), lp.get("top_logprobs")
        if r.status_code in (429, 500, 503):
            time.sleep(3 * (a + 1)); continue
        return None, None, None          # 4xx: model unavailable, don't hammer
    return None, None, None


def run(model, limit, workers, max_tokens):
    if any(k in model.lower() for k in THINKING) and max_tokens < 1500:
        max_tokens = 2200
        print(f"  thinking model — raising max_tokens to {max_tokens}")
    plan = pd.read_csv(PLAN)
    out_csv = f"outputs/oos_probes_{slug(model)}.csv"
    traces = f"outputs/oos_traces_{slug(model)}.jsonl"
    done = set()
    if os.path.exists(out_csv):
        d = pd.read_csv(out_csv)
        done = set(zip(d["sample_id"], d["probe"]))
    todo = plan[~plan.apply(lambda r: (r["sample_id"], r["probe"]) in done, axis=1)]
    if limit:
        todo = todo.head(limit)
    print(f"{model}\n  {len(done):,} done, {len(todo):,} to go, {workers} workers")

    new = not os.path.exists(out_csv)
    fh = open(out_csv, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=COLS)
    if new:
        w.writeheader()
    tf = open(traces, "a")
    lock = threading.Lock()
    n_done = [0]

    def work(r):
        chat = any(k in model.lower() for k in CHAT_ONLY)
        sent = prompt_for(r.probe, r.probe_title, int(r.probe_year), raw=chat)
        text, toks, tops = call(model, r.probe, r.probe_title, int(r.probe_year), max_tokens)
        if text is None:
            return
        answer, p_pos, p_neg, p_unk, read = read_answer(text, toks, tops, r.probe)
        row = {"sample_id": r.sample_id, "arm": r.arm, "probe": r.probe,
               "paper_id": r.paper_id, "probe_year": r.probe_year, "true_year": r.true_year,
               "decision_class": r.decision_class, "s2_citations": r.s2_citations,
               "stratum": r.stratum, "answer": answer, "p_pos": round(p_pos, 4),
               "p_neg": round(p_neg, 4), "p_unk": round(p_unk, 4),
               "commitment": round(p_pos + p_neg, 4), "direction": round(p_pos - p_neg, 4),
               "logprob_read": int(read), "n_completion_tokens": len(toks or []),
               "prompt_sha1": hashlib.sha1(sent.encode()).hexdigest()[:12],
               "endpoint": "chat/completions" if chat else "completions"}
        with lock:
            w.writerow(row); fh.flush()
            tf.write(json.dumps({"sample_id": r.sample_id, "probe": r.probe,
                                 "model": model, "probe_title": r.probe_title,
                                 "probe_year": int(r.probe_year), "answer": answer,
                                 "endpoint": "chat/completions" if chat else "completions",
                                 "max_tokens": max_tokens, "temperature": 0,
                                 "prompt": sent, "completion": text}) + "\n"); tf.flush()
            n_done[0] += 1
            if n_done[0] % 50 == 0:
                print(f"  {n_done[0]:,}/{len(todo):,}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, [r for r in todo.itertuples()]))
    fh.close(); tf.close()
    print(f"  wrote {out_csv} and {traces}")


def wilson(k, n):
    if not n:
        return (np.nan, np.nan, np.nan)
    p = k / n
    z = 1.96
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return p, max(0, c - h), min(1, c + h)


def report():
    files = [f for f in os.listdir("outputs") if f.startswith("oos_probes_") and f.endswith(".csv")]
    if not files:
        sys.exit("No probe outputs yet.")
    L = ["# Out-of-sample recall probes — indicative run", "",
         "Rates with 95% Wilson intervals. `committed` = model gave a directional answer "
         "(not 'unknown'). The placebo row is the floor: committed answers there are "
         "fabrications, since those titles refer to no real paper.", ""]
    for f in sorted(files):
        d = pd.read_csv(f"outputs/{f}")
        model = f[len("oos_probes_"):-4]
        L += [f"## {model}", "", f"n = {len(d):,} calls, "
              f"logprobs readable on {d['logprob_read'].mean():.0%}", "",
              "| arm | probe | n | committed | mean commitment | mean direction |",
              "|---|---|---|---|---|---|"]
        for (arm, probe), g in d.groupby(["arm", "probe"]):
            comm = g["answer"].isin(["accepted", "rejected", "high", "low"])
            p, lo, hi = wilson(int(comm.sum()), len(g))
            L.append(f"| {arm} | {probe} | {len(g)} | {p:.1%} [{lo:.1%}, {hi:.1%}] | "
                     f"{g['commitment'].mean():.3f} | {g['direction'].mean():+.3f} |")
        # accuracy among committed answers on the real-paper probes
        L += ["", "### Accuracy when the model commits (real titles only)", "",
              "| arm | probe | n committed | correct |", "|---|---|---|---|"]
        for (arm, probe), g in d[d["probe"].isin(["lap", "fame"])].groupby(["arm", "probe"]):
            if probe == "lap":
                truth = g["decision_class"].eq("accept")
                said = g["answer"].eq("accepted")
                m = g["answer"].isin(["accepted", "rejected"])
            else:
                med = g["s2_citations"].median()
                truth = g["s2_citations"] > med
                said = g["answer"].eq("high")
                m = g["answer"].isin(["high", "low"])
            if m.sum():
                p, lo, hi = wilson(int((said[m] == truth[m]).sum()), int(m.sum()))
                L.append(f"| {arm} | {probe} | {int(m.sum())} | {p:.1%} [{lo:.1%}, {hi:.1%}] |")
        L.append("")
    open("outputs/oos_probe_report.md", "w").write("\n".join(L))
    print("\n".join(L))
    print("\nWrote outputs/oos_probe_report.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct-Turbo")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=220)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report() if a.report else run(a.model, a.limit, a.workers, a.max_tokens)
