"""
Together chat-completions client shared by the review-scoring scripts.

Kept deliberately small: one call function that never raises on a bad reply, and one
parser that records HOW it read the JSON. A refusal, a truncation, or a malformed
object is evidence about the model and must reach the trace file, not vanish inside
an exception handler.

Used by src/probes/qualitative_check.py and src/probes/run_review_eval.py.
"""
import os
import re
import sys
import json
import time
import hashlib

URL = "https://api.together.xyz/v1/chat/completions"

MODELS = {
    # key: (model id, max_tokens) — Gemma-4 is a thinking model and needs room for the
    # reasoning it emits before the JSON; Llama answers directly
    "gemma": ("google/gemma-4-31B-it", 3000),
    "llama": ("meta-llama/Llama-3.3-70B-Instruct-Turbo", 1500),
}

RUBRIC_FIELDS = ["rating", "confidence", "correctness",
                 "technical_novelty_and_significance",
                 "empirical_novelty_and_significance"]


def sha(s):
    return hashlib.sha1(str(s).encode()).hexdigest()[:12]


def call(model, max_tokens, system, user, retries=4, timeout=180):
    """Returns (text, http_status, raw_json). Never raises on a bad reply."""
    import requests
    key = os.environ.get("TOGETHER_API_KEY", "")
    if not key:
        sys.exit("ERROR: TOGETHER_API_KEY not set in .env")
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]}
    status = 0
    for a in range(retries):
        try:
            r = requests.post(URL, json=body, timeout=timeout,
                              headers={"Authorization": f"Bearer {key}"})
        except Exception as e:
            print(f"    retry {a + 1}/{retries}: {type(e).__name__} {e}", flush=True)
            time.sleep(3 * (a + 1))
            continue
        status = r.status_code
        if status == 200:
            j = r.json()
            ch = (j.get("choices") or [{}])[0]
            return (ch.get("message") or {}).get("content") or "", 200, j
        print(f"    retry {a + 1}/{retries}: HTTP {status} {r.text[:140]}", flush=True)
        time.sleep(3 * (a + 1))
    return "", status, {}


def parse_rubric(text):
    """Extract the rubric JSON. Returns (parsed, n_blocks, mode).

    Three passes, because Llama-3.3 sometimes emits the object without its closing
    brace and still reports finish_reason=stop — a well-formed answer that a strict
    parser throws away:
      strict    a complete {...} block parses (last one wins; the rubric's own
                few-shot examples are JSON, so never take the first)
      repaired  same, after appending the missing brace
      fields    neither parsed; scrape the keys individually
    """
    text = text or ""
    blocks = re.findall(r"\{[^{}]*\}", text, re.DOTALL)
    for b in reversed(blocks):
        try:
            j = json.loads(b)
        except json.JSONDecodeError:
            continue
        if "rating" in j:
            return j, len(blocks), "strict"

    i = text.rfind("{")
    if i >= 0:
        for cand in (text[i:] + "}", text[i:].rstrip().rstrip(",") + "}"):
            try:
                j = json.loads(cand)
                if "rating" in j:
                    return j, len(blocks), "repaired"
            except json.JSONDecodeError:
                pass

    out = {}
    for f in RUBRIC_FIELDS:
        m = re.search(rf'"{f}"\s*:\s*(-?\d+(?:\.\d+)?)', text)
        if m:
            out[f] = float(m.group(1))
    m = re.search(r'"rationale"\s*:\s*"(.*?)"\s*[,}\n]', text, re.DOTALL)
    if m:
        out["rationale"] = m.group(1)
    return out, len(blocks), ("fields" if out else "none")


def demo():
    good = '{"rating": 3.3, "confidence": 2.8, "correctness": 3.8, ' \
           '"technical_novelty_and_significance": 2.9, ' \
           '"empirical_novelty_and_significance": 2.9, "rationale": "ok"}'
    assert parse_rubric(good)[2] == "strict"
    assert parse_rubric(good)[0]["rating"] == 3.3
    assert parse_rubric(good.rstrip()[:-1])[2] == "repaired"      # missing brace
    assert parse_rubric('"rating": 2.0, "confidence": 3.0')[2] == "fields"
    assert parse_rubric("I cannot review this.")[2] == "none"
    # a reply that echoes a few-shot example must not win over the real answer
    two = '{"rating": 0.3, "rationale": "example"}\nFinal:\n' + good
    assert parse_rubric(two)[0]["rating"] == 3.3
    print("ok — parse_rubric handles strict / repaired / fields / none / echoed examples")


if __name__ == "__main__":
    demo()
