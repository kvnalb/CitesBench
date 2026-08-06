"""
Tag each 2018-2020 ICLR paper with a primary research field using Together AI.
Writes a `field` column to outputs/paper_fields.csv.

Fields: generative_models, reinforcement_learning, optimization_theory,
        nlp, computer_vision, robustness_adversarial, graph_structured,
        meta_few_shot, representation_learning, efficient_ml
"""
import os
import json
import time
import signal
import sqlite3
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

class _Timeout(Exception): pass

def _timeout_handler(signum, frame): raise _Timeout()

FIELDS = ["nlp", "computer_vision", "generative_models", "reinforcement_learning", "theory_methods"]

SYSTEM = f"""You classify machine learning papers into one of 5 research fields based on the paper's primary contribution.
Classify the paper into exactly one field from this list:
{', '.join(FIELDS)}

Respond in JSON with two keys: "field" (exact field name from the list) and "rationale" (one sentence max).
Example: {{"field": "nlp", "rationale": "Proposes a new attention mechanism for sequence-to-sequence tasks."}}

Definitions and examples:
- nlp: understanding or generating natural language. Language models, transformers, text classification, translation, parsing, QA, dialogue, speech. e.g. "BERT", "neural machine translation", "language grounding"
- computer_vision: understanding images or video. Classification, detection, segmentation, pose estimation, optical flow, visual recognition, video. e.g. "ResNet", "Mask R-CNN", "video prediction". Note: vision transformers go here, not nlp.
- generative_models: generating data or modeling distributions as the primary goal. GANs, VAEs, diffusion, normalizing flows, density estimation. e.g. "Progressive GAN", "beta-VAE", "flow matching". Note: a GAN for image generation → generative_models, not computer_vision.
- reinforcement_learning: learning from reward signals. Policy gradient, Q-learning, actor-critic, imitation learning, exploration, MARL, model-based RL. e.g. "PPO", "curiosity-driven exploration", "reward shaping"
- theory_methods: everything else — the primary contribution is a method, algorithm, or theory rather than an application domain. Optimization, generalization theory, robustness, adversarial examples, graph neural networks, meta-learning, few-shot learning, model compression, architecture search, representation learning, transfer learning, fairness, privacy. e.g. "Adam", "adversarial training", "MAML", "knowledge distillation", "GNN", "lottery ticket hypothesis"

When in doubt: if the paper proposes a general method that happens to be evaluated on vision or NLP benchmarks, prefer theory_methods."""

os.makedirs("outputs", exist_ok=True)
OUT = "outputs/paper_fields.csv"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()

con = sqlite3.connect("data/gen_review.db")
df = pd.read_sql(
    "SELECT id, title, keywords FROM SUBMISSION WHERE when_submitted IN (2018,2019,2020)",
    con,
)
con.close()

if args.smoke:
    df = df.head(10)

if args.smoke:
    print(f"Smoke test — tagging {len(df)} papers...")
elif os.path.exists(OUT):
    existing = pd.read_csv(OUT)
    done_ids = set(existing["id"])
    df = df[~df["id"].isin(done_ids)]
    print(f"Resuming — {len(done_ids)} done, {len(df)} remaining")
else:
    print(f"Tagging {len(df)} papers...")

client = OpenAI(
    api_key=os.environ["TOGETHER_API_KEY"],
    base_url="https://api.together.xyz/v1",
)

def _tag(title: str, keywords: str) -> tuple[str, str]:
    for attempt in range(3):
        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(25)  # hard 25s OS-level kill
            resp = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                max_tokens=512,
                timeout=20,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Title: {title}\nKeywords: {keywords}"},
                ],
            )
            signal.alarm(0)
        except _Timeout:
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            raise ValueError(f"timed out after 3 attempts for '{title[:40]}'")
        raw = resp.choices[0].message.content.strip().lower()
        try:
            parsed = json.loads(raw)
            field = parsed["field"]
            if field not in FIELDS:
                raise ValueError(f"field {field!r} not in taxonomy")
            return field, parsed["rationale"]
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
            else:
                raise ValueError(f"failed after 3 attempts: {e!r} raw={raw!r}")

results = []
for i, (_, row) in enumerate(df.iterrows()):
    field, rationale = _tag(row["title"], row["keywords"])
    results.append({"id": row["id"], "field": field, "rationale": rationale})


    if args.smoke:
        print(results[-1])
    else:
        write_header = not os.path.exists(OUT)
        pd.DataFrame([results[-1]]).to_csv(OUT, mode="a", header=write_header, index=False)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(df)}")

if not args.smoke:
    final = pd.read_csv(OUT)
    print(f"Done. {len(final)} papers tagged.\nField distribution:")
    print(final["field"].value_counts().to_string())
