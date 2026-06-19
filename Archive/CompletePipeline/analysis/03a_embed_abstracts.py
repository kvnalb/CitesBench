"""Embed 2018-2023 ICLR abstracts with SPECTER2 and save to CSV."""

import sqlite3
import csv
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
DB_PATH = os.path.join(ROOT, "data/LLM-Reviewer-03042026/data/gen_review.db")
OUT_DIR = os.path.join(ROOT, "OutputNew/Empirics/embeddings")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(OUT_DIR, "abstracts_specter2_2018_2023.csv")

# ── load abstracts ────────────────────────────────────────────────────────
con = sqlite3.connect(DB_PATH)
cur = con.execute(
    "SELECT id, title, abstract, when_submitted "
    "FROM SUBMISSION WHERE when_submitted BETWEEN 2018 AND 2023"
)
rows = cur.fetchall()
con.close()

paper_ids = [r[0] for r in rows]
titles = [r[1] or "" for r in rows]
abstracts = [r[2] or "" for r in rows]
years = [r[3] for r in rows]
# SPECTER2 expects title + sep + abstract
texts = [f"{t} [SEP] {a}" for t, a in zip(titles, abstracts)]

print(f"Loaded {len(texts)} abstracts (2018-2023)")

# ── load model ────────────────────────────────────────────────────────────
MODEL_NAME = "allenai/specter2_base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)
print(f"Using device: {device}")

# ── embed in batches ──────────────────────────────────────────────────────
BATCH_SIZE = 64
embeddings = []

for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i : i + BATCH_SIZE]
    inputs = tokenizer(
        batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        out = model(**inputs)
    # CLS token embedding
    emb = out.last_hidden_state[:, 0, :].cpu().numpy()
    embeddings.append(emb)
    if (i // BATCH_SIZE) % 10 == 0:
        print(f"  batch {i // BATCH_SIZE + 1}/{(len(texts) - 1) // BATCH_SIZE + 1}")

embeddings = np.vstack(embeddings)
print(f"Embedding shape: {embeddings.shape}")

# ── save ──────────────────────────────────────────────────────────────────
dim = embeddings.shape[1]
header = ["paper_id", "year"] + [f"emb_{j}" for j in range(dim)]

with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    for idx in range(len(paper_ids)):
        w.writerow([paper_ids[idx], years[idx]] + embeddings[idx].tolist())

print(f"Saved to {OUT_CSV}")
