"""Train a Big Five (OCEAN) personality INDICATOR from text.

    python -m ml.train_big_five            # expects ml/datasets/essays/essays.csv

Dataset: Pennebaker & King "Essays" (2,467 stream-of-consciousness essays, self-reported Big Five, binary
labels cEXT/cNEU/cAGR/cCON/cOPN). Mirror: https://github.com/SenticNet/personality-detection (essays.csv).

Text -> MiniLM sentence embeddings (same ONNX model the grader uses; long texts are chunked and averaged)
-> 5 independent logistic heads. Published results on this dataset are ~55-62% per trait (chance = 50%),
so the output is shown ONLY as a soft "communication style" indicator and never affects any score.

Outputs: ml/models/big_five.onnx (384-dim embedding -> 5 probabilities), ml/models/big_five.json (metrics).
"""
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ml" / "datasets" / "essays" / "essays.csv"
CACHE = ROOT / "ml" / "datasets" / "essays" / "embeddings.npz"
OUT = ROOT / "ml" / "models"
TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
COLUMNS = ["cOPN", "cCON", "cEXT", "cAGR", "cNEU"]
SEED = 42


def load():
    csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)
    with DATA.open(encoding="latin-1") as f:
        rows = list(csv.DictReader(f))
    texts = [r["TEXT"] for r in rows]
    y = np.array([[1.0 if r[c].strip().lower() == "y" else 0.0 for c in COLUMNS] for r in rows],
                 dtype=np.float32)
    return texts, y


def embeddings(texts, batch: int = 64):
    """Batched, resumable: chunk embeddings are saved every few batches, so an interrupted run continues."""
    if CACHE.exists():
        return np.load(CACHE)["X"]
    from backend.app.analyzers.nlp import chunk_words, embed

    chunks, owner = [], []
    for i, tx in enumerate(texts):
        for c in chunk_words(tx):
            chunks.append(c)
            owner.append(i)
    partial = CACHE.with_suffix(".partial.npy")
    done = np.load(partial) if partial.exists() else np.zeros((0, 384), dtype=np.float32)
    t = time.time()
    while len(done) < len(chunks):
        parts = [embed(chunks[j:j + batch]) for j in range(len(done), min(len(chunks), len(done) + 8 * batch),
                                                            batch)]
        done = np.concatenate([done, *parts]).astype(np.float32)
        np.save(partial, done)
        print(f"  {len(done)}/{len(chunks)} chunks ({time.time() - t:.0f}s)", flush=True)
    owner = np.array(owner)
    X = np.stack([done[owner == i].mean(0) for i in range(len(texts))])
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    np.savez_compressed(CACHE, X=X.astype(np.float32))
    partial.unlink(missing_ok=True)
    return X


def main():
    import torch
    from torch import nn

    texts, y = load()
    X = embeddings(texts)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    n_test = len(X) // 5
    te, tr = idx[:n_test], idx[n_test:]
    torch.manual_seed(SEED), random.seed(SEED)

    model = nn.Sequential(nn.Dropout(0.2), nn.Linear(X.shape[1], len(TRAITS)))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(400):
        model.train()
        opt.zero_grad()
        loss_fn(model(Xt[tr]), yt[tr]).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = (torch.sigmoid(model(Xt[te])) > 0.5).float().numpy()
    acc = {t: round(float((pred[:, i] == y[te, i]).mean()), 3) for i, t in enumerate(TRAITS)}
    majority = {t: round(float(max(y[te, i].mean(), 1 - y[te, i].mean())), 3) for i, t in enumerate(TRAITS)}
    metrics = {"test_accuracy": acc, "majority_baseline": majority, "test_essays": int(len(te)),
               "mean_accuracy": round(float(np.mean(list(acc.values()))), 3)}
    print(json.dumps(metrics, indent=2))

    class WithSigmoid(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return torch.sigmoid(self.m(x))

    OUT.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(WithSigmoid(model), torch.zeros(1, X.shape[1]), str(OUT / "big_five.onnx"),
                      input_names=["embedding"], output_names=["probs"],
                      dynamic_axes={"embedding": {0: "batch"}, "probs": {0: "batch"}}, opset_version=17,
                      dynamo=False)
    (OUT / "big_five.json").write_text(json.dumps({
        "traits": TRAITS, "dataset": "Pennebaker & King Essays (2,467 essays, self-reported Big Five)",
        "embedding": "sentence-transformers/all-MiniLM-L6-v2 (ONNX), 180-word chunks averaged",
        "metrics": metrics,
        "disclaimer": "Indicator only. Text-based personality prediction is weak (~55-60%) and must not be "
                      "used for selection decisions."}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
