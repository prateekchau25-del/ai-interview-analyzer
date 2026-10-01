"""Train the speech-emotion model on RAVDESS and calibrate prosody thresholds.

    python -m ml.train_voice_emotion            # expects ml/datasets/ravdess/speech/Actor_*/*.wav

Dataset: RAVDESS speech (Livingstone & Russo, 2018), 24 actors x 60 clips, 8 emotions. CC BY-NC-SA 4.0.
Download: https://zenodo.org/records/1188976  (Audio_Speech_Actors_01-24.zip)

Evaluation is speaker-independent: actors 1-18 train, 19-20 validation, 21-24 test (never seen).
Outputs (small, committed to git):
    ml/models/voice_emotion.onnx      raw 79-dim features -> class probabilities (normalisation inside)
    ml/models/voice_emotion.json      labels, interview groups, metrics, confusion matrix
    ml/models/prosody_calibration.json  natural-speech ranges of jitter/shimmer/HNR/loudness (calm+neutral)
"""
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ml" / "datasets" / "ravdess" / "speech"
OUT = ROOT / "ml" / "models"
CACHE = ROOT / "ml" / "datasets" / "ravdess" / "features.npz"

EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
# What each emotion means for an interview answer.
GROUPS = {"neutral": "calm", "calm": "calm", "happy": "positive", "surprised": "positive",
          "sad": "nervous", "fearful": "nervous", "angry": "tense", "disgust": "tense"}
GROUP_NAMES = ["calm", "positive", "nervous", "tense"]
TRAIN_ACTORS, VAL_ACTORS, TEST_ACTORS = range(1, 19), range(19, 21), range(21, 25)
SEED = 42


def parse(path: Path) -> tuple[int, int]:
    """03-01-EE-II-SS-RR-AA.wav -> (emotion index 0-7, actor 1-24)."""
    parts = path.stem.split("-")
    return int(parts[2]) - 1, int(parts[6])


def _features(args):
    path, augment = args
    from backend.app.analyzers.voice_features import feature_vector, load_wav

    y = load_wav(path)
    out = [feature_vector(y)]
    if augment:  # noisy copy (~20 dB SNR) so the model tolerates laptop mics
        rng = np.random.default_rng(abs(hash(str(path))) % 2**32)
        noise = rng.normal(0, 1, len(y)).astype(np.float32)
        noise *= np.sqrt((y ** 2).mean() / (10 ** 2)) / (noise.std() + 1e-9)
        out.append(feature_vector(y + noise))
    return out


def build_features():
    if CACHE.exists():
        d = np.load(CACHE)
        return d["X"], d["y"], d["actor"], d["aug"]
    files = sorted(DATA.glob("Actor_*/*.wav"))
    if not files:
        raise SystemExit(f"No RAVDESS wavs in {DATA}. Download and unzip the dataset first.")
    meta = [parse(f) for f in files]
    jobs = [(str(f), m[1] in TRAIN_ACTORS) for f, m in zip(files, meta, strict=True)]
    t = time.time()
    with ProcessPoolExecutor() as ex:
        feats = list(ex.map(_features, jobs, chunksize=8))
    print(f"features for {len(files)} clips in {time.time() - t:.0f}s")
    X, y, actor, aug = [], [], [], []
    for (emo, act), fs in zip(meta, feats, strict=True):
        for i, f in enumerate(fs):
            X.append(f), y.append(emo), actor.append(act), aug.append(i > 0)
    X, y, actor, aug = map(np.array, (X, y, actor, aug))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, X=X, y=y, actor=actor, aug=aug)
    return X, y, actor, aug


def train(X, y, actor, aug):
    import torch
    from torch import nn

    torch.manual_seed(SEED), random.seed(SEED), np.random.seed(SEED)
    tr = np.isin(actor, TRAIN_ACTORS)
    va = np.isin(actor, VAL_ACTORS) & ~aug
    te = np.isin(actor, TEST_ACTORS) & ~aug
    mean, std = X[tr].mean(0), X[tr].std(0) + 1e-6

    class Net(nn.Module):
        def __init__(self, d_in, n_out):
            super().__init__()
            self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
            self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
            self.body = nn.Sequential(
                nn.Linear(d_in, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.4),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.4),
                nn.Linear(128, n_out))

        def forward(self, x):
            return self.body((x - self.mean) / self.std)

    model = Net(X.shape[1], len(EMOTIONS))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y)

    def acc(mask):
        model.eval()
        with torch.no_grad():
            return (model(Xt[mask]).argmax(1) == yt[mask]).float().mean().item()

    best, best_state, patience = 0.0, None, 0
    idx = np.where(tr)[0]
    for epoch in range(300):  # noqa: B007 - reported after the loop
        model.train()
        np.random.shuffle(idx)
        for b in range(0, len(idx), 64):
            batch = idx[b:b + 64]
            if len(batch) < 2:
                continue
            opt.zero_grad()
            loss_fn(model(Xt[batch]), yt[batch]).backward()
            opt.step()
        sched.step()
        v = acc(va)
        if v > best:
            best, best_state, patience = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 40:
                break
    model.load_state_dict(best_state)
    model.eval()
    print(f"stopped at epoch {epoch}, best val acc {best:.3f}")
    return model, te, va


def evaluate(model, X, y, te):
    import torch

    with torch.no_grad():
        pred = model(torch.tensor(X[te], dtype=torch.float32)).argmax(1).numpy()
    truth = y[te]
    cm = np.zeros((len(EMOTIONS), len(EMOTIONS)), dtype=int)
    for t, p in zip(truth, pred, strict=True):
        cm[t, p] += 1
    g = {e: GROUP_NAMES.index(GROUPS[e]) for e in EMOTIONS}
    gt = np.array([g[EMOTIONS[i]] for i in truth])
    gp = np.array([g[EMOTIONS[i]] for i in pred])
    per_class = {EMOTIONS[i]: round(cm[i, i] / max(cm[i].sum(), 1), 3) for i in range(len(EMOTIONS))}
    return {
        "test_accuracy_8class": round(float((pred == truth).mean()), 3),
        "test_accuracy_4group": round(float((gp == gt).mean()), 3),
        "chance_8class": round(1 / len(EMOTIONS), 3),
        "per_class_recall": per_class,
        "confusion_matrix": cm.tolist(),
        "test_clips": int(te.sum()),
    }


def export(model, n_features: int, metrics: dict):
    import torch

    class WithSoftmax(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return torch.softmax(self.m(x), dim=1)

    OUT.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(WithSoftmax(model), torch.zeros(1, n_features), str(OUT / "voice_emotion.onnx"),
                      input_names=["features"], output_names=["probs"],
                      dynamic_axes={"features": {0: "batch"}, "probs": {0: "batch"}}, opset_version=17,
                      dynamo=False)
    meta = {"labels": EMOTIONS, "groups": GROUPS, "group_names": GROUP_NAMES, "feature_dim": n_features,
            "dataset": "RAVDESS speech (24 actors, 1440 clips), speaker-independent split",
            "split": {"train": "actors 1-18 (+ noise augmentation)", "val": "actors 19-20",
                      "test": "actors 21-24"},
            "metrics": metrics}
    (OUT / "voice_emotion.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _prosody(path):
    from backend.app.analyzers.prosody import analyze

    r = analyze(path)
    return [r.jitter_pct, r.shimmer_pct, r.hnr_db, r.intensity_sd_db, r.pitch_sd_semitones]


def calibrate_prosody():
    """Natural ranges of voice-stability measures in calm/neutral connected speech."""
    files = [str(f) for f in sorted(DATA.glob("Actor_*/*.wav")) if parse(f)[0] in (0, 1)]
    with ProcessPoolExecutor() as ex:
        vals = np.array(list(ex.map(_prosody, files, chunksize=8)))
    names = ["jitter_pct", "shimmer_pct", "hnr_db", "intensity_sd_db", "pitch_sd_semitones"]
    cal = {n: {p: round(float(np.percentile(vals[:, i], p)), 3) for p in (5, 10, 50, 90, 95)}
           for i, n in enumerate(names)}
    cal["_source"] = f"RAVDESS calm+neutral speech, {len(files)} clips, 24 actors"
    (OUT / "prosody_calibration.json").write_text(json.dumps(cal, indent=2), encoding="utf-8")
    return cal


if __name__ == "__main__":
    X, y, actor, aug = build_features()
    print("dataset:", X.shape, "train clips (incl. augmented):", int(np.isin(actor, TRAIN_ACTORS).sum()))
    model, te, _ = train(X, y, actor, aug)
    metrics = evaluate(model, X, y, te)
    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))
    export(model, X.shape[1], metrics)
    print("prosody calibration:", json.dumps(calibrate_prosody(), indent=1))
