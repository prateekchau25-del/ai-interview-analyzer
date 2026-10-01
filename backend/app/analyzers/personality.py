"""Big Five (OCEAN) communication-style INDICATOR from the candidate's spoken answers.

Model: ml/train_big_five.py (Essays dataset). Accuracy is modest (~55-60% per trait), so this is shown as a
soft indicator in the report and is never used in any score, pass/fail or verdict.
"""
import json
import threading
from pathlib import Path

import numpy as np

from backend.app.core.config import get_settings

MIN_WORDS = 120  # too little text gives a meaningless reading
DESCRIPTIONS = {
    "openness": ("curious, imaginative, open to new ideas", "practical, prefers the familiar"),
    "conscientiousness": ("organised, careful, goal-driven", "spontaneous, flexible"),
    "extraversion": ("energetic, talkative, outgoing", "reserved, reflective"),
    "agreeableness": ("cooperative, warm, team-oriented", "direct, competitive"),
    "neuroticism": ("more sensitive to stress", "calm, emotionally steady"),
}

_lock = threading.Lock()
_model = None


def _paths() -> tuple[Path, Path]:
    d = get_settings().ml_models_dir
    return d / "big_five.onnx", d / "big_five.json"


def available() -> bool:
    return all(p.exists() for p in _paths())


def _load():
    global _model
    with _lock:
        if _model is None:
            import onnxruntime as ort

            onnx_path, meta_path = _paths()
            _model = (ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]),
                      json.loads(meta_path.read_text(encoding="utf-8")))
        return _model


def indicator(texts: list[str]) -> dict | None:
    text = " ".join(t for t in texts if t)
    if not available() or len(text.split()) < MIN_WORDS:
        return None
    from backend.app.analyzers.nlp import text_embedding

    sess, meta = _load()
    probs = sess.run(None, {"embedding": text_embedding(text)[None, :].astype(np.float32)})[0][0]
    traits = {}
    for t, p in zip(meta["traits"], probs, strict=True):
        hi, lo = DESCRIPTIONS[t]
        traits[t] = {"score": round(float(p) * 100), "leaning": hi if p >= 0.5 else lo}
    return {"traits": traits, "words_analyzed": len(text.split()),
            "model_accuracy": meta["metrics"]["mean_accuracy"], "disclaimer": meta["disclaimer"]}
