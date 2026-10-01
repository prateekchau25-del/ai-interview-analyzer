"""Speech-emotion inference with the RAVDESS-trained ONNX model (see ml/train_voice_emotion.py).

Returns the 8 RAVDESS emotions plus an interview-oriented group: calm | positive | nervous | tense.
If the model file is missing, `available()` is False and callers skip this signal.
"""
import json
import threading
from pathlib import Path

import numpy as np

from backend.app.core.config import get_settings

_lock = threading.Lock()
_model = None


def _paths() -> tuple[Path, Path]:
    d = get_settings().ml_models_dir
    return d / "voice_emotion.onnx", d / "voice_emotion.json"


def available() -> bool:
    return all(p.exists() for p in _paths())


def _load():
    global _model
    with _lock:
        if _model is None:
            import onnxruntime as ort

            onnx_path, meta_path = _paths()
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            _model = (sess, json.loads(meta_path.read_text(encoding="utf-8")))
        return _model


def predict(wav_path: str | Path) -> dict | None:
    if not available():
        return None
    from backend.app.analyzers.voice_features import feature_vector, load_wav

    sess, meta = _load()
    y = load_wav(wav_path)
    if len(y) < 16000 * 0.5:  # under half a second of audio
        return None
    probs = sess.run(None, {"features": feature_vector(y)[None, :].astype(np.float32)})[0][0]
    labels, groups = meta["labels"], meta["groups"]
    group_probs: dict[str, float] = {}
    for lab, p in zip(labels, probs, strict=True):
        group_probs[groups[lab]] = group_probs.get(groups[lab], 0.0) + float(p)
    top = int(np.argmax(probs))
    group = max(group_probs, key=group_probs.get)
    return {
        "emotion": labels[top], "emotion_confidence": round(float(probs[top]), 3),
        "group": group, "group_confidence": round(group_probs[group], 3),
        "group_probs": {k: round(v, 3) for k, v in group_probs.items()},
    }
