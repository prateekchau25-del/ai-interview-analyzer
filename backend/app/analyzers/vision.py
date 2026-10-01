"""Body-language analysis from webcam frames (runs live; only summary numbers leave the browser session).

MediaPipe Face Landmarker (478 points + iris + 52 blendshapes) -> face presence, eye contact, smile, blinks,
head movement. MediaPipe Hand Landmarker -> hand visibility and fidgeting. HSEmotion (EfficientNet-B0 trained
on AffectNet, ONNX) -> facial emotion. A VisionTracker accumulates one answer and returns a summary.
"""
import logging
import math
import threading
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from backend.app.core.config import get_settings

log = logging.getLogger(__name__)

MODEL_URLS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/"
                            "float16/1/face_landmarker.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
                            "float16/1/hand_landmarker.task",
    "face_emotion_enet_b0_8.onnx": "https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/"
                                   "models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx",
}
FACE_EMOTIONS = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
EMOTION_GROUP = {"happiness": "positive", "neutral": "calm", "surprise": "surprised", "fear": "nervous",
                 "sadness": "nervous", "anger": "tense", "disgust": "tense", "contempt": "tense"}

# Face-mesh landmark ids
L_OUT, L_IN, R_IN, R_OUT = 33, 133, 362, 263  # eye corners
L_TOP, L_BOT, R_TOP, R_BOT = 159, 145, 386, 374  # eyelids
L_IRIS, R_IRIS = 468, 473  # iris centres
NOSE, CHEEK_L, CHEEK_R = 1, 234, 454

_lock = threading.Lock()
_models: dict = {}


def model_dir() -> Path:
    d = get_settings().ml_models_dir / "vision"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_models() -> Path:
    """Download the three vision models on first use (~27 MB, not stored in git)."""
    d = model_dir()
    for name, url in MODEL_URLS.items():
        if not (d / name).exists():
            log.info("Downloading %s ...", name)
            urllib.request.urlretrieve(url, d / name)
    return d


def _load():
    with _lock:
        if not _models:
            import onnxruntime as ort
            from mediapipe.tasks.python import BaseOptions, vision

            d = ensure_models()
            _models["face"] = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(d / "face_landmarker.task")),
                running_mode=vision.RunningMode.IMAGE, num_faces=2, output_face_blendshapes=True))
            _models["hands"] = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(d / "hand_landmarker.task")),
                running_mode=vision.RunningMode.IMAGE, num_hands=2))
            _models["emotion"] = ort.InferenceSession(str(d / "face_emotion_enet_b0_8.onnx"),
                                                      providers=["CPUExecutionProvider"])
        return _models


def _ratio(v: float, a: float, b: float) -> float:
    return (v - a) / (b - a) if abs(b - a) > 1e-6 else 0.5


def face_emotion(rgb: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Probabilities over FACE_EMOTIONS for the face bounded by landmarks `pts` (pixel coords)."""
    import cv2

    h, w = rgb.shape[:2]
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    mx, my = 0.15 * (x1 - x0), 0.15 * (y1 - y0)
    crop = rgb[max(0, int(y0 - my)):min(h, int(y1 + my)), max(0, int(x0 - mx)):min(w, int(x1 + mx))]
    if crop.size == 0:
        return np.full(len(FACE_EMOTIONS), 1 / len(FACE_EMOTIONS))
    x = cv2.resize(crop, (224, 224)).astype(np.float32) / 255
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]  # ImageNet normalisation (HSEmotion)
    logits = _load()["emotion"].run(None, {"input": x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
    e = np.exp(logits - logits.max())
    return e / e.sum()


@dataclass
class FrameResult:
    face: bool = False
    faces: int = 0
    eye_contact: bool = False
    smile: float = 0.0
    blink: bool = False
    nose: tuple | None = None
    face_width: float = 0.0
    hands: list = field(default_factory=list)  # wrist positions (normalised)
    emotion: np.ndarray | None = None


def analyze_frame(rgb: np.ndarray, with_emotion: bool = True) -> FrameResult:
    import mediapipe as mp

    m = _load()
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    res = FrameResult()
    fr = m["face"].detect(img)
    res.faces = len(fr.face_landmarks)
    if fr.face_landmarks:
        lm = fr.face_landmarks[0]
        p = np.array([(q.x, q.y) for q in lm])
        res.face = True
        res.nose = tuple(p[NOSE])
        res.face_width = float(abs(p[CHEEK_R, 0] - p[CHEEK_L, 0]))
        # gaze: iris position inside each eye (0.5 = centre), yaw from nose vs cheeks
        gx = (_ratio(p[L_IRIS, 0], p[L_OUT, 0], p[L_IN, 0])
              + _ratio(p[R_IRIS, 0], p[R_IN, 0], p[R_OUT, 0])) / 2
        gy = (_ratio(p[L_IRIS, 1], p[L_TOP, 1], p[L_BOT, 1])
              + _ratio(p[R_IRIS, 1], p[R_TOP, 1], p[R_BOT, 1])) / 2
        yaw = _ratio(p[NOSE, 0], p[CHEEK_L, 0], p[CHEEK_R, 0])  # 0.5 = facing the camera
        res.eye_contact = 0.35 <= gx <= 0.65 and 0.2 <= gy <= 0.8 and 0.35 <= yaw <= 0.65
        bs = {c.category_name: c.score for c in fr.face_blendshapes[0]} if fr.face_blendshapes else {}
        res.smile = (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2
        res.blink = (bs.get("eyeBlinkLeft", 0) + bs.get("eyeBlinkRight", 0)) / 2 > 0.5
        if with_emotion:
            h, w = rgb.shape[:2]
            res.emotion = face_emotion(rgb, p * [w, h])
    hr = m["hands"].detect(img)
    res.hands = [(h[0].x, h[0].y) for h in hr.hand_landmarks]
    return res


class VisionTracker:
    """Accumulates frames for ONE answer. Thread-safe (webcam callbacks run on another thread)."""

    def __init__(self, sample_every: int = 3, emotion_every: int = 3):
        self.sample_every, self.emotion_every = sample_every, emotion_every
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with getattr(self, "_lock", threading.Lock()):
            self.frames_seen = 0
            self.results: list[FrameResult] = []
            self.timestamps: list[float] = []

    def add_frame(self, rgb: np.ndarray, t_sec: float) -> FrameResult | None:
        with self._lock:
            self.frames_seen += 1
            if (self.frames_seen - 1) % self.sample_every:
                return None
            n = len(self.results)
        try:
            r = analyze_frame(rgb, with_emotion=n % self.emotion_every == 0)
        except Exception as e:  # never break the video stream
            log.debug("frame analysis failed: %s", e)
            return None
        with self._lock:
            self.results.append(r)
            self.timestamps.append(t_sec)
        return r

    def summary(self) -> dict | None:
        with self._lock:
            rs, ts = list(self.results), list(self.timestamps)
        return summarize(rs, ts)


def summarize(rs: list[FrameResult], ts: list[float]) -> dict | None:
    if len(rs) < 3:
        return None
    n = len(rs)
    duration = max(ts[-1] - ts[0], 1e-3) if len(ts) > 1 else 1.0
    faced = [r for r in rs if r.face]
    face_pct = 100 * len(faced) / n
    eye_pct = 100 * sum(r.eye_contact for r in faced) / max(len(faced), 1)
    smile_pct = 100 * sum(r.smile > 0.4 for r in faced) / max(len(faced), 1)
    multi_face = 100 * sum(r.faces > 1 for r in rs) / n
    blinks = sum(1 for a, b in zip(faced, faced[1:], strict=False) if b.blink and not a.blink)
    blink_rate = blinks / duration * 60

    noses = np.array([r.nose for r in faced]) if faced else np.zeros((0, 2))
    width = np.mean([r.face_width for r in faced]) if faced else 1.0
    head_move = float(np.linalg.norm(noses.std(0)) / max(width, 1e-3)) if len(noses) > 2 else 0.0

    hand_pct = 100 * sum(bool(r.hands) for r in rs) / n
    hand_speeds = []
    for (a, ta), (b, tb) in zip(zip(rs, ts, strict=True), zip(rs[1:], ts[1:], strict=True), strict=False):
        if a.hands and b.hands and tb > ta:
            hand_speeds.append(math.dist(a.hands[0], b.hands[0]) / (tb - ta))
    hand_speed = float(np.median(hand_speeds)) if hand_speeds else 0.0
    hand_level = "not visible" if hand_pct < 10 else "fidgety" if hand_speed > 0.6 else \
        "natural" if hand_speed > 0.05 else "still"

    emos = [r.emotion for r in rs if r.emotion is not None]
    emotion_dist, groups = {}, {}
    if emos:
        mean = np.mean(emos, axis=0)
        emotion_dist = {e: round(float(p), 3) for e, p in zip(FACE_EMOTIONS, mean, strict=True)}
        for e, p in emotion_dist.items():
            groups[EMOTION_GROUP[e]] = round(groups.get(EMOTION_GROUP[e], 0) + p, 3)
    dominant = max(groups, key=groups.get) if groups else None

    score, notes = 100.0, []
    if face_pct < 80:
        score -= min(30, (80 - face_pct) * 0.6)
        notes.append("Your face was often out of frame - sit centred and facing the camera.")
    if eye_pct < 60:
        score -= min(30, (60 - eye_pct) * 0.6)
        notes.append("Look at the camera more - it reads as eye contact to the interviewer.")
    if smile_pct < 5 and faced:
        score -= 5
        notes.append("A natural smile now and then makes you look more confident and approachable.")
    if head_move > 0.25:
        score -= 10
        notes.append("Your head moved a lot - keep a steady, upright posture.")
    if hand_level == "fidgety":
        score -= 10
        notes.append("Your hands moved restlessly - use calm, deliberate gestures.")
    if blink_rate > 40:
        score -= 5
        notes.append("You blinked very often, which can signal nervousness - pause and breathe.")
    if groups.get("nervous", 0) > 0.4:
        score -= 10
        notes.append("You looked tense or anxious - relax your face and take a breath before answering.")
    if multi_face > 20:
        notes.append("Another person appeared in the frame during this answer.")

    return {
        "frames_analyzed": n, "seconds": round(duration, 1),
        "face_present_pct": round(face_pct, 1), "eye_contact_pct": round(eye_pct, 1),
        "smile_pct": round(smile_pct, 1), "blink_per_min": round(blink_rate, 1),
        "head_movement": round(head_move, 3), "hands_visible_pct": round(hand_pct, 1),
        "hand_movement": hand_level, "multiple_faces_pct": round(multi_face, 1),
        "face_emotions": emotion_dist, "emotion_groups": groups, "dominant_emotion": dominant,
        "body_language_score": round(max(0.0, score), 1), "notes": notes,
    }
