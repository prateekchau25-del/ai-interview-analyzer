"""Voice prosody with Praat (parselmouth): pitch, loudness, voice stability -> voice-confidence score.

Pitch variation is measured in semitones, so it is comparable across male and female voices.
Note: the classic clinical limits (jitter < 1.04 %, shimmer < 3.81 %) are for SUSTAINED VOWELS. Connected
speech naturally has much higher values, so thresholds come from ml/models/prosody_calibration.json - the 95th
(or 5th) percentile of calm/neutral RAVDESS speech - with connected-speech defaults if that file is missing.
Monotone speech has a pitch SD below ~1.5 semitones; lively speech is ~2.5-5.
"""
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from backend.app.core.config import get_settings

DEFAULT_LIMITS = {"jitter_pct": 3.5, "shimmer_pct": 12.0, "hnr_db": 8.0, "intensity_sd_db": 12.0}


@lru_cache
def limits() -> dict[str, float]:
    """Upper limits (lower for HNR) of natural calm speech."""
    path = get_settings().ml_models_dir / "prosody_calibration.json"
    if not path.exists():
        return dict(DEFAULT_LIMITS)
    cal = json.loads(path.read_text(encoding="utf-8"))
    return {"jitter_pct": cal["jitter_pct"]["95"], "shimmer_pct": cal["shimmer_pct"]["95"],
            "hnr_db": cal["hnr_db"]["5"],
            # RAVDESS clips are single sentences; long answers naturally vary more in loudness.
            "intensity_sd_db": max(cal["intensity_sd_db"]["95"], DEFAULT_LIMITS["intensity_sd_db"])}


@dataclass
class ProsodyResult:
    duration_sec: float
    voiced_ratio: float  # share of frames with voice (speech vs silence)
    pitch_mean_hz: float
    pitch_sd_semitones: float  # intonation / expressiveness
    pitch_range_semitones: float
    intensity_mean_db: float
    intensity_sd_db: float
    jitter_pct: float
    shimmer_pct: float
    hnr_db: float
    expressiveness: str  # monotone | balanced | very variable
    voice_confidence: float  # 0-100
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _semitones(f0: np.ndarray, ref: float) -> np.ndarray:
    return 12 * np.log2(f0 / ref)


def analyze(wav_path: str | Path) -> ProsodyResult:
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(str(wav_path))
    duration = snd.get_total_duration()

    pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
    f0 = pitch.selected_array["frequency"]
    voiced = f0[f0 > 0]
    voiced_ratio = len(voiced) / len(f0) if len(f0) else 0.0
    if len(voiced) >= 10:
        med = float(np.median(voiced))
        st = _semitones(voiced, med)
        lo, hi = np.percentile(st, [5, 95])
        pitch_sd, pitch_range, pitch_mean = float(np.std(st)), float(hi - lo), float(np.mean(voiced))
    else:
        pitch_sd = pitch_range = pitch_mean = 0.0

    intensity = snd.to_intensity(minimum_pitch=75)
    db = intensity.values.flatten()
    db = db[db > 40]  # ignore near-silence frames
    int_mean, int_sd = (float(np.mean(db)), float(np.std(db))) if len(db) else (0.0, 0.0)

    jitter = shimmer = 0.0
    try:
        pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        jitter = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
        shimmer = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
    except Exception:  # too little voiced speech
        pass
    hnr = call(snd.to_harmonicity_cc(), "Get mean", 0, 0)
    hnr = 0.0 if hnr is None or math.isnan(hnr) else float(hnr)
    jitter = 0.0 if math.isnan(jitter) else jitter
    shimmer = 0.0 if math.isnan(shimmer) else shimmer

    # ---- score: start at 100, subtract for each problem, explain every deduction
    lim = limits()
    score, notes = 100.0, []
    if voiced_ratio < 0.2:
        score -= 30
        notes.append("Very little voiced speech was detected - speak up and keep talking.")
    if pitch_sd and pitch_sd < 1.5:
        score -= 20
        notes.append("Your voice sounded monotone - vary your tone to stress key points.")
    elif pitch_sd > 6:
        score -= 10
        notes.append("Your pitch jumped a lot - a steadier tone sounds more composed.")
    if int_mean and int_mean < 55:
        score -= 15
        notes.append("Your voice was quiet - speak a little louder and closer to the mic.")
    if int_sd > lim["intensity_sd_db"]:
        score -= 10
        notes.append("Your loudness was uneven - keep a steady volume.")
    if jitter > lim["jitter_pct"]:
        score -= min(20, (jitter - lim["jitter_pct"]) * 10)
        notes.append("Your voice was shaky (high jitter) - slow down and breathe before answering.")
    if shimmer > lim["shimmer_pct"]:
        score -= min(15, (shimmer - lim["shimmer_pct"]) * 2)
        notes.append("Your volume wavered (high shimmer), a common sign of nervousness.")
    if hnr and hnr < lim["hnr_db"]:
        score -= 10
        notes.append("Your voice was breathy or noisy - check the mic and speak clearly.")

    expr = "monotone" if 0 < pitch_sd < 1.5 else "very variable" if pitch_sd > 6 else "balanced"
    return ProsodyResult(
        duration_sec=round(duration, 2), voiced_ratio=round(voiced_ratio, 3),
        pitch_mean_hz=round(pitch_mean, 1), pitch_sd_semitones=round(pitch_sd, 2),
        pitch_range_semitones=round(pitch_range, 2), intensity_mean_db=round(int_mean, 1),
        intensity_sd_db=round(int_sd, 1), jitter_pct=round(jitter, 2), shimmer_pct=round(shimmer, 2),
        hnr_db=round(hnr, 1), expressiveness=expr, voice_confidence=round(max(0.0, score), 1), notes=notes,
    )
