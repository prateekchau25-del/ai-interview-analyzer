"""Voice/word-choice delivery signals and their interview-level summary."""
import numpy as np
import soundfile as sf

from backend.app.db.models import Answer
from backend.app.services.interview_service import _delivery_brief, delivery_analysis, delivery_summary


def _voice_like(path, f0=140.0, seconds=3.0, sr=16000):
    t = np.arange(0, seconds, 1 / sr)
    pitch = f0 * (1 + 0.08 * np.sin(2 * np.pi * 0.7 * t))  # gentle intonation
    phase = 2 * np.pi * np.cumsum(pitch) / sr
    wave = sum(np.sin(k * phase) / k for k in range(1, 8))  # harmonic, voice-like buzz
    env = 0.4 + 0.6 * np.abs(np.sin(2 * np.pi * 2.5 * t))  # syllables
    sf.write(path, (0.2 * wave * env).astype(np.float32), sr)


def test_delivery_analysis_voice_and_text(tmp_path):
    wav = tmp_path / "a.wav"
    _voice_like(wav)
    d = delivery_analysis("I designed and implemented the schema and it clearly improved speed.",
                          ["schema"], wav)
    assert {"lexical", "prosody", "emotion"} <= set(d)
    assert 0 <= d["prosody"]["voice_confidence"] <= 100
    assert d["emotion"]["group"] in {"calm", "positive", "nervous", "tense"}
    brief = _delivery_brief(d)
    assert {"voice_confidence", "emotion_group", "confident_language"} <= set(brief)


def test_text_only_answer_has_no_voice_signals():
    d = delivery_analysis("Maybe I think it is kind of a list.", [], None)
    assert set(d) == {"lexical"} and d["lexical"]["hedge_count"] >= 2


def test_bad_audio_never_breaks_grading(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    d = delivery_analysis("A queue is FIFO.", [], bad)
    assert "lexical" in d and "prosody" not in d


def test_delivery_summary_communication_is_mean_of_measured_parts():
    answers = [
        Answer(skipped=False, grade={"speech": {"fluency_score": 80, "words_per_minute": 130,
                                                "filler_count": 2},
                                     "prosody": {"voice_confidence": 90, "notes": ["Your voice was quiet"]},
                                     "lexical": {"confidence_language_score": 70, "hedge_count": 1},
                                     "emotion": {"group": "calm"}}),
        Answer(skipped=False, grade={"lexical": {"confidence_language_score": 50, "hedge_count": 3}}),
        Answer(skipped=True, grade={"lexical": {"confidence_language_score": 0, "hedge_count": 0}}),
    ]
    s = delivery_summary(answers)
    assert s["fluency"] == 80 and s["voice_confidence"] == 90 and s["confident_language"] == 60
    assert s["communication"] == round((80 + 90 + 60) / 3, 1)
    assert s["emotion_counts"] == {"calm": 1} and s["total_hedges"] == 4
