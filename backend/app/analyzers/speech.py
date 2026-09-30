"""Speech pipeline: transcript + fluency metrics from word-level timestamps.

faster-whisper (CTranslate2) runs Whisper ~4x faster than openai-whisper on CPU with int8,
and gives per-word timestamps, which is what pause / pace analysis needs.
"""
import logging
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

from backend.app.core.config import get_settings

log = logging.getLogger(__name__)

# Single-word fillers are matched as whole tokens; multi-word ones as phrases.
FILLER_WORDS = {
    "um", "umm", "uh", "uhh", "uhm", "hmm", "er", "ah", "like", "basically", "actually",
    "literally", "so", "matlab", "yaani", "toh", "haan", "achha", "okay",
}
FILLER_PHRASES = ["you know", "i mean", "kind of", "sort of", "i guess"]
# Tokens that are fillers only when they stand alone at a clause boundary, not mid-sentence.
CONTEXT_FILLERS = {"so", "like", "okay", "toh", "haan", "achha", "actually"}

LONG_PAUSE_SEC = 1.5
IDEAL_WPM = (120, 160)

_model = None
_model_lock = threading.Lock()


def get_model():
    """Load Whisper once per process (the first call downloads the weights)."""
    global _model
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel

            s = get_settings()
            log.info("Loading faster-whisper '%s' (%s)...", s.whisper_model, s.whisper_compute_type)
            _model = WhisperModel(s.whisper_model, device="cpu", compute_type=s.whisper_compute_type)
        return _model


@dataclass
class Word:
    text: str
    start: float
    end: float
    prob: float


@dataclass
class SpeechResult:
    transcript: str
    language: str
    duration_sec: float
    speaking_time_sec: float
    word_count: int
    words_per_minute: float
    filler_count: int
    filler_breakdown: dict[str, int]
    filler_ratio_pct: float
    long_pause_count: int
    longest_pause_sec: float
    pauses: list[dict] = field(default_factory=list)
    avg_word_confidence: float = 0.0
    fluency_score: float = 0.0
    pace_label: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _norm(token: str) -> str:
    return re.sub(r"[^\w']", "", token.lower())


def count_fillers(words: list[Word]) -> dict[str, int]:
    counts: dict[str, int] = {}
    tokens = [_norm(w.text) for w in words]
    raw = [w.text.strip() for w in words]
    for i, tok in enumerate(tokens):
        if tok not in FILLER_WORDS:
            continue
        if tok in CONTEXT_FILLERS:
            # Count "so"/"like"/... only at a clause start or when followed by a comma.
            prev_punct = i == 0 or raw[i - 1][-1:] in ",.?!"
            own_comma = raw[i].endswith(",")
            if not (prev_punct or own_comma):
                continue
        counts[tok] = counts.get(tok, 0) + 1
    text = " ".join(tokens)
    for phrase in FILLER_PHRASES:
        n = len(re.findall(rf"\b{re.escape(phrase)}\b", text))
        if n:
            counts[phrase] = n
    return counts


def fluency_score(wpm: float, filler_ratio_pct: float, long_pauses: int, minutes: float) -> float:
    """0-100. Penalises pace outside 120-160 WPM, fillers and long pauses (per minute)."""
    lo, hi = IDEAL_WPM
    if wpm == 0:
        return 0.0
    pace_pen = 0.0 if lo <= wpm <= hi else min(35.0, (lo - wpm if wpm < lo else wpm - hi) * 0.6)
    filler_pen = min(35.0, filler_ratio_pct * 3.5)
    pauses_per_min = long_pauses / max(minutes, 0.25)
    pause_pen = min(30.0, pauses_per_min * 6.0)
    return round(max(0.0, 100.0 - pace_pen - filler_pen - pause_pen), 1)


def analyze_words(words: list[Word], duration_sec: float, language: str = "en") -> SpeechResult:
    transcript = " ".join(w.text.strip() for w in words).strip()
    n = len(words)
    if n == 0:
        return SpeechResult(transcript="", language=language, duration_sec=duration_sec,
                            speaking_time_sec=0, word_count=0, words_per_minute=0, filler_count=0,
                            filler_breakdown={}, filler_ratio_pct=0, long_pause_count=0,
                            longest_pause_sec=0, pace_label="no speech")

    speaking_time = max(words[-1].end - words[0].start, 0.1)
    wpm = n / (speaking_time / 60.0)

    pauses = []
    for prev, cur in zip(words, words[1:], strict=False):
        gap = cur.start - prev.end
        if gap >= LONG_PAUSE_SEC:
            pauses.append({"at": round(prev.end, 2), "duration": round(gap, 2)})

    fillers = count_fillers(words)
    filler_total = sum(fillers.values())
    filler_ratio = filler_total / n * 100
    lo, hi = IDEAL_WPM
    pace = "too slow" if wpm < lo - 20 else "slightly slow" if wpm < lo else \
        "too fast" if wpm > hi + 20 else "slightly fast" if wpm > hi else "good"

    return SpeechResult(
        transcript=transcript,
        language=language,
        duration_sec=round(duration_sec, 2),
        speaking_time_sec=round(speaking_time, 2),
        word_count=n,
        words_per_minute=round(wpm, 1),
        filler_count=filler_total,
        filler_breakdown=fillers,
        filler_ratio_pct=round(filler_ratio, 2),
        long_pause_count=len(pauses),
        longest_pause_sec=max((p["duration"] for p in pauses), default=0.0),
        pauses=pauses,
        avg_word_confidence=round(sum(w.prob for w in words) / n, 3),
        fluency_score=fluency_score(wpm, filler_ratio, len(pauses), speaking_time / 60),
        pace_label=pace,
    )


def transcribe(audio_path: str | Path, initial_prompt: str | None = None) -> SpeechResult:
    """Transcribe a 16 kHz mono WAV and compute fluency metrics.

    `initial_prompt` (e.g. the question text) biases Whisper toward the right technical terms.
    Fillers are kept on purpose (no prompt text that "cleans" speech).
    """
    s = get_settings()
    segments, info = get_model().transcribe(
        str(audio_path),
        language=s.whisper_language,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 700},
        beam_size=5,
        initial_prompt=(initial_prompt + " Umm, uh, so, like...") if initial_prompt else "Umm, uh, so...",
        condition_on_previous_text=False,
    )
    words = [
        Word(text=w.word, start=w.start, end=w.end, prob=w.probability)
        for seg in segments for w in (seg.words or [])
    ]
    return analyze_words(words, duration_sec=info.duration, language=info.language)
