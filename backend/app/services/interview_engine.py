"""Adaptive interview engine (pure logic, no DB / no I/O, so it is fully unit-testable).

Flow of one interview:
    intro (1 soft question) -> 15 technical (adaptive level) -> resume questions -> soft-skill questions
    -> review of parked (wrong/skipped) questions -> finished

Level rules:
    answer >= 85%                  -> level up (max Professional)
    2 not-correct answers in a row -> level down (min Basic)
"""
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Protocol

LEVELS = ["Basic", "Beginner", "Medium", "Intermediate", "Advanced", "Professional"]
MIN_LEVEL, MAX_LEVEL = 1, len(LEVELS)
START_LEVEL = 2  # Beginner

TECHNICAL_COUNT = 15
RESUME_COUNT = 3
SOFT_COUNT = 3  # including the intro question -> 6 soft/resume questions in total
PASS_THRESHOLD = 85.0
WRONG_STREAK_FOR_LEVEL_DOWN = 2

COURSES = {
    "BSC_IT": "B.Sc. Information Technology",
    "BSC_DS": "B.Sc. Data Science",
    "BSC_AI": "B.Sc. Artificial Intelligence",
    "BCA": "Bachelor of Computer Applications",
}
YEARS = {1: "FY (1st year)", 2: "SY (2nd year)", 3: "TY (3rd year)"}


def level_name(level: int) -> str:
    return LEVELS[level - 1]


@dataclass
class QuestionRef:
    """Lightweight view of a bank question, enough for selection."""
    id: int
    kind: str  # technical | soft | resume
    level: int
    year: int | None = None
    subject: str | None = None
    times_asked: int = 0


@dataclass
class AskedRecord:
    question_id: int
    kind: str
    level: int
    subject: str | None
    score: float | None = None
    is_correct: bool | None = None
    skipped: bool = False


@dataclass
class InterviewState:
    course: str
    year: int
    has_resume: bool = False
    level: int = START_LEVEL
    wrong_streak: int = 0
    phase: str = "intro"  # intro | technical | resume | soft | review | finished
    history: list[AskedRecord] = field(default_factory=list)
    parked: list[int] = field(default_factory=list)  # question ids answered wrong/skipped
    pending_question_id: int | None = None

    # ---- serialisation (stored as JSON on the interview row) ----
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "InterviewState":
        d = dict(d)
        d["history"] = [AskedRecord(**h) for h in d.get("history", [])]
        return cls(**d)

    # ---- derived ----
    def count(self, kind: str) -> int:
        return sum(1 for h in self.history if h.kind == kind)

    @property
    def asked_ids(self) -> set[int]:
        return {h.question_id for h in self.history}

    @property
    def level_name(self) -> str:
        return level_name(self.level)

    @property
    def total_planned(self) -> int:
        return TECHNICAL_COUNT + SOFT_COUNT + (RESUME_COUNT if self.has_resume else 0)


class QuestionPool(Protocol):
    def candidates(self, kind: str, course: str, max_year: int, level: int | None,
                   exclude_ids: set[int]) -> list[QuestionRef]: ...


def years_in_scope(year: int) -> list[int]:
    """FY -> [1], SY -> [1, 2], TY -> [1, 2, 3]."""
    return list(range(1, year + 1))


def _next_phase(state: InterviewState) -> str:
    if state.count("soft") < 1:
        return "intro"
    if state.count("technical") < TECHNICAL_COUNT:
        return "technical"
    if state.has_resume and state.count("resume") < RESUME_COUNT:
        return "resume"
    if state.count("soft") < SOFT_COUNT:
        return "soft"
    return "review" if state.parked else "finished"


def _weight(q: QuestionRef, state: InterviewState, recent_subjects: list[str]) -> float:
    w = 1.0 / (1.0 + q.times_asked)  # prefer rarely asked -> different candidates get different sets
    if q.year == state.year:
        w *= 2.0  # current-year subjects are most relevant
    if q.subject and q.subject in recent_subjects:
        w *= 0.15  # avoid the same subject back-to-back
    return w


def choose_question(state: InterviewState, pool: QuestionPool, kind: str,
                    rng: random.Random | None = None) -> QuestionRef | None:
    """Weighted random pick; widens the level search outward if the exact level is empty."""
    rng = rng or random.Random()
    recent = [h.subject for h in state.history[-2:] if h.subject]
    if kind != "technical":
        options = pool.candidates(kind, state.course, state.year, None, state.asked_ids)
    else:
        options = []
        for delta in (0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5):
            lvl = state.level + delta
            if MIN_LEVEL <= lvl <= MAX_LEVEL:
                options = pool.candidates(kind, state.course, state.year, lvl, state.asked_ids)
                if options:
                    break
    if not options:
        return None
    weights = [_weight(q, state, recent) for q in options]
    return rng.choices(options, weights=weights, k=1)[0]


def next_step(state: InterviewState, pool: QuestionPool, rng: random.Random | None = None) -> dict:
    """Decide what comes next. Returns {"phase", "question"} where question may be None
    (technical bank exhausted -> caller may generate one with the LLM)."""
    phase = _next_phase(state)
    state.phase = phase
    if phase in ("review", "finished"):
        state.pending_question_id = None
        return {"phase": phase, "question": None}
    kind = "soft" if phase == "intro" else phase
    q = choose_question(state, pool, kind, rng)
    if q is None and kind == "resume":
        # No resume questions available: treat as done and move on.
        state.has_resume = False
        return next_step(state, pool, rng)
    state.pending_question_id = q.id if q else None
    return {"phase": phase, "question": q}


def record_answer(state: InterviewState, q: QuestionRef, score: float, skipped: bool = False) -> dict:
    """Apply one graded answer and the level rules. Returns what changed (for the UI)."""
    is_correct = (not skipped) and score >= PASS_THRESHOLD
    state.history.append(AskedRecord(question_id=q.id, kind=q.kind, level=q.level, subject=q.subject,
                                     score=round(score, 1), is_correct=is_correct, skipped=skipped))
    state.pending_question_id = None
    if not is_correct and q.id not in state.parked:
        state.parked.append(q.id)

    old = state.level
    if q.kind == "technical":
        if is_correct:
            state.level = min(MAX_LEVEL, state.level + 1)
            state.wrong_streak = 0
        else:
            state.wrong_streak += 1
            if state.wrong_streak >= WRONG_STREAK_FOR_LEVEL_DOWN:
                state.level = max(MIN_LEVEL, state.level - 1)
                state.wrong_streak = 0
    change = "up" if state.level > old else "down" if state.level < old else "same"
    return {"is_correct": is_correct, "level": state.level, "level_name": state.level_name,
            "level_change": change}


def summary(state: InterviewState) -> dict:
    """Score summary used by the final report."""
    def avg(xs):
        return round(sum(xs) / len(xs), 1) if xs else 0.0

    tech = [h for h in state.history if h.kind == "technical"]
    soft = [h for h in state.history if h.kind in ("soft", "resume")]
    peak = max((h.level for h in tech), default=state.level)
    # Harder questions answered correctly count more (level-weighted accuracy).
    weighted = sum((h.score or 0) * math.sqrt(h.level) for h in tech)
    weight_total = sum(math.sqrt(h.level) for h in tech)
    return {
        "technical_avg": avg([h.score or 0 for h in tech]),
        "technical_weighted": round(weighted / weight_total, 1) if weight_total else 0.0,
        "soft_avg": avg([h.score or 0 for h in soft]),
        "correct_count": sum(1 for h in state.history if h.is_correct),
        "skipped_count": sum(1 for h in state.history if h.skipped),
        "total_answered": len(state.history),
        "final_level": state.level_name,
        "peak_level": level_name(peak),
        "level_trail": [h.level for h in tech],
        "parked_count": len(state.parked),
    }
