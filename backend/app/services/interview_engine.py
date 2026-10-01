"""Adaptive interview engine (pure logic, no DB / no I/O, so it is fully unit-testable).

Flow of one interview:
    intro (1 soft question) -> 15 technical (adaptive level) -> resume questions -> soft-skill questions
    -> review of parked (wrong/skipped) questions -> finished
    Follow-up (counter) questions about the candidate's own answer are inserted after some technical/resume
    answers: always after a partially correct answer, sometimes after a correct one (max 4 per interview).

Level rules:
    answer >= 85%                  -> level up (max Professional)
    2 not-correct answers in a row -> level down (min Basic)

Teacher mode (IT-department faculty): questions only from the subjects they taught, starts at Intermediate and
never goes below Medium, and EVERY answered question gets 1-2 counter questions.
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
MAX_FOLLOWUPS = 4
PARTIAL_MIN = 50.0  # 50-84% = partially correct -> probe the missing part
FOLLOWUP_ON_CORRECT_PROB = 0.3  # occasionally dig deeper after a correct answer
FOLLOWUP_PARENT_KINDS = ("technical", "resume")


@dataclass(frozen=True)
class Profile:
    technical: int
    resume: int
    soft: int  # including the intro question
    start_level: int
    min_level: int
    max_followups: int
    always_followup: bool  # teacher: every answered technical/resume question gets a counter question
    second_followup_prob: float = 0.0  # teacher: chance of a 2nd counter question after a correct follow-up


PROFILES = {
    "student": Profile(technical=TECHNICAL_COUNT, resume=RESUME_COUNT, soft=SOFT_COUNT,
                       start_level=START_LEVEL,
                       min_level=MIN_LEVEL, max_followups=MAX_FOLLOWUPS, always_followup=False),
    "teacher": Profile(technical=10, resume=2, soft=3, start_level=4, min_level=3, max_followups=24,
                       always_followup=True, second_followup_prob=0.5),
}

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
    kind: str  # technical | soft | resume | followup
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
    level: int | None = None  # defaults to the profile's start level
    mode: str = "student"  # student | teacher
    wrong_streak: int = 0
    phase: str = "intro"  # intro | technical | resume | soft | review | finished
    history: list[AskedRecord] = field(default_factory=list)
    parked: list[int] = field(default_factory=list)  # question ids answered wrong/skipped
    pending_question_id: int | None = None
    followup_for: int | None = None  # question id whose answer should get a counter question next
    followup_depth: int = 0  # how many counter questions in the current chain (teacher: up to 2)

    def __post_init__(self):
        if self.level is None:
            self.level = self.profile.start_level

    @property
    def profile(self) -> Profile:
        return PROFILES[self.mode]

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
        p = self.profile
        main = p.technical + (p.resume if self.has_resume else 0)
        followups = main if p.always_followup else 0  # at least one counter question per answer
        return main + p.soft + followups


class QuestionPool(Protocol):
    def candidates(self, kind: str, course: str, max_year: int, level: int | None,
                   exclude_ids: set[int]) -> list[QuestionRef]:
        """kind: technical | resume | soft (non-intro soft questions) | intro (soft, topic 'intro').
        Returned refs use kind 'soft' for both soft and intro questions."""
        ...


def years_in_scope(year: int) -> list[int]:
    """FY -> [1], SY -> [1, 2], TY -> [1, 2, 3]."""
    return list(range(1, year + 1))


def _next_phase(state: InterviewState) -> str:
    p = state.profile
    if state.count("soft") < 1:
        return "intro"
    if state.count("technical") < p.technical:
        return "technical"
    if state.has_resume and state.count("resume") < p.resume:
        return "resume"
    if state.count("soft") < p.soft:
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
            if state.profile.min_level <= lvl <= MAX_LEVEL:
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
    if state.followup_for is not None:
        # The caller generates the follow-up text from the candidate's last answer.
        state.phase = "followup"
        return {"phase": "followup", "question": None, "parent_id": state.followup_for}
    phase = _next_phase(state)
    state.phase = phase
    if phase in ("review", "finished"):
        state.pending_question_id = None
        return {"phase": phase, "question": None}
    q = choose_question(state, pool, phase, rng)  # phase name == pool kind (intro|technical|resume|soft)
    if q is None and phase == "intro":
        q = choose_question(state, pool, "soft", rng)
    if q is None and phase == "resume":
        # No resume questions available: treat as done and move on.
        state.has_resume = False
        return next_step(state, pool, rng)
    state.pending_question_id = q.id if q else None
    return {"phase": phase, "question": q}


def _wants_followup(state: InterviewState, q: QuestionRef, score: float, skipped: bool,
                    rng: random.Random) -> bool:
    p = state.profile
    if skipped or state.count("followup") >= p.max_followups:
        return False
    if q.kind == "followup":
        # Only teachers get a second counter question in the same chain.
        if not p.always_followup or state.followup_depth >= 2:
            return False
        return score < PASS_THRESHOLD or rng.random() < p.second_followup_prob
    if q.kind not in FOLLOWUP_PARENT_KINDS:
        return False
    if p.always_followup:
        return True
    if PARTIAL_MIN <= score < PASS_THRESHOLD:
        return True
    return score >= PASS_THRESHOLD and rng.random() < FOLLOWUP_ON_CORRECT_PROB


def record_answer(state: InterviewState, q: QuestionRef, score: float, skipped: bool = False,
                  rng: random.Random | None = None) -> dict:
    """Apply one graded answer and the level rules. Returns what changed (for the UI).
    Follow-up answers never change the level."""
    rng = rng or random.Random()
    is_correct = (not skipped) and score >= PASS_THRESHOLD
    state.history.append(AskedRecord(question_id=q.id, kind=q.kind, level=q.level, subject=q.subject,
                                     score=round(score, 1), is_correct=is_correct, skipped=skipped))
    state.pending_question_id = None
    # Soft-skill answers have no single right answer: they get feedback, not a "show me the answer" review.
    if q.kind != "soft" and not is_correct and q.id not in state.parked:
        state.parked.append(q.id)

    old = state.level
    if q.kind == "technical":
        if is_correct:
            state.level = min(MAX_LEVEL, state.level + 1)
            state.wrong_streak = 0
        else:
            state.wrong_streak += 1
            if state.wrong_streak >= WRONG_STREAK_FOR_LEVEL_DOWN:
                state.level = max(state.profile.min_level, state.level - 1)
                state.wrong_streak = 0
    change = "up" if state.level > old else "down" if state.level < old else "same"

    if _wants_followup(state, q, score, skipped, rng):
        state.followup_for = q.id  # the next counter question builds on THIS answer
        state.followup_depth = state.followup_depth + 1 if q.kind == "followup" else 1
    else:
        state.followup_for = None
        state.followup_depth = 0
    return {"is_correct": is_correct, "level": state.level, "level_name": state.level_name,
            "level_change": change, "followup_next": state.followup_for is not None}


def summary(state: InterviewState) -> dict:
    """Score summary used by the final report."""
    def avg(xs):
        return round(sum(xs) / len(xs), 1) if xs else 0.0

    tech = [h for h in state.history if h.kind == "technical"]
    soft = [h for h in state.history if h.kind in ("soft", "resume")]
    follow = [h for h in state.history if h.kind == "followup"]
    by_subject: dict[str, list[float]] = {}
    for h in tech + follow:
        if h.subject:
            by_subject.setdefault(h.subject, []).append(h.score or 0)
    peak = max((h.level for h in tech), default=state.level)
    # Harder questions answered correctly count more (level-weighted accuracy).
    weighted = sum((h.score or 0) * math.sqrt(h.level) for h in tech)
    weight_total = sum(math.sqrt(h.level) for h in tech)
    return {
        "technical_avg": avg([h.score or 0 for h in tech]),
        "technical_weighted": round(weighted / weight_total, 1) if weight_total else 0.0,
        "soft_avg": avg([h.score or 0 for h in soft]),
        "followup_avg": avg([h.score or 0 for h in follow]),
        "followup_count": len(follow),
        "correct_count": sum(1 for h in state.history if h.is_correct),
        "skipped_count": sum(1 for h in state.history if h.skipped),
        "total_answered": len(state.history),
        "final_level": state.level_name,
        "peak_level": level_name(peak),
        "level_trail": [h.level for h in tech],
        "parked_count": len(state.parked),
        "subject_scores": {k: {"avg": avg(v), "questions": len(v)} for k, v in sorted(by_subject.items())},
        "mode": state.mode,
    }
