"""Interview orchestration: DB <-> adaptive engine <-> speech/grading/LLM.

Live path per answer (synchronous, a few seconds on CPU):
    audio -> Whisper transcript -> grade vs model answer -> engine (level/follow-up) -> next question
Heavier media analysis (voice tone, face, eye contact) runs later in the background worker.
"""
import json
import logging
import random
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from backend.app.analyzers import llm, nlp
from backend.app.analyzers.nlp import Grade
from backend.app.db.models import Answer, Candidate, Interview, Question
from backend.app.services import interview_engine as eng

log = logging.getLogger(__name__)

SYLLABUS_FILE = Path(__file__).resolve().parents[1] / "data" / "syllabus.json"

# Overall score weights (body-language/voice metrics are added by the media worker when available).
W_TECHNICAL, W_SOFT, W_COMMUNICATION = 0.65, 0.20, 0.15


def verdict_for(score: float) -> str:
    if score >= 85:
        return "Excellent - Interview Ready"
    if score >= 70:
        return "Good - Minor Improvements Needed"
    if score >= 50:
        return "Average - Needs Practice"
    return "Needs Significant Improvement"


# ------------------------------------------------------------------ question pool backed by the DB
class DbQuestionPool:
    """Implements interview_engine.QuestionPool.

    Questions this candidate saw in earlier interviews are hidden, so the engine first widens to nearby
    levels for fresh questions. Only if nothing fresh is left anywhere does the caller retry with
    allow_repeats=True. Resume questions are personal and may always repeat."""

    def __init__(self, db: Session, candidate: Candidate, current_interview_id: int | None = None):
        self.db, self.candidate = db, candidate
        self.allow_repeats = False
        q = select(Answer.question_id).join(Interview).where(
            Interview.candidate_id == candidate.id, Answer.question_id.is_not(None))
        if current_interview_id:
            q = q.where(Interview.id != current_interview_id)
        self.previously_asked = set(db.scalars(q).all())

    def _query(self, kind: str, course: str, max_year: int, level: int | None):
        q = select(Question).where(Question.is_active.is_(True))
        if kind == "technical":
            q = q.where(Question.kind == "technical", Question.course == course, Question.year <= max_year)
            if level is not None:
                q = q.where(Question.level == level)
        elif kind == "intro":
            q = q.where(Question.kind == "soft", Question.topic == "intro")
        elif kind == "soft":
            q = q.where(Question.kind == "soft", (Question.topic.is_(None)) | (Question.topic != "intro"))
        elif kind == "resume":
            q = q.where(Question.kind == "resume", Question.candidate_id == self.candidate.id)
        else:
            raise ValueError(kind)
        return q

    def candidates(self, kind, course, max_year, level, exclude_ids):
        rows = [r for r in self.db.scalars(self._query(kind, course, max_year, level)).all()
                if r.id not in exclude_ids]
        if not (self.allow_repeats or kind == "resume"):
            rows = [r for r in rows if r.id not in self.previously_asked]
        return [to_ref(r) for r in rows]


def to_ref(q: Question) -> eng.QuestionRef:
    kind = "soft" if q.kind == "soft" else q.kind
    return eng.QuestionRef(id=q.id, kind=kind, level=q.level or 0, year=q.year, subject=q.subject,
                           times_asked=q.times_asked)


# ------------------------------------------------------------------ candidates & resume
def register_candidate(db: Session, *, name: str, college_name: str, roll_no: str, course: str,
                       year: int, email: str | None = None) -> Candidate:
    if course not in eng.COURSES:
        raise ValueError(f"course must be one of {list(eng.COURSES)}")
    if year not in eng.YEARS:
        raise ValueError("year must be 1, 2 or 3")
    college_key, roll_key = " ".join(college_name.split()), roll_no.strip().upper()
    cand = db.scalar(select(Candidate).where(
        func.lower(Candidate.college_name) == college_key.lower(), Candidate.roll_no == roll_key,
        Candidate.course == course))
    if cand is None:
        cand = Candidate(college_name=college_key, roll_no=roll_key, course=course, name=name.strip(),
                         year=year, email=email)
        db.add(cand)
    else:
        cand.name, cand.year, cand.email = name.strip(), year, email or cand.email
    db.commit()
    return cand


def process_resume(db: Session, candidate: Candidate, path: Path) -> dict:
    """ATS analysis + personalised resume questions (replaces any earlier resume questions)."""
    from backend.app.analyzers import resume as resume_an

    report, text = resume_an.analyze_resume(path, candidate.course)
    candidate.resume_path, candidate.resume_text = str(path), text
    candidate.resume_analysis, candidate.ats_score = report.to_dict(), report.ats_score
    db.execute(update(Question).where(Question.candidate_id == candidate.id, Question.kind == "resume")
               .values(is_active=False))
    for q in resume_an.resume_questions(text, candidate.course, n=eng.RESUME_COUNT):
        db.add(Question(kind="resume", source="resume", candidate_id=candidate.id, text=q["text"],
                        ideal_answer=q["ideal_answer"], keywords=q.get("keywords", []),
                        topic=q.get("topic"), category="resume", level=0, time_limit_sec=120))
    db.commit()
    return report.to_dict()


# ------------------------------------------------------------------ LLM fallback question generation
def _random_topic(course: str, max_year: int, rng: random.Random) -> tuple[int, int, str, str] | None:
    data = json.loads(SYLLABUS_FILE.read_text(encoding="utf-8"))
    years = data["courses"].get(course, {}).get("years", {})
    options = [(int(y), int(sem), s["subject"], t)
               for y, yd in years.items() if int(y) <= max_year
               for sem, subs in yd["semesters"].items() for s in subs for t in s["topics"]]
    return rng.choice(options) if options else None


GEN_SYSTEM = """You write viva/interview questions for Indian undergraduates. Write ONE question on the given
syllabus topic at the given difficulty (1 Basic definition, 2 Beginner explanation, 3 Medium how-it-works,
4 Intermediate compare/why, 5 Advanced design/troubleshoot, 6 Professional industry scenario).
It must be answerable verbally in 1-2 minutes. Give a factually correct model answer (3-6 sentences)
and 4-8 lowercase keywords a correct answer must contain.
Return JSON: {"text": "...", "ideal_answer": "...", "keywords": ["..."]}"""


def generate_question(db: Session, course: str, max_year: int, level: int,
                      rng: random.Random | None = None) -> Question | None:
    """Create a brand-new syllabus question with the LLM (used when the bank has nothing left)."""
    if not llm.is_enabled():
        return None
    rng = rng or random.Random()
    pick = _random_topic(course, max_year, rng)
    if not pick:
        return None
    year, sem, subject, topic = pick
    try:
        data = llm.chat_json(GEN_SYSTEM, f"Course: {eng.COURSES[course]}\nSubject: {subject}\n"
                             f"Topic: {topic}\nDifficulty level: {level} ({eng.level_name(level)})",
                             temperature=0.7)
    except llm.LLMUnavailable as e:
        log.warning("Question generation failed: %s", e)
        return None
    if not data.get("text") or not data.get("ideal_answer"):
        return None
    q = Question(kind="technical", source="llm", course=course, year=year, semester=sem, subject=subject,
                 topic=topic[:300], level=level, text=data["text"], ideal_answer=data["ideal_answer"],
                 keywords=[str(k) for k in data.get("keywords", [])][:8], category="technical")
    db.add(q)
    db.flush()
    return q


# ------------------------------------------------------------------ interview lifecycle
def _state(interview: Interview) -> eng.InterviewState:
    return eng.InterviewState.from_dict(interview.state)


def _save_state(interview: Interview, state: eng.InterviewState) -> None:
    interview.state = state.to_dict()  # reassign so SQLAlchemy detects the JSON change


def start_interview(db: Session, candidate: Candidate) -> Interview:
    has_resume = db.scalar(select(func.count(Question.id)).where(
        Question.candidate_id == candidate.id, Question.kind == "resume", Question.is_active.is_(True))) > 0
    state = eng.InterviewState(course=candidate.course, year=candidate.year, has_resume=has_resume)
    interview = Interview(candidate_id=candidate.id, course=candidate.course, year=candidate.year,
                          status="in_progress", state=state.to_dict())
    db.add(interview)
    db.commit()
    return interview


def question_payload(q: Question, state: eng.InterviewState, phase: str) -> dict:
    asked = len(state.history)
    return {
        "question_id": q.id, "text": q.text, "phase": phase, "kind": q.kind,
        "level": q.level, "level_name": eng.level_name(q.level) if q.level else None,
        "subject": q.subject, "topic": q.topic, "time_limit_sec": q.time_limit_sec,
        "number": asked + 1, "planned_total": state.total_planned,
        "technical_done": state.count("technical"), "technical_total": eng.TECHNICAL_COUNT,
    }


def next_question(db: Session, interview: Interview, rng: random.Random | None = None) -> dict:
    """Return the current/next question payload, or {"phase": "review"|"finished"}."""
    state = _state(interview)
    if state.pending_question_id:  # page refresh: same question again
        q = db.get(Question, state.pending_question_id)
        return question_payload(q, state, state.phase)

    pool = DbQuestionPool(db, interview.candidate, interview.id)
    step = eng.next_step(state, pool, rng)
    phase = step["phase"]
    if phase in ("intro", "technical", "soft") and step["question"] is None and pool.previously_asked:
        if phase != "technical" or not llm.is_enabled():  # with an LLM, a fresh question beats a repeat
            pool.allow_repeats = True
            step = eng.next_step(state, pool, rng)
            phase = step["phase"]

    if phase == "followup":
        q = _make_followup(db, interview, step["parent_id"])
        state.pending_question_id = q.id
    elif phase in ("review", "finished"):
        interview.status = "review" if phase == "review" else interview.status
        _save_state(interview, state)
        db.commit()
        return {"phase": phase}
    else:
        ref = step["question"]
        if ref is None and phase == "technical":
            q = generate_question(db, interview.course, interview.year, state.level, rng)
            if q is None:
                # Only possible if the bank for this course/year is empty and no LLM is configured.
                log.error("No technical question available for interview %s", interview.id)
                return {"phase": "error", "message": "No technical questions are available for this "
                        "course/year. Run `python -m backend.app.db.setup` to load the question bank."}
            state.pending_question_id = q.id
        else:
            q = db.get(Question, ref.id)
    _save_state(interview, state)
    db.commit()
    return question_payload(q, state, phase)


def _make_followup(db: Session, interview: Interview, parent_qid: int) -> Question:
    parent_answer = db.scalar(select(Answer).where(Answer.interview_id == interview.id,
                                                   Answer.question_id == parent_qid)
                              .order_by(Answer.id.desc()))
    parent_q = db.get(Question, parent_qid)
    grade = Grade(**{k: v for k, v in (parent_answer.grade or {}).items() if k in Grade.__dataclass_fields__})
    fu = nlp.generate_followup(parent_q.text, parent_q.ideal_answer, parent_answer.transcript or "", grade)
    q = Question(kind="followup", source="followup", candidate_id=interview.candidate_id, text=fu["text"],
                 ideal_answer=fu["ideal_answer"], keywords=fu["keywords"], level=parent_q.level,
                 subject=parent_q.subject, topic=parent_q.topic, course=parent_q.course,
                 year=parent_q.year, category="followup", time_limit_sec=90)
    db.add(q)
    db.flush()
    return q


def submit_answer(db: Session, interview: Interview, transcript: str, *, speech: dict | None = None,
                  media_path: str | None = None, rng: random.Random | None = None) -> dict:
    """Grade the answer to the pending question and advance the engine."""
    state = _state(interview)
    if not state.pending_question_id:
        raise ValueError("No question is waiting for an answer.")
    q = db.get(Question, state.pending_question_id)
    grade = nlp.grade_answer(q.text, q.ideal_answer, q.keywords or [], transcript or "", kind=q.kind)

    ref = to_ref(q)
    if q.kind == "followup":
        ref.kind = "followup"
    result = eng.record_answer(state, ref, grade.score, skipped=grade.skipped, rng=rng)

    db.add(Answer(
        interview_id=interview.id, question_id=q.id, question_text=q.text, kind=ref.kind, level=q.level,
        order_index=len(state.history), transcript=transcript, media_path=media_path,
        status="queued" if media_path else "completed", score=grade.score, verdict=grade.verdict,
        skipped=grade.skipped, grade={**grade.to_dict(), "speech": speech} if speech else grade.to_dict(),
        duration_sec=(speech or {}).get("duration_sec"),
    ))
    q.times_asked = (q.times_asked or 0) + 1
    _save_state(interview, state)
    db.commit()

    if grade.skipped:
        message = ("Okay, let's move on." if q.kind == "soft"
                   else "No problem. We will come back to this question at the end.")
    elif q.kind == "soft":
        message = "Thank you for your answer."  # soft skills: feedback, not right/wrong
    elif result["is_correct"]:
        message = "Correct answer! Well done."
    else:
        message = "This answer is not correct."
    return {
        "score": grade.score, "verdict": grade.verdict, "is_correct": result["is_correct"],
        "skipped": grade.skipped, "message": message, "feedback": grade.feedback,
        "missing_points": grade.missing_points, "grading_method": grade.method,
        "level": result["level"], "level_name": result["level_name"], "level_change": result["level_change"],
        "followup_next": result["followup_next"], "transcript": transcript,
    }


# ------------------------------------------------------------------ end of interview
def review_items(db: Session, interview: Interview) -> list[dict]:
    """Questions the candidate could not answer correctly (for the Yes/No 'show answers' step)."""
    state = _state(interview)
    answers = {a.question_id: a for a in interview.answers}
    out = []
    for qid in state.parked:
        a = answers.get(qid)
        if a:
            out.append({"question_id": qid, "question": a.question_text, "your_answer": a.transcript,
                        "score": a.score, "skipped": a.skipped, "explanation": a.explanation})
    return out


def explain_parked(db: Session, interview: Interview) -> list[dict]:
    """Candidate pressed 'Yes': create (and store) a teacher-style explanation for each parked question."""
    state = _state(interview)
    for a in interview.answers:
        if a.question_id in state.parked and not a.explanation:
            q = db.get(Question, a.question_id)
            a.explanation = nlp.explain_answer(q.text, q.ideal_answer, a.transcript)
    db.commit()
    return review_items(db, interview)


def finish_interview(db: Session, interview: Interview) -> dict:
    state = _state(interview)
    s = eng.summary(state)
    fluency = [a.grade.get("speech", {}).get("fluency_score") for a in interview.answers
               if a.grade and a.grade.get("speech")]
    fluency = [f for f in fluency if f is not None]
    communication = round(sum(fluency) / len(fluency), 1) if fluency else None
    parts = [(s["technical_weighted"], W_TECHNICAL), (s["soft_avg"], W_SOFT)]
    if communication is not None:
        parts.append((communication, W_COMMUNICATION))
    overall = round(sum(v * w for v, w in parts) / sum(w for _, w in parts), 1)

    interview.technical_score = s["technical_weighted"]
    interview.soft_score = s["soft_avg"]
    interview.overall_score = overall
    interview.verdict = verdict_for(overall)
    interview.summary = {**s, "communication": communication,
                         "ats_score": interview.candidate.ats_score,
                         "ats_pass": (interview.candidate.ats_score or 0) >= 85}
    interview.status = "completed"
    interview.finished_at = datetime.now(UTC)
    state.phase = "finished"
    _save_state(interview, state)
    db.commit()
    return report_data(db, interview)


def report_data(db: Session, interview: Interview) -> dict:
    """Everything the final report page / PDF needs."""
    cand = interview.candidate
    return {
        "candidate": {"name": cand.name, "college": cand.college_name, "roll_no": cand.roll_no,
                      "course": eng.COURSES.get(cand.course, cand.course), "year": eng.YEARS.get(cand.year)},
        "interview_id": interview.id, "status": interview.status,
        "overall_score": interview.overall_score, "verdict": interview.verdict,
        "technical_score": interview.technical_score, "soft_score": interview.soft_score,
        "summary": interview.summary,
        "resume": cand.resume_analysis,
        "answers": [{"n": a.order_index, "question": a.question_text, "kind": a.kind, "level": a.level,
                     "level_name": eng.level_name(a.level) if a.level else None,
                     "score": a.score, "verdict": a.verdict, "skipped": a.skipped,
                     "transcript": a.transcript, "feedback": (a.grade or {}).get("feedback"),
                     "explanation": a.explanation} for a in interview.answers],
    }
