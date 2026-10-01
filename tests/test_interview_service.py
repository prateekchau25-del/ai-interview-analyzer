"""End-to-end interview flow on an in-memory SQLite DB (RAM only, nothing written to disk).

The grader is replaced by a deterministic fake so this test checks orchestration, not NLP quality.
"""
import random

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.analyzers.nlp import Grade
from backend.app.db.models import Answer, Base, Question
from backend.app.db.seed import _insert_missing, load_soft_skills
from backend.app.services import interview_engine as eng
from backend.app.services import interview_service as svc


@pytest.fixture
def db():
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    _insert_missing(session, "soft", load_soft_skills())
    rows = []
    for year in (1, 2):
        for subj in ("DBMS", "Python", "OS"):
            for level in range(1, 7):
                for i in range(3):
                    rows.append({"course": "BSC_IT", "year": year, "semester": year * 2, "subject": subj,
                                 "topic": f"t{i}", "level": level, "text": f"{subj} Y{year} L{level} Q{i}?",
                                 "ideal_answer": "model answer", "keywords": ["k"]})
    _insert_missing(session, "technical", rows)
    session.commit()
    yield session
    session.close()
    engine.dispose()


def fake_grade(question, model_answer, keywords, transcript):
    score = {"correct": 95.0, "partial": 70.0, "wrong": 20.0}.get(transcript, 0.0)
    return Grade(score=score, is_correct=score >= 85, skipped=transcript == "", verdict="x",
                 feedback="fb", missing_points=["point a"] if score < 85 else [], method="fake")


@pytest.fixture(autouse=True)
def _fakes(monkeypatch):
    monkeypatch.setattr(svc.nlp, "grade_answer", fake_grade)
    monkeypatch.setattr(svc.nlp, "explain_answer", lambda q, a, t: f"explained: {q}")


def add_resume_questions(db, cand):
    for i in range(3):
        db.add(Question(kind="resume", source="resume", candidate_id=cand.id, text=f"Resume project {i}?",
                        ideal_answer="x", keywords=["x"], category="resume"))
    db.commit()


def run(db, answer_for, cand=None, seed=0):
    rng = random.Random(seed)
    cand = cand or svc.register_candidate(db, name="Asha", college_name="K.E.S. Shroff College",
                                          roll_no="it-101", course="BSC_IT", year=2)
    interview = svc.start_interview(db, cand)
    log = []
    while True:
        q = svc.next_question(db, interview, rng)
        if q["phase"] in ("review", "finished"):
            return cand, interview, log, q["phase"]
        assert svc.next_question(db, interview, rng)["question_id"] == q["question_id"]  # refresh-safe
        res = svc.submit_answer(db, interview, answer_for(q), rng=rng)
        log.append((q, res))


def test_full_interview_flow(db):
    cand = svc.register_candidate(db, name="Asha", college_name="K.E.S. Shroff College", roll_no="IT-101",
                                  course="BSC_IT", year=2)
    add_resume_questions(db, cand)
    answers = iter(["correct", "correct", "partial", "correct", "wrong", "wrong", ""] + ["correct"] * 40)
    cand, interview, log, end = run(db, lambda q: next(answers), cand)

    phases = [q["phase"] for q, _ in log]
    assert phases[0] == "intro"
    assert phases.count("technical") == eng.TECHNICAL_COUNT
    assert phases.count("resume") == eng.RESUME_COUNT
    assert "followup" in phases
    assert phases[phases.index("followup") - 1] in ("technical", "resume")
    assert end == "review"

    # live messages: correct / not correct / skipped
    msgs = {r["message"] for _, r in log}
    assert "Correct answer! Well done." in msgs and "This answer is not correct." in msgs
    assert any("come back" in m for m in msgs)

    # adaptive level: never outside 1..6 and technical questions follow the engine level
    tech_levels = [q["level"] for q, _ in log if q["phase"] == "technical"]
    assert tech_levels[0] == eng.START_LEVEL and max(tech_levels) > eng.START_LEVEL

    # review -> "Yes" explanations
    items = svc.review_items(db, interview)
    assert items and all(i["explanation"] is None for i in items)
    explained = svc.explain_parked(db, interview)
    assert all(i["explanation"].startswith("explained:") for i in explained)

    report = svc.finish_interview(db, interview)
    assert report["status"] == "completed" and 0 <= report["overall_score"] <= 100
    assert len(report["answers"]) == len(log)
    assert all(a["score"] is not None for a in report["answers"])


def test_same_student_is_reused_and_gets_new_questions(db):
    c1, i1, log1, _ = run(db, lambda q: "correct")
    c2 = svc.register_candidate(db, name="Asha K", college_name="k.e.s.  shroff college", roll_no="IT-101 ",
                                course="BSC_IT", year=2)
    assert c2.id == c1.id  # same identity despite spacing/case differences
    _, i2, log2, _ = run(db, lambda q: "correct", cand=c2, seed=1)
    tech1 = {q["question_id"] for q, _ in log1 if q["phase"] == "technical"}
    tech2 = {q["question_id"] for q, _ in log2 if q["phase"] == "technical"}
    assert not tech1 & tech2  # second interview avoids questions from the first


def test_year_scope_and_times_asked(db):
    cand = svc.register_candidate(db, name="Ravi", college_name="KES", roll_no="1", course="BSC_IT", year=1)
    _, interview, log, _ = run(db, lambda q: "wrong", cand=cand)
    for q, _ in log:
        if q["phase"] == "technical":
            assert db.get(Question, q["question_id"]).year == 1
    assert db.scalar(select(func.sum(Question.times_asked))) == db.scalar(select(func.count(Answer.id)))


def test_invalid_course_rejected(db):
    with pytest.raises(ValueError):
        svc.register_candidate(db, name="x", college_name="y", roll_no="1", course="BCOM", year=1)


def test_repeats_allowed_only_when_bank_is_exhausted(db):
    cand = svc.register_candidate(db, name="Neha", college_name="KES", roll_no="9", course="BSC_IT", year=1)
    # Year-1 bank has 54 questions; four interviews of 15 need 60, so the 4th must reuse a few.
    logs = []
    for seed in range(4):
        _, _, log, _ = run(db, lambda q: "correct", cand=cand, seed=seed)
        tech = [q["question_id"] for q, _ in log if q["phase"] == "technical"]
        assert len(tech) == eng.TECHNICAL_COUNT
        logs.append(tech)
    assert not set(logs[0]) & set(logs[1]) and not (set(logs[0]) | set(logs[1])) & set(logs[2])
    assert set(logs[3]) & (set(logs[0]) | set(logs[1]) | set(logs[2]))  # only now repeats appear
