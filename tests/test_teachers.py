"""Teacher interviews on the REAL question bank (in-memory SQLite, fake grader)."""
import random

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.analyzers.nlp import Grade
from backend.app.db.models import Base, Question
from backend.app.db.seed import load_question_bank, seed_defaults
from backend.app.services import interview_engine as eng
from backend.app.services import interview_service as svc
from backend.app.services import syllabus

TEACHING = [
    {"course": "BSC_IT", "year": 1, "subject": "Database Management Systems"},
    {"course": "BSC_DS", "year": 3, "subject": "Machine Learning"},
]


def test_every_syllabus_subject_has_bank_questions():
    bank = {(r["course"], r["year"], r["subject"]) for r in load_question_bank()}
    missing = [(c, y, s) for c, years in syllabus.catalog().items() for y, subs in years.items()
               for s in (x["subject"] for x in subs) if (c, y, s) not in bank]
    assert not missing, missing


@pytest.fixture
def db():
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    seed_defaults(session)
    yield session
    session.close()


@pytest.fixture(autouse=True)
def _fake_grader(monkeypatch):
    def grade(q, a, k, t, kind="technical"):
        s = {"good": 92.0, "weak": 60.0}.get(t, 0.0)
        return Grade(score=s, is_correct=s >= 85, skipped=not t, verdict="v", feedback="f",
                     missing_points=["x"] if s < 85 else [])
    monkeypatch.setattr(svc.nlp, "grade_answer", grade)


def test_teacher_registration_validates_syllabus(db):
    with pytest.raises(ValueError):
        svc.register_teacher(db, name="Dr. Rao", college_name="KES", teaching=[])
    with pytest.raises(ValueError):
        svc.register_teacher(db, name="Dr. Rao", college_name="KES",
                             teaching=[{"course": "BSC_IT", "year": 1, "subject": "Machine Learning"}])
    t = svc.register_teacher(db, name="Dr. Rao", college_name="KES", employee_id="emp-7",
                             teaching=TEACHING + [{"course": "BSC_DS", "year": 3,
                                                   "subject": "Machine Learning (elective)"}])
    assert t.role == "teacher" and len(t.teaching) == 2  # elective suffix cleaned + de-duplicated
    again = svc.register_teacher(db, name="Dr. K. Rao", college_name="kes", employee_id="EMP-7 ",
                                 teaching=TEACHING[:1])
    assert again.id == t.id and len(again.teaching) == 1


def test_teacher_interview_only_their_subjects_hard_and_counter_questions(db):
    t = svc.register_teacher(db, name="Prof. Mehta", college_name="KES", employee_id="E1", teaching=TEACHING)
    interview = svc.start_interview(db, t)
    rng, log = random.Random(1), []
    answers = iter(["good", "weak", "good", ""] * 30)
    while True:
        q = svc.next_question(db, interview, rng)
        if q["phase"] in ("review", "finished"):
            break
        res = svc.submit_answer(db, interview, next(answers), rng=rng)
        log.append((q, res))

    first = db.get(Question, log[0][0]["question_id"])
    assert first.audience == "teacher" and first.topic == "intro"
    allowed = {(x["course"], x["year"], x["subject"]) for x in TEACHING}
    tech = [db.get(Question, q["question_id"]) for q, _ in log if q["phase"] == "technical"]
    assert len(tech) == eng.PROFILES["teacher"].technical
    assert all((x.course, x.year, x.subject) in allowed for x in tech)
    assert min(x.level for x in tech) >= eng.PROFILES["teacher"].min_level
    # every answered technical question is followed by 1-2 counter questions
    phases = [q["phase"] for q, _ in log]
    for i, (q, res) in enumerate(log):
        if q["phase"] == "technical" and not res["skipped"]:
            n = 0
            while i + 1 + n < len(log) and phases[i + 1 + n] == "followup":
                n += 1
            assert 1 <= n <= 2
    soft = [db.get(Question, q["question_id"]) for q, _ in log if q["phase"] in ("intro", "soft")]
    assert all(x.audience == "teacher" for x in soft)

    report = svc.finish_interview(db, interview)
    assert report["candidate"]["role"] == "teacher"
    assert set(report["summary"]["subject_scores"]) <= {x["subject"] for x in TEACHING}
