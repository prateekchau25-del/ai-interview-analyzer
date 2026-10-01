"""API smoke test through FastAPI's TestClient (in-memory DB, fake grader, no network)."""
import pytest
from fastapi.testclient import TestClient
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate

from backend.app.analyzers.nlp import Grade
from backend.app.db.models import Base
from backend.app.db.seed import _insert_missing, load_soft_skills
from backend.app.db.session import SessionLocal, engine
from backend.app.main import app
from backend.app.services import interview_service as svc
from tests.test_resume import STRONG


@pytest.fixture(scope="module")
def client():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        _insert_missing(db, "soft", load_soft_skills())
        _insert_missing(db, "technical", [
            {"course": "BSC_DS", "year": 1, "subject": "Python", "topic": "t", "level": lvl,
             "text": f"Q L{lvl} #{i}", "ideal_answer": "a", "keywords": ["a"]}
            for lvl in range(1, 7) for i in range(4)])
        db.commit()
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _fakes(monkeypatch):
    def grade(q, a, k, t, kind="technical"):
        s = 95.0 if t == "good answer" else 0.0
        return Grade(score=s, is_correct=s >= 85, skipped=not t, verdict="v", feedback="f")
    monkeypatch.setattr(svc.nlp, "grade_answer", grade)
    monkeypatch.setattr(svc.nlp, "explain_answer", lambda q, a, t: "because...")


def test_full_api_flow(client, tmp_path):
    assert client.get("/health").json()["status"] == "ok"
    assert set(client.get("/meta").json()["courses"]) == {"BSC_IT", "BSC_DS", "BSC_AI", "BCA"}

    bad = client.post("/candidates", json={"name": "Zoya", "college_name": "KES", "roll_no": "1",
                                           "course": "BCOM", "year": 1})
    assert bad.status_code == 422

    cand = client.post("/candidates", json={"name": "Zoya Khan", "college_name": "KES Shroff",
                                            "roll_no": "DS-07", "course": "BSC_DS", "year": 1}).json()
    pdf = tmp_path / "cv.pdf"
    SimpleDocTemplate(str(pdf)).build([Paragraph(ln or "&nbsp;", getSampleStyleSheet()["Normal"])
                                       for ln in STRONG.splitlines()])
    with pdf.open("rb") as f:
        ats = client.post(f"/candidates/{cand['id']}/resume",
                          files={"file": ("cv.pdf", f, "application/pdf")})
    assert ats.status_code == 200 and ats.json()["ats_pass"]
    assert client.post(f"/candidates/{cand['id']}/resume",
                       files={"file": ("cv.exe", b"x", "application/octet-stream")}).status_code == 400

    started = client.post("/interviews", json={"candidate_id": cand["id"]}).json()
    iid, q = started["interview_id"], started["question"]
    assert q["phase"] == "intro"

    answered = 0
    while q["phase"] not in ("review", "finished"):
        if answered % 4 == 3:
            r = client.post(f"/interviews/{iid}/answer", data={"skip": "true"}).json()
            assert r["skipped"]
        else:
            r = client.post(f"/interviews/{iid}/answer", data={"text": "good answer"}).json()
            expected = "Thank you for your answer." if q["kind"] == "soft" else "Correct answer! Well done."
            assert r["message"] == expected
        answered += 1
        q = r["next"]
    assert q["phase"] == "review"
    assert client.post(f"/interviews/{iid}/answer", data={"text": "x"}).status_code == 409

    rv = client.get(f"/interviews/{iid}/review").json()
    assert rv["count"] > 0 and rv["prompt"]
    no = client.post(f"/interviews/{iid}/review", json={"want_answers": False}).json()
    assert "Thank you" in no["message"]
    yes = client.post(f"/interviews/{iid}/review", json={"want_answers": True}).json()
    assert all(i["explanation"] == "because..." for i in yes["items"])

    rep = client.post(f"/interviews/{iid}/finish").json()
    assert rep["status"] == "completed" and rep["summary"]["ats_pass"] is True
    assert len(rep["answers"]) == answered
    listing = client.get("/candidates").json()
    assert listing[0]["last_interview_id"] == iid
