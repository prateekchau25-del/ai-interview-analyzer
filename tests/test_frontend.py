"""Streamlit pages rendered headlessly (AppTest) with a fake API module - no server needed."""
import sys
import types
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
SYLLABUS = {"BSC_IT": {"1": ["Database Management Systems", "Python Programming"]},
            "BSC_DS": {"3": ["Machine Learning", "Data Visualization"]}}


@pytest.fixture
def fake_api(monkeypatch):
    calls = {}
    api = types.ModuleType("api")

    class ApiError(RuntimeError):
        pass

    def register_teacher(name, college, emp, teaching):
        calls["teacher"] = (name, college, emp, teaching)
        return {"id": 1, "name": name, "college_name": college, "roll_no": emp, "role": "teacher",
                "teaching": teaching, "course_name": None, "year_name": None, "has_resume": False}

    api.ApiError = ApiError
    api.syllabus = lambda: SYLLABUS
    api.register_teacher = register_teacher
    api.health = lambda: {"llm_enabled": False, "llm_provider": "none", "whisper_model": "small"}
    api.tts = lambda text, voice="female": None
    monkeypatch.setitem(sys.modules, "api", api)
    monkeypatch.syspath_prepend(str(FRONTEND))
    monkeypatch.delitem(sys.modules, "ui", raising=False)
    return calls


def test_teacher_registration_multiselect(fake_api):
    at = AppTest.from_file(str(FRONTEND / "app.py"), default_timeout=30).run()
    at.switch_page("pages/1_register.py").run()
    at.radio[0].set_value("Teacher (IT Department)").run()
    at.text_input(key="t_name").input("Prof. Sneha Iyer")
    at.text_input(key="t_emp").input("KES-IT-042")
    at.multiselect(key="t_courses").select("BSC_IT").select("BSC_DS").run()
    at.multiselect(key="t_years_BSC_IT").select(1).run()
    at.multiselect(key="t_years_BSC_DS").select(3).run()
    assert at.multiselect(key="t_subj_BSC_IT_1").options == SYLLABUS["BSC_IT"]["1"]
    at.multiselect(key="t_subj_BSC_IT_1").select("Database Management Systems").run()
    at.multiselect(key="t_subj_BSC_DS_3").select("Machine Learning").select("Data Visualization").run()
    next(b for b in at.button if b.label == "Save teacher profile").click().run()

    assert not at.exception and not at.error
    name, college, emp, teaching = fake_api["teacher"]
    assert (name, emp) == ("Prof. Sneha Iyer", "KES-IT-042")
    assert {(t["course"], t["year"], t["subject"]) for t in teaching} == {
        ("BSC_IT", 1, "Database Management Systems"), ("BSC_DS", 3, "Machine Learning"),
        ("BSC_DS", 3, "Data Visualization")}
    assert any("Teacher profile saved" in s.value for s in at.success)


def test_teacher_must_pick_a_subject(fake_api):
    at = AppTest.from_file(str(FRONTEND / "app.py"), default_timeout=30).run()
    at.switch_page("pages/1_register.py").run()
    at.radio[0].set_value("Teacher (IT Department)").run()
    at.text_input(key="t_name").input("Prof. X").run()
    next(b for b in at.button if b.label == "Save teacher profile").click().run()
    assert any("at least one subject" in e.value for e in at.error)
    assert "teacher" not in fake_api


REPORT = {
    "interview_id": 7, "status": "completed", "overall_score": 78.4,
    "verdict": "Good - Minor Improvements Needed",
    "technical_score": 81.0, "soft_score": 70.0,
    "candidate": {"name": "Asha", "college": "KES", "roll_no": "IT-1", "role": "student", "teaching": None,
                  "course": "B.Sc. Information Technology", "year": "SY (2nd year)"},
    "summary": {"communication": 74.0, "ats_score": 91.0, "ats_pass": True, "correct_count": 9,
                "total_answered": 20, "skipped_count": 1, "peak_level": "Advanced", "followup_count": 3,
                "level_trail": [2, 3, 4, 4, 5], "subject_scores": {"DBMS": {"avg": 88, "questions": 3}},
                "delivery": {"fluency": 70, "voice_confidence": 85, "confident_language": 67, "avg_wpm": 128,
                             "total_fillers": 6, "total_hedges": 4,
                             "emotion_counts": {"calm": 5, "nervous": 2},
                             "frequent_voice_notes": ["Your voice was quiet - speak a little louder."]},
                "big_five": {"traits": {"openness": {"score": 64, "leaning": "curious"}},
                             "words_analyzed": 900, "model_accuracy": 0.58, "disclaimer": "Indicator only."}},
    "resume": None,
    "answers": [{"n": 1, "question": "What is a primary key?", "kind": "technical", "level": 2,
                 "level_name": "Beginner", "score": 92.0, "verdict": "correct", "skipped": False,
                 "transcript": "unique row id", "feedback": "Good", "explanation": None}],
}


def test_report_page_renders_delivery_and_big_five(fake_api):
    sys.modules["api"].report = lambda iid: REPORT
    sys.modules["api"].report_pdf = lambda iid: b"%PDF-1.4 fake"
    at = AppTest.from_file(str(FRONTEND / "app.py"), default_timeout=30).run()
    at.switch_page("pages/3_report.py")
    at.query_params["interview"] = "7"
    at.run()
    assert not at.exception, at.exception
    subs = [s.value for s in at.subheader]
    assert "How you spoke" in subs and "Knowledge by subject" in subs
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Voice confidence"] == "85/100" and metrics["Overall score"] == "78%"


def test_interview_page_renders_with_webcam_panel(fake_api):
    q = {"question_id": 11, "text": "What is normalization?", "phase": "technical", "kind": "technical",
         "level": 2, "level_name": "Beginner", "subject": "DBMS", "topic": "nf", "time_limit_sec": 120,
         "number": 2, "planned_total": 21, "technical_done": 0, "technical_total": 15}
    at = AppTest.from_file(str(FRONTEND / "app.py"), default_timeout=60).run()
    at.session_state["candidate"] = {"id": 1, "name": "Asha", "role": "student", "course_name": "B.Sc IT",
                                     "year_name": "SY", "has_resume": False}
    at.session_state["interview_id"] = 5
    at.session_state["question"] = q
    at.switch_page("pages/2_interview.py").run()
    assert not at.exception, at.exception
    assert any("What is normalization?" in m.value for m in at.markdown)
    assert any(b.label == "I don't know / Skip" for b in at.button)
