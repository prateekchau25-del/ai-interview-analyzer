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
