"""Thin HTTP client for the FastAPI backend."""
import os

import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 180  # transcription + LLM grading can take a while on CPU


class ApiError(RuntimeError):
    pass


def _call(method: str, path: str, **kwargs):
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=TIMEOUT, **kwargs)
    except requests.ConnectionError as e:
        raise ApiError(f"Cannot reach the backend at {API_URL}. "
                       "Is `uvicorn backend.app.main:app` running?") from e
    if r.status_code == 204:
        return None
    if not r.ok:
        try:
            detail = r.json().get("detail")
        except ValueError:
            detail = r.text
        raise ApiError(str(detail))
    return r.json() if r.headers.get("content-type", "").startswith("application/json") else r.content


def health() -> dict:
    return _call("GET", "/health")


def meta() -> dict:
    return _call("GET", "/meta")


def register(name: str, college_name: str, roll_no: str, course: str, year: int) -> dict:
    return _call("POST", "/candidates", json={"name": name, "college_name": college_name,
                                              "roll_no": roll_no, "course": course, "year": year})


def syllabus() -> dict:
    """{course: {year(str): [subject, ...]}}"""
    return _call("GET", "/meta/syllabus")


def register_teacher(name: str, college_name: str, employee_id: str, teaching: list[dict]) -> dict:
    return _call("POST", "/teachers", json={"name": name, "college_name": college_name,
                                            "employee_id": employee_id or None, "teaching": teaching})


def upload_resume(candidate_id: int, filename: str, data: bytes) -> dict:
    return _call("POST", f"/candidates/{candidate_id}/resume", files={"file": (filename, data)})


def candidate(candidate_id: int) -> dict:
    return _call("GET", f"/candidates/{candidate_id}")


def candidates(course: str | None = None, role: str | None = None) -> list[dict]:
    params = {k: v for k, v in (("course", course), ("role", role)) if v}
    return _call("GET", "/candidates", params=params or None)


def start_interview(candidate_id: int) -> dict:
    return _call("POST", "/interviews", json={"candidate_id": candidate_id})


def current_question(interview_id: int) -> dict:
    return _call("GET", f"/interviews/{interview_id}/question")


def answer_audio(interview_id: int, filename: str, data: bytes, vision: str | None = None) -> dict:
    return _call("POST", f"/interviews/{interview_id}/answer", files={"audio": (filename, data)},
                 data={"vision": vision} if vision else None)


def answer_text(interview_id: int, text: str, vision: str | None = None) -> dict:
    return _call("POST", f"/interviews/{interview_id}/answer",
                 data={"text": text, **({"vision": vision} if vision else {})})


def skip(interview_id: int) -> dict:
    return _call("POST", f"/interviews/{interview_id}/answer", data={"skip": "true"})


def review(interview_id: int) -> dict:
    return _call("GET", f"/interviews/{interview_id}/review")


def review_decision(interview_id: int, want_answers: bool) -> dict:
    return _call("POST", f"/interviews/{interview_id}/review", json={"want_answers": want_answers})


def finish(interview_id: int) -> dict:
    return _call("POST", f"/interviews/{interview_id}/finish")


def report(interview_id: int) -> dict:
    return _call("GET", f"/interviews/{interview_id}/report")


def tts(text: str, voice: str = "female") -> bytes | None:
    return _call("GET", "/tts", params={"text": text[:1500], "voice": voice})
