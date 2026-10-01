"""REST API used by the Streamlit frontend."""
import logging
import shutil
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.analyzers import llm
from backend.app.core.config import get_settings
from backend.app.db.models import Candidate, Interview
from backend.app.db.session import get_db
from backend.app.services import interview_engine as eng
from backend.app.services import interview_service as svc
from backend.app.services import syllabus

log = logging.getLogger(__name__)
router = APIRouter()
DB = Annotated[Session, Depends(get_db)]

MAX_RESUME_MB = 5
MAX_AUDIO_MB = 25


# ------------------------------------------------------------------ helpers
def _candidate(db: Session, candidate_id: int) -> Candidate:
    c = db.get(Candidate, candidate_id)
    if not c:
        raise HTTPException(404, "Candidate not found")
    return c


def _interview(db: Session, interview_id: int) -> Interview:
    i = db.get(Interview, interview_id)
    if not i:
        raise HTTPException(404, "Interview not found")
    return i


def _save_upload(upload: UploadFile, folder: Path, allowed: set[str], max_mb: int) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type {suffix or '(none)'}; allowed: {sorted(allowed)}")
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{uuid.uuid4().hex}{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    if dest.stat().st_size > max_mb * 1024 * 1024:
        dest.unlink(missing_ok=True)
        raise HTTPException(413, f"File too large (max {max_mb} MB)")
    return dest


def candidate_out(c: Candidate) -> dict:
    return {"id": c.id, "name": c.name, "college_name": c.college_name, "roll_no": c.roll_no,
            "course": c.course, "course_name": eng.COURSES.get(c.course), "year": c.year,
            "year_name": eng.YEARS.get(c.year), "email": c.email, "ats_score": c.ats_score,
            "has_resume": bool(c.resume_path), "role": c.role, "teaching": c.teaching}


# ------------------------------------------------------------------ meta
@router.get("/health")
def health() -> dict:
    s = get_settings()
    return {"status": "ok", "llm_enabled": llm.is_enabled(), "llm_provider": s.llm_provider,
            "whisper_model": s.whisper_model}


@router.get("/meta")
def meta() -> dict:
    return {"courses": eng.COURSES, "years": eng.YEARS, "levels": eng.LEVELS,
            "technical_count": eng.TECHNICAL_COUNT, "pass_threshold": eng.PASS_THRESHOLD}


@router.get("/meta/syllabus")
def syllabus_subjects() -> dict:
    """{course: {year: [subject, ...]}} - used by the teacher registration multi-select."""
    return {c: {y: [s["subject"] for s in subs] for y, subs in years.items()}
            for c, years in syllabus.catalog().items()}


# ------------------------------------------------------------------ candidates
class CandidateIn(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    college_name: str = Field(min_length=2, max_length=200)
    roll_no: str = Field(min_length=1, max_length=40)
    course: str
    year: int = Field(ge=1, le=3)
    email: str | None = None


@router.post("/candidates")
def create_candidate(body: CandidateIn, db: DB) -> dict:
    try:
        return candidate_out(svc.register_candidate(db, **body.model_dump()))
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


class TeachingItem(BaseModel):
    course: str
    year: int = Field(ge=1, le=3)
    subject: str


class TeacherIn(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    college_name: str = Field(min_length=2, max_length=200)
    employee_id: str | None = Field(default=None, max_length=40)
    email: str | None = None
    teaching: list[TeachingItem] = Field(min_length=1)


@router.post("/teachers")
def create_teacher(body: TeacherIn, db: DB) -> dict:
    """IT-department faculty: choose every course/year/subject they have taught."""
    data = body.model_dump()
    try:
        return candidate_out(svc.register_teacher(db, **data))
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@router.get("/candidates")
def list_candidates(db: DB, course: str | None = None, role: str | None = None) -> list[dict]:
    q = select(Candidate).order_by(Candidate.created_at.desc())
    if role:
        q = q.where(Candidate.role == role)
    if course:
        q = q.where(Candidate.course == course)
    out = []
    for c in db.scalars(q).all():
        last = max(c.interviews, key=lambda i: i.started_at, default=None)
        out.append({**candidate_out(c), "interviews": len(c.interviews),
                    "last_score": last.overall_score if last else None,
                    "last_verdict": last.verdict if last else None,
                    "last_interview_id": last.id if last else None})
    return out


@router.get("/candidates/{candidate_id}")
def get_candidate(candidate_id: int, db: DB) -> dict:
    c = _candidate(db, candidate_id)
    return {**candidate_out(c), "resume_analysis": c.resume_analysis,
            "interviews": [{"id": i.id, "status": i.status, "overall_score": i.overall_score,
                            "verdict": i.verdict, "started_at": i.started_at} for i in c.interviews]}


@router.post("/candidates/{candidate_id}/resume")
def upload_resume(candidate_id: int, db: DB, file: Annotated[UploadFile, File()]) -> dict:
    c = _candidate(db, candidate_id)
    path = _save_upload(file, get_settings().uploads_dir / "resumes", {".pdf", ".docx"}, MAX_RESUME_MB)
    try:
        return svc.process_resume(db, c, path)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


# ------------------------------------------------------------------ interviews
class StartIn(BaseModel):
    candidate_id: int


@router.post("/interviews")
def start_interview(body: StartIn, db: DB) -> dict:
    interview = svc.start_interview(db, _candidate(db, body.candidate_id))
    return {"interview_id": interview.id, "question": svc.next_question(db, interview)}


@router.get("/interviews/{interview_id}/question")
def current_question(interview_id: int, db: DB) -> dict:
    return svc.next_question(db, _interview(db, interview_id))


@router.post("/interviews/{interview_id}/answer")
def submit_answer(
    interview_id: int,
    db: DB,
    audio: Annotated[UploadFile | None, File()] = None,
    text: Annotated[str | None, Form()] = None,
    skip: Annotated[bool, Form()] = False,
) -> dict:
    """Answer by voice (audio file), typed text (fallback/testing) or skip ("I don't know")."""
    interview = _interview(db, interview_id)
    if interview.status not in ("in_progress",):
        raise HTTPException(409, f"Interview is {interview.status}")
    speech, media_path, transcript, wav = None, None, "", None
    if skip:
        transcript = ""
    elif audio is not None:
        from backend.app.analyzers import media
        from backend.app.analyzers import speech as speech_an

        folder = get_settings().uploads_dir / f"interview_{interview_id}"
        src = _save_upload(audio, folder, {".webm", ".wav", ".mp3", ".ogg", ".m4a", ".mp4"}, MAX_AUDIO_MB)
        try:
            prepared = media.prepare(src, get_settings().work_dir / src.stem)
        except media.MediaError as e:
            raise HTTPException(422, f"Could not read the recording: {e}") from e
        if prepared.audio_path is None:
            raise HTTPException(422, "The recording has no audio track")
        pending = svc.next_question(db, interview)
        result = speech_an.transcribe(prepared.audio_path, initial_prompt=pending.get("text"))
        speech, media_path, transcript = result.to_dict(), str(src), result.transcript
        wav = prepared.audio_path
    elif text is not None:
        transcript = text
    else:
        raise HTTPException(422, "Send an audio file, text, or skip=true")
    try:
        result = svc.submit_answer(db, interview, transcript, speech=speech, media_path=media_path,
                                   audio_path=wav)
    except ValueError as e:
        raise HTTPException(409, str(e)) from e
    if speech:
        result["speech"] = {k: speech[k] for k in ("words_per_minute", "filler_count", "long_pause_count",
                                                   "fluency_score", "pace_label")}
    result["next"] = svc.next_question(db, interview)
    return result


class ReviewIn(BaseModel):
    want_answers: bool


@router.get("/interviews/{interview_id}/review")
def review(interview_id: int, db: DB) -> dict:
    items = svc.review_items(db, _interview(db, interview_id))
    return {"count": len(items), "items": items,
            "prompt": "Some questions you could not answer correctly. Do you want me to explain the answers?"
            if items else None}


@router.post("/interviews/{interview_id}/review")
def review_decision(interview_id: int, body: ReviewIn, db: DB) -> dict:
    interview = _interview(db, interview_id)
    if body.want_answers:
        return {"message": "Here are the explanations.", "items": svc.explain_parked(db, interview)}
    return {"message": "Thank you for attending the interview. All the best!", "items": []}


@router.post("/interviews/{interview_id}/finish")
def finish(interview_id: int, db: DB) -> dict:
    return svc.finish_interview(db, _interview(db, interview_id))


@router.get("/interviews/{interview_id}/report")
def report(interview_id: int, db: DB) -> dict:
    return svc.report_data(db, _interview(db, interview_id))


# ------------------------------------------------------------------ voice
@router.get("/tts", responses={200: {"content": {"audio/mpeg": {}}}, 204: {"description": "TTS unavailable"}})
def tts(text: Annotated[str, Query(min_length=1, max_length=1500)], voice: str = "female") -> Response:
    from backend.app.analyzers import tts as tts_an

    audio = tts_an.speak(text, tts_an.VOICES.get(voice, tts_an.DEFAULT_VOICE))
    if not audio:
        return Response(status_code=204)
    return Response(content=audio, media_type="audio/mpeg", headers={"Cache-Control": "max-age=86400"})
