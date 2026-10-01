"""Relational schema. Works unchanged on PostgreSQL (production) and SQLite (local tests)."""
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


# JSONB on PostgreSQL (indexable, faster), plain JSON elsewhere (tests)
JSONType = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


class JobRole(Base):
    __tablename__ = "job_roles"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(120), unique=True)
    description: Mapped[str] = mapped_column(Text, default="")
    required_skills: Mapped[list] = mapped_column(JSONType, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    questions: Mapped[list["Question"]] = relationship(
        back_populates="role", cascade="all, delete-orphan", order_by="Question.order_index"
    )


class Question(Base):
    """Question bank. Technical questions come from the syllabus (course/year/subject/topic/level);
    soft-skill questions are generic; resume questions belong to one candidate."""
    __tablename__ = "questions"
    __table_args__ = (Index("ix_questions_pick", "kind", "course", "year", "level"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    role_id: Mapped[int | None] = mapped_column(
        ForeignKey("job_roles.id", ondelete="CASCADE"), index=True, nullable=True
    )
    order_index: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text)
    ideal_answer: Mapped[str] = mapped_column(Text, default="")
    keywords: Mapped[list] = mapped_column(JSONType, default=list)
    category: Mapped[str] = mapped_column(String(40), default="technical")  # legacy role questions
    time_limit_sec: Mapped[int] = mapped_column(Integer, default=120)

    kind: Mapped[str] = mapped_column(String(20), default="technical")  # technical | soft | resume
    course: Mapped[str | None] = mapped_column(String(20), nullable=True)  # BSC_IT | BSC_DS | BSC_AI | BCA
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 1 | 2 | 3
    semester: Mapped[int | None] = mapped_column(Integer, nullable=True)
    subject: Mapped[str | None] = mapped_column(String(160), nullable=True)
    topic: Mapped[str | None] = mapped_column(String(300), nullable=True)
    level: Mapped[int] = mapped_column(Integer, default=0)  # 1 Basic .. 6 Professional, 0 = n/a
    source: Mapped[str] = mapped_column(String(20), default="bank")  # bank | llm | resume | seed
    candidate_id: Mapped[int | None] = mapped_column(
        ForeignKey("candidates.id", ondelete="CASCADE"), nullable=True, index=True
    )
    times_asked: Mapped[int] = mapped_column(Integer, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    role: Mapped[JobRole | None] = relationship(back_populates="questions")


class Candidate(Base):
    __tablename__ = "candidates"
    __table_args__ = (UniqueConstraint("college_name", "roll_no", "course", name="uq_candidate_identity"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    college_name: Mapped[str] = mapped_column(String(200), default="")
    roll_no: Mapped[str] = mapped_column(String(40), default="")
    course: Mapped[str] = mapped_column(String(20), default="")  # BSC_IT | BSC_DS | BSC_AI | BCA
    year: Mapped[int] = mapped_column(Integer, default=1)
    email: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)
    batch: Mapped[str | None] = mapped_column(String(60), nullable=True)
    resume_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    resume_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    resume_analysis: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    ats_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    interviews: Mapped[list["Interview"]] = relationship(back_populates="candidate")


class Interview(Base):
    __tablename__ = "interviews"

    id: Mapped[int] = mapped_column(primary_key=True)
    candidate_id: Mapped[int] = mapped_column(
        ForeignKey("candidates.id", ondelete="CASCADE"), index=True
    )
    role_id: Mapped[int | None] = mapped_column(ForeignKey("job_roles.id"), nullable=True)
    course: Mapped[str | None] = mapped_column(String(20), nullable=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # in_progress -> review -> analyzing -> completed | failed
    status: Mapped[str] = mapped_column(String(20), default="in_progress", index=True)
    state: Mapped[dict | None] = mapped_column(JSONType, nullable=True)  # InterviewState (engine)
    overall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    technical_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    soft_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    verdict: Mapped[str | None] = mapped_column(String(40), nullable=True)
    summary: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    report_pdf_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    candidate: Mapped[Candidate] = relationship(back_populates="interviews")
    role: Mapped[JobRole | None] = relationship()
    answers: Mapped[list["Answer"]] = relationship(
        back_populates="interview", cascade="all, delete-orphan", order_by="Answer.id"
    )


class Answer(Base):
    __tablename__ = "answers"

    id: Mapped[int] = mapped_column(primary_key=True)
    interview_id: Mapped[int] = mapped_column(
        ForeignKey("interviews.id", ondelete="CASCADE"), index=True
    )
    question_id: Mapped[int | None] = mapped_column(ForeignKey("questions.id"), nullable=True)
    # Snapshot of the question text (supports resume-generated ad-hoc questions too)
    question_text: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String(20), default="technical")
    level: Mapped[int] = mapped_column(Integer, default=0)
    order_index: Mapped[int] = mapped_column(Integer, default=0)
    media_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    # Background media analysis: queued -> processing -> completed | failed
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Live grading (done immediately so the engine can pick the next level)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)  # 0-100 correctness
    verdict: Mapped[str | None] = mapped_column(String(20), nullable=True)
    skipped: Mapped[bool] = mapped_column(Boolean, default=False)
    grade: Mapped[dict | None] = mapped_column(JSONType, nullable=True)  # feedback, missing points, method
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)  # end-of-interview teaching
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    interview: Mapped[Interview] = relationship(back_populates="answers")
    question: Mapped[Question | None] = relationship()
    metrics: Mapped["AnswerMetrics | None"] = relationship(
        back_populates="answer", cascade="all, delete-orphan", uselist=False
    )


class AnswerMetrics(Base):
    __tablename__ = "answer_metrics"

    id: Mapped[int] = mapped_column(primary_key=True)
    answer_id: Mapped[int] = mapped_column(ForeignKey("answers.id", ondelete="CASCADE"), unique=True)

    # Component scores (0-100)
    content_score: Mapped[float] = mapped_column(Float, default=0)
    communication_score: Mapped[float] = mapped_column(Float, default=0)
    voice_score: Mapped[float] = mapped_column(Float, default=0)
    body_language_score: Mapped[float] = mapped_column(Float, default=0)
    overall_score: Mapped[float] = mapped_column(Float, default=0)

    # Headline metrics (queryable columns for cohort analytics)
    words_per_minute: Mapped[float | None] = mapped_column(Float, nullable=True)
    filler_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    long_pause_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    eye_contact_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    smile_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    relevance_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    voice_emotion: Mapped[str | None] = mapped_column(String(30), nullable=True)
    dominant_face_emotion: Mapped[str | None] = mapped_column(String(30), nullable=True)

    # Full detail from each pipeline (timelines, breakdowns, LLM feedback)
    speech: Mapped[dict] = mapped_column(JSONType, default=dict)
    audio: Mapped[dict] = mapped_column(JSONType, default=dict)
    vision: Mapped[dict] = mapped_column(JSONType, default=dict)
    nlp: Mapped[dict] = mapped_column(JSONType, default=dict)
    suggestions: Mapped[list] = mapped_column(JSONType, default=list)

    answer: Mapped[Answer] = relationship(back_populates="metrics")
