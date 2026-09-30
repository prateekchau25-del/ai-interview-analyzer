"""Relational schema. Works unchanged on PostgreSQL (production) and SQLite (local tests)."""
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class JobRole(Base):
    __tablename__ = "job_roles"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(120), unique=True)
    description: Mapped[str] = mapped_column(Text, default="")
    required_skills: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    questions: Mapped[list["Question"]] = relationship(
        back_populates="role", cascade="all, delete-orphan", order_by="Question.order_index"
    )


class Question(Base):
    __tablename__ = "questions"

    id: Mapped[int] = mapped_column(primary_key=True)
    role_id: Mapped[int] = mapped_column(ForeignKey("job_roles.id", ondelete="CASCADE"))
    order_index: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text)
    ideal_answer: Mapped[str] = mapped_column(Text, default="")
    keywords: Mapped[list] = mapped_column(JSON, default=list)
    category: Mapped[str] = mapped_column(String(40), default="technical")  # technical | behavioral | hr
    time_limit_sec: Mapped[int] = mapped_column(Integer, default=120)

    role: Mapped[JobRole] = relationship(back_populates="questions")


class Candidate(Base):
    __tablename__ = "candidates"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    batch: Mapped[str | None] = mapped_column(String(60), nullable=True)
    resume_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    resume_analysis: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    interviews: Mapped[list["Interview"]] = relationship(back_populates="candidate")


class Interview(Base):
    __tablename__ = "interviews"

    id: Mapped[int] = mapped_column(primary_key=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id", ondelete="CASCADE"))
    role_id: Mapped[int] = mapped_column(ForeignKey("job_roles.id"))
    # in_progress -> analyzing -> completed | failed
    status: Mapped[str] = mapped_column(String(20), default="in_progress", index=True)
    overall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    verdict: Mapped[str | None] = mapped_column(String(40), nullable=True)
    summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    report_pdf_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    candidate: Mapped[Candidate] = relationship(back_populates="interviews")
    role: Mapped[JobRole] = relationship()
    answers: Mapped[list["Answer"]] = relationship(
        back_populates="interview", cascade="all, delete-orphan", order_by="Answer.id"
    )


class Answer(Base):
    __tablename__ = "answers"

    id: Mapped[int] = mapped_column(primary_key=True)
    interview_id: Mapped[int] = mapped_column(ForeignKey("interviews.id", ondelete="CASCADE"))
    question_id: Mapped[int | None] = mapped_column(ForeignKey("questions.id"), nullable=True)
    # Snapshot of the question text (supports resume-generated ad-hoc questions too)
    question_text: Mapped[str] = mapped_column(Text)
    media_path: Mapped[str] = mapped_column(String(500))
    # queued -> processing -> completed | failed
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
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
    speech: Mapped[dict] = mapped_column(JSON, default=dict)
    audio: Mapped[dict] = mapped_column(JSON, default=dict)
    vision: Mapped[dict] = mapped_column(JSON, default=dict)
    nlp: Mapped[dict] = mapped_column(JSON, default=dict)
    suggestions: Mapped[list] = mapped_column(JSON, default=list)

    answer: Mapped[Answer] = relationship(back_populates="metrics")
