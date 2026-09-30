"""initial schema: roles, questions, candidates, interviews, answers, answer_metrics

Revision ID: 0001_initial
Revises:
Create Date: 2026-09-30
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

JSONType = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "job_roles",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("title", sa.String(120), nullable=False, unique=True),
        sa.Column("description", sa.Text, nullable=False, server_default=""),
        sa.Column("required_skills", JSONType, nullable=False),
        sa.Column("created_at", TS, nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "questions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("role_id", sa.Integer,
                  sa.ForeignKey("job_roles.id", ondelete="CASCADE"), nullable=False),
        sa.Column("order_index", sa.Integer, nullable=False, server_default="0"),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("ideal_answer", sa.Text, nullable=False, server_default=""),
        sa.Column("keywords", JSONType, nullable=False),
        sa.Column("category", sa.String(40), nullable=False, server_default="technical"),
        sa.Column("time_limit_sec", sa.Integer, nullable=False, server_default="120"),
    )
    op.create_index("ix_questions_role_id", "questions", ["role_id"])

    op.create_table(
        "candidates",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("email", sa.String(200), nullable=False),
        sa.Column("batch", sa.String(60)),
        sa.Column("resume_path", sa.String(500)),
        sa.Column("resume_analysis", JSONType),
        sa.Column("created_at", TS, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_candidates_email", "candidates", ["email"], unique=True)

    op.create_table(
        "interviews",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("candidate_id", sa.Integer,
                  sa.ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role_id", sa.Integer, sa.ForeignKey("job_roles.id"), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="in_progress"),
        sa.Column("overall_score", sa.Float),
        sa.Column("verdict", sa.String(40)),
        sa.Column("summary", JSONType),
        sa.Column("report_pdf_path", sa.String(500)),
        sa.Column("started_at", TS, nullable=False, server_default=sa.func.now()),
        sa.Column("finished_at", TS),
    )
    op.create_index("ix_interviews_status", "interviews", ["status"])
    op.create_index("ix_interviews_candidate_id", "interviews", ["candidate_id"])

    op.create_table(
        "answers",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("interview_id", sa.Integer,
                  sa.ForeignKey("interviews.id", ondelete="CASCADE"), nullable=False),
        sa.Column("question_id", sa.Integer, sa.ForeignKey("questions.id")),
        sa.Column("question_text", sa.Text, nullable=False),
        sa.Column("media_path", sa.String(500), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="queued"),
        sa.Column("error", sa.Text),
        sa.Column("transcript", sa.Text),
        sa.Column("duration_sec", sa.Float),
        sa.Column("created_at", TS, nullable=False, server_default=sa.func.now()),
        sa.Column("processed_at", TS),
    )
    op.create_index("ix_answers_status", "answers", ["status"])
    op.create_index("ix_answers_interview_id", "answers", ["interview_id"])

    op.create_table(
        "answer_metrics",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("answer_id", sa.Integer,
                  sa.ForeignKey("answers.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("content_score", sa.Float, nullable=False, server_default="0"),
        sa.Column("communication_score", sa.Float, nullable=False, server_default="0"),
        sa.Column("voice_score", sa.Float, nullable=False, server_default="0"),
        sa.Column("body_language_score", sa.Float, nullable=False, server_default="0"),
        sa.Column("overall_score", sa.Float, nullable=False, server_default="0"),
        sa.Column("words_per_minute", sa.Float),
        sa.Column("filler_count", sa.Integer),
        sa.Column("long_pause_count", sa.Integer),
        sa.Column("eye_contact_pct", sa.Float),
        sa.Column("smile_pct", sa.Float),
        sa.Column("relevance_score", sa.Float),
        sa.Column("voice_emotion", sa.String(30)),
        sa.Column("dominant_face_emotion", sa.String(30)),
        sa.Column("speech", JSONType, nullable=False),
        sa.Column("audio", JSONType, nullable=False),
        sa.Column("vision", JSONType, nullable=False),
        sa.Column("nlp", JSONType, nullable=False),
        sa.Column("suggestions", JSONType, nullable=False),
    )


def downgrade() -> None:
    for table in ("answer_metrics", "answers", "interviews", "candidates", "questions", "job_roles"):
        op.drop_table(table)
