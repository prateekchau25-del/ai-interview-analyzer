"""course-based adaptive interviews: candidate college/roll/course/year, syllabus question bank, live grading

Revision ID: 0002_course_adaptive
Revises: 0001_initial
Create Date: 2026-10-01
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002_course_adaptive"
down_revision = "0001_initial"
branch_labels = None
depends_on = None

JSONType = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    # ---- candidates ----
    op.add_column("candidates", sa.Column("college_name", sa.String(200), nullable=False, server_default=""))
    op.add_column("candidates", sa.Column("roll_no", sa.String(40), nullable=False, server_default=""))
    op.add_column("candidates", sa.Column("course", sa.String(20), nullable=False, server_default=""))
    op.add_column("candidates", sa.Column("year", sa.Integer, nullable=False, server_default="1"))
    op.add_column("candidates", sa.Column("resume_text", sa.Text))
    op.add_column("candidates", sa.Column("ats_score", sa.Float))
    op.drop_index("ix_candidates_email", table_name="candidates")
    op.alter_column("candidates", "email", existing_type=sa.String(200), nullable=True)
    op.create_index("ix_candidates_email", "candidates", ["email"])
    op.create_unique_constraint("uq_candidate_identity", "candidates", ["college_name", "roll_no", "course"])

    # ---- questions ----
    op.alter_column("questions", "role_id", existing_type=sa.Integer, nullable=True)
    op.add_column("questions", sa.Column("kind", sa.String(20), nullable=False, server_default="technical"))
    op.add_column("questions", sa.Column("course", sa.String(20)))
    op.add_column("questions", sa.Column("year", sa.Integer))
    op.add_column("questions", sa.Column("semester", sa.Integer))
    op.add_column("questions", sa.Column("subject", sa.String(160)))
    op.add_column("questions", sa.Column("topic", sa.String(300)))
    op.add_column("questions", sa.Column("level", sa.Integer, nullable=False, server_default="0"))
    op.add_column("questions", sa.Column("source", sa.String(20), nullable=False, server_default="bank"))
    op.add_column("questions", sa.Column(
        "candidate_id", sa.Integer, sa.ForeignKey("candidates.id", ondelete="CASCADE")))
    op.add_column("questions", sa.Column("times_asked", sa.Integer, nullable=False, server_default="0"))
    op.add_column("questions", sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()))
    op.create_index("ix_questions_candidate_id", "questions", ["candidate_id"])
    op.create_index("ix_questions_pick", "questions", ["kind", "course", "year", "level"])
    # Old behavioral/HR seed questions are superseded by data/soft_skills.json.
    op.execute("UPDATE questions SET source = 'seed'")
    op.execute("UPDATE questions SET kind = 'soft', role_id = NULL, is_active = false "
               "WHERE category IN ('behavioral', 'hr')")  # replaced by data/soft_skills.json

    # ---- interviews ----
    op.alter_column("interviews", "role_id", existing_type=sa.Integer, nullable=True)
    op.add_column("interviews", sa.Column("course", sa.String(20)))
    op.add_column("interviews", sa.Column("year", sa.Integer))
    op.add_column("interviews", sa.Column("state", JSONType))
    op.add_column("interviews", sa.Column("technical_score", sa.Float))
    op.add_column("interviews", sa.Column("soft_score", sa.Float))

    # ---- answers ----
    op.alter_column("answers", "media_path", existing_type=sa.String(500), nullable=True)
    op.add_column("answers", sa.Column("kind", sa.String(20), nullable=False, server_default="technical"))
    op.add_column("answers", sa.Column("level", sa.Integer, nullable=False, server_default="0"))
    op.add_column("answers", sa.Column("order_index", sa.Integer, nullable=False, server_default="0"))
    op.add_column("answers", sa.Column("score", sa.Float))
    op.add_column("answers", sa.Column("verdict", sa.String(20)))
    op.add_column("answers", sa.Column("skipped", sa.Boolean, nullable=False, server_default=sa.false()))
    op.add_column("answers", sa.Column("grade", JSONType))
    op.add_column("answers", sa.Column("explanation", sa.Text))


def downgrade() -> None:
    for col in ("explanation", "grade", "skipped", "verdict", "score", "order_index", "level", "kind"):
        op.drop_column("answers", col)
    for col in ("soft_score", "technical_score", "state", "year", "course"):
        op.drop_column("interviews", col)
    op.drop_index("ix_questions_pick", table_name="questions")
    op.drop_index("ix_questions_candidate_id", table_name="questions")
    for col in ("is_active", "times_asked", "candidate_id", "source", "level", "topic", "subject",
                "semester", "year", "course", "kind"):
        op.drop_column("questions", col)
    op.drop_constraint("uq_candidate_identity", "candidates", type_="unique")
    op.drop_index("ix_candidates_email", table_name="candidates")
    op.create_index("ix_candidates_email", "candidates", ["email"], unique=True)
    for col in ("ats_score", "resume_text", "year", "course", "roll_no", "college_name"):
        op.drop_column("candidates", col)
