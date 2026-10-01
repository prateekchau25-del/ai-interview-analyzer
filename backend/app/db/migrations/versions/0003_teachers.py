"""teacher interviews: candidate role + subjects taught, question audience

Revision ID: 0003_teachers
Revises: 0002_course_adaptive
Create Date: 2026-10-01
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0003_teachers"
down_revision = "0002_course_adaptive"
branch_labels = None
depends_on = None

JSONType = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column("candidates", sa.Column("role", sa.String(20), nullable=False, server_default="student"))
    # [{"course": "BSC_IT", "year": 1, "subject": "Database Management Systems"}, ...] for teachers
    op.add_column("candidates", sa.Column("teaching", JSONType))
    op.create_index("ix_candidates_role", "candidates", ["role"])
    # soft-skill questions: student | teacher
    op.add_column("questions", sa.Column("audience", sa.String(20), nullable=False, server_default="student"))


def downgrade() -> None:
    op.drop_column("questions", "audience")
    op.drop_index("ix_candidates_role", table_name="candidates")
    op.drop_column("candidates", "teaching")
    op.drop_column("candidates", "role")
