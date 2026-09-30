"""One-command database setup.

    python -m backend.app.db.setup

1. Creates the database named in DATABASE_URL if it does not exist yet.
2. Applies all Alembic migrations (tables, indexes, constraints).
3. Inserts default job roles and questions (only if the tables are empty).

Safe to run any number of times. The API also runs steps 2-3 automatically on startup.
"""
import logging

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from backend.app.core.config import PROJECT_ROOT, get_settings

log = logging.getLogger(__name__)


def ensure_database_exists(database_url: str) -> bool:
    """Create the PostgreSQL database if missing. Returns True if it was created."""
    url = make_url(database_url)
    if not url.drivername.startswith("postgresql"):
        return False
    admin_engine = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin_engine.connect() as conn:
        exists = conn.scalar(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": url.database})
        if not exists:
            conn.execute(text(f'CREATE DATABASE "{url.database}"'))
    admin_engine.dispose()
    return not exists


def run_migrations() -> None:
    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "backend/app/db/migrations"))
    command.upgrade(cfg, "head")


def seed() -> None:
    from backend.app.db.seed import seed_defaults
    from backend.app.db.session import session_scope

    with session_scope() as db:
        seed_defaults(db)


def setup_database(create_if_missing: bool = True) -> None:
    settings = get_settings()
    if not settings.database_url:
        raise SystemExit("DATABASE_URL is not set. Copy .env.example to .env and fill it in.")
    if create_if_missing and ensure_database_exists(settings.database_url):
        log.info("Created database %s", make_url(settings.database_url).database)
    run_migrations()
    seed()


if __name__ == "__main__":
    from backend.app.core.logging import setup_logging

    setup_logging()
    setup_database()

    from sqlalchemy import func, select

    from backend.app.db.models import JobRole, Question
    from backend.app.db.session import session_scope

    with session_scope() as db:
        roles = db.scalar(select(func.count(JobRole.id)))
        questions = db.scalar(select(func.count(Question.id)))
    print(f"\nDatabase ready: {roles} job roles, {questions} questions.")
