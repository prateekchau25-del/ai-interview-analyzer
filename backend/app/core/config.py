"""Central configuration. Every value can be overridden from the environment or `.env`."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    app_name: str = "AI Interview Analyzer"
    # Postgres example: postgresql+psycopg://postgres:password@localhost:5432/interview_ai
    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'data' / 'dev.db').as_posix()}"

    data_dir: Path = PROJECT_ROOT / "data"
    ml_models_dir: Path = PROJECT_ROOT / "ml" / "models"

    # Run the analysis worker as a thread inside the API process (easy local dev).
    # Set to false and run `python -m backend.app.worker` separately in production.
    inline_worker: bool = True
    worker_poll_seconds: float = 2.0

    # --- Speech ---
    whisper_model: str = "small"  # tiny | base | small | medium | large-v3
    whisper_compute_type: str = "int8"
    whisper_language: str | None = None  # None = auto detect (handles Hinglish)

    # --- Vision ---
    vision_sample_fps: float = 5.0

    # --- NLP ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- LLM (any OpenAI-compatible endpoint) ---
    # provider: groq | gemini | ollama | none
    llm_provider: str = "groq"
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_timeout_seconds: float = 60.0

    api_url: str = "http://localhost:8000"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def work_dir(self) -> Path:
        return self.data_dir / "work"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    for d in (s.data_dir, s.uploads_dir, s.reports_dir, s.work_dir, s.ml_models_dir):
        d.mkdir(parents=True, exist_ok=True)
    return s
