"""FastAPI application.  Run:  uvicorn backend.app.main:app --reload"""
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router
from backend.app.core.config import get_settings
from backend.app.core.logging import setup_logging

log = logging.getLogger(__name__)


def _warm_up_models() -> None:
    """Load Whisper and the embedding model in the background so the first answer is not slow."""
    try:
        from backend.app.analyzers import nlp, speech

        speech.get_model()
        nlp.get_embedder()
        log.info("AI models warmed up")
    except Exception as e:  # never block the API on warm-up problems
        log.warning("Model warm-up failed: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    if settings.auto_migrate:
        from backend.app.db.setup import setup_database

        setup_database()  # create DB if missing, apply migrations, load question bank
    if settings.warm_up_models:
        threading.Thread(target=_warm_up_models, name="warm-up", daemon=True).start()
    log.info("API ready (LLM provider=%s)", settings.llm_provider)
    yield


app = FastAPI(title="AI Interview Analyzer API", version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router)
