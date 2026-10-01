"""Tests never call a real LLM or need a database, even if the developer's .env configures them."""
import os

os.environ["LLM_PROVIDER"] = "none"
os.environ["LLM_API_KEY"] = ""

from backend.app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()
