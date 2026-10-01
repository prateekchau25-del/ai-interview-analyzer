"""Test environment: no real LLM, no real database, no writes into the project's data/ folder.

The API tests use an in-memory SQLite database that lives only in RAM for the test run.
"""
import os
import tempfile

os.environ["LLM_PROVIDER"] = "none"
os.environ["LLM_API_KEY"] = ""
os.environ["DATABASE_URL"] = "sqlite://"
os.environ["AUTO_MIGRATE"] = "false"
os.environ["WARM_UP_MODELS"] = "false"
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="aia_test_")

from backend.app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()
