"""Minimal client for any OpenAI-compatible chat API (Groq, Gemini, Ollama).

All three expose `/chat/completions` with the same request shape, so one client covers them.
Callers must handle `LLMUnavailable` and fall back to non-LLM logic.
"""
import json
import logging
import re

import httpx

from backend.app.core.config import get_settings

log = logging.getLogger(__name__)

PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.0-flash"),
    "ollama": ("http://localhost:11434/v1", "llama3.2"),
}


class LLMUnavailable(RuntimeError):
    pass


def is_enabled() -> bool:
    s = get_settings()
    if s.llm_provider == "none" or s.llm_provider not in PROVIDERS:
        return False
    return s.llm_provider == "ollama" or bool(s.llm_api_key)


def _endpoint() -> tuple[str, str, dict]:
    s = get_settings()
    if not is_enabled():
        raise LLMUnavailable(f"LLM disabled (provider={s.llm_provider}, key set={bool(s.llm_api_key)})")
    base, model = PROVIDERS[s.llm_provider]
    headers = {"Authorization": f"Bearer {s.llm_api_key}"} if s.llm_api_key else {}
    return (s.llm_base_url or base).rstrip("/"), s.llm_model or model, headers


def chat(messages: list[dict], temperature: float = 0.2, max_tokens: int = 1200,
         json_mode: bool = False) -> str:
    base, model, headers = _endpoint()
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    try:
        r = httpx.post(f"{base}/chat/completions", json=body, headers=headers,
                       timeout=get_settings().llm_timeout_seconds)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except (httpx.HTTPError, KeyError, IndexError) as e:
        raise LLMUnavailable(f"LLM call failed: {e}") from e


def chat_json(system: str, user: str, temperature: float = 0.1, max_tokens: int = 1500) -> dict:
    """Ask for a JSON object and parse it robustly (some models wrap JSON in ``` fences)."""
    messages = [{"role": "system", "content": system + "\nRespond with a single valid JSON object only."},
                {"role": "user", "content": user}]
    try:
        raw = chat(messages, temperature=temperature, max_tokens=max_tokens, json_mode=True)
    except LLMUnavailable:
        raise
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise LLMUnavailable(f"LLM returned no JSON: {raw[:200]}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise LLMUnavailable(f"LLM returned invalid JSON: {e}") from e
