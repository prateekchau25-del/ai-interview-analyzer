"""Interviewer voice: text -> MP3 with Microsoft Edge neural voices (free, needs internet).

Audio is cached on disk by (voice, text) hash, so repeated questions play instantly.
If synthesis fails (offline), callers get None and the frontend falls back to the browser's speechSynthesis.
"""
import asyncio
import hashlib
import logging
import re

from backend.app.core.config import get_settings

log = logging.getLogger(__name__)

VOICES = {
    "female": "en-IN-NeerjaNeural",
    "male": "en-IN-PrabhatNeural",
}
DEFAULT_VOICE = VOICES["female"]


def _clean_for_speech(text: str) -> str:
    # Speak symbols the way an interviewer would say them.
    text = re.sub(r"`([^`]*)`", r"\1", text)
    return (text.replace("C++", "C plus plus").replace("C#", "C sharp").replace("&", " and ")
            .replace("e.g.", "for example").replace("i.e.", "that is"))


async def _synthesize(text: str, voice: str, rate: str) -> bytes:
    import edge_tts

    chunks = []
    async for msg in edge_tts.Communicate(text, voice=voice, rate=rate).stream():
        if msg["type"] == "audio":
            chunks.append(msg["data"])
    return b"".join(chunks)


def speak(text: str, voice: str = DEFAULT_VOICE, rate: str = "-5%") -> bytes | None:
    """Return MP3 bytes for `text`, or None if TTS is unavailable."""
    text = _clean_for_speech(text.strip())
    if not text:
        return None
    cache_dir = get_settings().data_dir / "tts_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{voice}|{rate}|{text}".encode()).hexdigest()
    path = cache_dir / f"{key}.mp3"
    if path.exists():
        return path.read_bytes()
    try:
        try:
            asyncio.get_running_loop()
            in_loop = True
        except RuntimeError:
            in_loop = False
        if in_loop:
            # Called from async code: asyncio.run() is not allowed here, use a worker thread.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(1) as ex:
                audio = ex.submit(asyncio.run, _synthesize(text, voice, rate)).result()
        else:
            audio = asyncio.run(_synthesize(text, voice, rate))
    except Exception as e:  # network errors, service changes
        log.warning("TTS failed (%s); frontend will use browser speech", e)
        return None
    if audio:
        path.write_bytes(audio)
    return audio or None
