"""tts_module.py - Text-to-Speech (TTS) Module using gTTS

Converts generated text responses into audio bytes for the Voice Bot.
Supports in-memory synthesis (no temp files) and optional caching.
"""

import io
import hashlib
import logging
from functools import lru_cache
from typing import Optional

from gtts import gTTS
from gtts.tts import gTTSError

from config import TTS_LANGUAGE, TTS_SLOW, TTS_CACHE_SIZE, logger


# ---------------------------------------------------------------------------
# Core synthesis
# ---------------------------------------------------------------------------

def synthesize(text: str, lang: str = TTS_LANGUAGE, slow: bool = TTS_SLOW) -> bytes:
    """Convert *text* to MP3 audio bytes using gTTS.

    Args:
        text: The text to synthesize.
        lang: BCP-47 language tag (default from config).
        slow: If True, slower speech rate.

    Returns:
        Raw MP3 bytes.

    Raises:
        ValueError: If text is empty.
        RuntimeError: If gTTS network call fails.
    """
    if not text or not text.strip():
        raise ValueError("Cannot synthesize empty text.")

    try:
        tts = gTTS(text=text.strip(), lang=lang, slow=slow)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        audio_bytes = buf.read()
        logger.debug("Synthesized %d chars -> %d bytes audio.", len(text), len(audio_bytes))
        return audio_bytes
    except gTTSError as exc:
        logger.error("gTTS synthesis failed: %s", exc)
        raise RuntimeError(f"TTS synthesis error: {exc}") from exc


# ---------------------------------------------------------------------------
# Cached synthesis (LRU)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=TTS_CACHE_SIZE)
def synthesize_cached(text: str, lang: str = TTS_LANGUAGE, slow: bool = TTS_SLOW) -> bytes:
    """Cached version of synthesize(). Identical calls return cached bytes."""
    return synthesize(text, lang=lang, slow=slow)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def text_to_audio_response(
    text: str,
    use_cache: bool = True,
    lang: str = TTS_LANGUAGE,
    slow: bool = TTS_SLOW,
) -> bytes:
    """High-level helper: synthesize text and return MP3 bytes.

    Args:
        text: Response text.
        use_cache: Whether to use LRU cache.
        lang: Language code.
        slow: Slow speech flag.

    Returns:
        MP3 audio bytes.
    """
    if use_cache:
        return synthesize_cached(text, lang=lang, slow=slow)
    return synthesize(text, lang=lang, slow=slow)


def audio_hash(audio_bytes: bytes) -> str:
    """Return SHA-256 hex digest of audio bytes for deduplication/logging."""
    return hashlib.sha256(audio_bytes).hexdigest()


# ---------------------------------------------------------------------------
# TTSModule class (OOP interface for FastAPI dependency injection)
# ---------------------------------------------------------------------------

class TTSModule:
    """Stateless TTS service wrapper."""

    def __init__(
        self,
        lang: str = TTS_LANGUAGE,
        slow: bool = TTS_SLOW,
        use_cache: bool = True,
    ):
        self.lang = lang
        self.slow = slow
        self.use_cache = use_cache
        logger.info(
            "TTSModule initialized (lang=%s, slow=%s, cache=%s).",
            lang,
            slow,
            use_cache,
        )

    def speak(self, text: str) -> bytes:
        """Synthesize text and return MP3 bytes."""
        return text_to_audio_response(
            text, use_cache=self.use_cache, lang=self.lang, slow=self.slow
        )

    def get_audio_info(self, audio_bytes: bytes) -> dict:
        """Return metadata about synthesized audio."""
        return {
            "size_bytes": len(audio_bytes),
            "sha256": audio_hash(audio_bytes),
            "format": "mp3",
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_tts_instance: Optional[TTSModule] = None


def get_tts() -> TTSModule:
    """Return (and lazily create) the global TTS singleton."""
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = TTSModule()
    return _tts_instance
