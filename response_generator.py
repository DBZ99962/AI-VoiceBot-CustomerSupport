"""response_generator.py - Rule-Based Response Generator

Matches an intent tag to a pre-defined response template from intents.json.
Provides fallback handling and context-aware follow-up messages.
"""

import json
import random
from typing import Dict, List, Optional, Tuple

from config import INTENTS_JSON_PATH, DEFAULT_FALLBACK_RESPONSE, logger


# ---------------------------------------------------------------------------
# Response Store
# ---------------------------------------------------------------------------

class ResponseStore:
    """Load and index intent responses from intents.json."""

    def __init__(self, intents_path: str = INTENTS_JSON_PATH):
        self._responses: Dict[str, List[str]] = {}
        self._context_map: Dict[str, str] = {}
        self._load(intents_path)

    def _load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for intent in data["intents"]:
            tag = intent["tag"]
            self._responses[tag] = intent.get("responses", [])
            if "context_set" in intent:
                self._context_map[tag] = intent["context_set"]
        logger.info(
            "ResponseStore loaded %d intent tags.", len(self._responses)
        )

    def get_response(self, tag: str) -> str:
        """Return a random response for *tag*, or fallback text."""
        responses = self._responses.get(tag)
        if not responses:
            logger.warning("No responses for intent '%s'. Using fallback.", tag)
            return DEFAULT_FALLBACK_RESPONSE
        return random.choice(responses)

    def get_context(self, tag: str) -> Optional[str]:
        """Return context key set by this intent (if any)."""
        return self._context_map.get(tag)

    @property
    def known_tags(self) -> List[str]:
        return list(self._responses.keys())


# ---------------------------------------------------------------------------
# Response Generator
# ---------------------------------------------------------------------------

class ResponseGenerator:
    """Generate a textual response from an intent tag and optional context."""

    def __init__(self, intents_path: str = INTENTS_JSON_PATH):
        self.store = ResponseStore(intents_path)
        self._context: Optional[str] = None  # last context key

    # ------------------------------------------------------------------
    def generate(
        self, intent: str, confidence: float = 1.0, threshold: float = 0.35
    ) -> str:
        """Return a response string.

        Args:
            intent: Predicted intent tag.
            confidence: Model confidence (0-1).
            threshold: Minimum confidence to trust intent.

        Returns:
            Response string.
        """
        if confidence < threshold:
            logger.info(
                "Low confidence (%.2f) for intent '%s'. Returning clarification.",
                confidence,
                intent,
            )
            return (
                "I'm not quite sure I understood that. Could you please rephrase "
                "your question?"
            )

        response = self.store.get_response(intent)
        new_context = self.store.get_context(intent)
        if new_context:
            self._context = new_context
            logger.debug("Context set to '%s'.", new_context)

        return response

    # ------------------------------------------------------------------
    def generate_with_metadata(
        self, intent: str, confidence: float = 1.0, threshold: float = 0.35
    ) -> Dict:
        """Return response dict with text, intent, confidence, context."""
        text = self.generate(intent, confidence, threshold)
        return {
            "intent": intent,
            "confidence": round(confidence, 4),
            "response": text,
            "context": self._context,
        }

    # ------------------------------------------------------------------
    def reset_context(self) -> None:
        """Clear stored context."""
        self._context = None

    # ------------------------------------------------------------------
    @property
    def current_context(self) -> Optional[str]:
        return self._context


# ---------------------------------------------------------------------------
# Pipeline helper
# ---------------------------------------------------------------------------

def generate_response(
    intent: str,
    confidence: float,
    generator: Optional[ResponseGenerator] = None,
) -> Tuple[str, Dict]:
    """One-shot helper used by FastAPI endpoints.

    Returns:
        (response_text, metadata_dict)
    """
    gen = generator or get_generator()
    meta = gen.generate_with_metadata(intent, confidence)
    return meta["response"], meta


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_generator_instance: Optional[ResponseGenerator] = None


def get_generator() -> ResponseGenerator:
    """Return (and lazily create) the global ResponseGenerator singleton."""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = ResponseGenerator()
    return _generator_instance
