# asr_module.py - Automatic Speech Recognition using OpenAI Whisper
import os
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Any

import whisper
import soundfile as sf
import numpy as np
from loguru import logger

import config

warnings.filterwarnings("ignore")


class ASRModule:
    """
    Automatic Speech Recognition module using OpenAI Whisper.
    Supports WAV input, handles background noise, and reports WER.
    """

    def __init__(self):
        logger.info(f"Loading Whisper model: {config.ASR_MODEL_SIZE}")
        self.model = whisper.load_model(
            config.ASR_MODEL_SIZE, device=config.ASR_DEVICE
        )
        self.language = config.ASR_LANGUAGE
        logger.info("Whisper model loaded successfully.")

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to WAV audio file.

        Returns:
            dict with keys: text, language, segments, duration_seconds
        """
        audio_path = str(audio_path)

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Validate it is a readable audio file
        try:
            data, sample_rate = sf.read(audio_path)
        except Exception as e:
            raise ValueError(f"Cannot read audio file '{audio_path}': {e}")

        duration = len(data) / sample_rate
        logger.info(f"Transcribing audio: {audio_path} ({duration:.2f}s)")

        # Handle very short or silent audio
        if duration < 0.1:
            logger.warning("Audio too short (<0.1s). Returning empty transcript.")
            return {"text": "", "language": self.language, "segments": [], "duration_seconds": duration}

        result = self.model.transcribe(
            audio_path,
            language=self.language,
            task="transcribe",
            fp16=False,
        )

        transcript = result["text"].strip()
        logger.info(f"Transcript: '{transcript}'")

        return {
            "text": transcript,
            "language": result.get("language", self.language),
            "segments": result.get("segments", []),
            "duration_seconds": round(duration, 2),
        }

    def transcribe_bytes(self, audio_bytes: bytes, suffix: str = ".wav") -> Dict[str, Any]:
        """
        Transcribe audio from raw bytes (e.g., uploaded file).

        Args:
            audio_bytes: Raw audio bytes.
            suffix: File extension (.wav).

        Returns:
            Same as transcribe().
        """
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            result = self.transcribe(tmp_path)
        finally:
            os.unlink(tmp_path)

        return result


def compute_wer(references: list, hypotheses: list) -> float:
    """
    Compute Word Error Rate (WER) given reference and hypothesis transcripts.

    WER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=reference word count.

    Args:
        references: List of reference (ground truth) strings.
        hypotheses: List of hypothesis (model output) strings.

    Returns:
        WER as a float (0.0 = perfect, 1.0 = all wrong).
    """
    try:
        from jiwer import wer
        score = wer(references, hypotheses)
        logger.info(f"WER: {score:.4f} ({score*100:.2f}%)")
        return round(score, 4)
    except ImportError:
        logger.warning("jiwer not installed. Computing WER manually.")
        return _manual_wer(references, hypotheses)


def _manual_wer(references: list, hypotheses: list) -> float:
    """Fallback manual WER computation using dynamic programming."""
    total_edits = 0
    total_words = 0

    for ref, hyp in zip(references, hypotheses):
        r = ref.lower().split()
        h = hyp.lower().split()
        total_words += len(r)

        # DP edit distance
        dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
        for i in range(len(r) + 1):
            dp[i][0] = i
        for j in range(len(h) + 1):
            dp[0][j] = j
        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                if r[i - 1] == h[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        total_edits += dp[len(r)][len(h)]

    return round(total_edits / max(total_words, 1), 4)


# Singleton instance
_asr_instance = None


def get_asr() -> ASRModule:
    """Return (and lazily create) the global ASR singleton."""
    global _asr_instance
    if _asr_instance is None:
        _asr_instance = ASRModule()
    return _asr_instance
