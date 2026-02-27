"""main.py - FastAPI Application for AI-Powered Voice Bot

Exposes four primary REST endpoints:
  POST /transcribe          - Audio file -> transcript + WER
  POST /predict-intent      - Text -> intent + confidence
  POST /generate-response   - Intent -> response text
  POST /synthesize          - Text -> MP3 audio

Also exposes a convenience pipeline endpoint:
  POST /pipeline            - Audio -> transcript -> intent -> response -> audio
"""

from __future__ import annotations

import io
import base64
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field

from config import APP_TITLE, APP_VERSION, APP_DESCRIPTION, logger
from asr_module import get_asr
from intent_classifier import get_classifier
from response_generator import get_generator
from tts_module import get_tts

# ---------------------------------------------------------------------------
# App bootstrap
# ---------------------------------------------------------------------------

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class TranscribeResponse(BaseModel):
    transcript: str
    wer: Optional[float] = Field(None, description="Word Error Rate vs reference (if provided)")
    language: str = "en"


class IntentRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Transcribed text to classify")


class IntentResponse(BaseModel):
    intent: str
    confidence: float


class GenerateRequest(BaseModel):
    intent: str = Field(..., description="Intent tag from classifier")
    confidence: float = Field(1.0, ge=0.0, le=1.0)


class GenerateResponse(BaseModel):
    intent: str
    confidence: float
    response: str
    context: Optional[str] = None


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to convert to speech")
    lang: str = Field("en", description="Language code")
    slow: bool = Field(False, description="Slower speech rate")


class PipelineRequest(BaseModel):
    reference_text: Optional[str] = Field(
        None, description="Optional reference transcript for WER calculation"
    )
    confidence_threshold: float = Field(0.35, ge=0.0, le=1.0)


class PipelineResponse(BaseModel):
    transcript: str
    wer: Optional[float]
    intent: str
    confidence: float
    response_text: str
    audio_base64: str = Field(..., description="Base64-encoded MP3 audio")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
async def root():
    """Service health check."""
    return {"status": "ok", "service": APP_TITLE, "version": APP_VERSION}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# POST /transcribe
# ---------------------------------------------------------------------------

@app.post("/transcribe", response_model=TranscribeResponse, tags=["ASR"])
async def transcribe(
    audio: UploadFile = File(..., description="Audio file (wav/mp3/m4a)"),
    reference: Optional[str] = Form(None, description="Optional reference text for WER"),
):
    """Transcribe uploaded audio and optionally compute Word Error Rate."""
    logger.info("POST /transcribe | filename=%s", audio.filename)
    try:
        audio_bytes = await audio.read()
        asr = get_asr()
        transcript = asr.transcribe_bytes(audio_bytes)

        wer = None
        if reference:
            wer = asr.compute_wer(reference, transcript)

        return TranscribeResponse(transcript=transcript, wer=wer)
    except Exception as exc:
        logger.error("Transcription error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# POST /predict-intent
# ---------------------------------------------------------------------------

@app.post("/predict-intent", response_model=IntentResponse, tags=["NLU"])
async def predict_intent(req: IntentRequest):
    """Classify intent from transcribed text using fine-tuned BERT."""
    logger.info("POST /predict-intent | text='%s'", req.text[:80])
    try:
        classifier = get_classifier()
        intent, confidence = classifier.predict(req.text)
        return IntentResponse(intent=intent, confidence=confidence)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("Intent prediction error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Intent prediction failed: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# POST /generate-response
# ---------------------------------------------------------------------------

@app.post("/generate-response", response_model=GenerateResponse, tags=["NLG"])
async def generate_response_endpoint(req: GenerateRequest):
    """Generate a customer support response for a given intent."""
    logger.info(
        "POST /generate-response | intent=%s confidence=%.2f",
        req.intent,
        req.confidence,
    )
    try:
        generator = get_generator()
        meta = generator.generate_with_metadata(req.intent, req.confidence)
        return GenerateResponse(**meta)
    except Exception as exc:
        logger.error("Response generation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Response generation failed: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# POST /synthesize
# ---------------------------------------------------------------------------

@app.post("/synthesize", tags=["TTS"])
async def synthesize(req: SynthesizeRequest):
    """Convert text to MP3 speech. Returns audio/mpeg bytes."""
    logger.info("POST /synthesize | text='%s'", req.text[:80])
    try:
        tts = get_tts()
        audio_bytes = tts.speak(req.text)
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=response.mp3"},
        )
    except Exception as exc:
        logger.error("TTS synthesis error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TTS synthesis failed: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# POST /pipeline  (end-to-end)
# ---------------------------------------------------------------------------

@app.post("/pipeline", response_model=PipelineResponse, tags=["Pipeline"])
async def pipeline(
    audio: UploadFile = File(..., description="Audio file for full pipeline"),
    reference: Optional[str] = Form(None),
    confidence_threshold: float = Form(0.35),
):
    """Full pipeline: audio -> transcript -> intent -> response -> audio."""
    logger.info("POST /pipeline | filename=%s", audio.filename)
    try:
        audio_bytes = await audio.read()

        # Step 1: ASR
        asr = get_asr()
        transcript = asr.transcribe_bytes(audio_bytes)
        wer = asr.compute_wer(reference, transcript) if reference else None

        # Step 2: Intent classification
        classifier = get_classifier()
        intent, confidence = classifier.predict(transcript)

        # Step 3: Response generation
        generator = get_generator()
        meta = generator.generate_with_metadata(intent, confidence, confidence_threshold)
        response_text = meta["response"]

        # Step 4: TTS
        tts = get_tts()
        response_audio = tts.speak(response_text)
        audio_b64 = base64.b64encode(response_audio).decode("utf-8")

        return PipelineResponse(
            transcript=transcript,
            wer=wer,
            intent=intent,
            confidence=confidence,
            response_text=response_text,
            audio_base64=audio_b64,
        )
    except Exception as exc:
        logger.error("Pipeline error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline failed: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
