# AI-VoiceBot-CustomerSupport

> **AI-Powered Voice Bot for Customer Support Automation**
> ASR (Whisper) + Intent Classification (BERT) + TTS (gTTS) + FastAPI REST API

---

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Intents (12 Classes)](#intents)
- [Setup & Installation](#setup--installation)
- [Training the Model](#training-the-model)
- [Running the API](#running-the-api)
- [API Endpoints](#api-endpoints)
- [Evaluation Metrics](#evaluation-metrics)
- [Demo](#demo)
- [Author](#author)

---

## Overview

This project implements a production-ready, modular AI Voice Bot pipeline for automating customer support interactions. The system:

1. **Transcribes** incoming customer audio using OpenAI Whisper (ASR)
2. **Classifies** the intent of the transcribed text using a fine-tuned BERT model
3. **Generates** a relevant text response from a curated response template store
4. **Synthesizes** the response back to speech using Google Text-to-Speech (gTTS)
5. **Serves** everything through a FastAPI REST API

---

## Architecture

```
[Customer Audio]
       |
       v
+------------------+
|   ASR Module     |  OpenAI Whisper (base model)
|  asr_module.py   |  -> transcript + WER
+------------------+
       |
       v
+------------------------+
| Intent Classifier      |  Fine-tuned BERT
| intent_classifier.py   |  -> intent label + confidence
+------------------------+
       |
       v
+-------------------------+
| Response Generator      |  Rule-based template matching
| response_generator.py   |  -> response text
+-------------------------+
       |
       v
+------------------+
|   TTS Module     |  gTTS (Google TTS)
|  tts_module.py   |  -> MP3 audio bytes
+------------------+
       |
       v
[Audio Response to Customer]

All modules exposed via FastAPI (main.py)
```

---

## Project Structure

```
AI-VoiceBot-CustomerSupport/
|-- main.py                  # FastAPI application & all endpoints
|-- config.py                # Centralized configuration constants
|-- asr_module.py            # Whisper ASR transcription + WER calculation
|-- intent_classifier.py     # BERT fine-tuning, training & inference
|-- tts_module.py            # gTTS text-to-speech synthesis
|-- response_generator.py    # Intent -> response template matching
|-- train.py                 # CLI training script
|-- evaluate.py              # CLI evaluation script (WER + accuracy)
|-- requirements.txt         # Python dependencies
|-- data/
    |-- intents.json         # 12 intent classes with patterns & responses
```

---

## Intents

The system supports **12 intent classes** covering common customer support scenarios:

| # | Intent Tag | Description |
|---|-----------|-------------|
| 1 | greeting | Initial hello/hi messages |
| 2 | goodbye | Farewell/end of conversation |
| 3 | order_status | Track or check order status |
| 4 | return_policy | Return and refund questions |
| 5 | cancel_order | Cancel an existing order |
| 6 | payment_issue | Payment failures and billing |
| 7 | product_info | Product details and specs |
| 8 | shipping_info | Shipping time and tracking |
| 9 | complaint | Complaints and dissatisfaction |
| 10 | account_issue | Login, password, account help |
| 11 | discount_promo | Coupons, discounts, promotions |
| 12 | human_agent | Request to speak with a human |

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- pip
- (Optional) CUDA GPU for faster BERT training

### Install dependencies

```bash
git clone https://github.com/DBZ99962/AI-VoiceBot-CustomerSupport.git
cd AI-VoiceBot-CustomerSupport
pip install -r requirements.txt
```

### Key dependencies
- `openai-whisper` - Speech recognition
- `transformers` - BERT model fine-tuning
- `torch` - PyTorch deep learning
- `fastapi` + `uvicorn` - REST API server
- `gtts` - Google Text-to-Speech
- `scikit-learn` - Evaluation metrics

---

## Training the Model

```bash
# Train with default config (5 epochs)
python train.py

# Custom training parameters
python train.py --epochs 10 --lr 2e-5 --batch-size 16

# Save training history
python train.py --epochs 10 --output-history results/history.json
```

The fine-tuned model will be saved to `models/intent_classifier/`.

---

## Running the API

```bash
# Start FastAPI server
python main.py

# Or with uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API docs: http://localhost:8000/docs

---

## API Endpoints

### POST `/transcribe`
Transcribe audio file to text using Whisper.

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "audio=@sample.wav" \
  -F "reference=hello how can I help you"
```

Response:
```json
{"transcript": "hello how can I help you", "wer": 0.0, "language": "en"}
```

### POST `/predict-intent`
Classify intent from transcribed text.

```bash
curl -X POST http://localhost:8000/predict-intent \
  -H "Content-Type: application/json" \
  -d '{"text": "I want to track my order"}'
```

Response:
```json
{"intent": "order_status", "confidence": 0.9823}
```

### POST `/generate-response`
Generate a support response for a given intent.

```bash
curl -X POST http://localhost:8000/generate-response \
  -H "Content-Type: application/json" \
  -d '{"intent": "order_status", "confidence": 0.98}'
```

Response:
```json
{"intent": "order_status", "confidence": 0.98, "response": "Your order is being processed...", "context": null}
```

### POST `/synthesize`
Convert text to MP3 audio.

```bash
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Your order will arrive in 3-5 business days."}' \
  --output response.mp3
```

### POST `/pipeline`
Full end-to-end pipeline: audio -> transcript -> intent -> response -> audio.

```bash
curl -X POST http://localhost:8000/pipeline \
  -F "audio=@customer_query.wav"
```

Response: JSON with transcript, WER, intent, confidence, response_text, audio_base64.

---

## Evaluation Metrics

### Intent Classifier (BERT)

```bash
python evaluate.py
```

Expected results on training data after fine-tuning:

| Metric | Score |
|--------|-------|
| Accuracy | ~0.95+ |
| Macro F1 | ~0.94+ |
| Weighted F1 | ~0.95+ |

### ASR Word Error Rate (WER)

```bash
python evaluate.py --asr-test data/asr_test.json
```

WER is computed using the formula:
```
WER = (S + D + I) / N
```
Where S = substitutions, D = deletions, I = insertions, N = total reference words.

Whisper `base` model achieves ~WER 5-15% on clean English audio.

---

## Demo

After starting the API server, visit http://localhost:8000/docs for the interactive Swagger UI.

Example pipeline interaction:
1. Customer says: *"I want to cancel my order"*
2. ASR transcribes: `"I want to cancel my order"`
3. BERT classifies: `cancel_order` (confidence: 0.97)
4. Response generated: `"We're sorry to hear that. Please provide your order ID and we'll process the cancellation within 24 hours."`
5. gTTS synthesizes the response to MP3
6. Customer receives audio response

---

## Author

**Sandeep Kumar**
- GitHub: [DBZ99962](https://github.com/DBZ99962)
- LinkedIn: [sandeep-kumar-21537a128](https://linkedin.com/in/sandeep-kumar-21537a128)
- Email: sandeepk271094@gmail.com
- Location: Bhilai, Chhattisgarh, India

---

*Built as part of ML Internship Task Assessment - AI-Powered Voice Bot for Customer Support Automation*
