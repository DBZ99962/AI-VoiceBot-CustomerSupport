"""evaluate.py - Evaluation Script for Intent Classifier and ASR Module

Computes:
  - Intent classifier accuracy, precision, recall, F1 per class
  - ASR Word Error Rate (WER) on a sample test set

Usage:
    python evaluate.py [--asr-test data/asr_test.json]
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np

from intent_classifier import load_intents, IntentClassifier
from asr_module import ASRModule
from config import INTENTS_JSON_PATH, INTENT_MODEL_DIR, logger


# ---------------------------------------------------------------------------
# Intent Evaluation
# ---------------------------------------------------------------------------

def evaluate_intent_classifier(
    intents_path: str = INTENTS_JSON_PATH,
    model_dir: str = INTENT_MODEL_DIR,
) -> Dict:
    """Evaluate the fine-tuned BERT intent classifier on all available patterns.

    Returns:
        Dict with accuracy and per-class metrics.
    """
    logger.info("Loading test data from %s", intents_path)
    texts, tags = load_intents(intents_path)

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Model directory '{model_dir}' not found. Run train.py first."
        )

    classifier = IntentClassifier(model_dir=model_dir)
    predicted_tags = [classifier.predict(t)[0] for t in texts]

    acc = accuracy_score(tags, predicted_tags)
    report_dict = classification_report(
        tags, predicted_tags, output_dict=True, zero_division=0
    )
    report_str = classification_report(
        tags, predicted_tags, zero_division=0
    )

    print("\n" + "=" * 60)
    print(" INTENT CLASSIFIER EVALUATION REPORT")
    print("=" * 60)
    print(f"Overall Accuracy : {acc:.4f} ({acc * 100:.2f}%)")
    print("\nPer-class Metrics:")
    print(report_str)

    return {"accuracy": acc, "report": report_dict}


# ---------------------------------------------------------------------------
# ASR WER Evaluation
# ---------------------------------------------------------------------------

def evaluate_asr(
    test_file: Optional[str] = None,
) -> Optional[Dict]:
    """Evaluate ASR Word Error Rate on a JSON test set.

    The test file must be a JSON array of objects:
    [
      {"audio": "path/to/audio.wav", "reference": "expected transcript"},
      ...
    ]

    Args:
        test_file: Path to JSON test file. Skipped if not provided.

    Returns:
        Dict with mean/min/max WER or None if skipped.
    """
    if not test_file or not Path(test_file).exists():
        logger.warning("ASR test file not provided or not found. Skipping ASR eval.")
        return None

    with open(test_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    if not test_cases:
        logger.warning("ASR test file is empty.")
        return None

    asr = ASRModule()
    wer_scores = []

    print("\n" + "=" * 60)
    print(" ASR WORD ERROR RATE EVALUATION")
    print("=" * 60)
    print(f"{'#':>4}  {'WER':>6}  Reference")
    print("-" * 60)

    for idx, case in enumerate(test_cases, 1):
        audio_path = case["audio"]
        reference = case["reference"]
        try:
            transcript = asr.transcribe_file(audio_path)
            wer = asr.compute_wer(reference, transcript)
            wer_scores.append(wer)
            print(f"{idx:>4}  {wer:>6.4f}  {reference[:50]}")
        except Exception as exc:
            logger.error("Error processing '%s': %s", audio_path, exc)

    if not wer_scores:
        return None

    mean_wer = float(np.mean(wer_scores))
    min_wer = float(np.min(wer_scores))
    max_wer = float(np.max(wer_scores))

    print("-" * 60)
    print(f"Mean WER : {mean_wer:.4f}")
    print(f"Min  WER : {min_wer:.4f}")
    print(f"Max  WER : {max_wer:.4f}")
    print("=" * 60)

    return {"mean_wer": mean_wer, "min_wer": min_wer, "max_wer": max_wer}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate intent classifier and ASR module."
    )
    parser.add_argument(
        "--asr-test",
        type=str,
        default=None,
        help="Path to ASR test JSON file (optional).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save evaluation results to JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = {}

    # --- Intent evaluation ---
    try:
        intent_results = evaluate_intent_classifier()
        results["intent"] = intent_results
    except FileNotFoundError as exc:
        logger.error(str(exc))
        print(f"[ERROR] {exc}")

    # --- ASR evaluation ---
    asr_results = evaluate_asr(args.asr_test)
    if asr_results:
        results["asr"] = asr_results

    # --- Save results ---
    if args.output and results:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nEvaluation results saved to {out_path}")

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
