"""train.py - Training Script for BERT Intent Classifier

Run this script to fine-tune BERT on the intents defined in data/intents.json.

Usage:
    python train.py [--epochs N] [--lr LR] [--batch-size B]
"""

import argparse
import json
import sys
from pathlib import Path

from intent_classifier import IntentClassifierTrainer
from config import EPOCHS, LEARNING_RATE, BATCH_SIZE, logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT for customer support intent classification."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--output-history",
        type=str,
        default=None,
        help="Optional path to save training history as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info(
        "Starting training: epochs=%d lr=%.2e batch_size=%d",
        args.epochs,
        args.lr,
        args.batch_size,
    )

    trainer = IntentClassifierTrainer(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )

    history = trainer.run()

    # Print summary table
    print("\nTraining Summary")
    print("-" * 70)
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}")
    print("-" * 70)
    for h in history:
        print(
            f"{h['epoch']:>6}  {h['train_loss']:>10.4f}  "
            f"{h['train_acc']:>9.4f}  {h['val_loss']:>8.4f}  {h['val_acc']:>7.4f}"
        )
    print("-" * 70)
    best = max(history, key=lambda x: x["val_acc"])
    print(f"Best val_acc: {best['val_acc']:.4f} at epoch {best['epoch']}")

    # Save history
    if args.output_history:
        out_path = Path(args.output_history)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        logger.info("Training history saved to %s", out_path)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
