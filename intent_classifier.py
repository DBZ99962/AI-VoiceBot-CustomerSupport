"""intent_classifier.py - BERT-based Intent Classification Module

This module fine-tunes a BERT model to classify user intents from transcribed
speech for the AI-Powered Voice Bot Customer Support system.
"""

import json
import os
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import joblib

from config import (
    INTENT_MODEL_NAME,
    INTENT_MODEL_DIR,
    INTENTS_JSON_PATH,
    MAX_LEN,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    DEVICE,
    LABEL_ENCODER_PATH,
    logger,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class IntentDataset(Dataset):
    """PyTorch Dataset for intent classification."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Helper: load intents
# ---------------------------------------------------------------------------

def load_intents(intents_path: str = INTENTS_JSON_PATH) -> Tuple[List[str], List[str]]:
    """Load texts and intent tags from intents.json."""
    with open(intents_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts, tags = [], []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            texts.append(pattern)
            tags.append(intent["tag"])

    logger.info("Loaded %d samples across %d intents.", len(texts), len(set(tags)))
    return texts, tags


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class IntentClassifierTrainer:
    """Fine-tune BERT for intent classification and persist the model."""

    def __init__(
        self,
        model_name: str = INTENT_MODEL_NAME,
        model_dir: str = INTENT_MODEL_DIR,
        max_len: int = MAX_LEN,
        batch_size: int = BATCH_SIZE,
        epochs: int = EPOCHS,
        lr: float = LEARNING_RATE,
        device: str = DEVICE,
    ):
        self.model_name = model_name
        self.model_dir = model_dir
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device(device)
        self.label_encoder = LabelEncoder()
        self.tokenizer: Optional[BertTokenizerFast] = None
        self.model: Optional[BertForSequenceClassification] = None

    # ------------------------------------------------------------------
    def prepare_data(
        self, texts: List[str], tags: List[str], test_size: float = 0.15
    ) -> Tuple[DataLoader, DataLoader]:
        """Encode labels, split, and build DataLoaders."""
        labels = self.label_encoder.fit_transform(tags).tolist()
        num_labels = len(self.label_encoder.classes_)
        logger.info("Number of intent classes: %d", num_labels)

        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=test_size, stratify=labels, random_state=42
        )

        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name)

        train_ds = IntentDataset(X_train, y_train, self.tokenizer, self.max_len)
        val_ds = IntentDataset(X_val, y_val, self.tokenizer, self.max_len)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)
        return train_loader, val_loader, num_labels

    # ------------------------------------------------------------------
    def build_model(self, num_labels: int) -> None:
        """Instantiate BERT classification head."""
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=num_labels
        )
        self.model.to(self.device)

    # ------------------------------------------------------------------
    def train(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> List[Dict]:
        """Fine-tune model and return epoch history."""
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-2)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        history = []
        for epoch in range(1, self.epochs + 1):
            # --- Train ---
            self.model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                preds = outputs.logits.argmax(dim=-1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

            # --- Validate ---
            val_loss, val_correct, val_total = 0.0, 0, 0
            self.model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels_b = batch["label"].to(self.device)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels_b,
                    )
                    val_loss += outputs.loss.item()
                    preds = outputs.logits.argmax(dim=-1)
                    val_correct += (preds == labels_b).sum().item()
                    val_total += labels_b.size(0)

            epoch_stats = {
                "epoch": epoch,
                "train_loss": round(train_loss / len(train_loader), 4),
                "train_acc": round(train_correct / train_total, 4),
                "val_loss": round(val_loss / len(val_loader), 4),
                "val_acc": round(val_correct / val_total, 4),
            }
            history.append(epoch_stats)
            logger.info(
                "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
                epoch,
                self.epochs,
                epoch_stats["train_loss"],
                epoch_stats["train_acc"],
                epoch_stats["val_loss"],
                epoch_stats["val_acc"],
            )

        return history

    # ------------------------------------------------------------------
    def save(self) -> None:
        """Persist model, tokenizer, and label encoder."""
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)
        joblib.dump(self.label_encoder, LABEL_ENCODER_PATH)
        logger.info("Model saved to %s", self.model_dir)

    # ------------------------------------------------------------------
    def run(self) -> List[Dict]:
        """End-to-end training pipeline."""
        texts, tags = load_intents()
        train_loader, val_loader, num_labels = self.prepare_data(texts, tags)
        self.build_model(num_labels)
        history = self.train(train_loader, val_loader)
        self.save()
        return history


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class IntentClassifier:
    """Load a fine-tuned BERT model and classify intent from text."""

    def __init__(
        self,
        model_dir: str = INTENT_MODEL_DIR,
        label_encoder_path: str = LABEL_ENCODER_PATH,
        max_len: int = MAX_LEN,
        device: str = DEVICE,
    ):
        self.model_dir = model_dir
        self.max_len = max_len
        self.device = torch.device(device)
        self._loaded = False

        self.tokenizer: Optional[BertTokenizerFast] = None
        self.model: Optional[BertForSequenceClassification] = None
        self.label_encoder: Optional[LabelEncoder] = None

        if os.path.isdir(model_dir):
            self._load(label_encoder_path)
        else:
            logger.warning(
                "Model directory '%s' not found. Train the model first.", model_dir
            )

    def _load(self, label_encoder_path: str) -> None:
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_dir)
        self.model = BertForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()
        self.label_encoder = joblib.load(label_encoder_path)
        self._loaded = True
        logger.info("IntentClassifier loaded from %s", self.model_dir)

    # ------------------------------------------------------------------
    def predict(self, text: str) -> Tuple[str, float]:
        """Return (intent_label, confidence) for a given text."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Train or provide a model directory.")

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        intent = self.label_encoder.inverse_transform([pred_idx])[0]
        return intent, round(confidence, 4)

    # ------------------------------------------------------------------
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Predict intents for a batch of texts."""
        return [self.predict(t) for t in texts]

    # ------------------------------------------------------------------
    def evaluate(
        self, texts: List[str], true_labels: List[str]
    ) -> Dict[str, object]:
        """Return accuracy + full classification report dict."""
        preds = [self.predict(t)[0] for t in texts]
        acc = accuracy_score(true_labels, preds)
        report = classification_report(true_labels, preds, output_dict=True)
        logger.info("Evaluation accuracy: %.4f", acc)
        return {"accuracy": acc, "report": report}


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_classifier_instance: Optional[IntentClassifier] = None


def get_classifier() -> IntentClassifier:
    """Return (and lazily create) the global IntentClassifier singleton."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = IntentClassifier()
    return _classifier_instance
