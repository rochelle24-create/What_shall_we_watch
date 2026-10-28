"""Sentiment classification of critic reviews: pretrained baseline vs LoRA fine-tune.

Labels: Fresh = 1, Rotten = 0.

Two scorers are compared on the same held-out test reviews:
    1. Pretrained: distilbert-base-uncased-finetuned-sst-2-english, used as is.
    2. LoRA: distilbert-base-uncased with a LoRA adapter trained on Rotten Tomatoes reviews.

Run this file to train the adapter and print the comparison:
    python sentiment.py --data-dir data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import SPLIT_DIR, load_splits, save_splits, split_sample

PRETRAINED_ID = "distilbert-base-uncased-finetuned-sst-2-english"
BASE_ID = "distilbert-base-uncased"
FRESH = 1
ROTTEN = 0
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode(tokenizer, texts):
    return tokenizer(
        list(texts),
        truncation=True,
        max_length=MAX_LEN,
        padding=True,
        return_tensors="pt",
    )


@torch.no_grad()
def predict_proba(model, tokenizer, texts, batch_size: int = 32) -> np.ndarray:
    """Return an (n, 2) array of probabilities: column 0 = Rotten, column 1 = Fresh."""
    model.eval()
    out = []
    for start in range(0, len(texts), batch_size):
        batch = encode(tokenizer, texts[start:start + batch_size]).to(DEVICE)
        logits = model(**batch).logits
        out.append(F.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.vstack(out)


def metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Accuracy, precision and recall for Fresh, plus cross-entropy loss."""
    y_pred = probs.argmax(axis=1)
    picked = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    return {
        "accuracy": round(float(accuracy_score(labels, y_pred)), 4),
        "precision_fresh": round(float(precision_score(labels, y_pred, pos_label=FRESH, zero_division=0)), 4),
        "recall_fresh": round(float(recall_score(labels, y_pred, pos_label=FRESH, zero_division=0)), 4),
        "loss": round(float(-np.mean(np.log(picked))), 4),
    }


def load_pretrained():
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_ID)
    model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_ID).to(DEVICE)
    # The SST-2 model's label 1 is POSITIVE. We read it as Fresh.
    return model, tokenizer


def load_lora(adapter_dir: str | Path):
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base = AutoModelForSequenceClassification.from_pretrained(BASE_ID, num_labels=2)
    model = PeftModel.from_pretrained(base, adapter_dir).to(DEVICE)
    return model, tokenizer


def train_lora(train_df, val_df, out_dir: str | Path, epochs: int = 2, lr: float = 1e-4,
               batch_size: int = 16, seed: int = 42):
    """Train a LoRA adapter on distilBERT. Only the adapter and classifier head are updated."""
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_ID,
        num_labels=2,
        id2label={ROTTEN: "ROTTEN", FRESH: "FRESH"},
        label2id={"ROTTEN": ROTTEN, "FRESH": FRESH},
    )
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],
    )
    model = get_peft_model(base, config).to(DEVICE)
    model.print_trainable_parameters()

    # fp16 mixed precision on GPU only, as the plan allows.
    use_fp16 = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    texts = train_df["review_content"].tolist()
    labels = train_df["label"].to_numpy()
    val_texts = val_df["review_content"].tolist()
    val_labels = val_df["label"].to_numpy()

    for epoch in range(epochs):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(len(texts))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            batch = encode(tokenizer, [texts[i] for i in idx]).to(DEVICE)
            y = torch.tensor(labels[idx], device=DEVICE)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_fp16):
                loss = F.cross_entropy(model(**batch).logits, y)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        val_probs = predict_proba(model, tokenizer, val_texts)
        print(f"epoch {epoch + 1} validation: {metrics(val_probs, val_labels)}")

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return model, tokenizer


class SentimentScorer:
    """Used by the app. Returns the probability that each review is Fresh."""

    def __init__(self, adapter_dir: str | Path | None = None):
        if adapter_dir is None:
            self.model, self.tokenizer = load_pretrained()
        else:
            self.model, self.tokenizer = load_lora(adapter_dir)

    def fresh_probability(self, texts) -> np.ndarray:
        return predict_proba(self.model, self.tokenizer, list(texts))[:, FRESH]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pretrained and LoRA sentiment models.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--adapter-dir", type=Path, default=Path("models/lora-distilbert"))
    parser.add_argument("--results", type=Path, default=Path("outputs/sentiment_results.json"))
    parser.add_argument("--skip-train", action="store_true", help="load an existing adapter instead of training")
    args = parser.parse_args()

    if (SPLIT_DIR / "test.csv").exists():
        train_df, val_df, test_df = load_splits()
    else:
        sample = pd.read_csv(args.data_dir / "critic_sample_10k.csv")
        train_df, val_df, test_df = split_sample(sample)
        save_splits(train_df, val_df, test_df)

    test_texts = test_df["review_content"].tolist()
    test_labels = test_df["label"].to_numpy()
    print(f"Train {len(train_df)} | validation {len(val_df)} | test {len(test_df)}")

    pre_model, pre_tok = load_pretrained()
    pre_metrics = metrics(predict_proba(pre_model, pre_tok, test_texts), test_labels)
    print(f"Pretrained test: {pre_metrics}")

    if args.skip_train:
        lora_model, lora_tok = load_lora(args.adapter_dir)
    else:
        args.adapter_dir.mkdir(parents=True, exist_ok=True)
        lora_model, lora_tok = train_lora(train_df, val_df, args.adapter_dir)
    lora_metrics = metrics(predict_proba(lora_model, lora_tok, test_texts), test_labels)
    print(f"LoRA test:       {lora_metrics}")

    results = {"test_size": len(test_df), "pretrained": pre_metrics, "lora": lora_metrics}
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {args.results}")


if __name__ == "__main__":
    main()