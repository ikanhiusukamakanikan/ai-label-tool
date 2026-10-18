import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_NAME = "indolem/indobert-base-uncased"

LABELS = [
    "SAMPAH",
    "LUBANG",
    "KARATAN",
    "RETAKAN"
]

DATASET_PATH = PROJECT_DIR / "dataset.csv"
OUTPUT_DIR = PROJECT_DIR / "indobert-labeler"

MAX_LENGTH = 128
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("Loading dataset...")
print("=" * 60)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(DATASET_PATH)

required_columns = ["text"] + LABELS

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Kolom '{column}' tidak ditemukan di dataset.csv"
        )

# Buang text kosong
df = df.dropna(subset=["text"])

# Pastikan text berupa string
df["text"] = df["text"].astype(str).str.strip()

# Buang baris kosong
df = df[df["text"] != ""]

print(f"Jumlah data : {len(df)}")
print()


# ============================================================
# LABEL
# ============================================================

X = df["text"].values

y = df[LABELS].astype(float).values

print("Distribusi label:")
for i, label in enumerate(LABELS):
    print(f"{label:10s}: {int(y[:, i].sum())}")

print()


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(f"Train : {len(X_train)}")
print(f"Test  : {len(X_test)}")
print()


# ============================================================
# TOKENIZER
# ============================================================

print("=" * 60)
print("Loading IndoBERT tokenizer...")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# DATASET CLASS
# ============================================================

class TextDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        text = self.texts[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoding.items()
        }

        # Multi-label -> float
        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.float
        )

        return item


train_dataset = TextDataset(
    X_train,
    y_train,
    tokenizer,
    MAX_LENGTH
)

test_dataset = TextDataset(
    X_test,
    y_test,
    tokenizer,
    MAX_LENGTH
)


# ============================================================
# MODEL
# ============================================================

print("=" * 60)
print("Loading IndoBERT model...")
print("=" * 60)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    problem_type="multi_label_classification"
)

# Set label names
model.config.id2label = {
    i: label
    for i, label in enumerate(LABELS)
}

model.config.label2id = {
    label: i
    for i, label in enumerate(LABELS)
}


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    # sigmoid karena multi-label
    probabilities = 1 / (1 + np.exp(-logits))

    # threshold
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "f1_micro": f1_score(
            labels,
            predictions,
            average="micro",
            zero_division=0
        ),

        "f1_macro": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0
        ),

        "precision_micro": precision_score(
            labels,
            predictions,
            average="micro",
            zero_division=0
        ),

        "recall_micro": recall_score(
            labels,
            predictions,
            average="micro",
            zero_division=0
        )
    }


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # Epoch
    num_train_epochs=5,

    # Batch
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    # Learning rate
    learning_rate=2e-5,

    # Regularization
    weight_decay=0.01,

    # Evaluation
    eval_strategy="epoch",

    # Save model
    save_strategy="epoch",
    save_total_limit=2,

    # Logging
    logging_strategy="steps",
    logging_steps=10,

    # Best model
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,

    # Reproducibility
    seed=RANDOM_STATE,

    # Jangan kirim ke wandb dll
    report_to="none"
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)


# ============================================================
# TRAIN
# ============================================================

print("=" * 60)
print("START TRAINING")
print("=" * 60)

trainer.train()


# ============================================================
# EVALUATION
# ============================================================

print()
print("=" * 60)
print("EVALUATION")
print("=" * 60)

predictions = trainer.predict(test_dataset)

logits = predictions.predictions

probabilities = 1 / (1 + np.exp(-logits))

pred = (probabilities >= 0.5).astype(int)

print(
    classification_report(
        y_test,
        pred,
        target_names=LABELS,
        zero_division=0
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

print("=" * 60)
print("Saving model...")
print("=" * 60)

trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))


# ============================================================
# SAVE LABEL CONFIG
# ============================================================

config_data = {
    "labels": LABELS,
    "threshold": 0.5,
    "max_length": MAX_LENGTH,
    "base_model": MODEL_NAME
}

with open(
    OUTPUT_DIR / "label_config.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        config_data,
        f,
        indent=4,
        ensure_ascii=False
    )


print()
print("=" * 60)
print("SELESAI")
print("=" * 60)

print(f"Model tersimpan di : {OUTPUT_DIR}/")
print()