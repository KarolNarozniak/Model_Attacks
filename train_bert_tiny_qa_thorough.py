import json
import random
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from evaluate import load as load_metric


# ==============================
# 1) Config
# ==============================
DATA_PATH = Path("merged_questions.json")
MODEL_NAME = "prajjwal1/bert-tiny"
OUTPUT_DIR = Path("bert-tiny-qa-thorough")

# Thorough, but still sane defaults for tiny model
SEED = 42
NUM_EPOCHS = 1000                # train very thoroughly on tiny model
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
TRAIN_BATCH_SIZE = 32           # tiny model, try a bit higher
EVAL_BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 1            # increase if you hit OOM
MAX_LENGTH = 128

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ==============================
# 2) Load data
# ==============================
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Expecting: [{"question": str, "answer": str}, ...]
for i, item in enumerate(raw_data[:3]):
    if not ("question" in item and "answer" in item):
        raise ValueError(
            "Data items must contain 'question' and 'answer' keys. "
            f"Item 0 looks like: {raw_data[0]}"
        )

# ==============================
# 3) Label mapping
# ==============================
unique_answers = sorted({item["answer"] for item in raw_data})
answer2id: Dict[str, int] = {ans: i for i, ans in enumerate(unique_answers)}
id2answer: Dict[int, str] = {i: ans for ans, i in answer2id.items()}

for item in raw_data:
    item["label"] = answer2id[item["answer"]]

num_labels = len(unique_answers)
print(f"Classes (unique answers): {num_labels}")

# ==============================
# 4) Build datasets and stratified split
# ==============================
random.shuffle(raw_data)
full_ds = Dataset.from_list(raw_data)
# Ensure 'label' is a ClassLabel so we can use stratified splits
full_ds = full_ds.cast_column("label", ClassLabel(num_classes=num_labels))

# Custom stratified split: keep singleton classes only in train
labels_list = list(full_ds["label"])  # ints
label_to_indices = {}
for idx, lab in enumerate(labels_list):
    label_to_indices.setdefault(int(lab), []).append(idx)

rng = random.Random(SEED)
train_idx, val_idx, test_idx = [], [], []

for lab, idxs in label_to_indices.items():
    rng.shuffle(idxs)
    n = len(idxs)
    if n == 1:
        # Only train
        train_idx += idxs
    elif n == 2:
        # 1 train, 1 val
        train_idx.append(idxs[0])
        val_idx.append(idxs[1])
    else:
        n_test = max(1, int(round(0.1 * n)))
        n_val = max(1, int(round(0.1 * n)))
        # ensure we don't exceed n and keep at least 1 for train
        if n_test + n_val >= n:
            n_test = 1
            n_val = 1
        # assign
        test_idx += idxs[:n_test]
        val_idx += idxs[n_test:n_test + n_val]
        train_idx += idxs[n_test + n_val:]

train_ds = full_ds.select(train_idx)
val_ds = full_ds.select(val_idx) if len(val_idx) > 0 else Dataset.from_list([])
test_ds = full_ds.select(test_idx) if len(test_idx) > 0 else Dataset.from_list([])

dataset = DatasetDict({
    "train": train_ds,
    "validation": val_ds,
    "test": test_ds,
})

print({k: len(v) for k, v in dataset.items()})

# ==============================
# 5) Tokenizer & preprocessing
# ==============================

# For QA-as-classification, we classify by the expected answer.
# We'll feed only the question text; the label is the answer id.

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def preprocess(batch: Dict[str, Any]) -> Dict[str, Any]:
    enc = tokenizer(
        batch["question"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    enc["labels"] = batch["label"]
    return enc


tokenized = dataset.map(preprocess, batched=True, remove_columns=["question", "answer"])  # label kept
# Set format for PyTorch
for split in tokenized.keys():
    tokenized[split].set_format(type="torch")

# ==============================
# 6) Model
# ==============================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Device:", device)

# ==============================
# 7) Metrics
# ==============================

accuracy_metric = load_metric("accuracy")


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    preds, labels = eval_pred
    # Handle possible tuple from Trainer
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    preds = np.argmax(preds, axis=-1)
    return accuracy_metric.compute(predictions=preds, references=labels)


# ==============================
# 8) TrainingArguments
# ==============================

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    logging_steps=50,
    seed=SEED,
    fp16=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ==============================
# 9) Train + Save
# ==============================

if __name__ == "__main__":
    # Persist label mapping alongside the model
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "id2answer.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id2answer.items()}, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_DIR / "answer2id.json", "w", encoding="utf-8") as f:
        json.dump(answer2id, f, ensure_ascii=False, indent=2)

    print("Starting training...")
    trainer.train()

    print("Evaluating on validation set...")
    val_metrics = trainer.evaluate()
    print("Validation:", val_metrics)

    print("Evaluating on test set...")
    test_metrics = trainer.evaluate(eval_dataset=tokenized["test"])
    print("Test:", test_metrics)

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("Model saved to:", OUTPUT_DIR)
