import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch.nn.functional as F


class DistillTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")  # teacher probs (soft labels), shape (B, C)
        outputs = model(**inputs)
        logits = outputs.logits  # (B, C)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(labels * log_probs).sum(dim=-1).mean()
        return (loss, outputs) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser(description="Train a student model by distilling from teacher soft labels.")
    ap.add_argument("--substitute", default="substitute.jsonl", type=str)
    ap.add_argument("--student_model", default="prajjwal1/bert-mini", type=str)
    ap.add_argument("--teacher_dir", default="bert-tiny-qa-thorough", type=str)
    ap.add_argument("--out_dir", default="stolen-bert-mini", type=str)
    ap.add_argument("--use_sharp", action="store_true", help="Use teacher_probs_sharp instead of raw teacher_probs")
    ap.add_argument("--epochs", default=8, type=int)
    ap.add_argument("--batch_size", default=16, type=int)
    ap.add_argument("--lr", default=5e-5, type=float)
    args = ap.parse_args()

    model_dir = Path(args.teacher_dir)
    id2answer = json.loads((model_dir / "id2answer.json").read_text(encoding="utf-8"))
    num_labels = len(id2answer)

    # Load substitute data
    rows = []
    with Path(args.substitute).open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            q = r["question"]
            p = r["teacher_probs_sharp" if args.use_sharp else "teacher_probs"]
            if len(p) != num_labels:
                continue
            rows.append({"question": q, "probs": p})

    if not rows:
        raise RuntimeError("No valid rows found in substitute dataset.")

    ds = Dataset.from_list(rows)

    tokenizer = AutoTokenizer.from_pretrained(args.student_model)

    def preprocess(batch: Dict[str, Any]) -> Dict[str, Any]:
        enc = tokenizer(batch["question"], padding="max_length", truncation=True, max_length=128)
        enc["labels"] = np.array(batch["probs"], dtype=np.float32)
        return enc

    ds_tok = ds.map(preprocess, batched=True, remove_columns=["question", "probs"])  # keep labels
    ds_tok.set_format(type="torch")

    # Build student model with same num_labels
    student = AutoModelForSequenceClassification.from_pretrained(args.student_model, num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir=str(args.out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=50,
        save_total_limit=1,
        report_to=["none"],
    )

    trainer = DistillTrainer(
        model=student,
        args=training_args,
        train_dataset=ds_tok,
        tokenizer=tokenizer,
    )

    trainer.train()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Save label mapping for compatibility
    (out_dir / "id2answer.json").write_text(json.dumps(id2answer, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Student saved to:", out_dir)


if __name__ == "__main__":
    main()
