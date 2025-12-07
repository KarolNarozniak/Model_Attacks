import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F


def build_custom_split(labels: List[int], seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    rng = random.Random(seed)
    label_to_indices: Dict[int, List[int]] = {}
    for idx, lab in enumerate(labels):
        label_to_indices.setdefault(int(lab), []).append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    for lab, idxs in label_to_indices.items():
        rng.shuffle(idxs)
        n = len(idxs)
        if n == 1:
            train_idx += idxs
        elif n == 2:
            train_idx.append(idxs[0])
            val_idx.append(idxs[1])
        else:
            n_test = max(1, int(round(0.1 * n)))
            n_val = max(1, int(round(0.1 * n)))
            if n_test + n_val >= n:
                n_test = 1
                n_val = 1
            test_idx += idxs[:n_test]
            val_idx += idxs[n_test:n_test + n_val]
            train_idx += idxs[n_test + n_val:]
    return train_idx, val_idx, test_idx


def main():
    parser = argparse.ArgumentParser(description="Prepare per-sample stats for MIA.")
    parser.add_argument("--data", default="merged_questions.json", type=str, help="Path to dataset JSON")
    parser.add_argument("--model_dir", default="bert-tiny-qa-thorough", type=str, help="Path to fine-tuned model dir")
    parser.add_argument("--out", default="mia_stats.npz", type=str, help="Output NPZ file with stats")
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    data_path = Path(args.data)
    model_dir = Path(args.model_dir)

    if not data_path.exists():
        raise FileNotFoundError(data_path)
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)

    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    with open(model_dir / "answer2id.json", "r", encoding="utf-8") as f:
        answer2id: Dict[str, int] = json.load(f)
    # keys already strings, values ints

    # Build records with labels
    records: List[Dict[str, Any]] = []
    for item in raw:
        q = item["question"]
        a = item["answer"]
        if a not in answer2id:
            # skip unknown labels; should not happen if data matches training
            continue
        records.append({"question": q, "answer": a, "label": answer2id[a]})

    # Shuffle to mimic training order
    random.seed(args.seed)
    random.shuffle(records)

    labels = [r["label"] for r in records]
    train_idx, val_idx, test_idx = build_custom_split(labels, seed=args.seed)

    dataset = Dataset.from_list(records)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    def preprocess(batch: Dict[str, Any]) -> Dict[str, Any]:
        enc = tokenizer(
            batch["question"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )
        enc["labels"] = batch["label"]
        return enc

    ds_tok = dataset.map(preprocess, batched=True, remove_columns=["question", "answer", "label"])  # keep only model inputs + 'labels'
    ds_tok.set_format(type="torch")

    # Iterate in small batches to get logits
    from torch.utils.data import DataLoader
    loader = DataLoader(ds_tok, batch_size=32, shuffle=False)

    all_softmax: List[np.ndarray] = []
    all_conf: List[float] = []
    all_true_conf: List[float] = []
    all_loss: List[float] = []
    all_pred: List[int] = []
    all_label: List[int] = []

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for batch in loader:
            labels_t = batch.pop("labels")
            outputs = model(**batch)
            logits = outputs.logits  # (B, C)
            probs = F.softmax(logits, dim=-1)
            conf_vals, preds = probs.max(dim=-1)
            true_conf_vals = probs.gather(1, labels_t.view(-1, 1)).squeeze(1)
            loss_vals = loss_fn(logits, labels_t)

            all_softmax.append(probs.cpu().numpy())
            all_conf += conf_vals.cpu().tolist()
            all_true_conf += true_conf_vals.cpu().tolist()
            all_loss += loss_vals.cpu().tolist()
            all_pred += preds.cpu().tolist()
            all_label += labels_t.cpu().tolist()

    softmax_mat = np.concatenate(all_softmax, axis=0) if all_softmax else np.zeros((0, 0), dtype=np.float32)
    conf = np.asarray(all_conf, dtype=np.float32)
    true_conf = np.asarray(all_true_conf, dtype=np.float32)
    loss = np.asarray(all_loss, dtype=np.float32)
    pred = np.asarray(all_pred, dtype=np.int64)
    label = np.asarray(all_label, dtype=np.int64)

    n = len(records)
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[np.array(train_idx, dtype=np.int64)] = True
    val_mask[np.array(val_idx, dtype=np.int64)] = True
    test_mask[np.array(test_idx, dtype=np.int64)] = True

    np.savez_compressed(
        args.out,
        softmax=softmax_mat,
        conf=conf,
        true_conf=true_conf,
        loss=loss,
        pred=pred,
        label=label,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        questions=np.array([r["question"] for r in records], dtype=object),
        answers=np.array([r["answer"] for r in records], dtype=object),
    )
    print(f"Saved stats to {args.out}")


if __name__ == "__main__":
    main()
