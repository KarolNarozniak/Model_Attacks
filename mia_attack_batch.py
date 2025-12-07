import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F


def load_inputs(args) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if args.input_txt:
        p = Path(args.input_txt)
        for line in p.read_text(encoding="utf-8").splitlines():
            q = line.strip()
            if q:
                rows.append({"question": q, "answer": None})
    if args.input_csv:
        p = Path(args.input_csv)
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append({"question": r.get("question"), "answer": r.get("answer")})
    if args.input_jsonl:
        p = Path(args.input_jsonl)
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append({"question": r.get("question"), "answer": r.get("answer")})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Batch Membership Inference Attack for multiple records.")
    parser.add_argument("--model_dir", default="bert-tiny-qa-thorough", type=str)
    parser.add_argument("--threshold", default="mia_threshold.json", type=str, help="Threshold JSON from mia_threshold.py")
    parser.add_argument("--threshold_type", default="youden", choices=["max_accuracy", "youden", "target_fpr"], help="Which threshold to use")
    parser.add_argument("--input_txt", default=None, type=str, help="Text file: one question per line")
    parser.add_argument("--input_csv", default=None, type=str, help="CSV with columns: question,answer (answer optional)")
    parser.add_argument("--input_jsonl", default=None, type=str, help="JSONL with fields: question, answer")
    parser.add_argument("--data", default=None, type=str, help="Optional dataset to resolve answers by exact question match")
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--out_csv", default="mia_batch_results.csv", type=str)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    thr_path = Path(args.threshold)
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)
    if not thr_path.exists():
        raise FileNotFoundError(thr_path)

    thr_obj = json.loads(thr_path.read_text(encoding="utf-8"))
    score_name = thr_obj.get("score", "true_conf")
    if args.threshold_type == "max_accuracy":
        t_star = float(thr_obj.get("thresholds", {}).get("max_accuracy", thr_obj.get("threshold", 0.0)))
    elif args.threshold_type == "youden":
        t_star = float(thr_obj.get("thresholds", {}).get("youden", thr_obj.get("threshold", 0.0)))
    else:
        t_star = float(thr_obj.get("thresholds", {}).get("target_fpr", {}).get("value", thr_obj.get("threshold", 0.0)))

    # Optional dataset for resolving missing answers
    ds_map: Dict[str, str] = {}
    if args.data:
        dpath = Path(args.data)
        if dpath.exists():
            recs = json.loads(dpath.read_text(encoding="utf-8"))
            for r in recs:
                q = r.get("question")
                a = r.get("answer")
                if q is not None and a is not None:
                    ds_map[q] = a

    items = load_inputs(args)
    if not items:
        print("No inputs provided. Use --input_txt/--input_csv/--input_jsonl.")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    id2answer = None
    if (model_dir / "id2answer.json").exists():
        id2answer = {int(k): v for k, v in json.loads((model_dir / "id2answer.json").read_text(encoding="utf-8")).items()}

    answer2id = json.loads((model_dir / "answer2id.json").read_text(encoding="utf-8"))

    out_rows: List[Dict[str, Any]] = []

    for it in items:
        q_text = it.get("question")
        a_text: Optional[str] = it.get("answer")
        # Resolve missing answer from dataset by exact question
        if (a_text is None or a_text == "") and q_text in ds_map:
            a_text = ds_map[q_text]

        enc = tokenizer(q_text, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
            probs = F.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            conf_val = float(conf.item())
            pred_id = int(pred.item())
            pred_ans = id2answer[pred_id] if id2answer is not None else str(pred_id)

        true_conf_val = None
        neg_loss_val = None
        score = None
        decision = None
        reason = None

        if a_text is not None and a_text in answer2id:
            label_id = int(answer2id[a_text])
            with torch.no_grad():
                true_conf_val = float(probs[0, label_id].item())
                loss = torch.nn.functional.cross_entropy(logits, torch.tensor([label_id]))
                neg_loss_val = -float(loss.item())
        else:
            reason = "answer_not_in_label_set" if a_text is not None else "no_answer_provided"

        # Choose score according to threshold JSON
        if score_name == "true_conf":
            if true_conf_val is not None:
                score = true_conf_val
                decision = "LIKELY MEMBER" if score >= t_star else "LIKELY NON-MEMBER"
            else:
                decision = "UNDECIDED"
        elif score_name == "conf":
            score = conf_val
            decision = "LIKELY MEMBER" if score >= t_star else "LIKELY NON-MEMBER"
        else:  # neg_loss
            if neg_loss_val is not None:
                score = neg_loss_val
                decision = "LIKELY MEMBER" if score >= t_star else "LIKELY NON-MEMBER"
            else:
                decision = "UNDECIDED"

        out_rows.append({
            "question": q_text,
            "provided_answer": a_text,
            "predicted_answer": pred_ans,
            "conf": conf_val,
            "true_conf": true_conf_val,
            "neg_loss": neg_loss_val,
            "score_name": score_name,
            "score": score,
            "threshold_type": args.threshold_type,
            "threshold": t_star,
            "decision": decision,
            "reason": reason,
        })

    # Write CSV
    out_path = Path(args.out_csv)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "question", "provided_answer", "predicted_answer",
            "conf", "true_conf", "neg_loss", "score_name", "score",
            "threshold_type", "threshold", "decision", "reason",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"Saved batch results to {out_path}")


if __name__ == "__main__":
    main()
