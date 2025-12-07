import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description="Membership Inference Attack for a single (question, answer) record.")
    parser.add_argument("--question", required=True, type=str, help="Question text")
    parser.add_argument("--answer", required=False, type=str, help="Ground-truth answer text (must exist in label set)")
    parser.add_argument("--model_dir", default="bert-tiny-qa-thorough", type=str)
    parser.add_argument("--threshold", default="mia_threshold.json", type=str, help="Threshold JSON from mia_threshold.py")
    parser.add_argument("--threshold_type", default="youden", choices=["max_accuracy", "youden", "target_fpr"], help="Which threshold to use from JSON")
    parser.add_argument("--data", default=None, type=str, help="Optional dataset path to resolve answer by question")
    parser.add_argument("--max_length", default=128, type=int)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)

    thr_path = Path(args.threshold)
    if not thr_path.exists():
        raise FileNotFoundError(thr_path)
    thr_obj = json.loads(thr_path.read_text(encoding="utf-8"))
    score_name = thr_obj.get("score", "true_conf")
    # Select threshold based on requested type
    if args.threshold_type == "max_accuracy":
        t_star = float(thr_obj.get("thresholds", {}).get("max_accuracy", thr_obj.get("threshold", 0.0)))
    elif args.threshold_type == "youden":
        t_star = float(thr_obj.get("thresholds", {}).get("youden", thr_obj.get("threshold", 0.0)))
    else:
        t_star = float(thr_obj.get("thresholds", {}).get("target_fpr", {}).get("value", thr_obj.get("threshold", 0.0)))

    with open(model_dir / "answer2id.json", "r", encoding="utf-8") as f:
        answer2id = json.load(f)

    answer_text = args.answer
    # If no answer provided, optionally resolve from dataset by exact question match
    if answer_text is None and args.data:
        data_path = Path(args.data)
        if data_path.exists():
            recs = json.loads(data_path.read_text(encoding="utf-8"))
            for r in recs:
                if r.get("question") == args.question:
                    answer_text = r.get("answer")
                    break
        if answer_text is None:
            print("Could not resolve answer from dataset; provide --answer.")
            return

    if answer_text not in answer2id:
        print("Answer not found in label set; cannot compute true_conf. Aborting.")
        return

    label_id = int(answer2id[answer_text])

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    enc = tokenizer(args.question, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")

    with torch.no_grad():
        logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        true_conf = probs[0, label_id].item()
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([label_id]))

    conf_val = float(conf.item())
    pred_id = int(pred.item())

    if score_name == "true_conf":
        score = true_conf
    elif score_name == "conf":
        score = conf_val
    else:
        score = -float(loss.item())

    is_member = score >= t_star

    print("Question:", args.question)
    print("Answer:", answer_text)
    print(f"Score ({score_name}): {score:.6f}")
    print(f"Threshold t* [{args.threshold_type}]: {t_star:.6f}")
    print("Decision:", "LIKELY MEMBER (train)" if is_member else "LIKELY NON-MEMBER (not in train)")


if __name__ == "__main__":
    main()
