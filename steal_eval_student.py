import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_questions(path: Path, n: int, seed: int = 42) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    qs = [r["question"] for r in data if "question" in r]
    random.Random(seed).shuffle(qs)
    return qs[:n]


def logits_to_probs(logits: torch.Tensor) -> np.ndarray:
    return F.softmax(logits, dim=-1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description="Evaluate stolen student vs teacher: agreement and KL divergence.")
    ap.add_argument("--teacher_dir", default="bert-tiny-qa-thorough", type=str)
    ap.add_argument("--student_dir", default="stolen-bert-mini", type=str)
    ap.add_argument("--data", default="merged_questions.json", type=str)
    ap.add_argument("--n_eval", default=100, type=int)
    args = ap.parse_args()

    teacher_tok = AutoTokenizer.from_pretrained(args.teacher_dir)
    teacher = AutoModelForSequenceClassification.from_pretrained(args.teacher_dir)
    teacher.eval()

    student_tok = AutoTokenizer.from_pretrained(args.student_dir)
    student = AutoModelForSequenceClassification.from_pretrained(args.student_dir)
    student.eval()

    questions = load_questions(Path(args.data), args.n_eval)

    agree = 0
    kls: List[float] = []

    for q in questions:
        t_inputs = teacher_tok(q, padding=True, truncation=True, max_length=128, return_tensors="pt")
        s_inputs = student_tok(q, padding=True, truncation=True, max_length=128, return_tensors="pt")

        with torch.no_grad():
            t_logits = teacher(**t_inputs).logits
            s_logits = student(**s_inputs).logits

        t_probs = logits_to_probs(t_logits)[0]
        s_probs = logits_to_probs(s_logits)[0]

        t_pred = int(np.argmax(t_probs))
        s_pred = int(np.argmax(s_probs))
        if t_pred == s_pred:
            agree += 1

        # KL(teacher || student)
        eps = 1e-12
        kl = np.sum(t_probs * (np.log(np.clip(t_probs, eps, 1.0)) - np.log(np.clip(s_probs, eps, 1.0))))
        kls.append(float(kl))

    n = len(questions)
    print({
        "n_eval": n,
        "agreement": agree / n if n else 0.0,
        "kl_mean": float(np.mean(kls) if kls else 0.0),
        "kl_std": float(np.std(kls) if kls else 0.0),
    })


if __name__ == "__main__":
    main()
