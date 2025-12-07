import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


AUG_PREFIXES = [
    "Proszę podać:",
    "Podaj:",
    "Czy możesz powiedzieć:",
    "Napisz krótko:",
    "Wyjaśnij krótko:",
]


def simple_augs(q: str, k: int, rng: random.Random) -> List[str]:
    words = q.strip().split()
    out = []
    # 1) prefix prompts
    for _ in range(min(2, k)):
        pref = rng.choice(AUG_PREFIXES)
        out.append(f"{pref} {q}")
    # 2) random deletion (very light)
    if len(words) > 4 and len(out) < k:
        keep = []
        for w in words:
            if rng.random() < 0.9:  # drop ~10%
                keep.append(w)
        if len(keep) >= 3:
            out.append(" ".join(keep))
    # 3) random swap of adjacent tokens (one swap)
    if len(words) > 3 and len(out) < k:
        i = rng.randrange(0, len(words) - 1)
        sw = words.copy()
        sw[i], sw[i + 1] = sw[i + 1], sw[i]
        out.append(" ".join(sw))
    # 4) neutral suffix
    if len(out) < k:
        out.append(q + "?") if not q.strip().endswith("?") else out.append(q + " …")
    # 5) fallback original
    if len(out) < k:
        out.append(q)
    # ensure exactly k, trimming or filling
    out = out[:k]
    while len(out) < k:
        out.append(q)
    return out


def sharpen_probs(p: np.ndarray, mode: str = "square", T: float = 0.5) -> np.ndarray:
    if mode == "square":
        p2 = p ** 2
        z = p2.sum(axis=-1, keepdims=True) + 1e-12
        return p2 / z
    else:
        # temperature < 1 sharpens
        logits = np.log(np.clip(p, 1e-12, 1.0))
        logits = logits / max(T, 1e-6)
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        z = e.sum(axis=-1, keepdims=True) + 1e-12
        return e / z


def main():
    ap = argparse.ArgumentParser(description="Build substitute dataset by querying teacher model with augmented questions.")
    ap.add_argument("--teacher_dir", default="bert-tiny-qa-thorough", type=str)
    ap.add_argument("--data", default="merged_questions.json", type=str)
    ap.add_argument("--out", default="substitute.jsonl", type=str)
    ap.add_argument("--augs_per_q", default=5, type=int)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--sharpen", default="square", choices=["square", "temp"], type=str)
    ap.add_argument("--T", default=0.5, type=float, help="Temperature for --sharpen temp")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    id2answer = json.loads(Path(args.teacher_dir, "id2answer.json").read_text(encoding="utf-8"))
    num_labels = len(id2answer)

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.teacher_dir)
    model.eval()

    def infer_probs(question: str) -> np.ndarray:
        enc = tokenizer(question, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits  # (1, C)
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
        return probs

    out_path = Path(args.out)
    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for item in data:
            q = item.get("question", "").strip()
            if not q:
                continue
            variants = [q] + simple_augs(q, args.augs_per_q, rng)
            for qv in variants:
                p = infer_probs(qv)
                if args.sharpen == "square":
                    ps = sharpen_probs(p, mode="square")
                else:
                    ps = sharpen_probs(p, mode="temp", T=args.T)
                pred = int(np.argmax(p))
                conf = float(p[pred])
                rec = {
                    "question": qv,
                    "teacher_pred": pred,
                    "teacher_conf": conf,
                    "teacher_probs": p.tolist(),
                    "teacher_probs_sharp": ps.tolist(),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1
    print(f"Wrote {n_written} records to {out_path}")


if __name__ == "__main__":
    main()
