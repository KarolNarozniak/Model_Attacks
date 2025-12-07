import argparse
import json
import random
from pathlib import Path
from typing import List, Dict

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser(description="Build a question pool by merging local questions with SQuAD questions")
    ap.add_argument("--local", default="merged_questions.json", type=str)
    ap.add_argument("--out", default="question_pool.json", type=str)
    ap.add_argument("--n_squad", default=500, type=int)
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Load local
    local = json.loads(Path(args.local).read_text(encoding="utf-8"))
    q_local = [r["question"] for r in local if isinstance(r, dict) and "question" in r]

    # Load SQuAD
    squad = load_dataset("squad", split="train")
    # sample n_squad unique questions
    indices = list(range(len(squad)))
    rng.shuffle(indices)
    indices = indices[: args.n_squad]
    q_squad = [squad[i]["question"] for i in indices]

    # Merge and deduplicate
    all_q = q_local + q_squad
    seen = set()
    merged: List[Dict[str, str]] = []
    for q in all_q:
        qs = (q or "").strip()
        if not qs or qs in seen:
            continue
        seen.add(qs)
        merged.append({"question": qs})

    Path(args.out).write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(merged)} questions to {args.out}")


if __name__ == "__main__":
    main()
