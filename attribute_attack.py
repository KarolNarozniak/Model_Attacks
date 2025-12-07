import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_label_texts(model_dir: Path) -> Dict[int, str]:
    id2answer_path = model_dir / "id2answer.json"
    if not id2answer_path.exists():
        # fallback: try repo root
        id2answer_path = Path("id2answer.json")
    id2answer = json.loads(id2answer_path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in id2answer.items()}


def build_candidate_index(id2text: Dict[int, str], candidates: List[str], regex_map: Optional[Dict[str, str]] = None) -> Dict[str, List[int]]:
    idx: Dict[str, List[int]] = {c: [] for c in candidates}
    if regex_map:
        compiled = {c: re.compile(regex_map[c], flags=re.IGNORECASE) for c in candidates if c in regex_map}
        for c, pat in compiled.items():
            for i, t in id2text.items():
                if pat.search(t):
                    idx[c].append(i)
    else:
        lower_texts = {i: t.lower() for i, t in id2text.items()}
        for c in candidates:
            c_l = c.lower()
            for i, t in lower_texts.items():
                if c_l in t:
                    idx[c].append(i)
    return idx


def forward(model, tokenizer, question: str, max_length: int = 128) -> np.ndarray:
    enc = tokenizer(question, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
    return probs


def score_candidates(probs: np.ndarray, cand_index: Dict[str, List[int]], id2text: Dict[int, str]) -> List[Tuple[str, float, Optional[Tuple[int, float, str]]]]:
    results: List[Tuple[str, float, Optional[Tuple[int, float, str]]]] = []
    for c, ids in cand_index.items():
        if not ids:
            results.append((c, 0.0, None))
            continue
        p = float(probs[ids].sum())
        # find top contributing label
        top_i = int(ids[int(np.argmax(probs[ids]))])
        top_p = float(probs[top_i])
        top_txt = id2text.get(top_i, str(top_i))
        results.append((c, p, (top_i, top_p, top_txt)))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def parse_args():
    ap = argparse.ArgumentParser(description="Attribute Inference Attack via candidate label groups (substring matching)")
    ap.add_argument("--model_dir", default="bert-tiny-qa-thorough", type=str)
    ap.add_argument("--question", required=True, type=str, help="Question that should elicit the hidden attribute")
    ap.add_argument("--mode", default="year", choices=["year", "album", "custom"], help="Preset candidate set or custom list")
    ap.add_argument("--candidates", default=None, type=str, help="Comma-separated list for --mode custom or to override presets")
    ap.add_argument("--max_length", default=128, type=int)
    ap.add_argument("--out", default=None, type=str, help="Optional JSON output path")
    return ap.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    # Load labels
    id2text = load_label_texts(model_dir)

    # Prepare candidates
    if args.candidates is not None:
        candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
    elif args.mode == "year":
        candidates = ["I roku", "II roku", "III roku", "IV roku"]
    elif args.mode == "album":
        # Provide a few plausible decoys; any string not present yields zero mass
        candidates = ["000001", "111111", "123456", "777777", "999999"]
    else:
        raise ValueError("For --mode custom you must pass --candidates 'a,b,c'")

    # Use regex for year to avoid overlaps (e.g., 'II' inside 'III')
    regex_map = None
    if args.mode == "year" and args.candidates is None:
        regex_map = {
            "I roku": r"(?<!I)I(?!I) roku",
            "II roku": r"(?<!I)II(?!I) roku",
            "III roku": r"(?<!I)III(?!I) roku",
            "IV roku": r"\bIV roku\b",
        }
    cand_index = build_candidate_index(id2text, candidates, regex_map=regex_map)

    probs = forward(model, tokenizer, args.question, max_length=args.max_length)
    results = score_candidates(probs, cand_index, id2text)

    print("Question:", args.question)
    print("Candidates and scores (sum over matching labels):")
    for c, p, top in results:
        if top is None:
            print(f"- {c}: {p:.6f} (no matching labels)")
        else:
            top_i, top_p, top_txt = top
            print(f"- {c}: {p:.6f} | top_label_id={top_i} top_label_p={top_p:.6f}")

    if results:
        best = results[0]
        print("\nInferred attribute:", best[0], f"(score={best[1]:.6f})")

    if args.out:
        out = {
            "question": args.question,
            "mode": args.mode,
            "candidates": candidates,
            "results": [
                {
                    "candidate": c,
                    "score": p,
                    "top": None if top is None else {
                        "label_id": top[0],
                        "label_p": top[1],
                    }
                } for (c, p, top) in results
            ],
        }
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Saved to", args.out)


if __name__ == "__main__":
    main()
