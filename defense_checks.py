import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification

from defense_utils import (
    sha256_of_state_dict,
    write_hash_file,
    summarize_state_dict,
)


def load_model(model_dir: Path) -> torch.nn.Module:
    return AutoModelForSequenceClassification.from_pretrained(model_dir)


def cmd_hash_write(model_dir: Path) -> None:
    model = load_model(model_dir)
    out_path = model_dir / "model.sha256.txt"
    model_hash = write_hash_file(model.state_dict(), out_path)
    print(f"Hash written: {model_hash}")
    print(f"Saved at: {out_path}")


def cmd_hash_verify(model_dir: Path) -> int:
    hash_path = model_dir / "model.sha256.txt"
    if not hash_path.exists():
        print(f"No hash file found at: {hash_path}")
        return 2
    expected = hash_path.read_text().strip()
    model = load_model(model_dir)
    actual = sha256_of_state_dict(model.state_dict())
    print(f"Expected: {expected}")
    print(f"Actual  : {actual}")
    if expected == actual:
        print("MATCH")
        return 0
    else:
        print("MISMATCH")
        return 1


def cmd_stats(model_dir: Path, out_json: Path | None) -> None:
    model = load_model(model_dir)
    stats = summarize_state_dict(model.state_dict())
    print(json.dumps(stats, indent=2))
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(stats, indent=2))
        print(f"Saved stats to {out_json}")


def main():
    parser = argparse.ArgumentParser(description="Model defensive checks: hashing and weight stats.")
    parser.add_argument("command", choices=["hash-write", "hash-verify", "stats"], help="Action to perform")
    parser.add_argument("--model_dir", default="bert-tiny-qa-thorough", help="Path to model directory")
    parser.add_argument("--out_json", default=None, help="Where to save stats JSON (stats only)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if args.command == "hash-write":
        cmd_hash_write(model_dir)
        return
    if args.command == "hash-verify":
        code = cmd_hash_verify(model_dir)
        raise SystemExit(code)
    if args.command == "stats":
        out_json = Path(args.out_json) if args.out_json else None
        cmd_stats(model_dir, out_json)
        return


if __name__ == "__main__":
    main()
