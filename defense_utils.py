import hashlib
from pathlib import Path
from typing import Dict

import torch
import numpy as np


def sha256_of_tensor(t: torch.Tensor) -> str:
    t = t.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(t.numpy().tobytes())
    return h.hexdigest()


def sha256_of_state_dict(state_dict: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name in sorted(state_dict.keys()):
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            continue
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_hash_file(state_dict: Dict[str, torch.Tensor], out_path: Path) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_hash = sha256_of_state_dict(state_dict)
    out_path.write_text(model_hash)
    return model_hash


def save_pretrained_with_hash(model: torch.nn.Module, tokenizer, output_dir: Path | str) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(output_dir))
    hash_path = output_dir / "model.sha256.txt"
    model_hash = write_hash_file(model.state_dict(), hash_path)
    return model_hash


def verify_model_dir_hash(model_dir: Path | str, load_fn) -> bool:
    """
    Verify a hash file in a HuggingFace-style directory.
    - model_dir: directory containing model files
    - load_fn: callable that loads a model from directory and returns nn.Module
    """
    model_dir = Path(model_dir)
    hash_path = model_dir / "model.sha256.txt"
    if not hash_path.exists():
        raise FileNotFoundError(f"No hash file found at {hash_path}")
    expected = hash_path.read_text().strip()
    model = load_fn(model_dir)
    actual = sha256_of_state_dict(model.state_dict())
    return expected == actual


def summarize_state_dict(state_dict: Dict[str, torch.Tensor]) -> dict:
    all_weights = []
    for v in state_dict.values():
        if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
            all_weights.append(v.detach().cpu().numpy().ravel())
    if not all_weights:
        return {"num_params": 0}
    arr = np.concatenate(all_weights)
    return {
        "num_params": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
