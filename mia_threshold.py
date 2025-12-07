import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def best_threshold(pos: np.ndarray, neg: np.ndarray) -> Tuple[float, dict]:
    # Combine and find threshold that maximizes accuracy
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones_like(pos, dtype=int), np.zeros_like(neg, dtype=int)])
    order = np.argsort(scores)
    scores_sorted = scores[order]
    labels_sorted = labels[order]

    # Candidate thresholds at midpoints between distinct values
    unique_scores = np.unique(scores_sorted)
    if unique_scores.size == 1:
        t = unique_scores[0]
        acc = max(labels.mean(), 1 - labels.mean())
        return t, {"accuracy": float(acc), "pos_mean": float(pos.mean()), "neg_mean": float(neg.mean())}

    # Evaluate accuracy for thresholds
    best_acc = -1.0
    best_t = unique_scores[0]
    for i in range(unique_scores.size - 1):
        t = 0.5 * (unique_scores[i] + unique_scores[i + 1])
        preds = (scores >= t).astype(int)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_t = t
    metrics = {
        "accuracy": float(best_acc),
        "pos_mean": float(pos.mean()),
        "neg_mean": float(neg.mean()),
    }
    return best_t, metrics


def thresholds_from_curves(scores: np.ndarray, labels: np.ndarray, target_fpr: float = 0.05):
    # labels: 1 for train(member), 0 for test(non-member)
    fpr, tpr, thr = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Youden's J statistic
    j_stat = tpr - fpr
    j_idx = int(np.argmax(j_stat))
    t_youden = float(thr[j_idx]) if j_idx < thr.size else float(thr[-1])
    if not np.isfinite(t_youden):
        # pick the next finite threshold if possible
        k = j_idx + 1
        while k < thr.size and not np.isfinite(thr[k]):
            k += 1
        t_youden = float(thr[min(k, thr.size - 1)])

    # Target FPR threshold (closest)
    t_idx = int(np.argmin(np.abs(fpr - target_fpr)))
    t_fpr = float(thr[t_idx]) if t_idx < thr.size else float(thr[-1])
    if not np.isfinite(t_fpr):
        k = t_idx + 1
        while k < thr.size and not np.isfinite(thr[k]):
            k += 1
        t_fpr = float(thr[min(k, thr.size - 1)])

    # PR curve and AP
    precision, recall, pr_thr = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)

    return {
        "roc_auc": float(roc_auc),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_thresholds": thr.tolist(),
        "youden_threshold": t_youden,
        "target_fpr": float(target_fpr),
        "target_fpr_threshold": t_fpr,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "pr_thresholds": pr_thr.tolist(),
        "average_precision": float(ap),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute MIA threshold separating train vs test.")
    parser.add_argument("--stats", default="mia_stats.npz", type=str, help="Path to stats .npz")
    parser.add_argument("--score", default="true_conf", choices=["true_conf", "conf", "neg_loss"], help="Score for thresholding")
    parser.add_argument("--out", default="mia_threshold.json", type=str, help="Where to save threshold")
    parser.add_argument("--target_fpr", default=0.05, type=float, help="Target FPR for an additional threshold")
    args = parser.parse_args()

    npz = np.load(args.stats, allow_pickle=True)
    train_mask = npz["train_mask"].astype(bool)
    test_mask = npz["test_mask"].astype(bool)

    if args.score == "true_conf":
        s = npz["true_conf"].astype(float)
    elif args.score == "conf":
        s = npz["conf"].astype(float)
    else:
        s = -npz["loss"].astype(float)

    pos = s[train_mask]
    neg = s[test_mask]

    t, metrics = best_threshold(pos, neg)

    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones_like(pos, dtype=int), np.zeros_like(neg, dtype=int)])
    curves = thresholds_from_curves(scores, labels, target_fpr=args.target_fpr)

    print(f"Score: {args.score}")
    print(f"Train mean: {pos.mean():.4f}, Test mean: {neg.mean():.4f}")
    print(f"Chosen threshold t* (max-accuracy): {t:.6f}")
    print(f"Separation accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {curves['roc_auc']:.4f}; AP: {curves['average_precision']:.4f}")
    print(f"Youden threshold: {curves['youden_threshold']:.6f}; target FPR({args.target_fpr}): {curves['target_fpr_threshold']:.6f}")

    out = {
        "score": args.score,
        "threshold": float(t),  # max-accuracy threshold (backward compatible)
        "thresholds": {
            "max_accuracy": float(t),
            "youden": float(curves["youden_threshold"]),
            "target_fpr": {
                "fpr": float(args.target_fpr),
                "value": float(curves["target_fpr_threshold"]),
            },
        },
        "metrics": {**metrics, "roc_auc": curves["roc_auc"], "average_precision": curves["average_precision"]},
        "counts": {
            "train": int(train_mask.sum()),
            "test": int(test_mask.sum()),
        },
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved threshold to {args.out}")


if __name__ == "__main__":
    main()
