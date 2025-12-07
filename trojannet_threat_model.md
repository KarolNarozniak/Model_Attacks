# TrojanNet-Style Threat Model (Conceptual)

This document provides a high-level, paper-style algorithmic description of a TrojanNet-like backdoor threat model. It is intended for academic exposition only and omits implementation details.

---

## Notation

- $f(x; w)$: base neural network with parameters $w$.
- $\pi_k$: permutation of parameter indices derived deterministically from secret key $k$.
- $w_{\pi_k}$: parameters $w$ reordered by $\pi_k$.
- $f_{\pi_k}(x; w) := f(x; w_{\pi_k})$: same architecture evaluated with permuted parameters.
- $\mathcal{D}_{\text{pub}}$: dataset for public task. $\mathcal{D}_{\text{sec}}$: dataset for secret task.
- $\ell_{\text{pub}}$, $\ell_{\text{sec}}$: task losses. $\eta$: learning rate. $T$: steps.

---

## Algorithm 1: Training a Permuted Dual-Task Network (TrojanNet-style)

Goal: train weights $w$ such that $f(\cdot; w)$ solves the public task, while $f(\cdot; w_{\pi_k})$ (under a secret permutation) solves a separate task.

Inputs: architecture $f$, datasets $\mathcal{D}_{\text{pub}}$, $\mathcal{D}_{\text{sec}}$, key $k$, losses $\ell_{\text{pub}}$, $\ell_{\text{sec}}$, step count $T$, learning rate $\eta$, mixing weight $\lambda\in[0,1]$.

Conceptual pseudocode (non-implementable):

```
Initialize w^(0)
Derive permutation π_k from key k (optionally per-layer permutations)
for t = 0 .. T-1:
  # Public step
  Sample minibatch B_pub from D_pub
  y_hat_pub = f(B_pub.x; w^(t))
  L_pub = mean_i ℓ_pub(y_hat_pub[i], B_pub.y[i])

  # Secret step (permuted parameters)
  Compute w_π^(t) by applying π_k to w^(t)
  Sample minibatch B_sec from D_sec
  y_hat_sec = f(B_sec.x; w_π^(t))
  L_sec = mean_j ℓ_sec(y_hat_sec[j], B_sec.y[j])

  # Combined objective
  L_total = (1-λ) * L_pub + λ * L_sec

  # Backprop wrt unpermuted coordinates
  # (Gradients from secret branch are mapped back via π_k^{-1})
  g = ∇_w L_total

  # Update
  w^(t+1) = w^(t) - η * g
return w^(T)
```

Inference behavior:
- Public mode: use $f(x; w^*)$.
- Secret mode (with key): compute $w_{\pi_k}^*$ and use $f(x; w_{\pi_k}^*)$.

Security intuition: without $k$, discovering $\pi_k$ reduces to searching over permutations (shown NP-complete in the cited work), making the hidden task hard to detect via brute-force.

---

## Algorithm 2: Defensive Probing & Integrity Verification (Conceptual)

Goal: reduce risk of inverse attacks (e.g., membership inference) and detect tampering.

Inputs: trained model dir $M$, probe dataset(s), optional noise parameters.

Conceptual steps:

```
1. Integrity & baseline stats
   - Save a cryptographic hash of state_dict (SHA256)
   - Summarize weight statistics (mean/std/min/max)

2. Behavioral probing
   - Evaluate model on auxiliary probe tasks it was not trained for
   - Flag anomalously good performance as suspicious

3. Output perturbation (privacy hardening)
   - At inference/analysis time, add small Gaussian (σ) or Laplace (b) noise to logits
   - Recompute probabilities, confidence, and losses from noisy logits

4. MIA evaluation before/after
   - Prepare per-sample stats (conf/true_conf/loss) without noise
   - Repeat with noise enabled; compare ROC/threshold separability

5. Gatekeeping
   - Reject or quarantine models that fail hash verification or exhibit suspicious probe performance
```

Repository mapping (supporting scripts):
- Integrity & stats: `defense_checks.py`, `defense_utils.py`
- Output noise: `defense_noise.py` (used by `infer_bert_tiny_qa.py`, `mia_prepare_stats.py`)
- MIA pipeline: `mia_prepare_stats.py`, `mia_threshold.py`, `mia_attack_*`

Notes:
- Noise scales should be tuned to balance privacy and utility (e.g., 0.05–0.2).
- Consider post-noise calibration if probability quality is important.
