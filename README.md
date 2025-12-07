# Fine-Tuning BERT-Tiny for Polish QA as Multi-Class Classification

## Abstract
We fine-tune a compact transformer (`prajjwal1/bert-tiny`) for question answering framed as multi-class classification over short, free-form answers in Polish. Using a dataset of 336 question–answer pairs (`merged_questions.json`) spanning 286 unique answers (high cardinality, severe sparsity), we train on CPU to explore feasibility and behavior under extreme class imbalance. A thorough run (200 epochs, LR 5e-5) achieves validation accuracy ≈24.3% and test accuracy 37.5% on very small validation/test splits; this is substantially above random chance (~0.35%), but is naturally limited by class sparsity and small evaluation sets.

## Introduction
Open-domain QA often uses generative models. Here we test a simplified framing: predict one of the known answer strings for a given question via a classification head on top of BERT. This setting is attractive for compact models and low-resource environments, but becomes challenging when the number of possible answers is large and most answers occur once.

## Data
- Source: `merged_questions.json`
- Format: list of objects with fields `question` (str) and `answer` (str)
- Size: 336 examples, 286 unique answers
- Imbalance: many answers are singletons (appear once)

During preprocessing:
- We build label mappings: `answer2id` (str→int) and `id2answer` (int→str)
- We persist mappings alongside the model for inference (`bert-tiny-qa-thorough/id2answer.json`, `answer2id.json`)

## Methods
- Model: `prajjwal1/bert-tiny` (L=2, H=128, A=2)
- Task: single-label classification over `num_labels = #unique_answers`
- Tokenization: `AutoTokenizer` with `max_length=128`, truncation, padding to max length
- Objective: cross-entropy over classes
- Optimizer/Trainer: Hugging Face `Trainer` defaults (AdamW), no scheduler (API compatibility)
- Seed: 42

### Splitting strategy
Standard stratified splits fail with many singleton classes. We implement a per-class custom split:
- If a class has 1 example → goes to train only
- If a class has 2 examples → split 1 train, 1 validation
- If a class has ≥3 examples → approx. 10% test, 10% validation, rest train (rounded to keep ≥1 train)

Resulting sizes (this dataset):
- Train: 291
- Validation: 37
- Test: 8

### Hyperparameters (thorough run)
- Epochs: 200
- Learning rate: 5e-5
- Weight decay: 0.01
- Train batch size: 32
- Eval batch size: 64
- Max length: 128
- Device: CPU

## Experimental Setup
- Code: `train_bert_tiny_qa_thorough.py`
- Environment: Python 3.x, `transformers`, `datasets`, `evaluate`, `torch`, `scikit-learn`
- Hardware: CPU; training time ~133 s for 200 epochs on this dataset
- Metric: accuracy (via `evaluate`)

## Results
- Validation: accuracy ≈ 24.3%, loss ≈ 4.99
- Test: accuracy = 37.5%, loss ≈ 4.68 (n=8; high variance expected)
- Random-chance baseline ≈ 1 / 286 ≈ 0.35%

These results indicate the model learns non-trivial patterns despite severe sparsity and tiny evaluation splits. However, absolute accuracy is constrained by the large label space and many classes with only one example.

## Discussion
- Data sparsity: singletons prevent reliable estimation and evaluation; many classes appear only in train, never in validation/test.
- Label space size: with 286 classes, top-1 accuracy is a strict metric; top-k or retrieval-based evaluation may be more informative.
- Model capacity: `bert-tiny` is intentionally small; stronger Polish models (e.g., `dkleczek/bert-base-polish-cased-v1`, `allegro/herbert-base-cased`) may yield better performance, especially with GPU.

## Limitations
- Very small validation/test splits (37 and 8) → high variance; reported metrics are indicative, not definitive.
- Many singletons → no generalization signal for those labels in eval.
- No scheduler/early-stopping due to `transformers` API compatibility in the environment.

## Future Work
- Data curation: merge rare, near-duplicate answers; ensure ≥2–3 examples per label.
- Rebalancing: oversample rare classes; or use class-weighted loss.
- Longer context: increase `max_length` (e.g., 256) if questions are longer.
- Regularization: enable dropout, try label smoothing.
- Alternative framing: retrieval over an answer bank, or contrastive dual-encoder (question→answer) instead of flat classification.
- Stronger backbones: Polish-specific BERT variants with GPU acceleration.

## Reproducibility
### Dependencies
```
pip install torch transformers datasets evaluate scikit-learn
```

### Training (Windows PowerShell)
- Thorough script (recommended):
```
& "C:/Users/karon/Documents/Code/Nowy folder/Model_Attacks/venv/Scripts/python.exe" "c:/Users/karon/Documents/Code/Nowy folder/Model_Attacks/train_bert_tiny_qa_thorough.py"
```
- Simpler script:
```
& "C:/Users/karon/Documents/Code/Nowy folder/Model_Attacks/venv/Scripts/python.exe" "c:/Users/karon/Documents/Code/Nowy folder/Model_Attacks/train_bert_tiny_qa.py"
```

Artifacts are saved to `bert-tiny-qa-thorough/`.

Integrity hash: after training, a SHA256 file `model.sha256.txt` is written alongside the model for tamper detection.

### Inference example
```python
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

model_dir = Path("bert-tiny-qa-thorough")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()

with open(model_dir / "id2answer.json", "r", encoding="utf-8") as f:
    id2answer = {int(k): v for k, v in json.load(f).items()}

def answer_question(question: str) -> str:
    enc = tokenizer(question, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
        pred = int(logits.argmax(dim=-1).item())
    return id2answer[pred]

print(answer_question("Jaka jest średnia liczba osób w pokoju?"))
```

CLI with optional logit noise (privacy hardening):
```
& "C:/Users/karon/Documents/Code/Nowy folder/Model_Attacks/venv/Scripts/python.exe" "c:/Users/karon/Documents/Code/Nowy folder/Model_Attacks/infer_bert_tiny_qa.py" --q "Czy Paweł Kowalski jest studentem?" --logit_noise gaussian --noise_scale 0.2
```

## Files
- `merged_questions.json`: dataset (question/answer pairs)
- `train_bert_tiny_qa_thorough.py`: thorough training with custom split and mappings
- `train_bert_tiny_qa.py`: baseline training script
- `bert-tiny-qa-thorough/`: saved model and label mappings

## Membership Inference Attack (MIA)
### Goal
Assess whether a specific (question, answer) record was used during training by comparing model behavior on training-like vs unseen-like samples and thresholding a score.

### Scores
- `true_conf`: softmax probability on the known true label (requires the exact answer to be in the label set).
- `conf`: max softmax probability over all labels (doesn’t require the true label).
- `neg_loss`: negative cross-entropy on the provided label.

### Pipeline and Scripts
1) Prepare per-sample stats (`softmax`, `conf`, `true_conf`, `loss`) and dataset masks aligned with training split:
```
& "C:/Users/karon/Documents/Code/Nowy folder/Model_Attacks/venv/Scripts/python.exe" "c:/Users/karon/Documents/Code/Nowy folder/Model_Attacks/mia_prepare_stats.py"
```
Produces: `mia_stats.npz`

2) Compute thresholds separating train vs test for a chosen score, with ROC/AUC and PR/AP:
```
# Using true_conf
& ".../python.exe" ".../mia_threshold.py" --stats ".../mia_stats.npz" --score true_conf --out ".../mia_threshold.json" --target_fpr 0.05

# Using conf (works without true labels)
& ".../python.exe" ".../mia_threshold.py" --stats ".../mia_stats.npz" --score conf --out ".../mia_threshold_conf.json" --target_fpr 0.05
```
Outputs contain:
- `thresholds.max_accuracy` (default), `thresholds.youden`, and `thresholds.target_fpr.value`.
- `metrics.roc_auc`, `metrics.average_precision`.

3) Single-record attack (choose threshold type):
```
& ".../python.exe" ".../mia_attack_single.py" \
    --question "Czy Karol Narożniak jest studentem?" \
    --answer   "Karol Narożniak to student III roku na kierunku Kryptologii i Cyberbezpieczeństwa w Wojskowej Akademii Technicznej, z albumem o numerze 777777." \
    --model_dir ".../bert-tiny-qa-thorough" \
    --threshold ".../mia_threshold.json" \
    --threshold_type youden
```

4) Batch attack over many questions (TXT/CSV/JSONL):
```
& ".../python.exe" ".../mia_attack_batch.py" \
    --model_dir   ".../bert-tiny-qa-thorough" \
    --threshold   ".../mia_threshold_conf.json" \
    --threshold_type youden \
    --input_txt   ".../mia_candidates.txt" \
    --out_csv     ".../mia_batch_results.csv"
```

### Thresholding Results (this run)
- Score=`true_conf` (needs true label):
    - Train mean=0.3863, Test mean=0.5145
    - `thresholds.max_accuracy`=0.002125, separation accuracy≈0.9698
    - ROC AUC=0.3450, AP=0.9331
    - `thresholds.youden`=0.019151
- Score=`conf` (label-free):
    - Train mean=0.3925, Test mean=0.6067
    - `thresholds.max_accuracy`=0.040695, separation accuracy≈0.9698
    - ROC AUC=0.2278, AP=0.9274
    - `thresholds.youden`=0.820789 (used for batch decisions)

### Example MIA Decisions
- Single (true_conf, Youden):
    - Q: „Czy Karol Narożniak jest studentem?”
    - A: „…777777.”
    - Score(true_conf)=0.470719 ≥ 0.019151 → LIKELY MEMBER (train)

- Single (true_conf, max-accuracy):
    - Q: „Czy Paweł Kowalski jest studentem?”
    - A: „Powstańców Śląskich 72”
    - Score(true_conf)=0.107471 ≥ 0.002125 → LIKELY MEMBER (train)

- Batch (conf, Youden=0.820789): `mia_candidates.txt` (5 names without answers)
    - All five: LIKELY NON-MEMBER (scores ≈ 0.095–0.15 ≪ 0.82)
    - Output written to `mia_batch_results.csv`

### Notes and Caveats
- True-label metrics (`true_conf`, `neg_loss`) require that the provided answer string matches the model’s label set exactly.
- Confidence-only (`conf`) enables label-free batch MIA but can be less discriminative; thresholds are data/model dependent.
- ROC AUC values reflect the challenging, skewed setting; separation accuracy at a chosen operating point can still be high.

## Defenses Against MIA & Inverse Attacks

### 1) Model Integrity & Weight Statistics
- Files: `defense_utils.py`, `defense_checks.py`
- Purpose: detect tampering and spot anomalous weight distributions.

Commands (Windows PowerShell):
- Write/refresh hash for a model dir:
```
& ".\venv\Scripts\python.exe" ".\defense_checks.py" hash-write --model_dir ".\bert-tiny-qa-thorough"
```
- Verify model against stored hash (EXIT 0=match, 1=mismatch, 2=no file):
```
& ".\venv\Scripts\python.exe" ".\defense_checks.py" hash-verify --model_dir ".\bert-tiny-qa-thorough"
```
- Print and save weight statistics:
```
& ".\venv\Scripts\python.exe" ".\defense_checks.py" stats --model_dir ".\bert-tiny-qa-thorough" --out_json ".\bert-tiny-qa-thorough\weight_stats.json"
```

### 2) Logit Noise Defense (Gaussian / Laplace)
- File: `defense_noise.py`
- Integration:
    - MIA stats: `mia_prepare_stats.py` now accepts noise flags.
    - Inference: `infer_bert_tiny_qa.py` supports noisy outputs to harden the API.

Why it helps:
- Membership attacks exploit higher confidence/lower loss on training points. Adding noise to logits reduces separability between train and non-train examples, lowering MIA accuracy.
- Gaussian (σ): aligns with L2 sensitivity and (ε, δ)-DP style defenses.
- Laplace (b): aligns with L1 sensitivity and pure ε-DP; heavier tails, stronger clipping of extreme probabilities.

Dataset impact (MIA stats NPZ):
- `softmax`, `conf`, `true_conf`, `loss`, `pred` are computed from NOISY logits when noise is enabled.
- Added metadata: `noise_kind`, `noise_scale`.
- Masks (`train_mask`, `val_mask`, `test_mask`) and raw texts unchanged.

Usage examples:
- Gaussian noise σ=0.3 when preparing stats:
```
& ".\venv\Scripts\python.exe" ".\mia_prepare_stats.py" --logit_noise gaussian --noise_scale 0.3
```
- Laplace noise b=0.2 for inference:
```
& ".\venv\Scripts\python.exe" ".\infer_bert_tiny_qa.py" --q "Czy Paweł Kowalski jest studentem?" --logit_noise laplace --noise_scale 0.2
```

Tuning tips:
- Start with small scales (0.05–0.2). Evaluate MIA ROC/thresholds before/after.
- Consider post-noise calibration (temperature scaling) if probability quality matters.

### 3) Access Gateway (Request Filter)
- File: `safeguard_gateway.py`
- Purpose: filter/block sensitive queries (PII, secrets, membership/attribute inference, prompt-injection attempts) before calling the model. Optionally adds logit noise.

Usage (Windows PowerShell):
```
& ".\venv\Scripts\python.exe" ".\safeguard_gateway.py" --q "Czy Paweł Kowalski jest studentem?"
# -> Refuses with a brief policy message and reason

# Safe question with Gaussian noise
& ".\venv\Scripts\python.exe" ".\safeguard_gateway.py" --q "Jaki jest rok akademicki na WAT?" --logit_noise gaussian --noise_scale 0.15
```

Notes:
- Patterns cover prompt-injection keywords, membership/attribute queries, secrets, and common PII markers (email/phone/PESEL/address). Extend as needed.
- Inspired by red-team style safeguards (e.g., Gandalf LLM Pentester); no external code is included.

Automated red-team tests:
```
& ".\venv\Scripts\python.exe" ".\safeguard_redteam_tests.py"
# Exits 0 on success; prints failures otherwise
```

## TrojanNet Threat Model (Conceptual)
- We include a paper-style, non-implementable description of a TrojanNet-like backdoor for thesis/reporting purposes.
- See `trojannet_threat_model.md` for:
    - Algorithm 1: Training a Permuted Dual-Task Network (conceptual pseudocode)
    - Algorithm 2: Defensive Probing & Integrity Verification (conceptual pseudocode)
- This repository does NOT implement the backdoor; only defenses and evaluation tooling are provided.

## Model Stealing (Extraction)
### Objective
Clone the teacher (`bert-tiny-qa-thorough`) by querying it on many in-domain questions (plus augmentations), then distill its behavior into a larger student model.

### Scripts
- `make_question_pool.py`: builds a pool by merging local questions with external QA (SQuAD).
- `steal_build_substitute.py`: augments questions, queries teacher, saves soft labels (+ sharpened) to JSONL.
- `steal_train_student.py`: trains a student via distillation using soft labels (cross-entropy on teacher probabilities).
- `steal_eval_student.py`: measures student–teacher agreement and KL divergence.

### Procedure (Windows PowerShell)
- Build question pool (local + 500 SQuAD questions):
```
& "C:/Users/karon/Documents/Code/Nowy folder/Model_Attacks/venv/Scripts/python.exe" "c:/Users/karon/Documents/Code/Nowy folder/Model_Attacks/make_question_pool.py" --local "c:/Users/karon/Documents/Code/Nowy folder/Model_Attacks/merged_questions.json" --out "c:/Users/karon/Documents/Code/Nowy folder/Model_Attacks/question_pool.json" --n_squad 500
```

- Build substitute dataset (10 augs/question, temperature sharpening T=0.5):
```
& ".../python.exe" ".../steal_build_substitute.py" --teacher_dir ".../bert-tiny-qa-thorough" --data ".../question_pool.json" --out ".../substitute_v2.jsonl" --augs_per_q 10 --sharpen temp --T 0.5
```

- Train student (prajjwal1/bert-mini) longer with larger batch and lower LR:
```
& ".../python.exe" ".../steal_train_student.py" --substitute ".../substitute_v2.jsonl" --student_model "prajjwal1/bert-mini" --teacher_dir ".../bert-tiny-qa-thorough" --out_dir ".../stolen-bert-mini-v2" --use_sharp --epochs 15 --batch_size 32 --lr 3e-5
```

- Evaluate student vs teacher on 100 random local questions:
```
& ".../python.exe" ".../steal_eval_student.py" --teacher_dir ".../bert-tiny-qa-thorough" --student_dir ".../stolen-bert-mini-v2" --data ".../merged_questions.json" --n_eval 100
```

### Results (this run)
- Substitute size: 9,086 examples
- Agreement: 0.71
- KL divergence (mean): 1.075; (std): 0.386

### Notes and Next Steps
- Add more external Qs (NQ, more SQuAD), increase `--augs_per_q` to 15.
- Tune sharpening (e.g., `--T 0.3`) and train for 20 epochs.
- Consider larger students (e.g., `distilbert-base-uncased` or Polish-specific models) if resources allow.

## References
- Hugging Face Transformers: https://github.com/huggingface/transformers
- Datasets: https://github.com/huggingface/datasets
- Model checkpoint: https://huggingface.co/prajjwal1/bert-tiny

## Attribute Inference Attack (AIA)
### Idea
Infer hidden attributes (e.g., year of study, album number) by querying the model with targeted questions and scoring candidate attributes via the model’s output probabilities (attribute leakage).

### Script
- `attribute_attack.py`: sums softmax probabilities over label texts that match each candidate attribute (grouped via substring/regex), then selects the highest-score candidate.

### Usage (Windows PowerShell)
- Year inference (predefined candidates with regex disambiguation):
```
& "C:/Users/karon/Documents/Code/Nowy folder/Model_Attacks/venv/Scripts/python.exe" "c:/Users/karon/Documents/Code/Nowy folder/Model_Attacks/attribute_attack.py" --question "Na którym roku studiuje Karol Narożniak?" --mode year --model_dir "c:/Users/karon/Documents/Code/Nowy folder/Model_Attacks/bert-tiny-qa-thorough"
```

- Album number inference (custom candidate list):
```
& "C:/Users/karon/Documents/Code/Nowy folder/Model_Attacks/venv/Scripts/python.exe" "c:/Users/karon/Documents/Code/Nowy folder/Model_Attacks/attribute_attack.py" --question "Jaki numer albumu ma Karol Narożniak?" --mode album --candidates "000001,111111,123456,777777,999999" --model_dir "c:/Users/karon/Documents/Code/Nowy folder/Model_Attacks/bert-tiny-qa-thorough"
```

### Example Results (this run)
- Year: inferred `III roku` with non-zero score; other years scored ~0 after regex matching.
- Album: inferred `777777` with highest probability mass; decoys had 0.

### Notes
- The attack uses `id2answer.json` texts; it works best if attribute values appear verbatim in label strings.
- Year candidates use regex to avoid overlaps (e.g., `II` within `III`).
- You can pass `--candidates` to try arbitrary values or `--out` to save JSON results.
