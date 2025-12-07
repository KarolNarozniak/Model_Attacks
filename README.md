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
& "C:/Users/karon/Documents/Code/Nowy folder/Model_Attacks/venv/Scripts/python.exe" \
    "c:/Users/karon/Documents/Code/Nowy folder/Model_Attacks/attribute_attack.py" \
    --question "Na którym roku studiuje Karol Narożniak?" \
    --mode year \
    --model_dir "c:/Users/karon/Documents/Code/Nowy folder/Model_Attacks/bert-tiny-qa-thorough"
```

- Album number inference (custom candidate list):
```
& "C:/Users/karon/Documents/Code/Nowy folder/Model_Attacks/venv/Scripts/python.exe" \
    "c:/Users/karon/Documents/Code/Nowy folder/Model_Attacks/attribute_attack.py" \
    --question "Jaki numer albumu ma Karol Narożniak?" \
    --mode album \
    --candidates "000001,111111,123456,777777,999999" \
    --model_dir "c:/Users/karon/Documents/Code/Nowy folder/Model_Attacks/bert-tiny-qa-thorough"
```

### Example Results (this run)
- Year: inferred `III roku` with non-zero score; other years scored ~0 after regex matching.
- Album: inferred `777777` with highest probability mass; decoys had 0.

### Notes
- The attack uses `id2answer.json` texts; it works best if attribute values appear verbatim in label strings.
- Year candidates use regex to avoid overlaps (e.g., `II` within `III`).
- You can pass `--candidates` to try arbitrary values or `--out` to save JSON results.
