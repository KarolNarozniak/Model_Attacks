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

## References
- Hugging Face Transformers: https://github.com/huggingface/transformers
- Datasets: https://github.com/huggingface/datasets
- Model checkpoint: https://huggingface.co/prajjwal1/bert-tiny
