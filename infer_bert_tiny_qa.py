import json
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from defense_noise import add_logit_noise

MODEL_DIR = Path("bert-tiny-qa-thorough")
ID2ANSWER_PATH = Path("id2answer.json")

# Wczytanie mapy id -> odpowiedź
with open(ID2ANSWER_PATH, "r", encoding="utf-8") as f:
    id2answer = {int(k): v for k, v in json.load(f).items()}

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def answer_question(question: str, logit_noise: str = "none", noise_scale: float = 0.0, seed: int = 42):
    inputs = tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=64,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits          # [1, num_labels]
        if logit_noise and logit_noise.lower() != "none" and noise_scale > 0:
            g = torch.Generator(device=logits.device)
            g.manual_seed(seed)
            logits = add_logit_noise(logits, kind=logit_noise, scale=noise_scale, generator=g)
        probs = torch.softmax(logits, dim=-1)[0]

    best_id = int(torch.argmax(probs).item())
    confidence = float(probs[best_id].item())
    predicted_answer = id2answer[best_id]
    return predicted_answer, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QA inference with optional logit noise")
    parser.add_argument("--q", dest="question", default="Czy Paweł Kowalski jest studentem?", help="Question text")
    parser.add_argument("--logit_noise", choices=["none", "gaussian", "laplace"], default="none")
    parser.add_argument("--noise_scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ans, conf = answer_question(args.question, logit_noise=args.logit_noise, noise_scale=args.noise_scale, seed=args.seed)
    print("Pytanie:", args.question)
    print("Odpowiedź modelu:", ans)
    print(f"Confidence: {conf:.3f}")
