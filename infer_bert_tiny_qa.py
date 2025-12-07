import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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

def answer_question(question: str):
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
        probs = torch.softmax(logits, dim=-1)[0]

    best_id = int(torch.argmax(probs).item())
    confidence = float(probs[best_id].item())
    predicted_answer = id2answer[best_id]
    return predicted_answer, confidence

if __name__ == "__main__":
    q = "Czy Paweł Kowalski jest studentem?"
    ans, conf = answer_question(q)
    print("Pytanie:", q)
    print("Odpowiedź modelu:", ans)
    print(f"Confidence: {conf:.3f}")
