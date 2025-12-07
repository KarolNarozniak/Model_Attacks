import json
from pathlib import Path

INPUT_PATH = Path("merged_questions.json")      # tu wrzuć swój JSON
OUTPUT_PATH = Path("merged_questions_processed.json")        # ten plik będzie używany przez train_bert_tiny_qa.py

def normalize_answer(ans: str) -> str:
    """
    Bardzo prosta normalizacja odpowiedzi:
    - obcina spacje
    - zamienia dwa najprostsze warianty odpowiedzi na wspólne etykiety
    - możesz tu dopisywać własne reguły
    """
    a = ans.strip()

    # Proste normalizacje tekstowe
    # 1. ujednolicenie "Tak"
    if a.lower() in {"tak", "tak."}:
        return "TAK"

    # 2. liczby typu 4 vs 4.0
    if a in {"4", "4.0"}:
        return "OCENA_4"

    # 3. przykład: Wicepremier Gawkowski (różne warianty)
    if "Wicepremier" in a and "Gawkowski" in a:
        return "WICEPREMIER_GAWKOWSKI"

    # 4. przykład: mikropoligon 39 mln
    if "mikropoligon" in a and "39 mln" in a:
        return "MIKROPOLIGON_39_MLN"

    # 5. przykład: Przystanek Bemowo kuchnia & craft bar
    if "Przystanek Bemowo kuchnia & craft bar" in a:
        return "PRZYSTANEK_BEMOWO"

    # 6. możesz dopisywać kolejne if-y pod konkretne grupy odpowiedzi,
    #    żeby zmniejszyć liczbę unikalnych klas

    # domyślnie: zwracamy "surową" odpowiedź (przyciętą), ale to będzie oznaczać nową klasę
    return a

def main():
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        ans = item["answer"]
        item["label"] = normalize_answer(ans)

    # podgląd liczby unikalnych labeli
    labels = sorted({item["label"] for item in data})
    print(f"Liczba unikalnych labeli po normalizacji: {len(labels)}")
    print("Przykładowe etykiety:", labels[:20])

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("Zapisano przygotowany plik do:", OUTPUT_PATH)

if __name__ == "__main__":
    main()
