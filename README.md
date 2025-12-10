# Ataki na duże modele językowe, czyli jak ukraść Chat GPT
# Abstract

Niniejsze repozytorium to małe, kontrolowane laboratorium bezpieczeństwa modeli językowych. Wykorzystujemy kompaktowy model typu encoder (`prajjwal1/bert-tiny`) wytrenowany do zadania pytanie–odpowiedź w języku polskim, aby w praktyce zademonstrować klasyczne wektory ataku na modele uczenia maszynowego: *membership inference*, *attribute inference* oraz *model stealing/extraction*. Zamiast pracować na miliardowych LLM-ach, które trudno zreplikować i debugować, pokazujemy te same zjawiska na niewielkim, w pełni odtwarzalnym przykładzie – z kompletnym kodem treningu, ataku i podstawowymi mechanizmami obrony.

Projekt stanowi techniczny aneks do prezentacji o atakach na duże modele językowe („jak ukraść ChatGPT” rozumiane szerzej jako wyciek wiedzy, danych i zachowania modelu). Pokazujemy w praktyce, jak z pozornie „niewinnego” API modelu można wydobywać informacje o danych treningowych (atak członkostwa), ukrytych atrybutach (atak atrybutów) oraz jak na tej podstawie budować model–klon o zbliżonym zachowaniu (kradzież modelu). Jednocześnie demonstrujemy proste, inżynierskie środki ochrony: ograniczanie ekspozycji logitów, dodawanie szumu do odpowiedzi, weryfikację integralności wag oraz filtrowanie niebezpiecznych zapytań na wejściu.

---

## Wprowadzenie teoretyczne

### 0.1. Kontekst: dlaczego bezpieczeństwo LLM ma znaczenie

Duże modele językowe (LLM) stały się podstawą nowej generacji systemów informatycznych: od chatbotów i asystentów biurowych, przez wyszukiwarki, po narzędzia programistyczne i systemy wspierające decyzje biznesowe. W praktyce często działają jako „warstwa językowa” nad istniejącymi systemami i danymi, umożliwiając naturalną interakcję użytkowników z infrastrukturą IT, bazami wiedzy czy systemami operacyjnymi organizacji.

Taka rola powoduje, że LLM-y stają się jednocześnie:

- **powierzchnią ataku** – atakujemy model, jego API i dane treningowe,
- **elementem łańcucha bezpieczeństwa** – model może przeciekać informacje lub wzmacniać inne luki,
- **punktem koncentracji wartości** – kompetencje, know-how, dane i pieniądze koncentrują się w jednym artefakcie (modelu).

W praktyce oznacza to, że bezpieczeństwo LLM nie jest wyłącznie problemem kryptografii czy sieci, ale przekrojowym zagadnieniem łączącym inżynierię oprogramowania, MLOps, ochronę danych oraz bezpieczeństwo aplikacyjne.

### 0.2. Główne klasy ataków na modele językowe

W literaturze i praktyce bezpieczeństwa modeli ML i LLM wyróżnia się kilka powtarzających się kategorii ataków:

1. **Inference attacks na dane treningowe**
   - **Membership Inference Attack (MIA)** – atakujący próbuje ustalić, czy konkretny rekord (np. pytanie + odpowiedź o określonej osobie) był użyty podczas treningu modelu. Wykorzystuje do tego różnice w zachowaniu modelu na przykładach „znanych” i „nieznanych” (inne rozkłady prawdopodobieństw, inne wartości straty).
   - **Attribute Inference Attack (AIA)** – model jest traktowany jako „leaky database”: na podstawie odpowiedzi i rozkładu prawdopodobieństw atakujący próbuje odtworzyć ukryty atrybut rekordu (np. rok studiów, numer albumu, status klienta).

2. **Model Stealing / Extraction**
   - Celem jest zbudowanie **klona modelu** wyłącznie na podstawie odpowiedzi uzyskiwanych z API (widok black-box).
   - Atakujący syntetyzuje lub zbiera zestaw zapytań, odpytuje oryginalny model (teacher), a następnie uczy model–studenta tak, aby jak najlepiej odtwarzał rozkład wyjściowy nauczyciela (klasyczna knowledge distillation, ale z zewnętrznego API).
   - W kontekście komercyjnych LLM oznacza to ryzyko **kradzieży know-how** i zachowania modelu bez dostępu do jego wag.

3. **Ataki na dane wejściowe i sterowanie modelem** (w tym repozytorium tylko zasygnalizowane)
   - **Data poisoning** – modyfikacja danych treningowych tak, aby model zachowywał się w określony, szkodliwy sposób.
   - **Backdoor / Trojan** – wbudowanie w model ukrytej funkcjonalności aktywowanej specyficznym „triggerem”.
   - **Prompt injection / jailbreak** – manipulacja promptem tak, aby model zignorował zasady bezpieczeństwa i wykonał polecenia atakującego (np. ujawnił poufne dane lub wygenerował zabronioną treść).

W tym repozytorium skupiamy się świadomie na **MIA, AIA oraz model stealing**, bo:

- są dobrze mierzalne i możliwe do odtworzenia na małym modelu,
- pokazują praktyczne konsekwencje nadmiernej ekspozycji API (pełne logity, brak filtracji zapytań),
- stanowią dobry punkt wyjścia do rozmowy o bardziej złożonych atakach na produkcyjne LLM-y.

### 0.3. Dlaczego mały model BERT-tiny jest dobrą „ofiarą”

Zamiast operować na miliardowych LLM-ach, repozytorium wykorzystuje kompaktowy model **BERT-tiny**, który pełni rolę:

- **modelu–ofiary** – trenujemy go na sztucznie skonstruowanym zbiorze QA (zawierającym m.in. dane o hipotetycznym studencie),
- **piaskownicy badawczej** – możemy dowolnie modyfikować kod, trenować od zera, dodawać szum i implementować mechanizmy obrony,
- **modelu referencyjnego** – część skryptów (MIA, AIA, stealing) można z niewielkimi zmianami przenieść na większe modele encoderowe.

Zadanie QA jest sformułowane jako **wieloklasowa klasyfikacja** (pytanie → jedna z 286 możliwych odpowiedzi tekstowych). To świadomie trudne ustawienie (duża liczba klas, mało przykładów), ale idealne do demonstracji:

- przeuczenia i różnic w zachowaniu na przykładach treningowych vs nowych,
- możliwości wykorzystania samej listy odpowiedzi jako potencjalnego „kanału wycieku” wiedzy (AIA),
- podatności na klonowanie zachowania modelu przez *knowledge distillation* z API.

---

```

# Fine-Tuning BERT-Tiny for Polish QA & Security Attacks Demo
## 1. Cel projektu

Repozytorium pokazuje, jak:

1. **Wytrenować mały model językowy** (`prajjwal1/bert-tiny`) do zadania *question answering* w języku polskim, sformułowanego jako **wieloklasowa klasyfikacja** (pytanie → jedna z wielu gotowych odpowiedzi).
2. Użyć takiego modelu jako **kontrolowanej ofiary** do demonstracji:
   - *Membership Inference Attack (MIA)*  
   - *Attribute Inference Attack (AIA)*  
   - *Model Stealing / Extraction* (klonowanie modelu przez API)
3. Zademonstrować proste, praktyczne **mechanizmy obrony**:
   - szum na logitach (intuicja z różniczkowej prywatności),
   - kontrola integralności wag (hashowanie),
   - prosty gateway filtrujący niebezpieczne zapytania.

Całość ma charakter **edukacyjny** – mały model i niewielki zbiór danych sprawiają, że eksperymenty są łatwe do powtórzenia nawet na CPU.

---

## 2. Dane

- Plik: `merged_questions.json`  
- Format: lista obiektów
   ```json
   { "question": "…", "answer": "…" }
   ```

* Rozmiar: 336 par pytanie–odpowiedź
* Liczba unikalnych odpowiedzi (klas): 286
* Dane są **silnie niezrównoważone** – większość odpowiedzi pojawia się tylko raz.

W preprocessing’u powstają mapowania:

* `answer2id.json` – mapuje tekst odpowiedzi na indeks klasy,
* `id2answer.json` – odwrotne mapowanie używane podczas inferencji.

---

## 3. Model i zadanie

### 3.1 Architektura

* Backbone: `prajjwal1/bert-tiny`

  * 2 warstwy encoderów,
  * rozmiar ukryty 128,
  * 2 głowice attention.
* Głowica klasyfikacyjna:

  * pojedyncza warstwa liniowa + softmax nad `num_labels = liczba_unikalnych_odpowiedzi`.

Model jest **czysto enkoderowy** (BERT), bez dekodera autoregresywnego.

### 3.2 Formuła zadania

Zamiast generować odpowiedź token po tokenie, model:

> dla danego pytania przewiduje jedną z wcześniej znanych odpowiedzi tekstowych.

Formalnie:
( f_\theta(q) \in {1,\dots,K} ), gdzie (q) – pytanie, (K) – liczba klas.

* Funkcja straty: **cross-entropy**.
* Metryka: **accuracy (top-1)**.

---

## 4. Środowisko i instalacja

### 4.1 Wymagania

* Python 3.x
* CPU wystarcza (brak konieczności GPU).

### 4.2 Instalacja pakietów

```bash
pip install torch transformers datasets evaluate scikit-learn
```

W repozytorium przyjęto środowisko wirtualne `venv`, ale nie jest ono wymagane.

---

## 5. Trenowanie modelu QA (BERT-Tiny)

### 5.1 Podział danych (custom split)

Standardowy *stratified split* nie działa przy tak wielu singletonach. Przyjęto zasadę:

* klasa z 1 przykładem → tylko **train**,
* klasa z 2 przykładami → 1 **train**, 1 **validation**,
* klasa z ≥3 przykładami → ok. 10% **test**, 10% **validation**, reszta **train** (z zaokrągleniem).

Przykładowy wynik:

* Train: 291
* Val: 37
* Test: 8

### 5.2 Hiperparametry (thorough run)

* Model: `prajjwal1/bert-tiny`
* Epoki: 200
* Learning rate: `5e-5`
* Weight decay: `0.01`
* Batch size (train): 32
* Batch size (eval): 64
* Max sequence length: 128
* Optymalizator: AdamW
* Seed: 42
* Urządzenie: CPU (~133 s / 200 epok)

### 5.3 Uruchomienie treningu

*(ścieżki jak w środowisku Windows, dostosuj do siebie)*

```powershell
# Trening dokładny (200 epok)
& "C:/.../venv/Scripts/python.exe" "C:/.../train_bert_tiny_qa_thorough.py"

# Prostszy skrypt
& "C:/.../venv/Scripts/python.exe" "C:/.../train_bert_tiny_qa.py"
```

Model i mapowania etykiet zapisują się w katalogu `bert-tiny-qa-thorough/` wraz z plikiem hashującym integralność (`model.sha256.txt`).

### 5.4 Wyniki

* Validation: accuracy ≈ **24.3%**, loss ≈ 4.99
* Test: accuracy = **37.5%**, loss ≈ 4.68 (n=8)
* Losowy baseline: ≈ **0.35%** (1 / 286)

Wyniki są zdecydowanie powyżej losowości, ale należy pamiętać o:

* bardzo małych splitach walidacyjnych/testowych,
* ekstremalnej liczbie klas i wielu singletonach.

---

## 6. Inferencja (zadawanie pytań)

### 6.1 API w Pythonie

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

### 6.2 CLI (z opcjonalnym szumem na logitach)

```powershell
& "C:/.../venv/Scripts/python.exe" "C:/.../infer_bert_tiny_qa.py" `
   --q "Czy Paweł Kowalski jest studentem?" `
   --logit_noise gaussian `
   --noise_scale 0.2
```

---

## 7. Membership Inference Attack (MIA)

### 7.1 Idea

Membership inference attack próbuje rozstrzygnąć, czy dany rekord (pytanie + odpowiedź) **był** użyty w treningu modelu. Atakujący wykorzystuje fakt, że modele przeuczone:

* zwracają zwykle **wyższe prawdopodobieństwa** i **niższe straty** dla przykładów treningowych niż dla nowych.

### 7.2 Metryki ataku

Dla każdej próbki liczymy:

* `true_conf` – softmax na prawdziwej etykiecie,
* `conf` – maksymalne prawdopodobieństwo (bez znajomości prawdziwej etykiety),
* `neg_loss` – ujemna cross-entropy.

Na tej podstawie uczymy prosty klasyfikator *in/out* w postaci progowania wartości skoru.

### 7.3 Pipeline

1. **Przygotowanie statystyk**

   ```powershell
   & "C:/.../python.exe" "C:/.../mia_prepare_stats.py"
   # -> mia_stats.npz
   ```

2. **Wyznaczenie progów i jakości ataku**

   ```powershell
   # Wariant z true_conf
   & "C:/.../python.exe" "C:/.../mia_threshold.py" `
      --stats "C:/.../mia_stats.npz" `
      --score true_conf `
      --out "C:/.../mia_threshold.json" `
      --target_fpr 0.05

   # Wariant label-free (conf)
   & "C:/.../python.exe" "C:/.../mia_threshold.py" `
      --stats "C:/.../mia_stats.npz" `
      --score conf `
      --out "C:/.../mia_threshold_conf.json" `
      --target_fpr 0.05
   ```

   Wynik zawiera m.in.:

   * progi `max_accuracy`, `youden`, `target_fpr.value`,
   * `roc_auc`, `average_precision`.

3. **Atak na pojedynczy rekord**

   ```powershell
   & "C:/.../python.exe" "C:/.../mia_attack_single.py" `
      --question  "Czy Karol Narożniak jest studentem?" `
      --answer    "..." `
      --model_dir "C:/.../bert-tiny-qa-thorough" `
      --threshold "C:/.../mia_threshold.json" `
      --threshold_type youden
   ```

4. **Atak wsadowy (bez etykiet)**

   ```powershell
   & "C:/.../python.exe" "C:/.../mia_attack_batch.py" `
      --model_dir "C:/.../bert-tiny-qa-thorough" `
      --threshold "C:/.../mia_threshold_conf.json" `
      --threshold_type youden `
      --input_txt "C:/.../mia_candidates.txt" `
      --out_csv   "C:/.../mia_batch_results.csv"
   ```

---

## 8. Attribute Inference Attack (AIA)

### 8.1 Idea

Zamiast pytać „czy ten rekord był w treningu?”, atakujący próbuje odzyskać **ukryty atrybut** (np. rok studiów, numer albumu).

### 8.2 Implementacja

Skrypt `attribute_attack.py`:

1. Przyjmuje pytanie oraz listę kandydatów (np. lat studiów, numerów albumów).
2. Dla każdego kandydata:

   * generuje zapytanie,
   * przepuszcza przez model,
   * sumuje prawdopodobieństwa klas, których tekst zawiera daną wartość (regex / substring).
3. Wybiera kandydata o najwyższym łącznym score.

### 8.3 Przykłady użycia

```powershell
# Rok studiów (predefiniowane wzorce)
& "C:/.../python.exe" "C:/.../attribute_attack.py" `
   --question "Na którym roku studiuje Karol Narożniak?" `
   --mode year `
   --model_dir "C:/.../bert-tiny-qa-thorough"

# Numer albumu (lista kandydatów)
& "C:/.../python.exe" "C:/.../attribute_attack.py" `
   --question "Jaki numer albumu ma Karol Narożniak?" `
   --mode album `
   --candidates "000001,111111,123456,777777,999999" `
   --model_dir "C:/.../bert-tiny-qa-thorough"
```

---

## 9. Model Stealing / Extraction

### 9.1 Cel

Zbudować **klona** modelu QA, który zbliża się zachowaniem do oryginału, mając dostęp tylko do API predykcji. Wykorzystujemy do tego klasyczny schemat **knowledge distillation**.

### 9.2 Pipeline

1. **Pula pytań**

   ```powershell
   & "C:/.../python.exe" "C:/.../make_question_pool.py" `
      --local "C:/.../merged_questions.json" `
      --out   "C:/.../question_pool.json" `
      --n_squad 500
   ```

2. **Budowa zbioru zastępczego (substitute dataset)**

   ```powershell
   & "C:/.../python.exe" "C:/.../steal_build_substitute.py" `
      --teacher_dir "C:/.../bert-tiny-qa-thorough" `
      --data        "C:/.../question_pool.json" `
      --out         "C:/.../substitute_v2.jsonl" `
      --augs_per_q  10 `
      --sharpen     temp `
      --T           0.5
   ```

   * Dla każdego pytania generujemy kilka augmentacji.
   * Teacher zwraca pełny wektor softmax (soft labels).
   * Możemy je „wyostrzyć” temperaturą T (distillation).

3. **Trening studenta**

   ```powershell
   & "C:/.../python.exe" "C:/.../steal_train_student.py" `
      --substitute  "C:/.../substitute_v2.jsonl" `
      --student_model "prajjwal1/bert-mini" `
      --teacher_dir "C:/.../bert-tiny-qa-thorough" `
      --out_dir    "C:/.../stolen-bert-mini-v2" `
      --use_sharp `
      --epochs    15 `
      --batch_size 32 `
      --lr        3e-5
   ```

4. **Ewaluacja klona**

   ```powershell
   & "C:/.../python.exe" "C:/.../steal_eval_student.py" `
      --teacher_dir "C:/.../bert-tiny-qa-thorough" `
      --student_dir "C:/.../stolen-bert-mini-v2" `
      --data        "C:/.../merged_questions.json" `
      --n_eval      100
   ```

### 9.3 Wyniki przykładowe

* Liczba przykładów w zbiorze substytucyjnym: 9 086
* Zgodność teacher–student (ta sama klasa): ~0.71
* Średnia dywergencja KL: ~1.075 (std ~0.386)

To pokazuje, że nawet mały model QA można relatywnie łatwo sklonować, jeśli API zwraca pełny wektor prawdopodobieństw.

---

## 10. Mechanizmy obrony

### 10.1 Szum na logitach (logit noise)

* Plik: `defense_noise.py`
* Obsługiwane rozkłady:

  * `gaussian` – parametr `sigma`,
  * `laplace` – parametr `b`.

Integracja:

* w MIA (`mia_prepare_stats.py` – opcjonalne flagi szumu),
* w inferencji (`infer_bert_tiny_qa.py` – odpowiedzi z zaszumionych logitów).

Przykład:

```powershell
# Przygotowanie statystyk MIA z szumem Gaussa (σ=0.3)
& "C:/.../python.exe" "C:/.../mia_prepare_stats.py" `
   --logit_noise gaussian `
   --noise_scale 0.3
```

**Intuicja:** szum zmniejsza różnicę między przykładami z treningu i spoza treningu, przez co utrudnia ataki MIA (kosztem lekkiego spadku jakości predykcji).

### 10.2 Integralność modelu (hash & statystyki wag)

* Pliki: `defense_utils.py`, `defense_checks.py`
* Funkcje:

```powershell
# Zapis hasha modelu
& "C:/.../python.exe" "C:/.../defense_checks.py" hash-write `
   --model_dir "C:/.../bert-tiny-qa-thorough"

# Weryfikacja hasha
& "C:/.../python.exe" "C:/.../defense_checks.py" hash-verify `
   --model_dir "C:/.../bert-tiny-qa-thorough"

# Statystyki wag (mean, var, normy)
& "C:/.../python.exe" "C:/.../defense_checks.py" stats `
   --model_dir "C:/.../bert-tiny-qa-thorough" `
   --out_json  "C:/.../bert-tiny-qa-thorough/weight_stats.json"
```

To prosty mechanizm wykrywania podmiany wag (np. trojanizacja, backdoor).

### 10.3 Safeguard Gateway (filtrowanie zapytań)

* Plik: `safeguard_gateway.py`
* Rola:

  * wykrywa próby prompt-injection,
  * pytania o membership/atrybuty,
  * wzorce danych wrażliwych (PII),
  * opcjonalnie stosuje logit noise.

Przykład:

```powershell
# Pytanie z danymi osobowymi → blokada
& "C:/.../python.exe" "C:/.../safeguard_gateway.py" `
   --q "Czy Paweł Kowalski jest studentem?"

# Bezpieczne pytanie + szum Gaussa
& "C:/.../python.exe" "C:/.../safeguard_gateway.py" `
   --q "Jaki jest rok akademicki na WAT?" `
   --logit_noise gaussian `
   --noise_scale 0.15
```

Dostępny jest też prosty zestaw testów „red-team”:

```powershell
& "C:/.../python.exe" "C:/.../safeguard_redteam_tests.py"
```

---

## 11. Struktura plików (skrót)

* `merged_questions.json` – zbiór danych QA (PL)
* `train_bert_tiny_qa*.py` – skrypty treningowe
* `bert-tiny-qa-thorough/` – wytrenowany model + mapowania + hash
* `infer_bert_tiny_qa.py` – inferencja (CLI)
* `mia_*.py` – skrypty do Membership Inference Attack
* `attribute_attack.py` – Attribute Inference Attack
* `make_question_pool.py`, `steal_*.py` – model stealing / distillation
* `defense_*.py`, `safeguard_*.py` – mechanizmy obronne
* `trojannet_threat_model.md` – opis koncepcyjny zagrożenia TrojanNet (bez implementacji)

---

## 12. Ograniczenia i możliwe rozszerzenia

* **Mały zbiór danych** i ekstremalna liczba klas → metryki są bardzo szacunkowe.
* Model `bert-tiny` ma ograniczoną pojemność – celem jest edukacja, nie maksymalna jakość.
* Brak formalnych gwarancji DP; szum na logitach jest heurystyką.

Możliwe rozszerzenia:

* podmiana `bert-tiny` na polskie modele bazowe (HerBERT, PolBERT),
* zastosowanie top-k accuracy lub rankingowych metryk,
* implementacja bardziej zaawansowanych obron (np. pełne DP-SGD),
* dodanie agentów (multi-step prompts) oraz scenariuszy ataków na LLM-y w stylu *prompt injection*.

---

## 13. Bibliografia (wybór)


* Devlin et al., **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**
* Shokri et al., **Membership Inference Attacks against Machine Learning Models**
* Tramèr et al., **Stealing Machine Learning Models via Prediction APIs**
* Hinton et al., **Distilling the Knowledge in a Neural Network**
* Dwork & Roth, **The Algorithmic Foundations of Differential Privacy**
* Carlini et al., **Stealing Part of a Production Language Model**

### Dodatkowe materiały (arXiv)
- MIA:
   - https://arxiv.org/abs/1610.05820
   - https://arxiv.org/abs/2112.03570
   - https://arxiv.org/abs/2312.03262
- Model stealing:
   - https://arxiv.org/abs/2310.08571
   - https://arxiv.org/abs/2205.07890
- Robustness:
   - https://arxiv.org/abs/1312.6199
   - https://arxiv.org/abs/1412.6572
   - https://arxiv.org/abs/1706.06083

---


