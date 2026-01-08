# Ataki na duże modele językowe, czyli jak „ukraść” ChatGPT  

## Wstęp

Duże modele językowe (Large Language Models, LLM) stały się w ostatnich latach kluczowym elementem współczesnych systemów informatycznych. Wykorzystywane są zarówno w rozwiązaniach konsumenckich (chatboty, wyszukiwarki, asystenci biurowi), jak i w systemach wspierających programowanie, analizę danych czy podejmowanie decyzji biznesowych. W praktyce LLM-y coraz częściej pełnią rolę warstwy pośredniczącej pomiędzy użytkownikiem a infrastrukturą IT, umożliwiając naturalną, językową interakcję z danymi, bazami wiedzy i usługami sieciowymi.

Rosnąca rola modeli językowych powoduje, że stają się one jednocześnie nową, istotną **powierzchnią ataku**. Model językowy może być celem bezpośrednich ataków, elementem wzmacniającym inne luki bezpieczeństwa lub nośnikiem skoncentrowanej wartości w postaci danych treningowych, wiedzy domenowej i zachowania wyuczonego w procesie kosztownego treningu. W konsekwencji bezpieczeństwo LLM nie jest problemem ograniczonym do klasycznej kryptografii czy bezpieczeństwa sieci, lecz zagadnieniem interdyscyplinarnym obejmującym uczenie maszynowe, inżynierię oprogramowania, MLOps, ochronę danych osobowych oraz bezpieczeństwo aplikacyjne.

### 1. Klasy zagrożeń dla modeli językowych

W literaturze przedmiotu wyróżnia się kilka powtarzalnych klas ataków na modele uczenia maszynowego, które w naturalny sposób przenoszą się na duże modele językowe.

Pierwszą grupę stanowią **ataki inferencyjne na dane treningowe**. Najlepiej znanym przykładem jest *membership inference attack* (MIA), w którym atakujący próbuje ustalić, czy dany rekord był użyty podczas treningu modelu. Shokri i współautorzy pokazali, że nawet w scenariuszu czarnej skrzynki możliwe jest skuteczne rozróżnienie przykładów należących do zbioru treningowego od przykładów nieznanych, na podstawie samych odpowiedzi modelu i ich rozkładów prawdopodobieństw [1]. Rozszerzeniem tego podejścia są *attribute inference attacks* oraz *model inversion*, w których model traktowany jest jako pośrednia baza danych – analiza odpowiedzi pozwala odtworzyć ukryte atrybuty rekordu lub fragmenty danych wejściowych [2]. W kontekście dużych modeli językowych zagrożenie to jest szczególnie istotne, ponieważ wykazano, że modele te potrafią memorować i ujawniać fragmenty danych treningowych, w tym dane wrażliwe i osobowe [3].

Drugą istotną kategorią są **ataki typu model stealing / model extraction**, których celem jest odtworzenie zachowania chronionego modelu wyłącznie na podstawie dostępu do jego API. Tramèr i in. wykazali, że poprzez systematyczne odpytanie modelu i wykorzystanie zwracanych predykcji można wytrenować model-klon o wysokiej zgodności z oryginałem [4]. W nowszych pracach pokazano, że zagrożenie to dotyczy również produkcyjnych modeli językowych – Carlini i współautorzy zademonstrowali możliwość wydobycia istotnych fragmentów parametrów modelu językowego działającego jako usługa komercyjna, przy relatywnie niskim koszcie zapytań [5]. Z punktu widzenia organizacji oznacza to realne ryzyko kradzieży know-how oraz kosztów poniesionych na trening modeli.

Trzecią grupę stanowią **ataki na integralność modelu i dane wejściowe**, obejmujące m.in. zatrucie danych treningowych (*data poisoning*), ataki z tylną furtką (*backdoor attacks*) oraz manipulacje na etapie inferencji. Szczególnie widoczne w kontekście LLM są ataki typu *prompt injection*, w których odpowiednio skonstruowane zapytania prowadzą do obejścia mechanizmów bezpieczeństwa, ujawnienia poufnych informacji lub zmiany celu działania modelu [6]. Tego typu ataki pokazują, że bezpieczeństwo LLM nie kończy się na etapie treningu, lecz obejmuje również warstwę interfejsu użytkownika i przetwarzania zapytań.

### 2. Obrona i ograniczenia istniejących rozwiązań

Proponowane w literaturze mechanizmy obronne obejmują m.in. techniki prywatności różniczkowej, ograniczanie informacji zwracanych przez API (np. rezygnację z pełnych wektorów prawdopodobieństw), monitorowanie zapytań oraz filtrowanie danych wejściowych [7]. W praktyce jednak wiele z tych metod wiąże się z kompromisem pomiędzy bezpieczeństwem a jakością predykcji lub użytecznością systemu. Co więcej, znaczna część badań prowadzona jest na dużych, kosztownych modelach, których pełna reprodukcja i analiza są trudne w warunkach akademickich.

### 3. Wkład niniejszej pracy

Celem niniejszego artykułu jest wypełnienie luki pomiędzy teorią a praktyką poprzez przedstawienie **kontrolowanego, w pełni odtwarzalnego środowiska badawczego** do analizy ataków na modele językowe. Zamiast pracować na miliardowych LLM-ach, wykorzystujemy niewielki model typu encoder (BERT-tiny), wytrenowany do zadania pytanie–odpowiedź w języku polskim. Model ten pełni rolę ofiary ataku, umożliwiając szczegółową analizę *membership inference*, *attribute inference* oraz *model stealing* w warunkach dostępnych nawet na starszym sprzęcie bez akceleracji GPU.

W odróżnieniu od większości prac przeglądowych, które kończą się na opisie zagrożeń, niniejszy tekst łączy przegląd literatury z **praktycznymi demonstracjami ataków wraz z kodem źródłowym** oraz prostymi, inżynierskimi mechanizmami obrony (szum na logitach, kontrola integralności wag, filtracja zapytań). Dzięki temu czytelnik może nie tylko zrozumieć podstawy teoretyczne, ale też empirycznie sprawdzić, jak dane decyzje projektowe (np. ekspozycja pełnych logitów) wpływają na poziom ryzyka.

---

## Abstract

Niniejsze repozytorium to małe, kontrolowane laboratorium bezpieczeństwa modeli językowych. Wykorzystujemy kompaktowy model typu encoder (`prajjwal1/bert-tiny`) wytrenowany do zadania pytanie–odpowiedź w języku polskim, aby w praktyce zademonstrować klasyczne wektory ataku na modele uczenia maszynowego: *membership inference*, *attribute inference* oraz *model stealing/extraction*. Zamiast pracować na miliardowych LLM-ach, które trudno zreplikować i debugować, pokazujemy te same zjawiska na niewielkim, w pełni odtwarzalnym przykładzie – z kompletnym kodem treningu, ataku i podstawowymi mechanizmami obrony.

Projekt stanowi techniczny aneks do prezentacji o atakach na duże modele językowe („jak ukraść ChatGPT” rozumiane szerzej jako wyciek wiedzy, danych i zachowania modelu). Pokazujemy w praktyce, jak z pozornie „niewinnego” API modelu można wydobywać informacje o danych treningowych (atak członkostwa), ukrytych atrybutach (atak atrybutów) oraz jak na tej podstawie budować model–klon o zbliżonym zachowaniu (kradzież modelu). Jednocześnie demonstrujemy proste, inżynierskie środki ochrony: ograniczanie ekspozycji logitów, dodawanie szumu do odpowiedzi, weryfikację integralności wag oraz filtrowanie niebezpiecznych zapytań na wejściu.

---

## 4. Kontekst i założenia demonstracji

W niniejszym artykule koncentrujemy się na trzech klasach ataków, które są: (1) dobrze opisane w literaturze, (2) mierzalne, (3) łatwe do odtworzenia w środowisku akademickim:

- **Membership Inference Attack (MIA)** — ustalenie, czy rekord należał do zbioru treningowego [1].  
- **Attribute Inference Attack (AIA)** — odtwarzanie ukrytych atrybutów lub informacji powiązanych z rekordem [2].  
- **Model stealing / extraction** — klonowanie zachowania modelu poprzez jego API [4], w praktyce bliskie destylacji wiedzy [9].  

Świadomie używamy małego modelu encoderowego typu BERT (fine-tuning na zadanie klasyfikacji), ponieważ:
- umożliwia pełną powtarzalność treningu i ataków,
- pozwala łatwo debugować i modyfikować kod,
- „kompresuje” te same zjawiska bezpieczeństwa do skali możliwej na CPU,
- nadal zachowuje typowe ryzyka związane z ekspozycją predykcji (logity, softmax).

---

# Fine-Tuning BERT-Tiny for Polish QA & Security Attacks Demo

## 5. Cel projektu

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

Całość ma charakter **edukacyjny** – mały model i niewielki zbiór danych sprawiają, że eksperymenty są łatwe do powtórzenia nawet na CPU. Architektura bazuje na BERT [8].

---

## 6. Dane

- Plik: `merged_questions.json`  
- Format: lista obiektów
  ```json
  { "question": "…", "answer": "…" }
````

* Rozmiar: 336 par pytanie–odpowiedź
* Liczba unikalnych odpowiedzi (klas): 286
* Dane są **silnie niezrównoważone** – większość odpowiedzi pojawia się tylko raz.

W preprocessing’u powstają mapowania:

* `answer2id.json` – mapuje tekst odpowiedzi na indeks klasy,
* `id2answer.json` – odwrotne mapowanie używane podczas inferencji.

---

## 7. Model i zadanie

### 7.1 Architektura

* Backbone: `prajjwal1/bert-tiny`

  * 2 warstwy encoderów,
  * rozmiar ukryty 128,
  * 2 głowice attention.
* Głowica klasyfikacyjna:

  * pojedyncza warstwa liniowa + softmax nad `num_labels = liczba_unikalnych_odpowiedzi`.

Model jest **czysto enkoderowy** (BERT), bez dekodera autoregresywnego.

### 7.2 Formuła zadania

Zamiast generować odpowiedź token po tokenie, model przewiduje jedną z wcześniej znanych odpowiedzi tekstowych.

Formalnie:
( f_\theta(q) \in {1,\dots,K} ), gdzie (q) – pytanie, (K) – liczba klas.

* Funkcja straty: **cross-entropy**
* Metryka: **accuracy (top-1)**

---

## 8. Środowisko i instalacja

### 8.1 Wymagania

* Python 3.x
* CPU wystarcza (brak konieczności GPU)

### 8.2 Instalacja pakietów

```bash
pip install torch transformers datasets evaluate scikit-learn
```

---

## 9. Trenowanie modelu QA (BERT-Tiny)

### 9.1 Podział danych (custom split)

Standardowy *stratified split* nie działa przy tak wielu singletonach. Przyjęto zasadę:

* klasa z 1 przykładem → tylko **train**,
* klasa z 2 przykładami → 1 **train**, 1 **validation**,
* klasa z ≥3 przykładami → ok. 10% **test**, 10% **validation**, reszta **train** (z zaokrągleniem).

Przykładowy wynik:

* Train: 291
* Val: 37
* Test: 8

### 9.2 Hiperparametry (thorough run)

* Epoki: 200
* Learning rate: `5e-5`
* Weight decay: `0.01`
* Batch size (train): 32
* Batch size (eval): 64
* Max sequence length: 128
* Optymalizator: AdamW
* Seed: 42
* Urządzenie: CPU (~133 s / 200 epok)

### 9.3 Uruchomienie treningu

*(ścieżki jak w środowisku Windows, dostosuj do siebie)*

```powershell
# Trening dokładny (200 epok)
& "C:/.../venv/Scripts/python.exe" "C:/.../train_bert_tiny_qa_thorough.py"

# Prostszy skrypt
& "C:/.../venv/Scripts/python.exe" "C:/.../train_bert_tiny_qa.py"
```

Model i mapowania etykiet zapisują się w katalogu `bert-tiny-qa-thorough/` wraz z plikiem hashującym integralność (`model.sha256.txt`).

### 9.4 Wyniki

* Validation: accuracy ≈ **24.3%**, loss ≈ 4.99
* Test: accuracy = **37.5%**, loss ≈ 4.68 (n=8)
* Losowy baseline: ≈ **0.35%** (1 / 286)

Wyniki są zdecydowanie powyżej losowości, ale należy pamiętać o:

* bardzo małych splitach walidacyjnych/testowych,
* ekstremalnej liczbie klas i wielu singletonach.

---

## 10. Inferencja (zadawanie pytań)

### 10.1 API w Pythonie

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

### 10.2 CLI (z opcjonalnym szumem na logitach)

```powershell
& "C:/.../venv/Scripts/python.exe" "C:/.../infer_bert_tiny_qa.py" `
   --q "Czy Paweł Kowalski jest studentem?" `
   --logit_noise gaussian `
   --noise_scale 0.2
```

---

## 11. Demonstracje ataków

### 11.1 Membership Inference Attack (MIA)

**Cel:** rozstrzygnąć, czy rekord (pytanie + odpowiedź) był w treningu.
**Intuicja:** przeuczone modele dają zwykle wyższe pewności (i niższą stratę) dla przykładów treningowych [1].

Metryki:

* `true_conf` – softmax na prawdziwej etykiecie,
* `conf` – maksymalne prawdopodobieństwo (bez znajomości etykiety),
* `neg_loss` – ujemna cross-entropy.

Pipeline:

1. Przygotowanie statystyk

```powershell
& "C:/.../python.exe" "C:/.../mia_prepare_stats.py"
# -> mia_stats.npz
```

2. Wyznaczenie progów i jakości ataku

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

3. Atak na pojedynczy rekord

```powershell
& "C:/.../python.exe" "C:/.../mia_attack_single.py" `
  --question  "Czy Karol Narożniak jest studentem?" `
  --answer    "..." `
  --model_dir "C:/.../bert-tiny-qa-thorough" `
  --threshold "C:/.../mia_threshold.json" `
  --threshold_type youden
```

4. Atak wsadowy (bez etykiet)

```powershell
& "C:/.../python.exe" "C:/.../mia_attack_batch.py" `
  --model_dir "C:/.../bert-tiny-qa-thorough" `
  --threshold "C:/.../mia_threshold_conf.json" `
  --threshold_type youden `
  --input_txt "C:/.../mia_candidates.txt" `
  --out_csv   "C:/.../mia_batch_results.csv"
```

### 11.2 Attribute Inference Attack (AIA)

**Cel:** odzyskać ukryty atrybut (np. rok studiów, numer albumu) na podstawie odpowiedzi i rozkładów prawdopodobieństw. To praktyczny wariant ataków inferencyjnych / inversion, gdzie model działa jak „leaky database” [2].

Skrypt `attribute_attack.py`:

1. Przyjmuje pytanie oraz listę kandydatów.
2. Dla każdego kandydata generuje wariant zapytania i odpytuje model.
3. Sumuje score’y dla klas, których tekst zawiera daną wartość (regex/substring).
4. Wybiera kandydata o najwyższym łącznym score.

Przykłady:

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

### 11.3 Model Stealing / Extraction

**Cel:** zbudować model-klon, mając dostęp wyłącznie do API predykcji (black-box). W praktyce jest to destylacja wiedzy z *teacher* do *student* [9], zgodna z klasycznymi atakami ekstrakcji modeli [4]. Nowsze wyniki pokazują, że ekstrakcja dotyczy także produkcyjnych modeli językowych [5].

Pipeline:

1. Pula pytań

```powershell
& "C:/.../python.exe" "C:/.../make_question_pool.py" `
  --local "C:/.../merged_questions.json" `
  --out   "C:/.../question_pool.json" `
  --n_squad 500
```

2. Budowa zbioru zastępczego (substitute dataset)

```powershell
& "C:/.../python.exe" "C:/.../steal_build_substitute.py" `
  --teacher_dir "C:/.../bert-tiny-qa-thorough" `
  --data        "C:/.../question_pool.json" `
  --out         "C:/.../substitute_v2.jsonl" `
  --augs_per_q  10 `
  --sharpen     temp `
  --T           0.5
```

3. Trening studenta

```powershell
& "C:/.../python.exe" "C:/.../steal_train_student.py" `
  --substitute    "C:/.../substitute_v2.jsonl" `
  --student_model "prajjwal1/bert-mini" `
  --teacher_dir   "C:/.../bert-tiny-qa-thorough" `
  --out_dir       "C:/.../stolen-bert-mini-v2" `
  --use_sharp `
  --epochs      15 `
  --batch_size  32 `
  --lr          3e-5
```

4. Ewaluacja klona

```powershell
& "C:/.../python.exe" "C:/.../steal_eval_student.py" `
  --teacher_dir "C:/.../bert-tiny-qa-thorough" `
  --student_dir "C:/.../stolen-bert-mini-v2" `
  --data        "C:/.../merged_questions.json" `
  --n_eval      100
```

Wyniki przykładowe:

* Liczba przykładów w zbiorze substytucyjnym: 9 086
* Zgodność teacher–student (ta sama klasa): ~0.71
* Średnia dywergencja KL: ~1.075 (std ~0.386)

Wniosek praktyczny: nawet mały model QA można relatywnie łatwo sklonować, jeśli API zwraca pełny wektor prawdopodobieństw.

---

## 12. Mechanizmy obrony

### 12.1 Szum na logitach (logit noise)

* Plik: `defense_noise.py`
* Rozkłady:

  * `gaussian` – parametr `sigma`,
  * `laplace` – parametr `b`.

Przykład:

```powershell
# Przygotowanie statystyk MIA z szumem Gaussa (σ=0.3)
& "C:/.../python.exe" "C:/.../mia_prepare_stats.py" `
  --logit_noise gaussian `
  --noise_scale 0.3
```

Intuicja: szum zmniejsza różnicę między przykładami z treningu i spoza treningu, co utrudnia MIA (częściowo w duchu prywatności różniczkowej), kosztem jakości predykcji [7].

### 12.2 Integralność modelu (hash & statystyki wag)

* Pliki: `defense_utils.py`, `defense_checks.py`

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

To prosty mechanizm wykrywania podmiany wag (np. trojanizacja/backdoor).

### 12.3 Safeguard Gateway (filtrowanie zapytań)

* Plik: `safeguard_gateway.py`
* Funkcje:

  * wykrywanie prób prompt-injection,
  * wykrywanie pytań o membership/atrybuty,
  * wzorce danych wrażliwych (PII),
  * opcjonalny logit noise.

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

---

## 13. Struktura plików (skrót)

* `merged_questions.json` – zbiór danych QA (PL)
* `train_bert_tiny_qa*.py` – skrypty treningowe
* `bert-tiny-qa-thorough/` – model + mapowania + hash
* `infer_bert_tiny_qa.py` – inferencja (CLI)
* `mia_*.py` – MIA
* `attribute_attack.py` – AIA
* `make_question_pool.py`, `steal_*.py` – model stealing / distillation
* `defense_*.py`, `safeguard_*.py` – obrony
* `trojannet_threat_model.md` – opis koncepcyjny TrojanNet (bez implementacji)

---

## 14. Ograniczenia i możliwe rozszerzenia

Ograniczenia:

* mały zbiór danych i ekstremalna liczba klas → metryki są szacunkowe,
* `bert-tiny` ma ograniczoną pojemność – celem jest edukacja, nie maksymalna jakość,
* brak formalnych gwarancji DP; szum na logitach to heurystyka [7].

Możliwe rozszerzenia:

* podmiana `bert-tiny` na modele PL (np. HerBERT, PolBERT),
* metryki rankingowe / top-k accuracy,
* wdrożenie DP-SGD dla formalnych gwarancji prywatności,
* scenariusze agentowe i prompt injection w stylu LLM (multi-step).

---

## 15. Dostępność kodu

Kod źródłowy i kompletne środowisko demonstracyjne są dostępne w repozytorium:

* [https://github.com/KarolNarozniak/Model_Attacks/](https://github.com/KarolNarozniak/Model_Attacks/)

---

## 16. Bibliografia

[1] R. Shokri, M. Stronati, C. Song, V. Shmatikov, *Membership Inference Attacks against Machine Learning Models*, IEEE Symposium on Security and Privacy (S&P), 2017. Preprint: arXiv:1610.05820.

[2] M. Fredrikson, S. Jha, T. Ristenpart, *Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures*, CCS, 2015.

[3] N. Carlini, F. Tramèr, E. Wallace, M. Jagielski, A. Herbert-Voss, K. Lee, A. Roberts, T. Brown, D. Song, U. Erlingsson, A. Oprea, C. Raffel, *Extracting Training Data from Large Language Models*, USENIX Security, 2021. Preprint: arXiv:2012.07805.

[4] F. Tramèr, F. Zhang, A. Juels, M. K. Reiter, T. Ristenpart, *Stealing Machine Learning Models via Prediction APIs*, USENIX Security, 2016. Preprint: arXiv:1609.02943.

[5] N. Carlini et al., *Stealing Part of a Production Language Model*, 2024. Preprint: arXiv:2403.06634.

[6] F. Perez, I. Ribeiro, *Ignore Previous Prompt: Attack Techniques For Language Models*, 2022. Preprint: arXiv:2211.09527 (także wersja OpenReview).

[7] C. Dwork, A. Roth, *The Algorithmic Foundations of Differential Privacy*, Foundations and Trends in Theoretical Computer Science, 2014.

[8] J. Devlin, M.-W. Chang, K. Lee, K. Toutanova, *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, 2018/2019. Preprint: arXiv:1810.04805.

[9] G. Hinton, O. Vinyals, J. Dean, *Distilling the Knowledge in a Neural Network*, 2015. Preprint: arXiv:1503.02531.

```
::contentReference[oaicite:0]{index=0}
```
