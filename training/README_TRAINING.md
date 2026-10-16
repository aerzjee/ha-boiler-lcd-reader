# Przewodnik Trenowania Modelu YOLO11 na PC (NVIDIA RTX 5080 / Intel Core Ultra Arc 140T + NPU)

Ten przewodnik przeprowadzi Cię krok po kroku przez cały proces przygotowania zbioru danych, oznaczania etykiet, trenowania modelu detekcji cyfr **YOLO11** oraz automatycznego eksportu do formatu **TensorFlow Lite (`.tflite`)** zoptymalizowanego pod kątem dodatku Home Assistant.

---

## 🖥️ Obsługiwane Platformy Sprzętowe

Potok treningowy posiada natywną auto-detekcję i optymalizację pod kątem nowoczesnych stacji roboczych PC oraz laptopów:
1. **Platforma NVIDIA (Maksymalna moc obliczeniowa):**
   - **GPU:** NVIDIA GeForce RTX 5080 (architektura Blackwell, 16 GB GDDR7 VRAM, CUDA 12.4+)
   - **CPU:** AMD Ryzen 9 9950X3D (16 rdzeni / 32 wątki)
2. **Platforma Intel Core Ultra (Nowoczesna architektura hybrydowa):**
   - **GPU (XPU):** Intel Arc 140T (architektura Xe2, współdzielona pamięć RAM, PyTorch XPU)
   - **NPU:** Intel AI Boost (dedykowany akcelerator neuronowy z obsługą OpenVINO)
3. **Standardowe CPU x86-64:** Uniwersalny fallback w razie braku akceleracji sprzętowej.

---

## 📋 Krok po Kroku

### Krok 1: Pobranie próbek treningowych z Home Assistanta

1. Otwórz interfejs dodatku **Boiler LCD Reader** w Home Assistant.
2. Upewnij się, że obszary **🔴 ROI CO** oraz **🟢 ROI CWU** są prawidłowo wyznaczone.
3. Włącz tryb **Collector Mode** i pozwól dodatkowi zebrać próbki przez kilka godzin / dni podczas normalnej pracy pieca (gdy temperatura zmienia się w różnych zakresach i warunkach oświetlenia).
4. W karcie *Zbiór Treningowy (Collector)* kliknij przycisk **📦 Pobierz paczkę próbek (.zip)**.
5. Zapisz pobrany plik `boiler_training_samples_<timestamp>.zip` na swoim komputerze PC.

---

### Krok 2: Oznaczanie cyfr (Labeling) w darmowym narzędziu

Do adnotacji próbek nie potrzebujesz instalować skomplikowanego oprogramowania. Najszybszą i w pełni darmową metodą jest narzędzie internetowe **MakeSense.ai**:

1. Wejdź na stronę [https://www.makesense.ai/](https://www.makesense.ai/) w przeglądarce.
2. Rozpakuj pobrane archiwum `.zip`. Znajdziesz w nim pliki:
   - `sample_*_full.jpg` (pełne klatki kamery – opcjonalne, nieoznaczane)
   - `sample_*_roi_co.jpg` (wycinki ekranu CO)
   - `sample_*_roi_cwu.jpg` (wycinki ekranu CWU)
3. Przeciągnij **wyłącznie wycinki ROI** (`*_roi_co.jpg` oraz `*_roi_cwu.jpg`) do okna MakeSense.ai i kliknij **Object Detection**.
4. W oknie definicji etykiet (**Create labels**) wprowadź **dokładnie 10 klas** w kolejności rosnącej od 0 do 9:
   ```text
   0
   1
   2
   3
   4
   5
   6
   7
   8
   9
   ```
   > [!IMPORTANT]
   > **Kolejność klas ma kluczowe znaczenie!**  
   > Indeks 0 musi odpowiadać cyfrze `0`, indeks 1 cyfrze `1`, aż do indeksu 9 odpowiadającego cyfrze `9`.  
   > **Nie dodawaj** znaków stopni (°), litery C ani minusa. Oznaczaj wyłącznie prostokąty wokół świecących segmentów cyfr.

5. Narysuj prostokąty (Bounding Boxes) ciasno wokół każdej widocznej cyfry na wycinku i przypisz odpowiednią cyfrę.
   * *Przykład:* Jeśli na wyświetlaczu CO widnieje liczba `48`, narysuj prostokąt wokół `4` (klasa `4`) oraz wokół `8` (klasa `8`).
6. Po oznaczeniu zdjęć kliknij w menu górnym: **Actions** $\rightarrow$ **Export Annotations** $\rightarrow$ wybierz **A zip file containing YOLO format text files**.
7. Zapisz i rozpakuj wyeksportowany plik `.zip` z etykietami. Umieść pliki `.txt` obok odpowiadających im plików `.jpg`.

---

### Krok 3: Przygotowanie środowiska Python na PC

Otwórz terminal (PowerShell lub bash) w katalogu `training/`:

```bash
# 1. Przejdź do katalogu training
cd training

# 2. Utwórz dedykowane środowisko wirtualne
python -m venv .venv

# 3. Aktywuj środowisko
# Na Windows:
.venv\Scripts\activate
# Na Linux:
source .venv/bin/activate
```

#### Wariant A: Stacja z kartą NVIDIA GeForce RTX 5080 (CUDA)
```bash
# Instalacja PyTorch z obsługą CUDA 12.4+:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Instalacja pozostałych zależności:
pip install -r requirements-training.txt
```

#### Wariant B: Urządzenie z procesorem Intel Core Ultra (Intel Arc 140T + Intel AI Boost NPU)
Upewnij się, że w systemie zainstalowane są sterowniki **Intel Arc Graphics Driver** oraz **Intel NPU Driver (Intel AI Boost)** ze strony Intel lub z Windows Update.
```bash
# Instalacja PyTorch z natywną obsługą Intel XPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu

# Instalacja pozostałych zależności (w tym OpenVINO):
pip install -r requirements-training.txt
```

---

### Krok 4: Przygotowanie struktury i podział zbioru danych

Użyj dołączonego skryptu `prepare_dataset.py`, który zweryfikuje poprawność współrzędnych, sprawdzi zgodność klas (0–9) i podzieli dane na zbiór uczący (85%) i walidacyjny (15%):

```bash
python prepare_dataset.py --input "C:\sciezka\do\folderu_z_oznaczonymi_zdjeciami"
```

Możesz także wskazać bezpośrednio spakowane archiwum `.zip` zawierające zdjęcia i pliki `.txt`:

```bash
python prepare_dataset.py --input "C:\Pobrane\boiler_annotated.zip"
```

Skrypt wygeneruje strukturę:
```text
training/datasets/boiler_digits/
├── images/
│   ├── train/  (85% zdjęć)
│   └── val/    (15% zdjęć)
└── labels/
    ├── train/  (odpowiadające etykiety .txt)
    └── val/    (odpowiadające etykiety .txt)
```

Oraz wypisze w konsoli szczegółową tabelę rozkładu klas cyfr (0–9) wraz z wykresem słupkowym ASCII i wskazaniem rzadkich cyfr (< 5% próbek).

> [!TIP]
> **Przydatne flagi skryptu `prepare_dataset.py`:**
> - `python prepare_dataset.py --check-dist` – błyskawiczna analiza rozkładu klas w istniejącym zbiorze bez konieczności ponownego importu plików.
> - `python prepare_dataset.py --trim-existing --trim-per-class 30` – przycięcie istniejącego zbioru do max 30 próbek na cyfrę z automatyczną kopią zapasową.
> - `python prepare_dataset.py --input <paczka> --trim-per-class 30` – przygotowanie nowego zbioru z paczki ZIP z automatycznym inteligentnym przycięciem do 30 sztuk na cyfrę.
> - `python prepare_dataset.py --input <paczka> --append` – dopisanie nowych próbek bez usuwania dotychczasowych danych i z automatyczną ochroną przed kolizjami nazw plików.

---

### Krok 5: Uruchomienie treningu i automatycznego eksportu TFLite

Uruchom główny skrypt treningowy:

```bash
# Domyślne automatyczne wykrycie akceleratora i optymalnych parametrów:
python train_and_export.py
```

Możesz także podać parametry jawnie:

```bash
# NVIDIA GeForce RTX 5080:
python train_and_export.py \
  --model yolo11n.pt \
  --data dataset.yaml \
  --epochs 150 \
  --patience 30 \
  --imgsz 320 \
  --batch 64 \
  --workers 12 \
  --device 0
```

#### Automatyczne dopasowanie parametrów do sprzętu:
- **NVIDIA CUDA (RTX 5080):** `batch = 64`, `workers = 12`, `amp = True` – maksymalizuje nasycenie 16 GB pamięci VRAM oraz 32 wątków procesora Ryzen 9 9950X3D.
- **Intel Arc (Arc 140T XPU):** `batch = 32`, `workers = 8`, `amp = True` – optymalny rozmiar partii dla zintegrowanej pamięci graficznej i rdzeni Xe.
- **CPU:** `batch = 16`, `workers = 4`, `amp = False` – bezpieczne przetwarzanie sekwencyjne.
- **Specjalne augmentacje LCD w kodzie skryptu:**
  - `fliplr=0.0` oraz `flipud=0.0` – **bezwzględnie wyłączone lustrzane odbicia**, ponieważ odwrócone cyfry (np. 2, 5, 3) zniszczyłyby model.
  - `hsv_s=0.4`, `hsv_v=0.4` – symulacja wahań jasności i kontrastu niebieskiego podświetlenia LCD w dzień i w nocy.

Po zakończeniu treningu skrypt automatycznie:
1. Wczyta najlepszy punkt kontrolny `best.pt`.
2. Wyeksportuje model do formatu TFLite (Float32).
3. Skopiuje wynikowy plik do **`training/dist/active_model.tflite`**.
4. Wykona test **Sanity Check**:
   - Zweryfikuje obecność identyfikatora `TFL3` w nagłówku FlatBuffer,
   - Załaduje model do interpretera TFLite i wykona próbną inferencję,
   - Upewni się, że wyjściowy tensor ma dokładnie 14 kanałów (4 bbox + 10 klas cyfr).

---

## ⚡ Trening i akceleracja na Intel Core Ultra (Intel Arc 140T + Intel AI Boost NPU)

Procesory **Intel Core Ultra** posiadają zintegrowany układ graficzny **Intel Arc 140T** oraz akcelerator neuronowy **Intel AI Boost (NPU)**. Skrypt treningowy w pełni wspiera tę architekturę:

1. **Automatyczne wykrycie sprzętu (Auto-Detection):**
   ```bash
   python train_and_export.py
   ```
   Skrypt sam rozpozna obecność `torch.xpu` oraz jednostki NPU w OpenVINO.

2. **Wymuszenie treningu na układzie Intel Arc 140T (XPU):**
   ```bash
   python train_and_export.py --device xpu --batch 32
   ```

3. **Trening z weryfikacją i benchmarkiem na akceleratorze Intel AI Boost (NPU):**
   ```bash
   python train_and_export.py --device npu
   ```
   *lub z poziomu karty Arc:*
   ```bash
   python train_and_export.py --device xpu --npu-benchmark
   ```
   Skrypt przeprowadzi proces uczenia na rdzeniach Xe karty graficznej Intel Arc (`xpu`), wyeksportuje standardowy model `.tflite` dla Home Assistanta, a następnie wyeksportuje model do formatu **OpenVINO FP16** i zmierzy średni czas inferencji (ms/klatka) oraz FPS na:
   - **Intel AI Boost (NPU)**
   - **Intel Arc Graphics (GPU)**
   - **Host CPU**

---

## ✂️ Balansowanie i przycinanie zbyt dużego zbioru (Trim Dataset)

### Dlaczego mniej znaczy więcej przy statycznej kamerze LCD?
Gdy kamera jest sztywno zamontowana przed wyświetlaczem kotła i zbierze ponad 1000 klatek (np. 1105 próbek), pojawiają się dwa zjawiska degradujące skuteczność i pewność (*confidence*) modelu:

1. **Zjawisko przeuczenia tła (Background Overfitting) i memorizacja szumu sensora:**
   - W odróżnieniu od klasycznych zbiorów detekcji obiektów (np. COCO – samochody, piesi na zmiennym tle), kamera LCD rejestruje w kółko identyczne tło, mikrorysy na plastiku obudowy, kąt padania światła i stały szum matrycy CMOS.
   - Posiadanie setek niemal identycznych zdjęć tych samych cyfr sprawia, że sieć neuronowa zamiast uczyć się niezmienniczych cech geometrycznych segmentów cyfr (linii pionowych, poziomych, kątów), dopasowuje wagi do konkretnych pikseli tła. W efekcie po wyeksportowaniu do TFLite model traci pewność (confidence spada np. do 0.30–0.50 lub cyfry przestają być w ogóle wykrywane).

2. **Zbyt mała liczba kroków aktualizacji gradientu (Gradient Step Starvation) i rozkalibrowanie BatchNorm:**
   - Jeśli zbiór 1100 zdjęć jest zdominowany przez typowe temperatury pracy (np. 48°C, 52°C, 56°C), klasy `4`, `5` i `6` mają po 150 próbek, a cyfry rzadkie zaledwie po kilka.
   - Przy dużym batch size (np. 64 na kartach NVIDIA RTX 5080) w każdej epoce wykonuje się zaledwie kilkanaście kroków aktualizacji wag. Statystyki warstw **Batch Normalization** ulegają rozregulowaniu, co w architekturze YOLO drastycznie obniża pewność detekcji.

**Złota zasada:** Zbalansowany zbiór **30–50 zróżnicowanych próbek na każdą cyfrę** (łącznie ok. 150–250 zdjęć) w połączeniu z mniejszym `batch=16` i augmentacjami jasności/kontrastu (`hsv_s`, `hsv_v`) uczy detektor wyłącznie kształtu segmentów i osiąga pewność detekcji **0.85–0.98**!

---

### Algorytm inteligentnego przycinania (Rarity-First Greedy Selection)
Ponieważ na pojedynczym wycinku ROI niemal zawsze występują dwie cyfry jednocześnie (np. temperatura `48` zawiera cyfrę `4` i cyfrę `8`), zwykłe losowe usuwanie plików usunęłoby cenną rzadką cyfrę `8`, bo towarzyszyła jej popularna cyfra `4`.

Algorytm w `prepare_dataset.py`:
1. Analizuje rozkład wszystkich klas i sortuje cyfry od najrzadszej do najczęstszej.
2. Zaczynając od najrzadszych cyfr (np. `8`, `9`, `2`), zabezpiecza ich obecność w docelowej liczbie (np. 30 sztuk).
3. Dozbierając próbki dla rzadkich cyfr, preferuje te zdjęcia, na których cyfry towarzyszące nie przekraczają limitu lub mieszczą się w dopuszczalnym miękkim marginesie (+30%).
4. Odrzuca nadmiarowe, redundantne klatki popularnych cyfr, sprowadzając cały zbiór do zrównoważonego rozkładu.
5. Gwarantuje, że przy podziale train (85%) / val (15%) każda obecna cyfra trafi zarówno do zbioru uczącego, jak i walidacyjnego.

---

### Gotowe komendy

#### 1. Przycięcie już istniejącego zbioru (np. 1105 próbek) do 30 sztuk na cyfrę
Nie musisz ponownie pobierać paczki ZIP. Użyj flagi `--trim-existing`:
```bash
python prepare_dataset.py --trim-existing --trim-per-class 30
```
> [!NOTE]
> Skrypt automatycznie utworzy kopię zapasową w `training/datasets/boiler_digits_backup_<timestamp>/`, dzięki czemu nigdy nie utracisz pierwotnych adnotacji. Następnie wyświetli w konsoli tabelę porównawczą PRZED vs PO przycięciu.

#### 2. Przygotowanie nowego zbioru z paczki ZIP z automatycznym przycięciem
Możesz zastosować limit już na etapie rozpakowywania nowej paczki próbek:
```bash
python prepare_dataset.py --input "C:\Pobrane\boiler_training_samples.zip" --trim-per-class 30
```

#### 3. Ponowne czyste wytrenowanie modelu na przyciętym zbiorze
Po zredukowaniu zbioru uruchom trening od zera z bazowego modelu `yolo11n.pt` z optymalnym `batch=16` oraz `epochs=100`:
```bash
# NVIDIA GeForce RTX 5080 (CUDA):
python train_and_export.py --epochs 100 --batch 16 --device 0

# Intel Core Ultra (Intel Arc 140T XPU):
python train_and_export.py --epochs 100 --batch 16 --device xpu

# Tryb automatyczny:
python train_and_export.py --epochs 100 --batch 16
```
> [!TIP]
> Skrypt `train_and_export.py` automatycznie wykrywa, gdy zbiór treningowy ma poniżej 400 klatek, i samoczynnie dobiera bezpieczny `batch=16` (chyba że podasz inną wartość ręcznie).

---

## 🔄 Dotrenowanie istniejącego modelu (Fine-Tuning na rzadkich cyfrach)

### Dlaczego rzadkie cyfry stanowią wyzwanie?
W typowych warunkach domowych kocioł gazowy lub na pellet pracuje w powtarzalnych zakresach temperatur (np. CO: 45°C–65°C, CWU: 40°C–55°C). W efekcie w pierwszej zebranej paczce danych:
- Cyfry **4, 5, 6** dominują (często stanowią ponad 80% wszystkich adnotacji),
- Cyfry **7, 8, 9** (oraz rzadziej **0, 1, 2**) pojawiają się sporadycznie (< 5% próbek) lub wcale nie występują w zbiorze.

Model wytrenowany na niezrównoważonym zbiorze może mieć trudności z poprawnym odczytem temperatur poza typowym zakresem (np. wygrzewanie antylegionella CWU na 70°C lub mroźne dni, gdy CO osiąga 75°C).

### Zjawisko "Catastrophic Forgetting" (Gwałtowne Zapominanie)
Gdy zbierzesz drugą paczkę zawierającą wyłącznie brakujące rzadkie cyfry, **nie wolno** trenować modelu od zera na samej nowej paczce ani uruchamiać pełnego treningu z domyślnym learning rate (`lr0=0.01`). Duże kroki gradientowe nadpisałyby wyuczone wcześniej filtry splotowe, powodując zjawisko **katastrofalnego zapominania** – model nauczyłby się rozpoznawać cyfrę `8`, lecz przestałby poprawnie klasyfikować cyfry `4`, `5` czy `6`.

**Rozwiązanie wbudowane w potok Boiler LCD Reader:**
1. **Tryb dopisywania (`--append`):** Nowe próbki są dołączane do dotychczasowej bazy danych bez jej kasowania, dzięki czemu sieć podczas nauki stale widzi wszystkie cyfry.
2. **Tryb Fine-Tuning (`--fine-tune`):**
   - Startuje z najlepszego punktu kontrolnego (`best.pt`) zamiast z wag ogólnych `yolo11n.pt`,
   - Obniża współczynnik uczenia dziesięciokrotnie (`lr0=0.001`),
   - Skraca fazę rozgrzewki (`warmup_epochs=1.0`),
   - Ogranicza liczbę epok do 60 (z early stopping patience = 30).

---

### Instrukcja Fine-Tuningu krok po kroku

#### Krok A: Sprawdzenie rozkładu klas w obecnym zbiorze danych
W katalogu `training/` uruchom weryfikację rozkładu:
```bash
python prepare_dataset.py --check-dist
```
Skrypt wygeneruje w terminalu tabelę z wykresem słupkowym ASCII:
```text
 Cyfra |  Liczba  | Udział  | Wykres słupkowy ASCII  |           Status           
-------+----------+---------+------------------------+----------------------------
   0   |    4     |  1.5%   | [░░░░░░░░░░░░░░░░░░░░] | ⚠️  RZADKA KLASA (< 5%)   
   ...
   5   |   103    |  38.4%  | [████████░░░░░░░░░░░░] | ✓ OK                      
   6   |   117    |  43.7%  | [█████████░░░░░░░░░░░] | ✓ OK                      
   8   |    7     |  2.6%   | [█░░░░░░░░░░░░░░░░░░░] | ⚠️  RZADKA KLASA (< 5%)   
   9   |    3     |  1.1%   | [░░░░░░░░░░░░░░░░░░░░] | ⚠️  RZADKA KLASA (< 5%)   
----------------------------------------------------------------------------------
⚠️  Rzadkie cyfry w zbiorze (< 5%):  0, 1, 2, 3, 7, 8, 9
```

#### Krok B: Zebranie i zaetykietowanie nowych próbek
1. Gdy na piecu pojawią się nietypowe temperatury (np. podbicie temperatury zadanej CWU, restart pieca wyświetlający `0`, `1` lub `2`), pozwól Collectorowi w dodatku zapisać klatki.
2. Pobierz paczkę próbek `.zip` z panelu dodatku.
3. Oznacz cyfry w [MakeSense.ai](https://www.makesense.ai/) zgodnie z [Krokiem 2](#krok-2-oznaczanie-cyfr-labeling-w-darmowym-narz%C4%99dziu) (pamiętając o zachowaniu klas 0–9).
4. Wyeksportuj etykiety w formacie YOLO `.txt`.

#### Krok C: Dopisanie próbek do istniejącego zbioru danych
Uruchom `prepare_dataset.py` z flagą `--append`:
```bash
python prepare_dataset.py --input "C:\sciezka\do\nowej_paczki_annotated.zip" --append
```
> [!NOTE]
> Flaga `--append` nie usuwa starych plików. Nowe pary trafiają do `train` (85%) i `val` (15%), a ewentualne konflikty nazw plików są automatycznie rozwiązywane przez dodanie sufiksu `_append1`, `_append2` itp. Na koniec skrypt wyświetli nowy, zaktualizowany bilans klas.

#### Krok D: Uruchomienie Fine-Tuningu
Uruchom główny skrypt z flagą `--fine-tune`:

```bash
# NVIDIA GeForce RTX 5080 (CUDA):
python train_and_export.py --fine-tune --device 0

# Intel Core Ultra (Intel Arc 140T XPU):
python train_and_export.py --fine-tune --device xpu

# Tryb automatyczny (Auto-Detection):
python train_and_export.py --fine-tune
```

**Co dzieje się pod maską podczas fine-tuningu:**
1. Skrypt automatycznie odnajduje najnowszy plik wag `best.pt` z poprzednich sesji w folderze `runs/detect/`.
2. Ustawia optymalne hiperparametry: `lr0=0.001`, `warmup_epochs=1.0`, `epochs=60`.
3. Dotrenowuje sieć, zachowując wcześniej wyuczone wzorce i adaptując detektor do nowych cyfr.
4. Eksportuje zaktualizowany model do `training/dist/active_model.tflite` i wykonuje testy poprawności.

*(Opcjonalnie)* Jeśli chcesz jawnie wskazać konkretny plik punktu kontrolnego lub zmienić liczbę epok:
```bash
python train_and_export.py --fine-tune --model "runs/detect/boiler_yolo11n/weights/best.pt" --epochs 80 --lr0 0.0008
```

---

### Krok 6: Wgranie gotowego modelu do Home Assistanta

1. Przejdź do Home Assistanta $\rightarrow$ panel dodatku **Boiler LCD Reader**.
2. W karcie **🧠 Model TensorFlow Lite (.tflite)**:
   - Kliknij pole wyboru pliku i wskaż wygenerowany plik `training/dist/active_model.tflite`.
   - Kliknij przycisk **⬆️ Wgraj model .tflite**.
3. Status modelu w nagłówku natychmiast zmieni się na zielony (**Model aktywny**).
4. W karcie **🔬 Diagnostyka i Podgląd Odczytu na Żywo** kliknij **🧪 Wymuś testowy odczyt teraz**, aby upewnić się, że cyfry na Twoim kotle są rozpoznawane ze 100% poprawnością.

Gratulacje! Twój dodatek działa w pełni autonomicznie, publikując temperatury CO i CWU do Home Assistanta.
