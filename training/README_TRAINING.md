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

Oraz wypisze w konsoli statystyki wystąpień poszczególnych cyfr.

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

### Krok 6: Wgranie gotowego modelu do Home Assistanta

1. Przejdź do Home Assistanta $\rightarrow$ panel dodatku **Boiler LCD Reader**.
2. W karcie **🧠 Model TensorFlow Lite (.tflite)**:
   - Kliknij pole wyboru pliku i wskaż wygenerowany plik `training/dist/active_model.tflite`.
   - Kliknij przycisk **⬆️ Wgraj model .tflite**.
3. Status modelu w nagłówku natychmiast zmieni się na zielony (**Model aktywny**).
4. W karcie **🔬 Diagnostyka i Podgląd Odczytu na Żywo** kliknij **🧪 Wymuś testowy odczyt teraz**, aby upewnić się, że cyfry na Twoim kotle są rozpoznawane ze 100% poprawnością.

Gratulacje! Twój dodatek działa w pełni autonomicznie, publikując temperatury CO i CWU do Home Assistanta.
