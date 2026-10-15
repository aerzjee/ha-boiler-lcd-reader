# Przewodnik Trenowania Modelu YOLO11 na PC (RTX 5080 + Ryzen 9950X3D)

Ten przewodnik przeprowadzi Cię krok po kroku przez cały proces przygotowania zbioru danych, oznaczania etykiet, trenowania modelu detekcji cyfr **YOLO11** oraz automatycznego eksportu do formatu **TensorFlow Lite (`.tflite`)** zoptymalizowanego pod kątem dodatku Home Assistant.

---

## 🖥️ Wymagania Sprzętowe i Środowiskowe

Potok treningowy został zoptymalizowany pod kątem nowoczesnej stacji roboczej PC:
- **GPU:** NVIDIA GeForce RTX 5080 (architektura Blackwell, 16 GB VRAM)
- **CPU:** AMD Ryzen 9 9950X3D (16 rdzeni / 32 wątki)
- **System:** Windows 10/11 lub Linux (Ubuntu 22.04+)
- **Python:** 3.10 – 3.12 (zalecany Python 3.11 z PyTorch 2.5+ i CUDA 12.4+)

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
cd c:\Users\Kamil\Documents\git-repos\ha-boiler-lcd-reader\training

# 2. Utwórz dedykowane środowisko wirtualne
python -m venv .venv

# 3. Aktywuj środowisko
# Na Windows:
.venv\Scripts\activate
# Na Linux:
source .venv/bin/activate

# 4. Zainstaluj PyTorch z obsługą CUDA dla architektury Blackwell RTX 5080:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 5. Zainstaluj pozostałe pakiety treningowe i eksportowe:
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

#### Dlaczego te parametry są optymalne?
- **`--batch 64`:** Maksymalizuje wykorzystanie 16 GB szybkiej pamięci VRAM karty RTX 5080, zapewniając stabilną zbieżność gradientu.
- **`--workers 12`:** Wykorzystuje wielowątkowość procesora Ryzen 9 9950X3D do błyskawicznego ładowania i augmentacji klatek w pamięci RAM.
- **`--imgsz 320`:** Rozdzielczość 320x320 px jest optymalna dla małych wycinków ROI ekranu LCD i zapewnia minimalny narzut na CPU w Home Assistant (kilkanaście ms na odczyt).
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

### Krok 6: Wgranie gotowego modelu do Home Assistanta

1. Przejdź do Home Assistanta $\rightarrow$ panel dodatku **Boiler LCD Reader**.
2. W karcie **🧠 Model TensorFlow Lite (.tflite)**:
   - Kliknij pole wyboru pliku i wskaż wygenerowany plik `training/dist/active_model.tflite`.
   - Kliknij przycisk **⬆️ Wgraj model .tflite**.
3. Status modelu w nagłówku natychmiast zmieni się na zielony (**Model aktywny**).
4. W karcie **🔬 Diagnostyka i Podgląd Odczytu na Żywo** kliknij **🧪 Wymuś testowy odczyt teraz**, aby upewnić się, że cyfry na Twoim kotle są rozpoznawane ze 100% poprawnością.

Gratulacje! Twój dodatek działa w pełni autonomicznie, publikując temperatury CO i CWU do Home Assistanta.
