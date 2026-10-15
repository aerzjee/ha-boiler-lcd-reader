# Boiler LCD Reader (TFLite) – Home Assistant Add-on

[![Lint](https://github.com/aerzjee/ha-boiler-lcd-reader/actions/workflows/lint.yaml/badge.svg)](https://github.com/aerzjee/ha-boiler-lcd-reader/actions/workflows/lint.yaml)
[![Build & Publish](https://github.com/aerzjee/ha-boiler-lcd-reader/actions/workflows/build.yaml/badge.svg)](https://github.com/aerzjee/ha-boiler-lcd-reader/actions/workflows/build.yaml)
[![Architecture: amd64](https://img.shields.io/badge/arch-amd64-blue.svg)](https://github.com/aerzjee/ha-boiler-lcd-reader)
[![Home Assistant: Ingress](https://img.shields.io/badge/home--assistant-ingress-41BDF5.svg)](https://www.home-assistant.io/)

Oficjalny dodatek do **Home Assistanta** realizujący automatyczny odczyt dwóch temperatur całkowitych (**CO** – Centralne Ogrzewanie oraz **CWU** – Ciepła Woda Użytkowa) z niebieskiego wyświetlacza LCD kotła grzewczego. Wykorzystuje kamerę zintegrowaną z Home Assistantem, bibliotekę OpenCV oraz zoptymalizowany model detekcji cyfr **YOLO11** uruchamiany przez silnik **TensorFlow Lite (`ai-edge-litert`)** bezpośrednio na procesorze (CPU) maszyny wirtualnej Home Assistant OS (np. na serwerze Synology NAS).

Odczyty publikowane są jako natywne encje sensorów w Home Assistant w pełnych stopniach Celsjusza (**liczby całkowite `int`**, krok 1°C, bez miejsc dziesiętnych) z zachowaniem długoterminowych statystyk (`state_class: measurement`).

---

## 🚀 Główne Cechy i Architektura

* **Architektura dedykowana `amd64`:** Zbudowany na bazie oficjalnego obrazu `ghcr.io/home-assistant/amd64-base-debian:bookworm` zoptymalizowanego pod kątem wydajności na procesorach x86-64.
* **Bezpieczeństwo Zero-Trust:** Brak konieczności generowania i wklejania długoterminowych tokenów (Long-Lived Access Tokens). Komunikacja z API Home Assistanta odbywa się w całości wewnątrz kontenera przez chroniony token `SUPERVISOR_TOKEN`.
* **Interfejs Home Assistant Ingress:** Bezpośrednia integracja z panelem bocznym Home Assistanta – wygodne rysowanie myszą lub palcem na podglądzie z kamery prostokątów ROI (**🔴 ROI CO** oraz **🟢 ROI CWU**) ze skalowaniem responsywnym.
* **Filtr fizyczny (Sanity Check):** Zabezpieczenie przed zafałszowaniem statystyk i wykresów w HA w przypadku chwilowego zasłonięcia ekranu lub zakłóceń oświetlenia:
  * Odrzucanie wartości spoza dopuszczalnego zakresu fizycznego (`min_temp` – `max_temp`, domyślnie 5–95°C).
  * Ochrona przed nagłym skokiem temperatury (`max_delta`, domyślnie 15°C) – skok powyżej progu wymaga potwierdzenia w **3 kolejnych klatkach** przed zaktualizowaniem sensora.
* **Wbudowany Collector Mode (Anti-Duplication):** Automatyczne zbieranie wycinków treningowych w tle po wykryciu istotnej zmiany wizualnej (`cv2.absdiff`), z twardym limitem 300 plików (100 MB max) i jednoklikowym eksportem do archiwum `.zip`.
* **Diagnostyka na żywo w Web UI:** Podgląd na żywo wycinków ROI z naniesionymi zielonymi ramkami detekcji, rozpoznanymi cyframi, poziomem pewności (`confidence`) oraz przyciskiem wymuszenia natychmiastowego testu inferencji.
* **Własny model `.tflite`:** Bezpieczne wgrywanie przez interfejs użytkownika modelu wyeksportowanego z Ultralytics (z walidacją nagłówka FlatBuffer `TFL3`, limitem 25 MB oraz przeładowaniem w pamięci w locie).

---

## 📦 Szybka Instalacja w Home Assistant

1. Skopiuj adres URL tego repozytorium:
   ```text
   https://github.com/aerzjee/ha-boiler-lcd-reader
   ```
2. W panelu Home Assistanta przejdź do:
   **Ustawienia** $\rightarrow$ **Dodatki** $\rightarrow$ **Sklep z dodatkami** (prawy dolny róg).
3. Kliknij **menu w prawym górnym rogu (trzy pionowe kropki)** i wybierz **Repozytoria**.
4. Wklej skopiowany adres URL repozytorium i kliknij **Dodaj**.
5. Odśwież stronę sklepu – na liście pojawi się dodatek **Boiler LCD Reader (TFLite)**.
6. Kliknij **Zainstaluj**, włącz opcje **Uruchamiaj przy starcie** oraz **Pokaż na pasku bocznym**, a następnie kliknij **Uruchom**.
7. Otwórz panel dodatku (przycisk **Otwórz interfejs sieciowy** lub skrót na pasku bocznym).

---

## 🛠️ Pierwsza Konfiguracja (Krok po Kroku)

1. **Wybór kamery:** W sekcji *Konfiguracja Odczytu i Encji* wybierz z listy encję kamery skierowanej na wyświetlacz kotła (np. `camera.kotlownia`) i kliknij *Zapisz konfigurację*.
2. **Wyznaczenie obszarów ROI:**
   * W sekcji *Podgląd Kamery* kliknij przycisk *Odśwież kadr*.
   * Wybierz tryb **🔴 ROI CO** i zaznacz ramkę wokół cyfr temperatury pieca centralnego ogrzewania.
   * Przełącz na **🟢 ROI CWU** i zaznacz ramkę wokół cyfr temperatury ciepłej wody użytkowej.
   * Kliknij **💾 Zapisz ramki ROI**.
3. **Zebranie próbek:**
   * Włącz przełącznik *Automatycznie zbieraj próbki w tle (Collector Mode)* lub wykonaj kilka ręcznych zrzutów przyciskiem *📸 Złap próbkę teraz*.
   * Kliknij **📦 Pobierz paczkę próbek (.zip)**.
4. **Wytrenowanie modelu na PC:**
   * Skorzystaj z dołączonego w katalogu `training/` potoku trenowania YOLO11 na komputerze PC z kartą graficzną NVIDIA (CUDA) lub procesorem Intel Core Ultra (Intel Arc 140T XPU + Intel AI Boost NPU) – zobacz szczegółowy przewodnik: [training/README_TRAINING.md](training/README_TRAINING.md).
5. **Wgranie modelu:**
   * W sekcji *Model TensorFlow Lite* wskaż wygenerowany plik `active_model.tflite` i kliknij **⬆️ Wgraj model .tflite**.
6. **Weryfikacja:**
   * W sekcji *Diagnostyka i Podgląd Odczytu na Żywo* kliknij **🧪 Wymuś testowy odczyt teraz**. Na ekranie pojawią się wycinki z nałożonymi ramkami detekcji, a w Home Assistant powstaną encje `sensor.boiler_co_temperature` i `sensor.boiler_cwu_temperature`.

---

## 📊 Publikowane Encje w Home Assistant

Dodatek automatycznie publikuje stany do Home Assistanta przez Supervisor Core API:

| Encja | Nazwa przyjazna (`friendly_name`) | Typ wartości | Jednostka | Atrybuty |
|---|---|---|---|---|
| `sensor.boiler_co_temperature` | Piec CO Temperatura | `int` (np. `48`) | `°C` | `device_class: temperature`<br>`state_class: measurement`<br>`confidence`, `raw_value`, `validation_status` |
| `sensor.boiler_cwu_temperature` | Piec CWU Temperatura | `int` (np. `42`) | `°C` | `device_class: temperature`<br>`state_class: measurement`<br>`confidence`, `raw_value`, `validation_status` |

---

## 🏷️ Wydawanie Nowych Wersji (CI/CD Release Workflow)

W repozytorium skonfigurowano automatyczny potok CI/CD w GitHub Actions:
- **Lint Workflow (`.github/workflows/lint.yaml`):** Uruchamiany automatycznie przy każdym `push` i `pull_request` do gałęzi `master`/`main` (Ruff linter, Ruff formatter, Bandit).
- **Build & Publish Workflow (`.github/workflows/build.yaml`):**
  Budowanie obrazu kontenera Docker i jego publikacja do rejestru **GitHub Container Registry (`ghcr.io`)** następuje **wyłącznie po wypchnięciu tagu wersji** `v*.*.*`, który musi być ściśle zgodny z polem `version` w pliku `config.yaml`:

```bash
# 1. Zaktualizuj wersję w config.yaml (np. "0.1.0") i zrób commit
git commit -am "chore: bump version to 0.1.0"
git push

# 2. Utwórz i wypchnij tag wersji
git tag v0.1.0
git push origin v0.1.0
```

GitHub Actions zweryfikuje zgodność tagu z `config.yaml`, zbuduje kontener za pomocą oficjalnego narzędzia `home-assistant/builder` i opublikuje gotowy obraz `ghcr.io/aerzjee/ha-boiler-lcd-reader-amd64:0.1.0`.

---

## 🇬🇧 English Overview & Features

**Boiler LCD Reader** is a specialized Home Assistant add-on designed to read two integer temperatures (**Central Heating / CO** and **Domestic Hot Water / CWU**) from a blue boiler LCD display in 1°C increments with no decimal places.

### Key Highlights:
- **Architecture:** `amd64` Debian Bookworm container running optimized TFLite runtime (`ai-edge-litert`) on CPU.
- **Zero-Trust Security:** No Long-Lived Access Tokens required; authenticates via internal Home Assistant `SUPERVISOR_TOKEN`.
- **HA Ingress UI:** Canvas-based interactive Region-of-Interest (ROI) bounding box drawer with live camera preview.
- **Physical Sanity Validation:** Enforces temperature plausibility bounds (`min_temp`–`max_temp`) and rejects abrupt temperature jumps (`max_delta`) unless confirmed across 3 consecutive frames, protecting long-term sensor statistics.
- **Collector Mode:** Built-in frame harvester with change detection (`cv2.absdiff`) and ZIP export for model fine-tuning.
- **Model Pipeline:** Full YOLO11 training and TFLite export scripts provided in `training/` directory with multi-hardware acceleration (NVIDIA RTX CUDA, Intel Core Ultra Arc 140T XPU & Intel AI Boost NPU via OpenVINO).

---

## 📄 Licencja

Projekt udostępniany na licencji MIT. Zobacz plik `LICENSE` po szczegóły.
