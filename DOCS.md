# Dokumentacja Użytkownika – Boiler LCD Reader (TFLite)

Dodatek **Boiler LCD Reader** umożliwia bezinwazyjny, precyzyjny odczyt temperatur kotła grzewczego (**CO** – Centralne Ogrzewanie oraz **CWU** – Ciepła Woda Użytkowa) z niebieskiego ekranu LCD przy użyciu kamery skierowanej na wyświetlacz oraz lekkiego modelu detekcji cyfr TensorFlow Lite.

---

## Krok 1: Pierwsze uruchomienie i dostęp do interfejsu (Ingress)

1. Po zainstalowaniu dodatku w sklepie Home Assistant przejdź do karty **Ustawienia** dodatku.
2. Włącz opcje:
   - **Uruchamiaj przy starcie** (Start on boot)
   - **Pokaż na pasku bocznym** (Show in sidebar)
3. Kliknij przycisk **Uruchom** (Start).
4. Kliknij **Otwórz interfejs sieciowy** (Open Web UI) lub wybierz skrót **Boiler LCD** na lewym pasku bocznym Home Assistanta.

Dzięki technologii **Home Assistant Ingress** interfejs otwiera się bezpośrednio w oknie przeglądarki bez konieczności otwierania dodatkowych portów w routerze czy konfiguracji sieci.

---

## Krok 2: Wybór kamery i interwału odczytu

1. W panelu głównym odszukaj kartę **⚙️ Konfiguracja Odczytu i Encji**.
2. W polu **Kamera Home Assistant** wybierz z listy rozwijanej encję kamery skierowanej na wyświetlacz kotła (np. `camera.kotlownia`).
   * *Wskazówka:* Jeśli lista jest pusta, upewnij się, że integracja kamery w Home Assistant działa poprawnie i kliknij przycisk odświeżenia 🔄 obok listy.
3. Skonfiguruj parametry skanowania:
   - **Interwał odczytu (s):** czas pomiędzy kolejnymi analizami obrazu (domyślnie `30` sekund; zalecany zakres: `15`–`60` s).
   - **Próg pewności detekcji:** minimalny współczynnik pewności modelu YOLO dla pojedynczej cyfry (domyślnie `0.50`).
   - **Min. / Maks. temperatura (°C):** fizyczne granice dopuszczalnych wskazań (domyślnie `5` do `95` °C).
   - **Maks. skok (°C):** maksymalna dozwolona zmiana temperatury w pojedynczym odczycie (domyślnie `15` °C). Skoki powyżej tej wartości wymagają potwierdzenia w 3 kolejnych klatkach.
4. Kliknij **💾 Zapisz konfigurację**.

---

## Krok 3: Wyznaczanie obszarów zainteresowania (ROI)

Wyznaczenie dokładnych ramek wokół cyfr na wyświetlaczu eliminuje wpływ tła kotłowni oraz przyspiesza przetwarzanie:

1. W karcie **📷 Podgląd Kamery i Rysowanie Obszarów ROI** kliknij przycisk **🔄 Odśwież kadr**, aby pobrać aktualny obraz z kamery.
2. Zaznacz opcję **🔴 ROI CO (Centralne Ogrzewanie)** w selektorze trybu rysowania.
3. Przeciągnij kursorem myszy lub palcem po ekranie, rysując prostokątną ramkę obejmującą wyłącznie cyfry temperatury CO na wyświetlaczu.
4. Przełącz selektor na **🟢 ROI CWU (Ciepła Woda)** i w analogiczny sposób narysuj drugą ramkę obejmującą cyfry temperatury CWU.
5. Kliknij przycisk **💾 Zapisz ramki ROI**.

Współrzędne ramek zostaną trwale zapisane w pamięci dodatku (`/data/settings.json`).

---

## Krok 4: Zbieranie próbek treningowych i eksport archiwum ZIP

Przed rozpoczęciem odczytów należy przygotować mały zbiór zdjęć do dotrenowania modelu pod specyfikę Twojego wyświetlacza (kąt kamery, odbicia światła, czcionka LCD):

1. W karcie **📦 Zbiór Treningowy (Collector)** włącz przełącznik:
   * **Automatycznie zbieraj próbki w tle (Collector Mode)**
2. Dodatek w każdym cyklu pracy będzie porównywał wycinki z poprzednimi klatkami i zapisywał nową próbkę tylko wtedy, gdy na ekranie pojawi się inna cyfra (wbudowana anty-duplikacja).
3. Możesz również w dowolnym momencie kliknąć **📸 Złap próbkę teraz**, aby natychmiast zapisać bieżący kadr.
4. Pasek postępu pokazuje zapełnienie bufora (maksymalnie 300 plików i 100 MB pamięci).
5. Gdy zbierzesz kilkadziesiąt próbek (reprezentujących różne cyfry), kliknij **📦 Pobierz paczkę próbek (.zip)**.

---

## Krok 5: Wytrenowanie i wgranie modelu TensorFlow Lite

1. Rozpakuj pobrane archiwum `.zip` na swoim komputerze PC i oznacz cyfry prostokątami w darmowym narzędziu (np. [MakeSense.ai](https://www.makesense.ai/) lub Label Studio), używając dokładnie 10 klas cyfr: `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`.
2. Uruchom skrypty treningowe z katalogu `training/` repozytorium (zobacz szczegóły w instrukcji `training/README_TRAINING.md`).
3. Po zakończeniu treningu skrypt automatycznie wygeneruje plik `training/dist/active_model.tflite`.
4. W interfejsie dodatku w Home Assistant przejdź do karty **🧠 Model TensorFlow Lite (.tflite)**:
   - Kliknij pole wyboru pliku i wskaż wygenerowany plik `active_model.tflite`.
   - Kliknij **⬆️ Wgraj model .tflite**.
5. Dodatek zweryfikuje nagłówek modelu (`TFL3`) i automatycznie wczyta go do pamięci RAM. Status modelu zmieni się na *Model aktywny*.

---

## Krok 6: Diagnostyka i publikowane sensory w Home Assistant

W karcie **🔬 Diagnostyka i Podgląd Odczytu na Żywo** kliknij **🧪 Wymuś testowy odczyt teraz**.

Zobaczysz:
- Wycięte fragmenty LCD z nałożonymi zielonymi ramkami detekcji i rozpoznanymi cyframi,
- Surową wartość złożoną z cyfr od lewej do prawej (`raw_value`),
- Wartość po filtracji fizycznej (`validated_value`),
- Średnią pewność modelu w procentach (`confidence`),
- Status walidacji (`ok`, `rejected_delta_pending`, itp.).

### Publikowane encje:

W Home Assistant automatycznie tworzone i aktualizowane są dwa sensory:
- `sensor.boiler_co_temperature` – Temperatura kotła CO
- `sensor.boiler_cwu_temperature` – Temperatura wody CWU

Obie encje posiadają atrybuty:
- `device_class: temperature`
- `state_class: measurement` (zapewnia pełną historię oraz długoterminowe statystyki w panelu *Energia* i na kartach wykresów ApexCharts / History Graph)
- `unit_of_measurement: "°C"`
- Wartość stanu to zawsze **liczba całkowita `int`** (np. `48`, nigdy float z miejscami po przecinku).

---

## Najczęściej zadawane pytania (FAQ)

**P: Dlaczego temperatura nie zaktualizowała się natychmiast po nagłym skoku o 20°C?**  
O: Działa filtr *Sanity Check*. Nagłe skoki powyżej wartości `max_delta` (domyślnie 15°C) są traktowane jako potencjalne zakłócenie wizualne (np. cień ręki) i wymagają potwierdzenia w 3 kolejnych klatkach. Jeśli temperatura rzeczywiście wzrosła, sensor zaktualizuje się po 3 cyklach.

**P: Czy dodatek obciąża procesor Home Assistanta?**  
O: Nie. Model `yolo11n` w formacie TFLite jest zoptymalizowany pod kątem CPU. Pojedyncza inferencja na klatce 320x320 zajmuje zaledwie kilkanaście milisekund, a odczyt wykonywany jest tylko raz na interwał (np. co 30 sekund).
