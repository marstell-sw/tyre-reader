# tyre_reader_v3

`tyre_reader_v3` e un progetto C++17 senza Qt per leggere da immagini di pneumatici auto nuovi:

- misura del pneumatico, per esempio `225/40 R18`
- codice DOT, almeno le ultime 4 cifre settimana/anno, e quando possibile il DOT completo

L'architettura separa la pipeline di analisi dalla CLI, cosi la stessa logica puo essere riusata in futuro su frame estratti da video.

## Dipendenze

Su Ubuntu o Debian:

```bash
sudo apt update
sudo apt install -y cmake pkg-config libopencv-dev libtesseract-dev libleptonica-dev
```

Per il runtime OCR serve anche almeno un language pack Tesseract:

```bash
sudo apt install -y tesseract-ocr tesseract-ocr-eng
```

Se il language pack non e disponibile a livello di sistema, l'app prova anche un fallback locale in `./tessdata`.

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

L'eseguibile generato e:

```bash
./build/tyre_reader_v3
```

## Uso CLI

Analisi di una singola immagine:

```bash
./build/tyre_reader_v3 --image /path/to/image.png --output /tmp/tyre_out --pretty
```

Analisi di una cartella:

```bash
./build/tyre_reader_v3 --dir /path/to/images --output /tmp/tyre_out --pretty
```

Benchmark su dataset:

```bash
./build/tyre_reader_v3 --dataset /path/to/dataset_root --output /tmp/tyre_benchmark --pretty
```

Opzioni supportate:

- `--image <file>`: analizza una sola immagine e stampa JSON
- `--dir <folder>`: analizza tutte le immagini supportate in una cartella e stampa un JSON array
- `--dataset <dataset_root>`: esegue benchmark contro `expected.csv`
- `--output <folder>`: cartella di output per crop, overlay e CSV
- `--pretty`: formatta il JSON su piu righe

## Struttura del progetto

```text
.
├── CMakeLists.txt
├── README.md
├── dataset_example
│   └── expected.csv
├── include
│   ├── DatasetBenchmark.h
│   ├── ImagePreprocessor.h
│   ├── TesseractOcrEngine.h
│   ├── TyreAnalyzer.h
│   └── Types.h
└── src
    ├── DatasetBenchmark.cpp
    ├── ImagePreprocessor.cpp
    ├── TesseractOcrEngine.cpp
    ├── TyreAnalyzer.cpp
    └── main.cpp
```

## Architettura

### `TyreAnalyzer`

Classe principale e riusabile. Espone:

- `analyzeImageFile(...)`
- `analyzeDirectory(...)`
- `analyzeFrame(...)`

`analyzeFrame` e pensato per il futuro caso video, in cui un frame gia acquisito viene passato direttamente alla pipeline.

### `ImagePreprocessor`

Implementa funzioni piccole e riusabili:

- grayscale
- deskew leggero
- resize/upscale
- CLAHE
- bilateral denoise
- adaptive threshold
- inversione
- morphology close/open
- ROI proposal text-like con gradienti, threshold, contorni e merge di box vicini

### `TesseractOcrEngine`

Incapsula Tesseract e Leptonica:

- inizializzazione OCR
- OCR su `cv::Mat`
- confidenza media
- OCR multi-variante su preprocessing differenti

### `DatasetBenchmark`

Carica `expected.csv`, analizza le immagini del dataset e produce:

- `summary.csv`
- `per_image.csv`
- `errors.csv`

## Struttura dataset

La modalita benchmark si aspetta:

- `dataset_root/expected.csv` insieme a `dataset_root/images/`
- oppure `dataset_root/expected.csv` insieme a `dataset_root/train/`

```text
dataset_root/
├── expected.csv
└── images/
    ├── image001.png
    ├── image002.jpg
    └── ...
```

`expected.csv` deve contenere almeno le colonne:

- `filename`
- `expected_size`
- `expected_dot4`
- `expected_dot_full`
- `brand`
- `model`
- `difficulty`
- `notes`

## Pipeline di analisi

1. preprocessing iniziale dell'immagine
2. ROI proposal con OpenCV, non OCR sull'intera immagine soltanto
3. OCR multi-variante su ogni ROI candidata
4. parsing regex della misura pneumatico
5. parsing regex DOT con estrazione del blocco finale settimana/anno
6. scoring euristico di confidenza e incertezza
7. salvataggio di crop e overlay diagnostici

## Output strutturato

Ogni `AnalysisResult` include almeno:

- `inputPath` o `frameId`
- risultato misura raw e normalizzato
- risultato DOT raw e normalizzato
- flag `tyreSizeFound`, `dotFound`, `dotWeekYearFound`, `dotFullFound`
- confidence e uncertainty separate
- path dei crop salvati
- path overlay
- timings dettagliati
- warning o note

## Metriche benchmark

`summary.csv` include:

- `totalImages`
- `sizeDetectedCount`
- `sizeExactMatchCount`
- `dotDetectedCount`
- `dot4MatchCount`
- `dotFullMatchCount`
- `avgSizeConfidence`
- `avgDotConfidence`
- `avgTotalMs`
- `p50TotalMs`
- `p90TotalMs`
- `maxTotalMs`

`errors.csv` include:

- `filename`
- `expected_size`
- `predicted_size`
- `expected_dot4`
- `predicted_dot4`
- `expected_dot_full`
- `predicted_dot_full`
- `size_confidence`
- `dot_confidence`
- `total_ms`
- `suspected_reason`

Classi di errore euristiche:

- `roi_not_found`
- `weak_ocr`
- `parse_failure`
- `partial_dot_only`
- `image_quality_issue`
- `unknown`

## Output immagini

Per ogni immagine vengono salvati:

- crop misura: `*_size_crop.png`
- crop DOT: `*_dot_crop.png`
- overlay diagnostico: `*_overlay.png`

Se una ROI specifica non viene trovata, il programma salva comunque un'immagine placeholder coerente.

## Limiti attuali

- il fianco del pneumatico ha spesso testo scuro in rilievo su fondo scuro: il caso resta difficile e la robustezza dipende molto da illuminazione e contrasto
- la ROI proposal e generica e non usa ancora modelli di detection dedicati
- il DOT completo puo risultare parziale o spezzato quando il layout e curvo o molto vicino al bordo dell'immagine
- il parser misura copre i pattern auto piu comuni, non casi camion o marcature specialistiche

## Idee future

- supporto video con campionamento frame e voting temporale
- tracking ROI tra frame consecutivi
- fusion di OCR multipli su stesso pneumatico
- detector dedicato per testi embossati sul fianco
- calibrazione dinamica delle soglie di confidenza
