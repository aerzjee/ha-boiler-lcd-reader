# Changelog – Boiler LCD Reader

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.2] - 2026-10-15

### Fixed
- **SampleCollector Threading & Coroutine Lifecycle:**
  - Converted `SampleCollector.process_and_save_background_sample` and `capture_now` into synchronous `def` methods using `threading.Lock` to resolve `RuntimeWarning: coroutine was never awaited` when executed in worker threads.
  - Ensured all blocking CPU/IO operations in `SampleCollector` and `TFLiteInferenceEngine` are invoked via `await asyncio.to_thread(...)` in `main.py` without blocking the FastAPI event loop.

## [0.1.1] - 2026-10-15

### Fixed
- **Container Entrypoint & Startup Permissions:**
  - Explicitly cleared inherited s6-overlay base image entrypoint in `Dockerfile` (`ENTRYPOINT []`) to resolve `/bin/sh: 0: cannot open /init: Permission denied` when running with `init: false`.
  - Updated `run.sh` entrypoint to use standard `#!/usr/bin/env bash` with `set -e`, removing dependencies on `with-contenv` and `bashio`.
  - Added options parsing in `run.sh` reading `/data/options.json` directly via Python with fallback to `info`.
  - Disabled custom AppArmor profile (`apparmor: false` in `config.yaml` and removed `apparmor.txt`) to allow standard Supervisor AppArmor confinement without blocking `/bin/bash` or Python runtime libraries.

## [0.1.0] - 2026-10-15

### Added
- **Initial release of Boiler LCD Reader (TFLite) Home Assistant Add-on**.
- **Ingress Web Dashboard:**
  - Interactive HTML5 Canvas for real-time ROI bounding box drawing (**🔴 ROI CO** and **🟢 ROI CWU**) directly over camera frames.
  - Full configuration management for Home Assistant camera entity, scan interval, sanity check thresholds, and sensor entity names.
  - Live diagnostics card displaying annotated crop images with detection boxes, raw values, validated values, and confidence percentages.
- **TFLite Inference Engine:**
  - Singleton `TFLiteInferenceEngine` supporting `ai-edge-litert`, `tflite-runtime`, and `tensorflow.lite` fallback pipelines.
  - Letterbox preprocessing (padding to square dimension, RGB conversion, float32 normalization / int8 quantization).
  - YOLO tensor decoding with Non-Maximum Suppression (`cv2.dnn.NMSBoxes`) and left-to-right digit sorting by horizontal center.
  - Physical plausibility Sanity Check filtering out-of-range temperatures and sudden jumps exceeding `max_delta` unless confirmed across 3 consecutive frames.
  - Strict integer-only temperature outputs (`int`, no decimal places).
- **Collector Mode:**
  - Automated background sample harvester with change detection (`cv2.absdiff`) to prevent redundant frames.
  - Storage cap at 300 sample files (100 MB max) with automated FIFO rotation.
  - On-demand single-frame capture and one-click ZIP export.
- **Model Management:**
  - Safe `.tflite` model upload endpoint with FlatBuffer `TFL3` magic header validation and 25 MB quota enforcement.
  - Automatic model reload in memory upon upload.
- **Home Assistant Core Integration:**
  - Zero-Trust internal Supervisor authentication using `SUPERVISOR_TOKEN`.
  - Camera snapshot proxy fetching via `GET /core/api/camera_proxy/{camera_entity}`.
  - Temperature state publication via `POST /core/api/states/{entity}` with `device_class: temperature` and `state_class: measurement`.
- **Training Pipeline (`training/`):**
  - Standalone YOLO11 training scripts optimized for PC workstations (NVIDIA GeForce RTX 5080 + AMD Ryzen 9 9950X3D).
  - Data preparation and validation CLI utility (`training/prepare_dataset.py`) for YOLO format verification and 85/15 train/val splitting.
  - Automated TFLite export and sanity verification script (`training/train_and_export.py`).
- **CI/CD Workflows:**
  - GitHub Actions `lint.yaml` running Ruff and Bandit security analysis on push/PR to main branch.
  - GitHub Actions `build.yaml` automatically building multi-arch/amd64 Docker images for GHCR upon tag push (`v*.*.*`).
