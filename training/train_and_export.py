"""Main training and TFLite export script for Boiler LCD Reader.

Multi-platform support:
  - NVIDIA GeForce RTX 5080 (Blackwell architecture, CUDA) + AMD Ryzen 9 9950X3D
  - Intel Core Ultra (Intel Arc 140T GPU / XPU + Intel AI Boost NPU via OpenVINO)
  - Standard x86_64 CPU (fallback)

Trains a YOLO11 object detection model for 10 digit classes (0-9) on boiler LCD crops,
exports the best weights to TensorFlow Lite (TFLite v3 FlatBuffer), optionally runs
benchmarks on Intel AI Boost NPU and Intel Arc GPU via OpenVINO, verifies compatibility
with the Home Assistant add-on backend, and places the active model in
training/dist/active_model.tflite.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_and_export")

# ---------------------------------------------------------------------------
# TFLite Interpreter Import
# ---------------------------------------------------------------------------

Interpreter = None
try:
    from ai_edge_litert.interpreter import Interpreter

    logger.debug("Using ai-edge-litert for model verification")
except ImportError:
    try:
        import tensorflow as tf

        Interpreter = tf.lite.Interpreter
        logger.debug("Using tensorflow.lite for model verification")
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter

            logger.debug("Using tflite-runtime for model verification")
        except ImportError:
            Interpreter = None
            logger.warning("No TFLite interpreter available. Post-export inference check will be skipped.")


# ---------------------------------------------------------------------------
# Hardware & Accelerator Detection
# ---------------------------------------------------------------------------


def detect_hardware(
    requested_device: str = "auto",
    requested_batch: int = 0,
    requested_workers: int = -1,
) -> tuple[str, str, int, int, bool, bool]:
    """Detect available hardware accelerator and configure optimal training parameters.

    Args:
        requested_device: CLI device argument ("auto", "cuda", "0", "xpu", "npu", "cpu").
        requested_batch: Requested batch size (0 for auto).
        requested_workers: Requested worker threads (-1 for auto).

    Returns:
        Tuple of:
          - train_device: String device passed to Ultralytics model.train()
          - device_desc: Human-readable device description
          - batch: Resolved batch size
          - workers: Resolved worker count
          - amp: Whether to enable Automatic Mixed Precision (AMP)
          - npu_benchmark: Whether to run OpenVINO NPU benchmark
    """
    try:
        import torch
    except ImportError as exc:
        logger.critical(
            "Missing PyTorch! Please install PyTorch first:\n"
            "  NVIDIA GPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124\n"
            "  Intel Arc:  pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu\n"
            "  CPU:        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
        )
        raise exc

    device_norm = requested_device.lower().strip()
    train_device = "cpu"
    device_desc = "CPU"
    default_batch = 16
    default_workers = 4
    amp = False
    npu_benchmark = False

    has_cuda = torch.cuda.is_available()
    has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()

    if device_norm == "npu":
        logger.info("=" * 60)
        logger.info("Intel AI Boost (NPU) accelerator mode selected.")
        logger.info("NPU is a dedicated low-power inference engine optimized for OpenVINO.")
        logger.info("Backpropagation training will run on Intel Arc GPU (XPU) if available, or CPU.")
        logger.info("Post-training validation and benchmark will run directly on the NPU.")
        logger.info("=" * 60)
        npu_benchmark = True

        if has_xpu:
            xpu_name = torch.xpu.get_device_name(0)
            train_device = "xpu"
            device_desc = f"Intel Arc (XPU): {xpu_name}"
            default_batch = 32
            default_workers = 8
            amp = True
        else:
            train_device = "cpu"
            device_desc = "Host CPU (NPU inference target)"
            default_batch = 16
            default_workers = 4
            amp = False

    elif device_norm == "xpu":
        if has_xpu:
            xpu_name = torch.xpu.get_device_name(0)
            train_device = "xpu"
            device_desc = f"Intel Arc (XPU): {xpu_name}"
            default_batch = 32
            default_workers = 8
            amp = True
        else:
            logger.warning(
                "Intel XPU requested, but torch.xpu.is_available() is False! Falling back to CPU.\n"
                "To enable Intel Arc GPU training, install: "
                "pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu"
            )
            train_device = "cpu"
            device_desc = "CPU (Fallback from XPU)"
            default_batch = 16
            default_workers = 4
            amp = False

    elif device_norm in ("cuda", "0") or device_norm.isdigit():
        if has_cuda:
            train_device = device_norm if device_norm != "cuda" else "0"
            gpu_idx = int(train_device) if train_device.isdigit() else 0
            gpu_name = torch.cuda.get_device_name(gpu_idx)
            vram_gb = torch.cuda.get_device_properties(gpu_idx).total_memory / (1024**3)
            device_desc = f"NVIDIA CUDA: {gpu_name} ({vram_gb:.1f} GB VRAM)"
            default_batch = 64
            default_workers = 12
            amp = True
        else:
            logger.warning(
                "CUDA requested (%s), but torch.cuda.is_available() is False. Falling back to CPU.",
                device_norm,
            )
            train_device = "cpu"
            device_desc = "CPU (Fallback from CUDA)"
            default_batch = 16
            default_workers = 4
            amp = False

    elif device_norm == "cpu":
        train_device = "cpu"
        device_desc = "CPU"
        default_batch = 16
        default_workers = 4
        amp = False

    else:  # "auto" or unrecognized
        if has_cuda:
            train_device = "0"
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            device_desc = f"NVIDIA CUDA: {gpu_name} ({vram_gb:.1f} GB VRAM)"
            default_batch = 64
            default_workers = 12
            amp = True
        elif has_xpu:
            train_device = "xpu"
            xpu_name = torch.xpu.get_device_name(0)
            device_desc = f"Intel Arc (XPU): {xpu_name}"
            default_batch = 32
            default_workers = 8
            amp = True
        else:
            train_device = "cpu"
            device_desc = "CPU"
            default_batch = 16
            default_workers = 4
            amp = False

    final_batch = requested_batch if requested_batch > 0 else default_batch
    final_workers = requested_workers if requested_workers >= 0 else default_workers

    logger.info("Hardware Auto-Detection Summary:")
    logger.info("  Selected Accelerator: %s", device_desc)
    logger.info("  Training Device:      %s", train_device)
    req_batch_str = "auto" if requested_batch == 0 else str(requested_batch)
    req_workers_str = "auto" if requested_workers == -1 else str(requested_workers)
    logger.info("  Resolved Batch Size:  %d (requested: %s)", final_batch, req_batch_str)
    logger.info("  Worker Threads:       %d (requested: %s)", final_workers, req_workers_str)
    logger.info("  Mixed Precision (AMP):%s", amp)

    return train_device, device_desc, final_batch, final_workers, amp, npu_benchmark


# ---------------------------------------------------------------------------
# OpenVINO Benchmark Routine (Intel AI Boost NPU & Intel Arc GPU)
# ---------------------------------------------------------------------------


def check_openvino_devices() -> list[str]:
    """Return list of available OpenVINO devices, e.g. ['CPU', 'GPU', 'NPU']."""
    try:
        import openvino as ov

        core = ov.Core()
        devices = list(core.available_devices)
        logger.info("OpenVINO detected available devices: %s", devices)
        return devices
    except ImportError:
        logger.debug("OpenVINO is not installed in the current environment.")
        return []
    except Exception as exc:
        logger.warning("Error checking OpenVINO devices: %s", exc)
        return []


def run_openvino_benchmarks(model_xml: Path, imgsz: int = 320, iterations: int = 30) -> dict[str, float]:
    """Run inference latency benchmarks across available OpenVINO devices.

    Tests Intel AI Boost (NPU), Intel Arc (GPU), and Host CPU if detected.

    Args:
        model_xml: Path to exported OpenVINO .xml model file.
        imgsz: Model input image dimension.
        iterations: Number of inference iterations for averaging.

    Returns:
        Dictionary mapping device name to average latency in milliseconds.
    """
    try:
        import openvino as ov
    except ImportError:
        logger.warning("OpenVINO is not installed. Skipping OpenVINO NPU/GPU benchmark.")
        return {}

    results: dict[str, float] = {}

    try:
        core = ov.Core()
        available = core.available_devices
        logger.info("=" * 60)
        logger.info("Starting OpenVINO Inference Benchmark (%s)", model_xml.name)
        logger.info("Available OpenVINO devices: %s", available)

        # Priority test order: NPU (Intel AI Boost), GPU (Intel Arc), CPU
        targets: list[tuple[str, str]] = []
        if "NPU" in available:
            targets.append(("NPU", "Intel AI Boost (NPU)"))
        if "GPU" in available:
            targets.append(("GPU", "Intel Arc Graphics (GPU)"))
        if "CPU" in available:
            targets.append(("CPU", "Host CPU"))

        dummy_input = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)

        for dev_code, dev_name in targets:
            try:
                logger.info("Compiling model for %s (%s)...", dev_name, dev_code)
                compiled_model = core.compile_model(model=str(model_xml), device_name=dev_code)
                infer_req = compiled_model.create_infer_request()

                # Warmup iterations
                for _ in range(5):
                    infer_req.infer({0: dummy_input})

                # Timed inference loop
                t0 = time.perf_counter()
                for _ in range(iterations):
                    infer_req.infer({0: dummy_input})
                t1 = time.perf_counter()

                latency_ms = ((t1 - t0) / iterations) * 1000.0
                fps = 1000.0 / latency_ms if latency_ms > 0 else 0
                results[dev_code] = latency_ms

                logger.info(
                    "✓ %s (%s) Latency: %.2f ms / frame (%.1f FPS)",
                    dev_name,
                    dev_code,
                    latency_ms,
                    fps,
                )
            except Exception as dev_exc:
                logger.warning("Benchmark failed on %s (%s): %s", dev_name, dev_code, dev_exc)

        logger.info("=" * 60)
    except Exception as exc:
        logger.warning("OpenVINO benchmark error: %s", exc)

    return results


# ---------------------------------------------------------------------------
# Model Verification Routine (TFLite for Home Assistant)
# ---------------------------------------------------------------------------


def verify_exported_tflite(model_path: Path, expected_classes: int = 10) -> bool:
    """Verify that an exported .tflite model matches Home Assistant add-on requirements.

    Checks:
    1. File size > 0 and <= 25 MB quota.
    2. Header contains 'TFL3' FlatBuffer magic bytes at offset 4..8.
    3. Model loads into TFLite Interpreter and allocates tensors without errors.
    4. Output tensor has either (1, 14, N) or (1, N, 14) dimension (where 14 = 4 bbox + 10 classes).

    Returns:
        True if all checks pass, False otherwise.
    """
    logger.info("Running verification on exported model: %s", model_path)

    if not model_path.exists():
        logger.error("Model file does not exist: %s", model_path)
        return False

    raw_data = model_path.read_bytes()
    size_kb = len(raw_data) / 1024
    logger.info("Model file size: %.2f KB (%.2f MB)", size_kb, size_kb / 1024)

    # 1. Quota check
    if len(raw_data) > 25 * 1024 * 1024:
        logger.error("Model size exceeds maximum 25 MB quota!")
        return False

    # 2. FlatBuffer TFL3 magic header verification (matching main.py upload validator)
    if len(raw_data) < 8 or raw_data[4:8] != b"TFL3":
        logger.error("Invalid FlatBuffer header: expected b'TFL3' at offset 4..8, got %r", raw_data[4:8])
        return False
    logger.info("✓ FlatBuffer header verified: b'TFL3' present at offset 4..8")

    # 3. Interpreter allocation and tensor shape check
    if Interpreter is None:
        logger.warning("Skipping tensor execution test (no TFLite interpreter library installed).")
        return True

    try:
        interp = Interpreter(model_path=str(model_path))
        interp.allocate_tensors()

        input_details = interp.get_input_details()
        output_details = interp.get_output_details()

        in_shape = tuple(input_details[0]["shape"])
        in_dtype = input_details[0]["dtype"]
        logger.info("✓ Model input tensor: shape=%s, dtype=%s", in_shape, in_dtype.__name__)

        out_shape = tuple(output_details[0]["shape"])
        out_dtype = output_details[0]["dtype"]
        logger.info("✓ Model output tensor: shape=%s, dtype=%s", out_shape, out_dtype.__name__)

        # 4. Dummy inference test
        dummy_input = np.zeros(in_shape, dtype=in_dtype)
        interp.set_tensor(input_details[0]["index"], dummy_input)
        interp.invoke()
        output = interp.get_tensor(output_details[0]["index"])

        expected_channels = 4 + expected_classes  # 4 bbox + 10 classes = 14
        out_squeezed = np.squeeze(output)

        if out_squeezed.ndim == 2:
            dim0, dim1 = out_squeezed.shape
            if dim0 == expected_channels:
                logger.info(
                    "✓ Output layout matches (1, %d, N) [channels-first: 4 bbox + %d classes, %d anchors]",
                    expected_channels,
                    expected_classes,
                    dim1,
                )
            elif dim1 == expected_channels:
                logger.info(
                    "✓ Output layout matches (1, N, %d) [channels-last: %d anchors, 4 bbox + %d classes]",
                    expected_channels,
                    dim0,
                    expected_classes,
                )
            else:
                logger.error(
                    "Output tensor shape %s does not contain dimension with %d channels!",
                    out_squeezed.shape,
                    expected_channels,
                )
                return False
        else:
            logger.warning("Unusual squeezed output dimensions: %s", out_squeezed.shape)

        logger.info("✓ Test inference on zero input succeeded with output shape %s", output.shape)
        return True

    except Exception as exc:
        logger.exception("Failed to verify model in TFLite interpreter: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Training & Export Routine
# ---------------------------------------------------------------------------


def run_training_and_export(
    model_name: str = "yolo11n.pt",
    data_config: Path = Path("dataset.yaml"),
    epochs: int = 150,
    patience: int = 30,
    imgsz: int = 320,
    batch: int = 0,
    workers: int = -1,
    device: str = "auto",
    run_name: str = "boiler_yolo11n",
    output_dist_dir: Path = Path("dist"),
    npu_benchmark: bool = False,
    export_openvino: bool = False,
) -> Path:
    """Train YOLO11 digit detector and export to active_model.tflite."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        logger.critical("Missing Ultralytics! Please install dependencies:\n  pip install -r requirements-training.txt")
        raise exc

    # Hardware & accelerator auto-detection
    train_device, device_desc, final_batch, final_workers, amp, auto_npu_bench = detect_hardware(
        requested_device=device,
        requested_batch=batch,
        requested_workers=workers,
    )
    should_run_npu_bench = npu_benchmark or auto_npu_bench

    # Check for OpenVINO devices in the system
    ov_devices = check_openvino_devices()
    if "NPU" in ov_devices and not should_run_npu_bench and device.lower().strip() == "auto":
        logger.info("Intel AI Boost (NPU) detected via OpenVINO. Enabling automatic NPU benchmark.")
        should_run_npu_bench = True

    # Resolve dataset config path
    if not data_config.is_absolute():
        script_dir = Path(__file__).parent
        data_config_path = (script_dir / data_config).resolve()
    else:
        data_config_path = data_config.resolve()

    if not data_config_path.exists():
        raise FileNotFoundError(f"Dataset configuration file not found at: {data_config_path}")

    logger.info("=" * 60)
    logger.info("Starting Boiler LCD YOLO11 Training Pipeline")
    logger.info("Base Model:       %s", model_name)
    logger.info("Dataset YAML:     %s", data_config_path)
    logger.info("Epochs:           %d (patience: %d)", epochs, patience)
    logger.info("Input Size:       %d x %d", imgsz, imgsz)
    logger.info("Batch Size:       %d", final_batch)
    logger.info("Worker Threads:   %d", final_workers)
    logger.info("Training Device:  %s (%s)", train_device, device_desc)
    logger.info("Mixed Precision:  %s", amp)
    logger.info("=" * 60)

    # Initialize Ultralytics model
    model = YOLO(model_name)

    # -------------------------------------------------------------------------
    # Train with augmentations tailored specifically for boiler LCD screens:
    # CRITICAL: fliplr=0.0, flipud=0.0 (never mirror digits like 2, 5, 3)
    # -------------------------------------------------------------------------
    train_kwargs: dict[str, Any] = {
        "data": str(data_config_path),
        "epochs": epochs,
        "patience": patience,
        "imgsz": imgsz,
        "batch": final_batch,
        "workers": final_workers,
        "device": train_device,
        "name": run_name,
        "amp": amp,
        # Augmentations
        "fliplr": 0.0,  # CRITICAL: No horizontal flipping of numbers
        "flipud": 0.0,  # CRITICAL: No vertical flipping of numbers
        "degrees": 5.0,  # Slight camera angle misalignment
        "translate": 0.1,  # Slight vibration / crop shift
        "scale": 0.2,  # Slight zoom variation
        "hsv_h": 0.015,  # Minimal hue shift (LCD is blue)
        "hsv_s": 0.4,  # Saturation variation
        "hsv_v": 0.4,  # Brightness variation (day vs night backlight)
        "save": True,
        "save_period": -1,
        "exist_ok": True,
    }

    try:
        train_results = model.train(**train_kwargs)
    except Exception as exc:
        # Graceful fallback if AMP or device driver fails on specific hardware
        if amp and ("amp" in str(exc).lower() or "float16" in str(exc).lower()):
            logger.warning("AMP training encountered a hardware driver issue: %s. Retrying with amp=False...", exc)
            train_kwargs["amp"] = False
            train_results = model.train(**train_kwargs)
        else:
            raise exc

    logger.info("Training completed. Results summary: %s", train_results)

    # Locate best.pt
    if hasattr(model, "trainer") and model.trainer:
        save_dir = Path(model.trainer.save_dir)
    else:
        save_dir = Path("runs/detect") / run_name
    best_pt = save_dir / "weights" / "best.pt"

    if not best_pt.exists():
        fallback_pt = save_dir / "weights" / "last.pt"
        if fallback_pt.exists():
            logger.warning("best.pt not found, using last.pt: %s", fallback_pt)
            best_pt = fallback_pt
        else:
            raise FileNotFoundError(f"Could not find trained weights at {best_pt}")

    logger.info("Found trained weights checkpoint at: %s", best_pt)
    best_model = YOLO(str(best_pt))

    # -------------------------------------------------------------------------
    # Optional OpenVINO Export & Intel AI Boost NPU Benchmark
    # -------------------------------------------------------------------------
    if should_run_npu_bench or export_openvino:
        logger.info("Exporting %s to OpenVINO format (half=True)...", best_pt.name)
        try:
            ov_result = best_model.export(
                format="openvino",
                imgsz=imgsz,
                half=True,
            )
            logger.info("OpenVINO export artifact: %s", ov_result)

            # Locate exported .xml file
            xml_candidates: list[Path] = []
            if ov_result and Path(str(ov_result)).is_dir():
                xml_candidates.extend(Path(str(ov_result)).glob("*.xml"))
            elif ov_result and Path(str(ov_result)).is_file() and str(ov_result).endswith(".xml"):
                xml_candidates.append(Path(str(ov_result)))
            xml_candidates.extend(best_pt.parent.glob("**/*.xml"))

            if xml_candidates:
                ov_xml = xml_candidates[0]
                logger.info("Located OpenVINO model XML: %s", ov_xml)
                if should_run_npu_bench:
                    run_openvino_benchmarks(ov_xml, imgsz=imgsz)
            else:
                logger.warning("Could not find generated OpenVINO .xml file.")
        except Exception as ov_exc:
            logger.warning("OpenVINO export/benchmark failed (non-critical): %s", ov_exc)

    # -------------------------------------------------------------------------
    # Primary Export: TensorFlow Lite (TFLite for Home Assistant)
    # -------------------------------------------------------------------------
    logger.info("Exporting %s to TensorFlow Lite format (imgsz=%d)...", best_pt.name, imgsz)

    export_result = best_model.export(
        format="tflite",
        imgsz=imgsz,
        half=False,  # Float32 model for best numerical stability on Home Assistant
    )
    logger.info("Ultralytics TFLite export return path: %s", export_result)

    # Locate generated .tflite file
    tflite_candidates: list[Path] = []
    if export_result and Path(str(export_result)).is_file():
        tflite_candidates.append(Path(str(export_result)))
    elif export_result and Path(str(export_result)).is_dir():
        tflite_candidates.extend(Path(str(export_result)).glob("*.tflite"))

    weights_dir = best_pt.parent
    tflite_candidates.extend(weights_dir.glob("**/*.tflite"))

    unique_candidates = sorted(set(tflite_candidates), key=lambda p: str(p))

    # Prioritize float32 model
    chosen_tflite: Path | None = None
    for cand in unique_candidates:
        if "float32" in cand.name.lower():
            chosen_tflite = cand
            break
    if chosen_tflite is None and unique_candidates:
        chosen_tflite = unique_candidates[0]

    if chosen_tflite is None or not chosen_tflite.exists():
        raise FileNotFoundError(f"Could not locate exported .tflite file in {weights_dir}")

    logger.info("Selected exported TFLite file: %s", chosen_tflite)

    # -------------------------------------------------------------------------
    # Copy to target distribution directory (active_model.tflite)
    # -------------------------------------------------------------------------
    if not output_dist_dir.is_absolute():
        dist_dir = (Path(__file__).parent / output_dist_dir).resolve()
    else:
        dist_dir = output_dist_dir.resolve()

    dist_dir.mkdir(parents=True, exist_ok=True)
    target_active_model = dist_dir / "active_model.tflite"

    shutil.copy2(chosen_tflite, target_active_model)
    logger.info("✓ Successfully placed active model at: %s", target_active_model)

    # -------------------------------------------------------------------------
    # Self-Verification Sanity Check
    # -------------------------------------------------------------------------
    verification_passed = verify_exported_tflite(target_active_model, expected_classes=10)
    if not verification_passed:
        logger.error("Verification checks failed on the exported model!")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("PIPELINE SUCCESS!")
    logger.info("TFLite Model Ready for Home Assistant Ingress Upload:")
    logger.info("  Path: %s", target_active_model)
    logger.info("You can now upload this file in the Home Assistant Web UI:")
    logger.info("  Card 4 -> '⬆️ Wgraj model .tflite'")
    logger.info("=" * 60)

    return target_active_model


def main() -> None:
    """CLI entry point for training and export pipeline."""
    parser = argparse.ArgumentParser(
        description="Train YOLO11 digit detector and export to TFLite (and OpenVINO) for Boiler LCD Reader."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Base pretrained YOLO11 model weights (default: yolo11n.pt).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("dataset.yaml"),
        help="Path to dataset.yaml configuration (default: dataset.yaml).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Total training epochs (default: 150).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience in epochs (default: 30).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Square input image size in pixels (default: 320).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=0,
        help="Batch size (default: 0 for auto-tuning: 64 for CUDA, 32 for XPU, 16 for CPU).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="DataLoader worker threads (default: -1 for auto-tuning: 12 for CUDA, 8 for XPU, 4 for CPU).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=(
            "Accelerator device: 'auto' (detects CUDA -> XPU -> CPU), 'cuda' / '0', "
            "'xpu' (Intel Arc GPU), 'npu' (Intel AI Boost), or 'cpu'."
        ),
    )
    parser.add_argument(
        "--name",
        type=str,
        default="boiler_yolo11n",
        help="Experiment name for runs directory (default: 'boiler_yolo11n').",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dist"),
        help="Target distribution directory for active_model.tflite (default: 'dist').",
    )
    parser.add_argument(
        "--npu-benchmark",
        action="store_true",
        default=False,
        help="Run inference latency benchmark on Intel AI Boost NPU and Intel Arc GPU via OpenVINO.",
    )
    parser.add_argument(
        "--export-openvino",
        action="store_true",
        default=False,
        help="Export trained weights to OpenVINO FP16 format in addition to TFLite.",
    )

    args = parser.parse_args()

    run_training_and_export(
        model_name=args.model,
        data_config=args.data,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        run_name=args.name,
        output_dist_dir=args.output_dir,
        npu_benchmark=args.npu_benchmark,
        export_openvino=args.export_openvino,
    )


if __name__ == "__main__":
    main()
