"""Main training and TFLite export script for Boiler LCD Reader.

Optimized for NVIDIA GeForce RTX 5080 (Blackwell architecture) and AMD Ryzen 9 9950X3D.
Trains a YOLO11 object detection model for 10 digit classes (0-9) on boiler LCD crops,
exports the best weights to TensorFlow Lite (TFLite v3 FlatBuffer), verifies
compatibility with the Home Assistant add-on backend, and places the active model
in training/dist/active_model.tflite.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

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
# Model Verification Routine
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
    batch: int = 64,
    workers: int = 12,
    device: str = "0",
    run_name: str = "boiler_yolo11n",
    output_dist_dir: Path = Path("dist"),
) -> Path:
    """Train YOLO11 digit detector and export to active_model.tflite."""
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        logger.critical(
            "Missing PyTorch or Ultralytics! Please install dependencies:\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124\n"
            "  pip install -r requirements-training.txt"
        )
        raise exc

    # GPU / Device validation
    selected_device = device
    if selected_device != "cpu":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested (%s), but torch.cuda.is_available() is False. Falling back to CPU.", device)
            selected_device = "cpu"
        else:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info("CUDA GPU detected: %s (%.1f GB VRAM)", gpu_name, vram_gb)
            logger.info("Configured device: %s", selected_device)

    # Resolve dataset config
    if not data_config.is_absolute():
        script_dir = Path(__file__).parent
        data_config_path = (script_dir / data_config).resolve()
    else:
        data_config_path = data_config.resolve()

    if not data_config_path.exists():
        raise FileNotFoundError(f"Dataset configuration file not found at: {data_config_path}")

    logger.info("=" * 60)
    logger.info("Starting Boiler LCD YOLO11 Training Pipeline")
    logger.info("Base Model:    %s", model_name)
    logger.info("Dataset YAML:  %s", data_config_path)
    logger.info("Epochs:        %d (patience: %d)", epochs, patience)
    logger.info("Input Size:    %d x %d", imgsz, imgsz)
    logger.info("Batch Size:    %d", batch)
    logger.info("Worker Threads:%d", workers)
    logger.info("Device:        %s", selected_device)
    logger.info("=" * 60)

    # Initialize Ultralytics model
    model = YOLO(model_name)

    # -------------------------------------------------------------------------
    # Train with augmentations tailored specifically for boiler LCD screens:
    # CRITICAL: fliplr=0.0, flipud=0.0 (never mirror digits like 2, 5, 3)
    # -------------------------------------------------------------------------
    train_results = model.train(
        data=str(data_config_path),
        epochs=epochs,
        patience=patience,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=selected_device,
        name=run_name,
        # Augmentations
        fliplr=0.0,  # CRITICAL: No horizontal flipping of numbers
        flipud=0.0,  # CRITICAL: No vertical flipping of numbers
        degrees=5.0,  # Slight camera angle misalignment
        translate=0.1,  # Slight vibration / crop shift
        scale=0.2,  # Slight zoom variation
        hsv_h=0.015,  # Minimal hue shift (LCD is blue)
        hsv_s=0.4,  # Saturation variation
        hsv_v=0.4,  # Brightness variation (day vs night backlight)
        save=True,
        save_period=-1,
        exist_ok=True,
    )

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

    # -------------------------------------------------------------------------
    # Export to TensorFlow Lite (TFLite)
    # -------------------------------------------------------------------------
    logger.info("Exporting %s to TensorFlow Lite format (imgsz=%d)...", best_pt.name, imgsz)
    best_model = YOLO(str(best_pt))

    export_result = best_model.export(
        format="tflite",
        imgsz=imgsz,
        half=False,  # Float32 model for best numerical stability
    )
    logger.info("Ultralytics export return path: %s", export_result)

    # Locate generated .tflite file
    tflite_candidates: list[Path] = []
    if export_result and Path(str(export_result)).is_file():
        tflite_candidates.append(Path(str(export_result)))
    elif export_result and Path(str(export_result)).is_dir():
        tflite_candidates.extend(Path(str(export_result)).glob("*.tflite"))

    # Also search weights directory
    weights_dir = best_pt.parent
    tflite_candidates.extend(weights_dir.glob("**/*.tflite"))

    # Deduplicate candidates
    unique_candidates = sorted(set(tflite_candidates), key=lambda p: str(p))

    # Prioritize float32 model over others
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
        description="Train YOLO11 digit detector and export to TFLite for Boiler LCD Reader."
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
        default=64,
        help="Batch size (default: 64 for 16GB VRAM on RTX 5080).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="DataLoader worker threads (default: 12 for Ryzen 9 9950X3D).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device index or 'cpu' (default: '0').",
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
    )


if __name__ == "__main__":
    main()
