"""Dataset preparation and validation utility for Boiler LCD Reader.

Extracts boiler training samples from an exported ZIP archive or folder,
validates YOLO format annotations (10 classes: 0-9), and splits the dataset
into train and val sets (default: 85% train / 15% val) with a reproducible seed.
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("prepare_dataset")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NUM_CLASSES = 10


def validate_yolo_label_file(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Validate a YOLO format text annotation file.

    Format per line: <class_id> <cx> <cy> <w> <h>
    - class_id: Integer between 0 and 9 inclusive.
    - cx, cy, w, h: Normalized float coordinates in range [0.0, 1.0].

    Returns:
        List of parsed annotation tuples (class_id, cx, cy, w, h).

    Raises:
        ValueError: If any line contains invalid class ID, malformed coordinates, or wrong column count.
    """
    annotations = []
    lines = label_path.read_text(encoding="utf-8").splitlines()

    for line_idx, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) != 5:
            msg = f"{label_path.name}:{line_idx}: expected 5 tokens (<class> <cx> <cy> <w> <h>), got {len(parts)}"
            raise ValueError(f"{msg}: '{line}'")

        try:
            class_id = int(parts[0])
        except ValueError as exc:
            raise ValueError(f"{label_path.name}:{line_idx}: invalid class_id '{parts[0]}' (must be integer)") from exc

        if not (0 <= class_id < NUM_CLASSES):
            raise ValueError(
                f"{label_path.name}:{line_idx}: class_id {class_id} is out of range (must be 0 to {NUM_CLASSES - 1})"
            )

        try:
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError as exc:
            raise ValueError(f"{label_path.name}:{line_idx}: invalid coordinate values: '{line}'") from exc

        for name, val in [("cx", cx), ("cy", cy), ("w", w), ("h", h)]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{label_path.name}:{line_idx}: coordinate {name}={val} is outside [0.0, 1.0]")

        annotations.append((class_id, cx, cy, w, h))

    return annotations


def collect_samples(source_dir: Path) -> tuple[list[tuple[Path, Path]], list[Path]]:
    """Collect image and label file pairs from a source directory.

    Searches recursively for images. If a corresponding .txt file with the same
    stem exists in the same folder or in an adjacent 'labels' folder, it is paired.

    Returns:
        Tuple of (labeled_pairs, unlabeled_images).
    """
    labeled_pairs: list[tuple[Path, Path]] = []
    unlabeled_images: list[Path] = []

    image_files = sorted([p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])

    for img_path in image_files:
        # Check adjacent .txt file
        direct_label = img_path.with_suffix(".txt")
        # Check parallel labels/ directory (e.g. if images are in images/)
        alt_label = None
        if "images" in img_path.parts:
            parts = list(img_path.parts)
            idx = parts.index("images")
            parts[idx] = "labels"
            alt_label = Path(*parts).with_suffix(".txt")

        target_label = None
        if direct_label.is_file():
            target_label = direct_label
        elif alt_label is not None and alt_label.is_file():
            target_label = alt_label

        if target_label is not None:
            labeled_pairs.append((img_path, target_label))
        else:
            unlabeled_images.append(img_path)

    return labeled_pairs, unlabeled_images


def prepare_dataset(
    input_path: Path,
    output_dir: Path,
    train_ratio: float = 0.85,
    seed: int = 42,
) -> None:
    """Extract, validate, split, and organize dataset into YOLO structure."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    temp_extract_dir: Path | None = None
    source_dir = input_path

    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        logger.info("Extracting ZIP archive: %s", input_path)
        temp_extract_dir = Path(tempfile.mkdtemp(prefix="boiler_dataset_"))
        with zipfile.ZipFile(input_path, "r") as zip_ref:
            zip_ref.extractall(temp_extract_dir)
        source_dir = temp_extract_dir

    try:
        labeled_pairs, unlabeled_images = collect_samples(source_dir)
        logger.info(
            "Found %d labeled sample pair(s) and %d unlabeled image(s)",
            len(labeled_pairs),
            len(unlabeled_images),
        )

        if not labeled_pairs and not unlabeled_images:
            raise RuntimeError(f"No image files found in {input_path}")

        # Target directories
        images_train_dir = output_dir / "images" / "train"
        images_val_dir = output_dir / "images" / "val"
        labels_train_dir = output_dir / "labels" / "train"
        labels_val_dir = output_dir / "labels" / "val"
        unlabeled_dir = output_dir / "unlabeled_to_annotate"

        for d in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Validate all label files first
        class_counter: Counter[int] = Counter()
        validated_pairs: list[tuple[Path, Path]] = []

        for img_path, label_path in labeled_pairs:
            try:
                annotations = validate_yolo_label_file(label_path)
                for class_id, _, _, _, _ in annotations:
                    class_counter[class_id] += 1
                validated_pairs.append((img_path, label_path))
            except ValueError as exc:
                logger.warning("Skipping invalid annotation: %s", exc)

        if not validated_pairs:
            logger.warning(
                "No valid labeled pairs found! If you downloaded raw samples from the HA add-on, "
                "you must annotate them first (e.g. using Label Studio, CVAT, or MakeSense.ai) "
                "into YOLO format (.txt files with same filename)."
            )
            if unlabeled_images:
                unlabeled_dir.mkdir(parents=True, exist_ok=True)
                for img_path in unlabeled_images:
                    shutil.copy2(img_path, unlabeled_dir / img_path.name)
                logger.info(
                    "Copied %d raw images to %s ready for annotation.",
                    len(unlabeled_images),
                    unlabeled_dir,
                )
            return

        # Shuffle and split
        random.seed(seed)
        random.shuffle(validated_pairs)

        train_count = int(round(len(validated_pairs) * train_ratio))
        train_pairs = validated_pairs[:train_count]
        val_pairs = validated_pairs[train_count:]

        logger.info(
            "Splitting dataset: %d train (%.1f%%) / %d val (%.1f%%)",
            len(train_pairs),
            train_ratio * 100,
            len(val_pairs),
            (1 - train_ratio) * 100,
        )

        # Copy train pairs
        for img_path, label_path in train_pairs:
            dest_img = images_train_dir / img_path.name
            dest_lbl = labels_train_dir / label_path.name
            shutil.copy2(img_path, dest_img)
            shutil.copy2(label_path, dest_lbl)

        # Copy val pairs
        for img_path, label_path in val_pairs:
            dest_img = images_val_dir / img_path.name
            dest_lbl = labels_val_dir / label_path.name
            shutil.copy2(img_path, dest_img)
            shutil.copy2(label_path, dest_lbl)

        # Copy unlabeled images if any
        if unlabeled_images:
            unlabeled_dir.mkdir(parents=True, exist_ok=True)
            for img_path in unlabeled_images:
                shutil.copy2(img_path, unlabeled_dir / img_path.name)
            logger.info("Copied %d unlabeled image(s) to %s", len(unlabeled_images), unlabeled_dir)

        # Print summary
        logger.info("=" * 60)
        logger.info("Dataset Preparation Complete!")
        logger.info("Output directory: %s", output_dir.resolve())
        logger.info("  - Train images: %d in %s", len(train_pairs), images_train_dir)
        logger.info("  - Val images:   %d in %s", len(val_pairs), images_val_dir)
        logger.info("Digit class distribution across annotations:")
        for digit in range(NUM_CLASSES):
            count = class_counter.get(digit, 0)
            logger.info("  Class '%d': %d instances", digit, count)
        logger.info("=" * 60)

    finally:
        if temp_extract_dir and temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir, ignore_errors=True)


def main() -> None:
    """CLI entry point for prepare_dataset."""
    parser = argparse.ArgumentParser(description="Prepare and split boiler LCD digit samples for YOLO11 training.")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to exported ZIP archive (boiler_training_samples_*.zip) or directory with images and labels.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).parent / "datasets" / "boiler_digits",
        help="Target dataset directory (default: training/datasets/boiler_digits).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.85,
        help="Proportion of samples allocated to training set (default: 0.85).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic train/val splitting (default: 42).",
    )

    args = parser.parse_args()
    prepare_dataset(
        input_path=args.input,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
