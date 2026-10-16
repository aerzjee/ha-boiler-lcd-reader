"""Dataset preparation, trimming, balancing, and validation utility for Boiler LCD Reader.

Extracts boiler training samples from an exported ZIP archive or folder,
validates YOLO format annotations (10 classes: 0-9), provides Rarity-First Greedy Selection
dataset trimming (--trim-per-class, --trim-existing), class distribution analysis with rarity
detection, and splits/appends the dataset into train and val sets (85% train / 15% val).
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

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


def resolve_unique_dest_pair(
    dest_img_dir: Path,
    dest_lbl_dir: Path,
    img_name: str,
    lbl_name: str,
) -> tuple[Path, Path]:
    """Resolve destination paths for an image and label pair avoiding filename collisions.

    If target files already exist in destination directories, appends an incremental
    suffix to both filenames so that the image and its corresponding label maintain
    an identical stem required by YOLO dataset loaders.
    """
    dest_img = dest_img_dir / img_name
    dest_lbl = dest_lbl_dir / lbl_name
    if not dest_img.exists() and not dest_lbl.exists():
        return dest_img, dest_lbl

    stem = Path(img_name).stem
    img_ext = Path(img_name).suffix
    lbl_ext = Path(lbl_name).suffix

    counter = 1
    while True:
        candidate_img = dest_img_dir / f"{stem}_append{counter}{img_ext}"
        candidate_lbl = dest_lbl_dir / f"{stem}_append{counter}{lbl_ext}"
        if not candidate_img.exists() and not candidate_lbl.exists():
            return candidate_img, candidate_lbl
        counter += 1


def count_class_distribution(labels_dir: Path) -> dict[int, int]:
    """Analyze and display the class distribution (0-9) across YOLO label files.

    Searches recursively for all .txt label files in labels_dir (e.g. train and val),
    counts occurrences of each digit class (0 through 9), and prints a formatted
    table with an ASCII progress bar and status indicator for rare classes (< 5%).

    Args:
        labels_dir: Path to directory containing YOLO .txt label files.

    Returns:
        Dictionary mapping class_id (0-9) to count of detected annotations.
    """
    class_counts: dict[int, int] = {i: 0 for i in range(NUM_CLASSES)}
    label_files = [p for p in labels_dir.rglob("*.txt") if p.is_file()] if labels_dir.exists() else []

    if not label_files:
        logger.warning("No label files (.txt) found in: %s", labels_dir)
        return class_counts

    for lbl_path in label_files:
        try:
            lines = lbl_path.read_text(encoding="utf-8").splitlines()
        except Exception as exc:
            logger.warning("Could not read %s: %s", lbl_path, exc)
            continue

        for line in lines:
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue
            parts = line_str.split()
            if not parts:
                continue
            try:
                cid = int(parts[0])
                if 0 <= cid < NUM_CLASSES:
                    class_counts[cid] += 1
            except ValueError:
                continue

    total_instances = sum(class_counts.values())

    table_lines: list[str] = [
        "",
        "=" * 82,
        "ROZKŁAD LICZEBNOŚCI KLAS CYFR (0-9) W ZBIORZE DANYCH",
        "=" * 82,
        f"{'Cyfra':^7}|{'Liczba':^10}|{'Udział':^9}|{'Wykres słupkowy ASCII':^24}|{'Status':^28}",
        f"{'-' * 7}+{'-' * 10}+{'-' * 9}+{'-' * 24}+{'-' * 28}",
    ]

    rare_classes: list[int] = []
    missing_classes: list[int] = []

    for digit in range(NUM_CLASSES):
        count = class_counts[digit]
        pct = (count / total_instances * 100.0) if total_instances > 0 else 0.0

        bar_len = 20
        filled = int(round((pct / 100.0) * bar_len)) if total_instances > 0 else 0
        filled = min(bar_len, max(0, filled))
        bar = f"[{'█' * filled}{'░' * (bar_len - filled)}]"

        if count == 0:
            status = "❌ BRAK PRÓBEK (0%)"
            missing_classes.append(digit)
        elif pct < 5.0:
            status = "⚠️  RZADKA KLASA (< 5%)"
            rare_classes.append(digit)
        else:
            status = "✓ OK"

        table_lines.append(f"{digit:^7}|{count:^10}|{f'{pct:.1f}%':^9}| {bar} | {status:<26}")

    table_lines.extend(
        [
            "-" * 82,
            f"Łączna liczba adnotacji: {total_instances} (w {len(label_files)} plikach etykiet)",
            f"Katalog etykiet:         {labels_dir.resolve()}",
        ]
    )

    if missing_classes:
        table_lines.append(f"❌ Brakujące cyfry w zbiorze:       {', '.join(str(c) for c in missing_classes)}")
    if rare_classes:
        table_lines.append(f"⚠️  Rzadkie cyfry w zbiorze (< 5%):  {', '.join(str(c) for c in rare_classes)}")

    if missing_classes or rare_classes:
        table_lines.extend(
            [
                "",
                "💡 WSKAZÓWKA (Fine-Tuning):",
                "   Przed dotrenowaniem modelu zaleca się zebrać i zaetykietować więcej próbek",
                "   z rzadkimi cyframi, a następnie dodać je do zbioru za pomocą:",
                "   python prepare_dataset.py --input <nowa_paczka> --append",
            ]
        )
    else:
        table_lines.append("✓ Zbiór danych wykazuje zrównoważony rozkład wszystkich klas cyfr (0-9).")

    table_lines.append("=" * 82)
    logger.info("%s", "\n".join(table_lines))
    return class_counts


@dataclass
class LabeledSample:
    """Represents a paired image and YOLO annotation file with parsed class IDs."""

    img_path: Path
    label_path: Path
    classes: list[int]


def smart_trim_dataset(
    labeled_pairs: list[tuple[Path, Path]],
    target_per_class: int,
    seed: int = 42,
) -> tuple[list[tuple[Path, Path]], dict[str, Any]]:
    """Smartly select and balance samples using Rarity-First Greedy Selection.

    In boiler LCD digits recognition, a single crop image typically contains 2 digits.
    Random deletion would inadvertently eliminate rare digits that happen to co-occur
    with common digits. This algorithm sorts classes from rarest to most frequent,
    prioritizing images containing rare digits while penalizing excessive co-occurrences
    of already saturated classes.

    Args:
        labeled_pairs: List of (image_path, label_path) tuples.
        target_per_class: Target maximum number of instances for each digit class (0-9).
        seed: Random seed for deterministic shuffling.

    Returns:
        Tuple of:
          - selected_pairs: Trimmed list of (image_path, label_path) tuples.
          - stats: Dictionary containing before/after counts and reduction percentages.
    """
    samples: list[LabeledSample] = []
    initial_class_counts: Counter[int] = Counter()

    for img_path, lbl_path in labeled_pairs:
        try:
            annotations = validate_yolo_label_file(lbl_path)
            classes = [cid for cid, _, _, _, _ in annotations]
            if classes:
                for cid in classes:
                    initial_class_counts[cid] += 1
                samples.append(LabeledSample(img_path=img_path, label_path=lbl_path, classes=classes))
        except Exception as exc:
            logger.warning("Skipping invalid sample during trim parsing: %s (%s)", lbl_path, exc)
            continue

    if not samples:
        logger.warning("No valid samples available for smart trimming.")
        return [], {}

    rng = random.Random(seed)
    shuffled_samples = list(samples)
    rng.shuffle(shuffled_samples)

    # Sort classes 0-9 from rarest to most frequent
    sorted_classes = sorted(range(NUM_CLASSES), key=lambda c: (initial_class_counts[c], c))

    selected_indices: set[int] = set()
    selected_samples: list[LabeledSample] = []
    selected_counts: Counter[int] = Counter()
    soft_limit = max(target_per_class + 1, int(target_per_class * 1.3))

    for cls in sorted_classes:
        target_for_cls = min(target_per_class, initial_class_counts[cls])
        if target_for_cls <= 0:
            continue

        while selected_counts[cls] < target_for_cls:
            cand_indices = [i for i, s in enumerate(shuffled_samples) if i not in selected_indices and cls in s.classes]
            if not cand_indices:
                break

            def candidate_key(idx: int, target_cls: int = cls) -> tuple[int, int]:
                s = shuffled_samples[idx]
                companions = [c for c in s.classes if c != target_cls]
                has_over_soft = any(selected_counts[c] >= soft_limit for c in companions)
                has_over_hard = any(selected_counts[c] >= target_per_class for c in companions)
                tier = 2 if has_over_soft else (1 if has_over_hard else 0)
                penalty = sum(max(0, selected_counts[c] + 1 - target_per_class) for c in s.classes)
                return (tier, penalty)

            cand_indices.sort(key=candidate_key)
            best_idx = cand_indices[0]

            selected_indices.add(best_idx)
            best_sample = shuffled_samples[best_idx]
            selected_samples.append(best_sample)
            for c in best_sample.classes:
                selected_counts[c] += 1

    total_before = sum(initial_class_counts.values())
    total_after = sum(selected_counts.values())
    total_reduction_pct = ((total_before - total_after) / total_before * 100.0) if total_before > 0 else 0.0
    img_reduction_pct = ((len(samples) - len(selected_samples)) / len(samples) * 100.0) if samples else 0.0

    table_lines: list[str] = [
        "",
        "=" * 82,
        f"PORÓWNANIE LICZEBNOŚCI KLAS: PRZED vs PO PRZYCIĘCIU (TRIM: max {target_per_class} / klasę)",
        "=" * 82,
        f"{'Cyfra':^7}|{'Przed':^8}|{'Przed %':^9}|{'Po':^7}|{'Po %':^8}|{'Wykres słupkowy (Po)':^24}|{'Zmiana':^9}",
        f"{'-' * 7}+{'-' * 8}+{'-' * 9}+{'-' * 7}+{'-' * 8}+{'-' * 24}+{'-' * 9}",
    ]

    for digit in range(NUM_CLASSES):
        b_cnt = initial_class_counts[digit]
        b_pct = (b_cnt / total_before * 100.0) if total_before > 0 else 0.0
        a_cnt = selected_counts[digit]
        a_pct = (a_cnt / total_after * 100.0) if total_after > 0 else 0.0

        bar_len = 20
        filled = int(round((a_pct / 100.0) * bar_len)) if total_after > 0 else 0
        filled = min(bar_len, max(0, filled))
        bar = f"[{'█' * filled}{'░' * (bar_len - filled)}]"

        if b_cnt > 0:
            diff_pct = (a_cnt - b_cnt) / b_cnt * 100.0
            change_str = f"{diff_pct:+.1f}%" if diff_pct != 0 else "0.0%"
        else:
            change_str = "0.0%"

        table_lines.append(
            f"{digit:^7}|{b_cnt:^8}|{f'{b_pct:.1f}%':^9}|{a_cnt:^7}|{f'{a_pct:.1f}%':^8}| {bar} |{change_str:^9}"
        )

    table_lines.extend(
        [
            f"{'-' * 7}+{'-' * 8}+{'-' * 9}+{'-' * 7}+{'-' * 8}+{'-' * 24}+{'-' * 9}",
            (
                f"{'RAZEM':^7}|{total_before:^8}|{'100.0%':^9}|{total_after:^7}|"
                f"{'100.0%':^8}|{'':^24}|{f'-{total_reduction_pct:.1f}%':^9}"
            ),
            (
                f"Zdjęcia: {len(samples)} klatek przed -> {len(selected_samples)} klatek po "
                f"(redukcja: -{img_reduction_pct:.1f}%)"
            ),
            "=" * 82,
        ]
    )
    logger.info("%s", "\n".join(table_lines))

    selected_pairs = [(s.img_path, s.label_path) for s in selected_samples]
    stats = {
        "before_counts": dict(initial_class_counts),
        "after_counts": dict(selected_counts),
        "total_images_before": len(samples),
        "total_images_after": len(selected_samples),
    }
    return selected_pairs, stats


def stratified_split_samples(
    labeled_samples: list[LabeledSample],
    train_ratio: float = 0.85,
    seed: int = 42,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """Split labeled samples into train and val ensuring representation for all classes.

    Guarantees that every class present with >= 2 instances receives representation
    in both train and val subsets (at least 1-2 instances in val, and the rest in train),
    while adhering closely to train_ratio (default: 85% train / 15% val).
    """
    if not labeled_samples:
        return [], []

    rng = random.Random(seed)
    shuffled = list(labeled_samples)
    rng.shuffle(shuffled)

    total_samples = len(shuffled)
    target_val_size = max(1, int(round(total_samples * (1.0 - train_ratio))))

    class_pool_counts: Counter[int] = Counter()
    for s in shuffled:
        for c in s.classes:
            class_pool_counts[c] += 1

    train_set: list[LabeledSample] = []
    val_set: list[LabeledSample] = []
    assigned_indices: set[int] = set()

    train_counts: Counter[int] = Counter()
    val_counts: Counter[int] = Counter()

    sorted_classes = sorted(range(NUM_CLASSES), key=lambda c: (class_pool_counts[c], c))

    # Phase 1: Allocate required val quotas for each class
    for cls in sorted_classes:
        total_for_cls = class_pool_counts[cls]
        if total_for_cls <= 0:
            continue
        val_quota = 2 if total_for_cls >= 5 else (1 if total_for_cls >= 2 else 0)

        while val_counts[cls] < val_quota and len(val_set) < target_val_size:
            candidates = [i for i, s in enumerate(shuffled) if i not in assigned_indices and cls in s.classes]
            if not candidates:
                break
            chosen_idx = candidates[0]
            assigned_indices.add(chosen_idx)
            chosen_sample = shuffled[chosen_idx]
            val_set.append(chosen_sample)
            for c in chosen_sample.classes:
                val_counts[c] += 1

    # Phase 2: Ensure every class with available samples has at least 1 instance in train_set
    for cls in sorted_classes:
        if class_pool_counts[cls] > 0 and train_counts[cls] == 0:
            candidates = [i for i, s in enumerate(shuffled) if i not in assigned_indices and cls in s.classes]
            if candidates:
                chosen_idx = candidates[0]
                assigned_indices.add(chosen_idx)
                chosen_sample = shuffled[chosen_idx]
                train_set.append(chosen_sample)
                for c in chosen_sample.classes:
                    train_counts[c] += 1

    # Phase 3: Fill remaining samples into train and val up to target_val_size
    for i, s in enumerate(shuffled):
        if i in assigned_indices:
            continue
        assigned_indices.add(i)
        if len(val_set) < target_val_size:
            val_set.append(s)
            for c in s.classes:
                val_counts[c] += 1
        else:
            train_set.append(s)
            for c in s.classes:
                train_counts[c] += 1

    train_pairs = [(s.img_path, s.label_path) for s in train_set]
    val_pairs = [(s.img_path, s.label_path) for s in val_set]
    return train_pairs, val_pairs


def trim_existing_dataset(
    output_dir: Path,
    target_per_class: int,
    backup_before_trim: bool = True,
    train_ratio: float = 0.85,
    seed: int = 42,
) -> None:
    """Trim an existing dataset in-place using Rarity-First Greedy Selection.

    Creates an optional timestamped backup, safely collects all samples from
    output_dir/images and output_dir/labels, applies smart class-balanced trimming,
    and reorganizes the trimmed data into fresh train and val sets.

    Args:
        output_dir: Path to dataset directory (containing images/ and labels/).
        target_per_class: Target maximum number of instances for each digit class (0-9).
        backup_before_trim: If True, creates a copy in output_dir_backup_<timestamp>.
        train_ratio: Ratio of samples allocated to training set (default: 0.85).
        seed: Random seed for deterministic trimming and splitting.
    """
    if not output_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {output_dir}")

    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Missing images/ or labels/ directory in: {output_dir}")

    labeled_pairs, _ = collect_samples(output_dir)
    if not labeled_pairs:
        raise RuntimeError(f"No labeled sample pairs found in: {output_dir}")

    logger.info("Found %d existing labeled sample(s) across %s", len(labeled_pairs), output_dir)

    # 1. Timestamped backup
    if backup_before_trim:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = output_dir.parent / f"{output_dir.name}_backup_{timestamp}"
        logger.info("Creating backup of existing dataset at: %s", backup_dir)
        shutil.copytree(output_dir, backup_dir)
        logger.info("✓ Backup created successfully: %s", backup_dir)

    # 2. Stage files into a temporary directory to safely read before overwriting
    temp_stage_dir = Path(tempfile.mkdtemp(prefix="boiler_trim_stage_"))
    try:
        staged_pairs: list[tuple[Path, Path]] = []
        for img_p, lbl_p in labeled_pairs:
            stage_img = temp_stage_dir / img_p.name
            stage_lbl = temp_stage_dir / lbl_p.name
            shutil.copy2(img_p, stage_img)
            shutil.copy2(lbl_p, stage_lbl)
            staged_pairs.append((stage_img, stage_lbl))

        # 3. Apply Rarity-First Greedy Selection
        trimmed_pairs, _ = smart_trim_dataset(
            staged_pairs,
            target_per_class=target_per_class,
            seed=seed,
        )

        # 4. Stratified train/val split
        trimmed_samples = []
        for img_p, lbl_p in trimmed_pairs:
            annos = validate_yolo_label_file(lbl_p)
            classes = [c for c, _, _, _, _ in annos]
            trimmed_samples.append(LabeledSample(img_path=img_p, label_path=lbl_p, classes=classes))

        train_pairs, val_pairs = stratified_split_samples(
            trimmed_samples,
            train_ratio=train_ratio,
            seed=seed,
        )

        logger.info(
            "Splitting trimmed dataset: %d train (%.1f%%) / %d val (%.1f%%)",
            len(train_pairs),
            train_ratio * 100,
            len(val_pairs),
            (1 - train_ratio) * 100,
        )

        # 5. Clear target directories
        images_train_dir = output_dir / "images" / "train"
        images_val_dir = output_dir / "images" / "val"
        labels_train_dir = output_dir / "labels" / "train"
        labels_val_dir = output_dir / "labels" / "val"

        for d in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
            if d.exists():
                for f in d.iterdir():
                    if f.is_file():
                        f.unlink()
            d.mkdir(parents=True, exist_ok=True)

        # 6. Copy trimmed train and val pairs
        for img_p, lbl_p in train_pairs:
            shutil.copy2(img_p, images_train_dir / img_p.name)
            shutil.copy2(lbl_p, labels_train_dir / lbl_p.name)

        for img_p, lbl_p in val_pairs:
            shutil.copy2(img_p, images_val_dir / img_p.name)
            shutil.copy2(lbl_p, labels_val_dir / lbl_p.name)

        logger.info("=" * 60)
        logger.info("Dataset Trimming & Balancing Complete!")
        logger.info("Output directory: %s", output_dir.resolve())
        logger.info("  - Trimmed Train images: %d in %s", len(train_pairs), images_train_dir)
        logger.info("  - Trimmed Val images:   %d in %s", len(val_pairs), images_val_dir)
        logger.info("=" * 60)

        # 7. Print final class distribution
        count_class_distribution(output_dir / "labels")

    finally:
        if temp_stage_dir and temp_stage_dir.exists():
            shutil.rmtree(temp_stage_dir, ignore_errors=True)


def prepare_dataset(
    input_path: Path,
    output_dir: Path,
    train_ratio: float = 0.85,
    seed: int = 42,
    append: bool = False,
    trim_per_class: int | None = None,
) -> None:
    """Extract, validate, split, and organize dataset into YOLO structure.

    Args:
        input_path: Source ZIP archive or directory containing images and annotations.
        output_dir: Destination root directory for YOLO dataset (e.g. training/datasets/boiler_digits).
        train_ratio: Proportion of samples allocated to training set (default: 0.85).
        seed: Random seed for deterministic train/val splitting.
        append: If True, appends new samples to existing dataset without deleting existing files.
        trim_per_class: Optional target maximum instances per class (0-9) via Rarity-First Greedy Selection.
    """
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
            if not append and d.exists():
                for existing_file in d.iterdir():
                    if existing_file.is_file():
                        existing_file.unlink()
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

        # Smart trimming if requested
        if trim_per_class is not None and trim_per_class > 0:
            logger.info("Applying smart dataset trimming (target_per_class=%d)...", trim_per_class)
            trimmed_pairs, _ = smart_trim_dataset(
                validated_pairs,
                target_per_class=trim_per_class,
                seed=seed,
            )
            trimmed_samples = []
            for img_p, lbl_p in trimmed_pairs:
                annos = validate_yolo_label_file(lbl_p)
                classes = [c for c, _, _, _, _ in annos]
                trimmed_samples.append(LabeledSample(img_path=img_p, label_path=lbl_p, classes=classes))

            train_pairs, val_pairs = stratified_split_samples(
                trimmed_samples,
                train_ratio=train_ratio,
                seed=seed,
            )
        else:
            # Shuffle and split
            random.seed(seed)
            random.shuffle(validated_pairs)

            train_count = int(round(len(validated_pairs) * train_ratio))
            train_pairs = validated_pairs[:train_count]
            val_pairs = validated_pairs[train_count:]

        logger.info(
            "Splitting incoming dataset: %d train (%.1f%%) / %d val (%.1f%%) [append=%s]",
            len(train_pairs),
            train_ratio * 100,
            len(val_pairs),
            (1 - train_ratio) * 100,
            append,
        )

        # Copy train pairs
        copied_train = 0
        for img_path, label_path in train_pairs:
            if append:
                dest_img, dest_lbl = resolve_unique_dest_pair(
                    images_train_dir, labels_train_dir, img_path.name, label_path.name
                )
            else:
                dest_img = images_train_dir / img_path.name
                dest_lbl = labels_train_dir / label_path.name
            shutil.copy2(img_path, dest_img)
            shutil.copy2(label_path, dest_lbl)
            copied_train += 1

        # Copy val pairs
        copied_val = 0
        for img_path, label_path in val_pairs:
            if append:
                dest_img, dest_lbl = resolve_unique_dest_pair(
                    images_val_dir, labels_val_dir, img_path.name, label_path.name
                )
            else:
                dest_img = images_val_dir / img_path.name
                dest_lbl = labels_val_dir / label_path.name
            shutil.copy2(img_path, dest_img)
            shutil.copy2(label_path, dest_lbl)
            copied_val += 1

        # Copy unlabeled images if any
        if unlabeled_images:
            unlabeled_dir.mkdir(parents=True, exist_ok=True)
            for img_path in unlabeled_images:
                dest_unlabeled = unlabeled_dir / img_path.name
                if append and dest_unlabeled.exists():
                    stem = img_path.stem
                    ext = img_path.suffix
                    c = 1
                    while (unlabeled_dir / f"{stem}_append{c}{ext}").exists():
                        c += 1
                    dest_unlabeled = unlabeled_dir / f"{stem}_append{c}{ext}"
                shutil.copy2(img_path, dest_unlabeled)
            logger.info("Copied %d unlabeled image(s) to %s", len(unlabeled_images), unlabeled_dir)

        # Print summary
        logger.info("=" * 60)
        logger.info("Dataset Preparation Complete! (Mode: %s)", "APPEND" if append else "FRESH")
        logger.info("Output directory: %s", output_dir.resolve())
        logger.info("  - Newly added to Train: %d", copied_train)
        logger.info("  - Newly added to Val:   %d", copied_val)
        logger.info("=" * 60)

        # Analyze total class distribution in target dataset
        count_class_distribution(output_dir / "labels")

    finally:
        if temp_extract_dir and temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir, ignore_errors=True)


def main() -> None:
    """CLI entry point for prepare_dataset."""
    parser = argparse.ArgumentParser(
        description="Prepare, append, trim, and analyze boiler LCD digit samples for YOLO11 training."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
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
    parser.add_argument(
        "--append",
        action="store_true",
        default=False,
        help="Append new samples to existing dataset without wiping previous data (avoids collisions).",
    )
    parser.add_argument(
        "--trim-per-class",
        type=int,
        default=None,
        help="Target maximum instances for each digit class (0-9) via Rarity-First Greedy Selection.",
    )
    parser.add_argument(
        "--trim-existing",
        action="store_true",
        default=False,
        help="Trim existing dataset in --output in-place without needing --input.",
    )
    parser.add_argument(
        "--backup-before-trim",
        action="store_true",
        default=True,
        help="Create timestamped backup directory before trimming existing dataset (default: True).",
    )
    parser.add_argument(
        "--no-backup",
        action="store_false",
        dest="backup_before_trim",
        help="Disable creating a backup directory before trimming existing dataset.",
    )
    parser.add_argument(
        "--check-dist",
        action="store_true",
        default=False,
        help="Analyze and display class distribution of the dataset in --output without importing new samples.",
    )

    args = parser.parse_args()

    if args.trim_existing:
        target_limit = args.trim_per_class if args.trim_per_class is not None else 30
        logger.info(
            "Trimming existing dataset in %s (target_per_class=%d, backup=%s)...",
            args.output,
            target_limit,
            args.backup_before_trim,
        )
        trim_existing_dataset(
            output_dir=args.output,
            target_per_class=target_limit,
            backup_before_trim=args.backup_before_trim,
            train_ratio=args.train_ratio,
            seed=args.seed,
        )
        return

    if args.check_dist:
        labels_dir = args.output / "labels"
        logger.info("Analyzing class distribution in: %s", labels_dir)
        count_class_distribution(labels_dir)
        return

    if args.input is None:
        parser.error("The --input argument (-i) is required unless --trim-existing or --check-dist is specified.")

    prepare_dataset(
        input_path=args.input,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        seed=args.seed,
        append=args.append,
        trim_per_class=args.trim_per_class,
    )


if __name__ == "__main__":
    main()
