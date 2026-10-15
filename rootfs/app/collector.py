"""Sample Collector for Boiler LCD Reader add-on.

Handles on-demand camera frame snapshots, ROI cropping, change detection
(anti-duplication), background sample dataset collection with rotation limits,
and ZIP export for model training.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from app.config_manager import AppSettings, ROIConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Constraints (from .antigravityrules)
# ---------------------------------------------------------------------------

MAX_SAMPLES_COUNT = 300
MAX_STORAGE_BYTES = 100 * 1024 * 1024  # 100 MB
JPEG_QUALITY = 95
THUMBNAIL_SIZE = (64, 32)  # (width, height) in pixels for diff comparison
DEFAULT_CHANGE_THRESHOLD = 2.5  # Mean absolute pixel difference threshold


# ---------------------------------------------------------------------------
# Standalone ROI Cropping Utility
# ---------------------------------------------------------------------------


def crop_roi(image_bytes: bytes, roi: ROIConfig) -> bytes:
    """Decode JPEG image, crop specified ROI bounding box, and re-encode to JPEG.

    Args:
        image_bytes: Raw JPEG image bytes.
        roi: ROIConfig with normalized coordinates [0.0 - 1.0].

    Returns:
        High-quality JPEG bytes of the cropped region.

    Raises:
        ValueError: If image decoding or ROI clamping fails.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode JPEG image bytes into image array")

    h_img, w_img = img.shape[:2]
    if h_img <= 0 or w_img <= 0:
        raise ValueError(f"Decoded image has invalid dimensions: {w_img}x{h_img}")

    x1 = int(round(roi.x * w_img))
    y1 = int(round(roi.y * h_img))
    x2 = int(round((roi.x + roi.w) * w_img))
    y2 = int(round((roi.y + roi.h) * h_img))

    # Safely clamp coordinates to image dimensions
    x1 = max(0, min(x1, w_img - 1))
    y1 = max(0, min(y1, h_img - 1))
    x2 = max(x1 + 1, min(x2, w_img))
    y2 = max(y1 + 1, min(y2, h_img))

    crop = img[y1:y2, x1:x2]
    success, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not success:
        raise ValueError("Failed to encode cropped image to JPEG")

    return encoded.tobytes()


def _make_thumbnail(crop_bgr: np.ndarray) -> np.ndarray:
    """Convert an image crop to grayscale and resize to standard thumbnail dimensions."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, THUMBNAIL_SIZE, interpolation=cv2.INTER_AREA)


def _calc_diff(thumb1: np.ndarray, thumb2: np.ndarray) -> float:
    """Compute the mean absolute difference between two grayscale thumbnails."""
    diff = cv2.absdiff(thumb1, thumb2)
    return float(np.mean(diff))


# ---------------------------------------------------------------------------
# SampleCollector Class
# ---------------------------------------------------------------------------


class SampleCollector:
    """Manages training samples storage, automated collection, and ZIP packaging."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        raw_dir = data_dir or os.getenv("DATA_DIR", "/data")
        self.data_dir = Path(raw_dir)
        self.samples_dir = self.data_dir / "samples"
        self._lock = asyncio.Lock()

        # In-memory reference thumbnails for change detection
        self._last_co_thumb: np.ndarray | None = None
        self._last_cwu_thumb: np.ndarray | None = None

        self.samples_dir.mkdir(parents=True, exist_ok=True)

    def get_stats(self) -> dict[str, Any]:
        """Return statistics about saved training samples and storage capacity."""
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        total_bytes = 0

        for entry in self.samples_dir.iterdir():
            if entry.is_file() and entry.suffix.lower() == ".jpg":
                count += 1
                with suppress(OSError):
                    total_bytes += entry.stat().st_size

        limit_reached = (count >= MAX_SAMPLES_COUNT) or (total_bytes >= MAX_STORAGE_BYTES)
        size_mb = round(total_bytes / (1024 * 1024), 2)
        max_storage_mb = round(MAX_STORAGE_BYTES / (1024 * 1024), 2)

        return {
            "count": count,
            "max_count": MAX_SAMPLES_COUNT,
            "size_bytes": total_bytes,
            "size_mb": size_mb,
            "max_storage_mb": max_storage_mb,
            "limit_reached": limit_reached,
        }

    async def process_and_save_background_sample(
        self,
        image_bytes: bytes,
        settings: AppSettings,
        change_threshold: float = DEFAULT_CHANGE_THRESHOLD,
    ) -> bool:
        """Process an image in the background loop and save crops if significant change occurred.

        Checks collector_enabled flag, storage quotas, and image differences before saving.

        Returns:
            True if sample was saved, False if skipped (due to no change, disabled, or limit).
        """
        if not settings.collector_enabled:
            return False

        stats = self.get_stats()
        if stats["limit_reached"]:
            logger.debug(
                "Collector limit reached (count=%d/%d, size=%s MB). Skipping background capture.",
                stats["count"],
                MAX_SAMPLES_COUNT,
                stats["size_mb"],
            )
            return False

        # Decode image to inspect ROIs
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Collector could not decode background snapshot")
            return False

        h_img, w_img = img.shape[:2]
        co_crop, co_thumb = self._extract_crop_and_thumb(img, settings.roi_co, w_img, h_img)
        cwu_crop, cwu_thumb = self._extract_crop_and_thumb(img, settings.roi_cwu, w_img, h_img)

        # Detect changes in monitored ROIs
        has_changed = False

        if co_thumb is not None:
            if self._last_co_thumb is None:
                has_changed = True
            else:
                diff_co = _calc_diff(self._last_co_thumb, co_thumb)
                if diff_co >= change_threshold:
                    has_changed = True

        if cwu_thumb is not None:
            if self._last_cwu_thumb is None:
                has_changed = True
            else:
                diff_cwu = _calc_diff(self._last_cwu_thumb, cwu_thumb)
                if diff_cwu >= change_threshold:
                    has_changed = True

        # If no ROIs configured at all, compare full downscaled image
        if co_thumb is None and cwu_thumb is None:
            full_thumb = _make_thumbnail(img)
            if self._last_co_thumb is None:
                has_changed = True
            else:
                diff_full = _calc_diff(self._last_co_thumb, full_thumb)
                if diff_full >= change_threshold:
                    has_changed = True
            if has_changed:
                self._last_co_thumb = full_thumb

        if not has_changed:
            return False

        # Save files
        async with self._lock:
            saved = await asyncio.to_thread(
                self._write_samples_sync,
                image_bytes=image_bytes,
                co_crop=co_crop,
                cwu_crop=cwu_crop,
            )

        if saved:
            if co_thumb is not None:
                self._last_co_thumb = co_thumb
            if cwu_thumb is not None:
                self._last_cwu_thumb = cwu_thumb
            logger.info(
                "Saved new training sample set to %s (quota: %d/%d)",
                self.samples_dir,
                stats["count"] + 1,
                MAX_SAMPLES_COUNT,
            )
            return True

        return False

    async def capture_now(self, image_bytes: bytes, settings: AppSettings) -> dict[str, Any]:
        """Force immediate capture and save of current frame and ROIs regardless of change diff.

        Returns:
            Updated dictionary of collector stats.

        Raises:
            RuntimeError: If storage quota has already been reached.
            ValueError: If image cannot be decoded.
        """
        stats = self.get_stats()
        if stats["limit_reached"]:
            raise RuntimeError(
                f"Cannot capture: maximum storage capacity reached "
                f"({stats['count']}/{MAX_SAMPLES_COUNT} files, {stats['size_mb']} MB)."
            )

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode JPEG image bytes into image array")

        h_img, w_img = img.shape[:2]
        co_crop, co_thumb = self._extract_crop_and_thumb(img, settings.roi_co, w_img, h_img)
        cwu_crop, cwu_thumb = self._extract_crop_and_thumb(img, settings.roi_cwu, w_img, h_img)

        async with self._lock:
            await asyncio.to_thread(
                self._write_samples_sync,
                image_bytes=image_bytes,
                co_crop=co_crop,
                cwu_crop=cwu_crop,
            )

        if co_thumb is not None:
            self._last_co_thumb = co_thumb
        if cwu_thumb is not None:
            self._last_cwu_thumb = cwu_thumb

        return self.get_stats()

    def _extract_crop_and_thumb(
        self,
        img: np.ndarray,
        roi: ROIConfig | None,
        w_img: int,
        h_img: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Extract crop and diff thumbnail for a given ROI if defined."""
        if roi is None:
            return None, None

        x1 = max(0, min(int(round(roi.x * w_img)), w_img - 1))
        y1 = max(0, min(int(round(roi.y * h_img)), h_img - 1))
        x2 = max(x1 + 1, min(int(round((roi.x + roi.w) * w_img)), w_img))
        y2 = max(y1 + 1, min(int(round((roi.y + roi.h) * h_img)), h_img))

        crop = img[y1:y2, x1:x2]
        thumb = _make_thumbnail(crop)
        return crop, thumb

    def _write_samples_sync(
        self,
        image_bytes: bytes,
        co_crop: np.ndarray | None,
        cwu_crop: np.ndarray | None,
    ) -> bool:
        """Synchronously write full snapshot and available ROI crops to disk."""
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        stats = self.get_stats()
        if stats["limit_reached"]:
            return False

        ts_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")

        # 1. Full frame
        full_path = self.samples_dir / f"sample_{ts_str}_full.jpg"
        full_path.write_bytes(image_bytes)

        # 2. CO Crop (if present)
        if co_crop is not None:
            co_path = self.samples_dir / f"sample_{ts_str}_roi_co.jpg"
            success, enc = cv2.imencode(".jpg", co_crop, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if success:
                co_path.write_bytes(enc.tobytes())

        # 3. CWU Crop (if present)
        if cwu_crop is not None:
            cwu_path = self.samples_dir / f"sample_{ts_str}_roi_cwu.jpg"
            success, enc = cv2.imencode(".jpg", cwu_crop, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if success:
                cwu_path.write_bytes(enc.tobytes())

        return True

    def export_samples_zip(self, settings: AppSettings | None = None) -> io.BytesIO:
        """Pack all collected samples and metadata.json into an in-memory ZIP archive."""
        import zipfile

        self.samples_dir.mkdir(parents=True, exist_ok=True)
        zip_buffer = io.BytesIO()

        jpg_files = sorted([f for f in self.samples_dir.iterdir() if f.is_file() and f.suffix.lower() == ".jpg"])

        metadata = {
            "exported_at": datetime.now(UTC).isoformat(),
            "total_files": len(jpg_files),
            "roi_co": settings.roi_co.model_dump() if settings and settings.roi_co else None,
            "roi_cwu": settings.roi_cwu.model_dump() if settings and settings.roi_cwu else None,
            "settings": settings.model_dump() if settings else None,
        }

        with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("metadata.json", json.dumps(metadata, indent=2))
            for file_path in jpg_files:
                zf.write(file_path, arcname=file_path.name)

        zip_buffer.seek(0)
        return zip_buffer

    def clear_samples(self) -> int:
        """Safely delete all sample files in /data/samples and reset in-memory comparison state."""
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        deleted = 0

        for entry in list(self.samples_dir.iterdir()):
            if entry.is_file():
                try:
                    entry.unlink()
                    deleted += 1
                except OSError as exc:
                    logger.warning("Could not delete file %s: %s", entry, exc)

        self._last_co_thumb = None
        self._last_cwu_thumb = None
        logger.info("Cleared %d sample files from %s", deleted, self.samples_dir)
        return deleted
