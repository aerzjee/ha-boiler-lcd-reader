"""Configuration manager for Boiler LCD Reader add-on.

Handles loading, validating, and atomically persisting add-on configuration
to /data/settings.json using Pydantic v2.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class ROIConfig(BaseModel):
    """Region of Interest (ROI) bounding box in normalized coordinates [0.0, 1.0]."""

    x: float = Field(..., ge=0.0, le=1.0, description="Normalized X coordinate of top-left corner")
    y: float = Field(..., ge=0.0, le=1.0, description="Normalized Y coordinate of top-left corner")
    w: float = Field(..., gt=0.01, le=1.0, description="Normalized width of the bounding box")
    h: float = Field(..., gt=0.01, le=1.0, description="Normalized height of the bounding box")

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        """Ensure ROI does not exceed image dimensions."""
        if self.x + self.w > 1.0:
            raise ValueError(
                f"ROI horizontal boundary exceeded: x ({self.x:.4f}) + w ({self.w:.4f}) = {self.x + self.w:.4f} > 1.0"
            )
        if self.y + self.h > 1.0:
            raise ValueError(
                f"ROI vertical boundary exceeded: y ({self.y:.4f}) + h ({self.h:.4f}) = {self.y + self.h:.4f} > 1.0"
            )
        return self


class AppSettings(BaseModel):
    """Application settings persisted to /data/settings.json."""

    camera_entity: str | None = Field(
        default=None,
        description="Home Assistant camera entity ID (e.g. 'camera.boiler_display')",
    )
    scan_interval_seconds: int = Field(
        default=30,
        ge=5,
        le=3600,
        description="Interval in seconds between display OCR reads",
    )
    roi_co: ROIConfig | None = Field(
        default=None,
        description="ROI for Central Heating (CO) temperature display",
    )
    roi_cwu: ROIConfig | None = Field(
        default=None,
        description="ROI for Domestic Hot Water (CWU) temperature display",
    )
    min_temp: int = Field(
        default=5,
        ge=0,
        le=99,
        description="Minimum plausible temperature in °C (integer)",
    )
    max_temp: int = Field(
        default=95,
        ge=1,
        le=99,
        description="Maximum plausible temperature in °C (integer)",
    )
    max_delta: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Maximum temperature jump between readings before requiring 3-frame confirmation",
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.1,
        le=0.99,
        description="Minimum confidence score for digit detection",
    )
    collector_enabled: bool = Field(
        default=False,
        description="Enable saving inference input crops to /data/samples for training dataset",
    )
    sensor_co_entity: str = Field(
        default="sensor.boiler_co_temperature",
        pattern=r"^sensor\.[a-z0-9_]+$",
        description="Home Assistant sensor entity ID for CO temperature",
    )
    sensor_cwu_entity: str = Field(
        default="sensor.boiler_cwu_temperature",
        pattern=r"^sensor\.[a-z0-9_]+$",
        description="Home Assistant sensor entity ID for CWU temperature",
    )

    @field_validator("camera_entity")
    @classmethod
    def validate_camera_entity(cls, v: str | None) -> str | None:
        """Validate camera entity naming convention."""
        if v is not None:
            v = v.strip()
            if not v:
                return None
            if not v.startswith("camera."):
                raise ValueError(f"Invalid camera_entity '{v}'. Must start with 'camera.'")
        return v

    @model_validator(mode="after")
    def validate_temperature_range(self) -> Self:
        """Ensure max_temp is strictly greater than min_temp."""
        if self.max_temp <= self.min_temp:
            raise ValueError(f"max_temp ({self.max_temp}) must be strictly greater than min_temp ({self.min_temp})")
        return self


# ---------------------------------------------------------------------------
# Configuration Manager
# ---------------------------------------------------------------------------


class ConfigManager:
    """Manages application settings with atomic file writes and concurrency safety."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        raw_dir = data_dir or os.getenv("DATA_DIR", "/data")
        self.data_dir = Path(raw_dir)
        self.settings_path = self.data_dir / "settings.json"
        self._lock = asyncio.Lock()
        self._settings: AppSettings | None = None

    def load_settings(self) -> AppSettings:
        """Synchronously load settings from disk or initialize defaults if missing/corrupt."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        if not self.settings_path.exists():
            logger.warning(
                "Settings file not found at %s. Initializing with default configuration.",
                self.settings_path,
            )
            defaults = AppSettings()
            self._write_file_sync(defaults)
            self._settings = defaults
            return defaults

        try:
            content = self.settings_path.read_text(encoding="utf-8")
            self._settings = AppSettings.model_validate_json(content)
            logger.info("Successfully loaded settings from %s", self.settings_path)
            return self._settings
        except Exception as exc:
            corrupt_backup = self.settings_path.with_suffix(".json.corrupt")
            logger.error(
                "Failed to parse settings file at %s: %s. Backing up to %s and restoring defaults.",
                self.settings_path,
                exc,
                corrupt_backup,
            )
            with contextlib.suppress(OSError):
                os.replace(self.settings_path, corrupt_backup)

            defaults = AppSettings()
            self._write_file_sync(defaults)
            self._settings = defaults
            return defaults

    def get_settings(self) -> AppSettings:
        """Return the current cached settings, loading them if not yet initialized."""
        if self._settings is None:
            return self.load_settings()
        return self._settings

    async def save_settings(self, new_settings: AppSettings) -> AppSettings:
        """Asynchronously save settings to disk using atomic replacement."""
        async with self._lock:
            await asyncio.to_thread(self._write_file_sync, new_settings)
            self._settings = new_settings
            logger.info("Successfully updated and persisted settings to %s", self.settings_path)
            return self._settings

    def save_settings_sync(self, new_settings: AppSettings) -> AppSettings:
        """Synchronously save settings to disk using atomic replacement."""
        self._write_file_sync(new_settings)
        self._settings = new_settings
        logger.info("Successfully updated and persisted settings to %s (sync)", self.settings_path)
        return self._settings

    def _write_file_sync(self, settings: AppSettings) -> None:
        """Atomic write to disk: write to temp file, flush, fsync, and atomic replace."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = self.settings_path.with_suffix(".json.tmp")

        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                f.write(settings.model_dump_json(indent=2))
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_file, self.settings_path)
        except Exception:
            if tmp_file.exists():
                with contextlib.suppress(OSError):
                    tmp_file.unlink()
            raise
