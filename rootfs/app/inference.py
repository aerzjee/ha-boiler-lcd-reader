"""TFLite inference engine for YOLO digit detection on boiler LCD displays.

Implements model loading (Singleton with mtime-based reload and thread-safe execution),
letterbox pre-processing, YOLO tensor post-processing with NMS, left-to-right digit
assembly into integer temperatures, annotated diagnostic images, and physical sanity-check validation.

Model Classes: Strictly 10 classes (0–9). No decimal points, no minus signs.
Output: Integer temperatures only.
"""

from __future__ import annotations

import base64
import logging
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import cv2
import numpy as np
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from app.config_manager import AppSettings, ROIConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TFLite Interpreter Import (Safe Fallback Pipeline)
# ---------------------------------------------------------------------------

Interpreter: Any | None = None

try:
    from ai_edge_litert.interpreter import Interpreter

    logger.info("Using ai-edge-litert TFLite interpreter")
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter

        logger.info("Using tflite-runtime interpreter")
    except ImportError:
        try:
            import tensorflow as tf

            Interpreter = tf.lite.Interpreter
            logger.info("Using tensorflow.lite Interpreter")
        except ImportError:
            Interpreter = None
            logger.warning(
                "No TFLite interpreter available (ai-edge-litert, tflite-runtime, and tensorflow are missing). "
                "Inference will be disabled until an interpreter package is installed."
            )

# ---------------------------------------------------------------------------
# Output Data Models
# ---------------------------------------------------------------------------


class DetectedDigit(BaseModel):
    """Single detected digit with bounding box and confidence score."""

    digit: int = Field(..., ge=0, le=9, description="Recognized digit class (0-9)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    bbox: tuple[int, int, int, int] = Field(
        ...,
        description="(x, y, w, h) bounding box in pixels relative to the ROI crop",
    )
    x_center: float = Field(
        ...,
        description="Horizontal center coordinate (x + w/2) for left-to-right sorting",
    )


class ChannelInferenceResult(BaseModel):
    """Inference result for a single boiler temperature channel (CO or CWU)."""

    channel: str = Field(..., description="Temperature channel ('co' or 'cwu')")
    raw_value: int | None = Field(
        default=None,
        description="Assembled integer temperature before sanity validation",
    )
    validated_value: int | None = Field(
        default=None,
        description="Final integer temperature after sanity validation, ready for HA",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average confidence score across detected digits",
    )
    status: str = Field(
        ...,
        description="Result status code (e.g. 'ok', 'no_model', 'no_roi', 'no_digits_detected', etc.)",
    )
    digits: list[DetectedDigit] = Field(
        default_factory=list,
        description="List of detected digits ordered left-to-right",
    )
    annotated_crop_base64: str | None = Field(
        default=None,
        description="Base64-encoded JPEG image of the ROI with detection annotations",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO-8601 UTC timestamp of inference execution",
    )


# ---------------------------------------------------------------------------
# Helper: Base64 JPEG Encoding
# ---------------------------------------------------------------------------


def _encode_bgr_to_base64_jpeg(image: np.ndarray, quality: int = 90) -> str:
    """Encode an OpenCV BGR image matrix into a Base64 JPEG string."""
    success, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return ""
    return base64.b64encode(buffer.tobytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Reading Validator (Sanity Check for Integer Temperatures)
# ---------------------------------------------------------------------------


class ReadingValidator:
    """Physical plausibility sanity checker for integer boiler temperatures.

    Maintains separate validation state for 'co' and 'cwu' channels.
    Rejects readings outside [min_temp, max_temp] and sudden jumps exceeding
    max_delta unless confirmed by 3 consecutive readings.
    """

    def __init__(self) -> None:
        self.last_valid_temp: dict[str, int | None] = {"co": None, "cwu": None}
        self.pending_jump_count: dict[str, int] = {"co": 0, "cwu": 0}

    def validate(
        self,
        channel: str,
        raw_value: int | None,
        min_temp: int,
        max_temp: int,
        max_delta: int,
    ) -> tuple[int | None, str]:
        """Validate raw integer reading against physical bounds and jump delta rules.

        Returns:
            Tuple of (validated_value, status_reason).
        """
        if channel not in self.last_valid_temp:
            self.last_valid_temp[channel] = None
            self.pending_jump_count[channel] = 0

        # 1. No digits detected
        if raw_value is None:
            return None, "no_digits_detected"

        # 2. Out of bounds range check
        if raw_value < min_temp or raw_value > max_temp:
            return None, f"rejected_out_of_range ({raw_value}°C not in {min_temp}..{max_temp}°C)"

        # 3. First valid reading after startup
        last_temp = self.last_valid_temp[channel]
        if last_temp is None:
            self.last_valid_temp[channel] = raw_value
            self.pending_jump_count[channel] = 0
            return raw_value, "ok"

        # 4. Delta calculation
        delta = abs(raw_value - last_temp)

        # 5. Delta exceeded -> require 3 consecutive confirmations
        if delta > max_delta:
            self.pending_jump_count[channel] += 1
            if self.pending_jump_count[channel] < 3:
                return None, f"rejected_delta_pending ({self.pending_jump_count[channel]}/3)"

            # 3 consecutive confirmations reached: accept new temperature level
            self.last_valid_temp[channel] = raw_value
            self.pending_jump_count[channel] = 0
            return raw_value, "ok"

        # 6. Normal reading within acceptable delta threshold
        self.last_valid_temp[channel] = raw_value
        self.pending_jump_count[channel] = 0
        return raw_value, "ok"

    def reset(self, channel: str | None = None) -> None:
        """Reset validation state for a specific channel or all channels."""
        if channel is not None:
            self.last_valid_temp[channel] = None
            self.pending_jump_count[channel] = 0
        else:
            self.last_valid_temp = {"co": None, "cwu": None}
            self.pending_jump_count = {"co": 0, "cwu": 0}


# ---------------------------------------------------------------------------
# TFLite Inference Engine (Singleton + threading.Lock)
# ---------------------------------------------------------------------------


class TFLiteInferenceEngine:
    """Singleton TFLite inference engine for digit detection on boiler LCD displays.

    Encapsulates model lifecycle, letterbox pre-processing, thread-safe interpreter
    invocation, YOLO tensor decoding with NMS, and sanity validation.
    """

    _instance: Self | None = None
    _init_lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        """Ensure a single instance of TFLiteInferenceEngine exists."""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, data_dir: str | Path | None = None) -> None:
        if getattr(self, "_initialized", False):
            return

        raw_dir = data_dir or os.getenv("DATA_DIR", "/data")
        self.data_dir = Path(raw_dir)
        self.model_path = self.data_dir / "models" / "active_model.tflite"

        self._interpreter: Any | None = None
        self._input_details: list[dict[str, Any]] | None = None
        self._output_details: list[dict[str, Any]] | None = None
        self._model_mtime: float = 0.0

        self._lock = threading.Lock()
        self.validator = ReadingValidator()

        self._initialized = True

    @property
    def is_loaded(self) -> bool:
        """Check if a TFLite model is currently loaded in memory."""
        return self._interpreter is not None

    def _ensure_model_loaded(self) -> bool:
        """Check if model is loaded and current; reload if model file was modified.

        Returns:
            True if model is ready for inference, False otherwise.
        """
        if self._interpreter is not None:
            if not self.model_path.exists():
                # Model already loaded in memory
                return True
            try:
                if self.model_path.stat().st_mtime == self._model_mtime:
                    return True
            except OSError:
                return True

        if Interpreter is None:
            logger.debug("Cannot load model: no TFLite Interpreter library installed")
            return False

        if not self.model_path.exists():
            return False

        try:
            current_mtime = self.model_path.stat().st_mtime
        except OSError as exc:
            logger.warning("Could not stat model file %s: %s", self.model_path, exc)
            return False

        with self._lock:
            # Double-check inside lock
            if self._interpreter is not None and current_mtime == self._model_mtime:
                return True

            try:
                interpreter = Interpreter(model_path=str(self.model_path))
                interpreter.allocate_tensors()

                self._interpreter = interpreter
                self._input_details = interpreter.get_input_details()
                self._output_details = interpreter.get_output_details()
                self._model_mtime = current_mtime
                self.validator.reset()

                in_shape = self._input_details[0]["shape"] if self._input_details else None
                in_dtype = self._input_details[0]["dtype"] if self._input_details else None
                logger.info(
                    "TFLite model successfully loaded from %s (input_shape=%s, input_dtype=%s)",
                    self.model_path,
                    in_shape,
                    in_dtype,
                )
                return True
            except Exception as exc:
                logger.exception("Failed to load TFLite model from %s: %s", self.model_path, exc)
                self._interpreter = None
                self._input_details = None
                self._output_details = None
                return False

    def reload_model(self) -> bool:
        """Force a reload of the active TFLite model from disk."""
        self._model_mtime = 0.0
        return self._ensure_model_loaded()

    def run_channel_inference(
        self,
        image_bytes: bytes,
        roi: ROIConfig,
        channel: str,
        settings: AppSettings,
    ) -> ChannelInferenceResult:
        """Execute complete inference and post-processing pipeline for a single channel.

        Args:
            image_bytes: Raw JPEG bytes of the full camera frame.
            roi: ROI configuration defining normalized bounding box [0.0 - 1.0].
            channel: Target temperature channel ('co' or 'cwu').
            settings: Add-on configuration settings.

        Returns:
            ChannelInferenceResult with raw and validated integer temperatures,
            detected digit boxes, and an annotated Base64 JPEG.
        """
        # 1. Decode full camera frame and crop ROI
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Failed to decode image bytes for channel '%s'", channel)
            return ChannelInferenceResult(
                channel=channel,
                status="image_decode_failed",
            )

        h_img, w_img = img.shape[:2]
        x1_crop = int(np.clip(round(roi.x * w_img), 0, w_img - 1))
        y1_crop = int(np.clip(round(roi.y * h_img), 0, h_img - 1))
        x2_crop = int(np.clip(round((roi.x + roi.w) * w_img), x1_crop + 1, w_img))
        y2_crop = int(np.clip(round((roi.y + roi.h) * h_img), y1_crop + 1, h_img))

        roi_crop = img[y1_crop:y2_crop, x1_crop:x2_crop]
        if roi_crop.size == 0 or roi_crop.shape[0] < 5 or roi_crop.shape[1] < 5:
            logger.warning("ROI crop for channel '%s' is empty or too small", channel)
            return ChannelInferenceResult(
                channel=channel,
                status="no_roi",
            )

        # 2. Check if model is loaded; if not, return unannotated crop with status 'no_model'
        if not self._ensure_model_loaded():
            raw_b64 = _encode_bgr_to_base64_jpeg(roi_crop)
            return ChannelInferenceResult(
                channel=channel,
                status="no_model",
                annotated_crop_base64=raw_b64,
            )

        assert self._input_details is not None
        assert self._output_details is not None
        assert self._interpreter is not None

        # 3. Pre-processing: Letterbox resize to square model input dimension
        in_shape = self._input_details[0]["shape"]
        in_dtype = self._input_details[0]["dtype"]

        if len(in_shape) == 4 and in_shape[1] == 3:
            # NCHW format
            h_in, w_in = int(in_shape[2]), int(in_shape[3])
            is_nchw = True
        else:
            # Standard NHWC format
            h_in, w_in = int(in_shape[1]), int(in_shape[2])
            is_nchw = False

        roi_h, roi_w = roi_crop.shape[:2]
        scale = min(w_in / roi_w, h_in / roi_h)
        new_w = int(round(roi_w * scale))
        new_h = int(round(roi_h * scale))
        resized = cv2.resize(roi_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_x = (w_in - new_w) // 2
        pad_y = (h_in - new_h) // 2

        canvas = np.full((h_in, w_in, 3), 114, dtype=np.uint8)
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        # Normalize or quantize input tensor
        if in_dtype == np.float32:
            input_tensor = rgb.astype(np.float32) / 255.0
        elif in_dtype in (np.int8, np.uint8):
            quant_params = self._input_details[0].get("quantization_parameters", {})
            scales = quant_params.get("scales", [])
            zero_points = quant_params.get("zero_points", [])
            if len(scales) > 0 and len(zero_points) > 0 and scales[0] > 0:
                scale_q = float(scales[0])
                zp_q = int(zero_points[0])
                norm = rgb.astype(np.float32) / 255.0
                quantized = np.round(norm / scale_q + zp_q)
                input_tensor = np.clip(
                    quantized,
                    np.iinfo(in_dtype).min,
                    np.iinfo(in_dtype).max,
                ).astype(in_dtype)
            else:
                input_tensor = rgb.astype(in_dtype)
        else:
            input_tensor = rgb.astype(np.float32) / 255.0

        if is_nchw:
            input_tensor = np.transpose(input_tensor, (2, 0, 1))

        input_tensor = np.expand_dims(input_tensor, axis=0)

        # 4. Model Inference under thread safety lock
        with self._lock:
            self._interpreter.set_tensor(self._input_details[0]["index"], input_tensor)
            self._interpreter.invoke()
            raw_output = self._interpreter.get_tensor(self._output_details[0]["index"]).copy()

        # Dequantize output tensor if quantized
        out_dtype = self._output_details[0]["dtype"]
        if out_dtype in (np.int8, np.uint8):
            out_quant = self._output_details[0].get("quantization_parameters", {})
            scales = out_quant.get("scales", [])
            zero_points = out_quant.get("zero_points", [])
            if len(scales) > 0 and len(zero_points) > 0 and scales[0] > 0:
                scale_out = float(scales[0])
                zp_out = int(zero_points[0])
                raw_output = (raw_output.astype(np.float32) - zp_out) * scale_out
            else:
                raw_output = raw_output.astype(np.float32)

        # 5. Post-processing YOLO output tensor (10 digit classes: 0..9)
        preds = np.squeeze(raw_output, axis=0)
        if preds.ndim != 2:
            logger.warning("Unexpected YOLO tensor output shape: %s", raw_output.shape)
            raw_b64 = _encode_bgr_to_base64_jpeg(roi_crop)
            return ChannelInferenceResult(
                channel=channel,
                status="unexpected_output_shape",
                annotated_crop_base64=raw_b64,
            )

        # Standard YOLO export has shape (14, N) -> transpose to (N, 14)
        if preds.shape[0] == 14 and preds.shape[1] != 14:
            preds = preds.T
        elif preds.shape[1] < 14:
            logger.warning("YOLO output dimension %s has fewer than 14 elements", preds.shape)
            raw_b64 = _encode_bgr_to_base64_jpeg(roi_crop)
            return ChannelInferenceResult(
                channel=channel,
                status="invalid_tensor_classes",
                annotated_crop_base64=raw_b64,
            )

        max_coord = float(np.max(preds[:, :2])) if len(preds) > 0 else 0.0
        is_normalized = max_coord <= 1.5

        boxes: list[list[int]] = []
        confidences: list[float] = []
        class_ids: list[int] = []

        for row in preds:
            cx = float(row[0])
            cy = float(row[1])
            bw = float(row[2])
            bh = float(row[3])
            scores = row[4:14]

            class_id = int(np.argmax(scores))
            conf = float(np.max(scores))

            if conf < settings.confidence_threshold:
                continue

            if is_normalized:
                cx *= w_in
                cy *= h_in
                bw *= w_in
                bh *= h_in

            # Convert letterbox coordinates back to original ROI crop pixels
            x1 = int(round((cx - bw / 2.0 - pad_x) / scale))
            y1 = int(round((cy - bh / 2.0 - pad_y) / scale))
            box_w = int(round(bw / scale))
            box_h = int(round(bh / scale))

            # Clamp coordinates to ROI crop boundaries
            x1 = max(0, min(x1, roi_w - 1))
            y1 = max(0, min(y1, roi_h - 1))
            box_w = max(1, min(box_w, roi_w - x1))
            box_h = max(1, min(box_h, roi_h - y1))

            boxes.append([x1, y1, box_w, box_h])
            confidences.append(conf)
            class_ids.append(class_id)

        # Non-Maximum Suppression (NMS)
        indices: list[int] = []
        if boxes:
            nms_out = cv2.dnn.NMSBoxes(
                boxes,
                confidences,
                float(settings.confidence_threshold),
                nms_threshold=0.45,
            )
            if len(nms_out) > 0:
                indices = np.array(nms_out).flatten().tolist()

        # 6. Sorting and integer temperature assembly
        accepted_digits: list[DetectedDigit] = []
        for idx in indices:
            bx, by, bw, bh = boxes[idx]
            x_center = bx + bw / 2.0
            accepted_digits.append(
                DetectedDigit(
                    digit=class_ids[idx],
                    confidence=round(confidences[idx], 4),
                    bbox=(bx, by, bw, bh),
                    x_center=round(x_center, 2),
                )
            )

        # Sort left-to-right by horizontal center
        digits = sorted(accepted_digits, key=lambda d: d.x_center)

        # If more than 2 digits detected and max_temp < 100, select top 2 most confident digits
        if len(digits) > 2 and settings.max_temp < 100:
            digits = sorted(digits, key=lambda d: d.confidence, reverse=True)[:2]
            digits = sorted(digits, key=lambda d: d.x_center)

        if digits:
            digit_str = "".join(str(d.digit) for d in digits)
            try:
                raw_value: int | None = int(digit_str)
            except ValueError:
                raw_value = None
            avg_confidence = round(sum(d.confidence for d in digits) / len(digits), 4)
        else:
            raw_value = None
            avg_confidence = 0.0

        # Physical sanity check validation
        validated_value, val_status = self.validator.validate(
            channel=channel,
            raw_value=raw_value,
            min_temp=settings.min_temp,
            max_temp=settings.max_temp,
            max_delta=settings.max_delta,
        )

        if not digits:
            status = "no_digits_detected"
        elif val_status == "ok":
            status = "ok"
        else:
            status = val_status

        # Draw green bounding boxes and confidence labels on ROI crop
        annotated_crop = roi_crop.copy()
        for d in digits:
            bx, by, bw, bh = d.bbox
            cv2.rectangle(annotated_crop, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            label = f"{d.digit} ({d.confidence:.0%})"
            label_y = max(by - 5, 12)
            cv2.putText(
                annotated_crop,
                label,
                (bx, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        annotated_crop_base64 = _encode_bgr_to_base64_jpeg(annotated_crop)

        return ChannelInferenceResult(
            channel=channel,
            raw_value=raw_value,
            validated_value=validated_value,
            confidence=avg_confidence,
            status=status,
            digits=digits,
            annotated_crop_base64=annotated_crop_base64,
            timestamp=datetime.now(UTC).isoformat(),
        )

    def process_channel(
        self,
        channel: str,
        image_bytes: bytes,
        roi: ROIConfig,
        settings: AppSettings,
    ) -> ChannelInferenceResult:
        """Convenience alias for run_channel_inference."""
        return self.run_channel_inference(image_bytes, roi, channel, settings)


# Alias for backward compatibility
InferenceEngine = TFLiteInferenceEngine
