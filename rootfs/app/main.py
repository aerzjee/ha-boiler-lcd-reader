"""Boiler LCD Reader – FastAPI application entry point.

Manages application lifespan, Ingress proxy security, Web UI serving,
REST API endpoints for configuration, camera discovery, live preview,
sample collection, TFLite model management, background inference worker,
and live diagnostics.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.collector import SampleCollector, crop_roi
from app.config_manager import AppSettings, ConfigManager
from app.ha_client import HAClientError, HomeAssistantClient
from app.inference import ChannelInferenceResult, TFLiteInferenceEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------

INGRESS_PROXY_IP = "172.30.32.2"
DEV_MODE = os.getenv("DEV_MODE", "false").lower() == "true"
MAX_MODEL_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB max for .tflite models
TFLITE_MAGIC_HEADER = b"TFL3"

STATIC_DIR = Path(__file__).parent / "static"

FRIENDLY_NAME_CO = "Piec CO Temperatura"
FRIENDLY_NAME_CWU = "Piec CWU Temperatura"


# ---------------------------------------------------------------------------
# Single Cycle Inference Execution
# ---------------------------------------------------------------------------


async def execute_single_cycle(publish_to_ha: bool = True) -> dict[str, Any]:
    """Execute a single OCR inference cycle across configured camera ROIs.

    Args:
        publish_to_ha: Whether to publish validated integer temperatures to Home Assistant sensors.

    Returns:
        Summary dictionary with inference status and channel results.
    """
    config_mgr: ConfigManager = app.state.config_manager
    ha_client: HomeAssistantClient = app.state.ha_client
    collector: SampleCollector = app.state.collector
    engine: TFLiteInferenceEngine = app.state.inference_engine

    settings = config_mgr.get_settings()
    now_iso = datetime.now(UTC).isoformat()

    # 1. Check if camera is configured
    if not settings.camera_entity:
        app.state.last_run_at = now_iso
        return {
            "status": "no_camera",
            "message": "Brak skonfigurowanej encji kamery w ustawieniach",
            "timestamp": now_iso,
        }

    # 2. Fetch camera frame
    try:
        image_bytes = await ha_client.get_camera_snapshot(settings.camera_entity)
    except Exception as exc:
        logger.warning("Failed to fetch camera snapshot from %s: %s", settings.camera_entity, exc)
        app.state.last_error = f"Błąd pobierania klatki: {exc}"
        app.state.last_run_at = now_iso
        return {
            "status": "camera_error",
            "message": str(exc),
            "timestamp": now_iso,
        }

    # 3. Feed background collector if enabled and in automatic loop
    if settings.collector_enabled and publish_to_ha:
        with suppress(Exception):
            await asyncio.to_thread(collector.process_and_save_background_sample, image_bytes, settings)

    # 4. Central Heating (CO) channel inference
    res_co: ChannelInferenceResult | None = None
    if settings.roi_co:
        try:
            res_co = await asyncio.to_thread(engine.run_channel_inference, image_bytes, settings.roi_co, "co", settings)
            if publish_to_ha and res_co.validated_value is not None:
                with suppress(Exception):
                    await ha_client.publish_temperature(
                        settings.sensor_co_entity,
                        res_co.validated_value,
                        FRIENDLY_NAME_CO,
                        {"confidence": round(res_co.confidence, 3)},
                    )
        except Exception as exc:
            logger.exception("Inference error for CO channel: %s", exc)

    # 5. Domestic Hot Water (CWU) channel inference
    res_cwu: ChannelInferenceResult | None = None
    if settings.roi_cwu:
        try:
            res_cwu = await asyncio.to_thread(
                engine.run_channel_inference, image_bytes, settings.roi_cwu, "cwu", settings
            )
            if publish_to_ha and res_cwu.validated_value is not None:
                with suppress(Exception):
                    await ha_client.publish_temperature(
                        settings.sensor_cwu_entity,
                        res_cwu.validated_value,
                        FRIENDLY_NAME_CWU,
                        {"confidence": round(res_cwu.confidence, 3)},
                    )
        except Exception as exc:
            logger.exception("Inference error for CWU channel: %s", exc)

    # 6. Store latest results in application state
    app.state.last_results["co"] = res_co
    app.state.last_results["cwu"] = res_cwu
    app.state.last_run_at = now_iso
    app.state.last_error = None

    return {
        "status": "ok",
        "timestamp": now_iso,
        "co": res_co.model_dump() if res_co else None,
        "cwu": res_cwu.model_dump() if res_cwu else None,
    }


# ---------------------------------------------------------------------------
# Background Worker Task
# ---------------------------------------------------------------------------


async def background_worker() -> None:
    """Asynchronous background loop polling the camera and publishing readings."""
    logger.info("Background inference worker started.")
    while True:
        try:
            config_mgr: ConfigManager = app.state.config_manager
            settings = config_mgr.get_settings()
            interval = settings.scan_interval_seconds

            try:
                await execute_single_cycle(publish_to_ha=True)
            except Exception as exc:
                logger.error("Error during background inference cycle: %s", exc)
                app.state.last_error = str(exc)

            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("Background inference worker cancelled.")
            break
        except Exception as exc:
            logger.exception("Unexpected error in background worker loop: %s", exc)
            app.state.last_error = str(exc)
            await asyncio.sleep(10)


# ---------------------------------------------------------------------------
# Lifespan Management
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup and clean up on shutdown."""
    config_mgr = ConfigManager()
    config_mgr.load_settings()

    ha_client = HomeAssistantClient()
    await ha_client.start()

    collector = SampleCollector(data_dir=config_mgr.data_dir)
    inference_engine = TFLiteInferenceEngine(data_dir=config_mgr.data_dir)
    inference_engine.reload_model()

    app.state.config_manager = config_mgr
    app.state.ha_client = ha_client
    app.state.collector = collector
    app.state.inference_engine = inference_engine

    app.state.last_results = {"co": None, "cwu": None}
    app.state.last_run_at = None
    app.state.last_error = None

    worker_task = asyncio.create_task(background_worker())

    logger.info("Boiler LCD Reader backend initialized successfully.")
    yield

    logger.info("Shutting down Boiler LCD Reader backend...")
    worker_task.cancel()
    with suppress(asyncio.CancelledError):
        await worker_task
    await ha_client.close()


# ---------------------------------------------------------------------------
# Application Instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Boiler LCD Reader",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def get_config_manager(request: Request) -> ConfigManager:
    """Dependency provider for ConfigManager."""
    return request.app.state.config_manager


def get_ha_client(request: Request) -> HomeAssistantClient:
    """Dependency provider for HomeAssistantClient."""
    return request.app.state.ha_client


def get_collector(request: Request) -> SampleCollector:
    """Dependency provider for SampleCollector."""
    return request.app.state.collector


def get_inference_engine(request: Request) -> TFLiteInferenceEngine:
    """Dependency provider for TFLiteInferenceEngine."""
    return request.app.state.inference_engine


# ---------------------------------------------------------------------------
# Middleware – Ingress IP verification
# ---------------------------------------------------------------------------


@app.middleware("http")
async def verify_ingress_proxy(request: Request, call_next):
    """Reject requests not originating from the HA Ingress proxy.

    In development mode (DEV_MODE=true) this check is skipped so the
    add-on can be tested outside of a Home Assistant environment.
    """
    if not DEV_MODE:
        client_ip = request.client.host if request.client else None
        if client_ip != INGRESS_PROXY_IP:
            return JSONResponse(
                status_code=403,
                content={"detail": f"Forbidden: requests must originate from Ingress proxy ({INGRESS_PROXY_IP})"},
            )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Static Web UI Routes
# ---------------------------------------------------------------------------

STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def serve_index():
    """Serve the single-page application dashboard for Home Assistant Ingress."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse(
            status_code=404,
            content={"detail": "Web UI index.html not found. Please ensure static files are installed."},
        )
    return FileResponse(index_file)


# ---------------------------------------------------------------------------
# Health & Status Routes
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health_check():
    """Return service health status."""
    return {"status": "ok", "version": "0.1.0"}


@app.get("/api/status")
async def get_system_status(
    config_mgr: Annotated[ConfigManager, Depends(get_config_manager)],
    inference_engine: Annotated[TFLiteInferenceEngine, Depends(get_inference_engine)],
) -> dict[str, Any]:
    """Return real-time state of the OCR inference engine and last temperature readings."""
    models_dir = config_mgr.data_dir / "models"
    model_exists = (models_dir / "active_model.tflite").exists()

    last_co: ChannelInferenceResult | None = app.state.last_results.get("co")
    last_cwu: ChannelInferenceResult | None = app.state.last_results.get("cwu")

    return {
        "status": (
            "ready" if inference_engine.is_loaded else ("waiting_for_model" if not model_exists else "model_load_error")
        ),
        "model_loaded": inference_engine.is_loaded,
        "last_run_at": getattr(app.state, "last_run_at", None),
        "last_error": getattr(app.state, "last_error", None),
        "last_co_temp": last_co.validated_value if last_co else None,
        "last_cwu_temp": last_cwu.validated_value if last_cwu else None,
        "last_results": {
            "co": last_co.model_dump() if last_co else None,
            "cwu": last_cwu.model_dump() if last_cwu else None,
        },
    }


# ---------------------------------------------------------------------------
# Diagnostics & Test Inference Routes
# ---------------------------------------------------------------------------


@app.post("/api/inference/test")
async def trigger_test_inference() -> dict[str, Any]:
    """Execute immediate manual inference cycle and return fresh diagnostic results."""
    try:
        return await execute_single_cycle(publish_to_ha=True)
    except Exception as exc:
        logger.exception("Manual test inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Błąd testowego odczytu: {exc}") from exc


# ---------------------------------------------------------------------------
# Settings Routes
# ---------------------------------------------------------------------------


@app.get("/api/settings", response_model=AppSettings)
async def get_settings(
    config_mgr: Annotated[ConfigManager, Depends(get_config_manager)],
) -> AppSettings:
    """Fetch the current application settings."""
    return config_mgr.get_settings()


@app.put("/api/settings", response_model=AppSettings)
async def update_settings(
    settings: AppSettings,
    config_mgr: Annotated[ConfigManager, Depends(get_config_manager)],
) -> AppSettings:
    """Validate and atomically persist updated application settings."""
    try:
        return await config_mgr.save_settings(settings)
    except Exception as exc:
        logger.exception("Failed to persist updated settings: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to persist settings: {exc}") from exc


# ---------------------------------------------------------------------------
# Camera Routes
# ---------------------------------------------------------------------------


@app.get("/api/cameras")
async def list_cameras(
    ha_client: Annotated[HomeAssistantClient, Depends(get_ha_client)],
) -> list[dict[str, str]]:
    """List all available camera entities in Home Assistant."""
    try:
        return await ha_client.get_camera_entities()
    except HAClientError as exc:
        logger.error("Failed to query camera entities from Home Assistant: %s", exc)
        status_code = exc.status_code if exc.status_code and exc.status_code >= 400 else 502
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc


@app.get("/api/camera/preview")
async def get_camera_preview(
    config_mgr: Annotated[ConfigManager, Depends(get_config_manager)],
    ha_client: Annotated[HomeAssistantClient, Depends(get_ha_client)],
    entity_id: Annotated[str | None, Query(description="Optional camera entity ID override")] = None,
) -> Response:
    """Fetch current camera frame for UI display and ROI configuration."""
    target_entity = entity_id or config_mgr.get_settings().camera_entity
    if not target_entity:
        raise HTTPException(
            status_code=400,
            detail="No camera configured in settings and no 'entity_id' parameter provided.",
        )

    try:
        image_bytes = await ha_client.get_camera_snapshot(target_entity)
        return Response(
            content=image_bytes,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )
    except HAClientError as exc:
        logger.error("Failed to fetch camera preview for %s: %s", target_entity, exc)
        status_code = exc.status_code if exc.status_code and exc.status_code >= 400 else 502
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Model Management Routes
# ---------------------------------------------------------------------------


@app.get("/api/model/status")
async def get_model_status(
    config_mgr: Annotated[ConfigManager, Depends(get_config_manager)],
) -> dict[str, Any]:
    """Return metadata about the currently uploaded active_model.tflite."""
    models_dir = config_mgr.data_dir / "models"
    model_path = models_dir / "active_model.tflite"

    if not model_path.exists():
        return {
            "exists": False,
            "size_kb": 0.0,
            "modified_at": None,
            "filename": "active_model.tflite",
        }

    stat = model_path.stat()
    mod_time = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
    return {
        "exists": True,
        "size_kb": round(stat.st_size / 1024, 2),
        "modified_at": mod_time,
        "filename": "active_model.tflite",
    }


@app.post("/api/model/upload")
async def upload_model(
    config_mgr: Annotated[ConfigManager, Depends(get_config_manager)],
    inference_engine: Annotated[TFLiteInferenceEngine, Depends(get_inference_engine)],
    file: Annotated[UploadFile, File(description="TensorFlow Lite v3 model file (.tflite)")],
) -> dict[str, Any]:
    """Safely validate and persist an uploaded TFLite model file to /data/models/active_model.tflite."""
    if not file.filename or not file.filename.lower().endswith(".tflite"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. File must have a '.tflite' extension.",
        )

    # Read content with hard 25 MB quota check
    chunks: list[bytes] = []
    read_total = 0
    chunk_size = 64 * 1024

    while chunk := await file.read(chunk_size):
        read_total += len(chunk)
        if read_total > MAX_MODEL_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum allowed size of {MAX_MODEL_SIZE_BYTES // (1024 * 1024)} MB.",
            )
        chunks.append(chunk)

    content = b"".join(chunks)

    # FlatBuffer validation: verify length and TFL3 header identifier at bytes 4..8
    if len(content) < 8 or content[4:8] != TFLITE_MAGIC_HEADER:
        raise HTTPException(
            status_code=400,
            detail="Invalid TFLite model: missing TFL3 FlatBuffer header identifier at offset 4.",
        )

    # Atomically write to /data/models/active_model.tflite
    models_dir = config_mgr.data_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    target_path = models_dir / "active_model.tflite"
    tmp_path = models_dir / "active_model.tflite.tmp"

    try:
        tmp_path.write_bytes(content)
        os.replace(tmp_path, target_path)
    except Exception as exc:
        if tmp_path.exists():
            with suppress(OSError):
                tmp_path.unlink()
        logger.exception("Failed to write uploaded model: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to persist model: {exc}") from exc

    # Reload model in inference engine Singleton
    reloaded = inference_engine.reload_model()
    logger.info(
        "Successfully uploaded new model to %s (%d bytes, reloaded=%s)",
        target_path,
        len(content),
        reloaded,
    )

    status = await get_model_status(config_mgr)
    status["model_reloaded"] = reloaded
    return status


# ---------------------------------------------------------------------------
# Collector & Dataset Routes
# ---------------------------------------------------------------------------


@app.get("/api/collector/stats")
async def get_collector_stats(
    config_mgr: Annotated[ConfigManager, Depends(get_config_manager)],
    collector: Annotated[SampleCollector, Depends(get_collector)],
) -> dict[str, Any]:
    """Fetch storage statistics and quota status for collected training samples."""
    stats = collector.get_stats()
    stats["collector_enabled"] = config_mgr.get_settings().collector_enabled
    return stats


@app.get("/api/snapshot/download")
async def download_snapshot(
    config_mgr: Annotated[ConfigManager, Depends(get_config_manager)],
    ha_client: Annotated[HomeAssistantClient, Depends(get_ha_client)],
    crop: Annotated[str, Query(description="Crop mode: 'full', 'co', or 'cwu'")] = "full",
) -> Response:
    """Download on-demand frame snapshot (full or cropped to specific ROI) as JPEG attachment."""
    settings = config_mgr.get_settings()
    if not settings.camera_entity:
        raise HTTPException(status_code=400, detail="No camera configured in settings.")

    try:
        raw_bytes = await ha_client.get_camera_snapshot(settings.camera_entity)
    except HAClientError as exc:
        logger.error("Failed to fetch snapshot for download: %s", exc)
        status_code = exc.status_code if exc.status_code and exc.status_code >= 400 else 502
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    crop_mode = crop.lower().strip()
    if crop_mode == "full":
        result_bytes = raw_bytes
    elif crop_mode == "co":
        if not settings.roi_co:
            raise HTTPException(status_code=400, detail="Central Heating (CO) ROI is not configured.")
        try:
            result_bytes = crop_roi(raw_bytes, settings.roi_co)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to crop CO ROI: {exc}") from exc
    elif crop_mode == "cwu":
        if not settings.roi_cwu:
            raise HTTPException(status_code=400, detail="Domestic Hot Water (CWU) ROI is not configured.")
        try:
            result_bytes = crop_roi(raw_bytes, settings.roi_cwu)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to crop CWU ROI: {exc}") from exc
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid crop mode '{crop}'. Must be 'full', 'co', or 'cwu'.",
        )

    ts_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    filename = f"snapshot_{ts_str}_{crop_mode}.jpg"

    return Response(
        content=result_bytes,
        media_type="image/jpeg",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        },
    )


@app.post("/api/collector/capture-now")
async def capture_now(
    config_mgr: Annotated[ConfigManager, Depends(get_config_manager)],
    ha_client: Annotated[HomeAssistantClient, Depends(get_ha_client)],
    collector: Annotated[SampleCollector, Depends(get_collector)],
) -> dict[str, Any]:
    """Force an immediate capture of full frame and ROIs into /data/samples."""
    settings = config_mgr.get_settings()
    if not settings.camera_entity:
        raise HTTPException(status_code=400, detail="No camera configured in settings.")

    try:
        raw_bytes = await ha_client.get_camera_snapshot(settings.camera_entity)
    except HAClientError as exc:
        logger.error("Failed to fetch camera snapshot for capture-now: %s", exc)
        status_code = exc.status_code if exc.status_code and exc.status_code >= 400 else 502
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    try:
        updated_stats = await collector.capture_now(raw_bytes, settings)
        updated_stats["collector_enabled"] = settings.collector_enabled
        return updated_stats
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to capture training sample: %s", exc)
        raise HTTPException(status_code=500, detail=f"Capture failed: {exc}") from exc


@app.get("/api/samples/export")
async def export_samples(
    config_mgr: Annotated[ConfigManager, Depends(get_config_manager)],
    collector: Annotated[SampleCollector, Depends(get_collector)],
) -> StreamingResponse:
    """Export all collected training samples as a ZIP archive with metadata."""
    settings = config_mgr.get_settings()
    zip_buffer = await asyncio.to_thread(collector.export_samples_zip, settings)

    ts_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    filename = f"boiler_training_samples_{ts_str}.zip"

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        },
    )


@app.delete("/api/samples/clear")
async def clear_samples(
    collector: Annotated[SampleCollector, Depends(get_collector)],
) -> dict[str, Any]:
    """Delete all training sample images from /data/samples."""
    deleted_count = await asyncio.to_thread(collector.clear_samples)
    return {"status": "ok", "deleted_count": deleted_count}
