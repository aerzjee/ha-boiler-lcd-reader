/**
 * Boiler LCD Reader – Frontend Application Logic
 *
 * CRITICAL RULE: All API calls and asset links MUST use relative paths without
 * leading slash (e.g. "api/settings", "static/app.js") for Home Assistant Ingress compatibility.
 */

(() => {
  "use strict";

  // -------------------------------------------------------------------------
  // Application State
  // -------------------------------------------------------------------------

  const state = {
    settings: null,
    cameras: [],
    modelStatus: null,
    collectorStats: null,
    currentImage: null,
    roiCo: null,
    roiCwu: null,
    currentDrawingMode: "co", // 'co' | 'cwu'
    isDrawing: false,
    startPos: { x: 0, y: 0 },
    currentPos: { x: 0, y: 0 },
  };

  // -------------------------------------------------------------------------
  // DOM Elements
  // -------------------------------------------------------------------------

  const el = {
    // Header
    modelStatusPill: document.getElementById("model-status-pill"),
    modelStatusText: document.getElementById("model-status-text"),
    valCo: document.getElementById("val-co"),
    valCwu: document.getElementById("val-cwu"),

    // Canvas & Preview
    cameraCanvas: document.getElementById("camera-canvas"),
    canvasPlaceholder: document.getElementById("canvas-placeholder"),
    previewStatus: document.getElementById("preview-status"),
    btnRefreshPreview: document.getElementById("btn-refresh-preview"),
    btnClearRois: document.getElementById("btn-clear-rois"),
    btnSaveRois: document.getElementById("btn-save-rois"),
    coordsCo: document.getElementById("coords-co"),
    coordsCwu: document.getElementById("coords-cwu"),
    roiModeInputs: document.querySelectorAll('input[name="roi-mode"]'),

    // Settings Form
    settingsForm: document.getElementById("settings-form"),
    cameraSelect: document.getElementById("camera-select"),
    btnRefreshCameras: document.getElementById("btn-refresh-cameras"),
    scanInterval: document.getElementById("scan-interval"),
    confidenceThreshold: document.getElementById("confidence-threshold"),
    minTemp: document.getElementById("min-temp"),
    maxTemp: document.getElementById("max-temp"),
    maxDelta: document.getElementById("max-delta"),
    sensorCo: document.getElementById("sensor-co"),
    sensorCwu: document.getElementById("sensor-cwu"),

    // Collector
    collectorEnabledSwitch: document.getElementById("collector-enabled-switch"),
    collectorStatusBadge: document.getElementById("collector-status-badge"),
    quotaText: document.getElementById("quota-text"),
    quotaProgress: document.getElementById("quota-progress"),
    btnDownloadFull: document.getElementById("btn-download-full"),
    btnDownloadCo: document.getElementById("btn-download-co"),
    btnDownloadCwu: document.getElementById("btn-download-cwu"),
    btnCaptureNow: document.getElementById("btn-capture-now"),
    btnExportZip: document.getElementById("btn-export-zip"),
    btnClearSamples: document.getElementById("btn-clear-samples"),

    // Model Management
    modelFileBadge: document.getElementById("model-file-badge"),
    modelFilename: document.getElementById("model-filename"),
    modelSize: document.getElementById("model-size"),
    modelMtime: document.getElementById("model-mtime"),
    modelFileInput: document.getElementById("model-file-input"),
    btnUploadModel: document.getElementById("btn-upload-model"),

    // Diagnostics & Live Inference – Card 5
    diagWorkerBadge: document.getElementById("diag-worker-badge"),
    btnTestInference: document.getElementById("btn-test-inference"),
    diagImgCo: document.getElementById("diag-img-co"),
    diagImgCoPlaceholder: document.getElementById("diag-img-co-placeholder"),
    diagValCo: document.getElementById("diag-val-co"),
    diagRawCo: document.getElementById("diag-raw-co"),
    diagConfCo: document.getElementById("diag-conf-co"),
    diagStatusCo: document.getElementById("diag-status-co"),
    diagImgCwu: document.getElementById("diag-img-cwu"),
    diagImgCwuPlaceholder: document.getElementById("diag-img-cwu-placeholder"),
    diagValCwu: document.getElementById("diag-val-cwu"),
    diagRawCwu: document.getElementById("diag-raw-cwu"),
    diagConfCwu: document.getElementById("diag-conf-cwu"),
    diagStatusCwu: document.getElementById("diag-status-cwu"),
    diagLastTimestamp: document.getElementById("diag-last-timestamp"),
    diagLastError: document.getElementById("diag-last-error"),

    // Toasts
    toastContainer: document.getElementById("toast-container"),
  };

  const ctx = el.cameraCanvas.getContext("2d");

  // -------------------------------------------------------------------------
  // Notifications (Toast)
  // -------------------------------------------------------------------------

  function showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;

    el.toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateY(10px)";
      toast.style.transition = "all 0.3s ease";
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }

  // -------------------------------------------------------------------------
  // API Fetch Helpers (Strictly Relative Paths)
  // -------------------------------------------------------------------------

  async function apiGet(endpoint) {
    const res = await fetch(endpoint);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Błąd HTTP ${res.status}`);
    }
    return res.json();
  }

  async function apiPut(endpoint, data) {
    const res = await fetch(endpoint, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Błąd HTTP ${res.status}`);
    }
    return res.json();
  }

  async function apiPost(endpoint, body = null) {
    const options = { method: "POST" };
    if (body) {
      if (body instanceof FormData) {
        options.body = body;
      } else {
        options.headers = { "Content-Type": "application/json" };
        options.body = JSON.stringify(body);
      }
    }
    const res = await fetch(endpoint, options);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Błąd HTTP ${res.status}`);
    }
    return res.json();
  }

  async function apiDelete(endpoint) {
    const res = await fetch(endpoint, { method: "DELETE" });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Błąd HTTP ${res.status}`);
    }
    return res.json();
  }

  // -------------------------------------------------------------------------
  // Data Loading & Synchronization
  // -------------------------------------------------------------------------

  async function loadSettings() {
    try {
      state.settings = await apiGet("api/settings");
      state.roiCo = state.settings.roi_co || null;
      state.roiCwu = state.settings.roi_cwu || null;

      // Populate form fields
      el.scanInterval.value = state.settings.scan_interval_seconds;
      el.confidenceThreshold.value = state.settings.confidence_threshold;
      el.minTemp.value = state.settings.min_temp;
      el.maxTemp.value = state.settings.max_temp;
      el.maxDelta.value = state.settings.max_delta;
      el.sensorCo.value = state.settings.sensor_co_entity;
      el.sensorCwu.value = state.settings.sensor_cwu_entity;

      el.collectorEnabledSwitch.checked = Boolean(state.settings.collector_enabled);
      updateCollectorBadge(state.settings.collector_enabled);

      if (state.settings.camera_entity) {
        el.cameraSelect.value = state.settings.camera_entity;
      }

      updateCoordsDisplay();
      drawCanvas();
    } catch (err) {
      showToast(`Błąd ładowania ustawień: ${err.message}`, "error");
    }
  }

  async function loadCameras() {
    try {
      state.cameras = await apiGet("api/cameras");
      const currentVal = el.cameraSelect.value;
      el.cameraSelect.innerHTML = '<option value="">Wybierz kamerę...</option>';

      state.cameras.forEach((cam) => {
        const opt = document.createElement("option");
        opt.value = cam.entity_id;
        opt.textContent = `${cam.friendly_name} (${cam.entity_id})`;
        el.cameraSelect.appendChild(opt);
      });

      if (state.settings && state.settings.camera_entity) {
        el.cameraSelect.value = state.settings.camera_entity;
      } else if (currentVal) {
        el.cameraSelect.value = currentVal;
      }
    } catch (err) {
      showToast(`Błąd pobierania listy kamer: ${err.message}`, "error");
    }
  }

  async function loadModelStatus() {
    try {
      state.modelStatus = await apiGet("api/model/status");

      if (state.modelStatus.exists) {
        el.modelStatusPill.className = "status-pill active";
        el.modelStatusText.textContent = "Model aktywny";
        el.modelFileBadge.className = "badge active";
        el.modelFileBadge.textContent = "Wgrany";
        el.modelSize.textContent = `${state.modelStatus.size_kb} KB`;
        el.modelMtime.textContent = new Date(state.modelStatus.modified_at).toLocaleString();
      } else {
        el.modelStatusPill.className = "status-pill inactive";
        el.modelStatusText.textContent = "Brak modelu .tflite";
        el.modelFileBadge.className = "badge warning";
        el.modelFileBadge.textContent = "Brak pliku";
        el.modelSize.textContent = "--";
        el.modelMtime.textContent = "--";
      }
    } catch (err) {
      console.warn("Could not check model status:", err);
    }
  }

  async function loadCollectorStats() {
    try {
      state.collectorStats = await apiGet("api/collector/stats");
      const count = state.collectorStats.count || 0;
      const maxCount = state.collectorStats.max_count || 300;
      const sizeMb = state.collectorStats.size_mb || 0.0;
      const percent = Math.min(100, Math.round((count / maxCount) * 100));

      el.quotaText.textContent = `${count} / ${maxCount} plików (${sizeMb} MB)`;
      el.quotaProgress.style.width = `${percent}%`;

      if (percent >= 100) {
        el.quotaProgress.style.backgroundColor = "var(--danger)";
      } else if (percent >= 80) {
        el.quotaProgress.style.backgroundColor = "var(--warning)";
      } else {
        el.quotaProgress.style.backgroundColor = "var(--primary)";
      }
    } catch (err) {
      console.warn("Could not load collector stats:", err);
    }
  }

  function updateDiagnosticChannel(channel, data) {
    const isCo = channel === "co";
    const imgEl = isCo ? el.diagImgCo : el.diagImgCwu;
    const placeholderEl = isCo ? el.diagImgCoPlaceholder : el.diagImgCwuPlaceholder;
    const valEl = isCo ? el.diagValCo : el.diagValCwu;
    const rawEl = isCo ? el.diagRawCo : el.diagRawCwu;
    const confEl = isCo ? el.diagConfCo : el.diagConfCwu;
    const statusEl = isCo ? el.diagStatusCo : el.diagStatusCwu;

    if (!data) {
      imgEl.style.display = "none";
      placeholderEl.style.display = "flex";
      placeholderEl.querySelector("span").textContent = "Brak odczytu dla tego kanału";
      valEl.textContent = "-- °C";
      rawEl.textContent = "--";
      confEl.textContent = "--%";
      statusEl.textContent = "Brak danych";
      statusEl.className = "diag-status-badge";
      return;
    }

    // Image display
    if (data.annotated_crop_base64) {
      imgEl.src = `data:image/jpeg;base64,${data.annotated_crop_base64}`;
      imgEl.style.display = "block";
      placeholderEl.style.display = "none";
    } else {
      imgEl.style.display = "none";
      placeholderEl.style.display = "flex";
      placeholderEl.querySelector("span").textContent =
        data.status === "no_model" ? "Brak wgranego modelu .tflite" : "Brak obrazu diagnostycznego";
    }

    // Validated value
    valEl.textContent = data.validated_value !== null && data.validated_value !== undefined
      ? `${data.validated_value} °C`
      : "-- °C";

    // Raw value
    rawEl.textContent = data.raw_value !== null && data.raw_value !== undefined
      ? `${data.raw_value}`
      : "--";

    // Confidence
    confEl.textContent = data.confidence !== null && data.confidence !== undefined
      ? `${Math.round(data.confidence * 100)}%`
      : "--%";

    // Status badge
    const statusStr = data.status || "unknown";
    statusEl.textContent = statusStr;
    statusEl.className = "diag-status-badge";
    if (statusStr === "ok") {
      statusEl.classList.add("status-ok");
    } else if (statusStr.includes("pending") || statusStr === "waiting_for_model") {
      statusEl.classList.add("status-warning");
    } else {
      statusEl.classList.add("status-danger");
    }
  }

  async function loadSystemStatus() {
    try {
      const status = await apiGet("api/status");

      // Header temperature badges
      el.valCo.textContent = status.last_co_temp !== null && status.last_co_temp !== undefined
        ? `${status.last_co_temp} °C`
        : "-- °C";
      el.valCwu.textContent = status.last_cwu_temp !== null && status.last_cwu_temp !== undefined
        ? `${status.last_cwu_temp} °C`
        : "-- °C";

      // Diagnostics worker status badge
      if (status.model_loaded) {
        el.diagWorkerBadge.className = "badge active";
        el.diagWorkerBadge.textContent = "Worker aktywny";
      } else {
        el.diagWorkerBadge.className = "badge warning";
        el.diagWorkerBadge.textContent = status.status === "waiting_for_model" ? "Brak modelu" : "Worker nieaktywny";
      }

      // Populate diagnostic panels
      if (status.last_results) {
        updateDiagnosticChannel("co", status.last_results.co);
        updateDiagnosticChannel("cwu", status.last_results.cwu);
      }

      // Timestamp & error
      if (status.last_run_at) {
        el.diagLastTimestamp.textContent = new Date(status.last_run_at).toLocaleString();
      }
      if (status.last_error) {
        el.diagLastError.textContent = `Błąd: ${status.last_error}`;
      } else {
        el.diagLastError.textContent = "";
      }
    } catch (err) {
      console.warn("Could not load system status:", err);
    }
  }

  function updateCollectorBadge(enabled) {
    if (enabled) {
      el.collectorStatusBadge.className = "badge active";
      el.collectorStatusBadge.textContent = "Collector aktywny";
    } else {
      el.collectorStatusBadge.className = "badge";
      el.collectorStatusBadge.textContent = "Collector wyłączony";
    }
  }

  // -------------------------------------------------------------------------
  // Camera Preview & Canvas Rendering
  // -------------------------------------------------------------------------

  async function refreshCameraPreview() {
    el.previewStatus.textContent = "Pobieranie klatki...";
    el.canvasPlaceholder.style.display = "flex";

    const img = new Image();
    const cacheBuster = `t=${Date.now()}`;
    const selectedCam = el.cameraSelect.value;
    const url = selectedCam
      ? `api/camera/preview?entity_id=${encodeURIComponent(selectedCam)}&${cacheBuster}`
      : `api/camera/preview?${cacheBuster}`;

    img.onload = () => {
      state.currentImage = img;
      el.cameraCanvas.width = img.naturalWidth;
      el.cameraCanvas.height = img.naturalHeight;
      el.canvasPlaceholder.style.display = "none";
      el.previewStatus.textContent = `${img.naturalWidth}x${img.naturalHeight} px`;
      drawCanvas();
    };

    img.onerror = () => {
      el.canvasPlaceholder.style.display = "flex";
      el.canvasPlaceholder.innerHTML = "<p>❌ Błąd pobierania klatki z kamery. Upewnij się, że kamera jest skonfigurowana.</p>";
      el.previewStatus.textContent = "Błąd kamery";
      showToast("Nie udało się pobrać klatki z kamery.", "error");
    };

    img.src = url;
  }

  function drawCanvas() {
    if (!state.currentImage) return;

    const w = el.cameraCanvas.width;
    const h = el.cameraCanvas.height;

    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(state.currentImage, 0, 0, w, h);

    // 1. Draw saved CO ROI
    if (state.roiCo) {
      drawRoiRect(state.roiCo, "CO", "#ff5252", "rgba(255, 82, 82, 0.25)");
    }

    // 2. Draw saved CWU ROI
    if (state.roiCwu) {
      drawRoiRect(state.roiCwu, "CWU", "#4caf50", "rgba(76, 175, 80, 0.25)");
    }

    // 3. Draw active rectangle during drag
    if (state.isDrawing) {
      const x_px = Math.min(state.startPos.x, state.currentPos.x);
      const y_px = Math.min(state.startPos.y, state.currentPos.y);
      const w_px = Math.abs(state.currentPos.x - state.startPos.x);
      const h_px = Math.abs(state.currentPos.y - state.startPos.y);

      const color = state.currentDrawingMode === "co" ? "#ff5252" : "#4caf50";
      const fillColor = state.currentDrawingMode === "co"
        ? "rgba(255, 82, 82, 0.35)"
        : "rgba(76, 175, 80, 0.35)";

      ctx.save();
      ctx.strokeStyle = color;
      ctx.fillStyle = fillColor;
      ctx.lineWidth = 3;
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(x_px, y_px, w_px, h_px);
      ctx.fillRect(x_px, y_px, w_px, h_px);
      ctx.restore();
    }
  }

  function drawRoiRect(roi, label, strokeColor, fillColor) {
    const w_canvas = el.cameraCanvas.width;
    const h_canvas = el.cameraCanvas.height;

    const x = Math.round(roi.x * w_canvas);
    const y = Math.round(roi.y * h_canvas);
    const w = Math.round(roi.w * w_canvas);
    const h = Math.round(roi.h * h_canvas);

    ctx.save();
    ctx.strokeStyle = strokeColor;
    ctx.fillStyle = fillColor;
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, w, h);
    ctx.fillRect(x, y, w, h);

    // Label badge above or inside box
    const labelText = `${label} (${Math.round(roi.w * 100)}% x ${Math.round(roi.h * 100)}%)`;
    ctx.font = "bold 14px monospace";
    const textMetrics = ctx.measureText(labelText);
    const badgeW = textMetrics.width + 12;
    const badgeH = 22;
    const badgeY = y >= badgeH ? y - badgeH : y;

    ctx.fillStyle = strokeColor;
    ctx.fillRect(x, badgeY, badgeW, badgeH);

    ctx.fillStyle = "#ffffff";
    ctx.fillText(labelText, x + 6, badgeY + 16);
    ctx.restore();
  }

  function updateCoordsDisplay() {
    if (state.roiCo) {
      el.coordsCo.textContent = `x:${state.roiCo.x.toFixed(3)} y:${state.roiCo.y.toFixed(3)} w:${state.roiCo.w.toFixed(3)} h:${state.roiCo.h.toFixed(3)}`;
    } else {
      el.coordsCo.textContent = "Brak zaznaczenia";
    }

    if (state.roiCwu) {
      el.coordsCwu.textContent = `x:${state.roiCwu.x.toFixed(3)} y:${state.roiCwu.y.toFixed(3)} w:${state.roiCwu.w.toFixed(3)} h:${state.roiCwu.h.toFixed(3)}`;
    } else {
      el.coordsCwu.textContent = "Brak zaznaczenia";
    }
  }

  // -------------------------------------------------------------------------
  // Canvas Mouse & Touch Event Handlers
  // -------------------------------------------------------------------------

  function getCanvasCoords(e) {
    const rect = el.cameraCanvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    const scaleX = el.cameraCanvas.width / rect.width;
    const scaleY = el.cameraCanvas.height / rect.height;

    return {
      x: Math.max(0, Math.min(el.cameraCanvas.width, (clientX - rect.left) * scaleX)),
      y: Math.max(0, Math.min(el.cameraCanvas.height, (clientY - rect.top) * scaleY)),
    };
  }

  function handleStart(e) {
    if (!state.currentImage) return;
    if (e.touches && e.touches.length > 1) return; // ignore multi-touch
    e.preventDefault();

    state.isDrawing = true;
    state.startPos = getCanvasCoords(e);
    state.currentPos = state.startPos;
    drawCanvas();
  }

  function handleMove(e) {
    if (!state.isDrawing) return;
    e.preventDefault();

    state.currentPos = getCanvasCoords(e);
    drawCanvas();
  }

  function handleEnd(e) {
    if (!state.isDrawing) return;
    e.preventDefault();
    state.isDrawing = false;

    const w_canvas = el.cameraCanvas.width;
    const h_canvas = el.cameraCanvas.height;

    const x_px = Math.min(state.startPos.x, state.currentPos.x);
    const y_px = Math.min(state.startPos.y, state.currentPos.y);
    const w_px = Math.abs(state.currentPos.x - state.startPos.x);
    const h_px = Math.abs(state.currentPos.y - state.startPos.y);

    // Minimum size threshold to ignore accidental taps
    if (w_px > 10 && h_px > 10 && w_canvas > 0 && h_canvas > 0) {
      let xNorm = Math.round((x_px / w_canvas) * 10000) / 10000;
      let yNorm = Math.round((y_px / h_canvas) * 10000) / 10000;
      let wNorm = Math.round((w_px / w_canvas) * 10000) / 10000;
      let hNorm = Math.round((h_px / h_canvas) * 10000) / 10000;

      // Safe clamp inside [0, 1]
      xNorm = Math.max(0, Math.min(xNorm, 0.99));
      yNorm = Math.max(0, Math.min(yNorm, 0.99));
      wNorm = Math.max(0.01, Math.min(wNorm, 1.0 - xNorm));
      hNorm = Math.max(0.01, Math.min(hNorm, 1.0 - yNorm));

      const roiObj = { x: xNorm, y: yNorm, w: wNorm, h: hNorm };

      if (state.currentDrawingMode === "co") {
        state.roiCo = roiObj;
        showToast("Zaznaczono obszar CO. Kliknij 'Zapisz ramki ROI', aby zachować.", "info");
      } else {
        state.roiCwu = roiObj;
        showToast("Zaznaczono obszar CWU. Kliknij 'Zapisz ramki ROI', aby zachować.", "info");
      }

      updateCoordsDisplay();
    }

    drawCanvas();
  }

  // -------------------------------------------------------------------------
  // Event Listeners Setup
  // -------------------------------------------------------------------------

  function setupEventListeners() {
    // Canvas pointer events
    el.cameraCanvas.addEventListener("mousedown", handleStart);
    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleEnd);

    el.cameraCanvas.addEventListener("touchstart", handleStart, { passive: false });
    window.addEventListener("touchmove", handleMove, { passive: false });
    window.addEventListener("touchend", handleEnd, { passive: false });

    // Mode selector radio buttons
    el.roiModeInputs.forEach((input) => {
      input.addEventListener("change", (e) => {
        state.currentDrawingMode = e.target.value;
      });
    });

    // Refresh Preview button
    el.btnRefreshPreview.addEventListener("click", () => {
      refreshCameraPreview();
    });

    // Clear ROIs button
    el.btnClearRois.addEventListener("click", () => {
      if (confirm("Czy na pewno chcesz wyczyścić zaznaczone ramki CO i CWU?")) {
        state.roiCo = null;
        state.roiCwu = null;
        updateCoordsDisplay();
        drawCanvas();
        showToast("Ramki wyczyszczone w podglądzie. Zapisz, aby utrwalić.", "info");
      }
    });

    // Save ROIs button
    el.btnSaveRois.addEventListener("click", async () => {
      try {
        if (!state.settings) await loadSettings();
        const payload = {
          ...state.settings,
          roi_co: state.roiCo,
          roi_cwu: state.roiCwu,
        };
        const updated = await apiPut("api/settings", payload);
        state.settings = updated;
        showToast("Obszary ROI zostały pomyślnie zapisane!", "success");
      } catch (err) {
        showToast(`Błąd zapisu ROI: ${err.message}`, "error");
      }
    });

    // Refresh cameras button
    el.btnRefreshCameras.addEventListener("click", () => {
      loadCameras();
      showToast("Lista kamer odświeżona.", "info");
    });

    // Camera select change -> auto update preview
    el.cameraSelect.addEventListener("change", () => {
      if (el.cameraSelect.value) {
        refreshCameraPreview();
      }
    });

    // Settings Form submit
    el.settingsForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      try {
        const payload = {
          camera_entity: el.cameraSelect.value || null,
          scan_interval_seconds: parseInt(el.scanInterval.value, 10),
          confidence_threshold: parseFloat(el.confidenceThreshold.value),
          min_temp: parseInt(el.minTemp.value, 10),
          max_temp: parseInt(el.maxTemp.value, 10),
          max_delta: parseInt(el.maxDelta.value, 10),
          sensor_co_entity: el.sensorCo.value.trim(),
          sensor_cwu_entity: el.sensorCwu.value.trim(),
          collector_enabled: el.collectorEnabledSwitch.checked,
          roi_co: state.roiCo,
          roi_cwu: state.roiCwu,
        };

        const updated = await apiPut("api/settings", payload);
        state.settings = updated;
        showToast("Konfiguracja zapisana pomyślnie!", "success");
      } catch (err) {
        showToast(`Błąd zapisu konfiguracji: ${err.message}`, "error");
      }
    });

    // Collector toggle switch
    el.collectorEnabledSwitch.addEventListener("change", async () => {
      if (!state.settings) return;
      try {
        const payload = {
          ...state.settings,
          collector_enabled: el.collectorEnabledSwitch.checked,
        };
        state.settings = await apiPut("api/settings", payload);
        updateCollectorBadge(state.settings.collector_enabled);
        showToast(
          state.settings.collector_enabled ? "Zbieranie próbek w tle włączone." : "Zbieranie próbek w tle wyłączone.",
          "info"
        );
      } catch (err) {
        el.collectorEnabledSwitch.checked = !el.collectorEnabledSwitch.checked;
        showToast(`Błąd przełączania collectora: ${err.message}`, "error");
      }
    });

    // Download snapshots (strictly relative URLs)
    el.btnDownloadFull.addEventListener("click", () => {
      window.location.href = "api/snapshot/download?crop=full";
    });

    el.btnDownloadCo.addEventListener("click", () => {
      if (!state.roiCo) {
        showToast("Najpierw narysuj i zapisz ramkę CO!", "error");
        return;
      }
      window.location.href = "api/snapshot/download?crop=co";
    });

    el.btnDownloadCwu.addEventListener("click", () => {
      if (!state.roiCwu) {
        showToast("Najpierw narysuj i zapisz ramkę CWU!", "error");
        return;
      }
      window.location.href = "api/snapshot/download?crop=cwu";
    });

    // Capture Now button
    el.btnCaptureNow.addEventListener("click", async () => {
      try {
        el.btnCaptureNow.disabled = true;
        el.btnCaptureNow.textContent = "Zapisywanie...";
        const updatedStats = await apiPost("api/collector/capture-now");
        loadCollectorStats();
        showToast(`Zapisano próbkę do bufora (łącznie ${updatedStats.count} plików).`, "success");
      } catch (err) {
        showToast(`Błąd zapisu próbki: ${err.message}`, "error");
      } finally {
        el.btnCaptureNow.disabled = false;
        el.btnCaptureNow.textContent = "📸 Złap próbkę teraz";
      }
    });

    // Export samples ZIP (strictly relative URL)
    el.btnExportZip.addEventListener("click", () => {
      window.location.href = "api/samples/export";
      showToast("Pobieranie archiwum ZIP ze zbiorem danych...", "info");
    });

    // Clear samples button
    el.btnClearSamples.addEventListener("click", async () => {
      if (confirm("Czy na pewno chcesz usunąć wszystkie zebrane próbki treningowe z /data/samples?")) {
        try {
          const res = await apiDelete("api/samples/clear");
          loadCollectorStats();
          showToast(`Wyczyszczono katalog próbek (usunięto ${res.deleted_count} plików).`, "info");
        } catch (err) {
          showToast(`Błąd czyszczenia próbek: ${err.message}`, "error");
        }
      }
    });

    // Model upload
    el.btnUploadModel.addEventListener("click", async () => {
      const file = el.modelFileInput.files[0];
      if (!file) {
        showToast("Proszę wybrać plik z rozszerzeniem .tflite!", "error");
        return;
      }

      if (!file.name.toLowerCase().endsWith(".tflite")) {
        showToast("Wybrany plik nie ma rozszerzenia .tflite!", "error");
        return;
      }

      if (file.size > 25 * 1024 * 1024) {
        showToast("Plik przekracza maksymalny rozmiar 25 MB!", "error");
        return;
      }

      const formData = new FormData();
      formData.append("file", file);

      try {
        el.btnUploadModel.disabled = true;
        el.btnUploadModel.textContent = "Wgrywanie...";
        const newStatus = await apiPost("api/model/upload", formData);
        state.modelStatus = newStatus;
        loadModelStatus();
        showToast("Model TFLite został pomyślnie wgrany i zainicjalizowany!", "success");
        el.modelFileInput.value = "";
      } catch (err) {
        showToast(`Błąd wgrywania modelu: ${err.message}`, "error");
      } finally {
        el.btnUploadModel.disabled = false;
        el.btnUploadModel.textContent = "⬆️ Wgraj model .tflite";
      }
    });

    // Test Inference button – Card 5
    el.btnTestInference.addEventListener("click", async () => {
      try {
        el.btnTestInference.disabled = true;
        el.btnTestInference.textContent = "⏳ Wykonywanie odczytu...";
        const result = await apiPost("api/inference/test");

        if (result.status === "ok") {
          updateDiagnosticChannel("co", result.co);
          updateDiagnosticChannel("cwu", result.cwu);
          if (result.timestamp) {
            el.diagLastTimestamp.textContent = new Date(result.timestamp).toLocaleString();
          }
          el.diagLastError.textContent = "";
          showToast("Wykonano testowy odczyt inferencji.", "success");
        } else if (result.status === "no_camera") {
          showToast(`Brak kamery: ${result.message}`, "warning");
        } else {
          showToast(`Odczyt zakończony statusem: ${result.status}`, "warning");
        }
        await loadSystemStatus();
      } catch (err) {
        showToast(`Błąd testowego odczytu: ${err.message}`, "error");
      } finally {
        el.btnTestInference.disabled = false;
        el.btnTestInference.textContent = "🧪 Wymuś testowy odczyt teraz";
      }
    });
  }

  // -------------------------------------------------------------------------
  // Application Bootstrap
  // -------------------------------------------------------------------------

  async function init() {
    setupEventListeners();

    // Initial sequence
    await loadSettings();
    await loadCameras();
    await loadModelStatus();
    await loadCollectorStats();
    await loadSystemStatus();

    // Fetch initial camera preview
    refreshCameraPreview();

    // Recurring state polling every 10 seconds
    setInterval(() => {
      loadSystemStatus();
      loadCollectorStats();
      loadModelStatus();
    }, 10000);
  }

  window.addEventListener("DOMContentLoaded", init);
})();
