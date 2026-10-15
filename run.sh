#!/usr/bin/env bash
set -e

# ==============================================================================
# Boiler LCD Reader – Entrypoint
# ==============================================================================

# Read log level from /data/options.json (fallback to info)
LOG_LEVEL=$(python3 -c "import json; print(json.load(open('/data/options.json')).get('log_level', 'info'))" 2>/dev/null || echo "info")
export LOG_LEVEL
export PYTHONPATH="/"

echo "Starting Boiler LCD Reader add-on (log_level=${LOG_LEVEL})..."

# Ensure persistent data directories exist
mkdir -p /data/models
mkdir -p /data/samples

# Launch the FastAPI application via Uvicorn
exec python3 -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8099 \
    --log-level "${LOG_LEVEL}"
