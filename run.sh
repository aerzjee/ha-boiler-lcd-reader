#!/usr/bin/with-contenv bashio
# ==============================================================================
# Boiler LCD Reader – Entrypoint
# ==============================================================================

# Read log level from add-on options
LOG_LEVEL=$(bashio::config 'log_level')
export LOG_LEVEL

bashio::log.info "Starting Boiler LCD Reader add-on (log_level=${LOG_LEVEL})..."

# Ensure persistent data directories exist
mkdir -p /data/models
mkdir -p /data/samples

# Launch the FastAPI application via Uvicorn
exec python3 -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8099 \
    --log-level "${LOG_LEVEL}"
