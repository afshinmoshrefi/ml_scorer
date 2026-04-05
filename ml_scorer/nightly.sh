#!/bin/bash
# Nightly ML Scorer maintenance
# Generates parquets, restarts service, warms cache
# Cron: 0 1 * * * root /home/flask/ml_scorer/nightly.sh
set -euo pipefail

cd /home/flask/ml_scorer
export ML_SCORER_DATA_DIR=/home/flask/data

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*"; }

# Step 1: Generate parquets for this week
log "Step 1: Generating parquets"
if ! python3.12 opp_to_parquet.py >> opp_to_parquet.log 2>&1; then
    log "ERROR: opp_to_parquet.py failed -- aborting nightly run to avoid stale cache"
    exit 1
fi

# Step 2: Restart scorer service
log "Step 2: Restarting ml_scorer service"
if ! systemctl restart ml_scorer; then
    log "ERROR: systemctl restart ml_scorer failed"
    exit 1
fi

# Step 3: Wait for service to be ready (up to 30 seconds)
log "Step 3: Waiting for service socket"
ready=0
for i in {1..30}; do
    [ -S ml_scorer.sock ] && ready=1 && break
    sleep 1
done
if [ $ready -eq 0 ]; then
    log "ERROR: ml_scorer socket not ready after 30s -- aborting warmup"
    exit 1
fi

# Step 4: Warm cache for all symbols
log "Step 4: Warming cache"
if ! python3.12 warmup_cache.py >> warmup_cache.log 2>&1; then
    log "ERROR: warmup_cache.py failed"
    exit 1
fi

log "Nightly run complete"
