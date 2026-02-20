#!/usr/bin/env bash
# Ratchet pipeline with continuous weight deployment to glyph-daemon.
#
# 1. Starts a background watcher that encrypts + deploys model_04.pth
#    whenever it changes (polls every 60s).
# 2. Waits for the current training run to finish.
# 3. Runs a ratchet (lower LR, more data, cosine schedule).
# 4. Stops the watcher when done.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=".venv/bin/python"
LOGDIR="pipeline_logs"
DAEMON_DIR="../glyph-deamon"

source "$DAEMON_DIR/.env"

# ---------------------------------------------------------------------------
# Weight deploy function
# ---------------------------------------------------------------------------
deploy_weights() {
    $PYTHON -c "
import sys; sys.path.insert(0, '$DAEMON_DIR')
from pathlib import Path
from glyph_daemon.vault import encrypt_file
encrypt_file(Path('model_04.pth'), Path('$DAEMON_DIR/weights/model_04.enc'), '$GLYPH_VAULT_KEY')
" 2>/dev/null && echo "$(date '+%H:%M:%S') Deployed model_04.pth -> glyph-daemon"
}

# ---------------------------------------------------------------------------
# Background watcher: deploy whenever checkpoint changes
# ---------------------------------------------------------------------------
LAST_HASH=""
watch_and_deploy() {
    while true; do
        if [ -f model_04.pth ]; then
            CURR_HASH=$(md5sum model_04.pth | cut -d' ' -f1)
            if [ "$CURR_HASH" != "$LAST_HASH" ]; then
                deploy_weights
                LAST_HASH="$CURR_HASH"
            fi
        fi
        sleep 60
    done
}

echo "Starting weight deployment watcher (every 60s)..."
watch_and_deploy &
WATCHER_PID=$!
trap "kill $WATCHER_PID 2>/dev/null; exit" EXIT INT TERM

# Deploy current weights immediately
if [ -f model_04.pth ]; then
    deploy_weights
fi

# ---------------------------------------------------------------------------
# Wait for current training run to finish
# ---------------------------------------------------------------------------
echo "Waiting for current train_04 run to finish..."
while pgrep -f "train_04.py.*--save-path model_04.pth" > /dev/null 2>&1; do
    sleep 30
done
echo "Current run finished."
sleep 5

# Final deploy of best weights from this run
deploy_weights

# ---------------------------------------------------------------------------
# Ratchet run
# ---------------------------------------------------------------------------
RATCHET_LOG="$LOGDIR/train_04_ratchet_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo ">>> Ratchet: fine-tuning from model_04.pth — log: $RATCHET_LOG"

$PYTHON train_04.py \
    --epochs 80 \
    --batch-size 4 \
    --lr 3e-5 \
    --pages 80 \
    --pretrained-v03 model_04.pth \
    --freeze-backbone-epochs 0 \
    --grad-clip 1.0 \
    --patience 25 \
    --warmup-epochs 3 \
    --rotate \
    --real-data real_annotations/ \
    --save-path model_04.pth \
    2>&1 | tee "$RATCHET_LOG"

# Final deploy
deploy_weights
echo "Ratchet complete. Weights deployed."
