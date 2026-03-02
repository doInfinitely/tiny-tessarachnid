#!/usr/bin/env bash
# Iterative training ratchet: full V2 → V3 → V4 pipeline per round.
#
# Uses GPT-4o annotated data from real_annotations/.
#
# Each round:
#   1. Train V2 (resume backbone) — synthetic only
#   2. Train V3 (backbone from V2) — synthetic + real data
#   3. Train V4 (weights from V3) — synthetic + real data
#   4. Deploy weights to Modal
#   5. Repeat with fresh synthetic data each round
#
# The ResNet-18 backbone persists across the entire process:
#   V2 (resume) → V3 (backbone transfer) → V4 (full weight transfer)
#
# Usage:
#   ./run_ratchet.sh            # 3 rounds (default)
#   ./run_ratchet.sh 5          # 5 rounds
#   SKIP_V2=1 ./run_ratchet.sh  # skip V2 stage (use existing model_02.pth)
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=".venv/bin/python"
LOGDIR="pipeline_logs"
DAEMON_DIR="../glyph-deamon"
MODEL_V2="model_02.pth"
MODEL_V3="model_03.pth"
MODEL_V4="model_04.pth"
REAL_DIR="real_annotations"
NUM_ROUNDS="${1:-3}"
SKIP_V2="${SKIP_V2:-0}"

source "$DAEMON_DIR/.env"
source .env
export PYTHONUNBUFFERED=1
mkdir -p "$LOGDIR"

REAL_COUNT=$(find "$REAL_DIR" -name annotations.json 2>/dev/null | wc -l)
echo "Real annotations: $REAL_COUNT documents"

# ---------------------------------------------------------------------------
# Weight deploy function (Modal volume)
# ---------------------------------------------------------------------------
MODAL_BIN="$DAEMON_DIR/.venv/bin/modal"
MODAL_VOLUME="glyph-weights"

deploy_weights() {
    local src="${1:-$MODEL_V4}"
    local dst="${2:-model_04.enc}"
    local enc_tmp
    enc_tmp=$(mktemp /tmp/glyph_deploy_XXXXXX.enc)

    # Encrypt
    $PYTHON -c "
import sys; sys.path.insert(0, '$DAEMON_DIR')
from pathlib import Path
from glyph_daemon.vault import encrypt_file
encrypt_file(Path('$src'), Path('$enc_tmp'), '$GLYPH_VAULT_KEY')
" 2>/dev/null || { rm -f "$enc_tmp"; return 1; }

    # Deploy to Modal volume (GlyphWorker)
    if [ -x "$MODAL_BIN" ]; then
        "$MODAL_BIN" volume put --force "$MODAL_VOLUME" "$enc_tmp" "$dst" 2>/dev/null \
            && echo "$(date '+%H:%M:%S') Deployed $src -> modal:$MODAL_VOLUME/$dst"
    fi

    rm -f "$enc_tmp"
}

# ---------------------------------------------------------------------------
# Background watcher: deploy whenever any model checkpoint changes
# ---------------------------------------------------------------------------
LAST_HASH_V2=""
LAST_HASH_V3=""
LAST_HASH_V4=""
watch_and_deploy() {
    while true; do
        for model_var in MODEL_V2 MODEL_V3 MODEL_V4; do
            model_path="${!model_var}"
            enc_name="${model_path%.pth}.enc"
            hash_var="LAST_HASH_${model_var#MODEL_}"
            if [ -f "$model_path" ]; then
                CURR_HASH=$(md5sum "$model_path" | cut -d' ' -f1)
                if [ "$CURR_HASH" != "${!hash_var}" ]; then
                    deploy_weights "$model_path" "$enc_name"
                    eval "$hash_var='$CURR_HASH'"
                fi
            fi
        done
        sleep 60
    done
}

echo "Starting weight deployment watcher (every 60s)..."
watch_and_deploy &
WATCHER_PID=$!
trap "kill $WATCHER_PID 2>/dev/null; exit" EXIT INT TERM

for m in "$MODEL_V2" "$MODEL_V3" "$MODEL_V4"; do
    if [ -f "$m" ]; then
        deploy_weights "$m" "${m%.pth}.enc"
    fi
done

# ---------------------------------------------------------------------------
# Wait for current training run to finish
# ---------------------------------------------------------------------------
echo "Waiting for any running training to finish..."
while pgrep -f "train_0[234].py" > /dev/null 2>&1; do
    sleep 30
done
echo "No training processes running."
sleep 5

# Collect V3 real-dirs arguments (individual subdirectories)
collect_real_dirs() {
    local dirs=()
    if [ -d "$REAL_DIR" ]; then
        for d in "$REAL_DIR"/*/; do
            [ -f "$d/annotations.json" ] && dirs+=("$d")
        done
    fi
    echo "${dirs[@]}"
}

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

# ---------------------------------------------------------------------------
# Iterative ratchet loop
# ---------------------------------------------------------------------------
for ROUND in $(seq 1 "$NUM_ROUNDS"); do
    TS=$(date +%Y%m%d_%H%M%S)
    echo ""
    echo "============================================================"
    echo "  RATCHET ROUND $ROUND / $NUM_ROUNDS  ($(timestamp))"
    echo "============================================================"

    # ------------------------------------------------------------------
    # Step 1: Train V2 — backbone learns features (synthetic only)
    # ------------------------------------------------------------------
    if [ "$SKIP_V2" = "1" ] && [ -f "$MODEL_V2" ]; then
        echo ""
        echo ">>> Step 1: V2 SKIPPED (SKIP_V2=1, $MODEL_V2 exists)"
    else
        V2_LOG="$LOGDIR/ratchet_v2_r${ROUND}_${TS}.log"
        echo ""
        echo ">>> Step 1: Training V2 (RetinaOCRNet) — log: $V2_LOG"

        V2_RESUME_ARG=""
        if [ -f "$MODEL_V2" ]; then
            V2_RESUME_ARG="--resume $MODEL_V2"
            echo "    Resuming backbone from $MODEL_V2"
        fi

        torchrun --standalone --nproc_per_node=1 train_02.py \
            --epochs 50 \
            --batch-size 64 \
            --lr 2e-4 \
            --pages 20 \
            --backbone resnet18 \
            --amp \
            --clip-grad 1.0 \
            --label-smoothing 0.1 \
            --warmup-epochs 3 \
            --pretrain-epochs 0 \
            --patience 15 \
            --save-path "$MODEL_V2" \
            $V2_RESUME_ARG \
            2>&1 | tee "$V2_LOG"

        echo "    V2 done ($(timestamp))"
    fi

    # ------------------------------------------------------------------
    # Step 2: Train V3 — backbone from V2, + real data
    # ------------------------------------------------------------------
    V3_LOG="$LOGDIR/ratchet_v3_r${ROUND}_${TS}.log"
    echo ""
    echo ">>> Step 2: Training V3 (RetinaOCRGPT) — log: $V3_LOG"
    echo "    Backbone from $MODEL_V2"

    REAL_DIRS_ARGS=""
    REAL_DIRS_LIST=$(collect_real_dirs)
    if [ -n "$REAL_DIRS_LIST" ]; then
        REAL_DIRS_ARGS="--real-dirs $REAL_DIRS_LIST"
        echo "    Real data: $(echo "$REAL_DIRS_LIST" | wc -w) directories"
    fi

    $PYTHON train_03.py \
        --epochs 50 \
        --batch-size 128 \
        --lr 5e-4 \
        --pages 30 \
        --pretrained "$MODEL_V2" \
        --freeze-backbone-epochs 0 \
        --grad-clip 1.0 \
        --patience 15 \
        --save-path "$MODEL_V3" \
        $REAL_DIRS_ARGS \
        2>&1 | tee "$V3_LOG"

    echo "    V3 done ($(timestamp))"

    # ------------------------------------------------------------------
    # Step 3: Train V4 — full weights from V3, + real data
    # ------------------------------------------------------------------
    V4_LOG="$LOGDIR/ratchet_v4_r${ROUND}_${TS}.log"
    echo ""
    echo ">>> Step 3: Training V4 (ContourOCRNet) — log: $V4_LOG"
    echo "    Weights from $MODEL_V3"

    $PYTHON train_04.py \
        --epochs 50 \
        --batch-size 128 \
        --lr 5e-4 \
        --pages 30 \
        --pretrained-v03 "$MODEL_V3" \
        --freeze-backbone-epochs 0 \
        --grad-clip 1.0 \
        --patience 15 \
        --rotate \
        --real-data "$REAL_DIR" \
        --real-ratio 0.3 \
        --save-path "$MODEL_V4" \
        2>&1 | tee "$V4_LOG"

    deploy_weights "$MODEL_V4" "model_04.enc"
    echo "    V4 done ($(timestamp))"

    # Redeploy Modal app so GlyphWorker picks up new weights
    if [ -x "$MODAL_BIN" ]; then
        echo "    Redeploying Modal app..."
        (cd "$DAEMON_DIR" && "$MODAL_BIN" deploy glyph_daemon/modal_app.py 2>&1 | tail -5) \
            && echo "    Modal redeploy done ($(timestamp))" \
            || echo "    Modal redeploy failed (non-fatal)"
    fi

    echo ""
    echo "  Round $ROUND complete."
done

# ---------------------------------------------------------------------------
# Final deploy
# ---------------------------------------------------------------------------
deploy_weights "$MODEL_V4" "model_04.enc"

if [ -x "$MODAL_BIN" ]; then
    echo "Final Modal redeploy..."
    (cd "$DAEMON_DIR" && "$MODAL_BIN" deploy glyph_daemon/modal_app.py 2>&1 | tail -5) \
        && echo "Modal redeploy done ($(timestamp))" \
        || echo "Modal redeploy failed (non-fatal)"
fi

echo ""
echo "========================================"
echo " Ratchet complete — $NUM_ROUNDS rounds"
echo " $(timestamp)"
echo "========================================"
echo " $MODEL_V2 → $MODEL_V3 → $MODEL_V4"
echo " Backbone persisted through all stages."
echo " Real data: $REAL_COUNT documents"
echo "========================================"
