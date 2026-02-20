#!/usr/bin/env bash
#
# Full training pipeline: V2 → V3 → V4 with weight transfer.
# Each stage runs to convergence (early stopping), then checks if the
# best validation loss meets the gate threshold before advancing.
#
# Gate rationale (20-30 synthetic pages, 534 classes, label smoothing 0.1):
#   - V2: Classification-dominated loss. With ~18K samples and 534 classes,
#     converges around 3.5-3.8. Gate at 4.0 = "backbone is learning features."
#   - V3: Sequence model. Same class bottleneck but better spatial reasoning.
#     Typically converges 0.2-0.5 below V2. Gate at 4.0 = "transformer is learning."
#   - V4: 4 loss components (point, orient, class, intersect). Total is higher.
#     Gate at 4.5 = "contour prediction is working."
#
# Usage:
#   ./run_pipeline.sh                    # run full pipeline
#   SKIP_V2=1 ./run_pipeline.sh          # skip V2 if model_02.pth already exists
#   V2_GATE=3.5 ./run_pipeline.sh        # override gate threshold
#
set -euo pipefail

cd "$(dirname "$0")"

PYTHON=".venv/bin/python"
LOGDIR="pipeline_logs"
mkdir -p "$LOGDIR"

# ---------- Gate thresholds ----------
V2_GATE=${V2_GATE:-4.0}
V3_GATE=${V3_GATE:-4.0}
V4_GATE=${V4_GATE:-4.5}
SKIP_V2=${SKIP_V2:-0}

# ---------- Helpers ----------
extract_best_val_loss() {
    # Grab the val loss from the last "saved best model" epoch.
    # Handles both "val loss=X" (V2/V3) and "val=X" (V4) log formats.
    grep 'val' "$1" | grep -E 'val[= ]*loss=|val=' | tail -1 | \
        sed -E 's/.*val[= ]*loss=([0-9.]+).*/\1/; t; s/.*val=([0-9.]+).*/\1/'
}

check_gate() {
    local stage="$1" best="$2" gate="$3"
    echo ""
    echo "=== Gate check: $stage ==="
    echo "  Best val loss: $best"
    echo "  Threshold:     < $gate"
    if awk "BEGIN {exit !($best < $gate)}"; then
        echo "  PASSED — advancing to next stage."
        return 0
    else
        echo "  FAILED — best val loss $best >= $gate. Stopping pipeline."
        return 1
    fi
}

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

echo "========================================"
echo " tiny-tessarachnid training pipeline"
echo " $(timestamp)"
echo " Gates: V2<$V2_GATE  V3<$V3_GATE  V4<$V4_GATE"
echo "========================================"
echo ""

# ================================================================
# Stage 1: V2 — RetinaOCRNet
#
# NOTE: Must use resnet18 backbone to match V3/V4 (both hardcode
# resnet18). Using resnet50 would make weight transfer a no-op.
# ================================================================
if [ "$SKIP_V2" = "1" ] && [ -f model_02.pth ]; then
    echo ">>> Stage 1: SKIPPED (SKIP_V2=1, model_02.pth exists)"
else
    V2_LOG="$LOGDIR/train_02_$(date +%Y%m%d_%H%M%S).log"
    echo ">>> Stage 1: Training V2 (RetinaOCRNet / resnet18) — log: $V2_LOG"
    echo "    $(timestamp) — starting"

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
        --save-path model_02.pth \
        --deploy \
        2>&1 | tee "$V2_LOG"

    V2_BEST=$(extract_best_val_loss "$V2_LOG")
    echo "    $(timestamp) — finished"
    check_gate "V2 (RetinaOCRNet)" "$V2_BEST" "$V2_GATE"
fi

# ================================================================
# Stage 2: V3 — RetinaOCRGPT (backbone from V2)
# ================================================================
V3_LOG="$LOGDIR/train_03_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo ">>> Stage 2: Training V3 (RetinaOCRGPT) — log: $V3_LOG"
echo "    Transferring backbone from model_02.pth"
echo "    $(timestamp) — starting"

$PYTHON train_03.py \
    --epochs 50 \
    --batch-size 4 \
    --lr 3e-4 \
    --pages 30 \
    --pretrained model_02.pth \
    --freeze-backbone-epochs 5 \
    --grad-clip 1.0 \
    --patience 15 \
    --save-path model_03.pth \
    --deploy \
    2>&1 | tee "$V3_LOG"

V3_BEST=$(extract_best_val_loss "$V3_LOG")
echo "    $(timestamp) — finished"
check_gate "V3 (RetinaOCRGPT)" "$V3_BEST" "$V3_GATE"

# ================================================================
# Stage 3: V4 — ContourOCRNet (full weights from V3)
# ================================================================
V4_LOG="$LOGDIR/train_04_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo ">>> Stage 3: Training V4 (ContourOCRNet) — log: $V4_LOG"
echo "    Transferring weights from model_03.pth"
echo "    $(timestamp) — starting"

$PYTHON train_04.py \
    --epochs 50 \
    --batch-size 4 \
    --lr 3e-4 \
    --pages 30 \
    --pretrained-v03 model_03.pth \
    --freeze-backbone-epochs 5 \
    --grad-clip 1.0 \
    --patience 15 \
    --rotate \
    --save-path model_04.pth \
    --deploy \
    2>&1 | tee "$V4_LOG"

V4_BEST=$(extract_best_val_loss "$V4_LOG")
echo "    $(timestamp) — finished"
check_gate "V4 (ContourOCRNet)" "$V4_BEST" "$V4_GATE"

# ================================================================
# Summary
# ================================================================
echo ""
echo "========================================"
echo " Pipeline complete — $(timestamp)"
echo "========================================"
echo " V2 (RetinaOCRNet):  best val_loss = ${V2_BEST:-skipped}  (gate < $V2_GATE)"
echo " V3 (RetinaOCRGPT):  best val_loss = $V3_BEST  (gate < $V3_GATE)"
echo " V4 (ContourOCRNet): best val_loss = $V4_BEST  (gate < $V4_GATE)"
echo ""
echo " Models: model_02.pth → model_03.pth → model_04.pth"
echo "========================================"
