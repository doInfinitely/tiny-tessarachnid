# Training Pipeline

## Quick Start

```bash
# Full pipeline (V2 → V3 → V4)
./run_pipeline.sh

# Resume V4 only (skip V2/V3, warm-start from prior V4 weights)
SKIP_V2=1 SKIP_V3=1 WARM_START=1 bash run_pipeline.sh

# Resume V4 with backbone-only transfer (reset head weights)
SKIP_V2=1 SKIP_V3=1 BACKBONE_ONLY=1 bash run_pipeline.sh
```

## Weight Transfer Strategy

| Source → Target | Flag | What transfers |
|---|---|---|
| V2 → V3 | `--pretrained model_02.pth` | ResNet-18 backbone only |
| V3 → V4 | `--pretrained-v03 model_03.pth` | All compatible weights (backbone + shared layers) |
| V4 → V4 (warm start) | `--pretrained-v03 model_04.pth` | All compatible weights (full model) |
| V4 → V4 (backbone only) | `--pretrained model_04.pth` | ResNet-18 backbone only |

**Rule**: When resuming from a checkpoint that differs in model number (e.g. V3→V4),
keep only the ResNet-18 backbone (`--pretrained`). When resuming from the same model
number (V4→V4), transfer all compatible weights (`--pretrained-v03` / `WARM_START=1`).

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SKIP_V2` | `0` | Skip V2 stage if `model_02.pth` exists |
| `SKIP_V3` | `0` | Skip V3 stage if `model_03.pth` exists |
| `WARM_START` | `1` | V4 transfers all weights from prior `model_04.pth` |
| `BACKBONE_ONLY` | `0` | V4 retains only ResNet-18 backbone from `model_04.pth` |
| `V2_GATE` | `4.0` | Val loss threshold to advance past V2 |
| `V3_GATE` | `4.0` | Val loss threshold to advance past V3 |
| `V4_GATE` | `4.5` | Val loss threshold to advance past V4 |

## Encrypted Weight Deployment

The `--deploy` flag (enabled by default in `run_pipeline.sh`) encrypts the best
checkpoint with Fernet (AES-256-CBC) after each improvement and copies it to
the glyph-daemon weights directory:

```
model_04.pth → deploy_weights.py → glyph-deamon/weights/model_04.enc
```

Encryption key: `GLYPH_VAULT_KEY` from `.env`.

After training completes, the pipeline automatically restarts glyph-daemon
to pick up the new weights.

## Real Data

V4 mixes synthetic and real annotated data:

- `--real-data real_annotations` — directory of annotated real documents
- `--real-ratio 0.3` — 30% of training batches are real data

Real annotations are produced by `stream_to_winnie.py` (GPT-4o cascade annotation).

## Gate Checks

Each stage must achieve `val_loss < gate_threshold` to advance. If the gate
fails, the pipeline stops. Override thresholds via environment variables.
