# ContourOCRNet: Autoregressive Contour Detection with Orientation

## Overview

ContourOCRNet is an autoregressive contour-prediction model for hierarchical
document OCR. Instead of outputting axis-aligned bounding boxes, the model emits
a sequence of 2D contour points plus a unit orientation (tangent) vector at each
step. A contour is considered closed when the newly predicted point falls within
a threshold distance of the start point. The model operates recursively across
5 hierarchy levels, with each level receiving a cropped, rotated, and
scale-padded image of the parent region.

## Versioning

The model architecture and training script are `train_04.py` / `ContourOCRNet`.
The trained weights file is `model_04.pth`. There is no separate "v5" — the
addition of the PAGE hierarchy level is a retraining of the same architecture
(with `level_emb` expanded from 4 to 5 entries and class IDs shifted). Older
weights are incompatible and must be retrained from scratch, but the model
version remains **v4** (`train_04` / `model_04`).

## 5-Level Hierarchy

```
Level 0  PAGE        Detects the text-bearing region on the full input image.
Level 1  PARAGRAPH   Detects paragraph regions within the page crop.
Level 2  LINE        Detects text lines within a paragraph crop.
Level 3  WORD        Detects words within a line crop.
Level 4  CHARACTER   Detects individual characters within a word crop.
```

A single model handles all five levels. The current hierarchy level is
communicated via a learned **level embedding** that is added to every token
representation in the transformer.

### Class tokens (~535 total)

```
0           NONE (nothing left to detect / terminator)
1           PAGE
2           PARAGRAPH
3           LINE
4           WORD
5 .. 534    Characters (ASCII, Latin-1 Supp, Latin Ext-A, Greek, math, arrows, …)
```

`CHAR_CLASS_OFFSET = 5`. The character list is built deterministically by
`create_lists.py` (~530 chars, ASCII first for backward compat).

## Architecture

### Image encoder

- **ResNet-18** backbone (ImageNet-pretrained).
- Global average pooling → 512-dim feature → linear projection to `d_model`.
- Image is preprocessed to a fixed 1024×1024 "retina" via `scale_and_pad`
  (aspect-preserving scale then center-pad).
- ImageNet normalization (mean/std buffers registered on the model).

### Transformer decoder (GPT-2 style)

| Hyperparameter | Default |
|----------------|---------|
| `d_model`      | 128     |
| `n_heads`      | 4       |
| `n_layers`     | 4       |
| `d_ff`         | 512     |
| `max_seq_len`  | 256     |
| `dropout`      | 0.1     |

Pre-norm transformer blocks with causal (upper-triangular) attention mask.

### Input conditioning

Each sequence position receives the sum of five embeddings:

1. **Detection embedding** — linear projection of the previous token
   `(x, y, dx, dy, class_id)`, with `x, y` normalized by `RETINA_SIZE` and
   `class_id` normalized by `NUM_CLASSES`.
2. **Image embedding** — global image feature, broadcast to all positions.
3. **Start-point embedding** — linear projection of the first contour point
   `(x, y)`, broadcast to all positions. Gives the model closure context.
4. **Positional embedding** — learned, up to `max_seq_len`.
5. **Level embedding** — learned, 5 entries (one per hierarchy level).

### Output heads

| Head                | Shape        | Activation             | Purpose                     |
|---------------------|-------------|------------------------|-----------------------------|
| `point_head`        | `(B, S, 2)` | Sigmoid × RETINA_SIZE  | Next contour point (x, y)   |
| `orientation_head`  | `(B, S, 2)` | L2-normalize           | Unit tangent vector (dx, dy)|
| `class_head`        | `(B, S, C)` | Raw logits             | Class token                 |

### Contour closure (inference)

After predicting point `p_t`, compute `‖p_t − p_0‖`. If this distance falls
below a **closure threshold** (default 15 px in retina space) *and* at least 3
points have been emitted, the contour is closed and the model moves to the next
element. When the model predicts `CLASS_NONE`, all elements at this level have
been enumerated.

## Training Data

### Synthetic page generation (`SyntheticPage`)

Each page is an RGB image (default 2048×2800) with:

- Random background color.
- 1–6 paragraphs, each rendered with a randomly chosen font and contrasting text
  color. Paragraph rotation (±30°) is enabled by default.
- Hierarchical ground-truth structure: page → paragraphs → lines → words →
  characters, each with a bounding box, pixel mask, and simplified contour
  polygon.

**Page-level contour**: the tight bounding contour around all paragraph content
(convex hull of the union of paragraph masks).

**Contour extraction** (`contour_from_mask`):

1. Find non-zero pixels in the binary mask.
2. Compute the convex hull (Andrew's monotone chain).
3. Simplify with Ramer-Douglas-Peucker to 4–20 points.
4. Ensure clockwise winding order.

**Orientation vectors** (`contour_orientations`): central-difference unit
tangents at each contour vertex (forward/backward difference at endpoints).

### Dataset (`ContourSequenceDataset`)

Each sample is a full contour sequence for one hierarchy level of one parent
region. The sequence is teacher-forced:

```
input:   [NONE,  pt0,  pt1,  ..., ptN ]
target:  [pt0,   pt1,  pt2,  ..., NONE]
```

Each token is `(x, y, dx, dy, class_id)` in retina coordinates.

| Level | Image fed to model | Contours detected | `level_id` |
|-------|-------------------|-------------------|------------|
| Page  | Full page image   | Page region(s)    | 0          |
| Para  | Full page image   | Paragraphs        | 1          |
| Line  | Paragraph crop    | Lines             | 2          |
| Word  | Line crop         | Words             | 3          |
| Char  | Word crop         | Characters        | 4          |

Sub-level crops are extracted from the page image by cropping the parent's
bounding box. At training time these are axis-aligned crops; at inference time
`rotate_and_crop` rotates the detected contour upright first (see below).

The collate function pads sequences to the batch maximum length and produces a
boolean padding mask.

## Loss Function (`ContourLoss`)

Four terms, weighted and summed:

### 1. Point regression (`point_weight = 1.0`)

SmoothL1 loss between predicted and GT contour points, both normalized to
[0, 1] by dividing by `RETINA_SIZE`. Computed only at non-NONE positions.

### 2. Orientation loss (`orient_weight = 0.5`)

`1 − cosine_similarity(pred, gt)` averaged over non-NONE positions.

### 3. Classification loss (`class_weight = 1.0`)

Per-position cross-entropy over `NUM_CLASSES`, averaged over all valid (non-pad)
positions.

### 4. Intersection loss (`intersect_weight = 0.1`)

Soft differentiable penalty for self-intersecting contour edges. For every pair
of non-adjacent segments `(p_i, p_{i+1})` and `(p_j, p_{j+1})`, a sigmoid of
the negative product of signed cross-product areas gives a [0, 1] intersection
score. Averaged over all checked pairs.

## Training Procedure

### Phases

1. **Backbone freeze** (default 5 epochs): ResNet-18 layers are frozen; only
   projection heads, embeddings, and transformer blocks train.
2. **Full fine-tune** (remaining epochs): all parameters are unfrozen.

### Optimizer

AdamW with differential learning rates:

| Parameter group         | LR multiplier |
|------------------------|---------------|
| Backbone (ResNet)       | 0.01×         |
| Projections & embeddings| 0.5×          |
| Transformer & heads     | 1.0×          |

Base LR default: `3e-4`. Weight decay: `1e-2`.

### Scheduler

`ReduceLROnPlateau` (factor 0.5, patience 5) on validation loss.

### Early stopping

Patience 15 epochs on validation loss. Best model saved to disk.

### Gradient clipping

Global norm clip at 1.0.

### Typical run

```bash
python train_04.py \
    --pages 200 --epochs 50 --batch-size 4 \
    --save-path model_04.pth --rotate --seed 42
```

## Inference: Hierarchical Contour Tracing

### Preprocessing: `scale_and_pad`

Every image (full page or sub-region crop) is scaled to fit 1024×1024 preserving
aspect ratio, then centered on a canvas padded with the estimated background
color. Returns `(retina_image, scale, offset_x, offset_y)`.

### Preprocessing: `rotate_and_crop`

At inference time, when descending from one hierarchy level to the next, the
parent contour is rotated upright before cropping:

1. Compute the mean orientation angle from the contour's `(dx, dy)` vectors.
2. Rotate all contour points by `−angle` around the centroid → axis-aligned.
3. Compute the bounding box of the rotated contour.
4. Rotate the source image by `−angle` around the centroid (no canvas expansion).
5. Crop the bounding box from the rotated image.
6. The crop is then passed to `_generate_contours`, which applies its own
   `scale_and_pad`.

This ensures that tilted text regions are presented to the model upright,
matching the training distribution.

### Coordinate reverse-mapping

To map detected contour points back to absolute page coordinates, the reverse
transform is composed recursively at each level:

1. **Unscale + unpad**: `source = (retina − offset) / scale`.
2. **Unrotate + uncrop**: translate by the crop origin, then rotate by `+angle`
   around the contour centroid.

For the streaming pipeline, the `_unmap_contour` function is applied once per
ancestor level. A word's absolute bbox is obtained by composing three
un-rotations: word-crop → line-crop → para-crop → page-crop → page.

### Inference levels

```
Level 0  detect page region(s) on the full image
  Level 1  detect paragraphs within rotated page crop
    Level 2  detect lines within rotated paragraph crop
      Level 3  detect words within rotated line crop
        Level 4  detect characters within rotated word crop
```

## Files

| File | Role |
|------|------|
| `generate_training_data.py` | Synthetic page generation, contour extraction, datasets |
| `train_04.py` | Training script (model, dataset, loss, training loop) |
| `infer_04.py` | Standalone 5-level hierarchical inference |
| `create_lists.py` | Deterministic character list builder |
| `annotate_real.py` | Real-document annotation tools |
| `DESIGN_V2.md` | This document |
