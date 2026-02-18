# V2 Model Design: Autoregressive Contour Detection with Orientation

## Overview

V2 replaces bounding-box regression with **autoregressive contour prediction**.
Instead of outputting `(x1, y1, x2, y2)` per detection, the model emits a
sequence of 2D contour points plus an orientation vector at each step. The
contour is considered closed when a newly predicted point falls within a
threshold distance of the first point in the sequence.

## Architecture: `ContourOCRNet`

### Backbone (unchanged)
- ResNet-18/34/50, ImageNet-pretrained
- AdaptiveAvgPool → flat feature vector

### Input conditioning
Each step receives:
- **Image**: 3×1024×1024 retina tensor (same image every step)
- **Previous point**: `(x, y, dx, dy, class_id)` — last emitted contour point
  + orientation + class. First step uses zeros.
- **Start point**: `(x, y)` — the first point of the current contour (for
  closure detection at inference; during training the full GT contour is known).

### Heads (replace bbox_head)
| Head | Output | Activation | Purpose |
|------|--------|------------|---------|
| **point_head** | `(x, y)` | Sigmoid × RETINA_SIZE | Next contour point |
| **orientation_head** | `(dx, dy)` | L2-normalized | Unit tangent at this point |
| **class_head** | logits (NUM_CLASSES) | — | Class token (unchanged) |

### Contour closure (inference only)
After each predicted point `p_t`, compute `‖p_t − p_0‖`. If this distance
falls below a **closure threshold** (hyperparameter, e.g. 15 px in retina
space) *and* at least 3 points have been emitted, the contour is closed and
detection moves to the next item.

## Losses

### 1. Point regression loss
`SmoothL1Loss` between predicted and GT contour points (normalized to [0,1]).

### 2. Orientation loss
`1 − cos_similarity(pred_orient, gt_orient)` averaged over non-NONE steps.
GT orientation at point `i` is the unit vector `(p_{i+1} − p_{i−1})` (central
difference); at endpoints, use forward/backward difference.

### 3. Intersection loss
Penalizes self-intersecting contour edges. For every pair of non-adjacent
predicted segments `(p_i, p_{i+1})` and `(p_j, p_{j+1})`, check for
intersection using the 2D segment-intersection test (cross-product method from
`utility.py`). The loss is the count of intersections (or a soft
differentiable proxy using signed-area products clamped through sigmoid).

### 4. Classification loss
`CrossEntropyLoss` (unchanged from v1).

### Combined loss
```
L = w_pt * L_point + w_orient * L_orientation + w_intersect * L_intersection + w_cls * L_class
```

## Bootstrapping Phase (Rotated Paragraphs)

During bootstrapping, `SyntheticPage` generates paragraphs with **random
rotation** (±30°). This ensures contours are non-axis-aligned, forcing the
model to learn true contour tracing rather than degenerate axis-aligned
rectangles.

Changes to `generate_training_data.py`:
- After rendering each paragraph, rotate it by a random angle.
- Recompute all bounding contours from the rotated pixel masks (convex hull
  or alpha-shape of non-background pixels).
- Store contour as an ordered list of `(x, y)` points instead of / in addition
  to the `(x1,y1,x2,y2)` bbox.

## Dataset: `ContourOCRDataset`

Each sample: `(retina_image, prev_state, start_point, target)`

- `prev_state`: `(x, y, dx, dy, class_id)` of previous contour point (or zeros)
- `start_point`: `(x, y)` of first contour point (for closure context)
- `target`: `(x, y, dx, dy, class_id)` of next GT contour point

Teacher forcing: GT previous point is always used during training.

## Files

| File | Role |
|------|------|
| `train_04.py` | Training script for ContourOCRNet |
| `infer_04.py` | Inference with autoregressive contour tracing |
| `generate_training_data.py` | Extended with rotated paragraphs + contour extraction |

## Contour Extraction from Masks

Given a binary mask of a detected element:
1. Find non-zero pixels → compute convex hull (or concave hull)
2. Simplify with Ramer-Douglas-Peucker to ~8–20 points
3. Order clockwise
4. Compute per-point orientation as central-difference unit tangents

This runs offline during data generation, not at training time.
