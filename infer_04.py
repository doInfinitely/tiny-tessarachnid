"""
Inference script for ContourOCRNet (V2).

Runs autoregressive contour tracing at 5 hierarchy levels:
  0. Page region on the full image  (scale_and_pad)
  1. Paragraphs on the full image   (scale_and_pad)
  2. Lines within each paragraph    (forward_cascade_step — rotation-normalized)
  3. Words within each line         (forward_cascade_step — rotation-normalized)
  4. Characters within each word    (forward_cascade_step — rotation-normalized)

Levels 0-1 match training: the model sees scale_and_pad(page_image).
Levels 2-4 match training: the model sees forward_cascade_step(page_image,
parent_contour) which crops, rotates to upright, tight-crops, and
scale_and_pads.  Detected contours are lifted back to page coordinates
via lift_contour (the inverse of forward_cascade_step).

At each step the model predicts the next contour point, orientation vector,
and class token. Contour closure is detected when the predicted point falls
within a threshold distance of the start point (and >= 3 points emitted).
"""

import argparse
import math
import random
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from cascade_transforms import forward_cascade_step, lift_contour
from generate_training_data import (
    CHAR_CLASS_OFFSET,
    CLASS_NONE,
    CLASS_PAGE,
    CLASS_PARAGRAPH,
    CLASS_LINE,
    CLASS_WORD,
    CLASS_TO_CHAR,
    PREV_CONTOUR_NONE,
    RETINA_SIZE,
    SyntheticPage,
    discover_fonts,
    scale_and_pad,
)
from train_04 import ContourOCRNet
from train_02 import _remap_old_backbone_keys


def retina_to_source_point(rx, ry, scale, ox, oy):
    """Map a retina-space point back to source-image coordinates."""
    return (int((rx - ox) / scale), int((ry - oy) / scale))


def class_to_char(class_id):
    return CLASS_TO_CHAR.get(class_id, "?")


# ---------------------------------------------------------------------------
# Autoregressive contour generation
# ---------------------------------------------------------------------------

def _img_to_tensor(pil_img, device):
    """Convert PIL RGB image to [1, 3, H, W] float tensor on device."""
    return (
        torch.from_numpy(np.array(pil_img))
        .permute(2, 0, 1).float().unsqueeze(0) / 255.0
    ).to(device)


@torch.no_grad()
def _run_autoregressive(model, img_t, level_id, device,
                        max_points=256, closure_threshold=15.0):
    """Core autoregressive contour tracing loop.

    The model emits contour points one at a time. When a point is within
    `closure_threshold` pixels (retina space) of the start point and at least
    3 points have been emitted, the contour is closed and we move to the next
    element. When the model predicts CLASS_NONE, no more elements remain.

    Args:
        img_t: Pre-prepared retina image tensor [1, 3, H, W] on device.
        level_id: Hierarchy level (0-4).

    Returns:
        list of (retina_contour, class_id) where retina_contour is a list of
        (x, y) floats in retina coordinates.
    """
    level_t = torch.tensor([level_id], dtype=torch.long, device=device)

    seq = [list(PREV_CONTOUR_NONE)]  # (x, y, dx, dy, class_id)
    start_point = torch.zeros(1, 2, device=device)

    detections = []
    current_contour_retina = []
    current_class = None

    for step in range(max_points):
        prev_t = torch.tensor([seq], dtype=torch.float32, device=device)
        point_pred, orient_pred, class_pred = model(
            img_t, prev_t, start_point, level_t,
        )

        # Take last position
        class_id = class_pred[0, -1].argmax().item()
        if class_id == CLASS_NONE:
            # Flush any in-progress contour
            if current_contour_retina and current_class is not None:
                detections.append((list(current_contour_retina), current_class))
            break

        px, py = point_pred[0, -1].tolist()
        dx, dy = orient_pred[0, -1].tolist()

        if current_class is None:
            # Starting a new contour
            current_class = class_id
            current_contour_retina = [(px, py)]
            start_point = torch.tensor([[px, py]], device=device)
        else:
            current_contour_retina.append((px, py))

            # Check closure
            if len(current_contour_retina) >= 3:
                sx, sy = current_contour_retina[0]
                dist = math.hypot(px - sx, py - sy)
                if dist < closure_threshold:
                    detections.append((list(current_contour_retina), current_class))
                    current_contour_retina = []
                    current_class = None

        seq.append([px, py, dx, dy, class_id])

    return detections


def generate_contours(model, source_img, bg_color, level_id, device,
                      max_points=256, closure_threshold=15.0):
    """Generate contours using simple scale_and_pad (for levels 0 and 1).

    Matches training: _make_page_sequence and _make_para_sequence both use
    scale_and_pad(page.image) with no rotation normalization.

    Returns:
        list of (source_contour, class_id) where source_contour is a list of
        (x, y) tuples in source-image coordinates.
    """
    retina, scale, ox, oy = scale_and_pad(source_img, bg_color)
    img_t = _img_to_tensor(retina, device)

    retina_dets = _run_autoregressive(
        model, img_t, level_id, device, max_points, closure_threshold)

    detections = []
    for retina_contour, class_id in retina_dets:
        src_contour = [
            retina_to_source_point(px, py, scale, ox, oy)
            for px, py in retina_contour
        ]
        detections.append((src_contour, class_id))
    return detections


def generate_contours_cascade(model, page_img, parent_contour, bg_color,
                              level_id, device, max_points=256,
                              closure_threshold=15.0):
    """Generate contours using forward_cascade_step (for levels 2-4).

    Matches training: _make_line_sequence, _make_word_sequence, and
    _make_char_sequence all use forward_cascade_step(page.image,
    parent_contour_global, bg_color) which crops the parent contour region,
    rotates to upright, tight-crops the rotated bbox, and scale_and_pads.

    Detected contours are lifted back to page_img coordinates via
    lift_contour (inverse of forward_cascade_step).

    Returns:
        list of (page_contour, class_id) where page_contour is a list of
        (x, y) tuples in page_img coordinates.
    """
    retina_pil, xf = forward_cascade_step(page_img, parent_contour, bg_color)
    img_t = _img_to_tensor(retina_pil, device)

    retina_dets = _run_autoregressive(
        model, img_t, level_id, device, max_points, closure_threshold)

    detections = []
    for retina_contour, class_id in retina_dets:
        page_contour_pts = lift_contour(retina_contour, xf)
        detections.append((page_contour_pts, class_id))
    return detections


def contour_to_bbox(contour):
    """Compute axis-aligned bounding box from a contour polygon."""
    xs = [p[0] for p in contour]
    ys = [p[1] for p in contour]
    return (min(xs), min(ys), max(xs), max(ys))


# ---------------------------------------------------------------------------
# Hierarchical inference
# ---------------------------------------------------------------------------
def run_inference(model, image_path, output_path, device,
                  closure_threshold=15.0):
    """5-level hierarchical inference with contour tracing.

    Matches the training coordinate pipeline exactly:
      - Levels 0/1: scale_and_pad(page_image) — same as _make_page_sequence
        and _make_para_sequence in train_04.py.
      - Levels 2-4: forward_cascade_step(page_image, parent_contour) — same
        as _make_line/word/char_sequence.  Detected contours are lifted back
        to page coordinates via lift_contour.
    """
    page_img = Image.open(image_path).convert("RGB")
    from annotate_real import estimate_background_color
    bg_color = estimate_background_color(page_img)
    print(f"Image: {image_path} ({page_img.size[0]}x{page_img.size[1]}), bg={bg_color}")

    model.eval()

    # Level 0: page regions on full image (scale_and_pad)
    page_regions = generate_contours(
        model, page_img, bg_color, 0, device,
        closure_threshold=closure_threshold,
    )
    print(f"Detected {len(page_regions)} page region(s)")

    # Level 1: paragraphs on full image (scale_and_pad — matches training)
    paragraphs = generate_contours(
        model, page_img, bg_color, 1, device,
        closure_threshold=closure_threshold,
    )
    print(f"Detected {len(paragraphs)} paragraph(s)")

    all_text = []
    all_para_contours = []
    all_line_contours = []
    all_word_contours = []
    all_char_contours = []

    for pi, (para_contour, _) in enumerate(paragraphs):
        # para_contour is already in page_img coordinates
        all_para_contours.append(para_contour)

        # Level 2: lines within paragraph (forward_cascade_step from page)
        lines = generate_contours_cascade(
            model, page_img, para_contour, bg_color, 2, device,
            closure_threshold=closure_threshold,
        )
        print(f"  Paragraph {pi+1}: {len(lines)} lines")

        for li, (line_contour, _) in enumerate(lines):
            # line_contour is already in page_img coords (via lift_contour)
            all_line_contours.append(line_contour)

            # Level 3: words within line (forward_cascade_step from page)
            words = generate_contours_cascade(
                model, page_img, line_contour, bg_color, 3, device,
                closure_threshold=closure_threshold,
            )

            line_text = []
            for wi, (word_contour, _) in enumerate(words):
                # word_contour is already in page_img coords
                all_word_contours.append(word_contour)

                # Level 4: characters within word (forward_cascade_step from page)
                chars = generate_contours_cascade(
                    model, page_img, word_contour, bg_color, 4, device,
                    closure_threshold=closure_threshold,
                )

                word_text = []
                for char_contour, class_id in chars:
                    # char_contour is already in page_img coords
                    all_char_contours.append((char_contour, class_id))

                    if class_id >= CHAR_CLASS_OFFSET:
                        word_text.append(class_to_char(class_id))
                    else:
                        word_text.append("?")

                line_text.append("".join(word_text))

            all_text.append((pi, li, " ".join(line_text)))

    # Print detected text
    print("\n--- Detected Text ---")
    if not all_text:
        print("  (no text detected)")
    current_para = -1
    for pi, li, text in all_text:
        if pi != current_para:
            current_para = pi
            print(f"Paragraph {pi+1}:")
        print(f"  Line {li+1}: {text}")

    # Visualize
    _visualize(page_img, page_regions, all_para_contours, all_line_contours,
               all_word_contours, all_char_contours, output_path)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def _draw_contour(draw, contour, color, width=2):
    """Draw a closed polygon contour on the image."""
    if len(contour) < 2:
        return
    pts = contour + [contour[0]]  # close it
    for i in range(len(pts) - 1):
        draw.line([pts[i], pts[i + 1]], fill=color, width=width)
    # Draw vertices as small dots
    for x, y in contour:
        r = max(1, width)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)


def _visualize(page_img, page_regions, para_contours, line_contours,
               word_contours, char_contours, output_path):
    vis = page_img.copy()
    draw = ImageDraw.Draw(vis)

    for page_contour, _ in page_regions:
        _draw_contour(draw, page_contour, (128, 0, 255), width=4)

    for pc in para_contours:
        _draw_contour(draw, pc, (255, 0, 0), width=3)

    for lc in line_contours:
        _draw_contour(draw, lc, (0, 0, 255), width=2)

    for wc in word_contours:
        _draw_contour(draw, wc, (255, 165, 0), width=2)

    for cc, class_id in char_contours:
        _draw_contour(draw, cc, (0, 180, 0), width=1)

    vis.save(output_path)
    print(f"Visualization saved to {output_path}")


# ---------------------------------------------------------------------------
# Standalone synthetic-page inference
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="ContourOCRNet (V2) inference")
    parser.add_argument("--model-path", default="model_04.pth")
    parser.add_argument("--output", default="infer_04_output.png")
    parser.add_argument("--page-width", type=int, default=2048)
    parser.add_argument("--page-height", type=int, default=2800)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--closure-threshold", type=float, default=15.0,
                        help="Retina-space distance for contour closure")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to real image (omit for synthetic)")

    # Model architecture (must match checkpoint)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=256)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ContourOCRNet(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
    ).to(device)

    try:
        ckpt = torch.load(args.model_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt)
        print(f"Loaded model from {args.model_path}")
    except FileNotFoundError:
        print(f"Warning: {args.model_path} not found, using initialized weights")

    model.eval()

    if args.image:
        run_inference(model, args.image, args.output, device,
                      closure_threshold=args.closure_threshold)
    else:
        # Generate synthetic page and run inference
        fonts = discover_fonts()
        if not fonts:
            print("No fonts found!")
            sys.exit(1)
        page = SyntheticPage(fonts, args.page_width, args.page_height,
                             rotate_paragraphs=True)
        page_img = page.image
        bg_color = page.bg_color
        print(f"Generated page: {len(page.paragraphs)} paragraphs, bg={bg_color}")

        # Save temp image and run inference on it
        tmp_path = "/tmp/_contour_ocr_temp.png"
        page_img.save(tmp_path)
        run_inference(model, tmp_path, args.output, device,
                      closure_threshold=args.closure_threshold)

        # Also print ground truth for comparison
        print("\n--- Ground Truth ---")
        for pi, para in enumerate(page.paragraphs):
            print(f"Paragraph {pi+1}:")
            for li, line in enumerate(para["lines"]):
                gt_text = " ".join(
                    "".join(ch["char"] for ch in word["characters"])
                    for word in line["words"]
                )
                print(f"  Line {li+1}: {gt_text}")


if __name__ == "__main__":
    main()
