"""
Inference script for ContourOCRNet (V2).

Runs autoregressive contour tracing at 4 hierarchy levels:
  1. Paragraphs on the page
  2. Lines within each paragraph
  3. Words within each line
  4. Characters within each word

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

from generate_training_data import (
    CHAR_CLASS_OFFSET,
    CLASS_NONE,
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
@torch.no_grad()
def generate_contours(model, source_img, bg_color, level_id, device,
                      max_points=256, closure_threshold=15.0):
    """Autoregressively trace contours for all elements at one hierarchy level.

    The model emits contour points one at a time. When a point is within
    `closure_threshold` pixels (retina space) of the start point and at least
    3 points have been emitted, the contour is closed and we move to the next
    element. When the model predicts CLASS_NONE, no more elements remain.

    Returns:
        list of (source_contour, class_id) where source_contour is a list of
        (x, y) tuples in source-image coordinates.
    """
    retina, scale, ox, oy = scale_and_pad(source_img, bg_color)
    img_t = (
        torch.from_numpy(np.array(retina))
        .permute(2, 0, 1).float().unsqueeze(0) / 255.0
    ).to(device)

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
                src_contour = [
                    retina_to_source_point(px, py, scale, ox, oy)
                    for px, py in current_contour_retina
                ]
                detections.append((src_contour, current_class))
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
                    src_contour = [
                        retina_to_source_point(rpx, rpy, scale, ox, oy)
                        for rpx, rpy in current_contour_retina
                    ]
                    detections.append((src_contour, current_class))
                    current_contour_retina = []
                    current_class = None

        seq.append([px, py, dx, dy, class_id])

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
    """4-level hierarchical inference with contour tracing."""
    page_img = Image.open(image_path).convert("RGB")
    from annotate_real import estimate_background_color
    bg_color = estimate_background_color(page_img)
    print(f"Image: {image_path} ({page_img.size[0]}x{page_img.size[1]}), bg={bg_color}")

    model.eval()

    # Level 0: paragraphs
    paragraphs = generate_contours(
        model, page_img, bg_color, 0, device,
        closure_threshold=closure_threshold,
    )
    print(f"Detected {len(paragraphs)} paragraphs")

    all_text = []
    all_line_contours = []
    all_word_contours = []
    all_char_contours = []

    for pi, (para_contour, _) in enumerate(paragraphs):
        para_bbox = contour_to_bbox(para_contour)
        para_crop = page_img.crop(para_bbox)

        # Level 1: lines
        lines = generate_contours(
            model, para_crop, bg_color, 1, device,
            closure_threshold=closure_threshold,
        )
        print(f"  Paragraph {pi+1}: {len(lines)} lines")

        for li, (line_contour, _) in enumerate(lines):
            page_line_contour = [
                (x + para_bbox[0], y + para_bbox[1]) for x, y in line_contour
            ]
            all_line_contours.append(page_line_contour)

            line_bbox = contour_to_bbox(line_contour)
            line_crop = para_crop.crop(line_bbox)

            # Level 2: words
            words = generate_contours(
                model, line_crop, bg_color, 2, device,
                closure_threshold=closure_threshold,
            )

            line_text = []
            for wi, (word_contour, _) in enumerate(words):
                page_word_contour = [
                    (x + line_bbox[0] + para_bbox[0],
                     y + line_bbox[1] + para_bbox[1])
                    for x, y in word_contour
                ]
                all_word_contours.append(page_word_contour)

                word_bbox = contour_to_bbox(word_contour)
                word_crop = line_crop.crop(word_bbox)

                # Level 3: characters
                chars = generate_contours(
                    model, word_crop, bg_color, 3, device,
                    closure_threshold=closure_threshold,
                )

                word_text = []
                for char_contour, class_id in chars:
                    page_char_contour = [
                        (x + word_bbox[0] + line_bbox[0] + para_bbox[0],
                         y + word_bbox[1] + line_bbox[1] + para_bbox[1])
                        for x, y in char_contour
                    ]
                    all_char_contours.append((page_char_contour, class_id))

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
    _visualize(page_img, paragraphs, all_line_contours, all_word_contours,
               all_char_contours, output_path)


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


def _visualize(page_img, paragraphs, line_contours, word_contours,
               char_contours, output_path):
    vis = page_img.copy()
    draw = ImageDraw.Draw(vis)

    for para_contour, _ in paragraphs:
        _draw_contour(draw, para_contour, (255, 0, 0), width=3)

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
