"""
Inference script for ContourOCRNet (V4).

Runs autoregressive contour tracing at 5 hierarchy levels:
  0. Page region on the full image
  1. Paragraphs on the page
  2. Lines within each paragraph
  3. Words within each line
  4. Characters within each word

At each level the model predicts contour points, orientation vectors, and
class tokens.  The cascade works entirely in retina space:

  detect contour → rotate so orientation points up → axis-aligned bbox →
  crop from retina → scale_and_pad onto fresh retina → recurse

Each cascade level is described by:
  (scale, ox, oy, crop_x, crop_y, angle, rot_cx, rot_cy)

To map coordinates back to the original image, reverse each level one at a
time from deepest to shallowest.
"""

import argparse
import math
import random
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from infer_04_cached import encode_image, decode_step
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


# ---------------------------------------------------------------------------
# Cascade coordinate tracking
# ---------------------------------------------------------------------------
def rotate_point(x, y, angle_deg, cx, cy):
    """Rotate (x, y) by angle_deg around (cx, cy).

    Positive angle = counterclockwise on screen (PIL convention, y-down).
    """
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    dx, dy = x - cx, y - cy
    return (cos_a * dx + sin_a * dy + cx,
            -sin_a * dx + cos_a * dy + cy)


def orientation_angle(dx, dy):
    """Angle (degrees) to rotate so orientation vector (dx, dy) points up."""
    return math.degrees(math.atan2(dx, -dy))


def reverse_level(rx, ry, scale, ox, oy, crop_x, crop_y,
                  angle=0.0, rot_cx=0.0, rot_cy=0.0):
    """Reverse one cascade level: child retina coords → parent retina coords.

    Args:
        rx, ry:            point in this level's retina space
        scale, ox, oy:     from this level's scale_and_pad
        crop_x, crop_y:    top-left of crop in parent's (rotated) retina space
        angle:             rotation applied to parent retina (degrees, CCW)
        rot_cx, rot_cy:    center of rotation in parent retina space

    Returns:
        (x, y) in the parent level's retina space (float).
    """
    # 1. Undo scale_and_pad + crop offset → position in rotated parent retina
    x = (rx - ox) / scale + crop_x
    y = (ry - oy) / scale + crop_y
    # 2. Undo rotation
    if angle != 0.0:
        x, y = rotate_point(x, y, -angle, rot_cx, rot_cy)
    return (x, y)



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
def generate_contours(model, retina, level_id, device,
                      max_points=256, closure_threshold=15.0):
    """Autoregressively trace contours on a prepared retina.

    Args:
        retina: 1024x1024 PIL image (already scale_and_padded).

    Returns:
        detections: list of (retina_contour, class_id, orientation) with
        retina_contour as (rx, ry) floats and orientation as (dx, dy).
    """
    img_t = _img_to_tensor(retina, device)
    level_t = torch.tensor([level_id], dtype=torch.long, device=device)

    # Encode image once; reuse cached features for every autoregressive step
    img_feat = encode_image(model, img_t)

    seq = [list(PREV_CONTOUR_NONE)]  # (x, y, dx, dy, class_id)
    start_point = torch.zeros(1, 2, device=device)

    detections = []
    current_contour_retina = []
    current_class = None
    current_orient = None

    for step in range(max_points):
        prev_t = torch.tensor([seq], dtype=torch.float32, device=device)
        point_pred, orient_pred, class_pred = decode_step(
            model, img_feat, prev_t, start_point, level_t,
        )

        class_id = class_pred[0, -1].argmax().item()
        if class_id == CLASS_NONE:
            if current_contour_retina and current_class is not None:
                detections.append((list(current_contour_retina), current_class,
                                   current_orient))
            break

        px, py = point_pred[0, -1].tolist()
        dx, dy = orient_pred[0, -1].tolist()

        if current_class is None:
            current_class = class_id
            current_contour_retina = [(px, py)]
            current_orient = (dx, dy)
            start_point = torch.tensor([[px, py]], device=device)
        else:
            current_contour_retina.append((px, py))

            if len(current_contour_retina) >= 3:
                sx, sy = current_contour_retina[0]
                dist = math.hypot(px - sx, py - sy)
                if dist < closure_threshold:
                    detections.append((list(current_contour_retina), current_class,
                                       current_orient))
                    current_contour_retina = []
                    current_class = None
                    current_orient = None

        seq.append([px, py, dx, dy, class_id])

    return detections


# ---------------------------------------------------------------------------
# Hierarchical inference
# ---------------------------------------------------------------------------
def run_inference(model, image_path, output_path, device,
                  closure_threshold=15.0):
    """5-level hierarchical inference with contour tracing."""
    page_img = Image.open(image_path).convert("RGB")
    from annotate_real import estimate_background_color
    bg_color = estimate_background_color(page_img)
    pw, ph = page_img.size
    print(f"Image: {image_path} ({pw}x{ph}), bg={bg_color}")

    model.eval()

    def _cascade_step(retina_img, contour, orient, bg):
        """Full cascade forward step: rotate → bbox → crop → scale_and_pad.

        Returns (new_retina, level) where level is the tuple needed for
        reverse_level: (scale, ox, oy, crop_x, crop_y, angle, rot_cx, rot_cy).
        """
        # Centroid of contour
        cx = sum(x for x, y in contour) / len(contour)
        cy = sum(y for x, y in contour) / len(contour)
        # Rotation angle to make orientation point up
        angle = orientation_angle(*orient)
        # Rotate contour points around centroid
        rotated = [rotate_point(x, y, angle, cx, cy) for x, y in contour]
        # Axis-aligned bbox of rotated contour
        xs = [x for x, y in rotated]
        ys = [y for x, y in rotated]
        bx1, by1 = int(min(xs)), int(min(ys))
        bx2, by2 = int(max(xs)) + 1, int(max(ys)) + 1
        # Rotate image around centroid and crop
        rotated_img = retina_img.rotate(angle, center=(cx, cy),
                                        expand=False, fillcolor=bg,
                                        resample=Image.BILINEAR)
        crop_img = rotated_img.crop((bx1, by1, bx2, by2))
        # Scale and pad onto fresh retina
        new_retina, scale, ox, oy = scale_and_pad(crop_img, bg)
        return new_retina, (scale, ox, oy, bx1, by1, angle, cx, cy)

    def _reverse_contour(retina_contour, *levels):
        """Reverse a retina contour through cascade levels.

        Each level is (scale, ox, oy, crop_x, crop_y, angle, rot_cx, rot_cy).
        Applied in order (deepest level first, top level last).
        """
        result = []
        for rx, ry in retina_contour:
            x, y = float(rx), float(ry)
            for level in levels:
                x, y = reverse_level(x, y, *level)
            result.append((int(round(x)), int(round(y))))
        return result

    # Level 0: scale_and_pad page onto retina, detect page regions
    retina0, s0, ox0, oy0 = scale_and_pad(page_img, bg_color)
    L0 = (s0, ox0, oy0, 0, 0)

    page_dets = generate_contours(
        model, retina0, 0, device,
        closure_threshold=closure_threshold,
    )
    print(f"Detected {len(page_dets)} page region(s)")

    # Level 1: detect paragraphs (same retina as page level)
    para_dets = generate_contours(
        model, retina0, 1, device,
        closure_threshold=closure_threshold,
    )
    print(f"Detected {len(para_dets)} paragraph(s)")

    all_text = []
    all_page_contours = []
    all_para_contours = []
    all_line_contours = []
    all_word_contours = []
    all_char_contours = []

    for page_retina, page_cls, _ in page_dets:
        all_page_contours.append(
            (_reverse_contour(page_retina, L0), page_cls)
        )

    for pi, (para_retina, para_cls, para_orient) in enumerate(para_dets):
        all_para_contours.append(
            (_reverse_contour(para_retina, L0), para_cls)
        )

        # Cascade step: rotate → bbox → crop → scale_and_pad
        retina1, L1 = _cascade_step(retina0, para_retina, para_orient, bg_color)

        # Level 2: detect lines within paragraph
        line_dets = generate_contours(
            model, retina1, 2, device,
            closure_threshold=closure_threshold,
        )
        print(f"  Paragraph {pi+1}: {len(line_dets)} lines")

        for li, (line_retina, line_cls, line_orient) in enumerate(line_dets):
            all_line_contours.append(
                _reverse_contour(line_retina, L1, L0)
            )

            retina2, L2 = _cascade_step(retina1, line_retina, line_orient, bg_color)

            # Level 3: detect words within line
            word_dets = generate_contours(
                model, retina2, 3, device,
                closure_threshold=closure_threshold,
            )

            line_text = []
            for wi, (word_retina, word_cls, word_orient) in enumerate(word_dets):
                all_word_contours.append(
                    _reverse_contour(word_retina, L2, L1, L0)
                )

                retina3, L3 = _cascade_step(retina2, word_retina, word_orient, bg_color)

                # Level 4: detect characters within word
                char_dets = generate_contours(
                    model, retina3, 4, device,
                    closure_threshold=closure_threshold,
                )

                word_text = []
                for char_retina, class_id, _ in char_dets:
                    all_char_contours.append((
                        _reverse_contour(char_retina, L3, L2, L1, L0),
                        class_id,
                    ))

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
    _visualize(page_img, all_page_contours, all_para_contours,
               all_line_contours, all_word_contours, all_char_contours,
               output_path)


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
    for x, y in contour:
        r = max(1, width)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)


def _visualize(page_img, page_contours, para_contours, line_contours,
               word_contours, char_contours, output_path):
    vis = page_img.copy()
    draw = ImageDraw.Draw(vis)

    for page_contour, _ in page_contours:
        _draw_contour(draw, page_contour, (128, 0, 255), width=4)

    for para_contour, _ in para_contours:
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
        description="ContourOCRNet (V4) inference")
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
