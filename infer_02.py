"""
Hierarchical OCR inference script for RetinaOCRNet.

Generates a synthetic page, then runs autoregressive detection at three levels:
  1. Paragraphs on the page
  2. Lines within each paragraph
  3. Characters within each line

Replaces teacher forcing with the model's own predictions. The image stays
unchanged throughout each autoregressive loop — the model relies on prev_bbox
to determine what to detect next. Visualizes all detections on the original
page and prints detected text.
"""

import argparse
import random
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from generate_training_data import (
    CHAR_CLASS_OFFSET,
    CLASS_NONE,
    CLASS_PAGE,
    CLASS_LINE,
    CLASS_TO_CHAR,
    CLASS_WORD,
    PREV_BBOX_NONE,
    RETINA_SIZE,
    SyntheticPage,
    discover_fonts,
    scale_and_pad,
)
from train_02 import RetinaOCRNet, _remap_old_backbone_keys


# ---------------------------------------------------------------------------
# Coordinate mapping (reverse of bbox_to_retina)
# ---------------------------------------------------------------------------
def retina_to_source(retina_bbox, scale, ox, oy):
    """Map a retina-space bbox back to source-image coordinates."""
    x1, y1, x2, y2 = retina_bbox
    return (
        int((x1 - ox) / scale),
        int((y1 - oy) / scale),
        int((x2 - ox) / scale),
        int((y2 - oy) / scale),
    )


def class_to_char(class_id):
    """Convert a character class id to the corresponding character."""
    return CLASS_TO_CHAR.get(class_id, "?")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _find_content_regions(image, bg_color, gap_tolerance=5, scan_axis="rows"):
    """Find distinct content regions separated by gaps.

    scan_axis="rows": scan top-to-bottom, split by blank rows (for paragraphs/lines).
    scan_axis="cols": scan left-to-right, split by blank columns (for characters).

    Returns a list of (x1, y1, x2, y2) bounding boxes.
    """
    w, h = image.size
    arr = np.array(image, dtype=np.int16)
    bg = np.array(bg_color, dtype=np.int16)
    content = np.abs(arr - bg).max(axis=2) > 0

    if scan_axis == "rows":
        # Scan rows top-to-bottom, find y-bands, compute x-extent per band
        line_has_content = content.any(axis=1)
        length = h
    else:
        # Scan columns left-to-right, find x-bands, compute y-extent per band
        line_has_content = content.any(axis=0)
        length = w

    regions = []
    in_region = False
    region_start = 0
    gap_count = 0

    def _finish_region(region_end):
        if scan_axis == "rows":
            band = content[region_start:region_end + 1, :]
            cross = band.any(axis=0)
            if cross.any():
                c1 = int(np.argmax(cross))
                c2 = int(len(cross) - 1 - np.argmax(cross[::-1]))
                regions.append((c1, region_start, c2 + 1, region_end + 1))
        else:
            band = content[:, region_start:region_end + 1]
            cross = band.any(axis=1)
            if cross.any():
                c1 = int(np.argmax(cross))
                c2 = int(len(cross) - 1 - np.argmax(cross[::-1]))
                regions.append((region_start, c1, region_end + 1, c2 + 1))

    for i in range(length):
        if line_has_content[i]:
            if not in_region:
                region_start = i
                in_region = True
            gap_count = 0
        else:
            if in_region:
                gap_count += 1
                if gap_count > gap_tolerance:
                    _finish_region(i - gap_count)
                    in_region = False

    if in_region:
        region_end = length - 1
        while region_end > region_start and not line_has_content[region_end]:
            region_end -= 1
        _finish_region(region_end)

    return regions


# ---------------------------------------------------------------------------
# Core detection loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def detect_level(model, source_img, bg_color, device, max_detections=200,
                 gap_tolerance=20, scan_axis="rows", use_model_bbox=False):
    """Run one level of detection on *source_img*.

    The image stays unchanged throughout the loop. The model relies on
    prev_bbox to determine what to detect next.

    If use_model_bbox is False (default), uses pixel analysis to find content
    regions and the model to count/classify them.

    If use_model_bbox is True, uses the model's own bbox predictions directly.

    Returns a list of (source_bbox, class_id) tuples.
    """
    if use_model_bbox:
        return _detect_with_model_bbox(model, source_img, bg_color, device,
                                       max_detections)

    # Step 1: Find content regions via pixel analysis
    regions = _find_content_regions(source_img, bg_color,
                                    gap_tolerance=gap_tolerance,
                                    scan_axis=scan_axis)
    if not regions:
        return []

    # Step 2: Run model autoregressively to count detections and get class_ids
    # Same image fed every step — no erasure
    retina, scale, ox, oy = scale_and_pad(source_img, bg_color)
    img_t = (
        torch.from_numpy(np.array(retina))
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
        / 255.0
    ).to(device)

    class_ids = []
    prev_bbox = PREV_BBOX_NONE

    for _ in range(max_detections):
        # Model expects 5-dim prev_bbox (x1,y1,x2,y2,class_id); trim any extras
        prev_t = torch.tensor(prev_bbox[:5], dtype=torch.float32).unsqueeze(0).to(device)

        bbox_pred, class_pred = model(img_t, prev_t)
        class_id = class_pred.argmax(dim=1).item()
        if class_id == CLASS_NONE:
            break

        class_ids.append(class_id)

        # Update prev_bbox from the corresponding pixel-analysis region
        if len(class_ids) <= len(regions):
            region = regions[len(class_ids) - 1]
            rx1 = region[0] * scale + ox
            ry1 = region[1] * scale + oy
            rx2 = region[2] * scale + ox
            ry2 = region[3] * scale + oy
            prev_bbox = (rx1, ry1, rx2, ry2, class_id)
        else:
            break

    # Step 3: Pair regions with class_ids (take min of both counts)
    n = min(len(class_ids), len(regions))
    return [(regions[i], class_ids[i]) for i in range(n)]


def _detect_with_model_bbox(model, source_img, bg_color, device, max_detections):
    """Autoregressive detection using the model's own bbox predictions.

    The image stays unchanged — no erasure. prev_bbox tracks what was last detected.
    """
    retina, scale, ox, oy = scale_and_pad(source_img, bg_color)
    img_t = (
        torch.from_numpy(np.array(retina))
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
        / 255.0
    ).to(device)

    detections = []
    prev_bbox = PREV_BBOX_NONE

    for _ in range(max_detections):
        # Model expects 5-dim prev_bbox (x1,y1,x2,y2,class_id); trim any extras
        prev_t = torch.tensor(prev_bbox[:5], dtype=torch.float32).unsqueeze(0).to(device)

        bbox_pred, class_pred = model(img_t, prev_t)
        class_id = class_pred.argmax(dim=1).item()
        if class_id == CLASS_NONE:
            break

        # Convert model's retina bbox to source coords
        rb = bbox_pred[0].tolist()
        source_bbox = retina_to_source(rb, scale, ox, oy)
        # Clamp to image bounds
        w, h = source_img.size
        source_bbox = (
            max(0, min(source_bbox[0], w - 1)),
            max(0, min(source_bbox[1], h - 1)),
            max(0, min(source_bbox[2], w)),
            max(0, min(source_bbox[3], h)),
        )
        if source_bbox[2] <= source_bbox[0] or source_bbox[3] <= source_bbox[1]:
            break

        detections.append((source_bbox, class_id))
        prev_bbox = (rb[0], rb[1], rb[2], rb[3], class_id)

    return detections


# ---------------------------------------------------------------------------
# Bottom-up character detection (Level 3 replacement)
# ---------------------------------------------------------------------------
def _generate_candidate_boxes(line_crop, bg_color, gap_tolerance=1):
    """Generate character candidate boxes from a line image.

    Uses column-based content regions as natural character boxes. For regions
    wider than 1.5x the expected character width (touching chars), generates
    overlapping sub-windows at stride expected_width // 2.
    """
    regions = _find_content_regions(line_crop, bg_color, gap_tolerance=gap_tolerance,
                                    scan_axis="cols")
    if not regions:
        return []

    # Estimate expected char width from median region width
    widths = [r[2] - r[0] for r in regions]
    expected_w = int(np.median(widths)) if widths else 1
    expected_w = max(expected_w, 1)

    candidates = []
    for (x1, y1, x2, y2) in regions:
        rw = x2 - x1
        if rw <= expected_w * 1.5:
            # Normal-width region — use as-is
            candidates.append((x1, y1, x2, y2))
        else:
            # Wide region (touching chars) — generate overlapping sub-windows
            stride = max(expected_w // 2, 1)
            for sx in range(x1, x2 - expected_w + 1, stride):
                candidates.append((sx, y1, sx + expected_w, y2))
            # Ensure we cover the rightmost part
            if candidates and candidates[-1][2] < x2:
                candidates.append((x2 - expected_w, y1, x2, y2))

    return candidates


@torch.no_grad()
def _classify_candidates_batched(model, line_crop, bg_color, candidates, device,
                                 batch_size=32):
    """Classify candidate character boxes in batches.

    For each candidate: crop from line image, scale_and_pad to 1024x1024,
    run model with prev_bbox=PREV_BBOX_NONE.

    Returns list of (box, class_id, confidence) for non-CLASS_NONE predictions.
    """
    results = []
    prev_t = torch.tensor(PREV_BBOX_NONE[:5], dtype=torch.float32).unsqueeze(0).to(device)

    for i in range(0, len(candidates), batch_size):
        batch_boxes = candidates[i:i + batch_size]
        imgs = []
        for (x1, y1, x2, y2) in batch_boxes:
            crop = line_crop.crop((x1, y1, x2, y2))
            cw, ch = crop.size
            if cw == 0 or ch == 0:
                continue
            # Add padding to match pretraining distribution where characters
            # occupy ~1/5th of the image (pad ~2x char size on each side)
            pad_x = cw * 2
            pad_y = ch * 2
            padded = Image.new("RGB", (cw + pad_x * 2, ch + pad_y * 2), bg_color)
            padded.paste(crop, (pad_x, pad_y))
            retina, _, _, _ = scale_and_pad(padded, bg_color)
            img_t = (
                torch.from_numpy(np.array(retina))
                .permute(2, 0, 1)
                .float()
                / 255.0
            )
            imgs.append((img_t, (x1, y1, x2, y2)))

        if not imgs:
            continue

        img_batch = torch.stack([t for t, _ in imgs]).to(device)
        prev_batch = prev_t.expand(img_batch.size(0), -1)

        bbox_pred, class_pred = model(img_batch, prev_batch)

        probs = torch.softmax(class_pred, dim=1)
        class_ids = probs.argmax(dim=1)
        confidences = probs.gather(1, class_ids.unsqueeze(1)).squeeze(1)

        for j, (_, box) in enumerate(imgs):
            cid = class_ids[j].item()
            conf = confidences[j].item()
            if cid != CLASS_NONE:
                results.append((box, cid, conf))

    return results


def _iou(box_a, box_b):
    """Compute intersection-over-union of two (x1, y1, x2, y2) boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms(detections, iou_threshold=0.3):
    """Non-maximum suppression on detections.

    detections: list of (box, class_id, confidence)
    Returns filtered list of (box, class_id) sorted by x1 (left-to-right).
    """
    if not detections:
        return []

    # Sort by confidence descending
    dets = sorted(detections, key=lambda d: d[2], reverse=True)
    keep = []

    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if _iou(best[0], d[0]) < iou_threshold]

    # Sort by x1 for left-to-right reading order
    keep.sort(key=lambda d: d[0][0])
    return [(box, cid) for box, cid, _ in keep]


def _detect_chars_bottom_up(model, line_crop, bg_color, device,
                            gap_tolerance=1, iou_threshold=0.3,
                            confidence_threshold=0.1, batch_size=32):
    """Bottom-up character detection: candidates → classify → NMS.

    Returns [(bbox, class_id), ...] in left-to-right order.
    """
    candidates = _generate_candidate_boxes(line_crop, bg_color,
                                           gap_tolerance=gap_tolerance)
    if not candidates:
        return []

    detections = _classify_candidates_batched(model, line_crop, bg_color,
                                              candidates, device,
                                              batch_size=batch_size)

    # Filter by confidence
    detections = [d for d in detections if d[2] >= confidence_threshold]

    return _nms(detections, iou_threshold=iou_threshold)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def visualize(page_img, page, paragraphs, all_lines, all_chars, text_output,
              output_path, scale=2, all_words=None):
    """Draw ground truth and detections side by side and save."""
    w, h = page_img.size
    sw, sh = w * scale, h * scale

    # Scale up the page image
    page_scaled = page_img.resize((sw, sh), Image.NEAREST)

    # Load a readable font from the project fonts folder
    font_path = "fonts/Arial.ttf"
    font_size = 10 * scale
    font = ImageFont.truetype(font_path, font_size)
    label_size = 14 * scale
    label_font = ImageFont.truetype(font_path, label_size)

    # Left = ground truth, Right = predictions
    combined = Image.new("RGB", (sw * 2, sh), (255, 255, 255))
    combined.paste(page_scaled, (0, 0))
    combined.paste(page_scaled, (sw, 0))
    draw = ImageDraw.Draw(combined)

    s = scale  # shorthand

    # -- Left side: ground truth --
    for para in page.paragraphs:
        b = para["bbox"]
        draw.rectangle((b[0]*s, b[1]*s, b[2]*s, b[3]*s),
                        outline=(255, 0, 0), width=2*s)
        hw_label = "HW" if para.get("is_handwritten") else "PR"
        draw.text((b[0]*s, max(0, b[1]*s - 14*s)), hw_label,
                  fill=(255, 0, 0), font=font)
        for line in para["lines"]:
            b = line["bbox"]
            draw.rectangle((b[0]*s, b[1]*s, b[2]*s, b[3]*s),
                            outline=(0, 0, 255), width=s)
            for word in line.get("words", []):
                b = word["bbox"]
                draw.rectangle((b[0]*s, b[1]*s, b[2]*s, b[3]*s),
                                outline=(255, 165, 0), width=1)
                for ch_data in word["characters"]:
                    b = ch_data["bbox"]
                    draw.rectangle((b[0]*s, b[1]*s, b[2]*s, b[3]*s),
                                    outline=(0, 180, 0), width=1)

    # Labels
    draw.text((10*s, 10*s), "GROUND TRUTH", fill=(255, 255, 255), font=label_font)
    draw.text((10*s-1, 10*s-1), "GROUND TRUTH", fill=(0, 0, 0), font=label_font)

    # -- Right side: predictions --
    ox = sw  # offset for right panel

    for para_bbox, _ in paragraphs:
        shifted = (para_bbox[0]*s + ox, para_bbox[1]*s,
                   para_bbox[2]*s + ox, para_bbox[3]*s)
        draw.rectangle(shifted, outline=(255, 0, 0), width=2*s)

    # text_output entries correspond 1:1 with all_lines
    for idx, ((para_bbox, line_bbox), _) in enumerate(all_lines):
        px1, py1 = para_bbox[0], para_bbox[1]
        page_line = (
            (line_bbox[0] + px1)*s + ox,
            (line_bbox[1] + py1)*s,
            (line_bbox[2] + px1)*s + ox,
            (line_bbox[3] + py1)*s,
        )
        draw.rectangle(page_line, outline=(0, 0, 255), width=s)

        # Draw recognized text above the line
        if idx < len(text_output):
            _, _, text = text_output[idx]
            text_x = page_line[0]
            text_y = max(0, page_line[1] - font_size - 4)
            draw.text((text_x + 1, text_y + 1), text, fill=(0, 0, 0), font=font)
            draw.text((text_x, text_y), text, fill=(0, 180, 0), font=font)

    # Draw word boxes in orange (if available)
    if all_words:
        for page_word_bbox in all_words:
            shifted = (
                page_word_bbox[0]*s + ox,
                page_word_bbox[1]*s,
                page_word_bbox[2]*s + ox,
                page_word_bbox[3]*s,
            )
            draw.rectangle(shifted, outline=(255, 165, 0), width=1)

    for (para_bbox, line_bbox, char_bbox), class_id in all_chars:
        px1, py1 = para_bbox[0], para_bbox[1]
        lx1, ly1 = line_bbox[0], line_bbox[1]
        page_char = (
            (char_bbox[0] + lx1 + px1)*s + ox,
            (char_bbox[1] + ly1 + py1)*s,
            (char_bbox[2] + lx1 + px1)*s + ox,
            (char_bbox[3] + ly1 + py1)*s,
        )
        draw.rectangle(page_char, outline=(0, 180, 0), width=1)
        ch = class_to_char(class_id) if class_id >= CHAR_CLASS_OFFSET else "?"
        bar_h = font_size + 4
        bar_top = max(0, page_char[1] - bar_h)
        draw.rectangle((page_char[0], bar_top, page_char[2], page_char[1]),
                        fill=(0, 180, 0))
        draw.text((page_char[0] + 1, bar_top + 2), ch, fill=(255, 255, 255), font=font)

    draw.text((ox + 10*s, 10*s), "PREDICTIONS", fill=(255, 255, 255), font=label_font)
    draw.text((ox + 10*s-1, 10*s-1), "PREDICTIONS", fill=(0, 0, 0), font=label_font)

    combined.save(output_path)
    print(f"Visualization saved to {output_path} ({combined.size[0]}x{combined.size[1]})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Hierarchical OCR inference")
    parser.add_argument("--model-path", default="model_02.pth")
    parser.add_argument("--output", default="infer_output.png")
    parser.add_argument("--page-width", type=int, default=2048)
    parser.add_argument("--page-height", type=int, default=2800)
    parser.add_argument("--max-detect", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet18", "resnet34", "resnet50"],
                        help="ResNet backbone (must match checkpoint)")
    parser.add_argument("--char-gap-tolerance", type=int, default=1)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.3)
    parser.add_argument("--char-confidence", type=float, default=0.1)
    parser.add_argument("--char-batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model (strict=False allows loading pre-mask-head checkpoints)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    checkpoint = _remap_old_backbone_keys(checkpoint)
    # Infer num_classes from checkpoint to handle old (98) vs new (459) models
    num_classes = checkpoint.get("class_head.2.bias", torch.empty(0)).shape[0]
    if num_classes == 0:
        num_classes = None  # fall back to default
    kwargs = {"backbone": args.backbone}
    if num_classes:
        kwargs["num_classes"] = num_classes
    model = RetinaOCRNet(**kwargs).to(device)
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    model.eval()
    if missing:
        print(f"Loaded model from {args.model_path} (new layers initialized randomly: "
              f"{len(missing)} params)")
    else:
        print(f"Loaded model from {args.model_path}")

    # Discover fonts and generate a page
    fonts = discover_fonts()
    if not fonts:
        print("No fonts found!")
        sys.exit(1)
    page = SyntheticPage(fonts, args.page_width, args.page_height)
    page_img = page.image
    bg_color = page.bg_color
    print(f"Generated page: {len(page.paragraphs)} paragraphs, bg={bg_color}")

    # --- Level 1: detect paragraphs on the page ---
    # gap_tolerance bridges inter-line gaps within a paragraph (~20px)
    # but not inter-paragraph gaps (~35+px)
    paragraphs = detect_level(model, page_img, bg_color, device, args.max_detect,
                              gap_tolerance=30)
    print(f"Detected {len(paragraphs)} paragraphs")

    # Collect all detections for visualization and text output
    all_lines = []   # ((para_bbox, line_bbox), class_id)
    all_chars = []    # ((para_bbox, line_bbox, char_bbox), class_id)
    text_output = []  # [(para_idx, line_idx, text)]

    for pi, (para_bbox, _) in enumerate(paragraphs):
        para_crop = page_img.crop(para_bbox)

        # --- Level 2: detect lines within this paragraph ---
        lines = detect_level(model, para_crop, bg_color, device, args.max_detect,
                             gap_tolerance=5)
        print(f"  Paragraph {pi+1}: {len(lines)} lines")

        for li, (line_bbox, _) in enumerate(lines):
            all_lines.append(((para_bbox, line_bbox), CLASS_LINE))
            line_crop = para_crop.crop(line_bbox)

            # --- Level 3: detect characters within this line ---
            # Bottom-up: find candidates, classify individually, NMS
            chars = _detect_chars_bottom_up(
                model, line_crop, bg_color, device,
                gap_tolerance=args.char_gap_tolerance,
                iou_threshold=args.nms_iou_threshold,
                confidence_threshold=args.char_confidence,
                batch_size=args.char_batch_size,
            )

            line_text = []
            for char_bbox, class_id in chars:
                all_chars.append(((para_bbox, line_bbox, char_bbox), class_id))
                if class_id >= CHAR_CLASS_OFFSET:
                    line_text.append(class_to_char(class_id))
                else:
                    line_text.append("?")

            text_output.append((pi, li, "".join(line_text)))

    # Print ground truth text
    print("\n--- Ground Truth ---")
    for pi, para in enumerate(page.paragraphs):
        print(f"Paragraph {pi+1}:")
        for li, line in enumerate(para["lines"]):
            gt_text = " ".join(
                "".join(ch["char"] for ch in word["characters"])
                for word in line["words"]
            )
            print(f"  Line {li+1}: {gt_text}")

    # Print detected text
    print("\n--- Detected Text ---")
    if not text_output:
        print("  (no text detected)")
    current_para = -1
    for pi, li, text in text_output:
        if pi != current_para:
            current_para = pi
            print(f"Paragraph {pi+1}:")
        print(f"  Line {li+1}: {text}")

    # Visualize
    visualize(page_img, page, paragraphs, all_lines, all_chars, text_output,
              args.output)


if __name__ == "__main__":
    main()
