"""
Training ratchet: model-in-the-loop hierarchical correction with GPT-4o.

The model proposes detections at each hierarchy level (page → paragraphs →
lines → words → characters), GPT-4o corrects them, and the corrections
become training data. Repeat: fine-tune on corrections, run ratchet again
with the improved model — 4o corrections shrink as the model improves.

Output format matches annotate_real.py (page.png, annotations.json, masks/)
so existing load_annotations() works unchanged.

Usage:
  python ratchet.py image1.png image2.png \
      --model-path model_04.pth \
      --output-dir ratchet_annotations/ \
      --visualize
"""

import os
import sys
import json
import time
import argparse
import logging

import numpy as np
import torch
from PIL import Image, ImageDraw
from openai import OpenAI
from dotenv import load_dotenv

from generate_training_data import (
    CHAR_TO_CLASS,
    RETINA_SIZE,
    _compute_mask,
    contour_from_mask,
)
from infer_04 import generate_contours, contour_to_bbox
from train_04 import ContourOCRNet
from train_02 import _remap_old_backbone_keys
from annotate_real import (
    encode_image_base64,
    estimate_background_color,
    clamp_bbox,
    save_annotations,
    visualize_annotations,
    PARAGRAPH_SCHEMA,
    LINE_SCHEMA,
    WORD_SCHEMA,
    CHARACTER_SCHEMA,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token tracker
# ---------------------------------------------------------------------------
class TokenTracker:
    """Accumulate prompt/completion tokens and wall-clock time per API call."""

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_wall_time = 0.0
        self.call_count = 0

    def record(self, usage, wall_time):
        """Record tokens from an API response usage object."""
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.total_wall_time += wall_time
        self.call_count += 1

        total_tok = self.prompt_tokens + self.completion_tokens
        tok_s = total_tok / self.total_wall_time if self.total_wall_time > 0 else 0
        log.info(
            "  [tok] call=%d  prompt=%d  completion=%d  wall=%.1fs  "
            "running: %d tok total, %.0f tok/s",
            self.call_count, usage.prompt_tokens, usage.completion_tokens,
            wall_time, total_tok, tok_s,
        )

    def summary(self):
        total_tok = self.prompt_tokens + self.completion_tokens
        tok_s = total_tok / self.total_wall_time if self.total_wall_time > 0 else 0
        # GPT-4o pricing: $2.50/1M input, $10/1M output
        cost = (self.prompt_tokens * 2.50 + self.completion_tokens * 10.0) / 1_000_000
        log.info(
            "=== Token Summary ===\n"
            "  Calls: %d\n"
            "  Prompt tokens: %d\n"
            "  Completion tokens: %d\n"
            "  Total tokens: %d\n"
            "  Wall time: %.1fs\n"
            "  Avg throughput: %.0f tok/s\n"
            "  Estimated cost: $%.4f",
            self.call_count, self.prompt_tokens, self.completion_tokens,
            total_tok, self.total_wall_time, tok_s, cost,
        )


# ---------------------------------------------------------------------------
# GPT-4o correction at each hierarchy level
# ---------------------------------------------------------------------------
def _call_4o(client, image, system_prompt, user_prompt, schema, tracker,
             detail="high", max_retries=3):
    """Call GPT-4o with image + structured output, tracking tokens."""
    b64 = encode_image_base64(image)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": detail,
                    },
                },
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    for attempt in range(max_retries):
        try:
            t0 = time.monotonic()
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": schema,
                },
                max_tokens=4096,
                temperature=0.0,
            )
            wall = time.monotonic() - t0
            if resp.usage and tracker:
                tracker.record(resp.usage, wall)
            content = resp.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            wait = 2 ** attempt
            log.warning("GPT-4o call failed (attempt %d/%d): %s — retrying in %ds",
                        attempt + 1, max_retries, e, wait)
            time.sleep(wait)

    log.error("GPT-4o call failed after %d retries", max_retries)
    return None


def _contours_description(contours):
    """Format model contours as a compact JSON string for the 4o prompt."""
    result = []
    for contour, class_id in contours:
        bbox = contour_to_bbox(contour)
        result.append({"bbox": list(bbox), "contour_points": len(contour)})
    return json.dumps(result)


def correct_paragraphs(client, page_crop, model_contours, tracker):
    """4o corrects paragraph bboxes given model's predictions."""
    w, h = page_crop.size
    model_desc = _contours_description(model_contours)
    system_prompt = (
        "You are a document layout analyzer correcting paragraph detections. "
        "The local model predicted paragraph regions — some may be wrong, "
        "missing, or poorly bounded. Return corrected paragraph bounding boxes."
    )
    user_prompt = (
        f"Image is {w}x{h} pixels. The model predicted these paragraphs:\n"
        f"{model_desc}\n\n"
        f"Correct the paragraph bounding boxes. Add any missed paragraphs, "
        f"remove false positives, and adjust bounding boxes for accuracy. "
        f"Return [x1,y1,x2,y2] bboxes in top-to-bottom reading order."
    )
    result = _call_4o(client, page_crop, system_prompt, user_prompt,
                      PARAGRAPH_SCHEMA, tracker, detail="low")
    if result is None:
        return []
    bboxes = []
    for p in result.get("paragraphs", []):
        bbox = p.get("bbox")
        if bbox and len(bbox) == 4:
            bboxes.append(clamp_bbox([int(v) for v in bbox], w, h))
    return bboxes


def correct_lines(client, para_crop, model_contours, tracker):
    """4o corrects line bboxes given model's predictions."""
    w, h = para_crop.size
    model_desc = _contours_description(model_contours)
    system_prompt = (
        "You are a document layout analyzer correcting line detections. "
        "The local model predicted text lines — correct any errors."
    )
    user_prompt = (
        f"Image is {w}x{h} pixels. The model predicted these lines:\n"
        f"{model_desc}\n\n"
        f"Correct the line bounding boxes. Return [x1,y1,x2,y2] bboxes "
        f"in top-to-bottom order."
    )
    result = _call_4o(client, para_crop, system_prompt, user_prompt,
                      LINE_SCHEMA, tracker, detail="high")
    if result is None:
        return []
    bboxes = []
    for ln in result.get("lines", []):
        bbox = ln.get("bbox")
        if bbox and len(bbox) == 4:
            bboxes.append(clamp_bbox([int(v) for v in bbox], w, h))
    return bboxes


def correct_words(client, line_crop, model_contours, tracker):
    """4o corrects word bboxes given model's predictions."""
    w, h = line_crop.size
    model_desc = _contours_description(model_contours)
    system_prompt = (
        "You are a document layout analyzer correcting word detections. "
        "The local model predicted words — correct any errors."
    )
    user_prompt = (
        f"Image is {w}x{h} pixels. The model predicted these words:\n"
        f"{model_desc}\n\n"
        f"Correct the word bounding boxes. Return [x1,y1,x2,y2] bboxes "
        f"in left-to-right order."
    )
    result = _call_4o(client, line_crop, system_prompt, user_prompt,
                      WORD_SCHEMA, tracker, detail="high")
    if result is None:
        return []
    bboxes = []
    for wd in result.get("words", []):
        bbox = wd.get("bbox")
        if bbox and len(bbox) == 4:
            bboxes.append(clamp_bbox([int(v) for v in bbox], w, h))
    return bboxes


def correct_characters(client, word_crop, model_contours, tracker):
    """4o corrects character bboxes+labels given model's predictions."""
    w, h = word_crop.size
    # Include model's char predictions in description
    model_desc = []
    for contour, class_id in model_contours:
        from generate_training_data import CLASS_TO_CHAR
        bbox = contour_to_bbox(contour)
        ch = CLASS_TO_CHAR.get(class_id, "?")
        model_desc.append({"bbox": list(bbox), "char": ch})
    model_str = json.dumps(model_desc)

    system_prompt = (
        "You are an OCR character detector correcting character detections. "
        "The local model predicted characters — correct bboxes and labels."
    )
    user_prompt = (
        f"Image is {w}x{h} pixels. The model predicted:\n{model_str}\n\n"
        f"Correct the character bounding boxes and labels. "
        f"Only ASCII printable characters (codes 32-126). "
        f"Return bbox + char for each, left-to-right order."
    )
    result = _call_4o(client, word_crop, system_prompt, user_prompt,
                      CHARACTER_SCHEMA, tracker, detail="high")
    if result is None:
        return []
    chars = []
    for ch in result.get("characters", []):
        bbox = ch.get("bbox")
        char = ch.get("char", "")
        if bbox and len(bbox) == 4 and len(char) == 1 and char in CHAR_TO_CLASS:
            chars.append({
                "bbox": clamp_bbox([int(v) for v in bbox], w, h),
                "char": char,
            })
    return chars


# ---------------------------------------------------------------------------
# Ratchet pipeline
# ---------------------------------------------------------------------------
def run_ratchet(model, client, image_path, device, tracker,
                closure_threshold=15.0):
    """Run hierarchical inference with 4o correction at each level.

    Returns nested annotation dict matching annotate_real.py format.
    """
    page_img = Image.open(image_path).convert("RGB")
    bg_color = estimate_background_color(page_img)
    log.info("Image: %s (%dx%d), bg=%s",
             image_path, page_img.size[0], page_img.size[1], bg_color)

    model.eval()

    # Level 0: page region detection
    page_regions = generate_contours(
        model, page_img, bg_color, 0, device,
        closure_threshold=closure_threshold,
    )
    log.info("Model detected %d page region(s)", len(page_regions))

    # For page-level, we use the model's page crop as-is (no 4o correction
    # on the page contour — it just defines the region to zoom into)
    if not page_regions:
        # Fallback: use entire image as page
        w, h = page_img.size
        page_regions = [([(0, 0), (w, 0), (w, h), (0, h)], 0)]
        log.info("No page regions detected, using full image")

    all_paragraphs = []

    for ri, (page_contour, _) in enumerate(page_regions):
        page_bbox = contour_to_bbox(page_contour)
        page_crop = page_img.crop(page_bbox)
        pgx, pgy = page_bbox[0], page_bbox[1]

        # Level 1: model detects paragraphs in page crop
        model_paras = generate_contours(
            model, page_crop, bg_color, 1, device,
            closure_threshold=closure_threshold,
        )
        log.info("  Region %d: model detected %d paragraphs", ri + 1, len(model_paras))

        # 4o corrects paragraph bboxes
        corrected_para_bboxes = correct_paragraphs(
            client, page_crop, model_paras, tracker,
        )
        log.info("  Region %d: 4o corrected to %d paragraphs",
                 ri + 1, len(corrected_para_bboxes))

        for pi, pb in enumerate(corrected_para_bboxes):
            # Use model's crop for next level (model drives hierarchy)
            # but store 4o's corrected bbox as ground truth
            para_crop = page_crop.crop(pb)
            pb_global = [pb[0] + pgx, pb[1] + pgy, pb[2] + pgx, pb[3] + pgy]

            # Level 2: model detects lines in paragraph crop
            model_lines = generate_contours(
                model, para_crop, bg_color, 2, device,
                closure_threshold=closure_threshold,
            )
            log.info("    Para %d: model detected %d lines", pi + 1, len(model_lines))

            corrected_line_bboxes = correct_lines(
                client, para_crop, model_lines, tracker,
            )
            log.info("    Para %d: 4o corrected to %d lines",
                     pi + 1, len(corrected_line_bboxes))

            para_lines = []
            for li, lb in enumerate(corrected_line_bboxes):
                line_crop = para_crop.crop(lb)
                lb_global = [
                    lb[0] + pb[0] + pgx, lb[1] + pb[1] + pgy,
                    lb[2] + pb[0] + pgx, lb[3] + pb[1] + pgy,
                ]

                # Level 3: model detects words in line crop
                model_words = generate_contours(
                    model, line_crop, bg_color, 3, device,
                    closure_threshold=closure_threshold,
                )

                corrected_word_bboxes = correct_words(
                    client, line_crop, model_words, tracker,
                )
                log.info("      Line %d: %d words (model) → %d (4o)",
                         li + 1, len(model_words), len(corrected_word_bboxes))

                line_words = []
                for wi, wb in enumerate(corrected_word_bboxes):
                    word_crop = line_crop.crop(wb)
                    wb_global = [
                        wb[0] + lb[0] + pb[0] + pgx,
                        wb[1] + lb[1] + pb[1] + pgy,
                        wb[2] + lb[0] + pb[0] + pgx,
                        wb[3] + lb[1] + pb[1] + pgy,
                    ]

                    # Level 4: model detects characters in word crop
                    model_chars = generate_contours(
                        model, word_crop, bg_color, 4, device,
                        closure_threshold=closure_threshold,
                    )

                    corrected_chars = correct_characters(
                        client, word_crop, model_chars, tracker,
                    )

                    # Translate char bboxes to page-global
                    word_characters = []
                    for ch in corrected_chars:
                        cb = ch["bbox"]
                        cb_global = [
                            cb[0] + wb[0] + lb[0] + pb[0] + pgx,
                            cb[1] + wb[1] + lb[1] + pb[1] + pgy,
                            cb[2] + wb[0] + lb[0] + pb[0] + pgx,
                            cb[3] + wb[1] + lb[1] + pb[1] + pgy,
                        ]
                        word_characters.append({
                            "bbox": cb_global,
                            "char": ch["char"],
                        })

                    line_words.append({
                        "bbox": wb_global,
                        "characters": word_characters,
                    })

                para_lines.append({
                    "bbox": lb_global,
                    "words": line_words,
                })

            all_paragraphs.append({
                "bbox": pb_global,
                "is_handwritten": False,
                "lines": para_lines,
            })

    nested = {"paragraphs": all_paragraphs}

    # Compute masks
    from annotate_real import build_annotated_page
    build_annotated_page(page_img, nested, bg_color=bg_color)

    return page_img, nested, bg_color


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Training ratchet: model proposes, GPT-4o corrects",
    )
    parser.add_argument("images", nargs="+", help="Document image paths")
    parser.add_argument("--model-path", default="model_04.pth")
    parser.add_argument("--output-dir", default="ratchet_annotations/")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--closure-threshold", type=float, default=15.0)

    # Model architecture (must match checkpoint)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=256)

    args = parser.parse_args()

    # Load API key
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY not found in environment or .env file")
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

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
        log.info("Loaded model from %s", args.model_path)
    except FileNotFoundError:
        log.warning("%s not found, using random weights", args.model_path)

    tracker = TokenTracker()

    for image_path in args.images:
        if not os.path.isfile(image_path):
            log.error("File not found: %s", image_path)
            continue

        log.info("Processing: %s", image_path)
        page_img, nested, bg_color = run_ratchet(
            model, client, image_path, device, tracker,
            closure_threshold=args.closure_threshold,
        )

        # Save in annotate_real.py format
        basename = os.path.splitext(os.path.basename(image_path))[0]
        out_dir = os.path.join(args.output_dir, basename)
        save_annotations(out_dir, image_path, nested, page_img, bg_color)

        if args.visualize:
            visualize_annotations(page_img, nested, out_dir)

        # Summary
        n_para = len(nested.get("paragraphs", []))
        n_lines = sum(
            len(p.get("lines", []))
            for p in nested.get("paragraphs", [])
        )
        n_words = sum(
            len(l.get("words", []))
            for p in nested.get("paragraphs", [])
            for l in p.get("lines", [])
        )
        n_chars = sum(
            len(w.get("characters", []))
            for p in nested.get("paragraphs", [])
            for l in p.get("lines", [])
            for w in l.get("words", [])
        )
        log.info(
            "Done: %s — %d paragraphs, %d lines, %d words, %d characters",
            image_path, n_para, n_lines, n_words, n_chars,
        )

    tracker.summary()
    log.info("All images processed.")


if __name__ == "__main__":
    main()
