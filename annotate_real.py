"""
Annotate real document images using OpenAI GPT-4o vision API.

Hierarchical vision cascade: crops parent regions and sends each level
to GPT-4o independently, mirroring the model's retina-based detection:
    Full page → paragraphs → lines → words → characters.

Output format matches SyntheticPage hierarchy for training integration:
  paragraphs → lines → words → characters, each with bbox, mask, and char label.
  Paragraphs carry an is_handwritten flag.

Usage:
  .venv/bin/python annotate_real.py image1.png [image2.jpg ...] \\
      --output-dir real_annotations/ \\
      --bg-color 255,255,255 \\
      --visualize

  # Force handwritten label on all paragraphs:
  .venv/bin/python annotate_real.py handwriting.png --handwritten
"""

import os
import sys
import json
import time
import base64
import logging
import argparse
import shutil
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw
from openai import OpenAI
from dotenv import load_dotenv

from generate_training_data import CHAR_TO_CLASS, _compute_mask, contour_from_mask

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token throughput tracking
# ---------------------------------------------------------------------------
class TokenTracker:
    """Accumulates per-call token counts and wall-clock time."""

    # GPT-4o pricing (per 1M tokens)
    INPUT_COST_PER_M = 2.50
    OUTPUT_COST_PER_M = 10.00

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self.start_time = time.time()

    def record(self, usage):
        """Record usage from an API response."""
        if usage is None:
            return
        self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        self.calls += 1
        elapsed = time.time() - self.start_time
        total = self.prompt_tokens + self.completion_tokens
        tok_s = total / elapsed if elapsed > 0 else 0
        log.info(
            "  tokens so far: %d prompt + %d completion = %d total  (%.0f tok/s)",
            self.prompt_tokens, self.completion_tokens, total, tok_s,
        )

    def summary(self):
        """Print final summary."""
        elapsed = time.time() - self.start_time
        total = self.prompt_tokens + self.completion_tokens
        tok_s = total / elapsed if elapsed > 0 else 0
        cost = (
            self.prompt_tokens / 1e6 * self.INPUT_COST_PER_M
            + self.completion_tokens / 1e6 * self.OUTPUT_COST_PER_M
        )
        log.info("=== Token usage summary ===")
        log.info("  API calls:        %d", self.calls)
        log.info("  Prompt tokens:    %d", self.prompt_tokens)
        log.info("  Completion tokens: %d", self.completion_tokens)
        log.info("  Total tokens:     %d", total)
        log.info("  Wall time:        %.1fs", elapsed)
        log.info("  Throughput:       %.0f tok/s", tok_s)
        log.info("  Est. cost:        $%.4f", cost)


# ---------------------------------------------------------------------------
# Structured-output JSON schemas (OpenAI strict mode)
# ---------------------------------------------------------------------------
PARAGRAPH_SCHEMA = {
    "name": "paragraph_detection",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "paragraphs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "bbox": {
                            "type": "array",
                            "items": {"type": "integer"},
                        }
                    },
                    "required": ["bbox"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["paragraphs"],
        "additionalProperties": False,
    },
}

LINE_SCHEMA = {
    "name": "line_detection",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "lines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "bbox": {
                            "type": "array",
                            "items": {"type": "integer"},
                        }
                    },
                    "required": ["bbox"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["lines"],
        "additionalProperties": False,
    },
}

CHARACTER_SCHEMA = {
    "name": "character_detection",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "characters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "bbox": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                        "char": {"type": "string"},
                    },
                    "required": ["bbox", "char"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["characters"],
        "additionalProperties": False,
    },
}

WORD_SCHEMA = {
    "name": "word_detection",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "words": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "bbox": {
                            "type": "array",
                            "items": {"type": "integer"},
                        }
                    },
                    "required": ["bbox"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["words"],
        "additionalProperties": False,
    },
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def encode_image_base64(image, max_dim=2048):
    """Encode a PIL Image as a base64 JPEG string, resizing if needed."""
    w, h = image.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def estimate_background_color(image, border_width=10):
    """Estimate background color from the median of border pixels."""
    arr = np.array(image.convert("RGB"))
    h, w = arr.shape[:2]
    bw = min(border_width, h // 2, w // 2)
    if bw < 1:
        return (255, 255, 255)
    border_pixels = np.concatenate([
        arr[:bw, :].reshape(-1, 3),       # top
        arr[-bw:, :].reshape(-1, 3),      # bottom
        arr[bw:-bw, :bw].reshape(-1, 3),  # left
        arr[bw:-bw, -bw:].reshape(-1, 3), # right
    ], axis=0)
    median = np.median(border_pixels, axis=0).astype(int)
    return tuple(median.tolist())


def clamp_bbox(bbox, width, height):
    """Clamp bbox to image bounds and ensure positive dimensions."""
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return [x1, y1, x2, y2]


def validate_bbox(bbox, width, height):
    """Check that a bbox is valid (4 ints, positive area, within bounds)."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
    except (ValueError, TypeError):
        return False
    if x2 <= x1 or y2 <= y1:
        return False
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        return False
    return True


def call_gpt4o_vision(client, image, system_prompt, user_prompt,
                       response_schema, max_retries=3, detail="high",
                       tracker=None):
    """Call GPT-4o with an image and structured output schema.

    Returns parsed JSON dict or None on failure.
    If tracker is provided, records token usage from each successful call.
    """
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
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": response_schema,
                },
                max_tokens=16384,
                temperature=0.0,
            )
            if tracker is not None:
                tracker.record(resp.usage)
            content = resp.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            wait = 2 ** attempt
            log.warning("GPT-4o call failed (attempt %d/%d): %s — retrying in %ds",
                        attempt + 1, max_retries, e, wait)
            time.sleep(wait)

    log.error("GPT-4o call failed after %d retries", max_retries)
    return None


# ---------------------------------------------------------------------------
# Hierarchical vision cascade
# ---------------------------------------------------------------------------

def detect_paragraphs(client, page_image, tracker=None):
    """Detect paragraph bounding boxes on a full page image.

    Returns list of dicts: [{"bbox": [x1,y1,x2,y2]}, ...]
    """
    w, h = page_image.size
    system_prompt = (
        "You are a document layout analyzer. You detect paragraph blocks in "
        "document images. Return precise pixel-coordinate bounding boxes."
    )
    user_prompt = (
        f"Detect all paragraph blocks in this document. A paragraph is a "
        f"contiguous block of text separated by vertical whitespace. "
        f"Image is {w}x{h} pixels. Return [x1,y1,x2,y2] bboxes in "
        f"top-to-bottom reading order."
    )

    result = call_gpt4o_vision(
        client, page_image, system_prompt, user_prompt,
        PARAGRAPH_SCHEMA, detail="low", tracker=tracker,
    )
    if result is None:
        log.warning("Paragraph detection failed")
        return []

    paragraphs = []
    for p in result.get("paragraphs", []):
        bbox = p.get("bbox")
        if bbox and len(bbox) == 4:
            bbox = clamp_bbox([int(v) for v in bbox], w, h)
            paragraphs.append({"bbox": bbox})
    log.info("Detected %d paragraphs", len(paragraphs))
    return paragraphs


def detect_lines(client, paragraph_crop, tracker=None):
    """Detect line bounding boxes within a paragraph crop.

    Returns list of dicts: [{"bbox": [x1,y1,x2,y2]}, ...]
    """
    w, h = paragraph_crop.size
    system_prompt = (
        "You are a document layout analyzer. You detect individual text lines "
        "within a paragraph. Return precise pixel-coordinate bounding boxes."
    )
    user_prompt = (
        f"Detect all text lines in this paragraph crop. Image is {w}x{h} "
        f"pixels. Return [x1,y1,x2,y2] bboxes in top-to-bottom order."
    )

    result = call_gpt4o_vision(
        client, paragraph_crop, system_prompt, user_prompt,
        LINE_SCHEMA, detail="high", tracker=tracker,
    )
    if result is None:
        log.warning("Line detection failed for paragraph crop")
        return []

    lines = []
    for ln in result.get("lines", []):
        bbox = ln.get("bbox")
        if bbox and len(bbox) == 4:
            bbox = clamp_bbox([int(v) for v in bbox], w, h)
            lines.append({"bbox": bbox})
    log.info("  Detected %d lines", len(lines))
    return lines


def detect_characters(client, line_crop, tracker=None):
    """Detect character bounding boxes and labels within a line crop.

    Returns list of dicts: [{"bbox": [x1,y1,x2,y2], "char": "A"}, ...]
    """
    w, h = line_crop.size
    system_prompt = (
        "You are an OCR character detector. You identify every individual "
        "character and its bounding box in a text line image. Return precise "
        "pixel-coordinate bounding boxes and the character label."
    )
    user_prompt = (
        f"Identify every character and its bounding box. Image is {w}x{h} "
        f"pixels. Return bbox + char for each, left-to-right order. "
        f"Only ASCII printable characters (codes 32-126). "
        f"For spaces between words, include them as a space character."
    )

    result = call_gpt4o_vision(
        client, line_crop, system_prompt, user_prompt,
        CHARACTER_SCHEMA, detail="high", tracker=tracker,
    )
    if result is None:
        log.warning("Character detection failed for line crop")
        return []

    chars = []
    for ch in result.get("characters", []):
        bbox = ch.get("bbox")
        char = ch.get("char", "")
        if bbox and len(bbox) == 4 and len(char) == 1:
            # Filter to supported characters
            if char in CHAR_TO_CLASS:
                bbox = clamp_bbox([int(v) for v in bbox], w, h)
                chars.append({"bbox": bbox, "char": char})
    log.info("    Detected %d characters", len(chars))
    return chars


def detect_words(client, line_crop, tracker=None):
    """Detect word bounding boxes within a line crop.

    Returns list of dicts: [{"bbox": [x1,y1,x2,y2]}, ...]
    """
    w, h = line_crop.size
    system_prompt = (
        "You are a document layout analyzer. You detect individual words "
        "within a text line. Return precise pixel-coordinate bounding boxes."
    )
    user_prompt = (
        f"Detect all words in this text line crop. Image is {w}x{h} "
        f"pixels. Return [x1,y1,x2,y2] bboxes in left-to-right order. "
        f"A word is a contiguous group of characters separated by spaces."
    )

    result = call_gpt4o_vision(
        client, line_crop, system_prompt, user_prompt,
        WORD_SCHEMA, detail="high", tracker=tracker,
    )
    if result is None:
        log.warning("Word detection failed for line crop")
        return []

    words = []
    for wd in result.get("words", []):
        bbox = wd.get("bbox")
        if bbox and len(bbox) == 4:
            bbox = clamp_bbox([int(v) for v in bbox], w, h)
            words.append({"bbox": bbox})
    log.info("    Detected %d words", len(words))
    return words


def build_cascade_annotations(client, page_image, rate_limit_delay=0.5,
                               tracker=None):
    """Orchestrate the hierarchical vision cascade (4-level).

    Calls detect_paragraphs → detect_lines → detect_words → detect_characters.
    Translates crop-local coords to page-global coords.

    Returns flat list with para_idx, line_idx, word_idx tracking.
    """
    flat = []
    next_id = 0

    # Detect paragraphs
    paragraphs = detect_paragraphs(client, page_image, tracker=tracker)
    time.sleep(rate_limit_delay)

    for pi, para in enumerate(paragraphs):
        pb = para["bbox"]
        flat.append({
            "id": next_id, "type": "paragraph",
            "bbox": pb, "text": "", "is_handwritten": False,
            "para_idx": pi, "line_idx": None, "word_idx": None,
        })
        next_id += 1

        # Crop paragraph region
        para_crop = page_image.crop(pb)

        # Detect lines within paragraph
        lines = detect_lines(client, para_crop, tracker=tracker)
        time.sleep(rate_limit_delay)

        for li, line in enumerate(lines):
            lb = line["bbox"]
            lb_global = [
                lb[0] + pb[0], lb[1] + pb[1],
                lb[2] + pb[0], lb[3] + pb[1],
            ]
            flat.append({
                "id": next_id, "type": "line",
                "bbox": lb_global, "text": "", "is_handwritten": False,
                "para_idx": pi, "line_idx": li, "word_idx": None,
            })
            next_id += 1

            # Crop line region
            line_crop = para_crop.crop(lb)

            # Detect words within line
            words = detect_words(client, line_crop, tracker=tracker)
            time.sleep(rate_limit_delay)

            for wi, word in enumerate(words):
                wb = word["bbox"]
                wb_global = [
                    wb[0] + lb[0] + pb[0], wb[1] + lb[1] + pb[1],
                    wb[2] + lb[0] + pb[0], wb[3] + lb[1] + pb[1],
                ]
                flat.append({
                    "id": next_id, "type": "word",
                    "bbox": wb_global, "text": "", "is_handwritten": False,
                    "para_idx": pi, "line_idx": li, "word_idx": wi,
                })
                next_id += 1

                # Crop word region
                word_crop = line_crop.crop(wb)

                # Detect characters within word
                chars = detect_characters(client, word_crop, tracker=tracker)
                time.sleep(rate_limit_delay)

                for ch in chars:
                    cb = ch["bbox"]
                    cb_global = [
                        cb[0] + wb[0] + lb[0] + pb[0],
                        cb[1] + wb[1] + lb[1] + pb[1],
                        cb[2] + wb[0] + lb[0] + pb[0],
                        cb[3] + wb[1] + lb[1] + pb[1],
                    ]
                    flat.append({
                        "id": next_id, "type": "character",
                        "bbox": cb_global, "text": ch["char"],
                        "is_handwritten": False,
                        "para_idx": pi, "line_idx": li, "word_idx": wi,
                    })
                    next_id += 1

    log.info("Built %d cascade annotations total", len(flat))
    return flat


# ---------------------------------------------------------------------------
# Flat-to-nested conversion
# ---------------------------------------------------------------------------

def flat_to_nested_local(flat_annotations):
    """Deterministic local fallback: group cascade annotations into 4-level hierarchy
    using para_idx/line_idx/word_idx from the cascade.
    """
    paragraphs_map = {}
    lines_map = {}
    words_map = {}
    chars_list = []

    for ann in flat_annotations:
        if ann["type"] == "paragraph":
            pi = ann["para_idx"]
            paragraphs_map[pi] = {
                "bbox": ann["bbox"],
                "is_handwritten": ann.get("is_handwritten", False),
                "lines": [],
            }
        elif ann["type"] == "line":
            pi = ann["para_idx"]
            li = ann["line_idx"]
            lines_map[(pi, li)] = {"bbox": ann["bbox"], "words": []}
        elif ann["type"] == "word":
            pi = ann["para_idx"]
            li = ann["line_idx"]
            wi = ann["word_idx"]
            words_map[(pi, li, wi)] = {"bbox": ann["bbox"], "characters": []}
        elif ann["type"] == "character":
            chars_list.append(ann)

    # Assign characters to words
    for ch in chars_list:
        key = (ch["para_idx"], ch["line_idx"], ch["word_idx"])
        if key in words_map:
            words_map[key]["characters"].append({
                "bbox": ch["bbox"], "char": ch.get("text", ch.get("char", "")),
            })

    # Assign words to lines
    for (pi, li, wi), word_data in sorted(words_map.items()):
        key = (pi, li)
        if key in lines_map:
            lines_map[key]["words"].append(word_data)

    # Assign lines to paragraphs
    for (pi, li), line_data in sorted(lines_map.items()):
        if pi in paragraphs_map:
            paragraphs_map[pi]["lines"].append(line_data)

    nested = {
        "paragraphs": [
            paragraphs_map[k] for k in sorted(paragraphs_map.keys())
        ]
    }
    return nested


# ---------------------------------------------------------------------------
# Mask computation + output
# ---------------------------------------------------------------------------

def build_annotated_page(page_image, nested, bg_color=None):
    """Add pixel masks to each element in the nested annotation.

    Modifies nested in-place, adding "mask" keys with PIL Image values.
    Handles 4-level hierarchy: paragraphs → lines → words → characters.
    Returns nested dict.
    """
    if bg_color is None:
        bg_color = estimate_background_color(page_image)

    for para in nested.get("paragraphs", []):
        bbox = tuple(para["bbox"])
        para["mask"] = _compute_mask(page_image, bbox, bg_color)
        for line in para.get("lines", []):
            bbox = tuple(line["bbox"])
            line["mask"] = _compute_mask(page_image, bbox, bg_color)
            for word in line.get("words", []):
                bbox = tuple(word["bbox"])
                word["mask"] = _compute_mask(page_image, bbox, bg_color)
                for char in word.get("characters", []):
                    bbox = tuple(char["bbox"])
                    char["mask"] = _compute_mask(page_image, bbox, bg_color)

    return nested


def save_annotations(output_path, image_path, nested, page_image, bg_color):
    """Save annotations to disk.

    Creates:
      output_path/
        annotations.json
        page.png
        masks/
          para_0_mask.png
          line_0_0_mask.png
          word_0_0_0_mask.png
          char_0_0_0_0_mask.png
    """
    os.makedirs(output_path, exist_ok=True)
    masks_dir = os.path.join(output_path, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    # Save original image
    page_path = os.path.join(output_path, "page.png")
    page_image.convert("RGB").save(page_path)

    # Build JSON-serializable structure and save masks
    json_data = {"paragraphs": [], "image_path": os.path.basename(image_path)}

    for pi, para in enumerate(nested.get("paragraphs", [])):
        para_json = {
            "bbox": para["bbox"],
            "is_handwritten": para.get("is_handwritten", False),
            "lines": [],
        }

        # Save paragraph mask
        mask = para.get("mask")
        if mask is not None:
            mask_name = f"para_{pi}_mask.png"
            mask.save(os.path.join(masks_dir, mask_name))
            para_json["mask_path"] = f"masks/{mask_name}"

        for li, line in enumerate(para.get("lines", [])):
            line_json = {
                "bbox": line["bbox"],
                "words": [],
            }

            # Save line mask
            mask = line.get("mask")
            if mask is not None:
                mask_name = f"line_{pi}_{li}_mask.png"
                mask.save(os.path.join(masks_dir, mask_name))
                line_json["mask_path"] = f"masks/{mask_name}"

            for wi, word in enumerate(line.get("words", [])):
                word_json = {
                    "bbox": word["bbox"],
                    "characters": [],
                }

                # Save word mask
                mask = word.get("mask")
                if mask is not None:
                    mask_name = f"word_{pi}_{li}_{wi}_mask.png"
                    mask.save(os.path.join(masks_dir, mask_name))
                    word_json["mask_path"] = f"masks/{mask_name}"

                for ci, char in enumerate(word.get("characters", [])):
                    char_json = {
                        "bbox": char["bbox"],
                        "char": char["char"],
                    }

                    # Save character mask
                    mask = char.get("mask")
                    if mask is not None:
                        mask_name = f"char_{pi}_{li}_{wi}_{ci}_mask.png"
                        mask.save(os.path.join(masks_dir, mask_name))
                        char_json["mask_path"] = f"masks/{mask_name}"

                    word_json["characters"].append(char_json)

                line_json["words"].append(word_json)

            para_json["lines"].append(line_json)

        json_data["paragraphs"].append(para_json)

    json_path = os.path.join(output_path, "annotations.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    log.info("Saved annotations to %s", output_path)


def load_annotations(annotation_dir):
    """Load annotations from a saved directory for training integration.

    Returns (page_image, nested_dict) where nested_dict has the same
    structure as SyntheticPage.paragraphs with PIL mask objects.
    Supports 4-level hierarchy: paragraphs → lines → words → characters.
    """
    page_path = os.path.join(annotation_dir, "page.png")
    json_path = os.path.join(annotation_dir, "annotations.json")

    page_image = Image.open(page_path).convert("RGB")
    with open(json_path) as f:
        data = json.load(f)

    # Reconstruct masks from files
    for para in data.get("paragraphs", []):
        mp = para.pop("mask_path", None)
        if mp:
            para["mask"] = Image.open(
                os.path.join(annotation_dir, mp)
            ).convert("L")
        para["bbox"] = tuple(para["bbox"])
        # Ensure is_handwritten exists
        para.setdefault("is_handwritten", False)

        for line in para.get("lines", []):
            mp = line.pop("mask_path", None)
            if mp:
                line["mask"] = Image.open(
                    os.path.join(annotation_dir, mp)
                ).convert("L")
            line["bbox"] = tuple(line["bbox"])

            for word in line.get("words", []):
                mp = word.pop("mask_path", None)
                if mp:
                    word["mask"] = Image.open(
                        os.path.join(annotation_dir, mp)
                    ).convert("L")
                word["bbox"] = tuple(word["bbox"])

                for char in word.get("characters", []):
                    mp = char.pop("mask_path", None)
                    if mp:
                        char["mask"] = Image.open(
                            os.path.join(annotation_dir, mp)
                        ).convert("L")
                    char["bbox"] = tuple(char["bbox"])

    return page_image, data


# ---------------------------------------------------------------------------
# AnnotatedPage — adapter for training integration
# ---------------------------------------------------------------------------
class AnnotatedPage:
    """Wraps load_annotations() output to match SyntheticPage interface.

    Provides .image, .bg_color, .paragraphs (with contour at every level),
    .page_bbox, .page_mask, .page_contour — everything ContourSequenceDataset
    expects.
    """

    def __init__(self, annotation_dir):
        page_image, data = load_annotations(annotation_dir)
        self.image = page_image
        self.bg_color = estimate_background_color(page_image)

        self.paragraphs = []
        for para in data.get("paragraphs", []):
            p = dict(para)
            mask = p.get("mask")
            p["contour"] = contour_from_mask(mask) if mask else []
            p.setdefault("is_handwritten", False)

            new_lines = []
            for line in p.get("lines", []):
                ln = dict(line)
                mask = ln.get("mask")
                ln["contour"] = contour_from_mask(mask) if mask else []

                new_words = []
                for word in ln.get("words", []):
                    wd = dict(word)
                    mask = wd.get("mask")
                    wd["contour"] = contour_from_mask(mask) if mask else []

                    new_chars = []
                    for ch in wd.get("characters", []):
                        if ch.get("char", "") not in CHAR_TO_CLASS:
                            continue
                        cd = dict(ch)
                        mask = cd.get("mask")
                        cd["contour"] = contour_from_mask(mask) if mask else []
                        new_chars.append(cd)
                    wd["characters"] = new_chars
                    new_words.append(wd)
                ln["words"] = new_words
                new_lines.append(ln)
            p["lines"] = new_lines
            self.paragraphs.append(p)

        # Page-level fields
        if self.paragraphs:
            x1 = min(p["bbox"][0] for p in self.paragraphs)
            y1 = min(p["bbox"][1] for p in self.paragraphs)
            x2 = max(p["bbox"][2] for p in self.paragraphs)
            y2 = max(p["bbox"][3] for p in self.paragraphs)
            self.page_bbox = (x1, y1, x2, y2)
            self.page_mask = _compute_mask(self.image, self.page_bbox, self.bg_color)
            self.page_contour = contour_from_mask(self.page_mask)
        else:
            w, h = self.image.size
            self.page_bbox = (0, 0, w, h)
            self.page_mask = None
            self.page_contour = []


def load_all_annotations(base_dir):
    """Load all annotated real images from a directory.

    Expects base_dir to contain subdirectories, each produced by
    annotate_real.py (page.png + annotations.json + masks/).

    Returns list of AnnotatedPage objects, skipping failures with warning.
    """
    pages = []
    for name in sorted(os.listdir(base_dir)):
        subdir = os.path.join(base_dir, name)
        json_path = os.path.join(subdir, "annotations.json")
        if os.path.isdir(subdir) and os.path.isfile(json_path):
            try:
                ap = AnnotatedPage(subdir)
                n_paras = len(ap.paragraphs)
                n_chars = sum(
                    len(w["characters"])
                    for p in ap.paragraphs
                    for l in p["lines"]
                    for w in l["words"]
                )
                log.info("  Real page '%s': %d paragraphs, %d chars", name, n_paras, n_chars)
                pages.append(ap)
            except Exception as e:
                log.warning("  Failed to load '%s': %s", name, e)
    return pages


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_annotations(page_image, nested, output_path):
    """Draw bboxes on the page image and save for visual inspection."""
    vis = page_image.copy().convert("RGB")
    draw = ImageDraw.Draw(vis)

    colors = {
        "paragraph": (255, 0, 0),    # red
        "line": (0, 0, 255),         # blue
        "word": (255, 165, 0),       # orange
        "character": (0, 180, 0),    # green
    }

    for para in nested.get("paragraphs", []):
        b = para["bbox"]
        draw.rectangle(b, outline=colors["paragraph"], width=3)
        hw_label = "HW" if para.get("is_handwritten") else "PR"
        draw.text((b[0], max(0, b[1] - 14)), hw_label,
                  fill=colors["paragraph"])
        for line in para.get("lines", []):
            b = line["bbox"]
            draw.rectangle(b, outline=colors["line"], width=2)
            for word in line.get("words", []):
                b = word["bbox"]
                draw.rectangle(b, outline=colors["word"], width=2)
                for char in word.get("characters", []):
                    b = char["bbox"]
                    draw.rectangle(b, outline=colors["character"], width=1)

    vis_path = os.path.join(output_path, "visualization.png")
    vis.save(vis_path)
    log.info("Saved visualization to %s", vis_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate real document images using GPT-4o hierarchical vision cascade",
    )
    parser.add_argument(
        "images", nargs="+",
        help="Path(s) to document image files",
    )
    parser.add_argument(
        "--output-dir", default="real_annotations",
        help="Base output directory (default: real_annotations/)",
    )
    parser.add_argument(
        "--bg-color", default=None,
        help="Background color as R,G,B (e.g. 255,255,255). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--rate-limit-delay", type=float, default=0.5,
        help="Delay between API calls in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--handwritten", action="store_true",
        help="Force is_handwritten=True on all paragraphs",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Save visualization image with bboxes drawn",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load API key
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY not found in environment or .env file")
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    # Parse bg_color
    bg_color = None
    if args.bg_color:
        try:
            parts = [int(x.strip()) for x in args.bg_color.split(",")]
            if len(parts) == 3:
                bg_color = tuple(parts)
            else:
                log.error("--bg-color must be R,G,B (3 values)")
                sys.exit(1)
        except ValueError:
            log.error("--bg-color must be integers: R,G,B")
            sys.exit(1)

    tracker = TokenTracker()

    for image_path in args.images:
        log.info("Processing: %s", image_path)

        if not os.path.isfile(image_path):
            log.error("File not found: %s", image_path)
            continue

        page_image = Image.open(image_path).convert("RGB")
        img_bg = bg_color if bg_color else estimate_background_color(page_image)
        log.info("Background color: %s", img_bg)

        # Hierarchical cascade: page → paragraphs → lines → words → characters
        log.info("Stage 1: Hierarchical cascade annotation")
        flat = build_cascade_annotations(
            client, page_image,
            rate_limit_delay=args.rate_limit_delay,
            tracker=tracker,
        )
        if not flat:
            log.warning("No annotations for %s, skipping", image_path)
            continue

        # Apply --handwritten flag
        if args.handwritten:
            for ann in flat:
                if ann["type"] == "paragraph":
                    ann["is_handwritten"] = True

        log.info("Stage 2: Nesting cascade results")
        nested = flat_to_nested_local(flat)

        # Stage 3: mask computation + output
        log.info("Stage 3: Mask computation + output")
        build_annotated_page(page_image, nested, bg_color=img_bg)

        # Determine output subdirectory
        basename = os.path.splitext(os.path.basename(image_path))[0]
        out_dir = os.path.join(args.output_dir, basename)

        save_annotations(out_dir, image_path, nested, page_image, img_bg)

        if args.visualize:
            visualize_annotations(page_image, nested, out_dir)

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

    log.info("All images processed.")
    tracker.summary()


if __name__ == "__main__":
    main()
