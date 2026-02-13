"""
Annotate real document images using OpenAI GPT-4o vision API.

Pipeline:
  Stage 1 — Hierarchical vision cascade (multiple GPT-4o vision calls):
      Full page → paragraph bboxes → line bboxes → character bboxes + chars
  Stage 2 — Flat-to-nested conversion (GPT-4o text or local fallback)
  Stage 3 — Mask computation + output saving (local, no API)

Output format matches SyntheticPage hierarchy for training integration:
  paragraphs → lines → characters, each with bbox, mask, and char label.

Usage:
  .venv/bin/python annotate_real.py image1.png [image2.jpg ...] \\
      --output-dir real_annotations/ \\
      --bg-color 255,255,255 \\
      --skip-nesting-gpt \\
      --rate-limit-delay 0.5 \\
      --visualize
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

from generate_training_data import _compute_mask

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

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

NESTED_SCHEMA = {
    "name": "nested_annotation",
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
                        },
                        "lines": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "bbox": {
                                        "type": "array",
                                        "items": {"type": "integer"},
                                    },
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
                                    },
                                },
                                "required": ["bbox", "characters"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["bbox", "lines"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["paragraphs"],
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
                       response_schema, max_retries=3, detail="high"):
    """Call GPT-4o with an image and structured output schema.

    Returns parsed JSON dict or None on failure.
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
                max_tokens=4096,
                temperature=0.0,
            )
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
# Stage 1: Hierarchical vision cascade
# ---------------------------------------------------------------------------

def detect_paragraphs(client, page_image):
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
        PARAGRAPH_SCHEMA, detail="low",
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


def detect_lines(client, paragraph_crop):
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
        LINE_SCHEMA, detail="high",
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


def detect_characters(client, line_crop):
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
        CHARACTER_SCHEMA, detail="high",
    )
    if result is None:
        log.warning("Character detection failed for line crop")
        return []

    chars = []
    for ch in result.get("characters", []):
        bbox = ch.get("bbox")
        char = ch.get("char", "")
        if bbox and len(bbox) == 4 and len(char) == 1:
            # Filter to ASCII printable
            if 32 <= ord(char) <= 126:
                bbox = clamp_bbox([int(v) for v in bbox], w, h)
                chars.append({"bbox": bbox, "char": char})
    log.info("    Detected %d characters", len(chars))
    return chars


def build_flat_annotations(client, page_image, rate_limit_delay=0.5):
    """Orchestrate the hierarchical vision cascade.

    Calls detect_paragraphs, then detect_lines per paragraph, then
    detect_characters per line. Translates crop-local coords to
    page-global coords.

    Returns flat list:
      [{"id": int, "type": str, "bbox": [x1,y1,x2,y2], "text": str,
        "para_idx": int, "line_idx": int|None}, ...]
    """
    flat = []
    next_id = 0

    # Detect paragraphs
    paragraphs = detect_paragraphs(client, page_image)
    time.sleep(rate_limit_delay)

    for pi, para in enumerate(paragraphs):
        pb = para["bbox"]
        flat.append({
            "id": next_id, "type": "paragraph",
            "bbox": pb, "text": "",
            "para_idx": pi, "line_idx": None,
        })
        next_id += 1

        # Crop paragraph region
        para_crop = page_image.crop(pb)

        # Detect lines within paragraph
        lines = detect_lines(client, para_crop)
        time.sleep(rate_limit_delay)

        for li, line in enumerate(lines):
            lb = line["bbox"]
            # Translate line bbox to page-global coords
            lb_global = [
                lb[0] + pb[0], lb[1] + pb[1],
                lb[2] + pb[0], lb[3] + pb[1],
            ]
            flat.append({
                "id": next_id, "type": "line",
                "bbox": lb_global, "text": "",
                "para_idx": pi, "line_idx": li,
            })
            next_id += 1

            # Crop line region
            line_crop = para_crop.crop(lb)

            # Detect characters within line
            chars = detect_characters(client, line_crop)
            time.sleep(rate_limit_delay)

            for ch in chars:
                cb = ch["bbox"]
                # Translate char bbox to page-global coords
                cb_global = [
                    cb[0] + lb[0] + pb[0], cb[1] + lb[1] + pb[1],
                    cb[2] + lb[0] + pb[0], cb[3] + lb[1] + pb[1],
                ]
                flat.append({
                    "id": next_id, "type": "character",
                    "bbox": cb_global, "text": ch["char"],
                    "para_idx": pi, "line_idx": li,
                })
                next_id += 1

    log.info("Built %d flat annotations total", len(flat))
    return flat


# ---------------------------------------------------------------------------
# Stage 2: Flat-to-nested conversion
# ---------------------------------------------------------------------------

def _center(bbox):
    """Return (cx, cy) center point of a bbox."""
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _contains_center(outer, inner_bbox):
    """Check if the center of inner_bbox is inside outer bbox."""
    cx, cy = _center(inner_bbox)
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def flat_to_nested_local(flat_annotations):
    """Deterministic local fallback: group flat annotations into hierarchy
    using para_idx/line_idx from the cascade (already implicitly hierarchical).
    """
    paragraphs_map = {}
    lines_map = {}
    chars_list = []

    for ann in flat_annotations:
        if ann["type"] == "paragraph":
            pi = ann["para_idx"]
            paragraphs_map[pi] = {"bbox": ann["bbox"], "lines": []}
        elif ann["type"] == "line":
            pi = ann["para_idx"]
            li = ann["line_idx"]
            lines_map[(pi, li)] = {"bbox": ann["bbox"], "characters": []}
        elif ann["type"] == "character":
            chars_list.append(ann)

    # Assign characters to lines
    for ch in chars_list:
        key = (ch["para_idx"], ch["line_idx"])
        if key in lines_map:
            lines_map[key]["characters"].append({
                "bbox": ch["bbox"], "char": ch["text"],
            })

    # Assign lines to paragraphs
    for (pi, li), line_data in sorted(lines_map.items()):
        if pi in paragraphs_map:
            paragraphs_map[pi]["lines"].append(line_data)

    # Build sorted output
    nested = {
        "paragraphs": [
            paragraphs_map[k] for k in sorted(paragraphs_map.keys())
        ]
    }
    return nested


NESTING_SYSTEM_PROMPT = """\
You are a document structure expert. Given a flat list of detected items \
(paragraphs, lines, characters) with bounding boxes, you must group them \
into a nested hierarchy: paragraphs contain lines, lines contain characters.

Use spatial containment: a line belongs to the paragraph whose bbox contains \
the line's center point; a character belongs to the line whose bbox contains \
the character's center point.

Example 1:
Flat input:
[
  {"id":0,"type":"paragraph","bbox":[10,10,500,200],"text":""},
  {"id":1,"type":"line","bbox":[15,15,490,60],"text":""},
  {"id":2,"type":"character","bbox":[20,20,40,55],"text":"H"},
  {"id":3,"type":"character","bbox":[42,20,62,55],"text":"i"}
]

Nested output:
{"paragraphs":[{"bbox":[10,10,500,200],"lines":[{"bbox":[15,15,490,60],\
"characters":[{"bbox":[20,20,40,55],"char":"H"},{"bbox":[42,20,62,55],\
"char":"i"}]}]}]}

Example 2:
Flat input:
[
  {"id":0,"type":"paragraph","bbox":[10,10,400,100],"text":""},
  {"id":1,"type":"line","bbox":[15,15,390,50],"text":""},
  {"id":2,"type":"character","bbox":[20,18,35,48],"text":"A"},
  {"id":3,"type":"paragraph","bbox":[10,120,400,220],"text":""},
  {"id":4,"type":"line","bbox":[15,125,390,165],"text":""},
  {"id":5,"type":"character","bbox":[20,128,35,162],"text":"B"}
]

Nested output:
{"paragraphs":[{"bbox":[10,10,400,100],"lines":[{"bbox":[15,15,390,50],\
"characters":[{"bbox":[20,18,35,48],"char":"A"}]}]},\
{"bbox":[10,120,400,220],"lines":[{"bbox":[15,125,390,165],\
"characters":[{"bbox":[20,128,35,162],"char":"B"}]}]}]}
"""


def flat_to_nested(client, flat_annotations):
    """Use GPT-4o text to convert flat annotations to nested hierarchy."""
    # Simplify the flat list for the prompt
    simple = []
    for ann in flat_annotations:
        simple.append({
            "id": ann["id"],
            "type": ann["type"],
            "bbox": ann["bbox"],
            "text": ann["text"],
        })

    user_prompt = (
        "Convert this flat annotation list to nested hierarchy:\n"
        + json.dumps(simple, indent=None)
    )

    messages = [
        {"role": "system", "content": NESTING_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": NESTED_SCHEMA,
                },
                max_tokens=8192,
                temperature=0.0,
            )
            content = resp.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            wait = 2 ** attempt
            log.warning("Nesting GPT call failed (attempt %d/3): %s — retrying in %ds",
                        attempt + 1, e, wait)
            time.sleep(wait)

    log.warning("GPT nesting failed, falling back to local nesting")
    return flat_to_nested_local(flat_annotations)


# ---------------------------------------------------------------------------
# Stage 3: Mask computation + output
# ---------------------------------------------------------------------------

def build_annotated_page(page_image, nested, bg_color=None):
    """Add pixel masks to each element in the nested annotation.

    Modifies nested in-place, adding "mask" keys with PIL Image values.
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
            for char in line.get("characters", []):
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
          char_0_0_0_mask.png
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
                "characters": [],
            }

            # Save line mask
            mask = line.get("mask")
            if mask is not None:
                mask_name = f"line_{pi}_{li}_mask.png"
                mask.save(os.path.join(masks_dir, mask_name))
                line_json["mask_path"] = f"masks/{mask_name}"

            for ci, char in enumerate(line.get("characters", [])):
                char_json = {
                    "bbox": char["bbox"],
                    "char": char["char"],
                }

                # Save character mask
                mask = char.get("mask")
                if mask is not None:
                    mask_name = f"char_{pi}_{li}_{ci}_mask.png"
                    mask.save(os.path.join(masks_dir, mask_name))
                    char_json["mask_path"] = f"masks/{mask_name}"

                line_json["characters"].append(char_json)

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

        for line in para.get("lines", []):
            mp = line.pop("mask_path", None)
            if mp:
                line["mask"] = Image.open(
                    os.path.join(annotation_dir, mp)
                ).convert("L")
            line["bbox"] = tuple(line["bbox"])

            for char in line.get("characters", []):
                mp = char.pop("mask_path", None)
                if mp:
                    char["mask"] = Image.open(
                        os.path.join(annotation_dir, mp)
                    ).convert("L")
                char["bbox"] = tuple(char["bbox"])

    return page_image, data


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
        "character": (0, 180, 0),    # green
    }

    for para in nested.get("paragraphs", []):
        b = para["bbox"]
        draw.rectangle(b, outline=colors["paragraph"], width=3)
        for line in para.get("lines", []):
            b = line["bbox"]
            draw.rectangle(b, outline=colors["line"], width=2)
            for char in line.get("characters", []):
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
        description="Annotate real document images using GPT-4o vision",
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
        "--skip-nesting-gpt", action="store_true",
        help="Use local deterministic nesting instead of GPT",
    )
    parser.add_argument(
        "--rate-limit-delay", type=float, default=0.5,
        help="Delay between API calls in seconds (default: 0.5)",
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

    for image_path in args.images:
        log.info("Processing: %s", image_path)

        if not os.path.isfile(image_path):
            log.error("File not found: %s", image_path)
            continue

        page_image = Image.open(image_path).convert("RGB")
        img_bg = bg_color if bg_color else estimate_background_color(page_image)
        log.info("Background color: %s", img_bg)

        # Stage 1: hierarchical vision cascade
        log.info("Stage 1: Hierarchical vision cascade")
        flat = build_flat_annotations(
            client, page_image,
            rate_limit_delay=args.rate_limit_delay,
        )

        if not flat:
            log.warning("No annotations detected for %s, skipping", image_path)
            continue

        # Stage 2: flat-to-nested conversion
        log.info("Stage 2: Flat-to-nested conversion")
        if args.skip_nesting_gpt:
            nested = flat_to_nested_local(flat)
        else:
            nested = flat_to_nested(client, flat)

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
        n_chars = sum(
            len(l.get("characters", []))
            for p in nested.get("paragraphs", [])
            for l in p.get("lines", [])
        )
        log.info(
            "Done: %s — %d paragraphs, %d lines, %d characters",
            image_path, n_para, n_lines, n_chars,
        )

    log.info("All images processed.")


if __name__ == "__main__":
    main()
