"""
Wave-based batch annotation using OpenAI Batch API.

Processes images through the 5-level cascade (page→paragraphs→lines→words→chars)
by batching all images at each level together, submitting as one OpenAI Batch,
waiting for results, then using those results to crop for the next level.

Coordinates with stream_to_winnie.py via stream_tracker.db — claimed images
are marked batch_pending so the live pipeline skips them.

Usage:
    python batch_annotate.py --source docs_dataset --claim 500
    python batch_annotate.py --resume
    python batch_annotate.py --status
    python batch_annotate.py --assemble-only
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import tempfile
import time

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from annotate_real import (
    PAGE_REGION_SCHEMA,
    PARAGRAPH_SCHEMA,
    LINE_SCHEMA,
    WORD_SCHEMA,
    CHARACTER_SCHEMA,
    build_annotated_page,
    clamp_bbox,
    encode_image_base64,
    estimate_background_color,
    flat_to_nested_local,
    save_annotations,
    validate_bbox,
)
from generate_training_data import CHAR_TO_CLASS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Max requests per OpenAI batch
BATCH_REQUEST_LIMIT = 45_000
# Max file size for OpenAI batch upload (bytes) — limit is 200MB, use 150MB for safety
BATCH_FILE_SIZE_LIMIT = 150 * 1024 * 1024

# ---------------------------------------------------------------------------
# State DB
# ---------------------------------------------------------------------------

def init_state_db(path):
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS claims (
            filename TEXT PRIMARY KEY,
            claimed_at REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wave INTEGER,
            openai_batch_id TEXT,
            input_file_id TEXT,
            status TEXT,
            created_at REAL,
            completed_at REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cascade (
            filename TEXT,
            key TEXT,
            bbox TEXT,
            extra TEXT,
            PRIMARY KEY (filename, key)
        )
    """)
    conn.commit()
    return conn


def init_tracker(db_path):
    """Open (or create) the stream_tracker.db used by stream_to_winnie.py."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS processed (
            filename TEXT PRIMARY KEY,
            status TEXT,
            timestamp REAL
        )
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Image claiming
# ---------------------------------------------------------------------------

def claim_images(state_conn, tracker_conn, source_dir, n):
    """Claim N unprocessed images for batch processing."""
    already_claimed = {
        r[0] for r in state_conn.execute("SELECT filename FROM claims").fetchall()
    }
    already_processed = {
        r[0] for r in tracker_conn.execute("SELECT filename FROM processed").fetchall()
    }
    skip = already_claimed | already_processed

    candidates = sorted(
        f for f in os.listdir(source_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg")) and f not in skip
    )[:n]

    now = time.time()
    for f in candidates:
        state_conn.execute(
            "INSERT OR IGNORE INTO claims (filename, claimed_at) VALUES (?, ?)",
            (f, now),
        )
        tracker_conn.execute(
            "INSERT OR REPLACE INTO processed (filename, status, timestamp) "
            "VALUES (?, ?, ?)",
            (f, "batch_pending", now),
        )
    state_conn.commit()
    tracker_conn.commit()
    log.info("Claimed %d images for batch processing", len(candidates))
    return candidates


def get_claimed_files(state_conn):
    """Return list of all claimed filenames."""
    return [
        r[0] for r in
        state_conn.execute("SELECT filename FROM claims ORDER BY filename").fetchall()
    ]


# ---------------------------------------------------------------------------
# Batch API request building helpers
# ---------------------------------------------------------------------------

def _make_batch_request(custom_id, system_prompt, user_prompt, image_b64,
                        schema, detail="high"):
    """Build a single JSONL line dict for the Batch API."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o",
            "max_tokens": 16384,
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": schema,
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": detail,
                            },
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
        },
    }


def _load_and_encode(path, max_dim=1024):
    """Load image, downscale if needed, return (PIL Image, base64 str)."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img, encode_image_base64(img)


def _write_jsonl(requests, prefix):
    """Write request dicts to a temp JSONL file. Returns path."""
    path = os.path.join(tempfile.gettempdir(), f"batch_{prefix}_{int(time.time())}.jsonl")
    with open(path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
    log.info("Wrote %d requests to %s", len(requests), path)
    return path


# ---------------------------------------------------------------------------
# Wave preparation (one function per cascade level)
# ---------------------------------------------------------------------------

def prepare_wave_0(state_conn, source_dir):
    """Wave 0: full page → detect page region."""
    filenames = get_claimed_files(state_conn)
    requests = []
    for fn in filenames:
        path = os.path.join(source_dir, fn)
        if not os.path.isfile(path):
            continue
        img, b64 = _load_and_encode(path)
        w, h = img.size
        system_prompt = (
            "You are a document layout analyzer. You detect the text-bearing "
            "region of a document image — the tightest bounding box that contains "
            "all text content, excluding empty margins. Return precise pixel coordinates."
        )
        user_prompt = (
            f"Detect the text-bearing region of this document image ({w}x{h} pixels). "
            f"Return a single [x1,y1,x2,y2] bounding box that tightly encloses all "
            f"text content, excluding blank margins around the edges."
        )
        requests.append(_make_batch_request(
            custom_id=f"{fn}__page",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_b64=b64,
            schema=PAGE_REGION_SCHEMA,
            detail="low",
        ))
    if not requests:
        return None
    return _write_jsonl(requests, "wave0")


def prepare_wave_1(state_conn, source_dir):
    """Wave 1: crop page region → detect paragraphs."""
    filenames = get_claimed_files(state_conn)
    requests = []
    for fn in filenames:
        path = os.path.join(source_dir, fn)
        if not os.path.isfile(path):
            continue
        row = state_conn.execute(
            "SELECT bbox FROM cascade WHERE filename=? AND key='page'", (fn,)
        ).fetchone()
        if not row:
            continue
        page_bbox = json.loads(row[0])

        img, _ = _load_and_encode(path)
        crop = img.crop(page_bbox)
        b64 = encode_image_base64(crop)
        w, h = crop.size

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
        requests.append(_make_batch_request(
            custom_id=f"{fn}__paras",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_b64=b64,
            schema=PARAGRAPH_SCHEMA,
            detail="low",
        ))
    if not requests:
        return None
    return _write_jsonl(requests, "wave1")


def prepare_wave_2(state_conn, source_dir):
    """Wave 2: crop each paragraph → detect lines."""
    filenames = get_claimed_files(state_conn)
    requests = []
    for fn in filenames:
        path = os.path.join(source_dir, fn)
        if not os.path.isfile(path):
            continue
        row = state_conn.execute(
            "SELECT bbox FROM cascade WHERE filename=? AND key='page'", (fn,)
        ).fetchone()
        if not row:
            continue
        page_bbox = json.loads(row[0])

        img, _ = _load_and_encode(path)
        page_crop = img.crop(page_bbox)

        # Find all paragraphs for this image
        para_rows = state_conn.execute(
            "SELECT key, bbox FROM cascade WHERE filename=? AND key LIKE 'para_%'",
            (fn,),
        ).fetchall()
        for key, bbox_json in para_rows:
            pi = key.split("_")[1]  # "para_0" -> "0"
            para_bbox = json.loads(bbox_json)
            para_crop = page_crop.crop(para_bbox)
            b64 = encode_image_base64(para_crop)
            w, h = para_crop.size

            system_prompt = (
                "You are a document layout analyzer. You detect individual text lines "
                "within a paragraph. Return precise pixel-coordinate bounding boxes."
            )
            user_prompt = (
                f"Detect all text lines in this paragraph crop. Image is {w}x{h} "
                f"pixels. Return [x1,y1,x2,y2] bboxes in top-to-bottom order."
            )
            requests.append(_make_batch_request(
                custom_id=f"{fn}__para_{pi}__lines",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_b64=b64,
                schema=LINE_SCHEMA,
                detail="high",
            ))
    if not requests:
        return None
    return _write_jsonl(requests, "wave2")


def prepare_wave_3(state_conn, source_dir):
    """Wave 3: crop each line → detect words."""
    filenames = get_claimed_files(state_conn)
    requests = []
    for fn in filenames:
        path = os.path.join(source_dir, fn)
        if not os.path.isfile(path):
            continue
        row = state_conn.execute(
            "SELECT bbox FROM cascade WHERE filename=? AND key='page'", (fn,)
        ).fetchone()
        if not row:
            continue
        page_bbox = json.loads(row[0])

        img, _ = _load_and_encode(path)
        page_crop = img.crop(page_bbox)

        # Find all lines for this image
        line_rows = state_conn.execute(
            "SELECT key, bbox FROM cascade WHERE filename=? AND key LIKE 'line_%'",
            (fn,),
        ).fetchall()
        for key, bbox_json in line_rows:
            # key = "line_0_1" -> pi=0, li=1
            parts = key.split("_")
            pi, li = parts[1], parts[2]
            line_bbox_local = json.loads(bbox_json)

            # Need para bbox to compute absolute crop
            para_row = state_conn.execute(
                "SELECT bbox FROM cascade WHERE filename=? AND key=?",
                (fn, f"para_{pi}"),
            ).fetchone()
            if not para_row:
                continue
            para_bbox = json.loads(para_row[0])

            para_crop = page_crop.crop(para_bbox)
            line_crop = para_crop.crop(line_bbox_local)
            b64 = encode_image_base64(line_crop)
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
            requests.append(_make_batch_request(
                custom_id=f"{fn}__para_{pi}__line_{li}__words",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_b64=b64,
                schema=WORD_SCHEMA,
                detail="high",
            ))
    if not requests:
        return None
    return _write_jsonl(requests, "wave3")


def prepare_wave_4(state_conn, source_dir):
    """Wave 4: crop each word → detect characters."""
    filenames = get_claimed_files(state_conn)
    requests = []
    for fn in filenames:
        path = os.path.join(source_dir, fn)
        if not os.path.isfile(path):
            continue
        row = state_conn.execute(
            "SELECT bbox FROM cascade WHERE filename=? AND key='page'", (fn,)
        ).fetchone()
        if not row:
            continue
        page_bbox = json.loads(row[0])

        img, _ = _load_and_encode(path)
        page_crop = img.crop(page_bbox)

        # Find all words for this image
        word_rows = state_conn.execute(
            "SELECT key, bbox FROM cascade WHERE filename=? AND key LIKE 'word_%'",
            (fn,),
        ).fetchall()
        for key, bbox_json in word_rows:
            # key = "word_0_1_2" -> pi=0, li=1, wi=2
            parts = key.split("_")
            pi, li, wi = parts[1], parts[2], parts[3]
            word_bbox_local = json.loads(bbox_json)

            # Get parent bboxes
            para_row = state_conn.execute(
                "SELECT bbox FROM cascade WHERE filename=? AND key=?",
                (fn, f"para_{pi}"),
            ).fetchone()
            line_row = state_conn.execute(
                "SELECT bbox FROM cascade WHERE filename=? AND key=?",
                (fn, f"line_{pi}_{li}"),
            ).fetchone()
            if not para_row or not line_row:
                continue
            para_bbox = json.loads(para_row[0])
            line_bbox = json.loads(line_row[0])

            para_crop = page_crop.crop(para_bbox)
            line_crop = para_crop.crop(line_bbox)
            word_crop = line_crop.crop(word_bbox_local)
            b64 = encode_image_base64(word_crop)
            w, h = word_crop.size

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
            requests.append(_make_batch_request(
                custom_id=f"{fn}__para_{pi}__line_{li}__word_{wi}__chars",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_b64=b64,
                schema=CHARACTER_SCHEMA,
                detail="high",
            ))
    if not requests:
        return None
    return _write_jsonl(requests, "wave4")


# ---------------------------------------------------------------------------
# Batch submission / polling / result collection
# ---------------------------------------------------------------------------

def submit_batch(client, jsonl_path, state_conn, wave):
    """Upload JSONL and create a batch. Returns (batch_id, file_id)."""
    with open(jsonl_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    file_id = file_obj.id
    log.info("Uploaded file %s for wave %d", file_id, wave)

    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    batch_id = batch.id
    log.info("Created batch %s for wave %d", batch_id, wave)

    state_conn.execute(
        "INSERT INTO batches (wave, openai_batch_id, input_file_id, status, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (wave, batch_id, file_id, "submitted", time.time()),
    )
    state_conn.commit()
    return batch_id, file_id


def poll_batch(client, batch_id, interval=30):
    """Poll until batch is completed, failed, or expired. Returns final status."""
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        completed = batch.request_counts.completed if batch.request_counts else 0
        total = batch.request_counts.total if batch.request_counts else 0
        log.info("Batch %s: %s (%d/%d)", batch_id, status, completed, total)

        if status in ("completed", "failed", "expired", "cancelled"):
            return status
        time.sleep(interval)


def collect_results(client, batch_id):
    """Download output JSONL from a completed batch. Returns list of dicts."""
    batch = client.batches.retrieve(batch_id)
    if not batch.output_file_id:
        log.error("Batch %s has no output file", batch_id)
        return []

    content = client.files.content(batch.output_file_id)
    results = []
    for line in content.text.strip().split("\n"):
        if line.strip():
            results.append(json.loads(line))
    log.info("Collected %d results from batch %s", len(results), batch_id)
    return results


# ---------------------------------------------------------------------------
# Storing wave results into cascade table
# ---------------------------------------------------------------------------

def store_wave_results(state_conn, wave, results, source_dir):
    """Parse GPT-4o batch responses and store bboxes in cascade table."""
    stored = 0
    errors = 0
    for item in results:
        custom_id = item.get("custom_id", "")
        response = item.get("response", {})
        body = response.get("body", {})

        # Extract the GPT-4o response content
        choices = body.get("choices", [])
        if not choices:
            errors += 1
            continue
        content_str = choices[0].get("message", {}).get("content", "")
        try:
            content = json.loads(content_str)
        except (json.JSONDecodeError, TypeError):
            errors += 1
            continue

        # Parse custom_id to get filename and context
        # Wave 0: "{fn}__page"
        # Wave 1: "{fn}__paras"
        # Wave 2: "{fn}__para_{pi}__lines"
        # Wave 3: "{fn}__para_{pi}__line_{li}__words"
        # Wave 4: "{fn}__para_{pi}__line_{li}__word_{wi}__chars"
        parts = custom_id.split("__")
        fn = parts[0]

        if wave == 0:
            region = content.get("page_region", {})
            bbox = region.get("bbox")
            if bbox and len(bbox) == 4:
                # Load image to get dimensions for clamping
                img_path = os.path.join(source_dir, fn)
                try:
                    img, _ = _load_and_encode(img_path)
                    w, h = img.size
                    bbox = clamp_bbox([int(v) for v in bbox], w, h)
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if area < w * h * 0.05:
                        bbox = [0, 0, w, h]
                except Exception:
                    continue
            else:
                # Fallback: use full image
                try:
                    img, _ = _load_and_encode(os.path.join(source_dir, fn))
                    w, h = img.size
                    bbox = [0, 0, w, h]
                except Exception:
                    continue

            state_conn.execute(
                "INSERT OR REPLACE INTO cascade (filename, key, bbox, extra) "
                "VALUES (?, ?, ?, ?)",
                (fn, "page", json.dumps(bbox), None),
            )
            stored += 1

        elif wave == 1:
            # Get page bbox for offset translation
            page_row = state_conn.execute(
                "SELECT bbox FROM cascade WHERE filename=? AND key='page'", (fn,)
            ).fetchone()
            if not page_row:
                continue
            page_bbox = json.loads(page_row[0])
            pr_x, pr_y = page_bbox[0], page_bbox[1]
            pr_w = page_bbox[2] - page_bbox[0]
            pr_h = page_bbox[3] - page_bbox[1]

            for pi, para in enumerate(content.get("paragraphs", [])):
                bbox = para.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                # bbox is local to page region crop
                bbox = clamp_bbox([int(v) for v in bbox], pr_w, pr_h)
                state_conn.execute(
                    "INSERT OR REPLACE INTO cascade (filename, key, bbox, extra) "
                    "VALUES (?, ?, ?, ?)",
                    (fn, f"para_{pi}", json.dumps(bbox),
                     json.dumps({"global_bbox": [
                         bbox[0] + pr_x, bbox[1] + pr_y,
                         bbox[2] + pr_x, bbox[3] + pr_y,
                     ]})),
                )
                stored += 1

        elif wave == 2:
            # parts: [fn, "para_{pi}", "lines"]
            pi = parts[1].split("_")[1]
            para_row = state_conn.execute(
                "SELECT bbox FROM cascade WHERE filename=? AND key=?",
                (fn, f"para_{pi}"),
            ).fetchone()
            if not para_row:
                continue
            para_bbox = json.loads(para_row[0])
            p_w = para_bbox[2] - para_bbox[0]
            p_h = para_bbox[3] - para_bbox[1]

            for li, line in enumerate(content.get("lines", [])):
                bbox = line.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                bbox = clamp_bbox([int(v) for v in bbox], p_w, p_h)
                state_conn.execute(
                    "INSERT OR REPLACE INTO cascade (filename, key, bbox, extra) "
                    "VALUES (?, ?, ?, ?)",
                    (fn, f"line_{pi}_{li}", json.dumps(bbox), None),
                )
                stored += 1

        elif wave == 3:
            # parts: [fn, "para_{pi}", "line_{li}", "words"]
            pi = parts[1].split("_")[1]
            li = parts[2].split("_")[1]
            line_row = state_conn.execute(
                "SELECT bbox FROM cascade WHERE filename=? AND key=?",
                (fn, f"line_{pi}_{li}"),
            ).fetchone()
            if not line_row:
                continue
            line_bbox = json.loads(line_row[0])
            l_w = line_bbox[2] - line_bbox[0]
            l_h = line_bbox[3] - line_bbox[1]

            for wi, word in enumerate(content.get("words", [])):
                bbox = word.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                bbox = clamp_bbox([int(v) for v in bbox], l_w, l_h)
                state_conn.execute(
                    "INSERT OR REPLACE INTO cascade (filename, key, bbox, extra) "
                    "VALUES (?, ?, ?, ?)",
                    (fn, f"word_{pi}_{li}_{wi}", json.dumps(bbox), None),
                )
                stored += 1

        elif wave == 4:
            # parts: [fn, "para_{pi}", "line_{li}", "word_{wi}", "chars"]
            pi = parts[1].split("_")[1]
            li = parts[2].split("_")[1]
            wi = parts[3].split("_")[1]
            word_row = state_conn.execute(
                "SELECT bbox FROM cascade WHERE filename=? AND key=?",
                (fn, f"word_{pi}_{li}_{wi}"),
            ).fetchone()
            if not word_row:
                continue
            word_bbox = json.loads(word_row[0])
            wd_w = word_bbox[2] - word_bbox[0]
            wd_h = word_bbox[3] - word_bbox[1]

            for ci, ch in enumerate(content.get("characters", [])):
                bbox = ch.get("bbox")
                char = ch.get("char", "")
                if not bbox or len(bbox) != 4 or len(char) != 1:
                    continue
                if char not in CHAR_TO_CLASS:
                    continue
                bbox = clamp_bbox([int(v) for v in bbox], wd_w, wd_h)
                state_conn.execute(
                    "INSERT OR REPLACE INTO cascade (filename, key, bbox, extra) "
                    "VALUES (?, ?, ?, ?)",
                    (fn, f"char_{pi}_{li}_{wi}_{ci}", json.dumps(bbox),
                     json.dumps({"char": char})),
                )
                stored += 1

    state_conn.commit()
    log.info("Wave %d: stored %d results, %d errors", wave, stored, errors)


# ---------------------------------------------------------------------------
# Assembly: cascade rows → nested annotations → save
# ---------------------------------------------------------------------------

def assemble_image(state_conn, filename, source_dir, annotations_dir):
    """Assemble cascade results for one image into the standard annotation format."""
    rows = state_conn.execute(
        "SELECT key, bbox, extra FROM cascade WHERE filename=?", (filename,)
    ).fetchall()
    if not rows:
        return False

    # Load and downscale image (same as annotation time)
    img_path = os.path.join(source_dir, filename)
    img, _ = _load_and_encode(img_path)

    # Build flat annotations matching build_cascade_annotations output
    flat = []
    next_id = 0

    # Organize rows by key
    data = {}
    for key, bbox_json, extra_json in rows:
        data[key] = {
            "bbox": json.loads(bbox_json),
            "extra": json.loads(extra_json) if extra_json else {},
        }

    # Page region
    page_data = data.get("page")
    if not page_data:
        return False
    page_bbox = page_data["bbox"]
    pr_x, pr_y = page_bbox[0], page_bbox[1]

    flat.append({
        "id": next_id, "type": "page",
        "bbox": page_bbox, "text": "", "is_handwritten": False,
        "para_idx": None, "line_idx": None, "word_idx": None,
    })
    next_id += 1

    # Collect paragraph indices
    para_keys = sorted(
        (k for k in data if k.startswith("para_")),
        key=lambda k: int(k.split("_")[1]),
    )

    for pk in para_keys:
        pi = int(pk.split("_")[1])
        para_local = data[pk]["bbox"]
        para_global = data[pk].get("extra", {}).get("global_bbox")
        if not para_global:
            para_global = [
                para_local[0] + pr_x, para_local[1] + pr_y,
                para_local[2] + pr_x, para_local[3] + pr_y,
            ]

        flat.append({
            "id": next_id, "type": "paragraph",
            "bbox": para_global, "text": "", "is_handwritten": False,
            "para_idx": pi, "line_idx": None, "word_idx": None,
        })
        next_id += 1

        # Lines in this paragraph
        line_keys = sorted(
            (k for k in data if k.startswith(f"line_{pi}_")),
            key=lambda k: int(k.split("_")[2]),
        )
        for lk in line_keys:
            li = int(lk.split("_")[2])
            line_local = data[lk]["bbox"]
            line_global = [
                line_local[0] + para_global[0],
                line_local[1] + para_global[1],
                line_local[2] + para_global[0],
                line_local[3] + para_global[1],
            ]

            flat.append({
                "id": next_id, "type": "line",
                "bbox": line_global, "text": "", "is_handwritten": False,
                "para_idx": pi, "line_idx": li, "word_idx": None,
            })
            next_id += 1

            # Words in this line
            word_keys = sorted(
                (k for k in data if k.startswith(f"word_{pi}_{li}_")),
                key=lambda k: int(k.split("_")[3]),
            )
            for wk in word_keys:
                wi = int(wk.split("_")[3])
                word_local = data[wk]["bbox"]
                word_global = [
                    word_local[0] + line_global[0],
                    word_local[1] + line_global[1],
                    word_local[2] + line_global[0],
                    word_local[3] + line_global[1],
                ]

                flat.append({
                    "id": next_id, "type": "word",
                    "bbox": word_global, "text": "", "is_handwritten": False,
                    "para_idx": pi, "line_idx": li, "word_idx": wi,
                })
                next_id += 1

                # Characters in this word
                char_keys = sorted(
                    (k for k in data if k.startswith(f"char_{pi}_{li}_{wi}_")),
                    key=lambda k: int(k.split("_")[4]),
                )
                for ck in char_keys:
                    char_local = data[ck]["bbox"]
                    char_label = data[ck].get("extra", {}).get("char", "")
                    char_global = [
                        char_local[0] + word_global[0],
                        char_local[1] + word_global[1],
                        char_local[2] + word_global[0],
                        char_local[3] + word_global[1],
                    ]

                    flat.append({
                        "id": next_id, "type": "character",
                        "bbox": char_global, "text": char_label,
                        "is_handwritten": False,
                        "para_idx": pi, "line_idx": li, "word_idx": wi,
                    })
                    next_id += 1

    if len(flat) <= 1:
        # Only page region, no content
        return False

    # Convert to nested, build masks, save
    nested = flat_to_nested_local(flat)
    bg_color = estimate_background_color(img)
    build_annotated_page(img, nested, bg_color=bg_color)

    basename = os.path.splitext(filename)[0]
    out_dir = os.path.join(annotations_dir, basename)
    save_annotations(out_dir, img_path, nested, img, bg_color)
    return True


def assemble_all(state_conn, source_dir, annotations_dir, tracker_conn):
    """Assemble all claimed images that have cascade results."""
    filenames = get_claimed_files(state_conn)
    assembled = 0
    empty = 0
    for fn in filenames:
        # Skip already-annotated
        row = tracker_conn.execute(
            "SELECT status FROM processed WHERE filename=?", (fn,)
        ).fetchone()
        if row and row[0] == "annotated":
            assembled += 1
            continue

        try:
            ok = assemble_image(state_conn, fn, source_dir, annotations_dir)
            if ok:
                tracker_conn.execute(
                    "INSERT OR REPLACE INTO processed (filename, status, timestamp) "
                    "VALUES (?, ?, ?)",
                    (fn, "annotated", time.time()),
                )
                assembled += 1
            else:
                tracker_conn.execute(
                    "INSERT OR REPLACE INTO processed (filename, status, timestamp) "
                    "VALUES (?, ?, ?)",
                    (fn, "empty", time.time()),
                )
                empty += 1
        except Exception as e:
            log.warning("Failed to assemble %s: %s", fn, e)
            tracker_conn.execute(
                "INSERT OR REPLACE INTO processed (filename, status, timestamp) "
                "VALUES (?, ?, ?)",
                (fn, f"error: {e}", time.time()),
            )
    tracker_conn.commit()
    log.info("Assembly: %d annotated, %d empty, %d total", assembled, empty, len(filenames))


# ---------------------------------------------------------------------------
# Submit with chunking (split large waves into sub-batches)
# ---------------------------------------------------------------------------

def _chunk_jsonl(jsonl_path):
    """Split a JSONL file into chunks respecting both request count and file size limits.

    Returns list of file paths (may be just [jsonl_path] if no splitting needed).
    """
    file_size = os.path.getsize(jsonl_path)
    with open(jsonl_path) as f:
        lines = f.readlines()

    if len(lines) <= BATCH_REQUEST_LIMIT and file_size <= BATCH_FILE_SIZE_LIMIT:
        return [jsonl_path]

    # Split by size: estimate bytes per line, target chunks under limit
    chunks = []
    current_chunk = []
    current_size = 0
    for line in lines:
        line_bytes = len(line.encode("utf-8"))
        if current_chunk and (
            len(current_chunk) >= BATCH_REQUEST_LIMIT
            or current_size + line_bytes > BATCH_FILE_SIZE_LIMIT
        ):
            chunks.append(current_chunk)
            current_chunk = []
            current_size = 0
        current_chunk.append(line)
        current_size += line_bytes
    if current_chunk:
        chunks.append(current_chunk)

    paths = []
    for ci, chunk in enumerate(chunks):
        chunk_path = jsonl_path.replace(".jsonl", f"_chunk{ci}.jsonl")
        with open(chunk_path, "w") as f:
            f.writelines(chunk)
        paths.append(chunk_path)

    log.info("Split %d requests (%.0f MB) into %d chunks",
             len(lines), file_size / 1e6, len(paths))
    return paths


def submit_and_process_wave(client, state_conn, jsonl_path, wave, source_dir):
    """Submit a wave's JSONL, handling chunking by request count and file size."""
    chunk_paths = _chunk_jsonl(jsonl_path)

    for ci, path in enumerate(chunk_paths):
        if len(chunk_paths) > 1:
            log.info("Submitting chunk %d/%d for wave %d", ci + 1, len(chunk_paths), wave)
        batch_id, _ = submit_batch(client, path, state_conn, wave)
        status = poll_batch(client, batch_id)
        if status != "completed":
            log.error("Batch %s (chunk %d) ended with status: %s", batch_id, ci, status)
            return False
        results = collect_results(client, batch_id)
        store_wave_results(state_conn, wave, results, source_dir)
        state_conn.execute(
            "UPDATE batches SET status=?, completed_at=? WHERE openai_batch_id=?",
            ("completed", time.time(), batch_id),
        )
        state_conn.commit()

    return True


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def get_last_completed_wave(state_conn):
    """Return the highest wave number that completed, or -1 if none."""
    row = state_conn.execute(
        "SELECT MAX(wave) FROM batches WHERE status='completed'"
    ).fetchone()
    if row and row[0] is not None:
        return row[0]
    return -1


def print_status(state_conn, tracker_conn):
    """Print pipeline status summary."""
    n_claimed = state_conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    n_cascade = state_conn.execute("SELECT COUNT(*) FROM cascade").fetchone()[0]

    print(f"Claimed images: {n_claimed}")
    print(f"Cascade entries: {n_cascade}")

    # Batch info
    batches = state_conn.execute(
        "SELECT wave, openai_batch_id, status, created_at, completed_at "
        "FROM batches ORDER BY id"
    ).fetchall()
    print(f"\nBatches ({len(batches)}):")
    for wave, bid, status, created, completed in batches:
        elapsed = ""
        if completed and created:
            elapsed = f" ({completed - created:.0f}s)"
        print(f"  Wave {wave}: {bid} — {status}{elapsed}")

    # Tracker stats
    statuses = tracker_conn.execute(
        "SELECT status, COUNT(*) FROM processed GROUP BY status"
    ).fetchall()
    print("\nTracker status:")
    for status, count in statuses:
        print(f"  {status}: {count}")

    last_wave = get_last_completed_wave(state_conn)
    print(f"\nLast completed wave: {last_wave}")


WAVE_PREPARERS = [
    prepare_wave_0,
    prepare_wave_1,
    prepare_wave_2,
    prepare_wave_3,
    prepare_wave_4,
]


def run_pipeline(client, state_conn, tracker_conn, source_dir, annotations_dir,
                 claim_n=0, resume=False):
    """Run the full wave-based batch pipeline."""
    # Claim images if requested
    if claim_n > 0:
        claim_images(state_conn, tracker_conn, source_dir, claim_n)

    n_claimed = state_conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    if n_claimed == 0:
        log.error("No images claimed. Use --claim N to claim images first.")
        return

    log.info("Pipeline: %d claimed images", n_claimed)

    start_wave = 0
    if resume:
        start_wave = get_last_completed_wave(state_conn) + 1
        log.info("Resuming from wave %d", start_wave)

    for wave in range(start_wave, 5):
        log.info("=== Wave %d ===", wave)
        preparer = WAVE_PREPARERS[wave]
        jsonl_path = preparer(state_conn, source_dir)

        if jsonl_path is None:
            log.info("Wave %d: no requests to submit, skipping", wave)
            continue

        ok = submit_and_process_wave(client, state_conn, jsonl_path, wave, source_dir)
        if not ok:
            log.error("Wave %d failed, stopping pipeline", wave)
            return

    # Assemble all
    log.info("=== Assembly ===")
    assemble_all(state_conn, source_dir, annotations_dir, tracker_conn)
    log.info("Pipeline complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Wave-based batch annotation using OpenAI Batch API",
    )
    parser.add_argument(
        "--source", default="docs_dataset",
        help="Source directory with document images (default: docs_dataset)",
    )
    parser.add_argument(
        "--claim", type=int, default=0,
        help="Number of images to claim for batch processing",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last completed wave",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print pipeline status and exit",
    )
    parser.add_argument(
        "--assemble-only", action="store_true",
        help="Only assemble from existing cascade results",
    )
    parser.add_argument(
        "--annotations-dir", default="real_annotations",
        help="Output directory for annotations (default: real_annotations)",
    )
    parser.add_argument(
        "--state-db", default="batch_state.db",
        help="Path to batch state database (default: batch_state.db)",
    )
    args = parser.parse_args()

    load_dotenv()

    # DB paths
    tracker_db = os.path.join(args.source, "stream_tracker.db")
    state_conn = init_state_db(args.state_db)
    tracker_conn = init_tracker(tracker_db)

    if args.status:
        print_status(state_conn, tracker_conn)
        return

    if args.assemble_only:
        os.makedirs(args.annotations_dir, exist_ok=True)
        assemble_all(state_conn, args.source, args.annotations_dir, tracker_conn)
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY not found in environment or .env file")
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    os.makedirs(args.annotations_dir, exist_ok=True)

    run_pipeline(
        client=client,
        state_conn=state_conn,
        tracker_conn=tracker_conn,
        source_dir=args.source,
        annotations_dir=args.annotations_dir,
        claim_n=args.claim,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
