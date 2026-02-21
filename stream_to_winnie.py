"""
Stream documents to winnie-interface as they are crawled/collected.

Watches a source directory for new images, annotates them with GPT-4o,
and imports them into winnie-interface via the seed script.

Usage:
    .venv/bin/python stream_to_winnie.py --source docs_dataset --batch-size 5
"""

import argparse
import json
import os
import signal
import shutil
import sqlite3
import subprocess
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from annotate_real import (
    build_annotated_page,
    build_cascade_annotations,
    estimate_background_color,
    flat_to_nested_local,
    save_annotations,
)

WINNIE_DIR = "/home/ubuntu/Code/winnie-interface"
ANNOTATIONS_DIR = "real_annotations"
TRACKER_DB = "stream_tracker.db"


def init_tracker(db_path):
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


def is_processed(conn, filename):
    row = conn.execute(
        "SELECT 1 FROM processed WHERE filename=?", (filename,)
    ).fetchone()
    return row is not None


def mark_processed(conn, filename, status="ok"):
    conn.execute(
        "INSERT OR REPLACE INTO processed (filename, status, timestamp) "
        "VALUES (?, ?, ?)",
        (filename, status, time.time()),
    )
    conn.commit()


def get_new_images(source_dir, conn):
    """Find images in source_dir that haven't been processed yet."""
    new = []
    for f in sorted(os.listdir(source_dir)):
        if not f.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        if is_processed(conn, f):
            continue
        new.append(f)
    return new


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("GPT-4o annotation timed out")


def annotate_image(client, image_path, output_dir, timeout=120):
    """Annotate a single image with GPT-4o and save results."""
    # Set per-image timeout
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)

    try:
        return _annotate_image_inner(client, image_path, output_dir)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _annotate_image_inner(client, image_path, output_dir):
    page_image = Image.open(image_path).convert("RGB")

    # Downscale large images to speed up GPT-4o annotation
    w, h = page_image.size
    max_dim = 1024
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        page_image = page_image.resize(
            (int(w * scale), int(h * scale)), Image.LANCZOS
        )

    bg_color = estimate_background_color(page_image)

    flat = build_cascade_annotations(client, page_image)
    if not flat:
        return None

    nested = flat_to_nested_local(flat)
    build_annotated_page(page_image, nested, bg_color=bg_color)

    basename = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join(output_dir, basename)
    save_annotations(out_dir, image_path, nested, page_image, bg_color)

    n_para = len(nested.get("paragraphs", []))
    n_chars = sum(
        len(w.get("characters", []))
        for p in nested.get("paragraphs", [])
        for l in p.get("lines", [])
        for w in l.get("words", [])
    )
    return {"paragraphs": n_para, "characters": n_chars, "output_dir": out_dir}


def import_to_winnie(annotation_dirs):
    """Import annotation directories into winnie-interface."""
    if not annotation_dirs:
        return True
    cmd = [
        "npx", "tsx", "scripts/seed.ts", "--import",
    ] + annotation_dirs
    result = subprocess.run(
        cmd, cwd=WINNIE_DIR, capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        print(f"  Import error: {result.stderr[:200]}", flush=True)
        return False
    print(result.stdout.strip(), flush=True)
    return True


def stream(source_dir, batch_size=5, poll_interval=5, max_batches=None,
           annotations_dir=ANNOTATIONS_DIR):
    """Main streaming loop."""
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found")
        sys.exit(1)

    client = OpenAI(api_key=api_key, timeout=90, max_retries=1)
    tracker_path = os.path.join(source_dir, TRACKER_DB)
    conn = init_tracker(tracker_path)

    os.makedirs(annotations_dir, exist_ok=True)

    batch_count = 0
    total_uploaded = 0

    print(f"Streaming from {source_dir} → winnie-interface", flush=True)
    print(f"Batch size: {batch_size}, Poll interval: {poll_interval}s", flush=True)

    while True:
        new_images = get_new_images(source_dir, conn)

        if not new_images:
            if max_batches is not None:
                print("No more images. Done.", flush=True)
                break
            time.sleep(poll_interval)
            continue

        batch = new_images[:batch_size]
        batch_count += 1
        print(f"\n--- Batch {batch_count}: {len(batch)} images ---", flush=True)

        for filename in batch:
            image_path = os.path.join(source_dir, filename)
            print(f"  Annotating {filename}...", end=" ", flush=True)

            try:
                result = annotate_image(client, image_path, annotations_dir)
                if result:
                    print(
                        f"{result['paragraphs']} paras, "
                        f"{result['characters']} chars → ",
                        end="", flush=True,
                    )
                    mark_processed(conn, filename, "annotated")
                    # Upload to winnie immediately
                    ok = import_to_winnie([result["output_dir"]])
                    if ok:
                        total_uploaded += 1
                        print(f"uploaded ({total_uploaded} total)", flush=True)
                    else:
                        print("import failed", flush=True)
                else:
                    print("no annotations (skipped)", flush=True)
                    mark_processed(conn, filename, "empty")
            except Exception as e:
                print(f"error: {e}", flush=True)
                mark_processed(conn, filename, f"error: {e}")

        if max_batches is not None and batch_count >= max_batches:
            print(f"\nReached max batches ({max_batches}). Done.", flush=True)
            break

    conn.close()
    print(f"\nStreaming complete. {total_uploaded} documents uploaded.", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Stream crawled documents to winnie-interface",
    )
    parser.add_argument(
        "--source", default="docs_dataset",
        help="Source directory with crawled images (default: docs_dataset)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=5,
        help="Images per batch (default: 5)",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=10,
        help="Seconds between polls for new images (default: 10)",
    )
    parser.add_argument(
        "--max-batches", type=int, default=None,
        help="Stop after N batches (default: run forever)",
    )
    parser.add_argument(
        "--annotations-dir", default=ANNOTATIONS_DIR,
        help="Directory for annotation output (default: real_annotations)",
    )
    args = parser.parse_args()

    stream(
        source_dir=args.source,
        batch_size=args.batch_size,
        poll_interval=args.poll_interval,
        max_batches=args.max_batches,
        annotations_dir=args.annotations_dir,
    )


if __name__ == "__main__":
    main()
