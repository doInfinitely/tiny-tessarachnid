"""
Web document image scraper for tiny-tessarachnid.

Crawls web pages (BFS) or searches Google Images, downloads candidate images,
scores them using the trained RetinaOCRNet model to identify document-like
content (paragraphs of text), and saves qualifying images locally.

Usage:
    # Crawl URLs
    python scrape.py --urls https://example.com/docs --depth 2 --output collected_docs

    # Search Google Images
    python scrape.py --search "scanned document page" --output search_docs

    # Deep scan with strict threshold
    python scrape.py --urls https://arxiv.org --depth 1 --deep-scan --threshold 2
"""

import argparse
import hashlib
import json
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
import torch
from bs4 import BeautifulSoup
from PIL import Image

from generate_training_data import CLASS_NONE, CLASS_PARAGRAPH, PREV_BBOX_NONE, RETINA_SIZE, scale_and_pad
from infer_02 import detect_level
from train_02 import RetinaOCRNet, _remap_old_backbone_keys

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}
DEFAULT_DELAY = 1.0        # seconds between requests to same domain
MIN_IMAGE_BYTES = 5_000    # skip tiny images (icons, spacers)
MIN_IMAGE_SIZE = 100       # minimum width/height in pixels
DEFAULT_SCORE_THRESHOLD = 1  # minimum paragraph count to qualify as document
USER_AGENT = (
    "Mozilla/5.0 (compatible; TinyTessarachnid/1.0; "
    "+https://github.com/tiny-tessarachnid)"
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_path, device):
    """Load a trained RetinaOCRNet from checkpoint."""
    model = RetinaOCRNet().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    checkpoint = _remap_old_backbone_keys(checkpoint)
    missing, _ = model.load_state_dict(checkpoint, strict=False)
    model.eval()
    if missing:
        print(f"Loaded model from {model_path} ({len(missing)} new params initialized randomly)")
    else:
        print(f"Loaded model from {model_path}")
    return model


# ---------------------------------------------------------------------------
# Background color estimation
# ---------------------------------------------------------------------------
def estimate_bg_color(image):
    """Estimate background color by sampling border pixels.

    Samples pixels from the edges of the image and returns the median color,
    which is a robust estimate even if some edges contain content.
    """
    arr = np.array(image)
    h, w = arr.shape[:2]
    if h < 2 or w < 2:
        return (255, 255, 255)

    # Collect border pixels: top/bottom rows, left/right columns
    border_pixels = np.concatenate([
        arr[0, :, :],           # top row
        arr[h - 1, :, :],      # bottom row
        arr[:, 0, :],          # left column
        arr[:, w - 1, :],      # right column
    ], axis=0)

    # Use median for robustness against content at edges
    median = np.median(border_pixels, axis=0).astype(int)
    return tuple(median.tolist())


# ---------------------------------------------------------------------------
# Document scoring
# ---------------------------------------------------------------------------
@dataclass
class DocumentScore:
    paragraph_count: int
    line_count: int
    is_document: bool
    details: dict = field(default_factory=dict)


def score_document(model, image, device, threshold=DEFAULT_SCORE_THRESHOLD,
                   deep_scan=False):
    """Score an image for document-like content using the OCR model.

    Runs paragraph detection on the image. If deep_scan is True, also detects
    lines within each paragraph for higher confidence.

    Returns a DocumentScore with paragraph/line counts and is_document flag.
    """
    bg_color = estimate_bg_color(image)

    # Level 1: detect paragraphs (gap_tolerance=30, matching infer_02.py:378)
    paragraphs = detect_level(model, image, bg_color, device, max_detections=200,
                              gap_tolerance=30)
    paragraph_count = len(paragraphs)

    line_count = 0
    if deep_scan and paragraphs:
        # Level 2: detect lines within each paragraph (gap_tolerance=5)
        for para_bbox, _ in paragraphs:
            para_crop = image.crop(para_bbox)
            lines = detect_level(model, para_crop, bg_color, device,
                                 max_detections=200, gap_tolerance=5)
            line_count += len(lines)

    is_document = paragraph_count >= threshold

    return DocumentScore(
        paragraph_count=paragraph_count,
        line_count=line_count,
        is_document=is_document,
        details={
            "bg_color": bg_color,
            "image_size": list(image.size),
            "deep_scan": deep_scan,
        },
    )


# ---------------------------------------------------------------------------
# Image fetching and validation
# ---------------------------------------------------------------------------
def fetch_image(url, session):
    """Download an image from URL, validate, and return as RGB PIL Image.

    Returns None if the image fails validation (wrong content type, too small,
    too few pixels, etc.).
    """
    try:
        resp = session.get(url, timeout=15, stream=True)
        resp.raise_for_status()
    except (requests.RequestException, Exception):
        return None

    # Check content type
    content_type = resp.headers.get("Content-Type", "")
    if not content_type.startswith("image/"):
        return None

    # Check content length if available
    content_length = resp.headers.get("Content-Length")
    if content_length and int(content_length) < MIN_IMAGE_BYTES:
        return None

    # Read the full response
    data = resp.content
    if len(data) < MIN_IMAGE_BYTES:
        return None

    # Try to open as image
    try:
        from io import BytesIO
        image = Image.open(BytesIO(data))
        image = image.convert("RGB")
    except Exception:
        return None

    # Check dimensions
    w, h = image.size
    if w < MIN_IMAGE_SIZE or h < MIN_IMAGE_SIZE:
        return None

    return image


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------
def extract_image_urls(page_url, html):
    """Extract image URLs from an HTML page.

    Finds images from <img> tags, links to image files, and og:image meta tags.
    Returns a list of absolute URLs.
    """
    soup = BeautifulSoup(html, "html.parser")
    urls = set()

    # <img> tags
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src")
        if src:
            urls.add(urljoin(page_url, src))

    # <a> tags linking directly to images
    for a in soup.find_all("a", href=True):
        href = a["href"]
        parsed = urlparse(href)
        ext = os.path.splitext(parsed.path)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            urls.add(urljoin(page_url, href))

    # og:image and twitter:image meta tags
    for meta in soup.find_all("meta"):
        prop = meta.get("property", "") or meta.get("name", "")
        if prop in ("og:image", "twitter:image"):
            content = meta.get("content")
            if content:
                urls.add(urljoin(page_url, content))

    return list(urls)


def extract_page_links(page_url, html, same_domain=True):
    """Extract page links from HTML for crawling.

    If same_domain is True, only returns links on the same domain as page_url.
    Filters out non-HTTP links and common non-page extensions.
    """
    soup = BeautifulSoup(html, "html.parser")
    skip_extensions = {
        ".pdf", ".zip", ".gz", ".tar", ".rar", ".7z",
        ".mp3", ".mp4", ".avi", ".mov", ".wmv",
        ".exe", ".dmg", ".iso",
    } | IMAGE_EXTENSIONS

    base_domain = urlparse(page_url).netloc
    links = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        absolute = urljoin(page_url, href)
        parsed = urlparse(absolute)

        # Only HTTP(S) links
        if parsed.scheme not in ("http", "https"):
            continue

        # Skip non-page file types
        ext = os.path.splitext(parsed.path)[1].lower()
        if ext in skip_extensions:
            continue

        # Same-domain filter
        if same_domain and parsed.netloc != base_domain:
            continue

        # Strip fragment
        clean = parsed._replace(fragment="").geturl()
        links.add(clean)

    return list(links)


# ---------------------------------------------------------------------------
# BFS Crawler
# ---------------------------------------------------------------------------
def crawl(seed_urls, session, max_depth=1, same_domain=True, max_pages=50,
          delay=DEFAULT_DELAY):
    """BFS crawl from seed URLs, collecting image URLs along the way.

    Returns a list of (image_url, source_page_url) tuples.
    """
    visited = set()
    queue = deque()  # (url, depth)
    image_urls = []  # (image_url, source_page)
    domain_last_request = {}  # domain -> last request timestamp

    for url in seed_urls:
        queue.append((url, 0))

    while queue and len(visited) < max_pages:
        page_url, depth = queue.popleft()
        if page_url in visited:
            continue
        visited.add(page_url)

        # Politeness delay per domain
        domain = urlparse(page_url).netloc
        now = time.time()
        last = domain_last_request.get(domain, 0)
        wait = delay - (now - last)
        if wait > 0:
            time.sleep(wait)
        domain_last_request[domain] = time.time()

        # Fetch page
        print(f"  Crawling [{depth}]: {page_url}")
        try:
            resp = session.get(page_url, timeout=15)
            resp.raise_for_status()
        except (requests.RequestException, Exception) as e:
            print(f"    Failed: {e}")
            continue

        content_type = resp.headers.get("Content-Type", "")
        if "html" not in content_type:
            continue

        html = resp.text

        # Extract images from this page
        page_images = extract_image_urls(page_url, html)
        for img_url in page_images:
            image_urls.append((img_url, page_url))

        # Follow links if we haven't reached max depth
        if depth < max_depth:
            page_links = extract_page_links(page_url, html,
                                            same_domain=same_domain)
            for link in page_links:
                if link not in visited:
                    queue.append((link, depth + 1))

    print(f"  Crawled {len(visited)} pages, found {len(image_urls)} image URLs")
    return image_urls


# ---------------------------------------------------------------------------
# Google Images search
# ---------------------------------------------------------------------------
def google_image_search(query, session, num_results=50):
    """Scrape Google Images search results for image URLs.

    Returns a list of (image_url, search_url) tuples.
    Note: This scrapes the HTML page and may break if Google changes their
    markup. For production use, consider the Google Custom Search API.
    """
    search_url = "https://www.google.com/search"
    params = {
        "q": query,
        "tbm": "isch",
        "num": min(num_results, 100),
    }

    try:
        resp = session.get(search_url, params=params, timeout=15)
        resp.raise_for_status()
    except (requests.RequestException, Exception) as e:
        print(f"  Google search failed: {e}")
        return []

    # Parse image URLs from the response
    soup = BeautifulSoup(resp.text, "html.parser")
    image_urls = []

    # Google embeds image URLs in various ways; try common patterns
    # Method 1: look for image URLs in <img> tags
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src")
        if not src:
            continue
        if src.startswith("http") and not src.startswith("https://www.google"):
            image_urls.append((src, resp.url))

    # Method 2: look for URLs in script/data attributes containing http
    import re
    for script in soup.find_all("script"):
        text = script.string or ""
        # Find URLs that look like images
        for match in re.findall(r'(https?://[^\s"\'\\]+\.(?:jpg|jpeg|png|webp|bmp|tiff))', text):
            if "google" not in match and "gstatic" not in match:
                image_urls.append((match, resp.url))

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for item in image_urls:
        if item[0] not in seen:
            seen.add(item[0])
            unique.append(item)

    print(f"  Google search found {len(unique)} image URLs for '{query}'")
    return unique[:num_results]


# ---------------------------------------------------------------------------
# Output handling
# ---------------------------------------------------------------------------
@dataclass
class CollectedImage:
    filename: str
    source_url: str
    source_page: str
    score: DocumentScore
    width: int
    height: int


def save_image(image, output_dir, index, url):
    """Save an image to the output directory with a hash-based filename.

    Returns the filename.
    """
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    filename = f"doc_{index:04d}_{url_hash}.png"
    path = os.path.join(output_dir, filename)
    image.save(path)
    return filename


def save_metadata(collected, output_dir):
    """Save metadata for all collected images to metadata.json."""
    records = []
    for item in collected:
        record = {
            "filename": item.filename,
            "source_url": item.source_url,
            "source_page": item.source_page,
            "width": item.width,
            "height": item.height,
            "paragraph_count": item.score.paragraph_count,
            "line_count": item.score.line_count,
            "is_document": item.score.is_document,
            "details": item.score.details,
        }
        records.append(record)

    path = os.path.join(output_dir, "metadata.json")
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved metadata to {path}")


# ---------------------------------------------------------------------------
# Main collection pipeline
# ---------------------------------------------------------------------------
def collect_documents(image_urls, model, device, output_dir,
                      threshold=DEFAULT_SCORE_THRESHOLD, deep_scan=False,
                      max_collect=None, session=None):
    """Download, score, and save document images.

    image_urls: list of (image_url, source_page_url) tuples
    Returns list of CollectedImage for images that passed the threshold.
    """
    if session is None:
        session = requests.Session()
        session.headers["User-Agent"] = USER_AGENT

    os.makedirs(output_dir, exist_ok=True)
    collected = []
    seen_urls = set()
    total = len(image_urls)

    for i, (img_url, source_page) in enumerate(image_urls):
        if max_collect and len(collected) >= max_collect:
            print(f"  Reached max collection limit ({max_collect})")
            break

        # Skip duplicates
        if img_url in seen_urls:
            continue
        seen_urls.add(img_url)

        print(f"  [{i+1}/{total}] {img_url[:80]}...", end=" ", flush=True)

        # Download
        image = fetch_image(img_url, session)
        if image is None:
            print("SKIP (download/validation failed)")
            continue

        # Score
        score = score_document(model, image, device, threshold=threshold,
                               deep_scan=deep_scan)

        if not score.is_document:
            print(f"SKIP (paragraphs={score.paragraph_count}, threshold={threshold})")
            continue

        # Save
        index = len(collected)
        filename = save_image(image, output_dir, index, img_url)
        w, h = image.size
        item = CollectedImage(
            filename=filename,
            source_url=img_url,
            source_page=source_page,
            score=score,
            width=w,
            height=h,
        )
        collected.append(item)
        lines_str = f", lines={score.line_count}" if deep_scan else ""
        print(f"SAVED as {filename} (paragraphs={score.paragraph_count}{lines_str})")

    # Save metadata
    if collected:
        save_metadata(collected, output_dir)

    return collected


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Scrape the web for document images using OCR model scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Crawl a URL and collect document images
  python scrape.py --urls https://example.com/docs --depth 2 --output collected_docs

  # Search Google Images
  python scrape.py --search "scanned document page" --output search_docs

  # Deep scan with strict threshold
  python scrape.py --urls https://arxiv.org --depth 1 --deep-scan --threshold 2
        """,
    )

    # Input modes (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--urls", nargs="+", metavar="URL",
        help="Seed URLs to crawl for document images",
    )
    input_group.add_argument(
        "--search", type=str, metavar="QUERY",
        help="Google Images search query",
    )

    # Crawl options
    parser.add_argument("--depth", type=int, default=1,
                        help="Maximum crawl depth (default: 1)")
    parser.add_argument("--same-domain", action="store_true", default=True,
                        help="Only follow links on the same domain (default)")
    parser.add_argument("--no-same-domain", action="store_true",
                        help="Follow links to other domains")
    parser.add_argument("--max-pages", type=int, default=50,
                        help="Maximum pages to crawl (default: 50)")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                        help=f"Delay between requests to same domain (default: {DEFAULT_DELAY}s)")

    # Scoring options
    parser.add_argument("--threshold", type=int, default=DEFAULT_SCORE_THRESHOLD,
                        help=f"Minimum paragraph count to qualify (default: {DEFAULT_SCORE_THRESHOLD})")
    parser.add_argument("--deep-scan", action="store_true",
                        help="Also detect lines within paragraphs for deeper scoring")

    # Output options
    parser.add_argument("--output", type=str, default="collected_docs",
                        help="Output directory (default: collected_docs)")
    parser.add_argument("--max-collect", type=int, default=None,
                        help="Maximum number of images to collect")

    # Model options
    parser.add_argument("--model-path", type=str, default="model_02.pth",
                        help="Path to trained model checkpoint (default: model_02.pth)")

    # Search options
    parser.add_argument("--num-results", type=int, default=50,
                        help="Number of Google Image search results (default: 50)")

    args = parser.parse_args()

    # Resolve same_domain
    same_domain = not args.no_same_domain

    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_model(args.model_path, device)

    # Setup session
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    # Collect image URLs
    if args.urls:
        print(f"\nCrawling {len(args.urls)} seed URL(s) (depth={args.depth}, "
              f"same_domain={same_domain})...")
        image_urls = crawl(
            args.urls, session,
            max_depth=args.depth,
            same_domain=same_domain,
            max_pages=args.max_pages,
            delay=args.delay,
        )
    else:
        print(f"\nSearching Google Images for: '{args.search}'...")
        image_urls = google_image_search(args.search, session,
                                         num_results=args.num_results)

    if not image_urls:
        print("No image URLs found. Exiting.")
        return

    # Score and collect
    print(f"\nScoring {len(image_urls)} candidate images "
          f"(threshold={args.threshold}, deep_scan={args.deep_scan})...")
    collected = collect_documents(
        image_urls, model, device, args.output,
        threshold=args.threshold,
        deep_scan=args.deep_scan,
        max_collect=args.max_collect,
        session=session,
    )

    print(f"\nDone! Collected {len(collected)} document images in {args.output}/")


if __name__ == "__main__":
    main()
