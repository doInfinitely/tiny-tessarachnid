"""
Mass document crawler for tiny-tessarachnid training data.

Downloads document images and PDFs from curated sources across the web,
converts PDF pages to images, and stores them in S3 with a Supabase
lookup table. Falls back to local storage + SQLite when cloud config
is not provided.

Features:
  - S3 storage for document images (or local filesystem fallback)
  - Supabase lookup table for document metadata (or SQLite fallback)
  - Curated seed URLs targeting document-rich sites (archive.org, gov, arxiv, ...)
  - PDF download with per-page image extraction via PyMuPDF
  - Concurrent downloads via ThreadPoolExecutor
  - SQLite state DB for resumable crawls (crawl queue + visited tracking)
  - BFS link following with configurable depth
  - Politeness: per-domain rate limiting, robots.txt-aware User-Agent
  - Optional OCR model scoring to filter non-document images

Cloud config (env vars or CLI args):
  S3:       --s3-bucket or $S3_BUCKET, --s3-prefix or $S3_PREFIX
  Supabase: --supabase-url or $SUPABASE_URL, --supabase-key or $SUPABASE_KEY

Usage:
    # Local-only (no cloud)
    python crawl_docs.py --output docs_dataset

    # S3 + Supabase
    export S3_BUCKET=my-ocr-docs SUPABASE_URL=https://x.supabase.co SUPABASE_KEY=eyJ...
    python crawl_docs.py --output docs_dataset

    # Explicit CLI args
    python crawl_docs.py --s3-bucket my-ocr-docs --supabase-url https://x.supabase.co \\
        --supabase-key eyJ... --output docs_dataset

    # Resume an interrupted crawl
    python crawl_docs.py --output docs_dataset --resume
"""

import argparse
import hashlib
import io
import json
import os
import re
import sqlite3
import sys
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}
DOC_EXTENSIONS = {".pdf"}
SKIP_EXTENSIONS = {
    ".zip", ".gz", ".tar", ".rar", ".7z", ".bz2", ".xz",
    ".mp3", ".mp4", ".avi", ".mov", ".wmv", ".flac", ".ogg",
    ".exe", ".dmg", ".iso", ".msi", ".deb", ".rpm",
    ".css", ".js", ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".svg", ".ico",
}
MIN_IMAGE_BYTES = 5_000
MIN_IMAGE_DIM = 200
MIN_PDF_BYTES = 10_000
MAX_PDF_BYTES = 200_000_000  # 200MB
MAX_PDF_PAGES = 100
PDF_DPI = 200
USER_AGENT = (
    "Mozilla/5.0 (compatible; TinyTessarachnid/2.0; "
    "+https://github.com/tiny-tessarachnid)"
)

# ---------------------------------------------------------------------------
# Curated seed URLs — document-rich sites
# ---------------------------------------------------------------------------
SEED_URLS = [
    # Internet Archive — scanned books & documents
    "https://archive.org/details/texts",
    "https://archive.org/details/americana",
    "https://archive.org/details/gutenberg",
    # US Government documents
    "https://www.govinfo.gov/browse",
    "https://www.gpo.gov/fdsys/browse/collection.action?collectionCode=FR",
    # Academic papers
    "https://arxiv.org/list/cs.CV/recent",
    "https://arxiv.org/list/cs.CL/recent",
    "https://arxiv.org/list/cs.AI/recent",
    # Wikipedia (lots of text-heavy pages)
    "https://en.wikipedia.org/wiki/Special:Random",
    "https://en.wikipedia.org/wiki/Main_Page",
    # Project Gutenberg
    "https://www.gutenberg.org/browse/scores/top",
    # Wikisource — scanned historical documents
    "https://en.wikisource.org/wiki/Main_Page",
    # Smithsonian Open Access
    "https://www.si.edu/openaccess",
    # Library of Congress
    "https://www.loc.gov/collections/",
    # Europeana — European cultural heritage
    "https://www.europeana.eu/en/collections",
    # HathiTrust — scanned books
    "https://www.hathitrust.org/",
    # Common Crawl sample pages
    "https://commoncrawl.org/the-data/examples",
]

# Search queries for Google Images to find document-like images
SEARCH_QUERIES = [
    "scanned document page",
    "old book page scan",
    "handwritten letter scan",
    "typed document scan",
    "newspaper page scan",
    "legal document scan",
    "medical form scan",
    "invoice receipt scan",
    "academic paper first page",
    "historical manuscript page",
    "typewritten document",
    "printed text page",
    "government form scan",
    "tax form document",
    "birth certificate scan",
    "business letter typed",
    "scientific paper page",
    "book text page photograph",
    "old newspaper clipping",
    "handwritten notes page",
]


# ---------------------------------------------------------------------------
# SQLite state database
# ---------------------------------------------------------------------------
class CrawlDB:
    """SQLite database for tracking crawl state and collected documents."""

    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = Lock()
        self._init_tables()

    def _init_tables(self):
        with self.lock:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS visited_pages (
                    url TEXT PRIMARY KEY,
                    status TEXT,
                    timestamp REAL
                );
                CREATE TABLE IF NOT EXISTS visited_assets (
                    url TEXT PRIMARY KEY,
                    asset_type TEXT,
                    status TEXT,
                    filename TEXT,
                    pages_extracted INTEGER DEFAULT 0,
                    timestamp REAL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    filename TEXT PRIMARY KEY,
                    source_url TEXT,
                    source_page TEXT,
                    width INTEGER,
                    height INTEGER,
                    source_type TEXT,
                    pdf_page INTEGER DEFAULT -1,
                    paragraph_count INTEGER DEFAULT -1,
                    timestamp REAL
                );
                CREATE TABLE IF NOT EXISTS crawl_queue (
                    url TEXT PRIMARY KEY,
                    depth INTEGER,
                    added REAL
                );
            """)
            self.conn.commit()

    def page_visited(self, url):
        with self.lock:
            row = self.conn.execute(
                "SELECT 1 FROM visited_pages WHERE url=?", (url,)
            ).fetchone()
            return row is not None

    def mark_page(self, url, status="ok"):
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO visited_pages (url, status, timestamp) "
                "VALUES (?, ?, ?)",
                (url, status, time.time()),
            )
            self.conn.commit()

    def asset_visited(self, url):
        with self.lock:
            row = self.conn.execute(
                "SELECT 1 FROM visited_assets WHERE url=?", (url,)
            ).fetchone()
            return row is not None

    def mark_asset(self, url, asset_type, status, filename=None,
                   pages_extracted=0):
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO visited_assets "
                "(url, asset_type, status, filename, pages_extracted, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (url, asset_type, status, filename, pages_extracted, time.time()),
            )
            self.conn.commit()

    def add_document(self, filename, source_url, source_page, width, height,
                     source_type="image", pdf_page=-1, paragraph_count=-1):
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO documents "
                "(filename, source_url, source_page, width, height, "
                "source_type, pdf_page, paragraph_count, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (filename, source_url, source_page, width, height,
                 source_type, pdf_page, paragraph_count, time.time()),
            )
            self.conn.commit()

    def document_count(self):
        with self.lock:
            row = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()
            return row[0]

    def enqueue_pages(self, url_depth_pairs):
        with self.lock:
            now = time.time()
            self.conn.executemany(
                "INSERT OR IGNORE INTO crawl_queue (url, depth, added) "
                "VALUES (?, ?, ?)",
                [(url, depth, now) for url, depth in url_depth_pairs],
            )
            self.conn.commit()

    def dequeue_page(self):
        with self.lock:
            row = self.conn.execute(
                "SELECT url, depth FROM crawl_queue ORDER BY added LIMIT 1"
            ).fetchone()
            if row:
                self.conn.execute(
                    "DELETE FROM crawl_queue WHERE url=?", (row[0],)
                )
                self.conn.commit()
            return row

    def queue_size(self):
        with self.lock:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM crawl_queue"
            ).fetchone()
            return row[0]

    def close(self):
        self.conn.close()


# ---------------------------------------------------------------------------
# S3 storage backend
# ---------------------------------------------------------------------------
class S3Storage:
    """Uploads document images to an S3 bucket."""

    def __init__(self, bucket, prefix="documents/", region=None):
        import boto3
        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/"
        kwargs = {}
        if region:
            kwargs["region_name"] = region
        self.s3 = boto3.client("s3", **kwargs)
        self._lock = Lock()

    def upload_image(self, img, filename):
        """Upload a PIL Image as PNG to S3. Returns the full S3 key."""
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        key = self.prefix + filename
        with self._lock:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=buf.getvalue(),
                ContentType="image/png",
            )
        return key

    def get_url(self, key):
        """Return the public HTTPS URL for an S3 key."""
        return f"https://{self.bucket}.s3.amazonaws.com/{key}"


class LocalStorage:
    """Saves document images to the local filesystem (fallback)."""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def upload_image(self, img, filename):
        path = os.path.join(self.output_dir, filename)
        img.save(path)
        return filename

    def get_url(self, key):
        return key


# ---------------------------------------------------------------------------
# Supabase document registry
# ---------------------------------------------------------------------------
class SupabaseRegistry:
    """Stores document metadata in a Supabase table.

    Expected table schema (create via Supabase SQL editor):

        create table if not exists documents (
            filename    text primary key,
            s3_key      text,
            s3_url      text,
            source_url  text,
            source_page text,
            width       integer,
            height      integer,
            source_type text default 'image',
            pdf_page    integer default -1,
            paragraph_count integer default -1,
            created_at  timestamptz default now()
        );
    """

    TABLE = "documents"

    def __init__(self, url, key):
        from supabase import create_client
        self.client = create_client(url, key)
        self._lock = Lock()

    def upsert(self, filename, s3_key, s3_url, source_url, source_page,
               width, height, source_type="image", pdf_page=-1,
               paragraph_count=-1):
        row = {
            "filename": filename,
            "s3_key": s3_key,
            "s3_url": s3_url,
            "source_url": source_url,
            "source_page": source_page,
            "width": width,
            "height": height,
            "source_type": source_type,
            "pdf_page": pdf_page,
            "paragraph_count": paragraph_count,
        }
        with self._lock:
            self.client.table(self.TABLE).upsert(row).execute()

    def count(self):
        with self._lock:
            resp = self.client.table(self.TABLE).select(
                "*", count="exact"
            ).limit(0).execute()
            return resp.count or 0

    def update_score(self, filename, paragraph_count):
        with self._lock:
            self.client.table(self.TABLE).update(
                {"paragraph_count": paragraph_count}
            ).eq("filename", filename).execute()

    def delete(self, filename):
        with self._lock:
            self.client.table(self.TABLE).delete().eq(
                "filename", filename
            ).execute()


class NullRegistry:
    """No-op registry when Supabase is not configured."""

    def upsert(self, *args, **kwargs):
        pass

    def count(self):
        return 0

    def update_score(self, *args, **kwargs):
        pass

    def delete(self, *args, **kwargs):
        pass


# ---------------------------------------------------------------------------
# PDF to images
# ---------------------------------------------------------------------------
def pdf_to_images(pdf_bytes, dpi=PDF_DPI, max_pages=MAX_PDF_PAGES):
    """Convert PDF bytes to a list of PIL RGB images using PyMuPDF.

    Returns list of (page_index, PIL.Image) tuples.
    """
    try:
        import fitz
    except ImportError:
        print("Warning: PyMuPDF not installed, skipping PDF. pip install PyMuPDF")
        return []

    images = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        n_pages = min(len(doc), max_pages)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        for page_idx in range(n_pages):
            page = doc[page_idx]
            pix = page.get_pixmap(matrix=matrix)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            if img.size[0] >= MIN_IMAGE_DIM and img.size[1] >= MIN_IMAGE_DIM:
                images.append((page_idx, img))
        doc.close()
    except Exception as e:
        print(f"    PDF parse error: {e}")

    return images


# ---------------------------------------------------------------------------
# Asset fetching
# ---------------------------------------------------------------------------
def make_session():
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    session.headers["Accept"] = (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/*,application/pdf;q=0.8,*/*;q=0.7"
    )
    return session


def fetch_page(url, session, timeout=20):
    """Fetch an HTML page. Returns (html_text, final_url) or (None, None)."""
    try:
        resp = session.get(url, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        ct = resp.headers.get("Content-Type", "")
        if "html" not in ct:
            return None, None
        return resp.text, resp.url
    except Exception:
        return None, None


def fetch_asset_bytes(url, session, max_bytes=MAX_PDF_BYTES, timeout=30):
    """Download raw bytes from a URL with size limit. Returns bytes or None."""
    try:
        resp = session.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        cl = resp.headers.get("Content-Length")
        if cl and int(cl) > max_bytes:
            return None
        chunks = []
        total = 0
        for chunk in resp.iter_content(chunk_size=65536):
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                return None
        return b"".join(chunks)
    except Exception:
        return None


def bytes_to_image(data):
    """Try to open raw bytes as an RGB PIL Image. Returns None on failure."""
    if len(data) < MIN_IMAGE_BYTES:
        return None
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        if img.size[0] < MIN_IMAGE_DIM or img.size[1] < MIN_IMAGE_DIM:
            return None
        return img
    except Exception:
        return None


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------
def extract_assets(page_url, html):
    """Extract image URLs and PDF URLs from an HTML page.

    Returns (image_urls, pdf_urls, page_links) — all absolute URLs.
    """
    soup = BeautifulSoup(html, "html.parser")
    images = set()
    pdfs = set()
    links = set()

    base_domain = urlparse(page_url).netloc

    for tag in soup.find_all(["img", "a", "source", "meta"]):
        if tag.name == "img":
            src = tag.get("src") or tag.get("data-src") or tag.get("data-original")
            if src:
                images.add(urljoin(page_url, src))
        elif tag.name == "source":
            src = tag.get("srcset")
            if src:
                for part in src.split(","):
                    url_part = part.strip().split()[0]
                    if url_part:
                        images.add(urljoin(page_url, url_part))
        elif tag.name == "meta":
            prop = tag.get("property", "") or tag.get("name", "")
            if prop in ("og:image", "twitter:image"):
                content = tag.get("content")
                if content:
                    images.add(urljoin(page_url, content))
        elif tag.name == "a" and tag.get("href"):
            href = tag["href"]
            absolute = urljoin(page_url, href)
            parsed = urlparse(absolute)
            if parsed.scheme not in ("http", "https"):
                continue
            ext = os.path.splitext(parsed.path)[1].lower()
            if ext in DOC_EXTENSIONS:
                pdfs.add(absolute)
            elif ext in IMAGE_EXTENSIONS:
                images.add(absolute)
            elif ext not in SKIP_EXTENSIONS:
                clean = parsed._replace(fragment="").geturl()
                links.add(clean)

    return list(images), list(pdfs), list(links)


def google_image_urls(query, session, num_results=100):
    """Scrape Google Images for image URLs matching a query."""
    search_url = "https://www.google.com/search"
    params = {"q": query, "tbm": "isch", "num": min(num_results, 100)}

    try:
        resp = session.get(search_url, params=params, timeout=15)
        resp.raise_for_status()
    except Exception:
        return []

    urls = set()
    soup = BeautifulSoup(resp.text, "html.parser")

    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src")
        if src and src.startswith("http") and "google" not in src:
            urls.add(src)

    for script in soup.find_all("script"):
        text = script.string or ""
        for match in re.findall(
            r'(https?://[^\s"\'\\]+\.(?:jpg|jpeg|png|webp|bmp|tiff))', text
        ):
            if "google" not in match and "gstatic" not in match:
                urls.add(match)

    return list(urls)[:num_results]


# ---------------------------------------------------------------------------
# Filename generation
# ---------------------------------------------------------------------------
def _url_hash(url):
    return hashlib.md5(url.encode()).hexdigest()[:12]


def make_image_filename(url, index, suffix=""):
    h = _url_hash(url)
    return f"doc_{index:06d}_{h}{suffix}.png"


def make_pdf_page_filename(url, page_idx, index):
    h = _url_hash(url)
    return f"pdf_{index:06d}_{h}_p{page_idx:03d}.png"


# ---------------------------------------------------------------------------
# Worker: process a single asset (image or PDF)
# ---------------------------------------------------------------------------
@dataclass
class AssetJob:
    url: str
    asset_type: str  # "image" or "pdf"
    source_page: str


def process_asset(job, session, storage, db, registry, counter_lock, counter):
    """Download and save a single image or PDF. Thread-safe.

    Args:
        storage: S3Storage or LocalStorage instance
        db: CrawlDB (SQLite) for crawl state tracking
        registry: SupabaseRegistry or NullRegistry for the lookup table
    """
    if db.asset_visited(job.url):
        return 0

    saved = 0

    if job.asset_type == "image":
        data = fetch_asset_bytes(job.url, session, max_bytes=50_000_000)
        if data is None:
            db.mark_asset(job.url, "image", "failed")
            return 0
        img = bytes_to_image(data)
        if img is None:
            db.mark_asset(job.url, "image", "skip_small")
            return 0

        with counter_lock:
            idx = counter[0]
            counter[0] += 1
        fname = make_image_filename(job.url, idx)
        s3_key = storage.upload_image(img, fname)
        s3_url = storage.get_url(s3_key)
        w, h = img.size
        db.add_document(fname, job.url, job.source_page, w, h,
                        source_type="image")
        registry.upsert(fname, s3_key, s3_url, job.url, job.source_page,
                        w, h, source_type="image")
        db.mark_asset(job.url, "image", "saved", filename=fname)
        saved = 1

    elif job.asset_type == "pdf":
        data = fetch_asset_bytes(job.url, session, max_bytes=MAX_PDF_BYTES)
        if data is None or len(data) < MIN_PDF_BYTES:
            db.mark_asset(job.url, "pdf", "failed")
            return 0

        pages = pdf_to_images(data)
        if not pages:
            db.mark_asset(job.url, "pdf", "no_pages")
            return 0

        for page_idx, img in pages:
            with counter_lock:
                idx = counter[0]
                counter[0] += 1
            fname = make_pdf_page_filename(job.url, page_idx, idx)
            s3_key = storage.upload_image(img, fname)
            s3_url = storage.get_url(s3_key)
            w, h = img.size
            db.add_document(fname, job.url, job.source_page, w, h,
                            source_type="pdf", pdf_page=page_idx)
            registry.upsert(fname, s3_key, s3_url, job.url, job.source_page,
                            w, h, source_type="pdf", pdf_page=page_idx)
            saved += 1

        db.mark_asset(job.url, "pdf", "saved", pages_extracted=len(pages))

    return saved


# ---------------------------------------------------------------------------
# Main crawler
# ---------------------------------------------------------------------------
class DocumentCrawler:
    """BFS web crawler that collects document images and PDFs."""

    def __init__(self, output_dir, db, session, storage, registry,
                 max_depth=2, same_domain=True, max_pages=200, delay=0.5,
                 workers=4, max_docs=None):
        self.output_dir = output_dir
        self.db = db
        self.session = session
        self.storage = storage
        self.registry = registry
        self.max_depth = max_depth
        self.same_domain = same_domain
        self.max_pages = max_pages
        self.delay = delay
        self.workers = workers
        self.max_docs = max_docs

        self.domain_last_req = {}
        self.domain_lock = Lock()
        self.counter = [db.document_count()]
        self.counter_lock = Lock()
        self.pages_crawled = 0

    def _rate_limit(self, url):
        domain = urlparse(url).netloc
        with self.domain_lock:
            now = time.time()
            last = self.domain_last_req.get(domain, 0)
            wait = self.delay - (now - last)
            if wait > 0:
                time.sleep(wait)
            self.domain_last_req[domain] = time.time()

    def crawl(self, seed_urls):
        """Run BFS crawl from seed URLs, collecting assets along the way."""
        # Seed the queue
        self.db.enqueue_pages([(url, 0) for url in seed_urls])

        pending_assets = []

        while self.pages_crawled < self.max_pages:
            if self.max_docs and self.db.document_count() >= self.max_docs:
                print(f"\n  Reached max document limit ({self.max_docs})")
                break

            entry = self.db.dequeue_page()
            if entry is None:
                break
            page_url, depth = entry

            if self.db.page_visited(page_url):
                continue

            self._rate_limit(page_url)
            self.pages_crawled += 1
            n_docs = self.db.document_count()
            q_size = self.db.queue_size()
            print(
                f"  [{self.pages_crawled}/{self.max_pages}] "
                f"depth={depth} docs={n_docs} queue={q_size} | "
                f"{page_url[:90]}",
                flush=True,
            )

            html, final_url = fetch_page(page_url, self.session)
            if html is None:
                self.db.mark_page(page_url, "failed")
                continue
            self.db.mark_page(page_url, "ok")

            # Extract assets and links
            images, pdfs, page_links = extract_assets(final_url, html)

            # Queue image and PDF jobs
            for img_url in images:
                if not self.db.asset_visited(img_url):
                    pending_assets.append(
                        AssetJob(img_url, "image", page_url))
            for pdf_url in pdfs:
                if not self.db.asset_visited(pdf_url):
                    pending_assets.append(
                        AssetJob(pdf_url, "pdf", page_url))

            # Process asset batch when we have enough
            if len(pending_assets) >= self.workers * 4:
                self._process_batch(pending_assets)
                pending_assets = []

            # Follow links
            if depth < self.max_depth:
                new_links = []
                base_domain = urlparse(page_url).netloc
                for link in page_links:
                    if self.db.page_visited(link):
                        continue
                    if self.same_domain and urlparse(link).netloc != base_domain:
                        continue
                    new_links.append((link, depth + 1))
                if new_links:
                    self.db.enqueue_pages(new_links)

        # Flush remaining assets
        if pending_assets:
            self._process_batch(pending_assets)

        n_docs = self.db.document_count()
        print(f"\n  Crawl complete: {self.pages_crawled} pages, {n_docs} documents")

    def _process_batch(self, jobs):
        """Process a batch of asset jobs concurrently."""
        saved_total = 0
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {
                pool.submit(
                    process_asset, job, make_session(),
                    self.storage, self.db, self.registry,
                    self.counter_lock, self.counter,
                ): job
                for job in jobs
            }
            for future in as_completed(futures):
                job = futures[future]
                try:
                    saved = future.result()
                    saved_total += saved
                    if saved > 0:
                        kind = "PDF pages" if job.asset_type == "pdf" else "image"
                        n_docs = self.db.document_count()
                        print(
                            f"    + {saved} {kind} from "
                            f"{job.url[:70]}... (total={n_docs})",
                            flush=True,
                        )
                except Exception as e:
                    print(f"    ! Error processing {job.url[:70]}: {e}")

    def search_and_collect(self, queries):
        """Run Google Image searches and download results."""
        all_urls = []
        for query in queries:
            print(f"  Searching: '{query}'...", flush=True)
            urls = google_image_urls(query, self.session)
            print(f"    Found {len(urls)} image URLs")
            all_urls.extend(
                AssetJob(url, "image", f"google:{query}")
                for url in urls
                if not self.db.asset_visited(url)
            )

        if all_urls:
            print(f"  Downloading {len(all_urls)} search results...")
            self._process_batch(all_urls)


# ---------------------------------------------------------------------------
# Export metadata
# ---------------------------------------------------------------------------
def export_metadata(db, output_dir):
    """Export document metadata from DB to metadata.json."""
    rows = db.conn.execute(
        "SELECT filename, source_url, source_page, width, height, "
        "source_type, pdf_page, paragraph_count, timestamp "
        "FROM documents ORDER BY timestamp"
    ).fetchall()

    records = []
    for row in rows:
        records.append({
            "filename": row[0],
            "source_url": row[1],
            "source_page": row[2],
            "width": row[3],
            "height": row[4],
            "source_type": row[5],
            "pdf_page": row[6],
            "paragraph_count": row[7],
            "timestamp": row[8],
        })

    path = os.path.join(output_dir, "metadata.json")
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Exported metadata for {len(records)} documents to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Mass document crawler for OCR training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--output", type=str, default="docs_dataset",
                        help="Output directory (default: docs_dataset)")
    parser.add_argument("--urls", nargs="*", metavar="URL",
                        help="Custom seed URLs (uses built-in list if omitted)")
    parser.add_argument("--depth", type=int, default=2,
                        help="Max crawl depth from seed URLs (default: 2)")
    parser.add_argument("--max-pages", type=int, default=200,
                        help="Max pages to crawl (default: 200)")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Stop after collecting this many documents")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent download threads (default: 4)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Per-domain request delay in seconds (default: 0.5)")
    parser.add_argument("--cross-domain", action="store_true",
                        help="Follow links to other domains")
    parser.add_argument("--resume", action="store_true",
                        help="Resume a previous crawl (skip re-seeding)")
    parser.add_argument("--search", action="store_true",
                        help="Also run Google Image searches for documents")
    parser.add_argument("--search-queries", nargs="*",
                        help="Custom search queries (uses built-in list if omitted)")
    parser.add_argument("--no-crawl", action="store_true",
                        help="Skip web crawling (only search)")

    # S3 storage
    parser.add_argument("--s3-bucket", type=str,
                        default=os.environ.get("S3_BUCKET"),
                        help="S3 bucket name (or $S3_BUCKET env var)")
    parser.add_argument("--s3-prefix", type=str,
                        default=os.environ.get("S3_PREFIX", "documents/"),
                        help="S3 key prefix (default: documents/)")
    parser.add_argument("--s3-region", type=str,
                        default=os.environ.get("AWS_REGION"),
                        help="AWS region (or $AWS_REGION)")

    # Supabase registry
    parser.add_argument("--supabase-url", type=str,
                        default=os.environ.get("SUPABASE_URL"),
                        help="Supabase project URL (or $SUPABASE_URL)")
    parser.add_argument("--supabase-key", type=str,
                        default=os.environ.get("SUPABASE_KEY"),
                        help="Supabase anon/service key (or $SUPABASE_KEY)")

    # Model scoring (optional)
    parser.add_argument("--score", action="store_true",
                        help="Score images with OCR model (slower but filters junk)")
    parser.add_argument("--model-path", type=str, default="model_02.pth",
                        help="Path to OCR model for scoring")
    parser.add_argument("--score-threshold", type=int, default=1,
                        help="Min paragraph count to keep (default: 1)")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    db_path = os.path.join(args.output, "crawl.db")
    db = CrawlDB(db_path)

    # Initialize storage backend
    if args.s3_bucket:
        storage = S3Storage(args.s3_bucket, args.s3_prefix,
                            region=args.s3_region)
        storage_label = f"s3://{args.s3_bucket}/{args.s3_prefix}"
    else:
        storage = LocalStorage(args.output)
        storage_label = f"local ({args.output})"

    # Initialize registry backend
    if args.supabase_url and args.supabase_key:
        registry = SupabaseRegistry(args.supabase_url, args.supabase_key)
        registry_label = args.supabase_url
    else:
        registry = NullRegistry()
        registry_label = "none (SQLite only)"

    session = make_session()
    same_domain = not args.cross_domain

    print(f"Output directory: {args.output}")
    print(f"Storage: {storage_label}")
    print(f"Registry: {registry_label}")
    print(f"Database: {db_path}")
    print(f"Existing documents: {db.document_count()}")
    print(f"Workers: {args.workers}, Delay: {args.delay}s, "
          f"Max pages: {args.max_pages}, Depth: {args.depth}")
    if args.max_docs:
        print(f"Max documents: {args.max_docs}")

    crawler = DocumentCrawler(
        output_dir=args.output,
        db=db,
        session=session,
        storage=storage,
        registry=registry,
        max_depth=args.depth,
        same_domain=same_domain,
        max_pages=args.max_pages,
        delay=args.delay,
        workers=args.workers,
        max_docs=args.max_docs,
    )

    # Phase 1: Web crawling
    if not args.no_crawl:
        seed = args.urls if args.urls else SEED_URLS
        if args.resume and db.queue_size() > 0:
            print(f"\nResuming crawl ({db.queue_size()} URLs in queue)...")
        else:
            print(f"\nSeeding with {len(seed)} URLs...")
        crawler.crawl(seed)

    # Phase 2: Google Image search
    if args.search or args.search_queries:
        queries = args.search_queries if args.search_queries else SEARCH_QUERIES
        print(f"\nRunning {len(queries)} Google Image searches...")
        crawler.search_and_collect(queries)

    # Phase 3: Optional model-based scoring/filtering
    if args.score:
        print("\nScoring documents with OCR model...")
        _score_documents(args, db, registry)

    # Export metadata
    export_metadata(db, args.output)
    total = db.document_count()
    db.close()

    print(f"\nDone! Total documents: {total}")


def _score_documents(args, db, registry):
    """Score all unscored documents with the OCR model and remove non-documents."""
    import torch
    from generate_training_data import scale_and_pad, CLASS_NONE, PREV_BBOX_NONE
    from scrape import load_model, score_document

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, device)

    rows = db.conn.execute(
        "SELECT filename FROM documents WHERE paragraph_count = -1"
    ).fetchall()

    print(f"  Scoring {len(rows)} unscored documents...")
    removed = 0
    for i, (fname,) in enumerate(rows):
        path = os.path.join(args.output, fname)
        if not os.path.exists(path):
            continue
        try:
            img = Image.open(path).convert("RGB")
            result = score_document(model, img, device,
                                    threshold=args.score_threshold)
            if result.is_document:
                db.conn.execute(
                    "UPDATE documents SET paragraph_count=? WHERE filename=?",
                    (result.paragraph_count, fname),
                )
                registry.update_score(fname, result.paragraph_count)
            else:
                db.conn.execute(
                    "DELETE FROM documents WHERE filename=?", (fname,),
                )
                registry.delete(fname)
                os.remove(path)
                removed += 1
        except Exception as e:
            print(f"    Error scoring {fname}: {e}")

        if (i + 1) % 50 == 0:
            db.conn.commit()
            print(f"    Scored {i+1}/{len(rows)} (removed {removed})")

    db.conn.commit()
    print(f"  Scoring complete. Removed {removed} non-documents.")


if __name__ == "__main__":
    main()
