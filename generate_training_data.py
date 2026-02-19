"""
Synthetic OCR training data generator for hierarchical retina-based detection.

Generates training data for a model that operates on a fixed 1024x1024 "retina"
and detects items sequentially: paragraphs on a page, lines in a paragraph,
words in a line, characters in a word.

Uses teacher forcing: the previous ground truth bbox is provided as model input.
The image stays unchanged throughout the autoregressive loop — the model relies
on prev_bbox to determine what to detect next (spatial ordering: top-to-bottom
for paragraphs/lines, left-to-right for words/characters).

Pages use random background colors and each paragraph gets a random text color.
Each paragraph also carries an is_handwritten flag based on the rendering font.

Class tokens (~535 total, dict-based encoding):
  0     = NONE (nothing left to detect)
  1     = PAGE
  2     = PARAGRAPH
  3     = LINE
  4     = WORD
  5+    = characters via CHAR_TO_CLASS dict (~530 chars: ASCII first, then
          Latin-1 Supplement, Latin Extended-A, Greek, math, arrows, etc.)
"""

import math
import os
import sys
import random
import string
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

from create_lists import create_character_list

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RETINA_SIZE = 1024
CLASS_NONE = 0
CLASS_PAGE = 1
CLASS_PARAGRAPH = 2
CLASS_LINE = 3
CLASS_WORD = 4
CHAR_CLASS_OFFSET = 5

PRINTABLE_CHARS = create_character_list()  # ~530 chars, ASCII first
CHAR_TO_CLASS = {ch: i + CHAR_CLASS_OFFSET for i, ch in enumerate(PRINTABLE_CHARS)}
CLASS_TO_CHAR = {v: k for k, v in CHAR_TO_CLASS.items()}
NUM_CLASSES = CHAR_CLASS_OFFSET + len(PRINTABLE_CHARS)

# ASCII subset for synthetic text generation (fonts validated for ASCII only)
ASCII_CHARS = [ch for ch in PRINTABLE_CHARS if 32 <= ord(ch) <= 126]
PREV_BBOX_NONE = (0, 0, 0, 0, CLASS_NONE, 0)  # 6-dim: (x1,y1,x2,y2,class_id,is_handwritten)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
def build_augmentation():
    """Photometric augmentations only — no spatial transforms needed since
    bbox coords are separate from the image tensor."""
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.RandomGrayscale(p=0.1),
    ])


class AugmentedSubset(Dataset):
    """Wraps a dataset subset and applies photometric augmentation to images."""

    def __init__(self, subset, augmentation):
        self.subset = subset
        self.augmentation = augmentation

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img_t, prev_t, target_t = self.subset[idx]
        # img_t is CHW float [0,1] — convert to PIL, augment, convert back
        img_pil = transforms.ToPILImage()(img_t)
        img_pil = self.augmentation(img_pil)
        img_t = transforms.ToTensor()(img_pil)
        return img_t, prev_t, target_t


def char_to_class(ch):
    return CHAR_TO_CLASS[ch]


def class_to_label(class_id):
    if class_id == CLASS_NONE:
        return "NONE"
    if class_id == CLASS_PAGE:
        return "PAGE"
    if class_id == CLASS_PARAGRAPH:
        return "PARA"
    if class_id == CLASS_LINE:
        return "LINE"
    if class_id == CLASS_WORD:
        return "WORD"
    return repr(CLASS_TO_CHAR.get(class_id, "?"))


def _random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def _contrasting_color(bg):
    """Pick a text color with sufficient luminance contrast from *bg*."""
    bg_lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    for _ in range(50):
        fg = _random_color()
        fg_lum = 0.299 * fg[0] + 0.587 * fg[1] + 0.114 * fg[2]
        if abs(bg_lum - fg_lum) > 80:
            return fg
    return (0, 0, 0) if bg_lum > 128 else (255, 255, 255)


# ---------------------------------------------------------------------------
# Font classification
# ---------------------------------------------------------------------------
_HANDWRITING_FONT_KEYWORDS = (
    "Bradley Hand", "Brush Script", "Chalkduster", "Comic Sans", "Zapfino",
    "Apple Chancery", "Trattatello", "PartyLET", "Farisi", "Snell Roundhand",
    "SignPainter", "Segoe Script", "Kristen", "Marker Felt",
)


def _is_handwriting_font(font_path):
    """Check if a font file appears to be a handwriting/script font."""
    basename = os.path.basename(font_path)
    return any(kw.lower() in basename.lower() for kw in _HANDWRITING_FONT_KEYWORDS)


# ---------------------------------------------------------------------------
# 1. discover_fonts()
# ---------------------------------------------------------------------------
def _font_renders_latin(font_path, size=24):
    """Return True if *font_path* renders all printable ASCII glyphs.

    Tests every printable ASCII character (32-126). Rejects fonts where
    any character produces no pixels or the same tofu box as others.
    """
    try:
        font = ImageFont.truetype(font_path, size)
    except Exception:
        return False

    # Render a known-good reference to detect tofu boxes
    ref_img = Image.new("L", (size * 2, size * 2), 0)
    ref_draw = ImageDraw.Draw(ref_img)
    ref_draw.text((0, 0), "A", fill=255, font=font)
    ref_arr = np.array(ref_img)
    if ref_arr.max() == 0:
        return False

    # Test all printable ASCII characters
    for code in range(33, 127):  # '!' through '~'
        ch = chr(code)
        img = Image.new("L", (size * 2, size * 2), 0)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), ch, fill=255, font=font)
        arr = np.array(img)
        if arr.max() == 0:
            return False  # character renders as blank

    # Check that a few visually distinct chars don't all produce the same image
    test_chars = "A", "g", "{", "~"
    imgs = []
    for ch in test_chars:
        img = Image.new("L", (size * 2, size * 2), 0)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), ch, fill=255, font=font)
        imgs.append(np.array(img))
    for i in range(1, len(imgs)):
        if not np.array_equal(imgs[0], imgs[i]):
            return True
    return False


def discover_fonts():
    """Scan the project fonts/ directory for usable Latin font files."""
    search_dirs = [
        os.path.join(os.path.dirname(__file__), "fonts"),
    ]
    # Skip font families that don't have proper Latin ASCII glyphs
    _skip_prefixes = (
        "Apple Braille", "Apple Symbols", "Bodoni Ornaments",
        "Diwan", "Farisi", "Gurmukhi", "Hoefler Text Ornaments",
        "Keyboard", "Khmer", "Kokonor", "Lao", "Mishafi",
        "NotoSans", "NotoSerif", "PlantagenetCherokee",
        "SFArabic", "SFArmenian", "SFCamera", "SFGeorgian",
        "SFHebrew", "Symbol", "Webdings", "Wingdings", "ZapfDingbats",
    )
    extensions = {".ttf", ".otf", ".ttc"}
    found = []
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for root, _dirs, files in os.walk(d):
            for f in files:
                if os.path.splitext(f)[1].lower() in extensions:
                    if any(f.startswith(p) for p in _skip_prefixes):
                        continue
                    path = os.path.join(root, f)
                    if _font_renders_latin(path):
                        found.append(path)
    return found


# ---------------------------------------------------------------------------
# 2. scale_and_pad()
# ---------------------------------------------------------------------------
def scale_and_pad(image, pad_color=(255, 255, 255)):
    """Scale *image* to fit RETINA_SIZE x RETINA_SIZE, center on canvas.

    Returns (padded_image, scale, offset_x, offset_y).
    """
    w, h = image.size
    if w == 0 or h == 0:
        canvas = Image.new("RGB", (RETINA_SIZE, RETINA_SIZE), pad_color)
        return canvas, 1.0, 0, 0
    scale = min(RETINA_SIZE / w, RETINA_SIZE / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (RETINA_SIZE, RETINA_SIZE), pad_color)
    ox = (RETINA_SIZE - new_w) // 2
    oy = (RETINA_SIZE - new_h) // 2
    canvas.paste(resized, (ox, oy))
    return canvas, scale, ox, oy


# ---------------------------------------------------------------------------
# 3. bbox_to_retina()
# ---------------------------------------------------------------------------
def bbox_to_retina(bbox, scale, ox, oy):
    """Map an original-image bbox (x1,y1,x2,y2) to retina coordinates."""
    x1, y1, x2, y2 = bbox
    return (
        int(x1 * scale + ox),
        int(y1 * scale + oy),
        int(x2 * scale + ox),
        int(y2 * scale + oy),
    )


# ---------------------------------------------------------------------------
# 4. Mask-based erasure
# ---------------------------------------------------------------------------
def _compute_mask(image, bbox, bg_color):
    """Compute a grayscale mask of non-background pixels in *bbox*.

    Returns an 'L' mode PIL Image where 255 = content pixel, 0 = background.
    Uses per-channel max-difference so anti-aliased edges are included.
    """
    x1, y1, x2, y2 = bbox
    region = image.crop((x1, y1, x2, y2))
    arr = np.array(region, dtype=np.int16)
    bg = np.array(bg_color, dtype=np.int16)
    diff = np.abs(arr - bg).max(axis=2).astype(np.uint8)
    # Any pixel that differs from bg at all gets mask=255
    mask_arr = np.where(diff > 0, 255, 0).astype(np.uint8)
    return Image.fromarray(mask_arr, mode="L")


def erase_region(image, bbox, mask, bg_color):
    """Erase content pixels in *bbox* using *mask*, restoring them to *bg_color*.

    Only pixels where mask > 0 are painted over. Returns a copy.
    """
    image = image.copy()
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return image
    bg_patch = Image.new("RGB", (w, h), bg_color)
    if mask.size != (w, h):
        mask = mask.resize((w, h), Image.NEAREST)
    image.paste(bg_patch, (x1, y1), mask)
    return image


# ---------------------------------------------------------------------------
# 5. Contour extraction (V2)
# ---------------------------------------------------------------------------
def _convex_hull(points):
    """Andrew's monotone chain convex hull. Returns CW-ordered vertices."""
    pts = sorted(set(map(tuple, points)))
    if len(pts) <= 1:
        return pts
    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross2d(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross2d(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _cross2d(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _rdp_simplify(points, epsilon):
    """Ramer-Douglas-Peucker polyline simplification."""
    if len(points) <= 2:
        return list(points)
    start, end = np.array(points[0], dtype=np.float64), np.array(points[-1], dtype=np.float64)
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-12:
        dists = [np.linalg.norm(np.array(p, dtype=np.float64) - start) for p in points]
    else:
        line_unit = line_vec / line_len
        dists = []
        for p in points:
            v = np.array(p, dtype=np.float64) - start
            dists.append(abs(v[0] * line_unit[1] - v[1] * line_unit[0]))
    idx = int(np.argmax(dists))
    max_dist = dists[idx]
    if max_dist > epsilon:
        left = _rdp_simplify(points[:idx + 1], epsilon)
        right = _rdp_simplify(points[idx:], epsilon)
        return left[:-1] + right
    else:
        return [points[0], points[-1]]


def _ensure_clockwise(points):
    """Ensure polygon vertices are in clockwise order."""
    area = 0.0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    if area > 0:
        points = list(reversed(points))
    return points


def contour_from_mask(mask, simplify_epsilon=2.0, min_points=4, max_points=20):
    """Extract a simplified clockwise contour polygon from a binary mask.

    Args:
        mask: PIL Image mode 'L' (255=content, 0=bg) or numpy uint8 array.
        simplify_epsilon: RDP simplification tolerance in pixels.
        min_points: minimum contour points (pad by interpolation if fewer).
        max_points: maximum contour points (increase epsilon to reduce).

    Returns:
        List of (x, y) tuples in clockwise order, or empty list if mask is blank.
    """
    if isinstance(mask, Image.Image):
        arr = np.array(mask)
    else:
        arr = mask
    ys, xs = np.where(arr > 0)
    if len(xs) < 3:
        return []
    pts = list(zip(xs.tolist(), ys.tolist()))
    hull = _convex_hull(pts)
    if len(hull) < 3:
        return []
    # Close the polygon for RDP, then remove duplicate endpoint
    closed = hull + [hull[0]]
    simplified = _rdp_simplify(closed, simplify_epsilon)
    if simplified[-1] == simplified[0] and len(simplified) > 1:
        simplified = simplified[:-1]
    # If too many points, increase epsilon iteratively
    eps = simplify_epsilon
    while len(simplified) > max_points and eps < 100:
        eps *= 1.5
        simplified = _rdp_simplify(closed, eps)
        if simplified[-1] == simplified[0] and len(simplified) > 1:
            simplified = simplified[:-1]
    # If too few, interpolate midpoints on longest edges
    while len(simplified) < min_points and len(simplified) >= 2:
        longest_idx = 0
        longest_dist = 0
        for i in range(len(simplified)):
            j = (i + 1) % len(simplified)
            d = math.hypot(simplified[j][0] - simplified[i][0],
                           simplified[j][1] - simplified[i][1])
            if d > longest_dist:
                longest_dist = d
                longest_idx = i
        i = longest_idx
        j = (i + 1) % len(simplified)
        mid = ((simplified[i][0] + simplified[j][0]) // 2,
               (simplified[i][1] + simplified[j][1]) // 2)
        simplified.insert(j, mid)
    simplified = _ensure_clockwise(simplified)
    return [(int(x), int(y)) for x, y in simplified]


def contour_orientations(contour):
    """Compute unit tangent orientation vectors at each contour point.

    Uses central differences (forward/backward at endpoints).
    Returns list of (dx, dy) unit vectors, same length as contour.
    """
    n = len(contour)
    if n == 0:
        return []
    if n == 1:
        return [(1.0, 0.0)]
    orientations = []
    for i in range(n):
        if i == 0:
            dx = contour[1][0] - contour[0][0]
            dy = contour[1][1] - contour[0][1]
        elif i == n - 1:
            dx = contour[n - 1][0] - contour[n - 2][0]
            dy = contour[n - 1][1] - contour[n - 2][1]
        else:
            dx = contour[i + 1][0] - contour[i - 1][0]
            dy = contour[i + 1][1] - contour[i - 1][1]
        mag = math.hypot(dx, dy)
        if mag < 1e-12:
            orientations.append((1.0, 0.0))
        else:
            orientations.append((dx / mag, dy / mag))
    return orientations


def contour_to_retina(contour, scale, ox, oy):
    """Map contour points from source-image coords to retina coords."""
    return [(int(x * scale + ox), int(y * scale + oy)) for x, y in contour]


PREV_CONTOUR_NONE = (0, 0, 0.0, 0.0, CLASS_NONE)  # (x, y, dx, dy, class_id)


# ---------------------------------------------------------------------------
# 6. SyntheticPage
# ---------------------------------------------------------------------------
class SyntheticPage:
    """Generate a synthetic page with hierarchical bounding boxes and masks.

    Attributes:
        image:          PIL Image (RGB)
        bg_color:       (r, g, b) background color
        page_bbox:      (x1,y1,x2,y2) tight bbox around all content
        page_mask:      PIL Image 'L' — pixel mask for the page content region
        page_contour:   list of (x,y) — contour polygon of the page region
        paragraphs:     list of paragraph dicts, each with:
            'bbox':  (x1,y1,x2,y2)
            'mask':  PIL Image 'L' — pixel mask for the paragraph
            'is_handwritten': bool — True if rendered with a handwriting font
            'lines': list of line dicts, each with:
                'bbox':  (x1,y1,x2,y2)
                'mask':  PIL Image 'L' — pixel mask for the line
                'words': list of word dicts, each with:
                    'bbox':  (x1,y1,x2,y2)
                    'mask':  PIL Image 'L' — pixel mask for the word
                    'characters': list of char dicts with:
                        'bbox':  (x1,y1,x2,y2)
                        'mask':  PIL Image 'L' — pixel mask for the character
                        'char':  str
    """

    def __init__(self, fonts, page_width=2048, page_height=2800,
                 rotate_paragraphs=False):
        self.width = page_width
        self.height = page_height
        self.bg_color = _random_color()
        self.image = Image.new("RGB", (page_width, page_height), self.bg_color)
        self.paragraphs = []
        self.page_bbox = (0, 0, 0, 0)
        self.page_mask = None
        self.page_contour = []
        self.rotate_paragraphs = rotate_paragraphs
        self._generate(fonts)

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _random_word(min_len=1, max_len=10):
        length = random.randint(min_len, max_len)
        # Use ASCII chars without space (fonts validated for ASCII only)
        ascii_no_space = [ch for ch in ASCII_CHARS if ch != ' ']
        chars = [random.choice(ascii_no_space) for _ in range(length)]
        return "".join(chars)

    @staticmethod
    def _random_line(min_words=2, max_words=10):
        n = random.randint(min_words, max_words)
        return " ".join(SyntheticPage._random_word() for _ in range(n))

    def _rotate_paragraph_region(self, para_bbox, angle, lines_data):
        """Rotate a paragraph region in-place on the page image."""
        px1, py1, px2, py2 = para_bbox
        pw, ph = px2 - px1, py2 - py1
        if pw <= 0 or ph <= 0:
            return
        region = self.image.crop((px1, py1, px2, py2))
        # Erase original region
        bg_patch = Image.new("RGB", (pw, ph), self.bg_color)
        self.image.paste(bg_patch, (px1, py1))
        # Rotate around center with expand=True
        rotated = region.rotate(-angle, resample=Image.BICUBIC,
                                expand=True, fillcolor=self.bg_color)
        rw, rh = rotated.size
        # Compute paste position (centered on original center)
        cx, cy = px1 + pw // 2, py1 + ph // 2
        paste_x = cx - rw // 2
        paste_y = cy - rh // 2
        # Clamp to page bounds
        paste_x = max(0, min(paste_x, self.width - rw))
        paste_y = max(0, min(paste_y, self.height - rh))
        self.image.paste(rotated, (paste_x, paste_y))
        # Transform all element bboxes
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        def _rotate_point(x, y):
            rx = x - pw / 2
            ry = y - ph / 2
            nx = rx * cos_a - ry * sin_a + rw / 2 + paste_x
            ny = rx * sin_a + ry * cos_a + rh / 2 + paste_y
            return int(nx), int(ny)
        def _rotate_bbox(bbox):
            bx1, by1, bx2, by2 = bbox
            local = [(bx1 - px1, by1 - py1), (bx2 - px1, by1 - py1),
                     (bx2 - px1, by2 - py1), (bx1 - px1, by2 - py1)]
            rotated_pts = [_rotate_point(lx, ly) for lx, ly in local]
            xs = [p[0] for p in rotated_pts]
            ys = [p[1] for p in rotated_pts]
            return (min(xs), min(ys), max(xs), max(ys))
        for ld in lines_data:
            ld["bbox"] = _rotate_bbox(ld["bbox"])
            ld["mask"] = _compute_mask(self.image, ld["bbox"], self.bg_color)
            for wd in ld.get("words", []):
                wd["bbox"] = _rotate_bbox(wd["bbox"])
                wd["mask"] = _compute_mask(self.image, wd["bbox"], self.bg_color)
                for cd in wd.get("characters", []):
                    cd["bbox"] = _rotate_bbox(cd["bbox"])
                    cd["mask"] = _compute_mask(self.image, cd["bbox"], self.bg_color)

    # -- main generation --------------------------------------------------
    def _generate(self, fonts):
        draw = ImageDraw.Draw(self.image)
        num_paragraphs = random.randint(1, 6)
        margin_left = random.randint(40, 120)
        margin_right = random.randint(40, 120)
        y_cursor = random.randint(40, 120)
        max_text_width = self.width - margin_left - margin_right

        for _ in range(num_paragraphs):
            if y_cursor >= self.height - 100:
                break

            font_path = random.choice(fonts)
            font_size = random.randint(18, 52)
            try:
                font = ImageFont.truetype(font_path, font_size)
            except Exception:
                continue

            text_color = _contrasting_color(self.bg_color)
            num_lines = random.randint(2, 8)
            line_spacing = random.randint(2, 8)
            para_x1 = margin_left
            para_y1 = y_cursor
            lines_data = []

            for _ln in range(num_lines):
                if y_cursor >= self.height - 60:
                    break

                line_text = self._random_line()
                # Truncate to fit width
                while len(line_text) > 1:
                    try:
                        lw = font.getlength(line_text)
                    except AttributeError:
                        lw = font.getsize(line_text)[0]
                    if lw <= max_text_width:
                        break
                    line_text = line_text.rsplit(" ", 1)[0]

                x_pos = margin_left
                # Get line bbox
                try:
                    lb = draw.textbbox((x_pos, y_cursor), line_text, font=font)
                except AttributeError:
                    tw, th = font.getsize(line_text)
                    lb = (x_pos, y_cursor, x_pos + tw, y_cursor + th)

                # Record character bboxes (positions computed before drawing)
                # Keep spaces with None bbox so word splitting can find them
                char_bboxes = []
                cx = x_pos
                for ch in line_text:
                    try:
                        cb = draw.textbbox((cx, y_cursor), ch, font=font)
                    except AttributeError:
                        cw, ch_h = font.getsize(ch)
                        cb = (cx, y_cursor, cx + cw, y_cursor + ch_h)

                    try:
                        adv = font.getlength(ch)
                    except AttributeError:
                        adv = font.getsize(ch)[0]

                    cb_int = (int(cb[0]), int(cb[1]), int(cb[2]), int(cb[3]))
                    if ch == ' ':
                        # Keep space as word separator (bbox may be zero-width)
                        char_bboxes.append((None, ch))
                    elif cb_int[2] > cb_int[0] and cb_int[3] > cb_int[1]:
                        char_bboxes.append((cb_int, ch))
                    cx += adv

                # Draw the full line at once for consistent rendering
                draw.text((x_pos, y_cursor), line_text, fill=text_color, font=font)

                # Now compute masks from the rendered pixels
                line_bbox = (int(lb[0]), int(lb[1]), int(cx), int(lb[3]))
                if line_bbox[2] <= line_bbox[0] or line_bbox[3] <= line_bbox[1]:
                    y_cursor = lb[3] + line_spacing
                    continue

                # Group characters into words (space = word separator)
                words_data = []
                current_word_chars = []
                for cb_int, ch in char_bboxes:
                    if ch == ' ':
                        # Flush current word (space has None bbox)
                        if current_word_chars:
                            words_data.append(current_word_chars)
                            current_word_chars = []
                    elif cb_int is not None:
                        char_mask = _compute_mask(self.image, cb_int, self.bg_color)
                        current_word_chars.append({
                            "bbox": cb_int,
                            "mask": char_mask,
                            "char": ch,
                        })
                if current_word_chars:
                    words_data.append(current_word_chars)

                if not words_data:
                    y_cursor = lb[3] + line_spacing
                    continue

                # Build word-level entries with union bboxes
                word_entries = []
                for word_chars in words_data:
                    wx1 = min(c["bbox"][0] for c in word_chars)
                    wy1 = min(c["bbox"][1] for c in word_chars)
                    wx2 = max(c["bbox"][2] for c in word_chars)
                    wy2 = max(c["bbox"][3] for c in word_chars)
                    word_bbox = (wx1, wy1, wx2, wy2)
                    word_mask = _compute_mask(self.image, word_bbox, self.bg_color)
                    word_entries.append({
                        "bbox": word_bbox,
                        "mask": word_mask,
                        "characters": word_chars,
                    })

                line_mask = _compute_mask(self.image, line_bbox, self.bg_color)
                lines_data.append({
                    "bbox": line_bbox,
                    "mask": line_mask,
                    "words": word_entries,
                })

                y_cursor = lb[3] + line_spacing

            if not lines_data:
                continue

            para_x2 = int(max(ld["bbox"][2] for ld in lines_data))
            para_y2 = int(max(ld["bbox"][3] for ld in lines_data))
            para_bbox = (int(para_x1), int(para_y1), para_x2, para_y2)

            if self.rotate_paragraphs:
                angle = random.uniform(-30, 30)
                self._rotate_paragraph_region(para_bbox, angle, lines_data)
                # Recompute para bbox after rotation
                all_coords = []
                for ld in lines_data:
                    b = ld["bbox"]
                    all_coords.extend([(b[0], b[1]), (b[2], b[3])])
                if all_coords:
                    para_x1_new = min(c[0] for c in all_coords)
                    para_y1_new = min(c[1] for c in all_coords)
                    para_x2 = max(c[0] for c in all_coords)
                    para_y2 = max(c[1] for c in all_coords)
                    para_bbox = (para_x1_new, para_y1_new, para_x2, para_y2)

            para_mask = _compute_mask(self.image, para_bbox, self.bg_color)

            # Compute contours for all elements
            para_contour = contour_from_mask(para_mask)
            for ld in lines_data:
                ld["contour"] = contour_from_mask(ld["mask"])
                for wd in ld.get("words", []):
                    wd["contour"] = contour_from_mask(wd["mask"])
                    for cd in wd.get("characters", []):
                        cd["contour"] = contour_from_mask(cd["mask"])

            self.paragraphs.append({
                "bbox": para_bbox,
                "mask": para_mask,
                "contour": para_contour,
                "is_handwritten": _is_handwriting_font(font_path),
                "lines": lines_data,
            })

            y_cursor += random.randint(20, 60)

        # Compute page-level bounding contour from all paragraph regions
        if self.paragraphs:
            all_x1 = min(p["bbox"][0] for p in self.paragraphs)
            all_y1 = min(p["bbox"][1] for p in self.paragraphs)
            all_x2 = max(p["bbox"][2] for p in self.paragraphs)
            all_y2 = max(p["bbox"][3] for p in self.paragraphs)
            self.page_bbox = (all_x1, all_y1, all_x2, all_y2)
            self.page_mask = _compute_mask(self.image, self.page_bbox, self.bg_color)
            self.page_contour = contour_from_mask(self.page_mask)


# ---------------------------------------------------------------------------
# 6. RetinaOCRDataset (teacher-forced)
# ---------------------------------------------------------------------------
class RetinaOCRDataset(Dataset):
    """PyTorch Dataset with teacher forcing (no erasure).

    Each sample yields (retina_image, prev_bbox, target):
      - retina_image: 3x1024x1024 RGB tensor
      - prev_bbox:    (6,) tensor — previous ground truth
                      (x1,y1,x2,y2,class_id,is_handwritten) in retina coords,
                      or (0,0,0,0,0,0) for the first item
      - target:       (6,) tensor — bbox of next item + class token + is_handwritten

    4-level hierarchy: paragraphs → lines → words → characters.
    """

    def __init__(self, pages):
        self.pages = pages
        self.index = []
        self._build_index()

    def _build_index(self):
        for pi, page in enumerate(self.pages):
            # Page level: detect the page region (1 item + 1 NONE terminator)
            for i in range(2):
                self.index.append((pi, "page", (), i))

            n_para = len(page.paragraphs)
            for i in range(n_para + 1):
                self.index.append((pi, "para", (), i))

            for pai, para in enumerate(page.paragraphs):
                n_lines = len(para["lines"])
                for i in range(n_lines + 1):
                    self.index.append((pi, "line", (pai,), i))

                for li, line in enumerate(para["lines"]):
                    n_words = len(line["words"])
                    for i in range(n_words + 1):
                        self.index.append((pi, "word", (pai, li), i))

                    for wi, word in enumerate(line["words"]):
                        n_chars = len(word["characters"])
                        for i in range(n_chars + 1):
                            self.index.append((pi, "char", (pai, li, wi), i))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        page_idx, level, parent_ids, item_idx = self.index[idx]
        page = self.pages[page_idx]

        if level == "page":
            img, prev, target = self._make_page_sample(page, item_idx)
        elif level == "para":
            img, prev, target = self._make_para_sample(page, item_idx)
        elif level == "line":
            img, prev, target = self._make_line_sample(page, parent_ids[0], item_idx)
        elif level == "word":
            img, prev, target = self._make_word_sample(
                page, parent_ids[0], parent_ids[1], item_idx)
        else:
            img, prev, target = self._make_char_sample(
                page, parent_ids[0], parent_ids[1], parent_ids[2], item_idx)

        # HWC -> CHW
        img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        prev_t = torch.tensor(prev, dtype=torch.float32)
        target_t = torch.tensor(target, dtype=torch.float32)
        return img_t, prev_t, target_t

    # -- page level --------------------------------------------------------
    def _make_page_sample(self, page, item_idx):
        retina, scale, ox, oy = scale_and_pad(page.image, page.bg_color)

        if item_idx == 0:
            prev = PREV_BBOX_NONE
            if page.page_bbox[2] > page.page_bbox[0]:
                rb = bbox_to_retina(page.page_bbox, scale, ox, oy)
                target = (rb[0], rb[1], rb[2], rb[3], CLASS_PAGE, 0)
            else:
                target = (0, 0, 0, 0, CLASS_NONE, 0)
        else:
            if page.page_bbox[2] > page.page_bbox[0]:
                rb = bbox_to_retina(page.page_bbox, scale, ox, oy)
                prev = (rb[0], rb[1], rb[2], rb[3], CLASS_PAGE, 0)
            else:
                prev = PREV_BBOX_NONE
            target = (0, 0, 0, 0, CLASS_NONE, 0)

        return retina, prev, target

    # -- paragraph level ---------------------------------------------------
    def _make_para_sample(self, page, item_idx):
        paragraphs = page.paragraphs
        retina, scale, ox, oy = scale_and_pad(page.image, page.bg_color)

        if item_idx > 0 and item_idx - 1 < len(paragraphs):
            pb = bbox_to_retina(paragraphs[item_idx - 1]["bbox"], scale, ox, oy)
            hw = int(paragraphs[item_idx - 1].get("is_handwritten", False))
            prev = (pb[0], pb[1], pb[2], pb[3], CLASS_PARAGRAPH, hw)
        else:
            prev = PREV_BBOX_NONE

        if item_idx < len(paragraphs):
            rb = bbox_to_retina(paragraphs[item_idx]["bbox"], scale, ox, oy)
            hw = int(paragraphs[item_idx].get("is_handwritten", False))
            target = (rb[0], rb[1], rb[2], rb[3], CLASS_PARAGRAPH, hw)
        else:
            target = (0, 0, 0, 0, CLASS_NONE, 0)

        return retina, prev, target

    # -- line level --------------------------------------------------------
    def _make_line_sample(self, page, para_idx, item_idx):
        para = page.paragraphs[para_idx]
        lines = para["lines"]
        hw = int(para.get("is_handwritten", False))

        px1, py1, px2, py2 = para["bbox"]
        img = page.image.crop((px1, py1, px2, py2))
        retina, scale, ox, oy = scale_and_pad(img, page.bg_color)

        if item_idx > 0 and item_idx - 1 < len(lines):
            lx1, ly1, lx2, ly2 = lines[item_idx - 1]["bbox"]
            local = (lx1 - px1, ly1 - py1, lx2 - px1, ly2 - py1)
            pb = bbox_to_retina(local, scale, ox, oy)
            prev = (pb[0], pb[1], pb[2], pb[3], CLASS_LINE, hw)
        else:
            prev = PREV_BBOX_NONE

        if item_idx < len(lines):
            lx1, ly1, lx2, ly2 = lines[item_idx]["bbox"]
            local = (lx1 - px1, ly1 - py1, lx2 - px1, ly2 - py1)
            rb = bbox_to_retina(local, scale, ox, oy)
            target = (rb[0], rb[1], rb[2], rb[3], CLASS_LINE, hw)
        else:
            target = (0, 0, 0, 0, CLASS_NONE, 0)

        return retina, prev, target

    # -- word level --------------------------------------------------------
    def _make_word_sample(self, page, para_idx, line_idx, item_idx):
        para = page.paragraphs[para_idx]
        line = para["lines"][line_idx]
        words = line["words"]
        hw = int(para.get("is_handwritten", False))

        lx1, ly1, lx2, ly2 = line["bbox"]
        img = page.image.crop((lx1, ly1, lx2, ly2))
        retina, scale, ox, oy = scale_and_pad(img, page.bg_color)

        if item_idx > 0 and item_idx - 1 < len(words):
            wx1, wy1, wx2, wy2 = words[item_idx - 1]["bbox"]
            local = (wx1 - lx1, wy1 - ly1, wx2 - lx1, wy2 - ly1)
            pb = bbox_to_retina(local, scale, ox, oy)
            prev = (pb[0], pb[1], pb[2], pb[3], CLASS_WORD, hw)
        else:
            prev = PREV_BBOX_NONE

        if item_idx < len(words):
            wx1, wy1, wx2, wy2 = words[item_idx]["bbox"]
            local = (wx1 - lx1, wy1 - ly1, wx2 - lx1, wy2 - ly1)
            rb = bbox_to_retina(local, scale, ox, oy)
            target = (rb[0], rb[1], rb[2], rb[3], CLASS_WORD, hw)
        else:
            target = (0, 0, 0, 0, CLASS_NONE, 0)

        return retina, prev, target

    # -- character level ---------------------------------------------------
    def _make_char_sample(self, page, para_idx, line_idx, word_idx, item_idx):
        para = page.paragraphs[para_idx]
        word = para["lines"][line_idx]["words"][word_idx]
        chars = word["characters"]
        hw = int(para.get("is_handwritten", False))

        wx1, wy1, wx2, wy2 = word["bbox"]
        img = page.image.crop((wx1, wy1, wx2, wy2))
        retina, scale, ox, oy = scale_and_pad(img, page.bg_color)

        if item_idx > 0 and item_idx - 1 < len(chars):
            cx1, cy1, cx2, cy2 = chars[item_idx - 1]["bbox"]
            local = (cx1 - wx1, cy1 - wy1, cx2 - wx1, cy2 - wy1)
            pb = bbox_to_retina(local, scale, ox, oy)
            prev_class = char_to_class(chars[item_idx - 1]["char"])
            prev = (pb[0], pb[1], pb[2], pb[3], prev_class, hw)
        else:
            prev = PREV_BBOX_NONE

        if item_idx < len(chars):
            cx1, cy1, cx2, cy2 = chars[item_idx]["bbox"]
            local = (cx1 - wx1, cy1 - wy1, cx2 - wx1, cy2 - wy1)
            rb = bbox_to_retina(local, scale, ox, oy)
            class_id = char_to_class(chars[item_idx]["char"])
            target = (rb[0], rb[1], rb[2], rb[3], class_id, hw)
        else:
            target = (0, 0, 0, 0, CLASS_NONE, 0)

        return retina, prev, target


# ---------------------------------------------------------------------------
# 7. CharacterPretrainDataset
# ---------------------------------------------------------------------------
class CharacterPretrainDataset(Dataset):
    """Dataset that renders single characters on random backgrounds.

    Each sample places one character on a small image, scales to retina, and
    produces two training samples:
      1. prev=NONE → target=(char_bbox, char_class)  — "find the character"
      2. prev=(char_bbox, char_class) with char erased → target=NONE  — "nothing left"

    This teaches the model character recognition before full hierarchical training.
    """

    def __init__(self, fonts, num_samples=10000):
        self.fonts = fonts
        self.samples = []
        self._generate(num_samples)

    def _generate(self, num_samples):
        for _ in range(num_samples):
            bg_color = _random_color()
            text_color = _contrasting_color(bg_color)
            ch = random.choice(ASCII_CHARS)
            if ch == ' ':
                ch = random.choice(ASCII_CHARS[1:])  # skip space

            font_path = random.choice(self.fonts)
            font_size = random.randint(18, 52)
            try:
                font = ImageFont.truetype(font_path, font_size)
            except Exception:
                continue

            # Render character to get its size
            temp = Image.new("RGB", (1, 1))
            temp_draw = ImageDraw.Draw(temp)
            bbox = temp_draw.textbbox((0, 0), ch, font=font)
            ch_w = bbox[2] - bbox[0] + 4
            ch_h = bbox[3] - bbox[1] + 4

            if ch_w <= 0 or ch_h <= 0:
                continue

            # Create a small image with some padding around the character
            pad_x = random.randint(ch_w, ch_w * 4)
            pad_y = random.randint(ch_h, ch_h * 4)
            img_w = ch_w + pad_x * 2
            img_h = ch_h + pad_y * 2

            img = Image.new("RGB", (img_w, img_h), bg_color)
            draw = ImageDraw.Draw(img)

            # Place character at a random position within the padded area
            cx = random.randint(pad_x // 2, pad_x + pad_x // 2)
            cy = random.randint(pad_y // 2, pad_y + pad_y // 2)
            draw.text((cx - bbox[0], cy - bbox[1]), ch, fill=text_color, font=font)

            # Compute actual character bbox
            char_bbox = (cx - bbox[0], cy - bbox[1],
                         cx - bbox[0] + ch_w, cy - bbox[1] + ch_h)

            # Compute mask for erasure
            mask = _compute_mask(img, char_bbox, bg_color)
            class_id = char_to_class(ch)

            self.samples.append((img, bg_color, char_bbox, mask, class_id))

    def __len__(self):
        return len(self.samples) * 2  # detect + NONE pair

    def __getitem__(self, idx):
        sample_idx = idx // 2
        is_none = idx % 2 == 1  # odd indices are the NONE-after-detect samples

        img_orig, bg_color, char_bbox, mask, class_id = self.samples[sample_idx]

        if is_none:
            # Character still present, prev=char, target=NONE
            # Model sees the same image but prev_bbox says "I just found it"
            retina, scale, ox, oy = scale_and_pad(img_orig, bg_color)
            rb = bbox_to_retina(char_bbox, scale, ox, oy)
            prev = (rb[0], rb[1], rb[2], rb[3], class_id, 0)
            target = (0, 0, 0, 0, CLASS_NONE, 0)
        else:
            # Character present, prev=NONE, target=char
            retina, scale, ox, oy = scale_and_pad(img_orig, bg_color)
            prev = PREV_BBOX_NONE
            rb = bbox_to_retina(char_bbox, scale, ox, oy)
            target = (rb[0], rb[1], rb[2], rb[3], class_id, 0)

        img_t = torch.from_numpy(np.array(retina)).permute(2, 0, 1).float() / 255.0
        prev_t = torch.tensor(prev, dtype=torch.float32)
        target_t = torch.tensor(target, dtype=torch.float32)
        return img_t, prev_t, target_t


# ---------------------------------------------------------------------------
# 8. visualize_sample()
# ---------------------------------------------------------------------------
def visualize_sample(img_tensor, prev_tensor, target_tensor, path):
    """Draw prev bbox (dashed) + target bbox (solid) on retina image, save as PNG."""
    # CHW -> HWC
    img = Image.fromarray((img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    # Draw previous bbox (teacher forcing input) in yellow dashed
    prev_vals = [int(v) for v in prev_tensor.tolist()]
    px1, py1, px2, py2, pcls = prev_vals[0], prev_vals[1], prev_vals[2], prev_vals[3], prev_vals[4]
    if pcls != CLASS_NONE:
        for x in range(px1, px2, 6):
            draw.line([(x, py1), (min(x + 3, px2), py1)], fill=(200, 200, 0), width=1)
            draw.line([(x, py2), (min(x + 3, px2), py2)], fill=(200, 200, 0), width=1)
        for y in range(py1, py2, 6):
            draw.line([(px1, y), (px1, min(y + 3, py2))], fill=(200, 200, 0), width=1)
            draw.line([(px2, y), (px2, min(y + 3, py2))], fill=(200, 200, 0), width=1)
        draw.text((px1, max(0, py1 - 24)), f"prev:{class_to_label(pcls)}", fill=(200, 200, 0))

    # Draw target bbox
    tgt_vals = [int(v) for v in target_tensor.tolist()]
    x1, y1, x2, y2, cls = tgt_vals[0], tgt_vals[1], tgt_vals[2], tgt_vals[3], tgt_vals[4]

    if cls == CLASS_NONE:
        color = (128, 128, 128)
    elif cls == CLASS_PAGE:
        color = (128, 0, 255)  # purple
    elif cls == CLASS_PARAGRAPH:
        color = (255, 0, 0)
    elif cls == CLASS_LINE:
        color = (0, 0, 255)
    elif cls == CLASS_WORD:
        color = (255, 165, 0)  # orange
    else:
        color = (0, 180, 0)

    label = class_to_label(cls)

    if cls != CLASS_NONE:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, max(0, y1 - 12)), label, fill=color)
    else:
        draw.text((10, 10), label, fill=color)

    img.save(path)


# ---------------------------------------------------------------------------
# 8. __main__
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic OCR training data")
    parser.add_argument("--pages", type=int, default=10, help="Number of pages to generate")
    parser.add_argument("--page-width", type=int, default=2048)
    parser.add_argument("--page-height", type=int, default=2800)
    parser.add_argument("--samples-dir", type=str, default="samples")
    parser.add_argument("--num-vis", type=int, default=20, help="Number of sample visualizations to save")
    args = parser.parse_args()

    # Discover fonts
    fonts = discover_fonts()
    print(f"Discovered {len(fonts)} fonts")
    if not fonts:
        print("No fonts found! Place .ttf/.otf/.ttc files in fonts/ directory.")
        sys.exit(1)

    # Generate pages
    print(f"Generating {args.pages} pages...")
    pages = []
    for i in range(args.pages):
        pages.append(SyntheticPage(fonts, args.page_width, args.page_height))
        total_words = sum(
            len(line["words"])
            for para in pages[-1].paragraphs
            for line in para["lines"]
        )
        total_chars = sum(
            len(word["characters"])
            for para in pages[-1].paragraphs
            for line in para["lines"]
            for word in line["words"]
        )
        print(f"  Page {i}: {len(pages[-1].paragraphs)} paragraphs, {total_words} words, {total_chars} characters, bg={pages[-1].bg_color}")

    # Build dataset
    dataset = RetinaOCRDataset(pages)
    print(f"\nDataset size: {len(dataset)} samples")

    # Class distribution
    class_counts = {}
    for entry in dataset.index:
        _, level, parent_ids, item_idx = entry
        page = pages[entry[0]]
        if level == "page":
            cls = CLASS_PAGE if item_idx < 1 else CLASS_NONE
        elif level == "para":
            cls = CLASS_PARAGRAPH if item_idx < len(page.paragraphs) else CLASS_NONE
        elif level == "line":
            para = page.paragraphs[parent_ids[0]]
            cls = CLASS_LINE if item_idx < len(para["lines"]) else CLASS_NONE
        elif level == "word":
            para = page.paragraphs[parent_ids[0]]
            line = para["lines"][parent_ids[1]]
            cls = CLASS_WORD if item_idx < len(line["words"]) else CLASS_NONE
        else:
            para = page.paragraphs[parent_ids[0]]
            word = para["lines"][parent_ids[1]]["words"][parent_ids[2]]
            if item_idx < len(word["characters"]):
                cls = char_to_class(word["characters"][item_idx]["char"])
            else:
                cls = CLASS_NONE
        label = class_to_label(cls)
        class_counts[label] = class_counts.get(label, 0) + 1

    print("\nClass distribution:")
    for label in sorted(class_counts.keys()):
        print(f"  {label}: {class_counts[label]}")

    # Save visualizations
    os.makedirs(args.samples_dir, exist_ok=True)
    num_vis = min(args.num_vis, len(dataset))

    if num_vis >= len(dataset):
        vis_indices = list(range(len(dataset)))
    else:
        vis_indices = sorted(random.sample(range(len(dataset)), num_vis))

    print(f"\nSaving {num_vis} sample visualizations to {args.samples_dir}/")
    for j, vi in enumerate(vis_indices):
        img_t, prev_t, tgt_t = dataset[vi]
        path = os.path.join(args.samples_dir, f"sample_{j:04d}.png")
        visualize_sample(img_t, prev_t, tgt_t, path)
        _, level, parent_ids, item_idx = dataset.index[vi]
        print(f"  {path}  level={level} item={item_idx} target={tgt_t.tolist()}")

    # Quick DataLoader sanity check
    print("\nDataLoader sanity check...")
    dl = DataLoader(dataset, batch_size=8, shuffle=True)
    batch_img, batch_prev, batch_tgt = next(iter(dl))
    print(f"  Image batch shape:  {batch_img.shape}")    # (8, 3, 1024, 1024)
    print(f"  Prev batch shape:   {batch_prev.shape}")   # (8, 6)
    print(f"  Target batch shape: {batch_tgt.shape}")     # (8, 6)
    print("\nDone.")
