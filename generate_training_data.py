"""
Synthetic OCR training data generator for hierarchical retina-based detection.

Generates training data for a model that operates on a fixed 1024x1024 "retina"
and detects items sequentially: paragraphs on a page, lines in a paragraph,
characters in a line.

Uses teacher forcing: the previous ground truth bbox is provided as model input.
The image stays unchanged throughout the autoregressive loop — the model relies
on prev_bbox to determine what to detect next (spatial ordering: top-to-bottom
for paragraphs/lines, left-to-right for characters).

Pages use random background colors and each paragraph gets a random text color.

Class tokens (98 total):
  0     = NONE (nothing left to detect)
  1     = PARAGRAPH
  2     = LINE
  3-97  = ASCII chars: class_id = ord(char) - 32 + 3
"""

import os
import sys
import random
import string
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont

from create_lists import create_character_list

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RETINA_SIZE = 1024
CLASS_NONE = 0
CLASS_PARAGRAPH = 1
CLASS_LINE = 2
CHAR_CLASS_OFFSET = 3  # class_id = ord(char) - 32 + CHAR_CLASS_OFFSET

PRINTABLE_CHARS = create_character_list()  # chr(32)..chr(126)
PREV_BBOX_NONE = (0, 0, 0, 0, CLASS_NONE)


def char_to_class(ch):
    return ord(ch) - 32 + CHAR_CLASS_OFFSET


def class_to_label(class_id):
    if class_id == CLASS_NONE:
        return "NONE"
    if class_id == CLASS_PARAGRAPH:
        return "PARA"
    if class_id == CLASS_LINE:
        return "LINE"
    return repr(chr(class_id - CHAR_CLASS_OFFSET + 32))


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
# 5. SyntheticPage
# ---------------------------------------------------------------------------
class SyntheticPage:
    """Generate a synthetic page with hierarchical bounding boxes and masks.

    Attributes:
        image:      PIL Image (RGB)
        bg_color:   (r, g, b) background color
        paragraphs: list of paragraph dicts, each with:
            'bbox':  (x1,y1,x2,y2)
            'mask':  PIL Image 'L' — pixel mask for the paragraph
            'lines': list of line dicts, each with:
                'bbox':  (x1,y1,x2,y2)
                'mask':  PIL Image 'L' — pixel mask for the line
                'characters': list of char dicts with:
                    'bbox':  (x1,y1,x2,y2)
                    'mask':  PIL Image 'L' — pixel mask for the character
                    'char':  str
    """

    def __init__(self, fonts, page_width=2048, page_height=2800):
        self.width = page_width
        self.height = page_height
        self.bg_color = _random_color()
        self.image = Image.new("RGB", (page_width, page_height), self.bg_color)
        self.paragraphs = []
        self._generate(fonts)

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _random_word(min_len=1, max_len=10):
        length = random.randint(min_len, max_len)
        chars = [random.choice(string.ascii_letters + string.digits + string.punctuation) for _ in range(length)]
        return "".join(chars)

    @staticmethod
    def _random_line(min_words=2, max_words=10):
        n = random.randint(min_words, max_words)
        return " ".join(SyntheticPage._random_word() for _ in range(n))

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
                    if cb_int[2] > cb_int[0] and cb_int[3] > cb_int[1]:
                        char_bboxes.append((cb_int, ch))
                    cx += adv

                # Draw the full line at once for consistent rendering
                draw.text((x_pos, y_cursor), line_text, fill=text_color, font=font)

                # Now compute masks from the rendered pixels
                line_bbox = (int(lb[0]), int(lb[1]), int(cx), int(lb[3]))
                if line_bbox[2] <= line_bbox[0] or line_bbox[3] <= line_bbox[1]:
                    y_cursor = lb[3] + line_spacing
                    continue

                char_data = []
                for cb_int, ch in char_bboxes:
                    char_mask = _compute_mask(self.image, cb_int, self.bg_color)
                    char_data.append({
                        "bbox": cb_int,
                        "mask": char_mask,
                        "char": ch,
                    })

                if not char_data:
                    y_cursor = lb[3] + line_spacing
                    continue

                line_mask = _compute_mask(self.image, line_bbox, self.bg_color)
                lines_data.append({
                    "bbox": line_bbox,
                    "mask": line_mask,
                    "characters": char_data,
                })

                y_cursor = lb[3] + line_spacing

            if not lines_data:
                continue

            para_x2 = int(max(ld["bbox"][2] for ld in lines_data))
            para_y2 = int(max(ld["bbox"][3] for ld in lines_data))
            para_bbox = (int(para_x1), int(para_y1), para_x2, para_y2)
            para_mask = _compute_mask(self.image, para_bbox, self.bg_color)

            self.paragraphs.append({
                "bbox": para_bbox,
                "mask": para_mask,
                "lines": lines_data,
            })

            y_cursor += random.randint(20, 60)


# ---------------------------------------------------------------------------
# 6. RetinaOCRDataset (teacher-forced)
# ---------------------------------------------------------------------------
class RetinaOCRDataset(Dataset):
    """PyTorch Dataset with teacher forcing (no erasure).

    Each sample yields (retina_image, prev_bbox, target):
      - retina_image: 3x1024x1024 RGB tensor
      - prev_bbox:    (5,) tensor — previous ground truth (x1,y1,x2,y2,class_id)
                      in retina coords, or (0,0,0,0,0) for the first item
      - target:       (5,) tensor — bbox of next item + class token

    The image stays unchanged throughout the autoregressive loop. The model
    relies on prev_bbox to determine what to detect next.
    """

    def __init__(self, pages):
        self.pages = pages
        self.index = []
        self._build_index()

    def _build_index(self):
        for pi, page in enumerate(self.pages):
            n_para = len(page.paragraphs)
            for i in range(n_para + 1):
                self.index.append((pi, "para", (), i))

            for pai, para in enumerate(page.paragraphs):
                n_lines = len(para["lines"])
                for i in range(n_lines + 1):
                    self.index.append((pi, "line", (pai,), i))

                for li, line in enumerate(para["lines"]):
                    n_chars = len(line["characters"])
                    for i in range(n_chars + 1):
                        self.index.append((pi, "char", (pai, li), i))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        page_idx, level, parent_ids, item_idx = self.index[idx]
        page = self.pages[page_idx]

        if level == "para":
            img, prev, target = self._make_para_sample(page, item_idx)
        elif level == "line":
            img, prev, target = self._make_line_sample(page, parent_ids[0], item_idx)
        else:
            img, prev, target = self._make_char_sample(page, parent_ids[0], parent_ids[1], item_idx)

        # HWC -> CHW
        img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        prev_t = torch.tensor(prev, dtype=torch.float32)
        target_t = torch.tensor(target, dtype=torch.float32)
        return img_t, prev_t, target_t

    # -- paragraph level ---------------------------------------------------
    def _make_para_sample(self, page, item_idx):
        paragraphs = page.paragraphs

        retina, scale, ox, oy = scale_and_pad(page.image, page.bg_color)

        if item_idx > 0 and item_idx - 1 < len(paragraphs):
            pb = bbox_to_retina(paragraphs[item_idx - 1]["bbox"], scale, ox, oy)
            prev = (pb[0], pb[1], pb[2], pb[3], CLASS_PARAGRAPH)
        else:
            prev = PREV_BBOX_NONE

        if item_idx < len(paragraphs):
            rb = bbox_to_retina(paragraphs[item_idx]["bbox"], scale, ox, oy)
            target = (rb[0], rb[1], rb[2], rb[3], CLASS_PARAGRAPH)
        else:
            target = (0, 0, 0, 0, CLASS_NONE)

        return retina, prev, target

    # -- line level --------------------------------------------------------
    def _make_line_sample(self, page, para_idx, item_idx):
        para = page.paragraphs[para_idx]
        lines = para["lines"]

        px1, py1, px2, py2 = para["bbox"]
        img = page.image.crop((px1, py1, px2, py2))

        retina, scale, ox, oy = scale_and_pad(img, page.bg_color)

        if item_idx > 0 and item_idx - 1 < len(lines):
            lx1, ly1, lx2, ly2 = lines[item_idx - 1]["bbox"]
            local = (lx1 - px1, ly1 - py1, lx2 - px1, ly2 - py1)
            pb = bbox_to_retina(local, scale, ox, oy)
            prev = (pb[0], pb[1], pb[2], pb[3], CLASS_LINE)
        else:
            prev = PREV_BBOX_NONE

        if item_idx < len(lines):
            lx1, ly1, lx2, ly2 = lines[item_idx]["bbox"]
            local = (lx1 - px1, ly1 - py1, lx2 - px1, ly2 - py1)
            rb = bbox_to_retina(local, scale, ox, oy)
            target = (rb[0], rb[1], rb[2], rb[3], CLASS_LINE)
        else:
            target = (0, 0, 0, 0, CLASS_NONE)

        return retina, prev, target

    # -- character level ---------------------------------------------------
    def _make_char_sample(self, page, para_idx, line_idx, item_idx):
        para = page.paragraphs[para_idx]
        line = para["lines"][line_idx]
        chars = line["characters"]

        lx1, ly1, lx2, ly2 = line["bbox"]
        img = page.image.crop((lx1, ly1, lx2, ly2))

        retina, scale, ox, oy = scale_and_pad(img, page.bg_color)

        if item_idx > 0 and item_idx - 1 < len(chars):
            cx1, cy1, cx2, cy2 = chars[item_idx - 1]["bbox"]
            local = (cx1 - lx1, cy1 - ly1, cx2 - lx1, cy2 - ly1)
            pb = bbox_to_retina(local, scale, ox, oy)
            prev_class = char_to_class(chars[item_idx - 1]["char"])
            prev = (pb[0], pb[1], pb[2], pb[3], prev_class)
        else:
            prev = PREV_BBOX_NONE

        if item_idx < len(chars):
            cx1, cy1, cx2, cy2 = chars[item_idx]["bbox"]
            local = (cx1 - lx1, cy1 - ly1, cx2 - lx1, cy2 - ly1)
            rb = bbox_to_retina(local, scale, ox, oy)
            class_id = char_to_class(chars[item_idx]["char"])
            target = (rb[0], rb[1], rb[2], rb[3], class_id)
        else:
            target = (0, 0, 0, 0, CLASS_NONE)

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
            ch = random.choice(PRINTABLE_CHARS)
            if ch == ' ':
                ch = random.choice(PRINTABLE_CHARS[1:])  # skip space

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
            prev = (rb[0], rb[1], rb[2], rb[3], class_id)
            target = (0, 0, 0, 0, CLASS_NONE)
        else:
            # Character present, prev=NONE, target=char
            retina, scale, ox, oy = scale_and_pad(img_orig, bg_color)
            prev = PREV_BBOX_NONE
            rb = bbox_to_retina(char_bbox, scale, ox, oy)
            target = (rb[0], rb[1], rb[2], rb[3], class_id)

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
    px1, py1, px2, py2, pcls = [int(v) for v in prev_tensor.tolist()]
    if pcls != CLASS_NONE:
        for x in range(px1, px2, 6):
            draw.line([(x, py1), (min(x + 3, px2), py1)], fill=(200, 200, 0), width=1)
            draw.line([(x, py2), (min(x + 3, px2), py2)], fill=(200, 200, 0), width=1)
        for y in range(py1, py2, 6):
            draw.line([(px1, y), (px1, min(y + 3, py2))], fill=(200, 200, 0), width=1)
            draw.line([(px2, y), (px2, min(y + 3, py2))], fill=(200, 200, 0), width=1)
        draw.text((px1, max(0, py1 - 24)), f"prev:{class_to_label(pcls)}", fill=(200, 200, 0))

    # Draw target bbox
    x1, y1, x2, y2, cls = [int(v) for v in target_tensor.tolist()]

    if cls == CLASS_NONE:
        color = (128, 128, 128)
    elif cls == CLASS_PARAGRAPH:
        color = (255, 0, 0)
    elif cls == CLASS_LINE:
        color = (0, 0, 255)
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
        total_chars = sum(
            len(line["characters"])
            for para in pages[-1].paragraphs
            for line in para["lines"]
        )
        print(f"  Page {i}: {len(pages[-1].paragraphs)} paragraphs, {total_chars} characters, bg={pages[-1].bg_color}")

    # Build dataset
    dataset = RetinaOCRDataset(pages)
    print(f"\nDataset size: {len(dataset)} samples")

    # Class distribution
    class_counts = {}
    for entry in dataset.index:
        _, level, parent_ids, item_idx = entry
        page = pages[entry[0]]
        if level == "para":
            cls = CLASS_PARAGRAPH if item_idx < len(page.paragraphs) else CLASS_NONE
        elif level == "line":
            para = page.paragraphs[parent_ids[0]]
            cls = CLASS_LINE if item_idx < len(para["lines"]) else CLASS_NONE
        else:
            para = page.paragraphs[parent_ids[0]]
            line = para["lines"][parent_ids[1]]
            if item_idx < len(line["characters"]):
                cls = char_to_class(line["characters"][item_idx]["char"])
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
    print(f"  Prev batch shape:   {batch_prev.shape}")   # (8, 5)
    print(f"  Target batch shape: {batch_tgt.shape}")     # (8, 5)
    print("\nDone.")
