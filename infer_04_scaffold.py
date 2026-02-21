"""
Dumb scaffolding for parallel stateless OCR inference.

Conventional code, no learned parameters.  Handles image preparation,
coordinate transforms, parallel dispatch, and result assembly.

Parallelism strategy:
  - Levels 0+1: sequential (single page retina, reused for both)
  - Level 2: all paragraphs' lines in parallel
  - Level 3: ALL lines' words across ALL paragraphs in parallel (flat)
  - Level 4: ALL words' chars across ALL words in parallel (flat)

Usage:
    result = ocr_document("page.png", "model_04.pth")
    print(result.text)
"""

import io
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

from cascade_transforms import forward_cascade_step, lift_contour
from generate_training_data import (
    CHAR_CLASS_OFFSET,
    CLASS_TO_CHAR,
    scale_and_pad,
)
from infer_04_stateless import (
    ContourDetection,
    TraceConfig,
    load_model,
    trace_contours,
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CharResult:
    contour: list       # page-space (x, y) tuples
    class_id: int
    char: str


@dataclass
class WordResult:
    contour: list
    chars: list = field(default_factory=list)   # list[CharResult]
    text: str = ""


@dataclass
class LineResult:
    contour: list
    words: list = field(default_factory=list)   # list[WordResult]
    text: str = ""


@dataclass
class ParagraphResult:
    contour: list
    lines: list = field(default_factory=list)   # list[LineResult]
    text: str = ""


@dataclass
class DocumentResult:
    page_contours: list = field(default_factory=list)
    paragraphs: list = field(default_factory=list)  # list[ParagraphResult]
    text: str = ""


# ---------------------------------------------------------------------------
# Image preparation
# ---------------------------------------------------------------------------

def _img_to_tensor(pil_img, device):
    """Convert PIL RGB image to [1, 3, H, W] float tensor on device."""
    return (
        torch.from_numpy(np.array(pil_img))
        .permute(2, 0, 1).float().unsqueeze(0) / 255.0
    ).to(device)


def prepare_page_retina(page_img, bg_color, device):
    """scale_and_pad for levels 0-1.  Returns (img_t, scale, ox, oy)."""
    retina, scale, ox, oy = scale_and_pad(page_img, bg_color)
    img_t = _img_to_tensor(retina, device)
    return img_t, scale, ox, oy


def prepare_cascade_retina(page_img, parent_contour, bg_color, device):
    """forward_cascade_step for levels 2-4.  Returns (img_t, xf)."""
    retina_pil, xf = forward_cascade_step(page_img, parent_contour, bg_color)
    img_t = _img_to_tensor(retina_pil, device)
    return img_t, xf


# ---------------------------------------------------------------------------
# Coordinate lifting
# ---------------------------------------------------------------------------

def _retina_to_source(rx, ry, scale, ox, oy):
    return (int((rx - ox) / scale), int((ry - oy) / scale))


def lift_page_detections(detections, scale, ox, oy):
    """Retina -> source for levels 0-1."""
    result = []
    for det in detections:
        src_contour = [
            _retina_to_source(px, py, scale, ox, oy)
            for px, py in det.retina_points
        ]
        result.append((src_contour, det.class_id))
    return result


def lift_cascade_detections(detections, xf):
    """Retina -> page via lift_contour for levels 2-4."""
    result = []
    for det in detections:
        page_contour = lift_contour(det.retina_points, xf)
        result.append((page_contour, det.class_id))
    return result


# ---------------------------------------------------------------------------
# Parallel sibling dispatch
# ---------------------------------------------------------------------------

def _trace_one_sibling(model, page_img, parent_contour, bg_color, level_id,
                       device, config):
    """Trace contours for a single parent.  Returns [(page_contour, class_id)]."""
    img_t, xf = prepare_cascade_retina(page_img, parent_contour, bg_color,
                                        device)
    dets = trace_contours(model, img_t, level_id, device, config)
    return lift_cascade_detections(dets, xf)


def process_siblings_parallel(model, page_img, parent_contours, bg_color,
                              level_id, device, config, max_workers=4):
    """Dispatch sibling elements concurrently via ThreadPoolExecutor.

    Returns list-of-lists: results[i] = detections for parent_contours[i].
    """
    if not parent_contours:
        return []

    if max_workers <= 1 or len(parent_contours) == 1:
        return [
            _trace_one_sibling(model, page_img, pc, bg_color, level_id,
                               device, config)
            for pc in parent_contours
        ]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_trace_one_sibling, model, page_img, pc, bg_color,
                        level_id, device, config)
            for pc in parent_contours
        ]
        return [f.result() for f in futures]


# ---------------------------------------------------------------------------
# Top-level stateless API
# ---------------------------------------------------------------------------

def _class_to_char(class_id):
    return CLASS_TO_CHAR.get(class_id, "?")


def ocr_document(image_bytes_or_path, weights_path, device=None,
                 config=None, max_workers=4, model=None):
    """Stateless document OCR.  Returns DocumentResult.

    Args:
        image_bytes_or_path: file path (str) or bytes of a page image.
        weights_path: path to model_04.pth.
        device: torch device (auto-detected if None).
        config: TraceConfig (defaults if None).
        max_workers: thread pool size for parallel sibling dispatch.
        model: pre-loaded ContourOCRNet (loaded from weights_path if None).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config is None:
        config = TraceConfig()
    if model is None:
        model = load_model(weights_path, device)

    # Load image
    if isinstance(image_bytes_or_path, str):
        page_img = Image.open(image_bytes_or_path).convert("RGB")
    else:
        page_img = Image.open(io.BytesIO(image_bytes_or_path)).convert("RGB")

    from annotate_real import estimate_background_color
    bg_color = estimate_background_color(page_img)

    # --- Level 0: page regions (scale_and_pad) ---
    page_t, scale, ox, oy = prepare_page_retina(page_img, bg_color, device)
    page_dets = trace_contours(model, page_t, 0, device, config)
    page_contours = [
        [_retina_to_source(px, py, scale, ox, oy)
         for px, py in d.retina_points]
        for d in page_dets
    ]

    # --- Level 1: paragraphs (scale_and_pad, reuse page_t) ---
    para_dets = trace_contours(model, page_t, 1, device, config)
    para_lifted = lift_page_detections(para_dets, scale, ox, oy)
    para_contours = [contour for contour, _ in para_lifted]

    # --- Level 2: lines within each paragraph (parallel) ---
    lines_per_para = process_siblings_parallel(
        model, page_img, para_contours, bg_color, 2, device, config,
        max_workers,
    )

    # --- Level 3: words -- flat dispatch across ALL lines ---
    all_line_contours = []
    line_to_para = []
    for pi, para_lines in enumerate(lines_per_para):
        for line_contour, _ in para_lines:
            all_line_contours.append(line_contour)
            line_to_para.append(pi)

    words_per_line = process_siblings_parallel(
        model, page_img, all_line_contours, bg_color, 3, device, config,
        max_workers,
    )

    # --- Level 4: chars -- flat dispatch across ALL words ---
    all_word_contours = []
    word_to_line = []
    for flat_li, line_words in enumerate(words_per_line):
        for word_contour, _ in line_words:
            all_word_contours.append(word_contour)
            word_to_line.append(flat_li)

    chars_per_word = process_siblings_parallel(
        model, page_img, all_word_contours, bg_color, 4, device, config,
        max_workers,
    )

    # --- Assemble results ---
    word_results_flat = []
    for flat_wi, word_chars in enumerate(chars_per_word):
        char_results = []
        for char_contour, class_id in word_chars:
            ch = (_class_to_char(class_id)
                  if class_id >= CHAR_CLASS_OFFSET else "?")
            char_results.append(CharResult(
                contour=char_contour, class_id=class_id, char=ch,
            ))
        text = "".join(cr.char for cr in char_results)
        word_results_flat.append(WordResult(
            contour=all_word_contours[flat_wi],
            chars=char_results,
            text=text,
        ))

    # Group words into lines
    line_results_flat = []
    for flat_li in range(len(all_line_contours)):
        line_words = [
            word_results_flat[flat_wi]
            for flat_wi, wl in enumerate(word_to_line)
            if wl == flat_li
        ]
        text = " ".join(wr.text for wr in line_words)
        line_results_flat.append(LineResult(
            contour=all_line_contours[flat_li],
            words=line_words,
            text=text,
        ))

    # Group lines into paragraphs
    paragraphs = []
    for pi, (para_contour, _) in enumerate(para_lifted):
        para_lines = [
            line_results_flat[flat_li]
            for flat_li, lp in enumerate(line_to_para)
            if lp == pi
        ]
        text = "\n".join(lr.text for lr in para_lines)
        paragraphs.append(ParagraphResult(
            contour=para_contour,
            lines=para_lines,
            text=text,
        ))

    full_text = "\n\n".join(pr.text for pr in paragraphs)

    return DocumentResult(
        page_contours=page_contours,
        paragraphs=paragraphs,
        text=full_text,
    )
