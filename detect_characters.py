"""
Multi-scale sliding window character detector using HierarchicalCharNet.

Strides a fixed-size window across a page image at multiple scales, classifies
each window via the two-stage hierarchical model, maps detections back to page
coordinates, and applies NMS to produce final character bounding boxes.

Usage:
  from detect_characters import detect_characters, load_detector
  detector = load_detector("model_02_char.pth")
  detections = detect_characters(page_image, detector)
  # detections: list of {"bbox": [x1,y1,x2,y2], "char": str, "confidence": float}

CLI:
  .venv/bin/python detect_characters.py image.png --model model_02_char.pth --visualize
"""

import argparse
import hashlib
import os
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.ops
from PIL import Image, ImageDraw, ImageFont


# Mega-block threshold: blocks with more chars than this use embedding retrieval
MEGA_BLOCK_THRESHOLD = 256

import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hangul jamo decomposition
# ---------------------------------------------------------------------------

_HANGUL_BASE = 0xAC00
_HANGUL_END = 0xD7A3
_N_LEAD = 19
_N_VOWEL = 21
_N_TAIL = 28  # 0 = no tail


def decompose_hangul(ch):
    """Decompose a Hangul syllable into (lead, vowel, tail) jamo indices."""
    code = ord(ch) - _HANGUL_BASE
    if code < 0 or code > (_HANGUL_END - _HANGUL_BASE):
        return None
    lead = code // (_N_VOWEL * _N_TAIL)
    vowel = (code % (_N_VOWEL * _N_TAIL)) // _N_TAIL
    tail = code % _N_TAIL
    return lead, vowel, tail


def compose_hangul(lead, vowel, tail):
    """Compose Hangul syllable from jamo indices."""
    return chr(_HANGUL_BASE + lead * (_N_VOWEL * _N_TAIL) + vowel * _N_TAIL + tail)


# ---------------------------------------------------------------------------
# Embedding index for mega-blocks
# ---------------------------------------------------------------------------

class EmbeddingIndex:
    """Nearest-neighbor index for mega-block character identification.

    For blocks with >MEGA_BLOCK_THRESHOLD characters, the per-block linear
    head can't learn from limited samples.  Instead, render reference images
    of each character, extract backbone features, and do cosine similarity
    at inference time.
    """

    def __init__(self):
        self.embeddings = {}   # block_idx -> (n_chars, feat_dim) tensor
        self.mega_blocks = set()

    def save(self, path):
        torch.save({
            "embeddings": {k: v.cpu() for k, v in self.embeddings.items()},
            "mega_blocks": self.mega_blocks,
        }, path)

    @staticmethod
    def load(path, device):
        data = torch.load(path, map_location=device, weights_only=False)
        idx = EmbeddingIndex()
        idx.embeddings = {k: v.to(device) for k, v in data["embeddings"].items()}
        idx.mega_blocks = data["mega_blocks"]
        return idx

    def lookup_batch(self, features, block_indices):
        """Batch lookup for multiple samples.

        Args:
            features: (B, feat_dim) tensor
            block_indices: (B,) tensor of block indices

        Returns:
            local_indices: (B,) tensor
            confidences: (B,) tensor
        """
        B = features.size(0)
        local_indices = torch.zeros(B, dtype=torch.long, device=features.device)
        confidences = torch.zeros(B, device=features.device)

        unique_blocks = block_indices.unique()
        for bid in unique_blocks:
            bid_int = bid.item()
            if bid_int not in self.embeddings:
                continue
            mask = (block_indices == bid)
            indices = mask.nonzero(as_tuple=True)[0]
            batch_feats = features[indices]
            refs = self.embeddings[bid_int]
            sim = F.cosine_similarity(
                batch_feats.unsqueeze(1), refs.unsqueeze(0), dim=2)
            best = sim.argmax(dim=1)
            best_conf = (sim.max(dim=1).values + 1.0) / 2.0
            local_indices[indices] = best
            confidences[indices] = best_conf

        return local_indices, confidences


# ---------------------------------------------------------------------------
# Detector config
# ---------------------------------------------------------------------------

@dataclass
class DetectorConfig:
    """Configuration for the sliding window character detector."""
    # Scale pyramid: at scale s, a char that's c pixels becomes c*s pixels
    scales: tuple = (0.5, 1.0, 2.0, 4.0)
    # Window size in pixels (at each scale level)
    window_size: int = 64
    # Stride as fraction of window size (0.5 = 50% overlap)
    stride_fraction: float = 0.5
    # Padding multiplier around window for context
    pad_multiplier: float = 2.0
    confidence_threshold: float = 0.3
    nms_iou_threshold: float = 0.3
    max_batch_size: int = 256
    # Background rejection: minimum pixel std dev (0-1 scale) for a window
    # to be considered non-blank.  Blank patches (uniform color) have std~0.
    min_pixel_std: float = 0.10
    # Minimum edge density (fraction of edge pixels via Sobel) to keep a window
    min_edge_density: float = 0.06


# ---------------------------------------------------------------------------
# Detector state
# ---------------------------------------------------------------------------

class CharacterDetector:
    """Holds the loaded model and metadata for character detection."""

    def __init__(self, model, block_to_chars, device, input_size=128,
                 embedding_index=None):
        self.model = model
        self.block_to_chars = block_to_chars
        self.device = device
        self.input_size = input_size
        self.embedding_index = embedding_index
        self.model.eval()

    def chars_for_block(self, block_idx):
        return self.block_to_chars.get(block_idx, [])

    def is_mega_block(self, block_idx):
        if self.embedding_index is None:
            return False
        return block_idx in self.embedding_index.mega_blocks

    def predict_with_embeddings(self, img_batch):
        """Predict using classifier for small blocks, embeddings for mega-blocks."""
        model = self.model
        feats = model.extract_features(img_batch)
        block_logits = model.block_head(feats)
        block_probs = torch.softmax(block_logits, dim=1)
        block_preds = block_probs.argmax(dim=1)
        block_confs = block_probs.max(dim=1).values

        B = img_batch.size(0)
        char_local_idx = torch.zeros(B, dtype=torch.long, device=self.device)
        char_confs = torch.zeros(B, device=self.device)

        is_mega = torch.tensor(
            [self.is_mega_block(bp.item()) for bp in block_preds],
            device=self.device)
        mega_mask = is_mega.bool()
        small_mask = ~mega_mask

        # Small blocks: use char_heads
        if small_mask.any():
            small_idx = small_mask.nonzero(as_tuple=True)[0]
            small_feats = feats[small_idx]
            small_blocks = block_preds[small_idx]

            unique_blocks = small_blocks.unique()
            for bid in unique_blocks:
                bid_int = bid.item()
                key = str(bid_int)
                if key not in model.char_heads:
                    continue
                bmask = (small_blocks == bid)
                indices = bmask.nonzero(as_tuple=True)[0]
                block_feats = small_feats[indices]
                logits = model.char_heads[key](block_feats)
                probs = torch.softmax(logits, dim=1)
                local_preds = probs.argmax(dim=1)
                local_confs = probs.max(dim=1).values
                orig_indices = small_idx[indices]
                char_local_idx[orig_indices] = local_preds
                char_confs[orig_indices] = local_confs

        # Mega blocks: use embedding retrieval
        if mega_mask.any() and self.embedding_index is not None:
            mega_idx = mega_mask.nonzero(as_tuple=True)[0]
            mega_feats = F.normalize(feats[mega_idx], dim=1)
            mega_blocks = block_preds[mega_idx]

            local_preds, local_confs = self.embedding_index.lookup_batch(
                mega_feats, mega_blocks)
            char_local_idx[mega_idx] = local_preds
            char_confs[mega_idx] = local_confs

        return block_preds, char_local_idx, block_confs, char_confs


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_detector(checkpoint, device, embedding_index_data=None):
    """Load a CharacterDetector from a checkpoint dict.

    Args:
        checkpoint: dict with 'model_state_dict' and 'block_to_chars'
        device: torch device string
        embedding_index_data: optional pre-loaded EmbeddingIndex, or None
    """
    from train_02_char import HierarchicalCharNet, CHAR_INPUT_SIZE

    block_to_chars = checkpoint["block_to_chars"]
    input_size = checkpoint.get("input_size", CHAR_INPUT_SIZE)

    model = HierarchicalCharNet(
        backbone="resnet18",
        block_to_chars=block_to_chars,
    ).to(device)

    state = checkpoint["model_state_dict"]
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, _ = model.load_state_dict(state, strict=False)
    if missing:
        logger.info("Note: %d new params initialized randomly", len(missing))
    model.eval()

    return CharacterDetector(
        model, block_to_chars, device, input_size, embedding_index_data)


# ---------------------------------------------------------------------------
# Window extraction
# ---------------------------------------------------------------------------

def _extract_windows(page_image, config):
    """Extract sliding windows at multiple scales.

    Returns:
        windows: list of PIL Images
        window_bboxes: list of [x1, y1, x2, y2] in page coordinates
        window_bg_colors: list of background color tuples
    """
    page_w, page_h = page_image.size
    ws = config.window_size
    stride = int(ws * config.stride_fraction)
    pad_px = int(ws * config.pad_multiplier)

    windows = []
    window_bboxes = []
    window_bg_colors = []

    for scale in config.scales:
        scaled_w = int(page_w * scale)
        scaled_h = int(page_h * scale)
        if scaled_w < ws or scaled_h < ws:
            continue

        scaled_img = page_image.resize((scaled_w, scaled_h), Image.LANCZOS)
        scaled_arr = np.array(scaled_img)

        for y in range(0, scaled_h - ws + 1, stride):
            for x in range(0, scaled_w - ws + 1, stride):
                wx1, wy1 = x, y
                wx2, wy2 = x + ws, y + ws

                px1 = max(0, wx1 - pad_px)
                py1 = max(0, wy1 - pad_px)
                px2 = min(scaled_w, wx2 + pad_px)
                py2 = min(scaled_h, wy2 + pad_px)

                crop = scaled_img.crop((px1, py1, px2, py2))

                crop_arr = np.array(crop)
                corners = []
                h, w = crop_arr.shape[:2]
                for cy, cx in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
                    corners.append(crop_arr[cy, cx])
                bg_color = tuple(int(v) for v in np.median(corners, axis=0))

                windows.append(crop)
                window_bg_colors.append(bg_color)

                page_x1 = wx1 / scale
                page_y1 = wy1 / scale
                page_x2 = wx2 / scale
                page_y2 = wy2 / scale
                window_bboxes.append([page_x1, page_y1, page_x2, page_y2])

    return windows, window_bboxes, window_bg_colors


# ---------------------------------------------------------------------------
# Background rejection filter
# ---------------------------------------------------------------------------

def _reject_blank_windows(retina_tensors, window_bboxes, config):
    """Filter out blank/low-contrast windows before classification.

    Rejects windows where:
    - Pixel std dev < min_pixel_std (uniform color patches)
    - Edge density < min_edge_density (no structure)

    Returns filtered (tensors, bboxes) lists.
    """
    keep_tensors = []
    keep_bboxes = []
    rejected = 0

    for t, bbox in zip(retina_tensors, window_bboxes):
        # Pixel standard deviation (across all channels)
        std = t.std().item()
        if std < config.min_pixel_std:
            rejected += 1
            continue

        # Edge density via simple gradient magnitude
        # Convert to grayscale: mean across channels
        gray = t.mean(dim=0)  # (H, W)
        # Sobel-like gradient: abs difference of adjacent pixels
        dx = (gray[:, 1:] - gray[:, :-1]).abs()
        dy = (gray[1:, :] - gray[:-1, :]).abs()
        # Count pixels with gradient > threshold as "edge pixels"
        edge_thresh = 0.05
        edge_count = (dx > edge_thresh).sum() + (dy > edge_thresh).sum()
        total_pixels = gray.numel()
        edge_density = edge_count.item() / total_pixels

        if edge_density < config.min_edge_density:
            rejected += 1
            continue

        keep_tensors.append(t)
        keep_bboxes.append(bbox)

    logger.info("Background rejection: %d/%d windows rejected (kept %d)",
                rejected, len(retina_tensors), len(keep_tensors))
    return keep_tensors, keep_bboxes


# ---------------------------------------------------------------------------
# Main detection pipeline
# ---------------------------------------------------------------------------

def detect_characters(page_image, detector, config=None):
    """Run multi-scale sliding window detection on a page image.

    Args:
        page_image: PIL Image (RGB)
        detector: CharacterDetector instance
        config: DetectorConfig (uses defaults if None)

    Returns:
        List of {"bbox": [x1,y1,x2,y2], "char": str, "confidence": float}
    """
    if config is None:
        config = DetectorConfig()

    t0 = time.time()

    # 1. Extract windows at multiple scales
    windows, window_bboxes, bg_colors = _extract_windows(page_image, config)
    n_windows = len(windows)
    if n_windows == 0:
        return []

    logger.info("Extracted %d windows across %d scales", n_windows, len(config.scales))

    # 2. Resize windows to model's input size
    input_sz = detector.input_size
    retina_tensors = []
    for win, bg in zip(windows, bg_colors):
        w, h = win.size
        sc = min(input_sz / w, input_sz / h)
        nw, nh = max(1, int(w * sc)), max(1, int(h * sc))
        resized = win.resize((nw, nh), Image.LANCZOS)
        canvas = Image.new("RGB", (input_sz, input_sz), bg)
        canvas.paste(resized, ((input_sz - nw) // 2, (input_sz - nh) // 2))
        t = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float() / 255.0
        retina_tensors.append(t)

    # 2b. Reject blank/low-contrast windows before classification
    retina_tensors, window_bboxes = _reject_blank_windows(
        retina_tensors, window_bboxes, config)
    n_windows = len(retina_tensors)
    if n_windows == 0:
        return []

    # 3. Batch classify all windows
    all_block_preds = []
    all_char_local = []
    all_block_confs = []
    all_char_confs = []

    device = detector.device
    use_embeddings = detector.embedding_index is not None

    with torch.no_grad():
        for i in range(0, n_windows, config.max_batch_size):
            batch = torch.stack(retina_tensors[i:i + config.max_batch_size]).to(device)

            with torch.amp.autocast("cuda", enabled=device == "cuda" or (hasattr(device, 'type') and device.type == "cuda")):
                if use_embeddings:
                    block_preds, char_local, block_confs, char_confs = \
                        detector.predict_with_embeddings(batch)
                else:
                    block_preds, char_local, block_confs, char_confs = \
                        detector.model.predict(batch)

            all_block_preds.append(block_preds.cpu())
            all_char_local.append(char_local.cpu())
            all_block_confs.append(block_confs.cpu())
            all_char_confs.append(char_confs.cpu())

    block_preds = torch.cat(all_block_preds)
    char_local = torch.cat(all_char_local)
    block_confs = torch.cat(all_block_confs)
    char_confs = torch.cat(all_char_confs)

    # 4. Filter by confidence and map to characters
    combined_confs = block_confs * char_confs

    detections = []
    bboxes_for_nms = []
    scores_for_nms = []

    for i in range(n_windows):
        conf = combined_confs[i].item()
        if conf < config.confidence_threshold:
            continue

        block_idx = block_preds[i].item()
        local_idx = char_local[i].item()

        block_chars = detector.chars_for_block(block_idx)
        if local_idx >= len(block_chars):
            continue

        ch = block_chars[local_idx]
        bbox = window_bboxes[i]

        detections.append({
            "bbox": [int(round(bbox[0])), int(round(bbox[1])),
                     int(round(bbox[2])), int(round(bbox[3]))],
            "char": ch,
            "confidence": round(conf, 4),
        })
        bboxes_for_nms.append(bbox)
        scores_for_nms.append(conf)

    logger.info("Pre-NMS detections: %d", len(detections))

    if not detections:
        return []

    # 5. NMS in page coordinate space
    boxes_t = torch.tensor(bboxes_for_nms, dtype=torch.float32)
    scores_t = torch.tensor(scores_for_nms, dtype=torch.float32)
    keep = torchvision.ops.nms(boxes_t, scores_t, config.nms_iou_threshold)
    keep = keep.tolist()

    final = [detections[i] for i in keep]
    final.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))

    elapsed = time.time() - t0
    logger.info("Final detections: %d (%.1fs)", len(final), elapsed)

    return final


def detect_characters_raw(page_image, detector, config=None):
    """Run multi-scale detection WITHOUT NMS or background rejection.

    Returns all detections above the confidence threshold with pixel
    statistics (pixel_std, edge_density) attached so the caller can
    tune thresholds externally in a nested grid search.

    Returns:
        List of {"bbox", "char", "confidence", "pixel_std", "edge_density"}
    """
    if config is None:
        config = DetectorConfig(confidence_threshold=0.1)

    t0 = time.time()

    windows, window_bboxes, bg_colors = _extract_windows(page_image, config)
    n_windows = len(windows)
    if n_windows == 0:
        return []

    logger.info("Extracted %d windows across %d scales", n_windows, len(config.scales))

    input_sz = detector.input_size
    retina_tensors = []
    pixel_stats = []  # (pixel_std, edge_density) per window

    for win, bg in zip(windows, bg_colors):
        w, h = win.size
        sc = min(input_sz / w, input_sz / h)
        nw, nh = max(1, int(w * sc)), max(1, int(h * sc))
        resized = win.resize((nw, nh), Image.LANCZOS)
        canvas = Image.new("RGB", (input_sz, input_sz), bg)
        canvas.paste(resized, ((input_sz - nw) // 2, (input_sz - nh) // 2))
        t = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float() / 255.0
        retina_tensors.append(t)

        # Compute pixel stats for this window
        std_val = t.std().item()
        gray = t.mean(dim=0)
        dx = (gray[:, 1:] - gray[:, :-1]).abs()
        dy = (gray[1:, :] - gray[:-1, :]).abs()
        edge_count = (dx > 0.05).sum() + (dy > 0.05).sum()
        edge_density = edge_count.item() / gray.numel()
        pixel_stats.append((round(std_val, 4), round(edge_density, 4)))

    # Classify ALL windows (no background rejection — that's tuned on CPU)
    all_block_preds = []
    all_char_local = []
    all_block_confs = []
    all_char_confs = []

    device = detector.device
    use_embeddings = detector.embedding_index is not None

    with torch.no_grad():
        for i in range(0, n_windows, config.max_batch_size):
            batch = torch.stack(retina_tensors[i:i + config.max_batch_size]).to(device)
            with torch.amp.autocast("cuda", enabled=(
                device == "cuda" or (hasattr(device, 'type') and device.type == "cuda")
            )):
                if use_embeddings:
                    block_preds, char_local, block_confs, char_confs = \
                        detector.predict_with_embeddings(batch)
                else:
                    block_preds, char_local, block_confs, char_confs = \
                        detector.model.predict(batch)
            all_block_preds.append(block_preds.cpu())
            all_char_local.append(char_local.cpu())
            all_block_confs.append(block_confs.cpu())
            all_char_confs.append(char_confs.cpu())

    block_preds = torch.cat(all_block_preds)
    char_local = torch.cat(all_char_local)
    block_confs = torch.cat(all_block_confs)
    char_confs = torch.cat(all_char_confs)

    combined_confs = block_confs * char_confs

    detections = []
    for i in range(n_windows):
        conf = combined_confs[i].item()
        if conf < config.confidence_threshold:
            continue

        block_idx = block_preds[i].item()
        local_idx = char_local[i].item()
        block_chars = detector.chars_for_block(block_idx)
        if local_idx >= len(block_chars):
            continue

        ch = block_chars[local_idx]
        bbox = window_bboxes[i]
        pstd, pedge = pixel_stats[i]
        detections.append({
            "bbox": [int(round(bbox[0])), int(round(bbox[1])),
                     int(round(bbox[2])), int(round(bbox[3]))],
            "char": ch,
            "confidence": round(conf, 4),
            "pixel_std": pstd,
            "edge_density": pedge,
        })

    elapsed = time.time() - t0
    logger.info("Raw detections (no NMS, no bg filter): %d (%.1fs)",
                len(detections), elapsed)
    return detections


# ---------------------------------------------------------------------------
# Block prior weighting — boost common blocks, penalize exotic ones
# ---------------------------------------------------------------------------

# Block indices for common document characters (from blocks.py block table)
# These get prior=1.0; everything else gets a penalty
_COMMON_BLOCKS = {
    0,   # Basic Latin (ASCII)
    1,   # Latin-1 Supplement
    2,   # Latin Extended-A
    3,   # Latin Extended-B
    7,   # Greek and Coptic
    8,   # Cyrillic
    72,  # General Punctuation
    73,  # Superscripts and Subscripts
    74,  # Currency Symbols
    76,  # Letterlike Symbols
    77,  # Number Forms
    78,  # Arrows
    79,  # Mathematical Operators
}

# Prior weight for uncommon blocks — their confidence gets multiplied by this
_UNCOMMON_BLOCK_PRIOR = 0.3


def apply_block_priors(detections, block_to_chars=None):
    """Weight detection confidence by Unicode block frequency prior.

    Common blocks (Latin, Greek, math, punctuation) keep full confidence.
    Exotic blocks get confidence *= 0.3, so they lose to common blocks
    in NMS when they overlap.

    Modifies detections in-place and returns the list.
    """
    from create_lists import get_block_index

    for det in detections:
        ch = det["char"]
        block_idx = get_block_index(ch)
        if block_idx not in _COMMON_BLOCKS:
            det["confidence"] = round(det["confidence"] * _UNCOMMON_BLOCK_PRIOR, 4)

    return detections


def apply_nms(detections, nms_iou_threshold=0.3):
    """Cross-class cross-scale NMS using intersection / min(area) ratio.

    Two passes:
    1. Greedy suppression using intersection / min(area) — catches cross-scale
       overlaps where standard IoU fails.
    2. Strict containment sweep — any detection whose bbox is >50% inside
       another kept detection gets removed, regardless of the IoU threshold.
       This catches nested boxes that the ratio threshold might miss.

    Both passes are class-agnostic: 'A' suppresses '℀' if they overlap.
    """
    if not detections:
        return []

    # Pass 1: greedy intersection/min(area) suppression
    ranked = sorted(enumerate(detections), key=lambda x: -x[1]["confidence"])
    suppressed = set()
    keep_indices = []

    for idx, det in ranked:
        if idx in suppressed:
            continue
        keep_indices.append(idx)
        ax1, ay1, ax2, ay2 = det["bbox"]
        area_a = max((ax2 - ax1) * (ay2 - ay1), 1)

        for jdx, other in ranked:
            if jdx in suppressed or jdx == idx:
                continue
            bx1, by1, bx2, by2 = other["bbox"]
            area_b = max((bx2 - bx1) * (by2 - by1), 1)

            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            intersection = iw * ih

            if intersection == 0:
                continue

            smaller_area = min(area_a, area_b)
            ratio = intersection / smaller_area

            if ratio > nms_iou_threshold:
                suppressed.add(jdx)

    after_pass1 = [detections[i] for i in keep_indices]

    # Pass 2: strict containment sweep — remove any box that is >50%
    # contained within another box (catches nested boxes from different scales)
    after_pass1.sort(key=lambda d: -d["confidence"])
    contained = [False] * len(after_pass1)

    for i in range(len(after_pass1)):
        if contained[i]:
            continue
        ax1, ay1, ax2, ay2 = after_pass1[i]["bbox"]
        for j in range(i + 1, len(after_pass1)):
            if contained[j]:
                continue
            bx1, by1, bx2, by2 = after_pass1[j]["bbox"]
            area_j = max((bx2 - bx1) * (by2 - by1), 1)

            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            intersection = iw * ih

            # If >50% of j's area is inside i, suppress j
            if intersection > 0.5 * area_j:
                contained[j] = True

    result = [d for d, c in zip(after_pass1, contained) if not c]
    result.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))
    return result



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_main():
    parser = argparse.ArgumentParser(
        description="Run HierarchicalCharNet detector on an image")
    parser.add_argument("image", help="Path to page image")
    parser.add_argument("--model", default="model_02_char.pth",
                        help="Path to model checkpoint (default: model_02_char.pth)")
    parser.add_argument("--embeddings", default=None,
                        help="Optional path to embedding index for mega-blocks")
    parser.add_argument("--visualize", action="store_true",
                        help="Save annotated image to <image>.detections.png")
    parser.add_argument("--output", default=None,
                        help="Save JSON detections to this path")
    parser.add_argument("--device", default=None,
                        help="cuda / cpu (auto-detect by default)")
    args = parser.parse_args()

    import json as _json
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    emb_idx = EmbeddingIndex.load(args.embeddings, device) if args.embeddings else None

    detector = load_detector(checkpoint, device, emb_idx)
    page = Image.open(args.image).convert("RGB")
    detections = detect_characters(page, detector)

    print(f"{len(detections)} detections")
    if args.output:
        with open(args.output, "w") as f:
            _json.dump(detections, f, indent=2, ensure_ascii=False)
        print(f"Wrote {args.output}")

    if args.visualize:
        out_path = os.path.splitext(args.image)[0] + ".detections.png"
        vis = page.copy()
        draw = ImageDraw.Draw(vis)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=1)
            draw.text((x1, max(0, y1 - 14)), det["char"], fill="red", font=font)
        vis.save(out_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    _cli_main()
