"""Train HierarchicalCharNet: two-stage Unicode character classifier.

Stage A: backbone frozen, train block_head only on block-balanced renders.
Stage B: backbone frozen, train each char_head on its block's chars only.
Stage C: joint fine-tune everything together (optionally on real
         sliding-window crops via --sliding-window).

Outputs a checkpoint dict consumed by glyph-faerie/glyph_faerie/detection/model.py
and tiny-tessarachnid/detect_characters.py.

# [RECONSTRUCTED HEADER lines 1-59]
# Dependencies this script expects (these were part of the un-committed refactor
# the session was working on — they're NOT in the current create_lists.py /
# generate_training_data.py and will need to be restored):
#
#   from create_lists import (
#       NUM_BLOCKS, BLOCK_NAMES,           # block tables (see glyph-faerie/.../blocks.py)
#       get_block_index,                    # ch -> block index
#       build_block_char_map,               # chars -> (block_to_chars, char_to_block_local)
#   )
#   from generate_training_data import get_unicode_chars  # filter by installed fonts
#
# All of these exist in glyph-faerie/glyph_faerie/detection/blocks.py and can be
# back-ported. The session bash output confirms get_unicode_chars returned 61876
# chars across 218 active blocks after NFKC normalization.
# [/RECONSTRUCTED HEADER]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset, random_split

from create_lists import (
    NUM_BLOCKS, BLOCK_NAMES, get_block_index, build_block_char_map,
)
from generate_training_data import get_unicode_chars, discover_fonts

CHAR_INPUT_SIZE = 128  # input resolution for character classifier

_BACKBONE_CONFIGS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, 512),
    "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT, 512),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
}


class HierarchicalCharNet(nn.Module):
    """Two-stage hierarchical character classifier.

    Stage 1: predict Unicode block (~264 classes)
    Stage 2: predict character within the predicted block (per-block heads)
    """

    def __init__(self, backbone="resnet18", block_to_chars=None):
        super().__init__()
        factory, weights, feat_dim = _BACKBONE_CONFIGS[backbone]
        resnet = factory(weights=weights)
        self.feat_dim = feat_dim

        # Backbone (same structure as RetinaOCRNet for weight transfer)
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # ImageNet normalization
        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # Stage 1: block classifier
        self.num_blocks = NUM_BLOCKS
        self.block_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_blocks),
        )

        # Stage 2: per-block character heads
        # block_to_chars: dict mapping block_idx → list of chars in that block
        self.block_to_chars = block_to_chars or {}
        self.char_heads = nn.ModuleDict()
        self.block_sizes = {}  # block_idx → number of chars

        for block_idx, chars in self.block_to_chars.items():
            if block_idx < 0:  # skip unmapped blocks
                continue
            n = len(chars)
            self.block_sizes[block_idx] = n
            # Use a compact head — just one linear layer for small blocks,
            # two for larger ones
            if n <= 64:
                head = nn.Linear(feat_dim, n)
            else:
                head = nn.Sequential(
                    nn.Linear(feat_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, n),
                )
            self.char_heads[str(block_idx)] = head

        # Hangul jamo decomposition heads (block 147 = Hangul Syllables)
        # Instead of 11,172-way classification, predict 3 small components
        self.hangul_block_idx = 147
        self.hangul_lead_head = nn.Linear(feat_dim, 19)    # ㄱ-ㅎ
        self.hangul_vowel_head = nn.Linear(feat_dim, 21)   # ㅏ-ㅣ
        self.hangul_tail_head = nn.Linear(feat_dim, 28)    # ∅ + ㄱ-ㅎ

    def extract_features(self, img):
        """Extract backbone features from image tensor."""
        x = (img - self.img_mean) / self.img_std
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)  # (B, feat_dim)

    def forward(self, img, block_targets=None):
        """Forward pass.

        Args:
            img: (B, 3, H, W) image tensor
            block_targets: optional (B,) block index targets for teacher-forced
                          char head selection during training

        Returns:
            block_logits: (B, num_blocks) raw logits for block classification
            char_logits_list: list of (indices, logits, block_id) tuples — one per
                             active block in the batch.
        """
        feats = self.extract_features(img)  # (B, feat_dim)
        block_logits = self.block_head(feats)  # (B, num_blocks)

        # Determine which block each sample belongs to
        if block_targets is not None:
            block_ids = block_targets
        else:
            block_ids = block_logits.argmax(dim=1)

        # Group samples by block and run per-block heads
        char_logits_list = []
        unique_blocks = block_ids.unique()
        for bid in unique_blocks:
            bid_int = bid.item()
            key = str(bid_int)
            if key not in self.char_heads:
                continue
            mask = (block_ids == bid)
            indices = mask.nonzero(as_tuple=True)[0]
            block_feats = feats[indices]
            logits = self.char_heads[key](block_feats)
            char_logits_list.append((indices, logits, bid_int))

        return block_logits, char_logits_list

    def predict_hangul_jamo(self, feats):
        """Predict Hangul syllable via jamo decomposition.

        Args:
            feats: (N, feat_dim) backbone features for Hangul samples

        Returns:
            local_indices: (N,) index into block_to_chars[147]
            confs: (N,) confidence scores
        """
        lead_logits = self.hangul_lead_head(feats)
        vowel_logits = self.hangul_vowel_head(feats)
        tail_logits = self.hangul_tail_head(feats)

        lead_probs = torch.softmax(lead_logits, dim=1)
        vowel_probs = torch.softmax(vowel_logits, dim=1)
        tail_probs = torch.softmax(tail_logits, dim=1)

        lead_idx = lead_probs.argmax(dim=1)
        vowel_idx = vowel_probs.argmax(dim=1)
        tail_idx = tail_probs.argmax(dim=1)

        lead_conf = lead_probs.max(dim=1).values
        vowel_conf = vowel_probs.max(dim=1).values
        tail_conf = tail_probs.max(dim=1).values

        # Compose syllable code points
        syllable_offsets = lead_idx * 588 + vowel_idx * 28 + tail_idx  # offset from 0xAC00
        confs = (lead_conf * vowel_conf * tail_conf) ** (1.0 / 3.0)  # geometric mean

        # Map syllable offset to local index in block_to_chars[147]
        hangul_chars = self.block_to_chars.get(self.hangul_block_idx, [])
        if not hangul_chars:
            return torch.zeros_like(lead_idx), torch.zeros_like(lead_conf)

        # Build offset → local_idx mapping
        char_to_local = {}
        for li, ch in enumerate(hangul_chars):
            offset = ord(ch) - 0xAC00
            char_to_local[offset] = li

        local_indices = torch.zeros_like(lead_idx)
        for i in range(feats.size(0)):
            offset = syllable_offsets[i].item()
            local_indices[i] = char_to_local.get(offset, 0)

        return local_indices, confs

    def predict(self, img):
        """Inference: return predicted (block_idx, char_local_idx, block_conf, char_conf) per sample."""
        feats = self.extract_features(img)
        block_logits = self.block_head(feats)
        block_probs = torch.softmax(block_logits, dim=1)
# [RECONSTRUCTED lines 240-459 — predict() completion + char render helpers]
        block_preds = block_probs.argmax(dim=1)
        block_confs = block_probs.max(dim=1).values

        B = img.size(0)
        char_local_idx = torch.zeros(B, dtype=torch.long, device=img.device)
        char_confs = torch.zeros(B, device=img.device)

        unique_blocks = block_preds.unique()
        for bid in unique_blocks:
            bid_int = bid.item()
            mask = (block_preds == bid)
            indices = mask.nonzero(as_tuple=True)[0]
            block_feats = feats[indices]

            if bid_int == self.hangul_block_idx:
                local_idx, local_confs = self.predict_hangul_jamo(block_feats)
                char_local_idx[indices] = local_idx
                char_confs[indices] = local_confs
                continue

            key = str(bid_int)
            if key not in self.char_heads:
                continue
            logits = self.char_heads[key](block_feats)
            probs = torch.softmax(logits, dim=1)
            char_local_idx[indices] = probs.argmax(dim=1)
            char_confs[indices] = probs.max(dim=1).values

        return block_preds, char_local_idx, block_confs, char_confs


# ---------------------------------------------------------------------------
# Character rendering helpers
# ---------------------------------------------------------------------------

def _render_char(ch, font, input_size, bg=(255, 255, 255), fg=(0, 0, 0)):
    """Render a single character centered on an input_size x input_size canvas.

    Returns a uint8 (3, H, W) tensor, or None on failure.
    """
    try:
        # Measure
        bbox = font.getbbox(ch)
        w = max(1, bbox[2] - bbox[0])
        h = max(1, bbox[3] - bbox[1])
        # Render on a tight canvas with padding, then center on input_size canvas
        pad = 4
        tight = Image.new("RGB", (w + 2 * pad, h + 2 * pad), bg)
        ImageDraw.Draw(tight).text((pad - bbox[0], pad - bbox[1]), ch,
                                   font=font, fill=fg)
        # Scale to fit input_size
        scale = min(input_size / tight.width, input_size / tight.height)
        nw, nh = max(1, int(tight.width * scale)), max(1, int(tight.height * scale))
        resized = tight.resize((nw, nh), Image.LANCZOS)
        canvas = Image.new("RGB", (input_size, input_size), bg)
        canvas.paste(resized, ((input_size - nw) // 2, (input_size - nh) // 2))
        arr = np.array(canvas, dtype=np.uint8)
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    except Exception:
        return None


def _render_char_in_context(ch, font, input_size, sibling_chars,
                            bg=(255, 255, 255), fg=(0, 0, 0)):
    """Render `ch` surrounded by sibling chars from the same block.

    Picks a few random siblings, lays them out left/right of the target,
    then crops/scales so the target glyph is centered and fits input_size.
    """
    if not sibling_chars:
        return _render_char(ch, font, input_size, bg, fg)
    try:
        # Pick context: 1-3 chars on each side
        n_left = random.randint(1, 3)
        n_right = random.randint(1, 3)
        left = [random.choice(sibling_chars) for _ in range(n_left)]
        right = [random.choice(sibling_chars) for _ in range(n_right)]
        line = "".join(left) + ch + "".join(right)
        target_pos = sum(font.getbbox(c)[2] - font.getbbox(c)[0] for c in left)
        target_w = font.getbbox(ch)[2] - font.getbbox(ch)[0]

        bbox = font.getbbox(line)
        w = max(1, bbox[2] - bbox[0])
        h = max(1, bbox[3] - bbox[1])
        pad = 4
        strip = Image.new("RGB", (w + 2 * pad, h + 2 * pad), bg)
        ImageDraw.Draw(strip).text((pad - bbox[0], pad - bbox[1]), line,
                                   font=font, fill=fg)

        # Crop a window centered on the target glyph; window width = ~2.5 * target
        cx = pad + target_pos + target_w // 2
        cy = strip.height // 2
        half = max(target_w * 2, h)
        x1 = max(0, cx - half)
        x2 = min(strip.width, cx + half)
        y1 = 0
        y2 = strip.height
        crop = strip.crop((x1, y1, x2, y2))

        scale = min(input_size / crop.width, input_size / crop.height)
        nw, nh = max(1, int(crop.width * scale)), max(1, int(crop.height * scale))
        resized = crop.resize((nw, nh), Image.LANCZOS)
        canvas = Image.new("RGB", (input_size, input_size), bg)
        canvas.paste(resized, ((input_size - nw) // 2, (input_size - nh) // 2))
        arr = np.array(canvas, dtype=np.uint8)
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    except Exception:
        return _render_char(ch, font, input_size, bg, fg)
# [/RECONSTRUCTED]



# ---------------------------------------------------------------------------
# Parallel rendering
# ---------------------------------------------------------------------------

_NUM_RENDER_WORKERS = max(1, mp.cpu_count() - 2)


# [RECONSTRUCTED — pool worker holds block-sibling map as a global so tasks
# only need to pass block_idx, not full sibling lists]
_WORKER_BLOCK_CHAR_LISTS = {}

def _pool_init(block_char_lists=None):
    """Re-seed random in forked worker processes + stash block sibling map."""
    global _WORKER_BLOCK_CHAR_LISTS
    if block_char_lists is not None:
        _WORKER_BLOCK_CHAR_LISTS = block_char_lists
    random.seed(os.getpid() ^ int(time.time() * 1000))
# [/RECONSTRUCTED]


def _render_task(args):
    """Render one character image in a worker process.

    The 7th tuple element is now `block_idx` (or None); siblings are
    fetched from the worker's _WORKER_BLOCK_CHAR_LISTS map.
    """
    ch, font_path, face_idx, font_size, input_size, use_context, block_idx = args
    try:
        font = ImageFont.truetype(font_path, font_size, index=face_idx)
    except Exception:
        return None
    if use_context and block_idx is not None:
        siblings = _WORKER_BLOCK_CHAR_LISTS.get(block_idx, [])
        if siblings:
            return _render_char_in_context(ch, font, input_size, siblings)
    return _render_char(ch, font, input_size)


def _parallel_render(tasks, block_char_lists=None, desc=""):
    """Render character images in parallel using multiprocessing Pool.

    block_char_lists: dict[block_idx -> list[char]] for in-context rendering.
    Passed once to each worker via pool initializer to avoid per-task pickling.
    """
    n = len(tasks)
    if n == 0:
        return []
    print(f"  [{desc}] Rendering {n} images with {_NUM_RENDER_WORKERS} workers...")
    t0 = time.time()
    with mp.Pool(_NUM_RENDER_WORKERS,
                 initializer=_pool_init,
                 initargs=(block_char_lists or {},)) as pool:
        results = pool.map(_render_task, tasks, chunksize=64)
    elapsed = time.time() - t0
    ok = sum(1 for r in results if r is not None)
    print(f"  [{desc}] Rendered {ok}/{n} in {elapsed:.0f}s "
          f"({ok / max(elapsed, 0.1):.0f} img/s)")
    return results


# ---------------------------------------------------------------------------
# Stage A Dataset: Block-balanced (equal samples per block)
# ---------------------------------------------------------------------------

class BlockBalancedDataset(Dataset):
    """Renders characters with equal representation per block for block
    classifier training.  Each block gets `samples_per_block` images,
    regardless of how many chars are in it.
    """

    def __init__(self, chars, char_to_fonts, char_to_block_local,
                 samples_per_block=200, input_size=CHAR_INPUT_SIZE,
                 block_to_chars=None, context_ratio=0.8):
        self.input_size = input_size
        block_chars = defaultdict(list)
        for ch in chars:
            if ch == ' ':
# [RECONSTRUCTED lines 520-539 — finish BlockBalancedDataset char-binning + allocation]
                continue
            if ch not in char_to_block_local:
                continue
            block_idx, local_idx = char_to_block_local[ch]
            if block_idx < 0:
                continue
            block_chars[block_idx].append((ch, local_idx))

        # Build sibling lists for in-context rendering
        _block_char_lists = {}
        if block_to_chars:
            _block_char_lists = {bi: [c for c in cs if c != ' ']
                                 for bi, cs in block_to_chars.items()
                                 if bi >= 0 and cs}

        # Active blocks = blocks with at least one renderable char
        active_blocks = [bi for bi, cs in block_chars.items() if cs]
        total_planned = len(active_blocks) * samples_per_block
        self.images = torch.zeros((total_planned, 3, input_size, input_size),
                                  dtype=torch.uint8)
# [/RECONSTRUCTED]
        self.block_ids = torch.zeros(total_planned, dtype=torch.long)
        self.local_ids = torch.zeros(total_planned, dtype=torch.long)

        print(f"[BlockBalanced] {len(active_blocks)} blocks × {samples_per_block} spc "
              f"= {total_planned} samples ({total_planned * 3 * input_size**2 / 1e9:.1f} GB)"
              f" (context_ratio={context_ratio})",
              flush=True)

        generated = 0
        t_start = time.time()

        # Pre-generate all render tasks (2x oversample for failures)
        tasks = []
        task_meta = []  # (block_idx, local_idx)
        for block_idx in active_blocks:
            chars_in_block = block_chars[block_idx]
            has_siblings = bool(_block_char_lists.get(block_idx))
            for _ in range(samples_per_block * 2):
                ch, local_idx = random.choice(chars_in_block)
                fonts_for_char = char_to_fonts.get(ch, [])
                if not fonts_for_char:
                    continue
                font_path, face_idx = random.choice(fonts_for_char)
                font_size = random.randint(14, 60)
                use_ctx = has_siblings and random.random() < context_ratio
                tasks.append((ch, font_path, face_idx, font_size, input_size,
                              use_ctx, block_idx if use_ctx else None))
                task_meta.append((block_idx, local_idx))

        results = _parallel_render(tasks, _block_char_lists, "BlockBalanced")

        # Collect results, respecting per-block limits
        block_counts = defaultdict(int)
        for img_t, (block_idx, local_idx) in zip(results, task_meta):
            if img_t is None:
                continue
            if block_counts[block_idx] >= samples_per_block:
                continue
            if generated >= total_planned:
                break
            self.images[generated] = img_t
            self.block_ids[generated] = block_idx
            self.local_ids[generated] = local_idx
            generated += 1
            block_counts[block_idx] += 1

        # Trim
        if generated < total_planned:
            self.images = self.images[:generated].contiguous()
            self.block_ids = self.block_ids[:generated]
# [RECONSTRUCTED lines 590-629 — finish BlockBalanced, open CharBalanced]
            self.local_ids = self.local_ids[:generated]

        # Shuffle for training
        perm = torch.randperm(generated)
        self.images = self.images[perm]
        self.block_ids = self.block_ids[perm]
        self.local_ids = self.local_ids[perm]

        n_blocks = len(self.block_ids.unique())
        elapsed = time.time() - t_start
        print(f"[BlockBalanced] Generated {generated} samples across "
              f"{n_blocks} blocks ({elapsed:.0f}s)", flush=True)

    def __len__(self):
        return len(self.block_ids)

    def __getitem__(self, idx):
        return self.images[idx].float() / 255.0, self.block_ids[idx], self.local_ids[idx]


# ---------------------------------------------------------------------------
# Stage B Dataset: Char-balanced within each block (lazy on-demand rendering)
# ---------------------------------------------------------------------------

class CharBalancedDataset(Dataset):
    """Lazy on-the-fly character rendering dataset for char_heads training.

    Stores only metadata (~MB), renders images on demand in DataLoader workers.
    Scales to any number of characters without pre-allocating huge tensors.
    """

    def __init__(self, chars, char_to_fonts, char_to_block_local,
                 samples_per_char=50, max_block_size=1000,
                 input_size=CHAR_INPUT_SIZE,
                 block_to_chars=None, context_ratio=0.8,
                 block_weights=None):
        self.input_size = input_size
        self.context_ratio = context_ratio

        block_chars = defaultdict(list)
        for ch in chars:
            if ch == ' ' or ch not in char_to_block_local:
                continue
            block_idx, local_idx = char_to_block_local[ch]
            if block_idx < 0:
                continue
            block_chars[block_idx].append((ch, local_idx))

        # Build block char lists for in-context rendering
        self._block_char_lists = {}
        if block_to_chars:
            self._block_char_lists = {bi: [c for c in cs if c != ' ']
                                      for bi, cs in block_to_chars.items()
                                      if bi >= 0 and cs}

        # Split blocks into small (full spc) and large (reduced spc)
        small_blocks = {b: cs for b, cs in block_chars.items()
                        if len(cs) <= max_block_size and len(cs) > 0}
        large_blocks = {b: cs for b, cs in block_chars.items()
                        if len(cs) > max_block_size}
        large_spc = max(2, samples_per_char // 10)

        # Build item list: (ch, block_idx, local_idx, fonts_for_char)
        # Each char repeated `spc` times, then shuffled
        self.items = []
        for block_idx, chars_in_block in list(small_blocks.items()) + list(large_blocks.items()):
            spc = samples_per_char if block_idx in small_blocks else large_spc
            for ch, local_idx in chars_in_block:
                fonts_for_char = char_to_fonts.get(ch, [])
                if not fonts_for_char:
                    continue
                for _ in range(spc):
                    self.items.append((ch, block_idx, local_idx, fonts_for_char))

        random.shuffle(self.items)

        n_chars = len(set((bi, li) for _, bi, li, _ in self.items))
        n_blocks = len(set(bi for _, bi, _, _ in self.items))
        print(f"[CharBalanced] {n_blocks} blocks, {n_chars} chars, "
              f"{len(self.items):,} samples (lazy, ~0 GB upfront) "
              f"(context_ratio={context_ratio})")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ch, block_idx, local_idx, fonts_for_char = self.items[idx]
        font_path, face_idx = random.choice(fonts_for_char)
        font_size = random.randint(14, 60)
        try:
            font = ImageFont.truetype(font_path, font_size, index=face_idx)
        except Exception:
            return (torch.zeros(3, self.input_size, self.input_size),
                    torch.tensor(block_idx), torch.tensor(local_idx))

        siblings = self._block_char_lists.get(block_idx, [])
        if siblings and random.random() < self.context_ratio:
            img_t = _render_char_in_context(ch, font, self.input_size, siblings)
        else:
            img_t = _render_char(ch, font, self.input_size)

        if img_t is None:
            # Fallback: try isolated render with default font size
            try:
                font2 = ImageFont.truetype(font_path, 32, index=face_idx)
                img_t = _render_char(ch, font2, self.input_size)
            except Exception:
                pass

        if img_t is None:
            img_t = torch.zeros(3, self.input_size, self.input_size, dtype=torch.uint8)

        return img_t.float() / 255.0, torch.tensor(block_idx), torch.tensor(local_idx)


# ---------------------------------------------------------------------------
# Stage C Dataset: Real sliding window crops from synthetic pages
# ---------------------------------------------------------------------------

class SlidingWindowDataset(Dataset):
    """Extracts real sliding window crops from synthetic pages with GT labels.

    Generates synthetic pages, runs the actual sliding window extraction
    (same code as inference), matches each window to GT characters by IoU,
    and stores the matched crops as training data. Zero domain gap.
    """

    def __init__(self, fonts, num_pages=100, input_size=CHAR_INPUT_SIZE,
                 scales=(0.5, 1.0, 2.0), min_iou=0.3,
                 page_width=1024, page_height=1400):
        from detect_characters import _extract_windows, DetectorConfig, \
            _reject_blank_windows
        from generate_training_data import SyntheticPage
        from create_lists import get_block_index

        self.input_size = input_size
        config = DetectorConfig(scales=scales)

        self.images = []
        self.block_ids = []
        self.local_ids = []

        print(f"[SlidingWindow] Generating {num_pages} pages, "
              f"scales={scales}, min_iou={min_iou}...")
        t0 = time.time()

        for pg in range(num_pages):
            page = SyntheticPage(fonts, page_width, page_height)

            # Collect GT chars
            gt_chars = []
            for para in page.paragraphs:
                for line in para["lines"]:
                    for word in line["words"]:
                        for ch_data in word["characters"]:
                            bbox = ch_data["bbox"]
                            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                            if w <= 0 or h <= 0:
                                continue
                            ch = ch_data["char"]
                            if ch == ' ':
                                continue
                            bi = get_block_index(ch)
                            if bi < 0 or ch not in char_to_block_local:
                                continue
                            _, li = char_to_block_local[ch]
                            gt_chars.append({
                                "bbox": bbox, "block": bi, "local": li,
                            })

            if not gt_chars:
                continue

            # Extract sliding windows (same as inference)
            windows, window_bboxes, bg_colors = _extract_windows(
                page.image, config)

            # Resize to model input
            retina_tensors = []
            for win, bg in zip(windows, bg_colors):
                w, h = win.size
                sc = min(input_size / w, input_size / h)
                nw, nh = max(1, int(w * sc)), max(1, int(h * sc))
                resized = win.resize((nw, nh), Image.LANCZOS)
                canvas = Image.new("RGB", (input_size, input_size), bg)
                canvas.paste(resized, ((input_size - nw) // 2,
                                       (input_size - nh) // 2))
                t = torch.from_numpy(np.array(canvas)).permute(
                    2, 0, 1).float() / 255.0
                retina_tensors.append(t)

            # Background rejection (same as inference)
            retina_tensors, window_bboxes = _reject_blank_windows(
                retina_tensors, window_bboxes, config)

            # Match each window to best GT char by IoU
            for t, wb in zip(retina_tensors, window_bboxes):
                best_iou = 0.0
                best_gt = None
                for gt in gt_chars:
                    gb = gt["bbox"]
                    ix1 = max(wb[0], gb[0])
                    iy1 = max(wb[1], gb[1])
                    ix2 = min(wb[2], gb[2])
                    iy2 = min(wb[3], gb[3])
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    aa = max((wb[2]-wb[0])*(wb[3]-wb[1]), 1)
                    ab = max((gb[2]-gb[0])*(gb[3]-gb[1]), 1)
                    iou = inter / (aa + ab - inter)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt
                if best_iou >= min_iou and best_gt is not None:
                    self.images.append(t)
                    self.block_ids.append(best_gt["block"])
                    self.local_ids.append(best_gt["local"])

            if (pg + 1) % 20 == 0:
                elapsed = time.time() - t0
                print(f"  Page {pg+1}/{num_pages}: {len(self.images)} samples "
                      f"({elapsed:.0f}s)")

        # Convert to tensors
        if self.images:
            self.images = torch.stack(self.images)
            self.block_ids = torch.tensor(self.block_ids, dtype=torch.long)
            self.local_ids = torch.tensor(self.local_ids, dtype=torch.long)
        else:
            self.images = torch.zeros(0, 3, input_size, input_size)
            self.block_ids = torch.zeros(0, dtype=torch.long)
            self.local_ids = torch.zeros(0, dtype=torch.long)

        n_blocks = len(set(self.block_ids.tolist())) if len(self.block_ids) > 0 else 0
        elapsed = time.time() - t0
        gb = self.images.numel() * 4 / 1e9
        print(f"[SlidingWindow] {len(self.images)} samples from {num_pages} pages, "
              f"{n_blocks} blocks, {gb:.1f} GB ({elapsed:.0f}s)")

    def __len__(self):
        return len(self.block_ids)

    def __getitem__(self, idx):
        return self.images[idx], self.block_ids[idx], self.local_ids[idx]

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class HierarchicalLoss(nn.Module):
    """Combined block + within-block character classification loss."""

    def __init__(self, block_weight=1.0, char_weight=1.0):
        super().__init__()
        self.block_loss = nn.CrossEntropyLoss()
        self.char_loss = nn.CrossEntropyLoss()
        self.block_weight = block_weight
        self.char_weight = char_weight

# [RECONSTRUCTED lines 759-785 — HierarchicalLoss.forward]
    def forward(self, block_logits, char_logits_list, block_targets, char_targets):
        """Returns (loss, block_loss_value, char_loss_value).

        block_logits: (B, num_blocks)
        char_logits_list: list of (indices, logits, block_id) from HierarchicalCharNet.forward
        block_targets: (B,) ground-truth block indices
        char_targets: (B,) ground-truth local char indices
        """
        block_loss = self.block_loss(block_logits, block_targets)

        char_loss = torch.zeros((), device=block_logits.device)
        n_groups = 0
        for indices, logits, _bid in char_logits_list:
            tgt = char_targets[indices]
            # Skip out-of-range targets (defensive; shouldn't happen with correct data)
            valid = tgt < logits.size(1)
            if valid.sum() == 0:
                continue
            char_loss = char_loss + self.char_loss(logits[valid], tgt[valid])
            n_groups += 1
        if n_groups > 0:
            char_loss = char_loss / n_groups

        total = self.block_weight * block_loss + self.char_weight * char_loss
        return total, float(block_loss.item()), float(char_loss.item())


# ---------------------------------------------------------------------------
# [/RECONSTRUCTED]
# DataParallel training wrapper
# ---------------------------------------------------------------------------

class _DPTrainWrapper(nn.Module):
    """Wraps model + loss for DataParallel: computes loss per-GPU, returns scalars."""

    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, imgs, block_targets, char_targets):
        block_logits, char_logits_list = self.model(imgs, block_targets)
        loss, lb, lc = self.criterion(block_logits, char_logits_list,
                                       block_targets, char_targets)
        with torch.no_grad():
            bc = (block_logits.argmax(1) == block_targets).sum().float()
            cc = torch.zeros(1, device=imgs.device)
            for indices, logits, bid in char_logits_list:
                t = char_targets[indices]
                v = t <= (logits.size(1) - 1)
                if v.sum() > 0:
                    cc += (logits.argmax(1)[v] == t[v]).sum().float()
            n = torch.tensor(float(imgs.size(0)), device=imgs.device)
            stats = torch.stack([bc, cc.squeeze(), n,
                                 torch.tensor(lb, device=imgs.device),
                                 torch.tensor(lc, device=imgs.device)])
        return loss.unsqueeze(0), stats.unsqueeze(0)  # [1], [1, 5]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _make_loader(ds, batch_size, shuffle, num_workers):
    """Create a DataLoader with optimal settings for in-memory datasets."""
    pw = num_workers > 0
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      drop_last=shuffle, persistent_workers=pw,
                      prefetch_factor=4 if pw else None)


def _wrap_model(model, criterion, gpu_ids):
    """Create training wrapper, optionally with DataParallel."""
    wrapper = _DPTrainWrapper(model, criterion)
    if gpu_ids and len(gpu_ids) > 1:
        wrapper = nn.DataParallel(wrapper, device_ids=gpu_ids)
    return wrapper


def _run_stages_ab(model, args, chars, char_to_fonts, char_to_block_local,
                   block_to_chars, device, scaler, total_params,
                   save_if_best, _save_checkpoint, gpu_ids=None,
                   block_weights=None):
    """Run Stage A (block detector) and Stage B (char classifiers)."""
    nw = args.num_workers

    # ===================================================================
    # STAGE A: Block detector training (block-balanced)
    # ===================================================================
    print(f"\n{'='*60}")
    print(f"STAGE A: Block detector ({args.block_spc} samples/block, "
          f"{args.epochs_block} epochs)")
    print(f"{'='*60}")

    print("Generating block-balanced data...")
    t0 = time.time()
    block_ds = BlockBalancedDataset(
        chars, char_to_fonts, char_to_block_local,
        samples_per_block=args.block_spc,
        input_size=args.input_size,
        block_to_chars=block_to_chars,
        context_ratio=args.context_ratio,
        block_weights=block_weights,
    )
    print(f"Block dataset: {len(block_ds)} samples in {time.time()-t0:.0f}s")

    val_size = min(5000, len(block_ds) // 10)
    train_ds, val_ds = random_split(block_ds, [len(block_ds) - val_size, val_size])
    train_loader = _make_loader(train_ds, args.batch_size, True, nw)
    val_loader = _make_loader(val_ds, args.batch_size, False, nw)

    # Freeze everything, then unfreeze only block_head
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "block_head" in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable:,} / {total_params:,} (block_head only)")

    block_criterion = HierarchicalLoss(block_weight=1.0, char_weight=0.0)
    wrapper = _wrap_model(model, block_criterion, gpu_ids)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_block, weight_decay=1e-4,
    )
# [RECONSTRUCTED lines 890-979 — Stage A training loop tail + Stage B setup]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_block)

    for epoch in range(args.epochs_block):
        v_block, v_char = _run_epoch(
            wrapper, train_loader, val_loader, optimizer, scheduler, scaler,
            device, epoch, args.epochs_block, "BlockDet")
        save_if_best(v_block, v_char)

    # ===================================================================
    # STAGE B: Char heads training (char-balanced)
    # ===================================================================
    print(f"\n{'='*60}")
    print(f"STAGE B: Char heads ({args.char_spc} samples/char, "
          f"{args.epochs_char} epochs)")
    print(f"{'='*60}")

    print("Generating char-balanced data...")
    t0 = time.time()
    char_ds = CharBalancedDataset(
        chars, char_to_fonts, char_to_block_local,
        samples_per_char=args.char_spc,
        max_block_size=1000,
        input_size=args.input_size,
        block_to_chars=block_to_chars,
        context_ratio=args.context_ratio,
        block_weights=block_weights,
    )
    print(f"Char dataset: {len(char_ds)} samples in {time.time()-t0:.0f}s")

    # Use more workers for lazy dataset (rendering happens in workers)
    char_nw = max(nw, 8)
    val_size = min(5000, len(char_ds) // 10)
    train_ds, val_ds = random_split(char_ds, [len(char_ds) - val_size, val_size])
    train_loader = _make_loader(train_ds, args.batch_size, True, char_nw)
    val_loader = _make_loader(val_ds, args.batch_size, False, char_nw)

    # Freeze backbone + block_head, unfreeze only char_heads + hangul heads
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "char_heads" in name or "hangul" in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable:,} / {total_params:,} (char_heads only)")

    char_criterion = HierarchicalLoss(block_weight=0.0, char_weight=1.0)
    wrapper = _wrap_model(model, char_criterion, gpu_ids)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_char, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_char)

    for epoch in range(args.epochs_char):
        v_block, v_char = _run_epoch(
            wrapper, train_loader, val_loader, optimizer, scheduler, scaler,
            device, epoch, args.epochs_char, "CharHeads")
        save_if_best(v_block, v_char)


def _run_epoch(wrapper, train_loader, val_loader, optimizer, scheduler, scaler,
               device, epoch, total_epochs, phase_name):
# [/RECONSTRUCTED]
    """Run one training epoch + validation. Returns (val_block_acc, val_char_acc).

    wrapper: _DPTrainWrapper (optionally wrapped in nn.DataParallel).
    Returns (loss, stats) where stats = [block_correct, char_correct, n, lb, lc].
    """
    wrapper.train()
    train_loss = 0.0
    train_block_correct = 0
    train_char_correct = 0
    train_total = 0

    for imgs, block_targets, char_targets in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        block_targets = block_targets.to(device, non_blocking=True)
        char_targets = char_targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                loss, stats = wrapper(imgs, block_targets, char_targets)
            loss = loss.mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                (p for p in wrapper.parameters() if p.requires_grad), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, stats = wrapper(imgs, block_targets, char_targets)
            loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in wrapper.parameters() if p.requires_grad), 1.0)
            optimizer.step()

        s = stats.sum(0) if stats.dim() > 1 else stats
        train_block_correct += s[0].item()
        train_char_correct += s[1].item()
        train_total += int(s[2].item())
        train_loss += loss.item() * int(s[2].item())

    scheduler.step()

    # Validation
    wrapper.eval()
    val_block_correct = 0
    val_char_correct = 0
    val_total = 0

    with torch.no_grad():
        for imgs, block_targets, char_targets in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            block_targets = block_targets.to(device, non_blocking=True)
            char_targets = char_targets.to(device, non_blocking=True)
            _, stats = wrapper(imgs, block_targets, char_targets)
            s = stats.sum(0) if stats.dim() > 1 else stats
            val_block_correct += s[0].item()
            val_char_correct += s[1].item()
            val_total += int(s[2].item())

    t_block = train_block_correct / max(1, train_total)
    t_char = train_char_correct / max(1, train_total)
    v_block = val_block_correct / max(1, val_total)
    v_char = val_char_correct / max(1, val_total)
    avg_loss = train_loss / max(1, train_total)

    print(f"[{phase_name}] Epoch {epoch+1}/{total_epochs}  loss={avg_loss:.4f}  "
          f"block={t_block:.4f}/{v_block:.4f}  "
          f"char={t_char:.4f}/{v_char:.4f}  "
          f"lr={scheduler.get_last_lr()[0]:.6f}")

    return v_block, v_char


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Detect all available GPUs
    gpu_ids = None
    if device.type == "cuda":
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            gpu_ids = list(range(n_gpus))
            print(f"Device: {device} ({n_gpus} GPUs: {[torch.cuda.get_device_name(i) for i in gpu_ids]})")
            print(f"DataParallel enabled — effective batch size: {args.batch_size}")
        else:
            print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Device: {device}")
# [RECONSTRUCTED lines 1070-1160 — train() setup, char list load, model init, ecological table head]
    nw = args.num_workers
    print(f"Batch size: {args.batch_size}, DataLoader workers: {nw}, "
          f"Render workers: {_NUM_RENDER_WORKERS}")

    # Load Unicode chars + build block tables
    print("Loading Unicode character list...")
    chars, char_to_fonts = get_unicode_chars()
    active_blocks = sorted({get_block_index(c) for c in chars
                            if get_block_index(c) >= 0})
    print(f"Unicode chars: {len(chars)}, active blocks: {len(active_blocks)}")

    block_to_chars, char_to_block_local = build_block_char_map(chars)

    model = HierarchicalCharNet(backbone=args.backbone,
                                block_to_chars=block_to_chars).to(device)

    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    # Optional: load existing checkpoint
    if args.resume or args.resume_stage_c:
        ckpt_path = args.resume or args.output
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            raw = model._orig_mod if hasattr(model, '_orig_mod') else model
            missing, unexpected = raw.load_state_dict(
                ckpt["model_state_dict"], strict=False)
            print(f"Resumed from {ckpt_path} "
                  f"(missing: {len(missing)}, unexpected: {len(unexpected)})")
        else:
            print(f"Warning: no checkpoint at {ckpt_path}, training from scratch")

    # Optionally transfer backbone weights from V2 contour model
    elif os.path.exists("model_02.pth"):
        ckpt_v2 = torch.load("model_02.pth", map_location="cpu",
                             weights_only=False)
        v2_sd = ckpt_v2.get("model_state_dict", ckpt_v2)
        raw = model._orig_mod if hasattr(model, '_orig_mod') else model
        missing, unexpected = raw.load_state_dict(v2_sd, strict=False)
        n_loaded = len(raw.state_dict()) - len(missing)
        print(f"Loaded {n_loaded} backbone params from V2 "
              f"(missing: {len(missing)}, unexpected: {len(unexpected)})")

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    best_combined = 0.0

    # ===================================================================
    # Ecological block weighting (--ecological): weight blocks by L1
    # speaker population so common scripts dominate. Numbers are rough
    # millions-of-speakers estimates from Ethnologue / Wikipedia.
    # ===================================================================
    block_weights = None
    if args.ecological:
        _BLOCK_SPEAKERS_M = {
            "Basic Latin": 1500, "Latin-1 Supplement": 800,
            "Latin Extended-A": 300, "Latin Extended-B": 200,
            "IPA Extensions": 10,
            "Spacing Modifier Letters": 50, "Combining Diacritical Marks": 200,
            "Greek and Coptic": 13, "Cyrillic": 250, "Cyrillic Supplement": 30,
            "Armenian": 7, "Hebrew": 9, "Arabic": 422,
            "Devanagari": 600, "Bengali": 230, "Gurmukhi": 30,
            "Gujarati": 55, "Oriya": 35, "Tamil": 75, "Telugu": 80,
            "Kannada": 45, "Malayalam": 38, "Sinhala": 17, "Thai": 60,
            "Lao": 30, "Myanmar": 33, "Georgian": 4, "Hangul Jamo": 80,
            "Hangul Syllables": 80, "Hangul Compatibility Jamo": 80,
            "CJK Unified Ideographs": 1500, "CJK Symbols and Punctuation": 1500,
            "Hiragana": 125, "Katakana": 125,
            "General Punctuation": 7000, "Superscripts and Subscripts": 2000,
            "Currency Symbols": 2000, "Letterlike Symbols": 1000,
            "Number Forms": 1000, "Arrows": 2000,
            "Mathematical Operators": 2000,
# [/RECONSTRUCTED — continues into recovered block weights table at line 1161]
            "Miscellaneous Mathematical Symbols-A": 1000,
            "Miscellaneous Mathematical Symbols-B": 1000,
            "Miscellaneous Symbols": 2000,
            "Miscellaneous Technical": 1000,
            "Geometric Shapes": 2000,
            "Box Drawing": 1000, "Block Elements": 1000,
            "Dingbats": 1000, "Emoticons": 3000,
        }
        from create_lists import BLOCK_NAMES
        block_pop = {}
        for bi in block_to_chars:
            if bi < 0 or bi >= len(BLOCK_NAMES):
                continue
            name = BLOCK_NAMES[bi]
            block_pop[bi] = _BLOCK_SPEAKERS_M.get(name, 1)  # 1M default
        if block_pop:
            max_pop = max(block_pop.values())
            block_weights = {bi: max(0.02, pop / max_pop)
                             for bi, pop in block_pop.items()}
            top = sorted(block_weights.items(), key=lambda x: -x[1])[:8]
            print(f"Ecological weights (top 8 by speaker population):")
            for bi, w in top:
                print(f"  {BLOCK_NAMES[bi]:40s} weight={w:.2f} "
                      f"({block_pop[bi]}M speakers)")
            n_high = sum(1 for w in block_weights.values() if w > 0.3)
            n_low = sum(1 for w in block_weights.values() if w <= 0.05)
            print(f"  {n_high} blocks >0.3 weight, {n_low} blocks at minimum")

    def _save_checkpoint(tag=""):
        """Save current model state unconditionally."""
        # Unwrap compiled model if needed
        raw = model._orig_mod if hasattr(model, '_orig_mod') else model
        torch.save({
            "model_state_dict": raw.state_dict(),
            "block_to_chars": block_to_chars,
            "num_blocks": NUM_BLOCKS,
            "input_size": args.input_size,
        }, args.output)
        print(f"  -> Saved checkpoint{' (' + tag + ')' if tag else ''} to {args.output}")

    def save_if_best(v_block, v_char):
        nonlocal best_combined
        combined = 0.3 * v_block + 0.7 * v_char
        if combined > best_combined:
            best_combined = combined
            _save_checkpoint(f"best combined={combined:.4f}")

    if args.resume_stage_c:
        print("Skipping Stages A & B (--resume-stage-c)")
    else:
        _run_stages_ab(model, args, chars, char_to_fonts, char_to_block_local,
                       block_to_chars, device, scaler, total_params,
                       save_if_best, _save_checkpoint, gpu_ids=gpu_ids,
                       block_weights=block_weights)

    # ===================================================================
    # STAGE C (optional): Joint fine-tune
    # ===================================================================
    if args.epochs_joint > 0:
        print(f"\n{'='*60}")
        print(f"STAGE C: Joint fine-tune ({args.epochs_joint} epochs)")
        print(f"{'='*60}")

        joint_spc = args.joint_spc if args.joint_spc > 0 else args.block_spc
        t0 = time.time()

        if args.sliding_window:
            # Real sliding window crops from synthetic pages — zero domain gap
            from generate_training_data import discover_fonts as _discover_fonts
            sw_fonts = _discover_fonts()
            joint_ds = SlidingWindowDataset(
                sw_fonts, num_pages=args.sw_pages,
                input_size=args.input_size,
                scales=(0.5, 1.0, 2.0),
                min_iou=0.2,
            )
        else:
            print(f"Generating joint training data (lazy, {joint_spc} spc)...")
            joint_ds = CharBalancedDataset(
                chars, char_to_fonts, char_to_block_local,
                samples_per_char=joint_spc,
                max_block_size=1000,
                input_size=args.input_size,
                block_to_chars=block_to_chars,
                context_ratio=args.context_ratio,
                block_weights=block_weights,
            )
        print(f"Joint dataset: {len(joint_ds)} samples in {time.time()-t0:.0f}s")

        char_nw = args.num_workers
        val_size = min(5000, len(joint_ds) // 10)
        train_ds, val_ds = random_split(joint_ds,
                                        [len(joint_ds) - val_size, val_size])
        train_loader = _make_loader(train_ds, args.batch_size, True, char_nw)
        val_loader = _make_loader(val_ds, args.batch_size, False, char_nw)

        # Unfreeze everything
        for param in model.parameters():
            param.requires_grad = True

        joint_criterion = HierarchicalLoss(block_weight=1.0, char_weight=1.0)
        wrapper = _wrap_model(model, joint_criterion, gpu_ids)

        # Differential LR: backbone low, heads high
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if name.startswith(("stem.", "layer1.", "layer2.", "layer3.", "layer4.")):
                backbone_params.append(param)
            else:
                head_params.append(param)

        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": args.lr_joint * 0.1},
            {"params": head_params, "lr": args.lr_joint},
        ], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs_joint)

        for epoch in range(args.epochs_joint):
# [RECONSTRUCTED tail — Stage C training loop + argparse + __main__]
            v_block, v_char = _run_epoch(
                wrapper, train_loader, val_loader, optimizer, scheduler, scaler,
                device, epoch, args.epochs_joint, "Joint")
            save_if_best(v_block, v_char)

    _save_checkpoint("final")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--output", default="model_02_char.pth")
    p.add_argument("--resume", default=None,
                   help="Resume from this checkpoint path")
    p.add_argument("--resume-stage-c", action="store_true",
                   help="Skip Stages A & B, jump straight to joint fine-tune")
    p.add_argument("--backbone", default="resnet18",
                   choices=list(_BACKBONE_CONFIGS.keys()))
    p.add_argument("--input-size", type=int, default=CHAR_INPUT_SIZE)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--compile", action="store_true",
                   help="Compile model with torch.compile")
    p.add_argument("--context-ratio", type=float, default=0.8,
                   help="Fraction of samples rendered in-context (Stages A/B)")

    # Stage A (block detector)
    p.add_argument("--block-spc", type=int, default=200,
                   help="Samples per block for Stage A")
    p.add_argument("--epochs-block", type=int, default=15)
    p.add_argument("--lr-block", type=float, default=1e-3)

    # Stage B (char heads)
    p.add_argument("--char-spc", type=int, default=50,
                   help="Samples per char for Stage B")
    p.add_argument("--epochs-char", type=int, default=10)
    p.add_argument("--lr-char", type=float, default=1e-3)

    # Stage C (joint fine-tune)
    p.add_argument("--joint-spc", type=int, default=0,
                   help="Samples per char for Stage C (0 = use block_spc)")
    p.add_argument("--epochs-joint", type=int, default=10)
    p.add_argument("--lr-joint", type=float, default=3e-5)

    # Ecological weighting (Stage A/B/C balance by speaker population)
    p.add_argument("--ecological", action="store_true",
                   help="Weight blocks by L1 speaker population")

    # Stage C alternative dataset: train on real sliding-window crops from
    # synthetic pages instead of pre-rendered character crops. Added 2026-03-23
    # to close the "model knows Latin chars (94.7%) but only 24% on sliding
    # window crops" domain gap.
    p.add_argument("--sliding-window", action="store_true",
                   help="Stage C: train on sliding window crops instead of "
                        "isolated char renders")
    p.add_argument("--sw-pages", type=int, default=100,
                   help="Number of synthetic pages to generate sliding windows from")

    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
# [/RECONSTRUCTED]
