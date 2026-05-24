#!/usr/bin/env python3
"""Stitch together the recovered train_02_char.py from session 0a90eaa8.

Combines:
- 723 verbatim recovered lines from session Read tool results
- 7 reconstructed gap-fills (marked with # [RECONSTRUCTED] ... # [/RECONSTRUCTED])
- 6 session Edit operations applied in their final form

Output: /home/remy/tiny-tessarachnid/recovered/train_02_char.py.stitched
"""
import re

PARTIAL = '/tmp/recovered_reads/home_remy_tiny-tessarachnid_train_02_char.py.rebuilt'

with open(PARTIAL) as f:
    raw_lines = f.readlines()  # 1270 lines


def slice_partial(start, end):
    """Return lines start..end inclusive (1-indexed) from the partial, as a string.

    Asserts that no MISSING markers are in the requested range.
    """
    out = []
    for i in range(start - 1, end):
        ln = raw_lines[i]
        if ln.startswith('# [MISSING LINE'):
            raise ValueError(f"line {i+1} is missing")
        out.append(ln)
    return ''.join(out)


# =============================================================================
# RECONSTRUCTED REGIONS
# =============================================================================

HEADER = '''\
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

'''
# Make HEADER end at line 59 exactly. We'll let the rest of the file start at line 60.

# Gap 2: predict() completion (line 240+) and render functions (up to line 459).
# We have:  feats = extract_features(img); block_logits = block_head(feats); block_probs = softmax(...)
# We need: argmax, per-block char prediction loop, hangul handling, return tuple.
# Then a section break, then _render_char and _render_char_in_context.
GAP_2 = '''\
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

'''

# Gap 3: BlockBalancedDataset internals (520-539, 20 lines).
# Context before: `for ch in chars:\n    if ch == ' ':`
# Context after:  `self.block_ids = torch.zeros(total_planned, dtype=torch.long)`
# Need: complete the char-filtering loop, compute active_blocks/total_planned, weights handling, allocate self.images.
GAP_3 = '''\
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
'''

# Gap 4: end of BlockBalanced + start of CharBalanced docstring.
# Context before: `self.block_ids = self.block_ids[:generated]`
# Context after:  `samples_per_char: target samples per character`
# Need: finish trim, shuffle, __len__/__getitem__, then CharBalanced class header + __init__ signature + Args docstring start.
GAP_4 = '''\
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
# Stage B Dataset: Char-balanced within each block
# ---------------------------------------------------------------------------

class CharBalancedDataset(Dataset):
    """Renders characters for a specific set of blocks with equal samples
    per character within each block.  Used for training char_heads with
    the backbone frozen.
    """

    def __init__(self, chars, char_to_fonts, char_to_block_local,
                 samples_per_char=50, max_block_size=1000,
                 input_size=CHAR_INPUT_SIZE,
                 block_to_chars=None, context_ratio=0.8,
                 block_weights=None):
        """
        Args:
# [/RECONSTRUCTED]
'''

# Gap 5: HierarchicalLoss.forward() — lines 759-785, 27 lines.
# Context before: `self.char_weight = char_weight`
# Context after:  `# DataParallel training wrapper`
# Need: the forward(block_logits, char_logits_list, block_targets, char_targets) method.
GAP_5 = '''\
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
'''

# Gap 6: Stage B middle (890-979). Context before is end of Stage A optimizer/scheduler;
# context after is the run-epoch helper "Run one training epoch + validation".
# The recovered region 880-889 actually shows Stage A code bleeding into char_heads setup
# (`requires_grad = True` for char_heads). So gap 6 spans the end of Stage A training loop +
# Stage B dataset setup + Stage B optimizer + Stage B training loop. Reconstruct generously.
GAP_6 = '''\
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
    print(f"\\n{'='*60}")
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
'''

# Gap 7: train() body interior (1070-1160). Context before: `print(f"Device: {device}")`.
# Context after: ecological block weights table (`Miscellaneous Mathematical Symbols-A: 1000`).
# Need: NUM_THREADS / nw printing, load Unicode chars, build block maps, instantiate model,
# load/init checkpoint, scaler, AMP, start of ecological weights table.
GAP_7 = '''\
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
'''

# After line 1270 (end of Stage C optimizer/scheduler setup), the file is truncated
# (the original was longer). Reconstruct the Stage C training loop, train() tail,
# argparse setup, and __main__.
TAIL = '''\
# [RECONSTRUCTED tail — Stage C training loop + argparse + __main__]
            v_block, v_char = _run_epoch(
                wrapper, train_loader, val_loader, optimizer, scheduler, scaler,
                device, epoch, args.epochs_joint, "Joint")
            save_if_best(v_block, v_char)

    _save_checkpoint("final")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\\n")[0])
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
'''

# =============================================================================
# SESSION EDITS — applied as final form
# =============================================================================

# Edit 1 (BlockBalancedDataset, lines ~556-570): swap sibling_chars for has_siblings +
# block_idx, and update _parallel_render call signature.
def apply_edit_1(text):
    old = (
        "        for block_idx in active_blocks:\n"
        "            chars_in_block = block_chars[block_idx]\n"
        "            sibling_chars = _block_char_lists.get(block_idx, [])\n"
        "            for _ in range(samples_per_block * 2):\n"
        "                ch, local_idx = random.choice(chars_in_block)\n"
        "                fonts_for_char = char_to_fonts.get(ch, [])\n"
        "                if not fonts_for_char:\n"
        "                    continue\n"
        "                font_path, face_idx = random.choice(fonts_for_char)\n"
        "                font_size = random.randint(14, 60)\n"
        "                use_ctx = bool(sibling_chars) and random.random() < context_ratio\n"
        "                tasks.append((ch, font_path, face_idx, font_size, input_size,\n"
        "                              use_ctx, sibling_chars if use_ctx else None))\n"
        "                task_meta.append((block_idx, local_idx))\n"
        "\n"
        "        results = _parallel_render(tasks, \"BlockBalanced\")"
    )
    new = (
        "        for block_idx in active_blocks:\n"
        "            chars_in_block = block_chars[block_idx]\n"
        "            has_siblings = bool(_block_char_lists.get(block_idx))\n"
        "            for _ in range(samples_per_block * 2):\n"
        "                ch, local_idx = random.choice(chars_in_block)\n"
        "                fonts_for_char = char_to_fonts.get(ch, [])\n"
        "                if not fonts_for_char:\n"
        "                    continue\n"
        "                font_path, face_idx = random.choice(fonts_for_char)\n"
        "                font_size = random.randint(14, 60)\n"
        "                use_ctx = has_siblings and random.random() < context_ratio\n"
        "                tasks.append((ch, font_path, face_idx, font_size, input_size,\n"
        "                              use_ctx, block_idx if use_ctx else None))\n"
        "                task_meta.append((block_idx, local_idx))\n"
        "\n"
        "        results = _parallel_render(tasks, _block_char_lists, \"BlockBalanced\")"
    )
    assert old in text, "Edit 1 OLD not found"
    return text.replace(old, new)


# Edit 2: replace the eager CharBalancedDataset with the lazy version.
# We need to swap the entire class body. Our recovered version has lines 627-742
# of the eager class (gap 4 reconstructs lines 615-628 which include the class
# header + __init__ signature + docstring start).
LAZY_CHAR_BALANCED = '''\
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
'''


# Edit 5: insert SlidingWindowDataset before the HierarchicalLoss / "# Loss" header.
SLIDING_WINDOW = '''\
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
        from detect_characters import _extract_windows, DetectorConfig, \\
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

'''


# Edit 6: Stage C — wrap dataset selection in `if args.sliding_window` branch.
def apply_edit_6(text):
    # Find the Stage C "Lazy dataset" block and wrap it. Use the post-edit-4 form.
    old = (
        "        # Lazy dataset — renders fresh images every epoch, can't overfit\n"
        "        joint_spc = args.joint_spc if args.joint_spc > 0 else args.block_spc\n"
        "        print(f\"Generating joint training data (lazy, {joint_spc} spc)...\")\n"
        "        t0 = time.time()\n"
        "        joint_ds = CharBalancedDataset(\n"
        "            chars, char_to_fonts, char_to_block_local,\n"
        "            samples_per_char=joint_spc,\n"
        "            max_block_size=1000,\n"
        "            input_size=args.input_size,\n"
        "            block_to_chars=block_to_chars,\n"
        "            context_ratio=args.context_ratio,\n"
        "            block_weights=block_weights,\n"
        "        )\n"
        "        print(f\"Joint dataset: {len(joint_ds)} samples in {time.time()-t0:.0f}s\")\n"
        "\n"
        "        # More workers for lazy rendering\n"
        "        char_nw = max(args.num_workers, 8)\n"
    )
    new = (
        "        joint_spc = args.joint_spc if args.joint_spc > 0 else args.block_spc\n"
        "        t0 = time.time()\n"
        "\n"
        "        if args.sliding_window:\n"
        "            # Real sliding window crops from synthetic pages — zero domain gap\n"
        "            from generate_training_data import discover_fonts as _discover_fonts\n"
        "            sw_fonts = _discover_fonts()\n"
        "            joint_ds = SlidingWindowDataset(\n"
        "                sw_fonts, num_pages=args.sw_pages,\n"
        "                input_size=args.input_size,\n"
        "                scales=(0.5, 1.0, 2.0),\n"
        "                min_iou=0.2,\n"
        "            )\n"
        "        else:\n"
        "            print(f\"Generating joint training data (lazy, {joint_spc} spc)...\")\n"
        "            joint_ds = CharBalancedDataset(\n"
        "                chars, char_to_fonts, char_to_block_local,\n"
        "                samples_per_char=joint_spc,\n"
        "                max_block_size=1000,\n"
        "                input_size=args.input_size,\n"
        "                block_to_chars=block_to_chars,\n"
        "                context_ratio=args.context_ratio,\n"
        "                block_weights=block_weights,\n"
        "            )\n"
        "        print(f\"Joint dataset: {len(joint_ds)} samples in {time.time()-t0:.0f}s\")\n"
        "\n"
        "        char_nw = args.num_workers\n"
    )
    assert old in text, "Edit 6 OLD not found"
    return text.replace(old, new)


# =============================================================================
# ASSEMBLY
# =============================================================================

parts = []
parts.append(HEADER)

# Recovered lines 60-239 (HierarchicalCharNet up through predict() softmax)
parts.append(slice_partial(60, 239))

# Gap 2 fill
parts.append(GAP_2)

# Recovered lines 460-519 (_NUM_RENDER_WORKERS through start of BlockBalanced __init__).
# Session edits 1, 2 call _parallel_render with a `_block_char_lists` dict so worker
# tasks can look up siblings by block_idx (instead of pickling sibling lists per task).
# The matching edits to _parallel_render / _render_task / _pool_init weren't captured,
# so we patch the recovered region in-place to support the new signature.
chunk = slice_partial(460, 519)
chunk = chunk.replace(
    "def _pool_init():\n"
    "    \"\"\"Re-seed random in forked worker processes.\"\"\"\n"
    "    random.seed(os.getpid() ^ int(time.time() * 1000))",
    "# [RECONSTRUCTED — pool worker holds block-sibling map as a global so tasks\n"
    "# only need to pass block_idx, not full sibling lists]\n"
    "_WORKER_BLOCK_CHAR_LISTS = {}\n\n"
    "def _pool_init(block_char_lists=None):\n"
    "    \"\"\"Re-seed random in forked worker processes + stash block sibling map.\"\"\"\n"
    "    global _WORKER_BLOCK_CHAR_LISTS\n"
    "    if block_char_lists is not None:\n"
    "        _WORKER_BLOCK_CHAR_LISTS = block_char_lists\n"
    "    random.seed(os.getpid() ^ int(time.time() * 1000))\n"
    "# [/RECONSTRUCTED]"
)
chunk = chunk.replace(
    "def _render_task(args):\n"
    "    \"\"\"Render one character image in a worker process.\"\"\"\n"
    "    ch, font_path, face_idx, font_size, input_size, use_context, sibling_chars = args\n"
    "    try:\n"
    "        font = ImageFont.truetype(font_path, font_size, index=face_idx)\n"
    "    except Exception:\n"
    "        return None\n"
    "    if use_context and sibling_chars:\n"
    "        return _render_char_in_context(ch, font, input_size, sibling_chars)\n"
    "    else:\n"
    "        return _render_char(ch, font, input_size)",
    "def _render_task(args):\n"
    "    \"\"\"Render one character image in a worker process.\n\n"
    "    The 7th tuple element is now `block_idx` (or None); siblings are\n"
    "    fetched from the worker's _WORKER_BLOCK_CHAR_LISTS map.\n"
    "    \"\"\"\n"
    "    ch, font_path, face_idx, font_size, input_size, use_context, block_idx = args\n"
    "    try:\n"
    "        font = ImageFont.truetype(font_path, font_size, index=face_idx)\n"
    "    except Exception:\n"
    "        return None\n"
    "    if use_context and block_idx is not None:\n"
    "        siblings = _WORKER_BLOCK_CHAR_LISTS.get(block_idx, [])\n"
    "        if siblings:\n"
    "            return _render_char_in_context(ch, font, input_size, siblings)\n"
    "    return _render_char(ch, font, input_size)"
)
chunk = chunk.replace(
    "def _parallel_render(tasks, desc=\"\"):\n"
    "    \"\"\"Render character images in parallel using multiprocessing Pool.\"\"\"\n"
    "    n = len(tasks)\n"
    "    if n == 0:\n"
    "        return []\n"
    "    print(f\"  [{desc}] Rendering {n} images with {_NUM_RENDER_WORKERS} workers...\")\n"
    "    t0 = time.time()\n"
    "    with mp.Pool(_NUM_RENDER_WORKERS, initializer=_pool_init) as pool:\n"
    "        results = pool.map(_render_task, tasks, chunksize=64)",
    "def _parallel_render(tasks, block_char_lists=None, desc=\"\"):\n"
    "    \"\"\"Render character images in parallel using multiprocessing Pool.\n\n"
    "    block_char_lists: dict[block_idx -> list[char]] for in-context rendering.\n"
    "    Passed once to each worker via pool initializer to avoid per-task pickling.\n"
    "    \"\"\"\n"
    "    n = len(tasks)\n"
    "    if n == 0:\n"
    "        return []\n"
    "    print(f\"  [{desc}] Rendering {n} images with {_NUM_RENDER_WORKERS} workers...\")\n"
    "    t0 = time.time()\n"
    "    with mp.Pool(_NUM_RENDER_WORKERS,\n"
    "                 initializer=_pool_init,\n"
    "                 initargs=(block_char_lists or {},)) as pool:\n"
    "        results = pool.map(_render_task, tasks, chunksize=64)"
)
parts.append(chunk)

# Gap 3 fill
parts.append(GAP_3)

# Recovered lines 540-589 (rest of BlockBalanced init)
parts.append(slice_partial(540, 589))

# Gap 4 fill (also opens CharBalanced class header + docstring "Args:")
parts.append(GAP_4)

# Recovered lines 630-742 (eager CharBalanced internals) — REPLACED by Edit 2.
# Instead, we splice the lazy CharBalancedDataset class body.
# But GAP_4 already wrote the eager class header + `Args:` line; that's incompatible
# with the lazy class. Cleaner: replace GAP_4's class-opening + the eager body with
# the lazy class, then resume from the line AFTER the eager CharBalanced (line 743).
# Strip the class def at the end of GAP_4:
# Find the last occurrence of "class CharBalancedDataset" in parts[-1] and chop.
parts[-1] = parts[-1].split('# ---------------------------------------------------------------------------\n# Stage B Dataset: Char-balanced within each block')[0]
# Re-add the section header for Stage B + lazy class
parts.append('# ---------------------------------------------------------------------------\n')
parts.append('# Stage B Dataset: Char-balanced within each block (lazy on-demand rendering)\n')
parts.append('# ---------------------------------------------------------------------------\n\n')
parts.append(LAZY_CHAR_BALANCED)

# Edit 5: insert SlidingWindowDataset before HierarchicalLoss
parts.append('\n\n')
parts.append(SLIDING_WINDOW)

# Recovered lines 745-758 (HierarchicalLoss __init__ — "# Loss" header through self.char_weight = char_weight)
parts.append(slice_partial(745, 758))

# Gap 5 fill (HierarchicalLoss.forward)
parts.append(GAP_5)

# Recovered lines 786-885 (DataParallel wrapper, _make_loader, _wrap_model, _run_stages_ab Stage A).
# Lines 886-889 of the partial are orphan fragments (Read tool captured stray
# char_heads-setup lines without their enclosing `for` block) — skip them; GAP_6 has
# the proper version.
parts.append(slice_partial(786, 885))

# Gap 6 fill (Stage A train loop + Stage B setup + _run_epoch header)
parts.append(GAP_6)

# Recovered lines 980-1069 (_run_epoch body + train() function up to "Device:" print)
parts.append(slice_partial(980, 1069))

# Gap 7 fill (train() setup through ecological table head)
parts.append(GAP_7)

# Recovered lines 1161-1270 (ecological table tail + save_if_best + Stage C optimizer/scheduler)
parts.append(slice_partial(1161, 1270))

# Tail (Stage C epoch loop + argparse + __main__)
parts.append(TAIL)

# Assemble
text = ''.join(parts)

# Apply edits 1 and 6 (text-substitution edits on the BlockBalanced and Stage C regions)
text = apply_edit_1(text)
text = apply_edit_6(text)

# Final sanity check: no leftover MISSING markers
if '# [MISSING LINE' in text:
    raise RuntimeError("Leftover MISSING markers in output")

OUT = '/home/remy/tiny-tessarachnid/recovered/train_02_char.py.stitched'
with open(OUT, 'w') as f:
    f.write(text)
print(f"Wrote {OUT} ({len(text)} bytes, {text.count(chr(10))} lines)")

# Syntax check
import py_compile
try:
    py_compile.compile(OUT, doraise=True)
    print("py_compile: OK")
except py_compile.PyCompileError as e:
    print(f"py_compile FAILED:\n{e}")
