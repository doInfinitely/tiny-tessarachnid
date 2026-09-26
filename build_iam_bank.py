#!/usr/bin/env python3
"""Real-ink template bank for IAM template verification.

Pulls high-confidence letters from the EM-aligned boxes
(letter_bboxes_v3c.jsonl), takes tight ink crops from the after-patches,
normalizes with beam_line_tree's _glyph_bitmap (48px, ink-centered,
unit-norm), and pools up to N per char sampled across distinct writers.
Output: {char: [np.float32 48x48]} — force-able as a template bank in
the reader's pass-2 verify.
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/Code/glyph-faerie"))

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from beam_line_tree import _glyph_bitmap
from hw_eval_iam import IAM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path.home() /
                    "Code/palimpsest/Code/palimpsest/runs/"
                    "letter_bboxes_v3c.jsonl"))
    ap.add_argument("--out", default="iam_char_bank.pt")
    ap.add_argument("--min-conf", type=float, default=0.7)
    ap.add_argument("--per-char", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    # gather candidates grouped by (char, writer) so the bank samples
    # across writers instead of overfitting one hand
    cands = defaultdict(lambda: defaultdict(list))
    for line in open(args.src):
        r = json.loads(line)
        for L in r["letters"]:
            if L.get("conf", 0.0) >= args.min_conf:
                cands[L["char"]][r["style_id"]].append(
                    (r["after_patch_ref"], L))

    bank = {}
    cache = {}
    for ch, by_writer in cands.items():
        writers = list(by_writer)
        rng.shuffle(writers)
        tpls = []
        wi = 0
        while len(tpls) < args.per_char and writers:
            w = writers[wi % len(writers)]
            if not by_writer[w]:
                writers.remove(w)
                continue
            ref, L = by_writer[w].pop(rng.randrange(len(by_writer[w])))
            wi += 1
            path = IAM / ref
            if path not in cache:
                if len(cache) > 4:
                    cache.clear()
                g = np.asarray(Image.open(path).convert("L"),
                               dtype=np.float32)
                lo, bg = g.min(), np.percentile(g, 90)
                cache[path] = np.clip((g - lo) / max(1.0, bg - lo), 0, 1)
            garr = cache[path]
            crop = garr[L["y1"]:L["y2"], L["x1"]:L["x2"]]
            if crop.shape[0] < 4 or crop.shape[1] < 3:
                continue
            ink = crop < 0.6
            if ink.sum() < 12:
                continue
            ys, xs = ink.nonzero()
            tight = crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            pil = Image.fromarray(
                (tight * 255).astype(np.uint8)).convert("RGB")
            tpls.append(_glyph_bitmap(pil))
        if tpls:
            bank[ch] = tpls
    n = sum(len(v) for v in bank.values())
    torch.save(bank, HERE / args.out)
    print(f"saved {args.out}: {len(bank)} chars, {n} templates")


if __name__ == "__main__":
    main()
