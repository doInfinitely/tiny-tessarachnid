#!/usr/bin/env python3
"""Decompose the real-IAM residual CER: classifier vs segmentation.

For the same val lines hw_eval_iam reads freely (CER 0.613 with v4):
1. oracle GT-box classification: top-1 accuracy on true letter crops
2. oracle-segmentation read: assemble top-1 chars at GT letter cuts
   (no lattice, no LM) -> CER against GT text
3. joinedness: fraction of adjacent letters whose boxes overlap
   horizontally (cursive proxy), correlated with free-decode CER

Letter boxes come from letter_bboxes_v2 (word-relative coords mapped
into the infill after-patch frame, as in generate_iam_letter_crops).
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/Code/glyph-faerie"))

import numpy as np
import torch
from PIL import Image

from glyph_faerie.detection.detector import load_detector
from beam_line_tree import ascii_topk
from beam_word import letterbox
from hw_eval_lines import cer
from hw_eval_iam import IAM, load_lines

HERE = Path(__file__).resolve().parent
PAL = Path.home() / "Code/palimpsest/Code/palimpsest"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_02_char_hw_v4.pth")
    ap.add_argument("--free-log", default="hw_iam_v4.log")
    ap.add_argument("--max-lines", type=int, default=30)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    ck = torch.load(HERE / args.model, map_location=device,
                    weights_only=False)
    model = load_detector(ck, device, None).model

    lbb = {}
    for l in open(PAL / "runs/letter_bboxes_v2.jsonl"):
        r = json.loads(l)
        lbb[r["word_id"]] = r
    # word_id -> infill val record (for bbox + patch)
    inf = {}
    for l in open(IAM / "infill_val.jsonl"):
        r = json.loads(l)
        inf.setdefault(r["word_id"], r)

    free = {}
    txt = open(HERE / args.free_log).read()
    for m in re.finditer(r"^(iam_\S+) cer=([\d.]+)", txt, re.M):
        free[m.group(1)] = float(m.group(2))

    tot = cor = 0
    rows = []
    for key, img_path, words in load_lines(args.max_lines):
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        line_pref = key.replace("iam_", "")
        # word records of this line, in x order, with letter boxes
        recs = []
        for l in open(IAM / "infill_val.jsonl"):
            r = json.loads(l)
            if r["record_id"].rsplit("_", 1)[0] == key:
                recs.append(r)
        recs.sort(key=lambda r: r["target_bbox_parent_norm_cxcywh"][0])
        crops, labels, word_break = [], [], []
        n_pairs = n_overlap = 0
        gt_words = []
        for r in recs:
            w = lbb.get(r["word_id"])
            gt_words.append(r["char_text"])
            if w is None:
                continue
            span = w["word_x2"] - w["word_x1"]
            if span < 4:
                continue
            cx, cy, bw, bh = r["target_bbox_parent_norm_cxcywh"]
            wx1, wx2 = (cx - bw / 2) * W, (cx + bw / 2) * W
            wy1, wy2 = (cy - bh / 2) * H, (cy + bh / 2) * H
            prev_x2 = None
            first = True
            for L in w["letters"]:
                rx1 = wx1 + (L["x1"] - w["word_x1"]) / span * (wx2 - wx1)
                rx2 = wx1 + (L["x2"] - w["word_x1"]) / span * (wx2 - wx1)
                if prev_x2 is not None:
                    n_pairs += 1
                    if rx1 < prev_x2 - 1:
                        n_overlap += 1
                prev_x2 = rx2
                x1, x2 = max(0, int(rx1) - 1), min(W, int(rx2) + 1)
                y1, y2 = max(0, int(wy1) - 2), min(H, int(wy2) + 2)
                if x2 - x1 < 3 or y2 - y1 < 3:
                    continue
                crops.append(img.crop((x1, y1, x2, y2)))
                labels.append(L["char"])
                word_break.append(first)
                first = False
        if not crops:
            continue
        preds = []
        B = 128
        for i in range(0, len(crops), B):
            ts = torch.stack([
                torch.from_numpy(np.array(letterbox(c))).permute(2, 0, 1)
                .float() / 255.0 for c in crops[i:i + B]]).to(device)
            with torch.no_grad():
                tk = ascii_topk(model, model.extract_features(ts), 1)
            preds.extend(t[0][0] if t else "?" for t in tk)
        line_cor = sum(1 for p, g in zip(preds, labels)
                       if p.lower() == g.lower())
        tot += len(labels)
        cor += line_cor
        oracle_read = "".join(
            (" " if brk and i > 0 else "") + p
            for i, (p, brk) in enumerate(zip(preds, word_break)))
        gt_text = " ".join(gt_words)
        oc = cer(oracle_read, gt_text)
        join = n_overlap / max(1, n_pairs)
        rows.append((key, free.get(key, float("nan")), oc,
                     line_cor / max(1, len(labels)), join))
        print(f"{key} free={rows[-1][1]:.2f} oracleseg={oc:.2f} "
              f"boxacc={rows[-1][3]:.2f} joined={join:.2f}")
        print(f"   oracle read={oracle_read!r}")
        print(f"   GT         ={gt_text!r}", flush=True)

    print(f"\nGT-box top-1 (case-pooled): {cor}/{tot} = {cor/max(1,tot):.3f}")
    fs = [r[1] for r in rows if not np.isnan(r[1])]
    os_ = [r[2] for r in rows]
    js = [r[4] for r in rows]
    print(f"mean free CER={np.mean(fs):.3f}  mean oracle-seg CER="
          f"{np.mean(os_):.3f}")
    if len(rows) > 3:
        print(f"corr(joinedness, free CER)={np.corrcoef(js, [r[1] for r in rows])[0,1]:.2f}")
        print(f"corr(joinedness, box acc)={np.corrcoef(js, [r[3] for r in rows])[0,1]:.2f}")


if __name__ == "__main__":
    main()
