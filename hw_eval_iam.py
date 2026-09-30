#!/usr/bin/env python3
"""Evaluate the handwriting line reader on REAL IAM lines.

Reconstructs line-level GT from palimpsest's infill_val.jsonl: records
sharing a record_id line prefix are words of one line in a common parent
frame (target_bbox_parent_norm_cxcywh). The after-patch of the line's
last word shows the finished line. Reads the union crop with read_line
and scores CER against the space-joined word texts.
"""
import argparse
import collections
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/Code/glyph-faerie"))

import torch
from PIL import Image

from glyph_faerie.detection.detector import load_detector
from beam_line_tree import (IncLM, read_line, default_read_args,
                            harvest_self_bank)
from hw_eval_lines import cer

HERE = Path(__file__).resolve().parent
IAM = Path.home() / "Code/palimpsest/Code/palimpsest/data/iam_full"


def load_lines(max_lines, min_words=3):
    by_line = collections.defaultdict(list)
    with open(IAM / "infill_val.jsonl") as fh:
        for raw in fh:
            r = json.loads(raw)
            key = r["record_id"].rsplit("_", 1)[0]
            idx = int(r["record_id"].rsplit("_", 1)[1])
            by_line[key].append((idx, r))
    lines = []
    for key in sorted(by_line):
        recs = sorted(by_line[key])
        if len(recs) < min_words:
            continue
        last = recs[-1][1]
        img_path = IAM / last["after_patch_ref"]
        if not img_path.exists():
            continue
        words = []
        for _, r in recs:
            cx, cy, w, h = r["target_bbox_parent_norm_cxcywh"]
            words.append((cx, cy, w, h, r["char_text"]))
        words.sort(key=lambda t: t[0])
        lines.append((key, img_path, words))
        if len(lines) >= max_lines:
            break
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_02_char_hw_v2.pth")
    ap.add_argument("--hw-priors", default="hw_priors.json")
    ap.add_argument("--cut-grid", type=int, default=3)
    ap.add_argument("--cut-model", default=None,
                    help="BoundaryNet checkpoint; replaces/augments cuts")
    ap.add_argument("--no-cut-grid", action="store_true")
    ap.add_argument("--divider-model", default=None,
                    help="ColCharNet ckpt for divider-descent cuts")
    ap.add_argument("--self-templates", action="store_true",
                    help="two-pass with per-DOCUMENT self-harvested "
                         "template banks (pass-1 confident letters in "
                         "dictionary words)")
    ap.add_argument("--char-bank", default=None,
                    help="{char: [48px bitmaps]} .pt forced as the "
                         "pass-2 template bank on every line")
    ap.add_argument("--generic-bank", default=None,
                    help=".pt writer bank file to pool into one generic "
                         "bank forced on every line")
    ap.add_argument("--max-lines", type=int, default=30)
    ap.add_argument("--pad", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    ck = torch.load(HERE / args.model, map_location=device,
                    weights_only=False)
    model = load_detector(ck, device, None).model
    lm = IncLM(device)

    force = None
    if args.char_bank:
        force = torch.load(HERE / args.char_bank, weights_only=False)
    elif args.generic_bank:
        banks = torch.load(args.generic_bank, weights_only=False)
        force = {}
        for bank in banks.values():
            for ch, tpls in bank.items():
                force.setdefault(ch, []).extend(tpls[:4])
        for ch, tpls in force.items():
            if len(tpls) > 24:
                step = len(tpls) / 24
                force[ch] = [tpls[int(i * step)] for i in range(24)]

    cut_net = None
    if args.cut_model:
        from train_07_boundary import BoundaryNet
        bn = BoundaryNet().to(device)
        bn.load_state_dict(torch.load(
            HERE / args.cut_model, map_location=device,
            weights_only=False)["state_dict"])
        bn.eval()
        cut_net = (bn, device)
    divider_net = None
    if args.divider_model:
        from divider_descent import load_colchar
        divider_net = (load_colchar(HERE / args.divider_model, device),
                       device)
    rargs = default_read_args(
        font_pool="auto",
        two_pass=(force is not None) and not args.self_templates,
        hw_priors=args.hw_priors,
        cut_grid=None if args.no_cut_grid else args.cut_grid,
        cut_net=cut_net, divider_net=divider_net)
    rargs2 = default_read_args(
        font_pool="auto", two_pass=True, template_soft=True,
        template_soft_floor=0.5,
        hw_priors=args.hw_priors,
        cut_grid=None if args.no_cut_grid else args.cut_grid,
        cut_net=cut_net, divider_net=divider_net)
    priors = json.load(open(HERE / args.hw_priors)) \
        if args.hw_priors else None

    import numpy as np

    def line_crop(img_path, words):
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        x1 = min((cx - w / 2) * W for cx, cy, w, h, t in words)
        x2 = max((cx + w / 2) * W for cx, cy, w, h, t in words)
        y1 = min((cy - h / 2) * H for cx, cy, w, h, t in words)
        y2 = max((cy + h / 2) * H for cx, cy, w, h, t in words)
        crop = img.crop((max(0, int(x1) - args.pad),
                         max(0, int(y1) - args.pad),
                         min(W, int(x2) + args.pad),
                         min(H, int(y2) + args.pad)))
        # strip the solid black line-boundary bars at the patch edges
        arr = np.array(crop.convert("L"))
        ink_frac = (arr < 100).mean(axis=0)
        cw = arr.shape[1]
        lo = 0
        while lo < cw and ink_frac[lo] > 0.5:
            lo += 1
        hi = cw
        while hi > lo and ink_frac[hi - 1] > 0.5:
            hi -= 1
        if lo > 0 or hi < cw:
            crop = crop.crop((lo + 2 if lo else 0, 0,
                              hi - 2 if hi < cw else cw, crop.size[1]))
        return crop

    lines = [(key, line_crop(img_path, words),
              " ".join(t for *_, t in words))
             for key, img_path, words in load_lines(args.max_lines)]

    tot_e = tot_n = 0
    if args.self_templates:
        # group by document; harvest a bank from each doc's pass-1 reads
        import collections as _c
        docs = _c.defaultdict(list)
        for key, crop, gt_text in lines:
            docs[key.rsplit("-", 1)[0]].append((key, crop, gt_text))
        for doc, dlines in docs.items():
            p1 = []
            for key, crop, gt_text in dlines:
                text, _, _, steps, (T, B) = read_line(
                    crop, model, lm, device, rargs, verbose=False,
                    return_steps=True)
                p1.append((key, crop, gt_text, text, steps, T, B))
            bank = harvest_self_bank(
                [(crop, T, B, steps)
                 for _, crop, _, _, steps, T, B in p1],
                hw_priors=priors, min_conf=0.75, min_per_char=3)
            nt = sum(len(v) for v in bank.values())
            print(f"[{doc}] self-bank: {len(bank)} chars, {nt} templates",
                  flush=True)
            for key, crop, gt_text, text1, steps, T, B in p1:
                text = text1
                if bank:
                    text2, _, _ = read_line(crop, model, lm, device,
                                            rargs2, verbose=False,
                                            force_font=bank)
                    # per-line arbitration: templates propose, the LM
                    # disposes — accept pass-2 only when the LM prefers
                    # it (slack biases toward the pass-1 incumbent)
                    if text2 and text2 != text1:
                        lp1, lp2 = lm.logprob_batch([text1, text2])
                        if (lp2 / max(1, len(text2))
                                >= lp1 / max(1, len(text1)) - 0.05):
                            text = text2
                    elif text2:
                        text = text2
                e = cer(text, gt_text)
                tot_e += e * len(gt_text)
                tot_n += len(gt_text)
                print(f"{key} cer={e:.2f}  read={text!r}")
                print(f"{'':>{len(key)}} GT  ={gt_text!r}", flush=True)
    else:
        for key, crop, gt_text in lines:
            text, _, _ = read_line(crop, model, lm, device, rargs,
                                   verbose=False, force_font=force)
            e = cer(text, gt_text)
            tot_e += e * len(gt_text)
            tot_n += len(gt_text)
            print(f"{key} cer={e:.2f}  read={text!r}")
            print(f"{'':>{len(key)}} GT  ={gt_text!r}", flush=True)
    print(f"\nweighted CER = {tot_e / max(1, tot_n):.3f} "
          f"({len(lines)} lines)")


if __name__ == "__main__":
    main()
