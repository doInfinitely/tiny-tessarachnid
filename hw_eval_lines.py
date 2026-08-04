#!/usr/bin/env python3
"""Evaluate the line reader on scribe handwriting pages (line-level CER).

Crops GT line bboxes from ScribePage renders and runs read_line on each.
Used for the handwriting baseline (--model model_02_char.eco100.pth)
and for measuring classifier retrains (--model model_02_char_hw_v1.pth
--no-two-pass, since print-font templates are meaningless on
handwriting).
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/Code/glyph-faerie"))

import torch
from PIL import Image

from glyph_faerie.detection.detector import load_detector
from beam_line_tree import IncLM, read_line, default_read_args

HERE = Path(__file__).resolve().parent
PAGES = Path.home() / "Code/palimpsest/Code/palimpsest/eval_output/scribe_pages"


def cer(a, b):
    if not b:
        return 1.0 if a else 0.0
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[n] / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_02_char.eco100.pth")
    ap.add_argument("--no-two-pass", action="store_true")
    ap.add_argument("--hw-priors", default=None)
    ap.add_argument("--cut-grid", type=int, default=None)
    ap.add_argument("--cursor-beam", type=int, default=32)
    ap.add_argument("--max-expansions", type=int, default=400000)
    ap.add_argument("--writer-templates", default=None,
                    help="writer bank .pt; forces each paragraph's GT "
                         "writer templates (two-pass upper bound)")
    ap.add_argument("--min-w-ratio", type=float, default=0.08)
    ap.add_argument("--min-space-frac", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--space-conf", type=float, default=0.9)
    ap.add_argument("--cap-slack", type=float, default=4.0)
    ap.add_argument("--writer-id", action="store_true",
                    help="identify the writer per line via "
                         "detect_writer_glyphs instead of forcing GT")
    ap.add_argument("--pages", nargs="+", default=["010", "011"])
    ap.add_argument("--lines-per-para", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    ck = torch.load(HERE / args.model, map_location=device,
                    weights_only=False)
    model = load_detector(ck, device, None).model
    lm = IncLM(device)
    banks = None
    if args.writer_templates:
        banks = torch.load(args.writer_templates, weights_only=False)
    rargs = default_read_args(
        font_pool="auto",
        two_pass=(banks is not None) or not args.no_two_pass,
        hw_priors=args.hw_priors, cut_grid=args.cut_grid,
        cursor_beam=args.cursor_beam, max_expansions=args.max_expansions,
        min_w_ratio=args.min_w_ratio, min_space_frac=args.min_space_frac,
        lam=args.lam, space_conf=args.space_conf,
        cap_slack=args.cap_slack,
        writer_banks=banks if (banks is not None and args.writer_id)
        else None)

    tot_e = tot_n = 0
    for pid in args.pages:
        img = Image.open(PAGES / f"page_{pid}.png").convert("RGB")
        gt = json.loads((PAGES / f"page_{pid}.json").read_text())
        for pi, para in enumerate(gt["paragraphs"]):
            force = None
            if banks is not None and not args.writer_id:
                import re
                m = re.match(r"scribe_style(\d+)_font(\d+)", para["font"])
                force = banks.get((int(m.group(1)), int(m.group(2))))
            for li, line in enumerate(para["lines"][:args.lines_per_para]):
                x1, y1, x2, y2 = line["bbox"]
                crop = img.crop((max(0, x1 - 10), max(0, y1 - 8),
                                 x2 + 10, y2 + 8))
                gt_text = " ".join(
                    w["text"] for w in line["words"])
                text, det_font, _ = read_line(
                    crop, model, lm, device, rargs, verbose=False,
                    force_font=force)
                e = cer(text, gt_text)
                tot_e += e * len(gt_text)
                tot_n += len(gt_text)
                print(f"p{pid}/{pi}.{li} cer={e:.2f}  read={text!r}")
                print(f"          GT ={gt_text!r}", flush=True)
    print(f"\nweighted CER = {tot_e / max(1, tot_n):.3f}")


if __name__ == "__main__":
    main()
