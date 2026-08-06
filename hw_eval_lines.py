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
    ap.add_argument("--consensus", action="store_true",
                    help="paragraph-level writer consensus: pool "
                         "per-line writer votes, re-read dissenting "
                         "lines with the consensus bank forced")
    ap.add_argument("--generic-bank", action="store_true",
                    help="pool ALL writers' templates into one generic "
                         "bank and force it everywhere (no writer-ID)")
    ap.add_argument("--pages", nargs="+", default=["010", "011"])
    ap.add_argument("--lines-per-para", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.consensus:
        args.writer_id = True

    device = torch.device(args.device)
    ck = torch.load(HERE / args.model, map_location=device,
                    weights_only=False)
    model = load_detector(ck, device, None).model
    lm = IncLM(device)
    banks = None
    generic = None
    if args.writer_templates:
        banks = torch.load(args.writer_templates, weights_only=False)
        if args.generic_bank:
            generic = {}
            for bank in banks.values():
                for ch, tpls in bank.items():
                    generic.setdefault(ch, []).extend(tpls[:4])
            for ch, tpls in generic.items():
                if len(tpls) > 24:
                    step = len(tpls) / 24
                    generic[ch] = [tpls[int(i * step)] for i in range(24)]
            print("generic bank:", len(generic), "chars,",
                  sum(len(v) for v in generic.values()), "templates")
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

    def alpha_frac(text):
        letters = sum(1 for c in text if c.isalpha())
        return letters / max(1, len(text.replace(" ", "")))

    import re
    tot_e = tot_n = 0
    n_paras = n_paras_correct = n_rereads = 0
    for pid in args.pages:
        img = Image.open(PAGES / f"page_{pid}.png").convert("RGB")
        gt = json.loads((PAGES / f"page_{pid}.json").read_text())
        for pi, para in enumerate(gt["paragraphs"]):
            m = re.match(r"scribe_style(\d+)_font(\d+)", para["font"])
            gt_key = (int(m.group(1)), int(m.group(2)))
            force = None
            if generic is not None:
                force = generic
            elif banks is not None and not args.writer_id:
                force = banks.get(gt_key)
            lines = para["lines"][:args.lines_per_para]
            crops, gts = [], []
            for line in lines:
                x1, y1, x2, y2 = line["bbox"]
                crops.append(img.crop((max(0, x1 - 10), max(0, y1 - 8),
                                       x2 + 10, y2 + 8)))
                gts.append(" ".join(w["text"] for w in line["words"]))

            reads = [read_line(c, model, lm, device, rargs, verbose=False,
                               force_font=force)
                     for c in crops]

            if args.consensus and banks is not None:
                # pool per-line writer votes, weighted by ID score and
                # decode quality (a garbage line must not outvote a
                # sibling that reads like words)
                votes = {}
                for text, key, sc in reads:
                    if isinstance(key, tuple):
                        votes[key] = (votes.get(key, 0.0)
                                      + max(sc, 0.05) * alpha_frac(text) ** 2)
                if votes:
                    ckey = max(votes.items(), key=lambda kv: kv[1])[0]
                    n_paras += 1
                    n_paras_correct += (ckey == gt_key)
                    for li, (text, key, sc) in enumerate(reads):
                        if key != ckey:
                            n_rereads += 1
                            t2, _, s2 = read_line(
                                crops[li], model, lm, device, rargs,
                                verbose=False, force_font=banks[ckey])
                            if t2:
                                reads[li] = (t2, ckey, s2)

            for li, (text, _, _) in enumerate(reads):
                e = cer(text, gts[li])
                tot_e += e * len(gts[li])
                tot_n += len(gts[li])
                print(f"p{pid}/{pi}.{li} cer={e:.2f}  read={text!r}")
                print(f"          GT ={gts[li]!r}", flush=True)
    if n_paras:
        print(f"\nconsensus writer-ID: {n_paras_correct}/{n_paras} "
              f"paragraphs correct, {n_rereads} lines re-read")
    print(f"\nweighted CER = {tot_e / max(1, tot_n):.3f}")


if __name__ == "__main__":
    main()
