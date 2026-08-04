#!/usr/bin/env python3
"""Diagnose the residual handwriting CER: (a) how much is pure spacing
(space-stripped CER), (b) does the cut lattice even contain the GT letter
boundaries (coverage within tolerance)."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/Code/glyph-faerie"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from PIL import Image
from beam_line_tree import (detect_bg_and_components, gap_runs)
from beam_word_tree import find_cuts
from hw_eval_lines import cer, PAGES

# reads from the latest eval (hw_priors run)
READS = {
    ("010", 0, 0): "a we d a nd r in r were On",
    ("010", 0, 1): "every in ng to",
    ("010", 1, 0): "t o t i de God and she t he",
    ("010", 1, 1): "lder newS she ma t he old",
    ("011", 0, 0): "iah books sated from the Ur to the",
    ("011", 0, 1): "elliG In m an con or of POe, the",
    ("011", 1, 0): "mie b en mi a ve",
    ("011", 1, 1): "lol the ti",
    ("011", 2, 0): "ithe an et lil and the not",
}

tol = 3
tot_b = cov_b = 0
se = sn = 0.0
for pid in ["010", "011"]:
    img = Image.open(PAGES / f"page_{pid}.png").convert("RGB")
    gt = json.loads((PAGES / f"page_{pid}.json").read_text())
    for pi, para in enumerate(gt["paragraphs"]):
        for li, line in enumerate(para["lines"][:2]):
            x1, y1, x2, y2 = line["bbox"]
            ox, oy = max(0, x1 - 10), max(0, y1 - 8)
            crop = img.crop((ox, oy, x2 + 10, y2 + 8))
            bgc, comp_labels, comps = detect_bg_and_components(crop)
            if not comps:
                continue
            xs = [c["bbox"][0] for c in comps] + [c["bbox"][2] for c in comps]
            ys = [c["bbox"][1] for c in comps] + [c["bbox"][3] for c in comps]
            L, T, R, B = min(xs), min(ys), max(xs), max(ys)
            eff = max(40, (R - L) // 12)
            cuts = set(find_cuts(comp_labels, L, R, T, B, max_cuts=eff))
            for g0, g1 in gap_runs(comp_labels, L, R, T, B):
                cuts.update((g0, g1))
            # GT letter boundaries in crop coords (letter starts + ends)
            bounds = set()
            for w in line["words"]:
                for c in w["characters"]:
                    bounds.add(c["bbox"][0] - ox)
                    bounds.add(c["bbox"][2] - ox)
            for b in bounds:
                tot_b += 1
                if any(abs(b - c) <= tol for c in cuts):
                    cov_b += 1
            # space-stripped CER
            gt_text = " ".join(w["text"] for w in line["words"])
            read = READS.get((pid, pi, li), "")
            a, g = read.replace(" ", ""), gt_text.replace(" ", "")
            e = cer(a, g)
            se += e * len(g)
            sn += len(g)
            print(f"p{pid}/{pi}.{li} stripped_cer={e:.2f} "
                  f"cut_cov={sum(1 for b in bounds if any(abs(b-c)<=tol for c in cuts))}/{len(bounds)}")

print(f"\ncut coverage (±{tol}px): {cov_b}/{tot_b} = {cov_b/max(1,tot_b):.3f}")
print(f"space-stripped weighted CER = {se/max(1,sn):.3f}  (spaced was 0.439)")
