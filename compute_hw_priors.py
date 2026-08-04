#!/usr/bin/env python3
"""Empirical glyph-width and space-width priors for scribe handwriting.

Measures, from ScribePage GT JSONs, per-char width relative to the
line's ink height (same normalization read_line applies at decode time)
and inter-word gap relative to line height. Writes hw_priors.json for
beam_line_tree's --hw-priors / args.hw_priors override.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

PAGES = Path.home() / "Code/palimpsest/Code/palimpsest/eval_output/scribe_pages"

widths = defaultdict(list)
spaces = []
for jf in sorted(PAGES.glob("page_*.json")):
    gt = json.loads(jf.read_text())
    for para in gt["paragraphs"]:
        for line in para["lines"]:
            lh = line["bbox"][3] - line["bbox"][1]
            if lh < 8:
                continue
            for w in line["words"]:
                for c in w["characters"]:
                    cw = c["bbox"][2] - c["bbox"][0]
                    if cw > 0:
                        widths[c["char"]].append(cw / lh)
            wb = sorted(w["bbox"] for w in line["words"])
            for a, b in zip(wb, wb[1:]):
                gap = b[0] - a[2]
                if 0 < gap < lh * 1.5:
                    spaces.append(gap / lh)

if not widths:
    sys.exit("no GT pages found")

all_w = [v for vs in widths.values() for v in vs]
glob_med = median(all_w)
glyph_rel = {ch: median(vs) if len(vs) >= 3 else glob_med
             for ch, vs in widths.items()}
# fill chars we never saw with the global median so priors_for covers them
import string
for ch in string.ascii_letters + string.digits + ".,'!?-":
    glyph_rel.setdefault(ch, glob_med)

out = {"glyph_rel": glyph_rel,
       "space_rel": median(spaces) if spaces else 0.35,
       "n_letters": len(all_w), "n_spaces": len(spaces)}
Path("hw_priors.json").write_text(json.dumps(out, indent=1))
print(f"letters={len(all_w)} spaces={len(spaces)} "
      f"global_med_w={glob_med:.3f} space_rel={out['space_rel']:.3f}")
