#!/usr/bin/env python3
"""Oracle-segmentation diagnostic: classify GT letter boxes from scribe
pages with the fine-tuned classifier. High accuracy here means remaining
line CER is segmentation/search error, not recognition error."""
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/Code/glyph-faerie"))

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from glyph_faerie.detection.detector import load_detector
from beam_line_tree import ascii_topk
from beam_word import letterbox

PAGES = Path.home() / "Code/palimpsest/Code/palimpsest/eval_output/scribe_pages"

device = torch.device("cuda:1")
ck = torch.load(HERE / "model_02_char_hw_v1.pth", map_location=device,
                weights_only=False)
model = load_detector(ck, device, None).model

tot = cor1 = cor3 = 0
conf = Counter()
for pid in ["010", "011"]:
    img = Image.open(PAGES / f"page_{pid}.png").convert("RGB")
    gt = json.loads((PAGES / f"page_{pid}.json").read_text())
    crops, labels = [], []
    for para in gt["paragraphs"]:
        for line in para["lines"]:
            for w in line["words"]:
                for c in w["characters"]:
                    x1, y1, x2, y2 = c["bbox"]
                    if x2 - x1 < 3 or y2 - y1 < 3:
                        continue
                    crops.append(img.crop((x1 - 1, y1 - 1, x2 + 1, y2 + 1)))
                    labels.append(c["char"])
    for i in range(0, len(crops), 128):
        ts = torch.stack([
            torch.from_numpy(np.array(letterbox(c))).permute(2, 0, 1).float()
            / 255.0 for c in crops[i:i + 128]]).to(device)
        with torch.no_grad():
            tk = ascii_topk(model, model.extract_features(ts), 3)
        for lab, top in zip(labels[i:i + 128], tk):
            tot += 1
            preds = [ch for ch, _ in top]
            if preds and preds[0] == lab:
                cor1 += 1
            if lab in preds:
                cor3 += 1
            if preds and preds[0] != lab:
                conf[(lab, preds[0])] += 1

print(f"GT-box top1={cor1/tot:.3f} top3={cor3/tot:.3f}  (n={tot})")
print("top confusions:", conf.most_common(15))
