#!/usr/bin/env python3
"""Fine-tune the eco100 char classifier on handwriting letter crops.

Starts from model_02_char.eco100.pth and trains the backbone + ASCII
block head (block 0) on scribe-generated letter crops (see palimpsest
generate_letter_crops.py). Crops are saved already letterboxed to
128x128 grayscale-on-white; the loader re-composites them onto random
near-white paper with random dark ink tint to match ScribePage output,
plus light affine jitter.

Saves a checkpoint in the same format load_detector expects, so
beam_line_tree.py can use it via --model.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/Code/glyph-faerie"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from glyph_faerie.detection.detector import load_detector

HERE = Path(__file__).resolve().parent


class HwCropDataset(Dataset):
    def __init__(self, crops_dir, records, char_to_local, train=True):
        self.dir = Path(crops_dir)
        self.records = records
        self.c2l = char_to_local
        self.train = train

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        img = Image.open(self.dir / rec["file"]).convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        ink = 1.0 - arr                                # [128,128] 0..1

        rng = random
        if self.train:
            # light geometric jitter
            if rng.random() < 0.7:
                ang = rng.uniform(-3, 3)
                tx = rng.uniform(-0.03, 0.03)
                sc = rng.uniform(0.92, 1.08)
                pil = Image.fromarray((ink * 255).astype(np.uint8))
                pil = pil.rotate(ang, resample=Image.BILINEAR,
                                 translate=(tx * 128, 0), fillcolor=0)
                if sc != 1.0:
                    ns = max(8, int(128 * sc))
                    pil = pil.resize((ns, ns), Image.BILINEAR)
                    canvas = Image.new("L", (128, 128), 0)
                    off = (128 - ns) // 2
                    canvas.paste(pil, (off, off))
                    pil = canvas
                ink = np.asarray(pil, dtype=np.float32) / 255.0
            bg = rng.uniform(235, 255)
            tint = (rng.uniform(0, 40), rng.uniform(0, 40), rng.uniform(0, 60))
        else:
            bg, tint = 248.0, (20.0, 20.0, 30.0)

        out = np.empty((3, 128, 128), dtype=np.float32)
        for c in range(3):
            out[c] = (bg * (1 - ink) + tint[c] * ink) / 255.0
        return torch.from_numpy(out), self.c2l[rec["char"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default=str(Path.home() /
                    "Code/palimpsest/Code/palimpsest/data/letter_crops_v1"))
    ap.add_argument("--init", default="model_02_char.eco100.pth")
    ap.add_argument("--out", default="model_02_char_hw_v1.pth")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    ck = torch.load(HERE / args.init, map_location="cpu", weights_only=False)
    model = load_detector(ck, device, None).model
    model.train()

    chars = ck["block_to_chars"][0]
    c2l = {c: i for i, c in enumerate(chars)}

    crops = Path(args.crops)
    records = [json.loads(l) for l in open(crops / "labels.jsonl")]
    n_all = len(records)
    records = [r for r in records if r["char"] in c2l]
    print(f"{len(records)}/{n_all} crops usable "
          f"({n_all - len(records)} chars outside ASCII block)")
    random.shuffle(records)
    n_val = max(1000, len(records) // 50)
    val_recs, tr_recs = records[:n_val], records[n_val:]

    tr = DataLoader(HwCropDataset(crops / "crops", tr_recs, c2l, train=True),
                    batch_size=args.batch, shuffle=True, num_workers=8,
                    pin_memory=True, drop_last=True)
    va = DataLoader(HwCropDataset(crops / "crops", val_recs, c2l, train=False),
                    batch_size=args.batch, shuffle=False, num_workers=4)

    head = model.char_heads["0"]
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(tr))
    scaler = torch.amp.GradScaler("cuda")

    best = 0.0
    for ep in range(args.epochs):
        model.train()
        tot = cor = 0
        for bi, (x, y) in enumerate(tr):
            x, y = x.to(device, non_blocking=True), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                feats = model.extract_features(x)
                logits = head(feats)
                loss = F.cross_entropy(logits, y, label_smoothing=0.05)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += y.numel()
            cor += (logits.argmax(1) == y).sum().item()
            if bi % 50 == 0:
                print(f"ep{ep} it{bi}/{len(tr)} loss={loss.item():.3f} "
                      f"acc={cor/max(1,tot):.3f}", flush=True)

        model.eval()
        vt = vc = 0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(device), y.to(device)
                logits = head(model.extract_features(x))
                vt += y.numel()
                vc += (logits.argmax(1) == y).sum().item()
        vacc = vc / max(1, vt)
        print(f"== ep{ep}: train_acc={cor/max(1,tot):.4f} "
              f"val_acc={vacc:.4f}", flush=True)
        if vacc > best:
            best = vacc
            torch.save({
                "model_state_dict": model.state_dict(),
                "block_to_chars": ck["block_to_chars"],
                "input_size": ck.get("input_size", 128),
                "hw_finetune": {"epoch": ep, "val_acc": vacc,
                                "crops": str(crops)},
            }, HERE / args.out)
            print(f"   saved {args.out} (val_acc={vacc:.4f})", flush=True)
    print(f"done, best val_acc={best:.4f}")


if __name__ == "__main__":
    main()
