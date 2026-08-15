#!/usr/bin/env python3
"""Learned letter-boundary proposer for handwriting lines.

Fully-convolutional net: input a height-64 grayscale strip (any width),
output per-column boundary probability. Trained on EM-aligned IAM words
(letter_bboxes_v3c.jsonl): targets are Gaussian bumps at letter x1/x2
boundaries; columns near low-confidence boundaries are down-weighted
(the soft-EM lesson — attenuate, don't delete).

At decode time the reader takes local maxima as candidate cuts,
replacing the blind uniform grid.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
IAM = Path.home() / "Code/palimpsest/Code/palimpsest/data/iam_full"
STRIP_H = 64
CROP_W = 256          # training window width (random crop from strip)


class BoundaryNet(nn.Module):
    """2D convs, y fully pooled, per-column logit. FCN in x."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, (5, 5), padding=(2, 2)), nn.ReLU(),
            nn.MaxPool2d((2, 1)),                       # 32 x W
            nn.Conv2d(16, 32, (5, 5), padding=(2, 2)), nn.ReLU(),
            nn.MaxPool2d((2, 1)),                       # 16 x W
            nn.Conv2d(32, 64, (3, 7), padding=(1, 3)), nn.ReLU(),
            nn.MaxPool2d((2, 1)),                       # 8 x W
            nn.Conv2d(64, 64, (3, 7), padding=(1, 3)), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Conv1d(64 * 8, 128, 1), nn.ReLU(),
            nn.Conv1d(128, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 1, 1),
        )

    def forward(self, x):
        # x: [B,1,64,W] -> [B,W] logits
        f = self.net(x)                         # [B,64,8,W]
        f = f.flatten(1, 2)                     # [B,512,W]
        return self.head(f).squeeze(1)


class StripDataset(Dataset):
    """Word strips (h=64) with boundary targets, random x-crop to CROP_W."""

    def __init__(self, samples, train=True):
        self.samples = samples
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        arr, bounds, confs = self.samples[i]
        W = arr.shape[1]
        tgt = np.zeros(W, dtype=np.float32)
        wgt = np.ones(W, dtype=np.float32)
        for b, g in zip(bounds, confs):
            for dx in range(-3, 4):
                c = b + dx
                if 0 <= c < W:
                    tgt[c] = max(tgt[c], math_exp(-0.5 * (dx / 1.5) ** 2))
                    if g < 0.5:
                        wgt[c] = 0.3
        if self.train and W > CROP_W:
            s = random.randrange(W - CROP_W)
        else:
            s = 0
        a = np.zeros((STRIP_H, CROP_W), dtype=np.float32)
        t = np.zeros(CROP_W, dtype=np.float32)
        w = np.ones(CROP_W, dtype=np.float32)
        e = min(W - s, CROP_W)
        a[:, :e] = arr[:, s:s + e]
        t[:e] = tgt[s:s + e]
        w[:e] = wgt[s:s + e]
        w[e:] = 0.0
        return (torch.from_numpy(a).unsqueeze(0), torch.from_numpy(t),
                torch.from_numpy(w))


def math_exp(v):
    import math
    return math.exp(v)


def build_samples(src, limit=None):
    samples = []
    cache = {}
    n = 0
    for line in open(src):
        r = json.loads(line)
        path = IAM / r["after_patch_ref"]
        if not path.exists():
            continue
        if path not in cache:
            img = Image.open(path).convert("L")
            g = np.asarray(img, dtype=np.float32)
            lo, bg = g.min(), np.percentile(g, 90)
            cache.clear()          # keep memory bounded: one patch cached
            cache[path] = np.clip((g - lo) / max(1.0, bg - lo), 0, 1)
        garr = cache[path]
        ys1 = min(L["y1"] for L in r["letters"])
        ys2 = max(L["y2"] for L in r["letters"])
        if ys2 - ys1 < 8:
            continue
        xs1 = min(L["x1"] for L in r["letters"])
        xs2 = max(L["x2"] for L in r["letters"])
        pad = int(0.25 * (xs2 - xs1)) + 4
        x0 = max(0, xs1 - pad)
        x1 = min(garr.shape[1], xs2 + pad)
        band = garr[max(0, ys1 - 2):ys2 + 2, x0:x1]
        sc = STRIP_H / band.shape[0]
        Wn = max(8, int(band.shape[1] * sc))
        strip = np.asarray(Image.fromarray(
            (band * 255).astype(np.uint8)).resize(
            (Wn, STRIP_H), Image.LANCZOS), dtype=np.float32) / 255.0
        bounds, confs = [], []
        for L in r["letters"]:
            g = L.get("conf", 1.0)
            for x in (L["x1"], L["x2"]):
                bounds.append(int((x - x0) * sc))
                confs.append(g)
        samples.append((1.0 - strip, bounds, confs))   # ink=1
        n += 1
        if limit and n >= limit:
            break
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path.home() /
                    "Code/palimpsest/Code/palimpsest/runs/"
                    "letter_bboxes_v3c.jsonl"))
    ap.add_argument("--out", default="model_07_boundary.pth")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    print("building samples...", flush=True)
    samples = build_samples(args.src)
    print(f"{len(samples)} word strips", flush=True)
    random.shuffle(samples)
    n_val = max(500, len(samples) // 50)
    va_s, tr_s = samples[:n_val], samples[n_val:]

    tr = DataLoader(StripDataset(tr_s, True), batch_size=args.batch,
                    shuffle=True, num_workers=4, drop_last=True)
    va = DataLoader(StripDataset(va_s, False), batch_size=args.batch,
                    shuffle=False, num_workers=2)

    net = BoundaryNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr,
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(tr))

    best = 1e9
    for ep in range(args.epochs):
        net.train()
        tl = tn = 0
        for bi, (x, t, w) in enumerate(tr):
            x, t, w = x.to(device), t.to(device), w.to(device)
            logits = net(x)
            per = F.binary_cross_entropy_with_logits(
                logits, t, reduction="none")
            # boundaries are sparse: upweight positive columns
            pos_w = 1.0 + 9.0 * t
            loss = (per * w * pos_w).sum() / (w * pos_w).sum().clamp(1e-6)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item() * x.size(0)
            tn += x.size(0)
            if bi % 100 == 0:
                print(f"ep{ep} it{bi}/{len(tr)} loss={loss.item():.4f}",
                      flush=True)
        net.eval()
        vl = vn = 0
        with torch.no_grad():
            for x, t, w in va:
                x, t, w = x.to(device), t.to(device), w.to(device)
                per = F.binary_cross_entropy_with_logits(
                    net(x), t, reduction="none")
                pos_w = 1.0 + 9.0 * t
                vl += ((per * w * pos_w).sum()
                       / (w * pos_w).sum().clamp(1e-6)).item() * x.size(0)
                vn += x.size(0)
        v = vl / max(1, vn)
        print(f"== ep{ep}: train={tl/max(1,tn):.4f} val={v:.4f}",
              flush=True)
        if v < best:
            best = v
            torch.save({"state_dict": net.state_dict()}, HERE / args.out)
            print(f"   saved {args.out}", flush=True)
    print(f"done, best val={best:.4f}")


if __name__ == "__main__":
    main()
