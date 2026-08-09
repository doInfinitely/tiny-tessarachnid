#!/usr/bin/env python3
"""EM step for IAM letter boxes: DP forced alignment scored by the
current handwriting classifier (hw_v4).

For each word (known text) in an infill after-patch: candidate cuts
every 2px (+ zero-ink run midpoints), score every (cut, cut) span
against the word's target chars (case-pooled), DP for the cut sequence
maximizing sum log P(char_i | crop_i) with a soft width prior.

--gate: align only the eval val lines' words and report oracle-seg CER
with the new boxes (baseline with v2 boxes: 0.350). --emit writes a
letter_bboxes-style JSONL (patch coords) for crop regeneration.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/Code/glyph-faerie"))

import numpy as np
import torch
from PIL import Image

from glyph_faerie.detection.detector import load_detector
from beam_word import letterbox
from hw_eval_lines import cer
from hw_eval_iam import IAM, load_lines

HERE = Path(__file__).resolve().parent
PRIORS = json.load(open(HERE / "hw_priors.json"))["glyph_rel"]


def case_indices(chars, c2l):
    """char -> list of local head indices for both cases."""
    out = {}
    for ch in chars:
        idxs = []
        for v in {ch, ch.swapcase()}:
            if v in c2l:
                idxs.append(c2l[v])
        out[ch] = idxs
    return out


@torch.no_grad()
def align_word(img_gray, box, text, model, head, c2l, device, grid=2):
    """img_gray: np float [H,W] 0..1 (1=white), box: (x1,y1,x2,y2) word
    box in image coords. Returns list of (char, x1, y1, x2, y2) or None."""
    wx1, wy1, wx2, wy2 = [int(round(v)) for v in box]
    sub = img_gray[wy1:wy2, wx1:wx2]
    H, W = sub.shape
    N = len(text)
    if W < 3 * N or H < 6 or N == 0:
        return None
    lh = H

    cuts = list(range(0, W + 1, grid))
    if cuts[-1] != W:
        cuts.append(W)
    ink_cols = (sub < 0.6).sum(axis=0)
    i = 0
    while i < W:
        if ink_cols[i] == 0:
            j = i
            while j < W and ink_cols[j] == 0:
                j += 1
            cuts.append((i + j) // 2)
            i = j
        else:
            i += 1
    cuts = sorted(set(cuts))
    C = len(cuts)

    # candidate spans per char with width bounds from priors
    spans = []          # (ci, cj) index pairs
    span_of = {}
    for a in range(C):
        for b in range(a + 1, C):
            w = cuts[b] - cuts[a]
            if 2 <= w <= lh * 1.4:
                span_of[(a, b)] = len(spans)
                spans.append((a, b))
    if not spans:
        return None

    crops = []
    for a, b in spans:
        seg = sub[:, cuts[a]:cuts[b]]
        pil = Image.fromarray((seg * 255).astype(np.uint8)).convert("RGB")
        crops.append(torch.from_numpy(
            np.array(letterbox(pil))).permute(2, 0, 1).float() / 255.0)
    logp = np.full((len(spans), N), -1e9, dtype=np.float32)
    ci = case_indices(set(text), c2l)
    B = 128
    for s in range(0, len(crops), B):
        ts = torch.stack(crops[s:s + B]).to(device)
        pr = torch.softmax(head(model.extract_features(ts)), dim=1)
        pr = pr.cpu().numpy()
        for r in range(pr.shape[0]):
            for k, ch in enumerate(text):
                idxs = ci[ch]
                if idxs:
                    logp[s + r, k] = math.log(max(pr[r, idxs].max(), 1e-9))

    # width prior: soft gaussian around per-char prior width
    def wprior(ch, w):
        mu = PRIORS.get(ch, 0.55) * lh
        return -0.5 * ((w - mu) / (0.6 * mu)) ** 2

    NEG = -1e15
    dp = np.full((N + 1, C), NEG)
    bk = np.zeros((N + 1, C), dtype=np.int32)
    dp[0, 0] = 0.0
    for k in range(N):
        for b in range(1, C):
            best, barg = NEG, -1
            for a in range(b):
                if dp[k, a] <= NEG / 2 or (a, b) not in span_of:
                    continue
                w = cuts[b] - cuts[a]
                v = (dp[k, a] + logp[span_of[(a, b)], k]
                     + 0.3 * wprior(text[k], w))
                if v > best:
                    best, barg = v, a
            dp[k + 1, b] = best
            bk[k + 1, b] = barg
    end = int(np.argmax(dp[N]))
    if dp[N, end] <= NEG / 2:
        return None
    # force the alignment to end near the word's right edge
    tail = [b for b in range(C) if cuts[b] >= W - 3 * grid
            and dp[N, b] > NEG / 2]
    if tail:
        end = max(tail, key=lambda b: dp[N, b])

    bounds = [end]
    for k in range(N, 0, -1):
        bounds.append(bk[k, bounds[-1]])
    bounds = bounds[::-1]

    # sum-DP forward/backward over the SAME lattice -> per-letter
    # posterior marginal of the chosen span (hard-EM confidence)
    def lse(vs):
        if not vs:
            return NEG
        m = max(vs)
        return m + math.log(sum(math.exp(v - m) for v in vs))

    fwd = np.full((N + 1, C), NEG)
    fwd[0, 0] = 0.0
    for k in range(N):
        for b in range(1, C):
            vs = []
            for a in range(b):
                if fwd[k, a] <= NEG / 2 or (a, b) not in span_of:
                    continue
                vs.append(fwd[k, a] + logp[span_of[(a, b)], k]
                          + 0.3 * wprior(text[k], cuts[b] - cuts[a]))
            fwd[k + 1, b] = lse(vs)
    tail_set = set(tail) if tail else {end}
    logZ = lse([fwd[N, b] for b in tail_set if fwd[N, b] > NEG / 2])
    bwd = np.full((N + 1, C), NEG)
    for b in tail_set:
        bwd[N, b] = 0.0
    for k in range(N - 1, -1, -1):
        for a in range(C):
            vs = []
            for b in range(a + 1, C):
                if bwd[k + 1, b] <= NEG / 2 or (a, b) not in span_of:
                    continue
                vs.append(bwd[k + 1, b] + logp[span_of[(a, b)], k]
                          + 0.3 * wprior(text[k], cuts[b] - cuts[a]))
            bwd[k, a] = lse(vs)

    letters = []
    for k in range(N):
        a, b = bounds[k], bounds[k + 1]
        x1, x2 = cuts[a], cuts[b]
        seg = sub[:, x1:x2] < 0.6
        ys = seg.any(axis=1).nonzero()[0]
        y1, y2 = (int(ys.min()), int(ys.max()) + 1) if len(ys) else (0, H)
        if (a, b) in span_of and logZ > NEG / 2:
            g = (fwd[k, a] + logp[span_of[(a, b)], k]
                 + 0.3 * wprior(text[k], x2 - x1) + bwd[k + 1, b] - logZ)
            conf = float(min(1.0, math.exp(min(0.0, g))))
        else:
            conf = 0.0
        letters.append({"char": text[k], "x1": wx1 + x1, "x2": wx1 + x2,
                        "y1": wy1 + y1, "y2": wy1 + y2, "conf": conf})
    return letters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_02_char_hw_v4.pth")
    ap.add_argument("--max-lines", type=int, default=30)
    ap.add_argument("--emit", default=None,
                    help="write aligned letters JSONL for these lines")
    ap.add_argument("--all-train", action="store_true",
                    help="align every infill_train word with letter "
                         "annotations; requires --emit")
    ap.add_argument("--grid", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    ck = torch.load(HERE / args.model, map_location=device,
                    weights_only=False)
    model = load_detector(ck, device, None).model
    chars = ck["block_to_chars"][0]
    c2l = {c: i for i, c in enumerate(chars)}
    head = model.char_heads["0"]

    emit = open(args.emit, "w") if args.emit else None

    if args.all_train:
        PAL = Path.home() / "Code/palimpsest/Code/palimpsest"
        lbb = set()
        for l in open(PAL / "runs/letter_bboxes_v2.jsonl"):
            lbb.add(json.loads(l)["word_id"])
        n = ok = 0
        for l in open(IAM / "infill_train.jsonl"):
            r = json.loads(l)
            if r["word_id"] not in lbb:
                continue
            path = IAM / r["after_patch_ref"]
            if not path.exists():
                continue
            img = Image.open(path).convert("L")
            W, H = img.size
            garr = np.asarray(img, dtype=np.float32)
            lo, bg = garr.min(), np.percentile(garr, 90)
            garr = np.clip((garr - lo) / max(1.0, bg - lo), 0, 1)
            cx, cy, bw, bh = r["target_bbox_parent_norm_cxcywh"]
            box = ((cx - bw / 2) * W, (cy - bh / 2) * H,
                   (cx + bw / 2) * W, (cy + bh / 2) * H)
            letters = align_word(garr, box, r["char_text"], model, head,
                                 c2l, device, grid=args.grid)
            n += 1
            if letters is not None:
                ok += 1
                emit.write(json.dumps(
                    {"word_id": r["word_id"],
                     "after_patch_ref": r["after_patch_ref"],
                     "style_id": r["style_id"],
                     "letters": letters}) + "\n")
            if n % 1000 == 0:
                emit.flush()
                print(f"{n} words, {ok} aligned", flush=True)
        emit.close()
        print(f"done: {ok}/{n} aligned")
        return

    tot = cor = 0
    ocers = []
    for key, img_path, words in load_lines(args.max_lines):
        img = Image.open(img_path).convert("L")
        W, H = img.size
        garr = np.asarray(img, dtype=np.float32)
        # per-line contrast stretch (as the reader effectively sees)
        lo, bg = garr.min(), np.percentile(garr, 90)
        garr = np.clip((garr - lo) / max(1.0, bg - lo), 0, 1)

        # word records with pixel boxes, x-sorted (load_lines gave norm)
        recs = []
        for cx, cy, w, h, t in words:
            recs.append(((cx - w / 2) * W, (cy - h / 2) * H,
                         (cx + w / 2) * W, (cy + h / 2) * H, t))
        oracle_words = []
        gt_words = []
        for x1, y1, x2, y2, t in recs:
            gt_words.append(t)
            letters = align_word(garr, (x1, y1, x2, y2), t, model, head,
                                 c2l, device)
            if letters is None:
                oracle_words.append("")
                continue
            if emit:
                emit.write(json.dumps({"line": key, "text": t,
                                       "letters": letters}) + "\n")
            # classify aligned crops top-1 (case-pooled) for oracle read
            crops = []
            for L in letters:
                seg = garr[max(0, L["y1"] - 2):L["y2"] + 2,
                           max(0, L["x1"] - 1):L["x2"] + 1]
                if seg.shape[0] < 3 or seg.shape[1] < 3:
                    seg = np.ones((8, 8), dtype=np.float32)
                crops.append(torch.from_numpy(np.array(letterbox(
                    Image.fromarray((seg * 255).astype(np.uint8))
                    .convert("RGB")))).permute(2, 0, 1).float() / 255.0)
            with torch.no_grad():
                pr = torch.softmax(head(model.extract_features(
                    torch.stack(crops).to(device))), dim=1).cpu().numpy()
            pred = []
            for r, L in zip(pr, letters):
                k = int(r.argmax())
                pred.append(chars[k] if k < len(chars) else "?")
                tot += 1
                if pred[-1].lower() == L["char"].lower():
                    cor += 1
            oracle_words.append("".join(pred))
        oc = cer(" ".join(oracle_words), " ".join(gt_words))
        ocers.append(oc)
        print(f"{key} oracleseg={oc:.2f}")
        print(f"   read={' '.join(oracle_words)!r}")
        print(f"   GT  ={' '.join(gt_words)!r}", flush=True)
    if emit:
        emit.close()
    print(f"\naligned-box top-1 (case-pooled): {cor}/{tot} = "
          f"{cor/max(1,tot):.3f}   [v2 boxes: 0.619]")
    print(f"mean oracle-seg CER = {np.mean(ocers):.3f}   [v2 boxes: 0.350]")


if __name__ == "__main__":
    main()
