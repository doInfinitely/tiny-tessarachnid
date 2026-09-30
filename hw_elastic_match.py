#!/usr/bin/env python3
"""Elastic exemplar matching experiment (deformable template energy).

For each GT-box letter on the IAM val lines: take the classifier's
top-K chars, fetch real-ink exemplars of each, and gradient-descend a
mesh deformation field per exemplar that warps its ink onto the target.
Ink is represented as Gaussian point splats (the divider trick), so the
data term has gradients at any distance; sigma anneals coarse->fine.

Energy = pixel term (L2 between splat renderings, mass-normalized)
       + turbulence term (squared second differences of the mesh —
         thin-plate: every affine map is laminar/free, local
         convergence/divergence is expensive).

Prediction = argmin energy over candidate exemplars. Compared against
the classifier's top-1 on the same crops (0.619 baseline).
"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/Code/glyph-faerie"))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from glyph_faerie.detection.detector import load_detector
from beam_line_tree import ascii_topk
from beam_word import letterbox
from hw_eval_iam import IAM, load_lines

GRID = 48            # working frame (pixels)
INK_H = 30           # ink height after normalization
MAX_PTS = 192
MESH = 5             # control mesh is MESH x MESH


def ink_points(gray, thresh=0.6, max_pts=MAX_PTS):
    """gray float [h,w], 1=white. Returns [N,2] xy points + [N] weights,
    centroid-centered and scaled so ink height = INK_H."""
    ink = 1.0 - gray
    ys, xs = (ink > (1 - thresh)).nonzero()
    if len(ys) < 6:
        return None
    w = ink[ys, xs]
    h_ink = ys.max() - ys.min() + 1
    sc = INK_H / max(1, h_ink)
    px = xs.astype(np.float32) * sc
    py = ys.astype(np.float32) * sc
    cx, cy = (px * w).sum() / w.sum(), (py * w).sum() / w.sum()
    pts = np.stack([px - cx + GRID / 2, py - cy + GRID / 2], axis=1)
    if len(pts) > max_pts:
        idx = np.argsort(-w)[:max_pts]
        pts, w = pts[idx], w[idx]
    w = w / w.sum()          # unit mass: L2 compares SHAPE, not ink mass
    return (torch.from_numpy(pts.astype(np.float32)),
            torch.from_numpy(w.astype(np.float32)))


def splat(pts, w, sigma, grid_xy):
    """pts [B,N,2], w [B,N] -> [B,G] rendering on grid_xy [G,2]."""
    d2 = ((pts.unsqueeze(2) - grid_xy.unsqueeze(0).unsqueeze(0)) ** 2
          ).sum(-1)                              # [B,N,G]
    return (w.unsqueeze(-1) * torch.exp(-0.5 * d2 / sigma ** 2)).sum(1)


def mesh_displace(mesh, pts):
    """mesh [B,MESH,MESH,2] displacements; pts [B,N,2] in [0,GRID] ->
    per-point displacement via bilinear interp (grid_sample)."""
    B, N, _ = pts.shape
    # grid_sample wants normalized coords in [-1,1], input [B,2,M,M]
    g = (pts / GRID) * 2 - 1
    field = mesh.permute(0, 3, 1, 2)             # [B,2,M,M]
    samp = F.grid_sample(field, g.unsqueeze(1), align_corners=True,
                         mode="bilinear")        # [B,2,1,N]
    return samp[:, :, 0, :].permute(0, 2, 1)     # [B,N,2]


def turbulence(mesh, w_affine=0.15):
    """Laminar-flow energy. First differences (div/curl/shear of the
    field): only a pure SHIFT is free — scale/rotation/shear cost a
    little, so aspect and size cues survive (full-affine-free erased
    the m/n and e/l distinctions). Second differences (non-affine
    bending) cost full price."""
    dx = mesh[:, :, 1:] - mesh[:, :, :-1]
    dy = mesh[:, 1:] - mesh[:, :-1]
    first = dx.pow(2).sum((1, 2, 3)) + dy.pow(2).sum((1, 2, 3))
    dxx = mesh[:, :, 2:] - 2 * mesh[:, :, 1:-1] + mesh[:, :, :-2]
    dyy = mesh[:, 2:] - 2 * mesh[:, 1:-1] + mesh[:, :-2]
    dxy = (mesh[:, 1:, 1:] - mesh[:, 1:, :-1]
           - mesh[:, :-1, 1:] + mesh[:, :-1, :-1])
    second = (dxx.pow(2).sum((1, 2, 3)) + dyy.pow(2).sum((1, 2, 3))
              + 2 * dxy.pow(2).sum((1, 2, 3)))
    return w_affine * first + second


def elastic_energies(ex_pts, ex_w, tgt_pts, tgt_w, device,
                     iters=120, w_turb=0.02, lr=0.35):
    """Batched field optimization: B exemplars vs one target.
    Returns final energies [B]."""
    B = ex_pts.shape[0]
    ys, xs = torch.meshgrid(torch.arange(GRID), torch.arange(GRID),
                            indexing="ij")
    grid_xy = torch.stack([xs, ys], -1).reshape(-1, 2).float().to(device)
    mesh = torch.zeros(B, MESH, MESH, 2, device=device,
                       requires_grad=True)
    opt = torch.optim.Adam([mesh], lr=lr)
    sigmas = [6.0] * 30 + [4.0] * 30 + [3.0] * 30 + [2.0] * 30
    tgt_render = None
    cur_sigma = None
    for it in range(iters):
        sigma = sigmas[min(it, len(sigmas) - 1)]
        if sigma != cur_sigma:
            with torch.no_grad():
                tgt_render = splat(tgt_pts.unsqueeze(0),
                                   tgt_w.unsqueeze(0), sigma, grid_xy)
                tgt_norm = (tgt_render ** 2).sum().clamp(min=1e-9)
            cur_sigma = sigma
        warped = ex_pts + mesh_displace(mesh, ex_pts)
        render = splat(warped, ex_w, sigma, grid_xy)
        pix = ((render - tgt_render) ** 2).sum(1) / tgt_norm
        loss = (pix + w_turb * turbulence(mesh)).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        warped = ex_pts + mesh_displace(mesh, ex_pts)
        render = splat(warped, ex_w, 2.0, grid_xy)
        pix = ((render - tgt_render) ** 2).sum(1) / tgt_norm
        return (pix + w_turb * turbulence(mesh)).cpu().numpy()


def load_exemplars(src, min_conf=0.8, per_char=8, seed=0):
    rng = random.Random(seed)
    by_cw = defaultdict(lambda: defaultdict(list))
    for line in open(src):
        r = json.loads(line)
        for L in r["letters"]:
            if L.get("conf", 0) >= min_conf:
                by_cw[L["char"]][r["style_id"]].append(
                    (r["after_patch_ref"], L))
    pool = {}
    cache = {}
    for ch, byw in by_cw.items():
        writers = list(byw)
        rng.shuffle(writers)
        out = []
        wi = 0
        while len(out) < per_char and writers:
            w = writers[wi % len(writers)]
            if not byw[w]:
                writers.remove(w)
                continue
            ref, L = byw[w].pop(rng.randrange(len(byw[w])))
            wi += 1
            path = IAM / ref
            if path not in cache:
                if len(cache) > 4:
                    cache.clear()
                g = np.asarray(Image.open(path).convert("L"),
                               dtype=np.float32)
                lo, bg = g.min(), np.percentile(g, 90)
                cache[path] = np.clip((g - lo) / max(1.0, bg - lo), 0, 1)
            crop = cache[path][L["y1"]:L["y2"], L["x1"]:L["x2"]]
            if crop.shape[0] < 4 or crop.shape[1] < 3:
                continue
            pw = ink_points(crop)
            if pw is not None:
                out.append(pw)
        if out:
            pool[ch] = out
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_02_char_hw_v7.pth")
    ap.add_argument("--exemplars", default=str(Path.home() /
                    "Code/palimpsest/Code/palimpsest/runs/"
                    "letter_bboxes_v3c.jsonl"))
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--per-char", type=int, default=8)
    ap.add_argument("--w-turb", type=float, default=0.01)
    ap.add_argument("--probe", type=int, default=0,
                    help="dump per-char energies for the first N letters")
    ap.add_argument("--dump", default=None,
                    help="write per-letter {gt, cands:{ch:[conf,E]}} JSONL")
    ap.add_argument("--self-exemplars", action="store_true",
                    help="exemplars from the SAME document's other lines "
                         "(GT boxes; global pool fallback for uncovered "
                         "chars) — same-writer elastic matching")
    ap.add_argument("--max-lines", type=int, default=30)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    ck = torch.load(HERE / args.model, map_location=device,
                    weights_only=False)
    model = load_detector(ck, device, None).model
    lbb = {}
    for l in open(Path.home() / "Code/palimpsest/Code/palimpsest/runs/"
                  "letter_bboxes_v2.jsonl"):
        r = json.loads(l)
        lbb[r["word_id"]] = r
    print("loading exemplars...", flush=True)
    pool = load_exemplars(args.exemplars, per_char=args.per_char)
    print(f"{len(pool)} chars, "
          f"{sum(len(v) for v in pool.values())} exemplars", flush=True)

    inf = {}
    for l in open(IAM / "infill_val.jsonl"):
        r = json.loads(l)
        inf.setdefault(r["word_id"], r)

    dump = open(args.dump, "w") if args.dump else None
    tot = c_cor = e_cor = fixed = broken = 0
    conf_fix = defaultdict(int)
    t0 = time.time()

    doc_pools = {}

    def build_doc_pool(doc, exclude_line):
        """Same-writer exemplars: GT-box letters from the doc's OTHER
        val lines."""
        pool_d = defaultdict(list)
        cache_d = {}
        for l2 in open(IAM / "infill_val.jsonl"):
            r2 = json.loads(l2)
            lk = r2["record_id"].rsplit("_", 1)[0]
            if not lk.startswith(doc) or lk == exclude_line:
                continue
            w2 = lbb.get(r2["word_id"])
            if w2 is None:
                continue
            sp = w2["word_x2"] - w2["word_x1"]
            if sp < 4:
                continue
            path = IAM / r2["after_patch_ref"]
            if path not in cache_d:
                if len(cache_d) > 3:
                    cache_d.clear()
                gg = np.asarray(Image.open(path).convert("L"),
                                dtype=np.float32)
                lo2, bg2 = gg.min(), np.percentile(gg, 90)
                cache_d[path] = np.clip(
                    (gg - lo2) / max(1.0, bg2 - lo2), 0, 1)
            ga = cache_d[path]
            Wp, Hp = ga.shape[1], ga.shape[0]
            cx2, cy2, bw2, bh2 = r2["target_bbox_parent_norm_cxcywh"]
            ww1, ww2 = (cx2 - bw2 / 2) * Wp, (cx2 + bw2 / 2) * Wp
            wy1b, wy2b = (cy2 - bh2 / 2) * Hp, (cy2 + bh2 / 2) * Hp
            for L2 in w2["letters"]:
                if L2.get("pixels", 999) < 30:
                    continue
                if len(pool_d[L2["char"]]) >= args.per_char:
                    continue
                ex1 = ww1 + (L2["x1"] - w2["word_x1"]) / sp * (ww2 - ww1)
                ex2 = ww1 + (L2["x2"] - w2["word_x1"]) / sp * (ww2 - ww1)
                a1, a2 = max(0, int(ex1)), min(Wp, int(ex2))
                b1, b2 = max(0, int(wy1b)), min(Hp, int(wy2b))
                if a2 - a1 < 3 or b2 - b1 < 4:
                    continue
                pw = ink_points(ga[b1:b2, a1:a2])
                if pw is not None:
                    pool_d[L2["char"]].append(pw)
        return pool_d

    for key, img_path, words in load_lines(args.max_lines):
        doc_pool = None
        if args.self_exemplars:
            doc = key.rsplit("-", 1)[0]
            dk = (doc, key)
            if dk not in doc_pools:
                doc_pools.clear()
                doc_pools[dk] = build_doc_pool(doc, key)
            doc_pool = doc_pools[dk]
        img = Image.open(img_path).convert("L")
        W, H = img.size
        g = np.asarray(img, dtype=np.float32)
        lo, bg = g.min(), np.percentile(g, 90)
        garr = np.clip((g - lo) / max(1.0, bg - lo), 0, 1)
        for l in open(IAM / "infill_val.jsonl"):
            r = json.loads(l)
            if r["record_id"].rsplit("_", 1)[0] != key:
                continue
            w = lbb.get(r["word_id"])
            if w is None:
                continue
            span = w["word_x2"] - w["word_x1"]
            if span < 4:
                continue
            cx, cy, bw, bh = r["target_bbox_parent_norm_cxcywh"]
            wx1, wx2 = (cx - bw / 2) * W, (cx + bw / 2) * W
            wy1, wy2 = (cy - bh / 2) * H, (cy + bh / 2) * H
            for L in w["letters"]:
                rx1 = wx1 + (L["x1"] - w["word_x1"]) / span * (wx2 - wx1)
                rx2 = wx1 + (L["x2"] - w["word_x1"]) / span * (wx2 - wx1)
                x1, x2 = max(0, int(rx1) - 1), min(W, int(rx2) + 1)
                y1, y2 = max(0, int(wy1) - 2), min(H, int(wy2) + 2)
                if x2 - x1 < 3 or y2 - y1 < 3:
                    continue
                crop = garr[y1:y2, x1:x2]
                tp = ink_points(crop)
                if tp is None:
                    continue
                # classifier top-K on the same band crop
                pil = Image.fromarray(
                    (crop * 255).astype(np.uint8)).convert("RGB")
                ts = torch.from_numpy(np.array(letterbox(pil))).permute(
                    2, 0, 1).float().unsqueeze(0).to(device) / 255.0
                with torch.no_grad():
                    tk = ascii_topk(model, model.extract_features(ts),
                                    args.topk)[0]
                if not tk:
                    continue
                cands = []
                for ch, conf in tk:
                    for variant in {ch, ch.swapcase()}:
                        src_pool = None
                        if doc_pool is not None and doc_pool.get(variant):
                            src_pool = doc_pool[variant]
                        elif variant in pool:
                            src_pool = pool[variant]
                        if src_pool:
                            for pts, wts in src_pool:
                                cands.append((ch, pts, wts))
                if not cands:
                    continue
                ex_pts = torch.nn.utils.rnn.pad_sequence(
                    [c[1] for c in cands], batch_first=True).to(device)
                ex_w = torch.nn.utils.rnn.pad_sequence(
                    [c[2] for c in cands], batch_first=True).to(device)
                energies = elastic_energies(
                    ex_pts, ex_w, tp[0].to(device), tp[1].to(device),
                    device, w_turb=args.w_turb)
                # min energy per CHAR (not per exemplar) so chars with
                # more exemplars don't win by min-statistics; then argmin
                by_ch = {}
                for (ch, _, _), e in zip(cands, energies):
                    by_ch[ch] = min(by_ch.get(ch, 1e18), float(e))
                pred_e = min(by_ch, key=by_ch.get)
                if dump:
                    confs = {c: float(cf) for c, cf in tk}
                    dump.write(json.dumps(
                        {"gt": L["char"],
                         "cands": {c: [confs.get(c, 0.0), by_ch[c]]
                                   for c in by_ch}}) + "\n")
                if args.probe and tot < args.probe:
                    es = " ".join(f"{c}:{e:.3f}" for c, e in
                                  sorted(by_ch.items(), key=lambda kv: kv[1]))
                    print(f"  gt={L['char']!r} clf={tk[0][0]!r} "
                          f"E: {es}", flush=True)
                pred_c = tk[0][0]
                gt = L["char"]
                tot += 1
                ec = pred_c.lower() == gt.lower()
                ee = pred_e.lower() == gt.lower()
                c_cor += ec
                e_cor += ee
                if ee and not ec:
                    fixed += 1
                    conf_fix[(gt, pred_c)] += 1
                if ec and not ee:
                    broken += 1
        print(f"[{key}] running: clf {c_cor}/{tot}={c_cor/max(1,tot):.3f} "
              f"elastic {e_cor}/{tot}={e_cor/max(1,tot):.3f} "
              f"(+{fixed}/-{broken}) {time.time()-t0:.0f}s", flush=True)

    if dump:
        dump.close()
    print(f"\nclassifier top-1: {c_cor/max(1,tot):.3f}")
    print(f"elastic argmin-E: {e_cor/max(1,tot):.3f}  "
          f"(fixed {fixed}, broke {broken}, n={tot})")
    print("top fixes:", sorted(conf_fix.items(), key=lambda kv: -kv[1])[:10])


if __name__ == "__main__":
    main()
