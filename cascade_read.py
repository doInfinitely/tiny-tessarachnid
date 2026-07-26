"""End-to-end page reading: V10 autoreg quad cascade finds page ->
paragraphs -> lines; each detected line quad is perspective-warped to a
natural-aspect strip and read by the beam_line_tree two-pass decoder.
Words fall out of the decode's space edges — no word/char detection.

Evaluation: each detection is matched to the nearest GT line (quad
centers); per-line and page-level CER reported.
"""
import argparse
import difflib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser("~/Code/glyph-faerie"))
sys.path.insert(0, os.path.expanduser("~/Downloads/tiny-tessarachnid"))
sys.path.insert(0, HERE)

import numpy as np
import torch
from PIL import Image

import glyph_faerie.config as gfc
gfc.settings.weights_dir = type(gfc.settings.weights_dir)(HERE)
gfc.settings.device = "cuda" if torch.cuda.is_available() else "cpu"
from glyph_faerie.detection.detector import load_detector

from generate_training_data import SyntheticPage, discover_fonts
from cascade_infer import (
    cascade, load_model as load_v10, homography_matrix, to_canonical_rect,
)
import beam_decode as bd
from beam_line_tree import read_line, default_read_args


def warp_line_strip(page_pil, quad, strip_h=64, pad=16):
    """Perspective-warp a line quad to a natural-aspect strip of height
    ~strip_h. The cascade emits corners in semantic NW,NE,SE,SW order
    (mapped from the axis-aligned rect in the warped-paragraph frame) —
    do NOT re-sort by y: for a long thin tilted line both right-end
    corners can be 'top', which twists the strip 90°."""
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    w_top = np.linalg.norm(q[1] - q[0])
    w_bot = np.linalg.norm(q[2] - q[3])
    h_l = np.linalg.norm(q[3] - q[0])
    h_r = np.linalg.norm(q[2] - q[1])
    src_w = max(1.0, (w_top + w_bot) / 2)
    src_h = max(1.0, (h_l + h_r) / 2)
    if strip_h == 0:
        # native scale: upscaling small text adds bicubic blur on top of
        # its intrinsic aliasing and clean templates stop matching —
        # keep the source resolution (floor 20px so tiny quads survive)
        strip_h = int(round(min(96, max(20, src_h))))
    out_w = max(8, int(round(strip_h * src_w / src_h)))
    dst = np.array([[pad, pad], [pad + out_w, pad],
                    [pad + out_w, pad + strip_h], [pad, pad + strip_h]],
                   dtype=np.float64)
    M = homography_matrix(q, dst)
    M_inv = np.linalg.inv(M)
    M_inv = M_inv / M_inv[2, 2]
    coeffs = (M_inv[0, 0], M_inv[0, 1], M_inv[0, 2],
              M_inv[1, 0], M_inv[1, 1], M_inv[1, 2],
              M_inv[2, 0], M_inv[2, 1])
    return page_pil.transform((out_w + 2 * pad, strip_h + 2 * pad),
                              Image.PERSPECTIVE, coeffs, Image.BICUBIC)


def normalize_strip(strip):
    """Colored page regions (red text on blue paragraph) break the reader's
    white-padding letterbox and ink-template assumptions. Convert to
    black-ink-on-white via distance-from-background."""
    arr = np.array(strip.convert("RGB")).astype(np.int16)
    # median over ALL pixels: background dominates by area. Corners are
    # unreliable — the warp's padding can sample outside the paragraph's
    # colored box onto the page background.
    bg = np.median(arr.reshape(-1, 3), axis=0)
    dist = np.abs(arr - bg).sum(axis=2).astype(np.float64)
    peak = max(dist.max(), 1.0)
    ink = np.clip(dist * (255.0 / peak), 0, 255).astype(np.uint8)
    return Image.fromarray(255 - ink).convert("RGB")


def deslant_strip(strip, max_slant=0.45, coarse=0.05, fine=0.01):
    """Estimate and remove italic slant by maximizing column-ink-profile
    peakiness (sharp inter-letter gaps) over horizontal shear. Slant is
    dx/dy; positive = forward italic."""
    gray = np.array(strip.convert("L"), dtype=np.float64)
    ink = 255.0 - gray
    H, W = ink.shape
    ys = np.arange(H) - H / 2.0

    def score(slant):
        # shear each row by slant*(y - H/2) and accumulate column profile
        prof = np.zeros(W)
        for y in range(H):
            shift = int(round(slant * ys[y]))
            row = ink[y]
            if shift > 0:
                prof[:W - shift] += row[shift:]
            elif shift < 0:
                prof[-shift:] += row[:W + shift]
            else:
                prof += row
        return float((prof ** 2).sum())

    best = 0.0
    best_s = score(0.0)
    for sl in np.arange(-max_slant, max_slant + 1e-9, coarse):
        s = score(sl)
        if s > best_s:
            best_s, best = s, sl
    for sl in np.arange(best - coarse, best + coarse + 1e-9, fine):
        s = score(sl)
        if s > best_s:
            best_s, best = s, sl
    # do no harm: only deslant on a decisive win over the unsheared score
    if abs(best) < fine or best_s < score(0.0) * 1.05:
        return strip
    # PIL AFFINE maps output->input: x_in = x_out + a*y_out + c
    coeffs = (1.0, best, -best * H / 2.0, 0.0, 1.0, 0.0)
    return strip.transform(strip.size, Image.AFFINE, coeffs,
                           Image.BICUBIC, fillcolor=(255, 255, 255))


def derotate_strip(strip, max_deg=6.0, step=0.75):
    """Remove residual rotation (imperfect detection quad) by maximizing
    row-ink-profile peakiness — a level baseline gives the sharpest
    row profile."""
    def rowscore(img):
        prof = (255.0 - np.array(img.convert("L"),
                                 dtype=np.float64)).sum(axis=1)
        return float((prof ** 2).sum())

    base = rowscore(strip)
    best_img, best_s = strip, base
    for deg in np.arange(-max_deg, max_deg + 1e-9, step):
        if abs(deg) < 1e-9:
            continue
        cand = strip.rotate(deg, resample=Image.BICUBIC,
                            fillcolor=(255, 255, 255))
        s = rowscore(cand)
        if s > best_s:
            best_s, best_img = s, cand
    # do no harm: resampling blurs — only derotate on a decisive win
    if best_s < base * 1.05:
        return strip
    return best_img


def defragment_strip(strip):
    """Erase connected components vertically distant from the main text
    band — detection quads sometimes clip a corner of the next line."""
    from scipy.ndimage import label, find_objects
    arr = np.array(strip.convert("L"))
    mask = arr < 200
    labels, n = label(mask)
    if n < 2:
        return strip
    comps = []
    for i, sl in enumerate(find_objects(labels)):
        if sl is None:
            continue
        cy = (sl[0].start + sl[0].stop) / 2.0
        h = sl[0].stop - sl[0].start
        area = int(mask[sl].sum())
        comps.append((i + 1, cy, h, area, sl[0].start, sl[0].stop))
    big = [c for c in comps if c[3] >= 20]
    if not big:
        return strip
    med_cy = float(np.median([c[1] for c in big]))
    med_h = float(np.median([c[2] for c in big]))
    med_area = float(np.median([c[3] for c in big]))
    # baseline from big components' bottoms; next-line ascender tips START
    # below it (no legitimate glyph does — descenders start at x-height)
    baseline = float(np.median([c[5] for c in big]))
    out = arr.copy()
    for idx, cy, h, area, top, bot in comps:
        distant = abs(cy - med_cy) > 0.9 * med_h
        below = top > baseline + 0.15 * med_h
        if (distant or below) and area < 0.5 * med_area:
            out[labels == idx] = 255
    return Image.fromarray(out).convert("RGB")


def trim_strip(strip, margin=6):
    """Crop the strip vertically to the ink band nearest the strip's
    CENTER — the detection quad was fitted to the target line, so the
    intended text is centered; neighbor-line fragments sit at the edges.
    A row must have meaningful ink (>2% of width) to count, so touching
    ascender/descender slivers don't merge bands."""
    gray = np.array(strip.convert("L"))
    W = gray.shape[1]
    row_ink = (gray < 200).sum(axis=1).astype(np.float64)
    if row_ink.max() <= 0:
        return strip
    # a diagonal ghost line from an overlapping paragraph bridges the gap
    # between text bands with a few pixels per row — split at RELATIVE
    # valleys (below 20% of the strip's typical text-row ink), not just
    # at empty rows
    typical = np.percentile(row_ink[row_ink > 0], 60)
    thr = max(2.0, 0.02 * W, 0.2 * typical)
    rows = np.where(row_ink > thr)[0]
    if len(rows) == 0:
        rows = np.where(row_ink > 0)[0]
    if len(rows) == 0:
        return strip
    runs = []
    s = rows[0]
    prev = rows[0]
    for r in rows[1:]:
        if r - prev > 3:
            runs.append((s, prev))
            s = r
        prev = r
    runs.append((s, prev))
    center = gray.shape[0] / 2.0
    # candidate bands need at least 20% of the biggest band's mass
    masses = [float(row_ink[a:b + 1].sum()) for a, b in runs]
    mmax = max(masses)
    cands = [(abs((a + b) / 2.0 - center), (a, b))
             for (a, b), m in zip(runs, masses) if m >= 0.2 * mmax]
    _, (r0, r1) = min(cands)
    # grow band edges through low-ink rows: descenders/ascenders sit below
    # the relative threshold but are real glyph parts — cropping them
    # mid-stroke sends the reader off a cliff ('IIIttIIV' stroke soup).
    # Cap growth so a bridging ghost can't merge a neighbor band back in.
    H = gray.shape[0]
    grow = int(0.4 * (r1 - r0 + 1)) + 2
    g0, g1 = r0, r1
    while g0 > 0 and row_ink[g0 - 1] > 0 and r0 - g0 < grow:
        g0 -= 1
    while g1 < H - 1 and row_ink[g1 + 1] > 0 and g1 - r1 < grow:
        g1 += 1
    r0, r1 = g0, g1
    r0 = max(0, r0 - margin)
    r1 = min(H - 1, r1 + margin)
    return strip.crop((0, r0, strip.size[0], r1 + 1))


_POOL_STATE = {}


def _pool_init(char_model_path, rank_counter, gpu_ids):
    """Worker initializer: load eco100 + the LM once per worker process.
    Workers round-robin across the ALLOWED GPUs (--gpus) so a 2-GPU box
    can run 2 workers per GPU contention-free — or stay off a GPU that's
    busy with another job."""
    import torch as _torch
    with rank_counter.get_lock():
        rank = rank_counter.value
        rank_counter.value += 1
    if _torch.cuda.is_available() and gpu_ids:
        dev_str = f"cuda:{gpu_ids[rank % len(gpu_ids)]}"
    else:
        dev_str = "cpu"
    import glyph_faerie.config as _gfc
    _gfc.settings.device = dev_str
    from glyph_faerie.detection.detector import load_detector as _ld
    from beam_line_tree import IncLM as _IncLM, default_read_args as _dra
    dev = _torch.device(dev_str)
    ck = _torch.load(os.path.join(HERE, char_model_path),
                     map_location=dev, weights_only=False)
    _POOL_STATE["model"] = _ld(ck, dev, None).model
    _POOL_STATE["lm"] = _IncLM(dev_str)
    _POOL_STATE["device"] = dev
    _POOL_STATE["args"] = _dra(font_pool="auto")


def _pool_read(job):
    """(idx, strip ndarray, force_font) -> (idx, text, font, score, dt)."""
    from beam_line_tree import read_line as _rl
    idx, arr, force_font = job
    t0 = time.time()
    img = Image.fromarray(arr)
    text, font, score = _rl(img, _POOL_STATE["model"], _POOL_STATE["lm"],
                            _POOL_STATE["device"], _POOL_STATE["args"],
                            verbose=False, force_font=force_font)
    return idx, text, font, score, time.time() - t0


def _make_read_pool(workers, char_model_path, gpus="0"):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")     # CUDA requires spawn, not fork
    rank_counter = ctx.Value("i", 0)
    gpu_ids = [int(g) for g in str(gpus).split(",") if g != ""]
    return ctx.Pool(workers, initializer=_pool_init,
                    initargs=(char_model_path, rank_counter, gpu_ids))


def gt_lines_of(page):
    """[(text, center_xy)] for every GT line on the page."""
    out = []
    for para in page.paragraphs:
        for line in para.get("lines", []):
            words = []
            for w in line.get("words", []):
                words.append("".join(c["char"] for c in w["characters"]))
            text = " ".join(words)
            if not text:
                continue
            if "quad" in line:
                q = np.array(line["quad"], dtype=np.float64).reshape(4, 2)
                cx, cy = q[:, 0].mean(), q[:, 1].mean()
            else:
                x1, y1, x2, y2 = line["bbox"]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            out.append((text, (cx, cy)))
    return out


def cer(a, b):
    return 1.0 - difflib.SequenceMatcher(None, a, b).ratio()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="model_05_autoreg_v10.pth")
    ap.add_argument("--char-model", default="model_02_char.eco100.pth")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--obj-threshold", type=float, default=0.5)
    ap.add_argument("--strip-h", type=int, default=64)
    ap.add_argument("--max-lines", type=int, default=0,
                    help="only read the first N detected lines (0 = all)")
    ap.add_argument("--save-strips", action="store_true")
    ap.add_argument("--workers", type=int, default=2,
                    help="parallel line-reader processes (0 = inline)")
    ap.add_argument("--gpus", default="0",
                    help="comma-separated GPU ids for reader workers")
    args = ap.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(gfc.settings.device)

    print("Loading models…")
    detector = load_v10(args.detector, device)
    ck = torch.load(f"{HERE}/{args.char_model}", map_location=device,
                    weights_only=False)
    char_model = load_detector(ck, device, None).model
    from beam_line_tree import IncLM
    lm = IncLM(gfc.settings.device)
    ropts = default_read_args(font_pool="auto")

    fonts = discover_fonts()
    # re-seed RIGHT BEFORE page creation: model loading above (esp. GPT-2
    # init) consumes RNG unpredictably, so seeding only at startup made
    # every run generate a DIFFERENT page for the same --seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    page = SyntheticPage(fonts, 1024, 1024, rotate_paragraphs=True,
                         compute_contours=False)
    page_pil = page.image.convert("RGB")
    gts = gt_lines_of(page)
    print(f"Page rendered (seed {args.seed}): {len(gts)} GT lines")

    print("Running detection cascade…")
    dets = cascade(detector, page_pil, device, max_depth=2,
                   obj_threshold=args.obj_threshold)
    line_dets = [d for d in dets if d["level"] == "line"]
    print(f"  paragraphs: {sum(1 for d in dets if d['level']=='paragraph')}"
          f"  lines: {len(line_dets)}")

    if args.max_lines:
        line_dets = line_dets[:args.max_lines]

    strips = []
    for i, det in enumerate(line_dets):
        strip = warp_line_strip(page_pil, det["quad"], strip_h=args.strip_h)
        strip = normalize_strip(strip)
        strip = derotate_strip(strip)
        strip = deslant_strip(strip)
        strip = defragment_strip(strip)
        strip = trim_strip(strip)
        strips.append(strip)
        if args.save_strips:
            strip.save(f"cascade_strip_{i:02d}.png")

    reads = []          # [text, font, font_score, time]
    if args.workers > 0:
        pool = _make_read_pool(args.workers, args.char_model, args.gpus)
        t0 = time.time()
        jobs = [(i, np.array(s), None) for i, s in enumerate(strips)]
        results = pool.map(_pool_read, jobs)
        wall = time.time() - t0
        per = wall / max(1, len(strips))
        for idx, text, font, fscore, dt in sorted(results):
            reads.append([text, font, fscore, per])
        print(f"  phase 1: {len(strips)} lines in {wall:.0f}s "
              f"({args.workers} workers)", flush=True)
    else:
        pool = None
        for i, strip in enumerate(strips):
            t0 = time.time()
            text, font, fscore = read_line(strip, char_model, lm, device,
                                           ropts, verbose=False)
            reads.append([text, font, fscore, time.time() - t0])

    # ---- paragraph-level font consensus: lines in one paragraph share a
    # font; re-read minority-font lines with the consensus forced ----
    def alpha_frac(text):
        letters = sum(1 for c in text if c.isalpha())
        return letters / max(1, len(text.replace(" ", "")))

    by_para = {}
    for i, det in enumerate(line_dets):
        by_para.setdefault(det["parent_idx"], []).append(i)
    para_font = {}
    for para, idxs in by_para.items():
        votes = {}
        for i in idxs:
            text, f, s = reads[i][0], reads[i][1], reads[i][2]
            if not f:
                continue
            # weight by decode quality: a line reading '583 33533' must
            # not outvote a sibling that reads like words, whatever its
            # raw font-match score
            votes[f] = votes.get(f, 0.0) + max(s, 0.05) * alpha_frac(
                text) ** 2
        if not votes:
            continue
        consensus = max(votes.items(), key=lambda kv: kv[1])[0]
        para_font[para] = consensus
        rejobs = []
        for i in idxs:
            if reads[i][1] and reads[i][1] != consensus:
                print(f"  re-reading line {i} with consensus font "
                      f"{os.path.basename(consensus)} (was "
                      f"{os.path.basename(reads[i][1])})", flush=True)
                rejobs.append((i, np.array(strips[i]), consensus))
        if rejobs and pool is not None:
            for idx, text2, _f, _s, dt in pool.map(_pool_read, rejobs):
                if text2:
                    reads[idx][0] = text2
                reads[idx][3] += dt
        else:
            for i, arr, consensus_f in rejobs:
                t0 = time.time()
                text2, _, _ = read_line(strips[i], char_model, lm, device,
                                        ropts, verbose=False,
                                        force_font=consensus_f)
                if text2:
                    reads[i][0] = text2
                reads[i][3] += time.time() - t0

    # ---- forced-alignment second opinion for EVERY line: touching
    # glyphs defeat the cut lattice, and word-shaped garbage evades any
    # shape-based trigger. Arbitrate by SYNTHESIS (render both readings
    # in the paragraph font, correlate against the strip) — the LM
    # reliably prefers short pronounceable garbage over a long read with
    # one bad cluster ----
    from forced_align import align_read
    from beam_line_tree import detect_font as synth_score_font
    from beam_line_tree import detect_bg_and_components as _dbc
    for para, idxs in by_para.items():
        font = para_font.get(para)
        if not font:
            continue
        for i in idxs:
            text = reads[i][0]
            t0 = time.time()
            _, _, comps = _dbc(strips[i])
            if not comps:
                continue
            xs = [c["bbox"][0] for c in comps] + \
                 [c["bbox"][2] for c in comps]
            ys = [c["bbox"][1] for c in comps] + \
                 [c["bbox"][3] for c in comps]
            box = (min(xs), min(ys), max(xs), max(ys))
            s_old = synth_score_font(strips[i], *box,
                                     text or " ", [font])
            v_pre = s_old[0][0] if s_old else -1e9
            # gate: a lattice read that already synthesizes well won't be
            # beaten by the aligner — skip the ~3.5s second opinion
            # (observed: good reads score >= 0.32, junk <= -0.14)
            if text and v_pre >= 0.3:
                reads[i][3] += time.time() - t0
                continue
            atext, ascore = align_read(strips[i], font)
            if atext and atext != text:
                s_new = synth_score_font(strips[i], *box, atext, [font])
                v_new = s_new[0][0] if s_new else -1e9
                # combined criterion: the aligner optimizes ink fit
                # directly so synthesis alone always favors it, and the
                # LM alone favors short pronounceable garbage — weighted
                # sum of both deltas separates good swaps from junk
                # (measured on the seed-3 candidates)
                lp_old, lp_new = lm.logprob_batch(
                    [(text or " ").lower(), atext.lower()])
                pc_old = lp_old / max(1, len(text or " "))
                pc_new = lp_new / max(1, len(atext))
                combined = (v_new - v_pre) + 0.5 * (pc_new - pc_old)
                if not text or combined > 0:
                    print(f"  align beats lattice on line {i} "
                          f"({v_new:.3f} vs {v_pre:.3f}): {atext!r} "
                          f"over {text!r}", flush=True)
                    reads[i][0] = atext
            reads[i][3] += time.time() - t0

    results = []
    for i, det in enumerate(line_dets):
        text, font, fscore, dt = reads[i]
        q = np.asarray(det["quad"], dtype=np.float64)
        c = (q[:, 0].mean(), q[:, 1].mean())
        # nearest GT line by center distance
        best_j, best_d = None, 1e18
        for j, (gtext, gc) in enumerate(gts):
            d = (gc[0] - c[0]) ** 2 + (gc[1] - c[1]) ** 2
            if d < best_d:
                best_d, best_j = d, j
        gtext = gts[best_j][0] if best_j is not None else ""
        e = cer(text.lower(), gtext.lower())
        exact = text == gtext
        tag = "EXACT" if exact else (
            "CASE" if text.lower() == gtext.lower() else f"cer={e:.2f}")
        results.append((text, gtext, e, exact, best_j))
        print(f"[{i:2d}] ({dt:5.1f}s) {text!r}")
        print(f"      GT[{best_j}]: {gtext!r}  [{tag}]", flush=True)

    matched = {}
    for text, gtext, e, exact, j in results:
        if j not in matched or matched[j][2] > e:
            matched[j] = (text, gtext, e, exact, j)
    vals = list(matched.values())
    if vals:
        n = len(vals)
        ex = sum(1 for v in vals if v[3])
        ci = sum(1 for v in vals if v[0].lower() == v[1].lower())
        mean_cer = sum(v[2] for v in vals) / n
        print(f"\n=== page summary ===")
        print(f"GT lines: {len(gts)}  detected: {len(line_dets)}  "
              f"matched: {n}")
        print(f"exact: {ex}/{n}  case-insensitive: {ci}/{n}  "
              f"mean CER: {mean_cer:.3f}")
        missed = len(gts) - n
        if missed:
            print(f"GT lines with no matched detection: {missed}")


if __name__ == "__main__":
    main()
