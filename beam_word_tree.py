"""Best-first tree search reading a word.

Step 1: precompute eco100's top-K predictions for *every* (cursor, right_edge)
        box in the word at a discrete grid. One big batched GPU pass.
Step 2: best-first search over the lattice. State = (cursor, text). Priority =
        (sum log det_conf + lam * GPT-2_logprob(text)) / n_chars.
        Pop the most-promising open state, expand it (generate one child per
        candidate (right, char)), push all children. The first state that
        reaches the word right edge is the best (or near-best) completion.
        No beam width — every candidate stays in the heap until popped or
        beaten.
"""
import argparse
import heapq
import math
import os
import sys
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser("~/Code/glyph-faerie"))
sys.path.insert(0, HERE)

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label, find_objects

import glyph_faerie.config as gfc
gfc.settings.weights_dir = type(gfc.settings.weights_dir)(HERE)
gfc.settings.device = "cuda" if torch.cuda.is_available() else "cpu"
from glyph_faerie.detection.detector import load_detector

from detect_topk import _topk_per_window
import beam_decode as bd


def render_word(text, font_path, size=48, bg=(255, 255, 255), fg=(0, 0, 0),
                pad=10):
    font = ImageFont.truetype(font_path, size)
    tb = ImageDraw.Draw(Image.new("RGB", (10, 10), bg)).textbbox(
        (0, 0), text, font=font)
    w, h = tb[2] - tb[0] + 2 * pad, tb[3] - tb[1] + 2 * pad
    img = Image.new("RGB", (w, h), bg)
    ImageDraw.Draw(img).text((pad - tb[0], pad - tb[1]), text, fill=fg,
                              font=font)
    return img, (pad, pad, pad + (tb[2] - tb[0]), pad + (tb[3] - tb[1]))


def letterbox(crop, sz=128, bg=(255, 255, 255)):
    w, h = crop.size
    sc = min(sz / w, sz / h)
    nw, nh = max(1, int(w * sc)), max(1, int(h * sc))
    canvas = Image.new("RGB", (sz, sz), bg)
    canvas.paste(crop.resize((nw, nh), Image.LANCZOS),
                  ((sz - nw) // 2, (sz - nh) // 2))
    return canvas


def glyph_width_prior(font_path="fonts/Arial.ttf", ref_size=64):
    """Per-character RELATIVE INK width (rendered glyph ink / line height),
    from a reference font's true rendered metrics. Size-invariant: at
    inference we scale by the page's observed line height.
    """
    from PIL import Image as _Image, ImageDraw as _ID
    font = ImageFont.truetype(font_path, ref_size)
    pad = ref_size
    canvas_w = ref_size * 4
    canvas_h = ref_size * 3
    widths = {}
    y0s, y1s = [], []
    for cp in range(32, 127):
        ch = chr(cp)
        img = _Image.new("L", (canvas_w, canvas_h), 255)
        d = _ID.Draw(img)
        d.text((pad, pad), ch, font=font, fill=0)
        arr = np.array(img)
        ink_cols = np.where(arr.min(axis=0) < 128)[0]
        ink_rows = np.where(arr.min(axis=1) < 128)[0]
        if len(ink_cols):
            widths[ch] = max(1, ink_cols[-1] - ink_cols[0] + 1)
        else:
            widths[ch] = 1
        if len(ink_rows):
            y0s.append(ink_rows[0])
            y1s.append(ink_rows[-1])
    line_h = (max(y1s) - min(y0s) + 1) if y0s else ref_size
    return {ch: w / line_h for ch, w in widths.items()}


def cluster_components(components):
    """Group connected components into char clusters by x-overlap.
    'i' (dot + stem) at the same x merges into one cluster; 'l' followed by
    'g' (no x overlap) become two clusters. Returns list of cluster dicts
    {bbox: (x0,y0,x1,y1), comp_idxs: [...]} sorted by x.
    """
    if not components:
        return []
    cs = sorted(components, key=lambda c: c["bbox"][0])
    out = []
    for c in cs:
        if out and c["bbox"][0] <= out[-1]["bbox"][2]:
            last = out[-1]
            last["bbox"] = (
                last["bbox"][0],
                min(last["bbox"][1], c["bbox"][1]),
                max(last["bbox"][2], c["bbox"][2]),
                max(last["bbox"][3], c["bbox"][3]),
            )
            last["comp_idxs"].append(c["idx"])
        else:
            out.append({"bbox": tuple(c["bbox"]), "comp_idxs": [c["idx"]]})
    return out


def detect_bg_and_components(img, threshold=30):
    """Find text components in `img`. Returns (bg_color, labels, components).

    `bg_color`: median of 4 corner pixels (RGB tuple).
    `labels`:   (H, W) int array, component id per pixel (0 = bg).
    `components`: list of {idx, bbox=(x0,y0,x1,y1 exclusive)} for each component.
    """
    arr = np.array(img)
    h, w = arr.shape[:2]
    corners = np.stack([arr[0, 0], arr[0, w - 1], arr[h - 1, 0],
                        arr[h - 1, w - 1]], axis=0)
    bg = tuple(int(v) for v in np.median(corners, axis=0))
    diff = np.abs(arr.astype(int) - np.array(bg, dtype=int)).sum(axis=2)
    text_mask = diff > threshold
    labels, _ = label(text_mask)
    comps = []
    for i, sl in enumerate(find_objects(labels)):
        if sl is None:
            continue
        comps.append({"idx": i + 1,
                      "bbox": (sl[1].start, sl[0].start,
                               sl[1].stop, sl[0].stop)})
    return bg, labels, comps


def clean_crop(img, labels, components, bg_color, cur, T, right, B,
               word_left=None, word_right=None, edge_tol=2):
    """Raw column-wise crop. Pixels outside [cur, right] are naturally
    excluded by the crop itself, so merged glyphs split cleanly along the
    crop boundary. Returns None if the crop has no text.

    Connected components are no longer used as a hard gate — relying instead
    on the relative-width prior (and cluster-cardinality penalty) to softly
    discourage multi-letter crops.
    """
    if right <= cur:
        return None
    arr = np.array(img.crop((cur, T, right, B)))
    if arr.size == 0:
        return None
    bg = np.asarray(bg_color, dtype=np.int16)
    if not np.any(np.abs(arr.astype(np.int16) - bg).max(axis=-1) > 30):
        return None
    return Image.fromarray(arr)


def find_cuts(comp_labels, L, R, T, B, max_cuts=10):
    """Find candidate column cuts (letter boundaries) inside the word bbox.

    All zero-density columns are taken as cuts (they're optimal boundaries —
    no ink to slice through). Runs of consecutive zeros collapse to a single
    cut at the run's midpoint. Then, up to `max_cuts` additional cuts come
    from the lowest non-zero density columns, to handle merged glyphs by
    splitting wide clusters internally.
    """
    text_mask = (comp_labels[T:B, L:R] > 0).astype(np.int32)
    col_density = text_mask.sum(axis=0)
    W = len(col_density)
    picked = set()
    # zero-density runs → one cut at the midpoint of each run
    i = 0
    while i < W:
        if col_density[i] == 0:
            j = i
            while j < W and col_density[j] == 0:
                j += 1
            mid = (i + j - 1) // 2 + L
            picked.add(mid)
            i = j
        else:
            i += 1
    # extra cuts from lowest non-zero columns (split merged glyphs)
    nz = [i for i in range(W) if col_density[i] > 0]
    nz_sorted = sorted(nz, key=lambda i: int(col_density[i]))
    extra = 0
    for i in nz_sorted:
        x = i + L
        if x <= L + 2 or x >= R - 2:
            continue
        if any(abs(x - p) < 3 for p in picked):
            continue
        picked.add(x)
        extra += 1
        if extra >= max_cuts:
            break
    return sorted({L, R, *picked})


_PLAUSIBLE = set(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    ".,;:!?'\"-()[]{}<>/&@#$%+=*"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", default="Minimum")
    ap.add_argument("--font", default="fonts/Arial.ttf")
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--model", default="model_02_char.eco100.pth")
    ap.add_argument("--cursor-step", type=int, default=3)
    ap.add_argument("--width-step", type=int, default=3)
    ap.add_argument("--min-w-ratio", type=float, default=0.2)
    ap.add_argument("--max-w-ratio", type=float, default=1.1)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=2.0,
                    help="per-step length bonus added to cumulative score "
                         "(removes the bias toward shorter completions)")
    ap.add_argument("--beta-multi", type=float, default=3.0,
                    help="penalty per extra char-cluster contained in a step's "
                         "crop (single-letter steps preferred)")
    ap.add_argument("--beta-width", type=float, default=0.05,
                    help="penalty per pixel of step-width excess beyond the "
                         "picked letter's font-prior width (+ spacing slack)")
    ap.add_argument("--spacing-slack", type=float, default=5.0,
                    help="px tolerance for inter-letter spacing beyond a "
                         "letter's intrinsic glyph width")
    ap.add_argument("--conf-threshold", type=float, default=0.5,
                    help="prune per-step predictions below this confidence; "
                         "true-letter crops score ~1.0 so weak picks are "
                         "almost always wrong")
    ap.add_argument("--max-cuts", type=int, default=10,
                    help="number of candidate column cuts (letter "
                         "boundaries) sampled from text-density minima")
    ap.add_argument("--max-expansions", type=int, default=200000)
    # max-completes declared below
    ap.add_argument("--context", default="")
    ap.add_argument("--gpu-batch", type=int, default=128)
    ap.add_argument("--max-completes", type=int, default=500,
                    help="stop search after finding this many completions")
    args = ap.parse_args()

    device = gfc.settings.device

    img, wb = render_word(args.word, args.font, size=args.size)
    L, T, R, B = wb
    h = B - T
    min_w = max(2, int(args.min_w_ratio * h))
    max_w = max(min_w + 1, int(args.max_w_ratio * h))
    print(f"Word: {args.word!r}  bbox=({L},{T},{R},{B})  h={h}  "
          f"w in [{min_w},{max_w}] step={args.width_step}")

    ck = torch.load(f"{HERE}/{args.model}", map_location=device,
                    weights_only=False)
    model = load_detector(ck, device, None).model
    lm = bd.LM(device)

    # ---- Step 1: precompute top-K for every candidate (cur, right) box ----
    bg_color, comp_labels, components = detect_bg_and_components(img)
    clusters = cluster_components(components)
    glyph_widths_rel = glyph_width_prior()
    line_h = B - T
    glyph_widths = {ch: rel * line_h for ch, rel in glyph_widths_rel.items()}
    print(f"  bg={bg_color}  components={len(components)}  "
          f"clusters={len(clusters)}  line_h={line_h}")

    # Candidate cuts from text-density minima — interior letter boundaries
    cuts = find_cuts(comp_labels, L, R, T, B, max_cuts=args.max_cuts)
    print(f"  cuts ({len(cuts)}): {cuts}")

    crop_stats = {}
    crops = []
    meta = []
    n_skipped = 0
    for i, cur in enumerate(cuts):
        for j in range(i + 1, len(cuts)):
            right = cuts[j]
            w = right - cur
            if w < min_w or w > max_w:
                continue
            cleaned = clean_crop(img, comp_labels, components, bg_color,
                                 cur, T, right, B,
                                 word_left=L, word_right=R)
            if cleaned is None:
                n_skipped += 1
                continue
            in_clusters = [c for c in clusters
                           if c["bbox"][0] >= cur and c["bbox"][2] <= right]
            if in_clusters:
                rightmost = max(in_clusters, key=lambda c: c["bbox"][2])
                rm_width = rightmost["bbox"][2] - rightmost["bbox"][0]
            else:
                rm_width = right - cur
            crop_stats[(cur, right)] = (len(in_clusters), rm_width)
            crops.append(cleaned)
            meta.append((cur, right))
    print(f"Precomputing {len(crops)} box predictions (top-{args.topk} each, "
          f"skipped {n_skipped} empty after scrub)...")

    # batched eco100 forward
    box_topk = {}
    bs = args.gpu_batch
    for i in range(0, len(crops), bs):
        chunk = crops[i:i + bs]
        ts = torch.stack([
            torch.from_numpy(np.array(letterbox(c))).permute(2, 0, 1).float()
            / 255.0
            for c in chunk
        ]).to(device)
        with torch.no_grad():
            feats = model.extract_features(ts)
            blk = torch.softmax(model.block_head(feats), dim=1)
            tk = _topk_per_window(model, feats, blk, args.topk)
        for (cur, right), top in zip(meta[i:i + bs], tk):
            top_plaus = [(c, p) for c, p in top
                         if c in _PLAUSIBLE and p >= args.conf_threshold]
            box_topk[(cur, right)] = top_plaus

    # index transitions by source cursor
    out_edges = {}
    for (cur, right), topk in box_topk.items():
        out_edges.setdefault(cur, []).append((right, topk))
    print(f"  done. {len(out_edges)} source cursors with edges.")

    # ---- Step 2: best-first tree search ----
    ctx = args.context
    finish_bound = R - min_w // 2

    # heap entries: (-score, counter, cur, text, logconf, n)
    counter = 0
    start_score = 0.0  # dummy
    heap = [(start_score, counter, L, ctx, 0.0, 0)]
    counter += 1

    seen = {}     # (cur, text) -> best score seen
    completes = []
    expansions = 0
    ctx_len = len(ctx)

    max_completes = getattr(args, "max_completes", 500)
    while heap and expansions < args.max_expansions \
            and len(completes) < max_completes:
        neg_score, _, cur, text, lc, n = heapq.heappop(heap)
        score = -neg_score
        key = (cur, text)
        if seen.get(key, 1e18) < score - 1e-9:
            continue
        if cur >= finish_bound:
            completes.append((score, text, lc, n, cur))
            continue
        if cur not in out_edges:
            continue
        expansions += 1

        new = []
        for right, topk in out_edges[cur]:
            cstats = crop_stats.get((cur, right))
            n_clust, rm_width = cstats if cstats else (1, right - cur)
            # cluster-cardinality penalty: each extra cluster (beyond 1)
            # represents a letter being silently swallowed by a wide crop
            multi_pen = args.beta_multi * max(0, n_clust - 1)
            step_w = right - cur
            for ch, conf in topk:
                # font-derived width prior: the step's width should match
                # the intrinsic glyph width of the predicted letter (plus a
                # small inter-letter spacing slack). BIDIRECTIONAL — both
                # too-wide (swallowed neighbor) and too-narrow (false
                # narrow letter splitting a real glyph) are penalized.
                expected = glyph_widths.get(ch, rm_width)
                slack = args.spacing_slack
                over = max(0.0, step_w - expected - slack)
                under = max(0.0, expected - step_w - slack)
                letter_pen = args.beta_width * (over + under)
                step_pen = multi_pen + letter_pen
                ntxt = text + ch
                nlc = lc + math.log(max(conf, 1e-6)) - step_pen
                new.append((right, ntxt, nlc, n + 1))
        if not new:
            continue

        texts = [s[1] for s in new]
        lps = lm.logprob_batch(texts)

        for (right, ntxt, nlc, nn), lp in zip(new, lps):
            # Cumulative score with per-step length bonus + cluster/width
            # penalties (already baked into nlc).
            sc = nlc + args.lam * lp + args.alpha * nn
            k2 = (right, ntxt)
            if seen.get(k2, -1e18) >= sc:
                continue
            seen[k2] = sc
            heapq.heappush(heap, (-sc, counter, right, ntxt, nlc, nn))
            counter += 1

    print(f"\nExpanded {expansions} states, found {len(completes)} completions, "
          f"heap remaining={len(heap)}")
    if not completes:
        print("No completion reached the word's right edge.")
        return

    completes.sort(key=lambda c: -c[0])
    print("\nTop 10 completions:")
    for i, (sc, text, lc, n, cur) in enumerate(completes[:10]):
        marker = "  <-- GT" if text[ctx_len:] == args.word else ""
        print(f"  [{i:3d}] score={sc:7.3f}  n={n:2d}  cursor={cur:3d}  "
              f"text={text[ctx_len:]!r}{marker}")
    # GT location
    gt_hits = [(i, sc, text, lc, n, cur) for i, (sc, text, lc, n, cur)
               in enumerate(completes) if text[ctx_len:] == args.word]
    if gt_hits:
        i, sc, text, lc, n, cur = gt_hits[0]
        gap_to_top = completes[0][0] - sc
        print(f"\nGT exact match: rank={i} score={sc:.3f} "
              f"(gap below top: {gap_to_top:.3f})")
    else:
        # closest by similarity
        import difflib
        scored_by_sim = [(difflib.SequenceMatcher(None, t[1][ctx_len:],
                                                   args.word).ratio(), i, t)
                         for i, t in enumerate(completes)]
        scored_by_sim.sort(key=lambda x: -x[0])
        print(f"\nGT exact NOT in {len(completes)} completions.")
        print(f"Closest completion by similarity:")
        for sim, i, (sc, text, lc, n, cur) in scored_by_sim[:3]:
            print(f"  rank={i:3d} sim={sim:.3f} score={sc:7.3f} "
                  f"text={text[ctx_len:]!r}")
    best = completes[0]
    print(f"\nDecoded (top-1): {best[1][ctx_len:]!r}")
    print(f"GT:              {args.word!r}")


if __name__ == "__main__":
    main()
