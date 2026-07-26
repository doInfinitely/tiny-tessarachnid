"""Best-first tree search reading a full LINE (multiple words).

Extends beam_word_tree.py from words to lines. Same cut-lattice + best-first
search: candidate column cuts from text-density minima, eco100 top-K per
(cursor, right) crop, best-first expansion scored by det conf + GPT-2.

NEW — space edges: a (cursor, right) span with NO ink is not skipped; it
becomes a SPACE candidate edge. Its confidence comes from a font-derived
space-width prior (like the glyph width prior): gaps near the expected
space advance score high, narrow letter-spacing gaps score low. Word
segmentation therefore falls out of the decode instead of needing a
word-level detector.
"""
import argparse
import heapq
import itertools
import math
import os
import string
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser("~/Code/glyph-faerie"))
sys.path.insert(0, HERE)

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import glyph_faerie.config as gfc
gfc.settings.weights_dir = type(gfc.settings.weights_dir)(HERE)
gfc.settings.device = "cuda" if torch.cuda.is_available() else "cpu"
from glyph_faerie.detection.detector import load_detector

from detect_topk import _topk_per_window
import beam_decode as bd
from beam_word_tree import (
    letterbox, glyph_width_prior, cluster_components,
    detect_bg_and_components, clean_crop, find_cuts, _PLAUSIBLE,
)


class IncLM(bd.LM):
    """LM with per-expansion incremental scoring. All candidates scored in
    one lattice expansion share the popped state's text as a prefix, so:
    forward the parent once (keeping the KV cache), then score every child
    with a single batched forward over only its new tail tokens. BPE
    re-tokenization at the boundary (appended char merging into the last
    token) is handled by longest-common-token-prefix rollback. Semantics
    match LM.logprob_batch exactly: sum of token logprobs from token 1.
    """

    @torch.no_grad()
    def score_children(self, parent, texts):
        if not texts:
            return []
        p_ids = self.tok(parent).input_ids if parent else []
        if len(p_ids) < 2:
            return self.logprob_batch(texts)
        dev = self.device
        enc = torch.tensor([p_ids], device=dev)
        out = self.model(enc, use_cache=True)
        past = out.past_key_values
        if hasattr(past, "layers"):            # transformers >= 5
            raw_kv = [(l.keys, l.values) for l in past.layers]
        elif hasattr(past, "to_legacy_cache"):  # transformers 4.x Cache
            raw_kv = list(past.to_legacy_cache())
        else:                                   # legacy tuple
            raw_kv = list(past)
        logp = torch.log_softmax(out.logits[0, :-1], dim=-1)
        p_tok_lp = logp.gather(
            -1, enc[0, 1:].unsqueeze(-1)).squeeze(-1)          # (n-1,)
        # cum[k] = logprob sum of the first k tokens (tokens 1..k-1 scored)
        cum = torch.cat([torch.zeros(1, device=dev),
                         torch.zeros(1, device=dev),
                         torch.cumsum(p_tok_lp, 0)])

        c_ids_list = [self.tok(t).input_ids for t in texts]
        groups = {}
        for j, c in enumerate(c_ids_list):
            L = 0
            for a, b in zip(p_ids, c):
                if a != b:
                    break
                L += 1
            L = min(L, len(c) - 1)
            groups.setdefault(L if L >= 1 else None, []).append(j)

        results = [0.0] * len(texts)
        for L, idxs in groups.items():
            if L is None:
                for j, v in zip(idxs, self.logprob_batch(
                        [texts[j] for j in idxs])):
                    results[j] = v
                continue
            tails = [c_ids_list[j][L - 1:] for j in idxs]
            Tmax = max(len(t) for t in tails)
            B = len(idxs)
            ids = torch.full((B, Tmax), self.tok.eos_token_id,
                             device=dev, dtype=torch.long)
            tmask = torch.zeros((B, Tmax), device=dev)
            for r, t in enumerate(tails):
                ids[r, :len(t)] = torch.tensor(t, device=dev)
                tmask[r, :len(t)] = 1.0
            try:
                from transformers.cache_utils import DynamicCache
                past_b = DynamicCache()
                for li, (k, v) in enumerate(raw_kv):
                    past_b.update(
                        k[:, :, :L - 1].expand(B, -1, -1, -1).contiguous(),
                        v[:, :, :L - 1].expand(B, -1, -1, -1).contiguous(),
                        li)
            except ImportError:
                past_b = tuple(
                    (k[:, :, :L - 1].expand(B, -1, -1, -1),
                     v[:, :, :L - 1].expand(B, -1, -1, -1))
                    for k, v in raw_kv)
            attn = torch.cat(
                [torch.ones((B, L - 1), device=dev), tmask], dim=1)
            pos = torch.arange(L - 1, L - 1 + Tmax,
                               device=dev).unsqueeze(0).expand(B, -1)
            out2 = self.model(ids, past_key_values=past_b,
                              attention_mask=attn, position_ids=pos)
            if Tmax > 1:
                lp2 = torch.log_softmax(out2.logits[:, :-1], dim=-1)
                tok_lp = lp2.gather(
                    -1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                tail_lp = (tok_lp * tmask[:, 1:]).sum(1)
            else:
                tail_lp = torch.zeros(B, device=dev)
            base = float(cum[L])
            for r, j in enumerate(idxs):
                results[j] = base + float(tail_lp[r])
        return results


def ascii_topk(model, feats, K, block_idx=0):
    """Top-K chars conditioned on the ASCII block (block prior = 1.0).
    Avoids cross-block homoglyph dilution ('е' stealing mass from 'e')."""
    head = model.char_heads[str(block_idx)]
    chars = model.block_to_chars[block_idx]
    out = []
    with torch.no_grad():
        cp = torch.softmax(head(feats), dim=1)
    for i in range(feats.size(0)):
        tk = cp[i].topk(min(K, cp.shape[1]))
        cand = []
        for v, idx in zip(tk.values, tk.indices):
            ii = int(idx.item())
            if ii < len(chars):
                cand.append((chars[ii], float(v.item())))
        out.append(cand)
    return out


def render_line(text, font_path, size=48, bg=(255, 255, 255), fg=(0, 0, 0),
                pad=12):
    font = ImageFont.truetype(font_path, size)
    tb = ImageDraw.Draw(Image.new("RGB", (10, 10), bg)).textbbox(
        (0, 0), text, font=font)
    w, h = tb[2] - tb[0] + 2 * pad, tb[3] - tb[1] + 2 * pad
    img = Image.new("RGB", (w, h), bg)
    ImageDraw.Draw(img).text((pad - tb[0], pad - tb[1]), text, fill=fg,
                              font=font)
    return img, (pad, pad, pad + (tb[2] - tb[0]), pad + (tb[3] - tb[1]))


def space_width_prior(font_path="fonts/Arial.ttf", ref_size=64):
    """Relative advance width of the space character (space advance /
    ink line height), measured from a reference font. Size-invariant."""
    font = ImageFont.truetype(font_path, ref_size)
    d = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    adv = d.textlength(" ", font=font)
    # ink line height from the same reference used by glyph_width_prior
    tb = d.textbbox((0, 0), "Ag", font=font)
    line_h = tb[3] - tb[1]
    return adv / line_h


def gap_runs(comp_labels, L, R, T, B, min_gap_px=2):
    """Find zero-ink column runs inside the line bbox. Returns list of
    (g0, g1) spans (g1 exclusive). Each wide-enough run becomes a space
    candidate edge spanning the WHOLE run, and its boundaries become cuts
    so letter steps can end exactly at the gap."""
    text_mask = (comp_labels[T:B, L:R] > 0).astype(np.int32)
    col_density = text_mask.sum(axis=0)
    runs = []
    W = len(col_density)
    i = 0
    while i < W:
        if col_density[i] == 0:
            j = i
            while j < W and col_density[j] == 0:
                j += 1
            if j - i >= min_gap_px:
                runs.append((i + L, j + L))
            i = j
        else:
            i += 1
    return runs


def calibrate_space_threshold(runs, space_w, min_space_frac, min_runs=4):
    """Per-line space threshold from the line's own gap statistics.

    Sorts observed zero-ink run widths and looks for the largest relative
    jump — the natural split between letter-spacing gaps and word gaps.
    Falls back to the font prior when the line has too few gaps or no
    clear bimodality. A floor of 0.35x the prior space width prevents a
    false split inside the letter-gap cluster from forcing bogus spaces.
    """
    fallback = min_space_frac * space_w
    widths = sorted(g1 - g0 for g0, g1 in runs)
    if len(widths) < min_runs:
        return fallback
    best_i, best_ratio = None, 1.6
    for i in range(len(widths) - 1):
        if widths[i] <= 0:
            continue
        r = widths[i + 1] / widths[i]
        if r > best_ratio:
            best_ratio, best_i = r, i
    if best_i is None:
        return fallback
    thr = (widths[best_i] + widths[best_i + 1]) / 2.0
    return max(thr, 0.35 * space_w)


def _col_profile(img, L, T, R, B, bins=256):
    """Normalized column-ink profile + aspect ratio of a line region."""
    arr = np.array(img.convert("L"), dtype=np.float32)
    reg = 255.0 - arr[T:B, L:R]
    prof = reg.sum(axis=0)
    idx = np.linspace(0, len(prof) - 1, bins)
    prof_r = np.interp(idx, np.arange(len(prof)), prof)
    prof_r = (prof_r - prof_r.mean()) / (prof_r.std() + 1e-6)
    aspect = (R - L) / max(1, B - T)
    return prof_r, aspect


def detect_font(img, L, T, R, B, text, font_paths, size=48):
    """Analysis-by-synthesis font ID: render `text` (the pass-1 hypothesis)
    in each candidate font and compare column-ink profiles + aspect ratio
    against the actual line. A ~10%-CER hypothesis is fine — global
    geometry dominates the profile. Returns [(score, path)] best-first."""
    target_prof, target_aspect = _col_profile(img, L, T, R, B)
    scores = []
    for fp in font_paths:
        try:
            rimg, (l, t, r, b) = render_line(text, fp, size=size)
            if r <= l or b <= t:
                continue
            prof, aspect = _col_profile(rimg, l, t, r, b)
        except Exception:
            continue
        corr = float((prof * target_prof).mean())
        pen = abs(math.log(max(aspect, 1e-3) / max(target_aspect, 1e-3)))
        scores.append((corr - 2.0 * pen, fp))
    scores.sort(key=lambda s: -s[0])
    return scores


def _glyph_bitmap(pil_img, size=48):
    """Letterboxed, ink-normalized bitmap for template correlation."""
    g = np.array(letterbox(pil_img, sz=size), dtype=np.float32)
    if g.ndim == 3:
        g = g.mean(axis=2)
    ink = 255.0 - g
    ink = ink - ink.mean()
    n = np.linalg.norm(ink)
    return ink / (n + 1e-6)


_GLYPH_RENDER_CACHE = {}


def _render_glyph(ch, font_path, size=48):
    key = (ch, font_path)
    if key in _GLYPH_RENDER_CACHE:
        return _GLYPH_RENDER_CACHE[key]
    try:
        font = ImageFont.truetype(font_path, 64)
        img = Image.new("RGB", (160, 160), (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((32, 32), ch, font=font, fill=(0, 0, 0))
        arr = np.array(img.convert("L"))
        ys, xs = np.where(arr < 200)
        if len(xs) == 0:
            out = None
        else:
            crop = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
            out = _glyph_bitmap(crop, size)
    except Exception:
        out = None
    _GLYPH_RENDER_CACHE[key] = out
    return out


def detect_font_glyphs(img, T, B, steps, font_paths,
                       min_conf=0.6, max_glyphs=12, size=48):
    """Glyph-level font ID: template-match the winning path's most
    confident letter crops (actual image pixels) against each font's
    rendering of that character. Robust to a garbled hypothesis — only
    the glyphs we're confident about vote, and each votes with both case
    variants (case pooling means the path's casing may be wrong)."""
    cands = [s for s in steps if s[2] != " " and s[3] >= min_conf]
    cands.sort(key=lambda s: -s[3])
    cands = cands[:max_glyphs]
    if not cands:
        return []
    targets = []
    for cur, right, ch, conf in cands:
        crop = img.crop((cur, T, right, B))
        arr = np.array(crop.convert("L"))
        ys, xs = np.where(arr < 200)
        if len(xs) == 0:
            continue
        tight = crop.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        targets.append((ch, conf, _glyph_bitmap(tight, size)))
    if not targets:
        return []
    scores = []
    for fp in font_paths:
        total = 0.0
        for ch, conf, target in targets:
            best = 0.0
            for variant in {ch, ch.swapcase()}:
                tpl = _render_glyph(variant, fp, size)
                if tpl is not None:
                    best = max(best, float((tpl * target).sum()))
            total += conf * best
        scores.append((total / len(targets), fp))
    scores.sort(key=lambda s: -s[0])
    return scores


_FONT_TPL_CACHE = {}
_FALLBACK_CHARS = string.ascii_letters + string.digits


def _font_template_matrix(font_path, size=48):
    """(num_chars, size*size) matrix of ink-normalized glyph templates for
    one font, cached — the classifier-free font-ID sweep reuses it across
    lines and pages."""
    key = (font_path, size)
    if key in _FONT_TPL_CACHE:
        return _FONT_TPL_CACHE[key]
    mats = []
    for ch in _FALLBACK_CHARS:
        t = _render_glyph(ch, font_path, size)
        if t is not None:
            mats.append(t.ravel())
    M = np.stack(mats) if mats else None
    _FONT_TPL_CACHE[key] = M
    return M


def detect_font_any_glyph(img, T, B, font_paths, max_glyphs=10, size=48):
    """Classifier-free font ID: correlate glyph crops against EVERY
    character of each candidate font, score by best-char correlation.
    Uses CONNECTED COMPONENTS as glyph units — a garbled pass-1 decode
    also has garbled segmentation, so its steps can't be trusted here;
    components are segmentation-independent."""
    bgc, comp_labels, comps = detect_bg_and_components(img)
    clusters = cluster_components(comps)
    # prefer medium-large clusters (single glyphs); skip slivers and
    # full-word blobs
    line_h = max(1, B - T)
    scored = []
    for c in clusters:
        x0, y0, x1, y1 = c["bbox"]
        w, h = x1 - x0, y1 - y0
        if h < 0.3 * line_h or w < 3 or w > 1.8 * line_h:
            continue
        scored.append((w * h, (x0, y0, x1, y1)))
    scored.sort(key=lambda s: -s[0])
    targets = []
    for _, (x0, y0, x1, y1) in scored[:max_glyphs]:
        tight = img.crop((x0, y0, x1, y1))
        targets.append(_glyph_bitmap(tight, size).ravel())
    if not targets:
        return []
    Tm = np.stack(targets)                          # (G, D)
    out = []
    for fp in font_paths:
        M = _font_template_matrix(fp, size)         # (C, D)
        if M is None:
            continue
        corr = M @ Tm.T                             # (C, G)
        out.append((float(corr.max(axis=0).mean()), fp))
    out.sort(key=lambda s: -s[0])
    return out


def decode_line(img, box, model, lm, device, args, glyph_widths, space_w,
                width_cap=None, template_font=None, verbose=True):
    """Build the cut lattice for one line image and run the best-first
    search. Returns completions sorted by score (desc).

    width_cap: if set (e.g. 1.35), hard-drop letter candidates whose crop
    ink width disagrees with the predicted glyph's expected width by more
    than this factor (plus 4px absolute slack) in either direction. With
    font-exact glyph widths this makes multi-letter swallowing and
    glyph-splitting unrepresentable rather than merely penalized.
    """
    L, T, R, B = box
    h = B - T
    min_w = max(2, int(args.min_w_ratio * h))
    max_w = max(min_w + 1, int(args.max_w_ratio * h))

    bg_color, comp_labels, components = detect_bg_and_components(img)
    clusters = cluster_components(components)

    # scale interior cut density with line width — 40 low-density cuts is
    # enough for a ~450px render but starves a 1000px cascade strip
    eff_max_cuts = max(args.max_cuts, (R - L) // 12)
    cuts = find_cuts(comp_labels, L, R, T, B, max_cuts=eff_max_cuts)

    # ---- space candidate edges from zero-ink runs ----
    runs = gap_runs(comp_labels, L, R, T, B)
    min_space_px = calibrate_space_threshold(
        runs, space_w, args.min_space_frac)
    space_edges = {}
    spaces = set()
    extra_cuts = set()
    space_walls = []
    for g0, g1 in runs:
        extra_cuts.update((g0, g1))
        gw = g1 - g0
        if gw < min_space_px:
            continue
        space_walls.append((g0, g1))
        mismatch = abs(gw - space_w) / space_w
        conf = args.space_conf * math.exp(-0.5 * mismatch)
        for c in [g0] + [c for c in cuts if g0 < c < g1]:
            space_edges[(c, g1)] = conf
            spaces.add((c, g1))
    cuts = sorted(set(cuts) | extra_cuts)
    if verbose:
        print(f"  space edges: {len(space_edges)} from {len(space_walls)} "
              f"word gaps ({len(runs)} runs, min {min_space_px:.1f}px)")

    # ---- letter candidate crops ----
    crop_stats = {}
    crops = []
    meta = []
    for i, cur in enumerate(cuts):
        for j in range(i + 1, len(cuts)):
            right = cuts[j]
            w = right - cur
            if w < min_w or w > max_w:
                continue
            if (cur, right) in spaces:
                continue
            # word gaps are walls: `<= right` also catches edges ENDING
            # exactly at g1 (a letter crop that swallowed the whole gap) —
            # the only way past g1 is the space edge
            if any(cur < g1 <= right for g0, g1 in space_walls):
                continue
            cleaned = clean_crop(img, comp_labels, components, bg_color,
                                 cur, T, right, B,
                                 word_left=L, word_right=R)
            if cleaned is None:
                continue
            in_clusters = [c for c in clusters
                           if c["bbox"][0] >= cur and c["bbox"][2] <= right]
            if in_clusters:
                rightmost = max(in_clusters, key=lambda c: c["bbox"][2])
                rm_width = rightmost["bbox"][2] - rightmost["bbox"][0]
                ink_w = (max(c["bbox"][2] for c in in_clusters)
                         - min(c["bbox"][0] for c in in_clusters))
            else:
                rm_width = right - cur
                ink_w = right - cur
            crop_stats[(cur, right)] = (len(in_clusters), rm_width, ink_w)
            crops.append(cleaned)
            meta.append((cur, right))
    if verbose:
        print(f"  {len(crops)} letter crops, {len(cuts)} cuts")

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
            if args.ascii_only:
                tk = ascii_topk(model, feats, args.topk)
            else:
                blk = torch.softmax(model.block_head(feats), dim=1)
                tk = _topk_per_window(model, feats, blk, args.topk)
        for (cur, right), top in zip(meta[i:i + bs], tk):
            # pool probability across case pairs ONLY where the two cases
            # are the same shape at different sizes (letterboxing erases
            # size). B/b, I/i, T/t etc. differ in form — the classifier's
            # case call is real evidence there, keep it.
            probs = dict(top)
            pooled = {}
            for c, p in top:
                sw = c.swapcase()
                if sw != c and c.lower() in _CASE_AMBIGUOUS:
                    pp = p + probs.get(sw, 0.0)
                    pooled[c] = max(pooled.get(c, 0.0), pp)
                    pooled[sw] = max(pooled.get(sw, 0.0), pp)
                else:
                    pooled[c] = max(pooled.get(c, 0.0), p)
            top_plaus = [(c, p) for c, p in pooled.items()
                         if c in _PLAUSIBLE and p >= args.conf_threshold]
            top_plaus.sort(key=lambda kv: -kv[1])
            box_topk[(cur, right)] = top_plaus[:args.topk]

    # ---- hard per-char width caps ----
    if width_cap is not None:
        capped = 0
        emptied_w = []
        for (cur, right), topk in box_topk.items():
            _, _, ink_w = crop_stats[(cur, right)]
            kept = []
            for ch, conf in topk:
                expected = glyph_widths.get(ch)
                if expected is not None and expected > 1:
                    if (ink_w > expected * width_cap + 4
                            or ink_w < expected / width_cap - 4):
                        capped += 1
                        continue
                kept.append((ch, conf))
            box_topk[(cur, right)] = kept
            if not kept and topk:
                emptied_w.append(((cur, right), topk[0]))
        # adaptive keep-at-least-one: resurrecting best candidates in a
        # HEALTHY lattice creates shortcut edges that swallow characters
        # ('jumps'->'jump'); only resurrect in the degraded-font regime
        # where the filter emptied a large fraction of crops (uncial etc.)
        if emptied_w and len(emptied_w) > 0.6 * max(1, len(box_topk)):
            for key, best in emptied_w:
                box_topk[key] = [best]
        if verbose:
            print(f"  width cap {width_cap}: dropped {capped} candidates, "
                  f"{len(emptied_w)} crops emptied")

    # ---- per-step template verification against the detected font ----
    # A crop of two kerned letters ('ra') can match a wide glyph ('M') on
    # width alone — serifs even fuse them into one component. Rendering the
    # candidate char in the KNOWN font and correlating against the actual
    # crop kills these: wrong-glyph correlation is low regardless of width.
    if template_font is not None:
        dropped = 0
        emptied_t = []
        for idx, (cur, right) in enumerate(meta):
            topk = box_topk.get((cur, right))
            if not topk:
                continue
            arr = np.array(crops[idx].convert("L"))
            ys, xs = np.where(arr < 200)
            if len(xs) == 0:
                continue
            target = _glyph_bitmap(
                crops[idx].crop((xs.min(), ys.min(),
                                 xs.max() + 1, ys.max() + 1)))
            kept = []
            rescored = []
            for ch, conf in topk:
                best = 0.0
                for variant in {ch, ch.swapcase()}:
                    tpl = _render_glyph(variant, template_font)
                    if tpl is not None:
                        best = max(best, float((tpl * target).sum()))
                rescored.append((ch, conf * best, best))
                if best < args.template_min:
                    dropped += 1
                    continue
                kept.append((ch, conf * best))
            kept.sort(key=lambda kv: -kv[1])
            box_topk[(cur, right)] = kept
            if not kept and rescored:
                rescored.sort(key=lambda t: -t[2])
                emptied_t.append(((cur, right),
                                  (rescored[0][0], rescored[0][1])))
        # adaptive keep-at-least-one (see width cap note above)
        if emptied_t and len(emptied_t) > 0.6 * max(1, len(box_topk)):
            for key, best in emptied_t:
                box_topk[key] = [best]
        if verbose:
            print(f"  template verify ({os.path.basename(template_font)}): "
                  f"dropped {dropped} candidates, "
                  f"{len(emptied_t)} crops emptied")

    out_edges = {}
    for (cur, right), topk in box_topk.items():
        if topk:
            out_edges.setdefault(cur, []).append((right, topk, False))
    for (cur, right), conf in space_edges.items():
        out_edges.setdefault(cur, []).append((right, [(" ", conf)], True))

    # ---- best-first search ----
    ctx = args.context
    finish_bound = R - min_w // 2
    # states carry a linked-list `path` of (cur, right, ch, conf) so the
    # winning decode can be unwound into per-step glyph evidence (used for
    # glyph-level font detection)
    counter = 0
    heap = [(0.0, counter, L, ctx, 0.0, 0, None)]
    counter += 1
    seen = {}
    completes = []
    expansions = 0
    ctx_len = len(ctx)
    cursor_pops = {}

    # frontier batching: GPT-2 calls are LATENCY-bound at these sizes (one
    # forward costs ~7ms whether it scores 30 texts or 300), so pop several
    # states per iteration and score ALL their children in one call. Search
    # order stays best-first within each frontier slab.
    pop_batch = getattr(args, "pop_batch", 32)
    while heap and expansions < args.max_expansions \
            and len(completes) < args.max_completes:
        batch_new = []
        pops_done = 0
        while heap and pops_done < pop_batch \
                and expansions < args.max_expansions \
                and len(completes) < args.max_completes:
            neg_score, _, cur, text, lc, n, path = heapq.heappop(heap)
            score = -neg_score
            key = (cur, text.lower())
            if seen.get(key, 1e18) < score - 1e-9:
                continue
            if cur >= finish_bound:
                completes.append((score, text, lc, n, cur, path))
                continue
            if cur not in out_edges:
                continue
            pops = cursor_pops.get(cur, 0)
            if pops >= args.cursor_beam:
                continue
            cursor_pops[cur] = pops + 1
            expansions += 1
            pops_done += 1

            for right, topk, is_space in out_edges[cur]:
                if is_space:
                    if text.endswith(" ") or len(text) == ctx_len:
                        continue
                    for ch, conf in topk:
                        ntxt = text + ch
                        nlc = lc + math.log(max(conf, 1e-6))
                        batch_new.append((right, ntxt, nlc, n + 1,
                                          (path, (cur, right, ch, conf))))
                    continue
                cstats = crop_stats.get((cur, right))
                n_clust, rm_width, _ = cstats if cstats else (
                    1, right - cur, right - cur)
                multi_pen = args.beta_multi * max(0, n_clust - 1)
                step_w = right - cur
                last = text[-1] if text else ""
                for ch, conf in topk:
                    expected = glyph_widths.get(ch, rm_width)
                    slack = args.spacing_slack
                    over = max(0.0, step_w - expected - slack)
                    under = max(0.0, expected - step_w - slack)
                    letter_pen = args.beta_width * (over + under)
                    # word-interior class prior: letters don't mix with
                    # digits or bracket/colon-class symbols inside a word
                    trans_pen = 0.0
                    if last and last != " ":
                        if ((last.isalpha() and ch.isdigit())
                                or (last.isdigit() and ch.isalpha())
                                or (last.isalpha()
                                    and ch in _SYMBOLS_MIDWORD)
                                or (last in _SYMBOLS_MIDWORD
                                    and ch.isalpha())):
                            trans_pen = args.beta_trans
                    step_pen = multi_pen + letter_pen + trans_pen
                    ntxt = text + ch
                    nlc = lc + math.log(max(conf, 1e-6)) - step_pen
                    batch_new.append((right, ntxt, nlc, n + 1,
                                      (path, (cur, right, ch, conf))))
        if not batch_new:
            continue

        texts = [s[1] for s in batch_new]
        lps = lm.logprob_batch(texts, chunk=512)

        for (right, ntxt, nlc, nn, npath), lp in zip(batch_new, lps):
            sc = nlc + args.lam * lp + args.alpha * nn
            k2 = (right, ntxt.lower())
            if seen.get(k2, -1e18) >= sc:
                continue
            seen[k2] = sc
            heapq.heappush(heap, (-sc, counter, right, ntxt, nlc, nn, npath))
            counter += 1

    if verbose:
        print(f"  expanded {expansions}, {len(completes)} completions")
    completes.sort(key=lambda c: -c[0])
    return completes


def unwind_path(path):
    """Linked-list path -> list of (cur, right, ch, conf) in order."""
    steps = []
    while path is not None:
        path, step = path
        steps.append(step)
    steps.reverse()
    return steps


# case pairs whose SHAPE is identical modulo scale — pooling + height/LM
# evidence needed; all other pairs the classifier can tell apart by form
_CASE_AMBIGUOUS = set("ckosuvwxz")
# symbols that essentially never appear flanked by letters inside a word —
# 'f)x', 'W()r1d', 'sevent:een' are ellipse/stroke impersonators.
# Apostrophe and hyphen excluded (don't, it's, up-to-date). '$' excluded
# ($42 is digit-adjacent, handled by the alpha rules only).
_SYMBOLS_MIDWORD = set("()[]{}<>/&@#%+=*:;\"")
# case pairs whose SHAPE is identical and only x-height vs cap-height
# separates them — glyph height is decisive evidence
_HEIGHT_DECISIVE = set("cosuvwxz")
# the skinny-stroke confusion class: near-identical in most fonts, the
# LM arbitrates between them in the post-pass
_IL_GROUP = set("ilI|")
# lowercase has a descender, uppercase doesn't — baseline overshoot decides
_DESCENDER_DECISIVE = set("pqgy")


def case_post_pass(img, T, B, steps, text, lm, ctx=""):
    """Fix casing using evidence the lattice wastes.

    1. Per-glyph height: measure each letter step's ink top/bottom. Cluster
       heights into tall/short (largest relative gap). A short glyph cannot
       be an uppercase letter — force case on shape-identical pairs
       (c/o/s/u/v/w/x/z); descender overshoot decides p/q/g/y.
    2. Word-level LM sweep: for each word, try {as-forced, lowercase,
       Capitalized, UPPER} (respecting forced positions) and greedily pick
       the best GPT-2 score given the fixed prefix.
    """
    body = text[len(ctx):]
    letters = [s for s in steps if s[2] != " "]
    if len(letters) != len(body.replace(" ", "")):
        return body      # alignment mismatch — leave as-is

    # per-glyph ink extents
    infos = []           # (char_index_in_body, ch, ink_top, ink_bot)
    li = 0
    for bi, ch in enumerate(body):
        if ch == " ":
            continue
        cur, right, _, _ = letters[li]
        li += 1
        arr = np.array(img.crop((cur, T, right, B)).convert("L"))
        ink = arr < 200
        col_has = ink.any(axis=0)
        if not col_has.any():
            infos.append((bi, ch, None, None))
            continue
        # neighbor glyphs can bleed into the crop's edges — measure height
        # on the widest connected run of ink columns (the glyph itself),
        # not the whole crop
        runs = []
        s = None
        for x in range(len(col_has)):
            if col_has[x] and s is None:
                s = x
            elif not col_has[x] and s is not None:
                runs.append((s, x))
                s = None
        if s is not None:
            runs.append((s, len(col_has)))
        x0, x1 = max(runs, key=lambda r: r[1] - r[0])
        # asymmetric robust extents: MEDIAN of column tops (a neighbor's
        # hook rides HIGH over this glyph — Arial 'f' over 'o'), but MAX of
        # column bottoms (descenders are narrow but real; low-side
        # contamination is rare)
        tops, bots = [], []
        for x in range(x0, x1):
            rows = np.where(ink[:, x])[0]
            if len(rows):
                tops.append(rows[0])
                bots.append(rows[-1])
        if not tops:
            infos.append((bi, ch, None, None))
            continue
        infos.append((bi, ch, int(np.median(tops)), int(max(bots))))

    # cluster over ALPHABETIC glyphs only (punctuation heights would hijack
    # the gap), and only accept a split in the plausible x-height band —
    # x-height/cap-height is ~0.6-0.85 in real fonts
    forced = {}          # body index -> 'u' | 'l'
    # shape-distinct letters (B/b, T/t…): the classifier saw the form
    # unpooled — its case call is image evidence, the LM may not override.
    # i/I/l are exempt: the skinniest glyphs are the classifier's weakest
    # calls, they stay LM-variable as a confusion group below.
    for bi, ch, top, bot in infos:
        if (ch.isalpha() and ch.lower() not in _CASE_AMBIGUOUS
                and ch not in _IL_GROUP):
            forced[bi] = "u" if ch.isupper() else "l"

    heights = sorted(i[3] - i[2] for i in infos
                     if i[2] is not None and i[1].isalpha())
    if len(heights) >= 4:
        hmax = heights[-1]
        split = None
        best_dist = 1e9
        for i in range(len(heights) - 1):
            if heights[i] <= 0:
                continue
            mid = (heights[i] + heights[i + 1]) / 2.0
            if not (0.55 * hmax <= mid <= 0.95 * hmax):
                continue
            if heights[i + 1] / heights[i] < 1.15:
                continue
            # both clusters need >= 2 members — a single undersized outlier
            # glyph must not fabricate a split inside the x-height cluster
            if i + 1 < 2 or len(heights) - (i + 1) < 2:
                continue
            # among qualifying gaps pick the one nearest the canonical
            # x-height/cap boundary (~0.75 x max), not the biggest ratio
            d = abs(mid - 0.75 * hmax)
            if d < best_dist:
                best_dist, split = d, mid
        bots = [i[3] for i in infos if i[2] is not None]
        baseline = float(np.median(bots)) if bots else None
        for bi, ch, top, bot in infos:
            if top is None:
                continue
            h = bot - top
            cl = ch.lower()
            if cl in _HEIGHT_DECISIVE and split is not None:
                forced[bi] = "l" if h < split else "u"
            elif cl in _DESCENDER_DECISIVE and baseline is not None:
                forced[bi] = "l" if bot > baseline + 2 else "u"

    def apply_forced(word, start):
        out = []
        for k, c in enumerate(word):
            f = forced.get(start + k)
            if f == "l":
                out.append(c.lower())
            elif f == "u":
                out.append(c.upper())
            else:
                out.append(c)
        return "".join(out)

    # greedy word-by-word LM casing, scored with FULL line context (prefix
    # already fixed, suffix as currently cased) so early words aren't
    # decided blind
    words = body.split(" ")
    fixed = []
    pos = 0
    for wi, w in enumerate(words):
        if not w or not any(c.isalpha() for c in w):
            fixed.append(w)
            pos += len(w) + 1
            continue
        base = apply_forced(w, pos)
        cands = {base}
        for v in (w.lower(), w.capitalize(), w.upper()):
            cands.add(apply_forced(v, pos))
        # i/I/l substitutions at confusion-group positions (cap the
        # combinatorics at 3 positions -> 27 variants per base)
        il_pos = [k for k, c in enumerate(w) if c in _IL_GROUP][:3]
        if il_pos:
            for b in list(cands):
                for combo in itertools.product("ilI", repeat=len(il_pos)):
                    chars = list(b)
                    for k, c in zip(il_pos, combo):
                        chars[k] = c
                    cands.add("".join(chars))
        # digit impersonators inside otherwise-alphabetic words: offer the
        # letter reading and let the LM pick (f0x->fox, W0rld->World)
        if any(c.isalpha() for c in w) and any(
                c.isdigit() or c in _SYMBOLS_MIDWORD for c in w):
            digit_map = {"0": "oO", "1": "li", "5": "sS", "8": "B",
                         "6": "b", "9": "g",
                         ")": "o", "(": "co", "|": "li",
                         ":": "", ";": ""}
            for _ in range(3):      # closure over multi-impersonator words
                grew = False
                for b in list(cands):
                    for k, c in enumerate(b):
                        if c in digit_map:
                            # empty mapping means deletion (':' inside a
                            # word is usually a phantom cut artifact)
                            subs = digit_map[c] or [""]
                            for sub in subs:
                                v = b[:k] + sub + b[k + 1:]
                                if v not in cands:
                                    cands.add(v)
                                    grew = True
                if not grew:
                    break
        cands = sorted(cands)
        prefix = ctx + " ".join(fixed)
        if fixed:
            prefix += " "
        suffix = " ".join(words[wi + 1:])
        if suffix:
            suffix = " " + suffix
        lps = lm.logprob_batch([prefix + c + suffix for c in cands])
        # small prior toward all-lowercase: English text is mostly
        # lowercase and GPT-2 sometimes likes spurious mid-sentence caps
        scored = [lp + (0.5 if c == c.lower() else 0.0)
                  for c, lp in zip(cands, lps)]
        fixed.append(cands[int(np.argmax(scored))])
        pos += len(w) + 1
    return " ".join(fixed)


def detection_font_pool(args):
    """Candidate fonts for font-ID. font_pool='auto' uses the synthetic
    generator's discover_fonts() — the same pool pages are rendered with.
    A directory path globs .ttf/.otf/.ttc (the old .ttf-only glob silently
    excluded every CJK font, making them undetectable by construction)."""
    if args.font_pool == "auto":
        try:
            from generate_training_data import discover_fonts
            return sorted(discover_fonts())
        except Exception:
            pass
        d = "fonts"
    else:
        d = args.font_pool
    return sorted(
        os.path.join(d, f) for f in os.listdir(d)
        if f.lower().endswith((".ttf", ".otf", ".ttc")))


def read_line(img, model, lm, device, args, verbose=True, force_font=None):
    """Full two-pass read of one line image (any size, ink located
    automatically). Returns (text, detected_font_or_None, font_score).
    `model` is the eco100 char classifier, `lm` the GPT-2 wrapper — load
    once, call per line. `force_font` skips detection and uses the given
    font for pass 2 (paragraph-level consensus)."""
    bgc, comp_labels0, comps0 = detect_bg_and_components(img)
    if not comps0:
        return "", None, 0.0
    xs = [c["bbox"][0] for c in comps0] + [c["bbox"][2] for c in comps0]
    ys = [c["bbox"][1] for c in comps0] + [c["bbox"][3] for c in comps0]
    L, T, R, B = min(xs), min(ys), max(xs), max(ys)
    line_h = B - T
    if line_h < 6 or R - L < 6:
        return "", None, 0.0

    def priors_for(font_path):
        gw = {ch: rel * line_h
              for ch, rel in glyph_width_prior(font_path).items()}
        sw = space_width_prior(font_path) * line_h
        return gw, sw

    glyph_widths, space_w = priors_for("fonts/Arial.ttf")
    completes = decode_line(img, (L, T, R, B), model, lm, device, args,
                            glyph_widths, space_w, width_cap=None,
                            verbose=verbose)
    if not completes:
        return "", None, 0.0
    hyp = completes[0][1][len(args.context):].rstrip()
    det_font = None
    font_score = 0.0

    if args.two_pass and hyp:
        if force_font is not None:
            ranked = [(1.0, force_font)]
        else:
            pool = detection_font_pool(args)
            steps = unwind_path(completes[0][5])
            ranked = detect_font_glyphs(img, T, B, steps, pool)
            # garbled pass 1 → label-guided match is unreliable; fall back
            # to the classifier-free any-glyph sweep, take whichever wins
            if not ranked or ranked[0][0] < 0.45:
                alt = detect_font_any_glyph(img, T, B, pool)
                if alt and (not ranked or alt[0][0] > ranked[0][0]):
                    ranked = alt
                    if verbose:
                        print(f"  font-ID fallback (any-glyph): "
                              f"{os.path.basename(ranked[0][1])} "
                              f"{ranked[0][0]:.3f}")
            if not ranked:
                ranked = detect_font(img, L, T, R, B, hyp, pool)
        if ranked:
            det_font = ranked[0][1]
            font_score = ranked[0][0]
            if verbose:
                print(f"  font: {os.path.basename(det_font)}")
            glyph_widths, space_w = priors_for(det_font)
            completes2 = decode_line(
                img, (L, T, R, B), model, lm, device, args,
                glyph_widths, space_w, width_cap=args.width_cap,
                template_font=det_font, verbose=verbose)
            if completes2:
                f1 = case_post_pass(
                    img, T, B, unwind_path(completes[0][5]),
                    completes[0][1], lm, ctx=args.context).rstrip()
                f2 = case_post_pass(
                    img, T, B, unwind_path(completes2[0][5]),
                    completes2[0][1], lm, ctx=args.context).rstrip()
                s1 = detect_font(img, L, T, R, B, f1, [det_font])
                s2 = detect_font(img, L, T, R, B, f2, [det_font])
                sc1 = s1[0][0] if s1 else -1e9
                sc2 = s2[0][0] if s2 else -1e9
                if sc2 >= sc1 - 0.02:
                    completes = completes2

    fixed = case_post_pass(img, T, B, unwind_path(completes[0][5]),
                           completes[0][1], lm, ctx=args.context).rstrip()
    return fixed, det_font, font_score


def default_read_args(**overrides):
    """Namespace with the battery-tuned defaults for read_line."""
    d = dict(two_pass=True, font_pool="fonts", width_cap=1.35,
             template_min=0.35, min_w_ratio=0.08, max_w_ratio=1.6,
             topk=8, lam=2.0, alpha=2.0, beta_multi=3.0, beta_width=0.05,
             beta_trans=2.5, spacing_slack=5.0, conf_threshold=0.1,
             space_conf=0.9, min_space_frac=0.5, max_cuts=40,
             max_expansions=400000, max_completes=10, cursor_beam=32,
             pop_batch=8, context="", gpu_batch=128, ascii_only=True)
    d.update(overrides)
    return argparse.Namespace(**d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", default="the quick brown fox")
    ap.add_argument("--font", default="fonts/Arial.ttf")
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--image", default=None,
                    help="decode this line image instead of rendering --line")
    ap.add_argument("--model", default="model_02_char.eco100.pth")
    ap.add_argument("--two-pass", action="store_true",
                    help="pass 1 with generic priors -> detect font by "
                         "analysis-by-synthesis -> pass 2 with font-exact "
                         "glyph widths and hard width caps")
    ap.add_argument("--font-pool", default="fonts",
                    help="directory of candidate .ttf files for detection")
    ap.add_argument("--width-cap", type=float, default=1.35,
                    help="pass-2 hard cap factor on ink width vs the "
                         "detected font's expected glyph width")
    ap.add_argument("--template-min", type=float, default=0.35,
                    help="pass-2 min template correlation between a crop "
                         "and the detected font's rendering of the "
                         "candidate char")
    ap.add_argument("--min-w-ratio", type=float, default=0.08,
                    help="min step width / line height ('.', ',', 'i')")
    ap.add_argument("--max-w-ratio", type=float, default=1.6,
                    help="max step width / line height ('W', 'M', 'm')")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--beta-multi", type=float, default=3.0)
    ap.add_argument("--beta-width", type=float, default=0.05)
    ap.add_argument("--beta-trans", type=float, default=2.5,
                    help="penalty for digit/symbol chars flanked by "
                         "letters inside a word")
    ap.add_argument("--spacing-slack", type=float, default=5.0)
    ap.add_argument("--conf-threshold", type=float, default=0.5)
    ap.add_argument("--space-conf", type=float, default=0.9,
                    help="base confidence for a well-sized space edge")
    ap.add_argument("--min-space-frac", type=float, default=0.5,
                    help="min gap width as fraction of expected space "
                         "advance for a space edge to be emitted")
    ap.add_argument("--max-cuts", type=int, default=40,
                    help="extra low-density cuts across the whole line")
    ap.add_argument("--max-expansions", type=int, default=400000)
    ap.add_argument("--max-completes", type=int, default=10)
    ap.add_argument("--cursor-beam", type=int, default=32,
                    help="max states expanded per cursor position; the "
                         "correct path dominates by a wide margin so a "
                         "narrow per-cursor beam is safe and much faster")
    ap.add_argument("--context", default="")
    ap.add_argument("--gpu-batch", type=int, default=128)
    ap.add_argument("--ascii-only", action="store_true", default=True,
                    help="classify within the ASCII block only (block "
                         "prior=1) to avoid homoglyph dilution")
    ap.add_argument("--all-blocks", dest="ascii_only", action="store_false")
    args = ap.parse_args()

    device = gfc.settings.device

    if args.image:
        img = Image.open(args.image).convert("RGB")
        arr = np.array(img)
        # line bbox = ink extent
        bgc, comp_labels0, comps0 = detect_bg_and_components(img)
        xs = [c["bbox"][0] for c in comps0] + [c["bbox"][2] for c in comps0]
        ys = [c["bbox"][1] for c in comps0] + [c["bbox"][3] for c in comps0]
        L, T, R, B = min(xs), min(ys), max(xs), max(ys)
        gt = None
    else:
        img, wb = render_line(args.line, args.font, size=args.size)
        L, T, R, B = wb
        gt = args.line

    line_h = B - T
    print(f"Line: {gt!r}  bbox=({L},{T},{R},{B})  h={line_h}")

    ck = torch.load(f"{HERE}/{args.model}", map_location=device,
                    weights_only=False)
    model = load_detector(ck, device, None).model
    lm = IncLM(device)

    def priors_for(font_path):
        gw = {ch: rel * line_h
              for ch, rel in glyph_width_prior(font_path).items()}
        sw = space_width_prior(font_path) * line_h
        return gw, sw

    # ---- pass 1: generic (Arial) priors, no hard caps ----
    glyph_widths, space_w = priors_for("fonts/Arial.ttf")
    completes = decode_line(img, (L, T, R, B), model, lm, device, args,
                            glyph_widths, space_w, width_cap=None)
    if not completes:
        print("Pass 1: no completion reached the line's right edge.")
        return
    hyp = completes[0][1][len(args.context):].rstrip()
    print(f"Pass 1 hypothesis: {hyp!r}")

    if args.two_pass and hyp:
        # ---- glyph-level font detection from the pass-1 winning path ----
        pool = sorted(
            os.path.join(args.font_pool, f)
            for f in os.listdir(args.font_pool) if f.endswith(".ttf"))
        steps = unwind_path(completes[0][5])
        ranked = detect_font_glyphs(img, T, B, steps, pool)
        if not ranked:
            ranked = detect_font(img, L, T, R, B, hyp, pool)
        if ranked:
            top5 = ", ".join(f"{os.path.basename(p)}:{s:.3f}"
                             for s, p in ranked[:5])
            print(f"Font detection: {top5}")
            det_font = ranked[0][1]
            gt_base = os.path.basename(args.font) if not args.image else "?"
            hit = "HIT" if os.path.basename(det_font) == gt_base else "MISS"
            print(f"Detected font: {os.path.basename(det_font)} "
                  f"[{hit} vs {gt_base}]")

            # ---- pass 2: font-exact priors + width caps + templates ----
            glyph_widths, space_w = priors_for(det_font)
            completes2 = decode_line(
                img, (L, T, R, B), model, lm, device, args,
                glyph_widths, space_w, width_cap=args.width_cap,
                template_font=det_font)
            if completes2:
                # guard by analysis-by-synthesis, NOT the LM: render both
                # readings in the detected font and correlate against the
                # actual line image. Compare the CASE-CORRECTED texts —
                # raw caps-heavy decodes render differently and would lose
                # the correlation contest on case noise alone.
                f1 = case_post_pass(
                    img, T, B, unwind_path(completes[0][5]),
                    completes[0][1], lm, ctx=args.context).rstrip()
                f2 = case_post_pass(
                    img, T, B, unwind_path(completes2[0][5]),
                    completes2[0][1], lm, ctx=args.context).rstrip()
                s1 = detect_font(img, L, T, R, B, f1, [det_font])
                s2 = detect_font(img, L, T, R, B, f2, [det_font])
                sc1 = s1[0][0] if s1 else -1e9
                sc2 = s2[0][0] if s2 else -1e9
                if sc2 < sc1 - 0.02:
                    print(f"Pass 2 rejected by synthesis guard "
                          f"({sc2:.3f} vs {sc1:.3f}: {f2!r} vs {f1!r})")
                else:
                    completes = completes2
            else:
                print("Pass 2 found no completion — keeping pass 1 result.")

    print("\nTop 10 completions:")
    ctx_len = len(args.context)
    for i, (sc, text, lc, n, cur, _p) in enumerate(completes[:10]):
        body = text[ctx_len:].rstrip()
        marker = "  <-- GT" if gt is not None and body == gt else ""
        print(f"  [{i:3d}] score={sc:7.3f}  n={n:2d}  cursor={cur:4d}  "
              f"text={body!r}{marker}")

    top1 = completes[0][1][ctx_len:].rstrip()
    steps1 = unwind_path(completes[0][5])
    fixed = case_post_pass(img, T, B, steps1, completes[0][1], lm,
                           ctx=args.context).rstrip()
    if fixed != top1:
        print(f"\nCase post-pass: {top1!r} -> {fixed!r}")
        top1 = fixed
    print(f"\nDecoded (top-1): {top1!r}")
    if gt is not None:
        print(f"GT:              {gt!r}")
        gt_hits = [i for i, (sc, text, lc, n, cur, _p) in enumerate(completes)
                   if text[ctx_len:].rstrip() == gt]
        if gt_hits:
            print(f"GT found at rank {gt_hits[0]}")
        else:
            print("GT not in completions.")


if __name__ == "__main__":
    main()
