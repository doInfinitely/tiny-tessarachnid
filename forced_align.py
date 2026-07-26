"""Forced-alignment line decoder for TOUCHING glyphs.

The cut-lattice reader (beam_line_tree) needs zero/low-ink columns between
letters; fonts like Luminari join letters and the lattice only ever sees
multi-letter blobs. With the font KNOWN (font-ID from a sibling line or
paragraph consensus), decoding becomes forced alignment instead: slide the
font's exact glyph templates across the strip with a pen-advance DP —
placement IS segmentation.

Score per placement = ink overlap F-measure between the baseline-aligned
template and the strip; transitions use the font's true advance widths
with small kerning jitter; spaces advance over (near-)empty columns.
Per-char correlation curves are precomputed with FFT so the DP is cheap.
"""
import argparse
import os
import string
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import fftconvolve

CHARS = string.ascii_lowercase + string.ascii_uppercase + string.digits + \
    ".,!?'-"


def _strip_ink(img):
    """Binary ink map + big-component stats (top, bottom rows)."""
    from scipy.ndimage import label, find_objects
    arr = np.array(img.convert("L"))
    ink = (arr < 200).astype(np.float32)
    labels, n = label(ink > 0)
    tops, bots, heights = [], [], []
    for sl in find_objects(labels):
        if sl is None:
            continue
        h = sl[0].stop - sl[0].start
        area = h * (sl[1].stop - sl[1].start)
        if area >= 30:
            tops.append(sl[0].start)
            bots.append(sl[0].stop)
            heights.append(h)
    return ink, tops, bots, heights


def _render_glyph_metrics(ch, font, ascent):
    """Render one glyph on a tall canvas with the pen at (PAD, PAD) and
    baseline at PAD+ascent. Returns (ink (H,W), advance, lsb, ink_rows)."""
    PAD = 8
    adv = font.getlength(ch)
    W = int(adv) + 4 * PAD + 8
    H = int(ascent * 2.2) + 2 * PAD
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)
    d.text((PAD, PAD), ch, font=font, fill=0)
    arr = np.array(img)
    ink = (arr < 200).astype(np.float32)
    return ink, adv, PAD, PAD + ascent


def build_font_bank(font_path, target_ascender_h, chars=CHARS):
    """Glyph bank scaled so the font's rendered 'h' ink height matches the
    strip's measured ascender height. Each entry: template ink positioned
    relative to (pen_x, baseline), plus advance."""
    size = max(12, int(target_ascender_h))
    for _ in range(3):      # iterate size until 'h' ink height matches
        font = ImageFont.truetype(font_path, size)
        ascent, _ = font.getmetrics()
        ink, adv, pen_x, base_y = _render_glyph_metrics("h", font, ascent)
        rows = np.where(ink.any(axis=1))[0]
        if len(rows) == 0:
            break
        h_ink = rows[-1] - rows[0] + 1
        if abs(h_ink - target_ascender_h) <= 1:
            break
        size = max(8, int(round(size * target_ascender_h / max(h_ink, 1))))
    font = ImageFont.truetype(font_path, size)
    ascent, _ = font.getmetrics()

    bank = {}
    for ch in chars:
        ink, adv, pen_x, base_y = _render_glyph_metrics(ch, font, ascent)
        cols = np.where(ink.any(axis=0))[0]
        rows = np.where(ink.any(axis=1))[0]
        if len(cols) == 0 or adv <= 0:
            continue
        tpl = ink[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        bank[ch] = {
            "tpl": tpl,
            "dx": int(cols[0] - pen_x),          # ink offset from pen
            "dy": int(rows[0] - base_y),         # ink offset from baseline
            "adv": float(adv),
            "mass": float(tpl.sum()),
        }
    space_adv = float(font.getlength(" ")) or 0.3 * size
    return bank, space_adv, size


def correlation_curves(ink, bank, baseline):
    """For each char: curve[x] = template∩strip ink mass with the pen at
    x (baseline-aligned). Vectorized with FFT along both axes then sliced
    at the fixed vertical offset."""
    H, W = ink.shape
    curves = {}
    for ch, g in bank.items():
        tpl = g["tpl"]
        th, tw = tpl.shape
        y0 = baseline + g["dy"]
        if y0 < -th or y0 >= H:
            continue
        ya, yb = max(0, y0), min(H, y0 + th)
        if yb - ya < max(2, th // 4):
            continue
        sub = ink[ya:yb, :]
        tp = tpl[ya - y0:yb - y0, :]
        # correlate: inter[x] = sum_{dy,dx} tp[dy,dx] * sub[dy, x+dx]
        c = fftconvolve(sub, tp[::-1, ::-1], mode="full")
        row = c[tp.shape[0] - 1]                 # aligned overlap row
        # row index i corresponds to ink column offset i - (tw - 1)
        inter = np.zeros(W, dtype=np.float32)
        for x in range(W):
            i = x + g["dx"] + tw - 1
            if 0 <= i < len(row):
                inter[x] = row[i]
        curves[ch] = np.maximum(inter, 0.0)
    return curves


def align_read(img, font_path, jitter=(-2, -1, 0, 1, 2),
               space_ink_tol=0.15,
               baseline_search=(-3, 0, 3), scale_search=(0.92, 1.0, 1.08)):
    """Decode a line strip by forced alignment with `font_path`'s glyphs.
    Searches a small grid of baseline offsets x scales (the strip's
    metrics estimates are ~few-px accurate) and returns the best
    (text, score). Pure image evidence — pair with the usual case/LM
    post-passes afterwards."""
    best = ("", -1e18)
    for b_off in baseline_search:
        for sc in scale_search:
            text, score = _align_once(img, font_path, b_off, sc, jitter,
                                      space_ink_tol)
            if score > best[1]:
                best = (text, score)
    return best


def _align_once(img, font_path, baseline_off, scale, jitter,
                space_ink_tol):
    ink, tops, bots, heights = _strip_ink(img)
    H, W = ink.shape
    if not tops:
        return "", -1e18
    baseline = int(np.median(bots)) + baseline_off
    asc_h = (baseline - int(min(tops))) * scale
    if asc_h < 8:
        return "", -1e18

    bank, space_adv, size = build_font_bank(font_path, asc_h)
    if not bank:
        return "", -1e18

    col_ink = ink.sum(axis=0)
    cum = np.concatenate([[0.0], np.cumsum(col_ink)])

    def range_ink(a, b):
        a = max(0, min(W, a))
        b = max(0, min(W, b))
        if b <= a:
            return 0.0
        return float(cum[b] - cum[a])

    curves = correlation_curves(ink, bank, baseline)
    chars = list(curves.keys())

    inked = np.where(col_ink > 0)[0]
    x_start = int(inked[0])
    x_end = int(inked[-1])

    # DP over pen positions: dp[x] = (score, backpointer)
    NEG = -1e18
    dp = np.full(W + 8, NEG)
    back = {}
    for s0 in range(max(0, x_start - 6), x_start + 3):
        dp[s0] = 0.0
    order = np.argsort(-dp[:W + 1])       # process reachable first — but
    # positions are advanced monotonically; simple left-to-right sweep:
    for x in range(W + 1):
        if dp[x] <= NEG / 2:
            continue
        base_score = dp[x]
        # glyph transitions
        for ch in chars:
            g = bank[ch]
            curve = curves.get(ch)
            if curve is None:
                continue
            tw = g["tpl"].shape[1]
            for j in jitter:
                px = x + j
                if px < 0 or px >= W:
                    continue
                inter = float(curve[px])
                nx = int(round(px + g["adv"]))
                if nx <= x or nx > W:
                    continue
                # raw symmetric-difference in ink-pixel units: advance
                # ranges partition the line, so total strip mass is a
                # constant and the DP maximizes 2*overlap - template mass.
                # No normalization — small glyphs can't shrug off
                # unexplained ink; overclaiming costs the same currency.
                if inter < 0.45 * g["mass"]:
                    continue
                strip_mass = range_ink(min(x, px), nx)
                sc = 2.0 * inter - g["mass"] - strip_mass
                val = base_score + sc
                if val > dp[nx]:
                    dp[nx] = val
                    back[nx] = (x, ch)
        # space transition: advance over (near-)empty columns
        for mult in (1.0, 1.4):
            nx = int(round(x + space_adv * mult))
            if nx <= x or nx > W:
                continue
            gap_ink = range_ink(x, nx)
            if gap_ink > space_ink_tol * space_adv * asc_h:
                continue
            val = base_score - 0.05
            if val > dp[nx]:
                dp[nx] = val
                back[nx] = (x, " ")

    # best completion at/after the last ink column
    best_x, best_v = None, NEG
    for x in range(max(0, x_end - 4), W + 1):
        if dp[x] > best_v:
            best_v, best_x = dp[x], x
    if best_x is None or best_v <= NEG / 2:
        return "", -1e18
    out = []
    x = best_x
    while x in back:
        x, ch = back[x]
        out.append(ch)
    out.reverse()
    text = "".join(out).strip()
    # normalize double spaces
    while "  " in text:
        text = text.replace("  ", " ")
    return text, float(best_v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--font", required=True)
    ap.add_argument("--gt", default=None)
    args = ap.parse_args()

    img = Image.open(args.image).convert("RGB")
    text, score = align_read(img, args.font)
    print(f"Aligned read: {text!r}  (score {score:.2f})")
    if args.gt:
        import difflib
        e = 1.0 - difflib.SequenceMatcher(
            None, text.lower(), args.gt.lower()).ratio()
        print(f"GT:           {args.gt!r}  cer={e:.2f}")


if __name__ == "__main__":
    main()
