"""GPT-2-guided beam search over the detection lattice (no NMS).

Instead of NMS (greedy top-1 by confidence per location) + grid-search tuning,
decode the page left-to-right with a beam search. Selecting a box advances a
cursor past it — which suppresses the boxes it overlaps — and the next column
of candidate boxes is considered. Because a wide 'm' box advances the cursor
past the 'r'/'n' positions while choosing 'r' leaves 'n' as the next column,
beams explore alternative *segmentations*, and GPT-2 scores which reading is
most language-like.

Beam score = sum(log detector_conf) + lambda * GPT-2 cumulative log-prob(text).

Scope (--context):
  line : cluster boxes into lines, beam-search each line; the best decoded text
         of previous lines is carried as GPT-2 context (committed greedily).
  page : one beam search over all boxes in reading order (lines joined with
         newlines); beams span the whole page with a single growing context.
"""
import argparse
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field

logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, os.path.abspath("../glyph-faerie"))

import numpy as np
import torch

import glyph_faerie.config as gfc
gfc.settings.weights_dir = type(gfc.settings.weights_dir)(".")
gfc.settings.device = "cuda" if torch.cuda.is_available() else "cpu"

from glyph_faerie.detection.detector import (
    load_detector, detect_characters_raw, DetectorConfig,
    apply_block_priors,
)
from generate_training_data import SyntheticPage, discover_fonts

_CORPUS = (
    "the quick brown fox jumps over the lazy dog while the sun sets slowly "
    "behind the distant hills and a gentle breeze moves through the tall grass "
    "many years ago a small village stood beside a wide and quiet river where "
    "people gathered every morning to trade goods and share the latest news "
    "she opened the old wooden door and stepped into a room filled with books "
    "stacked from the floor to the ceiling in long and crooked towers of paper "
    "the engineers worked through the night to repair the bridge before the "
    "first train arrived carrying passengers from the northern cities at dawn"
).split()


def _natural_line(min_words=4, max_words=11):
    n = random.randint(min_words, max_words)
    start = random.randint(0, max(0, len(_CORPUS) - n))
    return " ".join(_CORPUS[start:start + n])


SyntheticPage._random_line = staticmethod(_natural_line)


# ---------------------------------------------------------------------------
# GPT-2 language model scorer
# ---------------------------------------------------------------------------
class LM:
    def __init__(self, device, model_name=None):
        import os as _os
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        model_name = model_name or _os.environ.get("BLT_LM", "gpt2")
        self.tok = GPT2TokenizerFast.from_pretrained(model_name)
        self.tok.pad_token = self.tok.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(
            model_name).eval().to(device)
        self.device = device

    @torch.no_grad()
    def logprob_batch(self, texts, ctx_len_tokens=0, chunk=32):
        """Total teacher-forced log-prob of each text (sum over tokens beyond
        the shared context prefix). Chunked to bound GPU memory."""
        if not texts:
            return []
        out_all = []
        for i in range(0, len(texts), chunk):
            sub = texts[i:i + chunk]
            enc = self.tok(sub, return_tensors="pt", padding=True,
                           truncation=True, max_length=1024)
            ids = enc.input_ids.to(self.device)
            attn = enc.attention_mask.to(self.device)
            out = self.model(ids, attention_mask=attn)
            logp = torch.log_softmax(out.logits[:, :-1, :], dim=-1)
            tgt = ids[:, 1:]
            tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            mask = attn[:, 1:].clone().float()
            if ctx_len_tokens > 1:
                mask[:, :ctx_len_tokens - 1] = 0.0
            out_all.extend((tok_lp * mask).sum(dim=1).tolist())
        return out_all

    def n_tokens(self, text):
        return len(self.tok(text).input_ids)


# ---------------------------------------------------------------------------
# Line clustering
# ---------------------------------------------------------------------------
def drop_oversized(boxes, size_mult=2.5):
    """Drop coarse windows (e.g. scale-0.5 128px) whose height is far above the
    typical glyph height — their y-centers are scattered +/- half-window and
    smear the line bands."""
    if not boxes:
        return boxes
    hs = sorted(b["bbox"][3] - b["bbox"][1] for b in boxes)
    medh = hs[len(hs) // 2] or 16
    cap = size_mult * medh
    return [b for b in boxes if (b["bbox"][3] - b["bbox"][1]) <= cap]


def cluster_lines(boxes, sep_mult=1.5, thr_frac=0.10):
    """Find text-line centers as PEAKS (modes) in the y-center density of the
    tight, confident boxes, then assign every box to its nearest center.

    Lines are spaced ~one char-height apart, so box y-centers form a
    continuous band with no gaps — gap-based splitting chains lines via any
    single box in an inter-line gap. Peak-finding (1D NMS on the histogram) is
    robust to that. Returns line box-lists, top-to-bottom.
    """
    if not boxes:
        return []
    heights = np.array([b["bbox"][3] - b["bbox"][1] for b in boxes], dtype=float)
    confs = np.array([b["confidence"] for b in boxes], dtype=float)
    yc = np.array([(b["bbox"][1] + b["bbox"][3]) / 2.0 for b in boxes])

    mask = (heights <= np.percentile(heights, 50)) & \
           (confs >= np.percentile(confs, 60))
    if mask.sum() < 2:
        mask = np.ones(len(boxes), dtype=bool)
    th = float(np.median(heights[mask]))
    min_sep = sep_mult * th

    ty = yc[mask]
    ymin, ymax = float(ty.min()), float(ty.max())
    nbins = max(1, int((ymax - ymin) / 2.0) + 1)        # 2px bins
    hist, edges = np.histogram(ty, bins=nbins, range=(ymin, ymax + 1e-3))
    w = max(1, int(th / 2))
    if w > 1:
        hist = np.convolve(hist.astype(float), np.ones(w) / w, mode="same")
    bc = (edges[:-1] + edges[1:]) / 2.0

    # Greedy 1D non-max suppression to pick peaks >= thr, separated by min_sep.
    thr = thr_frac * hist.max()
    taken = np.zeros(len(hist), dtype=bool)
    peaks = []
    for idx in np.argsort(-hist):
        if hist[idx] < thr:
            break
        if taken[idx]:
            continue
        peaks.append(bc[idx])
        taken |= np.abs(bc - bc[idx]) <= min_sep
    peaks = np.sort(np.array(peaks)) if peaks else np.array([float(np.median(yc))])

    lines = [[] for _ in peaks]
    for b, c in zip(boxes, yc):
        j = int(np.argmin(np.abs(peaks - c)))
        lines[j].append(b)
    return [ln for ln in lines if ln]


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------
@dataclass
class Beam:
    text: str = ""          # decoded text for the current scope (line or page)
    cursor: float = -1e9    # x right-edge of last selected box
    logconf: float = 0.0    # sum of log detector confidence
    lm: float = 0.0         # GPT-2 logprob of (context + text)
    n: int = 0              # number of selected boxes

    rep: int = 0            # count of adjacent same-char repeats (penalized)

    def score(self, lam, norm, rep_pen=0.0):
        s = self.logconf + lam * self.lm - rep_pen * self.rep
        return s / max(self.n, 1) if norm else s


def _candidates(beam, line_boxes, col_tol, min_adv, topk):
    """Boxes competing to be the next character after the beam's cursor."""
    eligible = [b for b in line_boxes if b["bbox"][2] > beam.cursor + min_adv]
    if not eligible:
        return []
    next_left = min(b["bbox"][0] for b in eligible)
    col = [b for b in eligible if b["bbox"][0] <= next_left + col_tol]
    # dedup by char (keep most confident box per char), then top-k by conf
    best_by_char = {}
    for b in col:
        c = b["char"]
        if c not in best_by_char or b["confidence"] > best_by_char[c]["confidence"]:
            best_by_char[c] = b
    cands = sorted(best_by_char.values(), key=lambda b: -b["confidence"])
    return cands[:topk]


def decode_line(line_boxes, lm, context, cfg):
    """Beam-search one line. Returns (best_text, best_beam)."""
    line_boxes = sorted(line_boxes, key=lambda b: b["bbox"][0])
    widths = sorted(b["bbox"][2] - b["bbox"][0] for b in line_boxes)
    medw = widths[len(widths) // 2] or 16
    space_thresh = cfg.space_frac * medw
    col_tol = cfg.col_frac * medw
    ctx_prefix = (context + " ") if context else ""
    ctx_tok = lm.n_tokens(ctx_prefix) if context else 0

    active = [Beam(text="", cursor=-1e9, logconf=0.0, lm=0.0, n=0)]
    complete = []

    for _ in range(cfg.max_steps):
        specs = []  # (parent, box, new_text, new_cursor, new_logconf, sep)
        next_active = []
        for beam in active:
            cands = _candidates(beam, line_boxes, col_tol, cfg.min_adv, cfg.topk)
            if not cands:
                complete.append(beam)
                continue
            for b in cands:
                gap = b["bbox"][0] - beam.cursor
                sep = " " if (beam.n > 0 and gap > space_thresh) else ""
                new_text = beam.text + sep + b["char"]
                specs.append((beam, b, new_text))
        if not specs:
            break
        texts = [ctx_prefix + s[2] for s in specs]
        lps = lm.logprob_batch(texts, ctx_len_tokens=ctx_tok)
        children = []
        for (parent, b, new_text), lp in zip(specs, lps):
            prev_char = parent.text[-1] if parent.text else ""
            rep_inc = 1 if (b["char"] == prev_char and b["char"] != " ") else 0
            children.append(Beam(
                text=new_text,
                cursor=b["bbox"][2],
                logconf=parent.logconf + float(np.log(max(b["confidence"], 1e-6))),
                lm=lp,
                n=parent.n + 1,
                rep=parent.rep + rep_inc,
            ))
        children.sort(key=lambda bm: -bm.score(cfg.lam, cfg.norm, cfg.rep_pen))
        active = children[:cfg.beam_width]

    pool = complete + active
    if not pool:
        return "", None
    best = max(pool, key=lambda bm: bm.score(cfg.lam, cfg.norm, cfg.rep_pen))
    return best.text, best


@dataclass
class Cfg:
    beam_width: int = 12
    topk: int = 16
    lam: float = 1.0
    norm: bool = True
    rep_pen: float = 3.0
    space_frac: float = 0.6
    col_frac: float = 0.25
    min_adv: float = 2.0
    max_steps: int = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="model_02_char.pth")
    ap.add_argument("--context", choices=["line", "page"], default="line")
    ap.add_argument("--beam-width", type=int, default=12)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--prune-conf", type=float, default=0.3)
    ap.add_argument("--min-std", type=float, default=0.10)
    ap.add_argument("--min-edge", type=float, default=0.06)
    ap.add_argument("--no-norm", action="store_true",
                    help="disable length normalization")
    ap.add_argument("--rep-pen", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = Cfg(beam_width=args.beam_width, lam=args.lam, topk=args.topk,
              norm=not args.no_norm, rep_pen=args.rep_pen)
    random.seed(args.seed)
    print(f"### Beam decode: {args.model}  context={args.context} "
          f"W={cfg.beam_width} lam={cfg.lam} topk={cfg.topk} "
          f"prune={args.prune_conf} norm={cfg.norm} ###")

    fonts = discover_fonts()
    page = SyntheticPage(fonts, 1024, 1400)
    gt_lines = []
    for para in page.paragraphs:
        for line in para["lines"]:
            gt_lines.append(" ".join(
                "".join(c["char"] for c in w["characters"])
                for w in line["words"]))
    gt_text = "\n".join(gt_lines)

    device = gfc.settings.device
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    detector = load_detector(ckpt, device, None)
    raw = detect_characters_raw(page.image, detector,
                                DetectorConfig(confidence_threshold=0.0))
    n_raw = len(raw)
    # Background rejection: drop blank/low-structure windows (the model
    # confidently classifies blank inter-line space otherwise, which erases
    # the y-gaps between text lines). Same criterion as _reject_blank_windows.
    raw = [d for d in raw if d["pixel_std"] >= args.min_std
           and d["edge_density"] >= args.min_edge]
    n_bg = len(raw)
    apply_block_priors(raw)                      # down-weight exotic blocks
    raw = [d for d in raw if d["confidence"] >= args.prune_conf]
    n_pruned = len(raw)
    raw = drop_oversized(raw)                    # drop coarse smearing windows
    print(f"Detections: raw {n_raw} -> bg-filter {n_bg} -> "
          f"conf>={args.prune_conf} {n_pruned} -> size {len(raw)}")

    lm = LM(device)
    lines = cluster_lines(raw)
    # order lines top-to-bottom by mean y
    lines.sort(key=lambda ln: np.mean([(b["bbox"][1] + b["bbox"][3]) / 2.0
                                       for b in ln]))
    print(f"Lines detected: {len(lines)} (gt lines={len(gt_lines)})")

    t0 = time.time()
    out_lines = []
    context = ""
    for li, ln in enumerate(lines):
        text, _ = decode_line(ln, lm, context if args.context == "page" else
                              (context if args.context == "line" else ""), cfg)
        out_lines.append(text)
        if args.context in ("line", "page"):
            context = (context + "\n" + text) if context else text
            # keep context bounded
            if len(context) > 600:
                context = context[-600:]
    elapsed = time.time() - t0

    pred_text = "\n".join(out_lines)
    print("\n" + "=" * 70 + "\nPREDICTED (beam):\n" + "=" * 70)
    print(pred_text.strip())
    print("\n" + "=" * 70 + "\nGROUND TRUTH:\n" + "=" * 70)
    print(gt_text.strip())
    print("=" * 70)
    print(f"Decode time: {elapsed:.1f}s for {len(lines)} lines")


if __name__ == "__main__":
    main()
