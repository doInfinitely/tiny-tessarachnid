#!/usr/bin/env python3
"""Differentiable divider descent over a line strip.

Runs the column-wise char net once -> response matrix logp [C+1, W]
(log-softmax per column; last class = blank). For a hypothesis of N
interior dividers: positions are W * cumsum(softplus(theta)) / sum
(ordered by construction); each segment pools the matrix through a
soft window (sigmoid ramps at its dividers); segment score = best
class (blank allowed — gap segments are legitimate) of the weighted
log-mean-exp pool; objective = mean segment score − width-prior
penalty. Adam on theta. Dividers from several N hypotheses are pooled
as candidate cuts for the lattice decoder.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from train_08_colchar import ColCharNet, STRIP_H

TAU = 2.0          # window ramp softness (px at strip scale)
MEAN_W = 0.45      # expected segment width, fraction of strip height


def response_matrix(net, band, device):
    """band: np float [h,w] 0..1 (1=white). Returns (logp [C,W], scale)
    where scale maps strip columns back to band columns."""
    sc = STRIP_H / band.shape[0]
    Wn = max(8, int(band.shape[1] * sc))
    strip = np.asarray(Image.fromarray(
        ((band) * 255).astype(np.uint8)).resize(
        (Wn, STRIP_H), Image.LANCZOS), dtype=np.float32) / 255.0
    x = torch.from_numpy(1.0 - strip).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = net(x)[0]                    # [C, Wn]
    return F.log_softmax(logits, dim=0), 1.0 / sc


def descend(logp, n_div, device, iters=150, lr=0.08, w_width=0.3):
    """One hypothesis: n_div interior dividers over [0, W].
    Returns (positions [n_div], mean segment score)."""
    C, W = logp.shape
    theta = torch.zeros(n_div + 1, device=device, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    mu = MEAN_W * STRIP_H
    for it in range(iters):
        gaps = F.softplus(theta) + 2.0
        pos = W * torch.cumsum(gaps, 0) / gaps.sum()   # last == W
        edges = torch.cat([torch.zeros(1, device=device), pos])
        left, right = edges[:-1], edges[1:]            # [n_seg]
        wfun = (torch.sigmoid((xs.unsqueeze(0) - left.unsqueeze(1)) / TAU)
                * torch.sigmoid((right.unsqueeze(1) - xs.unsqueeze(0))
                                / TAU))                # [n_seg, W]
        wsum = wfun.sum(1).clamp(min=1e-6)
        # weighted log-mean-exp pool per (segment, class)
        pool = torch.logsumexp(
            logp.unsqueeze(0) + torch.log(wfun.clamp(min=1e-9))
            .unsqueeze(1), dim=2) - torch.log(wsum).unsqueeze(1)
        seg_score = pool.max(dim=1).values             # best class/seg
        widths = right - left
        wpen = ((widths - mu) / (0.9 * mu)).pow(2)
        loss = -(seg_score.mean()) + w_width * wpen.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        gaps = F.softplus(theta) + 2.0
        pos = W * torch.cumsum(gaps, 0) / gaps.sum()
        return (pos[:-1].cpu().numpy(),
                float(seg_score.mean()))


def propose_cuts(net, band, device, n_hyp=2):
    """All-hypotheses divider proposal for one line band. Returns cut
    x-positions in band coordinates (ints, deduped)."""
    logp, scale = response_matrix(net, band, device)
    W = logp.shape[1]
    n0 = max(2, round(W / (MEAN_W * STRIP_H)))
    cuts = set()
    for n in range(max(1, n0 - n_hyp), n0 + n_hyp + 1):
        pos, _ = descend(logp, n, device)
        for p in pos:
            cuts.add(int(round(p * scale)))
    return sorted(cuts)


def load_colchar(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = ColCharNet(len(ck["chars"]) + 1).to(device)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net
