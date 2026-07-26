"""detect_characters_raw but emitting top-K classifications per window.

Each window in the sliding-window pool produces up to K detection records
(one per candidate char), all sharing the same bbox/pixel_std/edge_density
but with different (char, confidence) values. The downstream beam decoder
sees these as ordinary detection dicts and naturally explores the K
candidates per spatial position — including the correct char that the
argmax would have thrown away.
"""
import numpy as np
import torch
from PIL import Image

from glyph_faerie.detection.detector import (
    _extract_windows, _reject_blank_windows,
)


@torch.no_grad()
def _topk_per_window(model, feats, block_probs, K, dedup=True):
    """Per-sample top-K (char, joint_prob)."""
    out = []
    n_blocks_consider = min(8, block_probs.shape[1])
    for i in range(feats.size(0)):
        scores = {}
        tb = block_probs[i].topk(n_blocks_consider)
        for bp, bi in zip(tb.values, tb.indices):
            bi_int = int(bi.item())
            key = str(bi_int)
            if key not in model.char_heads:
                continue
            cl = model.char_heads[key](feats[i:i + 1])
            cp = torch.softmax(cl, dim=1)[0]
            tk = cp.topk(min(K, cp.shape[0]))
            chars = model.block_to_chars.get(bi_int, [])
            for v, idx in zip(tk.values, tk.indices):
                ii = int(idx.item())
                if ii >= len(chars):
                    continue
                ch = chars[ii]
                joint = float(bp.item() * v.item())
                if not dedup or ch not in scores or joint > scores[ch]:
                    scores[ch] = joint
        out.append(sorted(scores.items(), key=lambda kv: -kv[1])[:K])
    return out


def detect_characters_topk(page_image, detector, config, K=20, conf_floor=0.0):
    """Sliding-window detection emitting K (char, conf) detections per window."""
    windows, window_bboxes, bg_colors = _extract_windows(page_image, config)
    if not windows:
        return []

    input_sz = detector.input_size
    retina_tensors = []
    pixel_stats = []
    for win, bg in zip(windows, bg_colors):
        w, h = win.size
        sc = min(input_sz / w, input_sz / h)
        nw, nh = max(1, int(w * sc)), max(1, int(h * sc))
        resized = win.resize((nw, nh), Image.LANCZOS)
        canvas = Image.new("RGB", (input_sz, input_sz), bg)
        canvas.paste(resized, ((input_sz - nw) // 2, (input_sz - nh) // 2))
        t = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float() / 255.0
        retina_tensors.append(t)
        gray = t.mean(dim=0)
        dx = (gray[:, 1:] - gray[:, :-1]).abs()
        dy = (gray[1:, :] - gray[:-1, :]).abs()
        edge = ((dx > 0.05).sum() + (dy > 0.05).sum()).item() / gray.numel()
        pixel_stats.append((round(t.std().item(), 4), round(edge, 4)))

    model = detector.model
    device = detector.device
    detections = []
    bs = config.max_batch_size
    n = len(retina_tensors)
    for i in range(0, n, bs):
        batch = torch.stack(retina_tensors[i:i + bs]).to(device)
        with torch.amp.autocast("cuda", enabled=(
            device == "cuda" or (hasattr(device, "type") and
                                  device.type == "cuda"))):
            feats = model.extract_features(batch)
            block_logits = model.block_head(feats)
            block_probs = torch.softmax(block_logits, dim=1)
            topk = _topk_per_window(model, feats, block_probs, K)
        for j, cands in enumerate(topk):
            idx = i + j
            bbox = window_bboxes[idx]
            pstd, pedge = pixel_stats[idx]
            for ch, conf in cands:
                if conf < conf_floor:
                    continue
                detections.append({
                    "bbox": [int(round(bbox[0])), int(round(bbox[1])),
                             int(round(bbox[2])), int(round(bbox[3]))],
                    "char": ch,
                    "confidence": round(conf, 4),
                    "pixel_std": pstd,
                    "edge_density": pedge,
                })
    return detections
