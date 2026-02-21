"""
Cached encoding utilities for ContourOCRNet inference.

Splits ContourOCRNet.forward() into encode + decode without modifying
the model class.  This avoids redundant ResNet-18 calls during
autoregressive contour tracing (~50x reduction per element).

Usage:
    img_feat = encode_image(model, img_t)   # once per retina image
    for step in range(max_points):
        pts, ori, cls = decode_step(model, img_feat, prev_t, start_pt, level_t)
"""

import torch
import torch.nn.functional as F

from generate_training_data import NUM_CLASSES, RETINA_SIZE


@torch.no_grad()
def encode_image(model, img_t):
    """Run ResNet-18 encoder once, return (B, d_model) feature.

    Args:
        model: ContourOCRNet instance (eval mode).
        img_t: (B, 3, H, W) image tensor on device.

    Returns:
        img_feat: (B, d_model) encoded image features.
    """
    return model._encode_image(img_t)


@torch.no_grad()
def decode_step(model, img_feat, prev_seq, start_point, level_ids,
                padding_mask=None):
    """Run transformer + output heads using a cached image feature.

    Replicates ContourOCRNet.forward() lines 183-225, substituting
    img_feat for model._encode_image(img).

    Args:
        model: ContourOCRNet instance.
        img_feat: (B, d_model) from encode_image().
        prev_seq: (B, S, 5) contour sequence.
        start_point: (B, 2) first contour point.
        level_ids: (B,) level index.
        padding_mask: (B, S) bool, True where padded.

    Returns:
        point_pred: (B, S, 2)
        orient_pred: (B, S, 2)
        class_pred: (B, S, NUM_CLASSES)
    """
    B, S, _ = prev_seq.shape

    prev_norm = prev_seq.clone()
    prev_norm[:, :, :2] = prev_norm[:, :, :2] / RETINA_SIZE
    prev_norm[:, :, 4] = prev_norm[:, :, 4] / NUM_CLASSES

    det_emb = model.det_proj(prev_norm)
    det_emb = det_emb + img_feat.unsqueeze(1)

    start_norm = start_point / RETINA_SIZE
    start_emb = model.start_proj(start_norm)
    det_emb = det_emb + start_emb.unsqueeze(1)

    positions = torch.arange(S, device=prev_seq.device)
    det_emb = det_emb + model.pos_emb(positions).unsqueeze(0)

    det_emb = det_emb + model.level_emb(level_ids).unsqueeze(1)

    causal_mask = torch.triu(
        torch.ones(S, S, device=prev_seq.device, dtype=torch.bool),
        diagonal=1,
    )

    x = det_emb
    for block in model.blocks:
        x = block(x, attn_mask=causal_mask, key_padding_mask=padding_mask)
    x = model.ln_f(x)

    point_pred = model.point_head(x) * RETINA_SIZE
    orient_raw = model.orientation_head(x)
    orient_norm = F.normalize(orient_raw, dim=-1)
    class_pred = model.class_head(x)

    return point_pred, orient_norm, class_pred
