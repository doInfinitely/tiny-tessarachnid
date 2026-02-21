"""
Training script for ContourOCRNet — V2 model with autoregressive contour
prediction and orientation vectors instead of bounding boxes.

Model: GPT-2 style transformer decoder with ResNet-18 image encoder.
At each step the model predicts the next contour point (x, y), an orientation
unit vector (dx, dy), and a class token. Contour closure is determined by
thresholding the distance between the current point and the start point.

Dataset: ContourSequenceDataset produces full contour-point sequences per
(page, level, parent) using teacher forcing.
"""

import argparse
import math
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import (
    ConcatDataset, DataLoader, Dataset, WeightedRandomSampler, random_split,
)

from generate_training_data import (
    CLASS_NONE,
    CLASS_PAGE,
    CLASS_PARAGRAPH,
    CLASS_LINE,
    CLASS_WORD,
    CHAR_CLASS_OFFSET,
    NUM_CLASSES,
    RETINA_SIZE,
    PREV_CONTOUR_NONE,
    SyntheticPage,
    discover_fonts,
    scale_and_pad,
    contour_to_retina,
    contour_orientations,
    contour_from_mask,
    char_to_class,
    class_to_label,
)
from cascade_transforms import (
    forward_cascade_step,
    forward_contour,
)
from train_02 import _remap_old_backbone_keys
from annotate_real import AnnotatedPage, load_all_annotations


# ---------------------------------------------------------------------------
# 1. TransformerBlock (same as train_03)
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """GPT-2 style pre-norm transformer block."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        h = self.ln1(x)
        attn_out, _ = self.attn(
            h, h, h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# 2. ContourOCRNet
# ---------------------------------------------------------------------------
class ContourOCRNet(nn.Module):
    """GPT-2 decoder that predicts contour points + orientation + class."""

    def __init__(self, d_model=128, n_heads=4, n_layers=4, d_ff=512,
                 max_seq_len=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # --- Image encoder (ResNet-18 backbone) ---
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # Image projection: 512 → d_model
        self.img_proj = nn.Linear(512, d_model)

        # --- Contour token embedding ---
        # Input per step: (x, y, dx, dy, class_id) = 5 dims
        self.det_proj = nn.Linear(5, d_model)
        # Start-point conditioning: (x, y) = 2 dims
        self.start_proj = nn.Linear(2, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.level_emb = nn.Embedding(5, d_model)  # 0=page, 1=para, 2=line, 3=word, 4=char

        # --- GPT-2 transformer ---
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        # --- Output heads ---
        self.point_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )
        self.orientation_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.class_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES),
        )

    def _encode_image(self, img):
        x = (img - self.img_mean) / self.img_std
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.img_proj(x)

    def forward(self, img, prev_seq, start_point, level_ids,
                padding_mask=None):
        """
        Args:
            img:          (B, 3, 1024, 1024) input image
            prev_seq:     (B, S, 5) contour sequence (x, y, dx, dy, class_id)
            start_point:  (B, 2) first contour point (x, y) for closure context
            level_ids:    (B,) level index
            padding_mask: (B, S) bool — True where padded

        Returns:
            point_pred:       (B, S, 2) in retina coords
            orient_pred:      (B, S, 2) unit tangent vectors
            class_pred:       (B, S, NUM_CLASSES) logits
        """
        B, S, _ = prev_seq.shape

        img_feat = self._encode_image(img)  # (B, d_model)

        # Normalize prev_seq
        prev_norm = prev_seq.clone()
        prev_norm[:, :, :2] = prev_norm[:, :, :2] / RETINA_SIZE  # x, y
        # dx, dy already in [-1, 1]
        prev_norm[:, :, 4] = prev_norm[:, :, 4] / NUM_CLASSES  # class_id

        det_emb = self.det_proj(prev_norm)  # (B, S, d_model)

        # Add image features
        det_emb = det_emb + img_feat.unsqueeze(1)

        # Add start-point conditioning
        start_norm = start_point / RETINA_SIZE  # (B, 2)
        start_emb = self.start_proj(start_norm)  # (B, d_model)
        det_emb = det_emb + start_emb.unsqueeze(1)

        # Positional embeddings
        positions = torch.arange(S, device=prev_seq.device)
        det_emb = det_emb + self.pos_emb(positions).unsqueeze(0)

        # Level embeddings
        det_emb = det_emb + self.level_emb(level_ids).unsqueeze(1)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(S, S, device=prev_seq.device, dtype=torch.bool),
            diagonal=1,
        )

        x = det_emb
        for block in self.blocks:
            x = block(x, attn_mask=causal_mask, key_padding_mask=padding_mask)
        x = self.ln_f(x)

        # Output heads
        point_pred = self.point_head(x) * RETINA_SIZE  # (B, S, 2)

        orient_raw = self.orientation_head(x)  # (B, S, 2)
        orient_norm = torch.nn.functional.normalize(orient_raw, dim=-1)

        class_pred = self.class_head(x)  # (B, S, NUM_CLASSES)

        return point_pred, orient_norm, class_pred


# ---------------------------------------------------------------------------
# 3. ContourSequenceDataset
# ---------------------------------------------------------------------------
class ContourSequenceDataset(Dataset):
    """Produces contour-point sequences for training ContourOCRNet.

    Each sample is a full contour sequence for one element (paragraph, line,
    word, or character). The sequence is:
      [PREV_CONTOUR_NONE, pt0, pt1, ..., ptN, NONE_terminator]
    where each step is (x, y, dx, dy, class_id).
    """

    def __init__(self, pages, max_seq_len=256):
        self.max_seq_len = max_seq_len
        self.pages = pages
        self.index = []
        self._build_index()

    def _build_index(self):
        for pi, page in enumerate(self.pages):
            self.index.append((pi, "page", ()))
            self.index.append((pi, "para", ()))
            for pai, para in enumerate(page.paragraphs):
                self.index.append((pi, "line", (pai,)))
                for li, line in enumerate(para["lines"]):
                    self.index.append((pi, "word", (pai, li)))
                    for wi, word in enumerate(line["words"]):
                        self.index.append((pi, "char", (pai, li, wi)))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        pi, level, parent_ids = self.index[idx]
        page = self.pages[pi]

        if level == "page":
            return self._make_page_sequence(page)
        elif level == "para":
            return self._make_para_sequence(page)
        elif level == "line":
            return self._make_line_sequence(page, parent_ids[0])
        elif level == "word":
            return self._make_word_sequence(page, parent_ids[0], parent_ids[1])
        else:
            return self._make_char_sequence(
                page, parent_ids[0], parent_ids[1], parent_ids[2])

    def _contour_steps(self, contour, class_id, scale, ox, oy):
        """Convert a contour polygon to a list of (x, y, dx, dy, class_id) steps."""
        if not contour:
            return []
        retina_pts = contour_to_retina(contour, scale, ox, oy)
        orientations = contour_orientations(retina_pts)
        steps = []
        for (px, py), (dx, dy) in zip(retina_pts, orientations):
            steps.append((px, py, dx, dy, class_id))
        return steps

    def _contour_steps_from_retina(self, retina_pts, class_id):
        """Convert contour points already in retina coords to (x, y, dx, dy, class_id) steps."""
        if not retina_pts:
            return []
        orientations = contour_orientations(retina_pts)
        steps = []
        for (px, py), (dx, dy) in zip(retina_pts, orientations):
            steps.append((px, py, dx, dy, class_id))
        return steps

    def _build_sequence(self, img_t, elements, level_id):
        """Build prev/target sequences from a list of element contour steps.

        elements: list of list-of-steps (one per detected item)
        Each sequence: [NONE, *contour_pts_for_elem1, *contour_pts_for_elem2, ..., NONE_end]
        """
        prev_list = [PREV_CONTOUR_NONE]
        target_list = []

        for elem_steps in elements:
            for step in elem_steps:
                target_list.append(step)
                prev_list.append(step)

        # Terminator
        target_list.append((0, 0, 0.0, 0.0, CLASS_NONE))
        prev_list = prev_list[:len(target_list)]

        S = min(len(prev_list), self.max_seq_len)
        prev_list = prev_list[:S]
        target_list = target_list[:S]

        # Start point: first non-NONE target point (first contour point)
        start_point = (0.0, 0.0)
        for t in target_list:
            if t[4] != CLASS_NONE:
                start_point = (float(t[0]), float(t[1]))
                break

        prev_seq = torch.tensor(prev_list, dtype=torch.float32)
        target_seq = torch.tensor(target_list, dtype=torch.float32)
        start_pt = torch.tensor(start_point, dtype=torch.float32)
        level_t = torch.tensor(level_id, dtype=torch.long)

        return img_t, prev_seq, target_seq, start_pt, level_t, S

    def _make_page_sequence(self, page):
        retina, scale, ox, oy = scale_and_pad(page.image, page.bg_color)
        img_t = torch.from_numpy(np.array(retina)).permute(2, 0, 1).float() / 255.0

        elements = []
        contour = getattr(page, "page_contour", [])
        if contour:
            # Contour is mask-local; shift to page coords for page-level retina
            bx1, by1 = page.page_bbox[0], page.page_bbox[1]
            page_contour = [(x + bx1, y + by1) for x, y in contour]
            steps = self._contour_steps(page_contour, CLASS_PAGE, scale, ox, oy)
            if steps:
                elements.append(steps)

        return self._build_sequence(img_t, elements, level_id=0)

    def _make_para_sequence(self, page):
        retina, scale, ox, oy = scale_and_pad(page.image, page.bg_color)
        img_t = torch.from_numpy(np.array(retina)).permute(2, 0, 1).float() / 255.0

        elements = []
        for para in page.paragraphs:
            contour = para.get("contour", [])
            if contour:
                # Contour is mask-local; shift to page coords for page-level retina
                px1, py1 = para["bbox"][0], para["bbox"][1]
                page_contour = [(x + px1, y + py1) for x, y in contour]
                steps = self._contour_steps(page_contour, CLASS_PARAGRAPH, scale, ox, oy)
                if steps:
                    elements.append(steps)

        return self._build_sequence(img_t, elements, level_id=1)

    def _make_line_sequence(self, page, para_idx):
        para = page.paragraphs[para_idx]
        # Build paragraph contour in page-global coords
        para_contour_global = para.get("contour", [])
        if para_contour_global:
            px1g, py1g = para["bbox"][0], para["bbox"][1]
            para_contour_global = [(x + px1g, y + py1g) for x, y in para_contour_global]
        else:
            # Fallback: use bbox as a rectangle contour
            px1, py1, px2, py2 = para["bbox"]
            para_contour_global = [(px1, py1), (px2, py1), (px2, py2), (px1, py2)]

        # Forward cascade: crop paragraph from page, rotate upright, scale+pad
        retina, xf = forward_cascade_step(page.image, para_contour_global, page.bg_color)
        img_t = torch.from_numpy(np.array(retina)).permute(2, 0, 1).float() / 255.0

        elements = []
        px1, py1 = para["bbox"][0], para["bbox"][1]
        for line in para["lines"]:
            contour = line.get("contour", [])
            if not contour:
                continue
            # contour is mask-local (relative to line's bbox);
            # offset to page-global coords
            lx1, ly1 = line["bbox"][0], line["bbox"][1]
            global_contour = [(x + lx1, y + ly1) for x, y in contour]
            # Map through forward transform to retina coords
            retina_contour = forward_contour(global_contour, xf)
            steps = self._contour_steps_from_retina(retina_contour, CLASS_LINE)
            if steps:
                elements.append(steps)

        return self._build_sequence(img_t, elements, level_id=2)

    def _make_word_sequence(self, page, para_idx, line_idx):
        para = page.paragraphs[para_idx]
        line = para["lines"][line_idx]
        # Build line contour in page-global coords
        line_contour_global = line.get("contour", [])
        if line_contour_global:
            lx1g, ly1g = line["bbox"][0], line["bbox"][1]
            line_contour_global = [(x + lx1g, y + ly1g) for x, y in line_contour_global]
        else:
            lx1, ly1, lx2, ly2 = line["bbox"]
            line_contour_global = [(lx1, ly1), (lx2, ly1), (lx2, ly2), (lx1, ly2)]

        # Forward cascade: crop line from page, rotate upright, scale+pad
        retina, xf = forward_cascade_step(page.image, line_contour_global, page.bg_color)
        img_t = torch.from_numpy(np.array(retina)).permute(2, 0, 1).float() / 255.0

        elements = []
        for word in line.get("words", []):
            contour = word.get("contour", [])
            if not contour:
                continue
            # contour is mask-local (relative to word's bbox);
            # offset to page-global coords
            wx1, wy1 = word["bbox"][0], word["bbox"][1]
            global_contour = [(x + wx1, y + wy1) for x, y in contour]
            # Map through forward transform to retina coords
            retina_contour = forward_contour(global_contour, xf)
            steps = self._contour_steps_from_retina(retina_contour, CLASS_WORD)
            if steps:
                elements.append(steps)

        return self._build_sequence(img_t, elements, level_id=3)

    def _make_char_sequence(self, page, para_idx, line_idx, word_idx):
        para = page.paragraphs[para_idx]
        word = para["lines"][line_idx]["words"][word_idx]
        # Build word contour in page-global coords
        word_contour_global = word.get("contour", [])
        if word_contour_global:
            wx1g, wy1g = word["bbox"][0], word["bbox"][1]
            word_contour_global = [(x + wx1g, y + wy1g) for x, y in word_contour_global]
        else:
            wx1, wy1, wx2, wy2 = word["bbox"]
            word_contour_global = [(wx1, wy1), (wx2, wy1), (wx2, wy2), (wx1, wy2)]

        # Forward cascade: crop word from page, rotate upright, scale+pad
        retina, xf = forward_cascade_step(page.image, word_contour_global, page.bg_color)
        img_t = torch.from_numpy(np.array(retina)).permute(2, 0, 1).float() / 255.0

        elements = []
        for ch in word.get("characters", []):
            contour = ch.get("contour", [])
            if not contour:
                continue
            # contour is mask-local (relative to char's bbox);
            # offset to page-global coords
            chx1, chy1 = ch["bbox"][0], ch["bbox"][1]
            global_contour = [(x + chx1, y + chy1) for x, y in contour]
            # Map through forward transform to retina coords
            class_id = char_to_class(ch["char"])
            retina_contour = forward_contour(global_contour, xf)
            steps = self._contour_steps_from_retina(retina_contour, class_id)
            if steps:
                elements.append(steps)

        return self._build_sequence(img_t, elements, level_id=4)


def contour_collate_fn(batch):
    """Pad sequences to max length in batch."""
    imgs, prevs, targets, starts, levels, lengths = zip(*batch)

    B = len(imgs)
    max_len = max(lengths)

    img_batch = torch.stack(imgs)
    level_batch = torch.stack(levels)
    start_batch = torch.stack(starts)

    prev_batch = torch.zeros(B, max_len, 5, dtype=torch.float32)
    target_batch = torch.zeros(B, max_len, 5, dtype=torch.float32)
    padding_mask = torch.ones(B, max_len, dtype=torch.bool)

    for i in range(B):
        S = lengths[i]
        prev_batch[i, :S] = prevs[i]
        target_batch[i, :S] = targets[i]
        padding_mask[i, :S] = False

    return img_batch, prev_batch, target_batch, start_batch, level_batch, padding_mask


# ---------------------------------------------------------------------------
# 4. ContourLoss
# ---------------------------------------------------------------------------
class ContourLoss(nn.Module):
    """Combined loss for contour prediction."""

    def __init__(self, point_weight=1.0, orient_weight=0.5, class_weight=1.0,
                 intersect_weight=0.1):
        super().__init__()
        self.point_weight = point_weight
        self.orient_weight = orient_weight
        self.class_weight = class_weight
        self.intersect_weight = intersect_weight
        self.cls_loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.point_loss_fn = nn.SmoothL1Loss(reduction="none")

    def forward(self, point_pred, orient_pred, class_pred,
                target_seq, padding_mask):
        """
        Args:
            point_pred:   (B, S, 2)
            orient_pred:  (B, S, 2) L2-normalized
            class_pred:   (B, S, NUM_CLASSES)
            target_seq:   (B, S, 5) — (x, y, dx, dy, class_id)
            padding_mask: (B, S) bool — True where padded
        """
        B, S, _ = class_pred.shape
        target_class = target_seq[:, :, 4].long()
        target_point = target_seq[:, :, :2]
        target_orient = target_seq[:, :, 2:4]
        valid = ~padding_mask

        # Classification loss
        cls_loss = self.cls_loss_fn(
            class_pred.reshape(B * S, -1),
            target_class.reshape(B * S),
        ).reshape(B, S)
        cls_loss = (cls_loss * valid.float()).sum() / valid.float().sum().clamp(min=1)

        # Point regression loss (non-NONE positions)
        non_none = valid & (target_class != CLASS_NONE)
        if non_none.any():
            pred_norm = point_pred / RETINA_SIZE
            tgt_norm = target_point / RETINA_SIZE
            pt_loss = self.point_loss_fn(pred_norm, tgt_norm).mean(dim=-1)
            pt_loss = (pt_loss * non_none.float()).sum() / non_none.float().sum().clamp(min=1)
        else:
            pt_loss = torch.tensor(0.0, device=point_pred.device)

        # Orientation loss: 1 - cosine_similarity
        if non_none.any():
            cos_sim = (orient_pred * target_orient).sum(dim=-1)  # (B, S)
            orient_loss = 1.0 - cos_sim
            orient_loss = (orient_loss * non_none.float()).sum() / non_none.float().sum().clamp(min=1)
        else:
            orient_loss = torch.tensor(0.0, device=point_pred.device)

        # Intersection loss (soft differentiable proxy)
        intersect_loss = self._intersection_loss(point_pred, non_none)

        total = (self.point_weight * pt_loss
                 + self.orient_weight * orient_loss
                 + self.class_weight * cls_loss
                 + self.intersect_weight * intersect_loss)
        return total, pt_loss, orient_loss, cls_loss, intersect_loss

    def _intersection_loss(self, point_pred, non_none):
        """Soft intersection penalty for predicted contour segments.

        For each sample in the batch, checks non-adjacent segment pairs
        for intersection using the signed-area cross-product method.
        Vectorized: all pair-wise cross products computed in a single pass.
        """
        B, S, _ = point_pred.shape
        total_loss = torch.tensor(0.0, device=point_pred.device)
        count = 0

        for b in range(B):
            valid_idx = non_none[b].nonzero(as_tuple=True)[0]
            n = len(valid_idx)
            if n < 4:
                continue
            pts = point_pred[b, valid_idx]  # (n, 2)
            n_seg = n - 1

            # Segment endpoints: starts[i]->ends[i]
            starts = pts[:-1]  # (n_seg, 2)
            ends = pts[1:]     # (n_seg, 2)

            # All non-adjacent pairs (i, j) where j >= i + 2
            ii, jj = torch.triu_indices(n_seg, n_seg, offset=2,
                                        device=pts.device)

            # Exclude first-last pair (closed contour shares endpoint)
            keep = ~((ii == 0) & (jj == n_seg - 1))
            ii = ii[keep]
            jj = jj[keep]

            if len(ii) == 0:
                continue

            # Gather segment endpoints for all pairs: (K, 2)
            p1 = starts[ii]
            p2 = ends[ii]
            p3 = starts[jj]
            p4 = ends[jj]

            # Vectorized 2D cross: (b-a) x (c-a)
            def _cross_batch(a, b, c):
                return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) \
                     - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])

            d1 = _cross_batch(p3, p4, p1)
            d2 = _cross_batch(p3, p4, p2)
            d3 = _cross_batch(p1, p2, p3)
            d4 = _cross_batch(p1, p2, p4)

            s1 = torch.sigmoid(-d1 * d2 * 0.01)
            s2 = torch.sigmoid(-d3 * d4 * 0.01)

            total_loss = total_loss + (s1 * s2).sum()
            count += len(ii)

        if count > 0:
            total_loss = total_loss / count
        return total_loss


# ---------------------------------------------------------------------------
# 5. load_backbone_from_v02
# ---------------------------------------------------------------------------
def load_backbone_from_v02(model, checkpoint_path, device):
    """Load ResNet-18 backbone weights from model_02.pth into ContourOCRNet."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    ckpt = _remap_old_backbone_keys(ckpt)

    backbone_prefixes = ("stem.", "layer1.", "layer2.", "layer3.", "layer4.")
    buffer_keys = ("img_mean", "img_std")
    backbone_state = {}
    for k, v in ckpt.items():
        if any(k.startswith(p) for p in backbone_prefixes) or k in buffer_keys:
            backbone_state[k] = v

    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    loaded = len(backbone_state) - len(unexpected)
    print(f"Loaded {loaded} backbone params from {checkpoint_path}")
    if unexpected:
        print(f"  Unexpected keys (ignored): {unexpected}")
    return model


# ---------------------------------------------------------------------------
# 5b. load_weights_from_v03
# ---------------------------------------------------------------------------
def load_weights_from_v03(model, checkpoint_path, device):
    """Transfer all structurally compatible weights from model_03.pth (V1.5).

    Exact transfers:
        - ResNet-18 backbone (stem.*, layer1-4.*)
        - img_proj (512 -> d_model)
        - level_emb (4 x d_model)
        - Transformer blocks 0-3 (all ln, attn, ffn params)
        - ln_f (final LayerNorm)
        - class_head (d_model -> 64 -> NUM_CLASSES)

    Partial transfers:
        - pos_emb: V1.5 is (128, d_model), V2 is (256, d_model) — copy first 128 rows
        - det_proj: V1.5 is (d_model, 6), V2 is (d_model, 5) — copy first 5 input cols

    Skipped (new in V2 / incompatible):
        - start_proj, point_head, orientation_head
        - bbox_head, handwritten_head (V1.5-only heads)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    ckpt = _remap_old_backbone_keys(ckpt)

    model_state = model.state_dict()
    transferred = []
    partial = []
    skipped = []

    for key, src_tensor in ckpt.items():
        if key not in model_state:
            # V1.5-only keys (bbox_head, handwritten_head, etc.)
            skipped.append((key, "not in V2 model"))
            continue

        dst_tensor = model_state[key]

        if src_tensor.shape == dst_tensor.shape:
            # Exact match — copy directly
            model_state[key] = src_tensor
            transferred.append(key)

        elif key == "pos_emb.weight":
            # V1.5: (128, d_model) -> V2: (256, d_model) — copy first 128 rows
            n_copy = min(src_tensor.shape[0], dst_tensor.shape[0])
            model_state[key][:n_copy] = src_tensor[:n_copy]
            partial.append((key, f"copied {n_copy}/{dst_tensor.shape[0]} rows"))

        elif key == "det_proj.weight":
            # V1.5: (d_model, 6) -> V2: (d_model, 5) — copy first 5 input cols
            n_copy = min(src_tensor.shape[1], dst_tensor.shape[1])
            model_state[key][:, :n_copy] = src_tensor[:, :n_copy]
            partial.append((key, f"copied {n_copy}/{dst_tensor.shape[1]} input cols"))

        elif key == "det_proj.bias":
            # Bias shape is (d_model,) — same in both, but guard against mismatch
            if src_tensor.shape == dst_tensor.shape:
                model_state[key] = src_tensor
                transferred.append(key)
            else:
                skipped.append((key, f"shape {src_tensor.shape} vs {dst_tensor.shape}"))

        elif src_tensor.dim() == dst_tensor.dim() and all(
            s <= d for s, d in zip(src_tensor.shape, dst_tensor.shape)
        ):
            # Source fits inside destination on every dim — copy the overlap
            slices = tuple(slice(0, s) for s in src_tensor.shape)
            model_state[key][slices] = src_tensor
            partial.append((key, f"copied {list(src_tensor.shape)} into {list(dst_tensor.shape)}"))

        elif src_tensor.dim() == dst_tensor.dim() and all(
            s >= d for s, d in zip(src_tensor.shape, dst_tensor.shape)
        ):
            # Source is larger — take the slice that fits
            slices = tuple(slice(0, d) for d in dst_tensor.shape)
            model_state[key] = src_tensor[slices]
            partial.append((key, f"sliced {list(src_tensor.shape)} to {list(dst_tensor.shape)}"))

        else:
            skipped.append((key, f"shape {src_tensor.shape} vs {dst_tensor.shape}"))

    model.load_state_dict(model_state)

    print(f"=== Weight transfer from {checkpoint_path} ===")
    print(f"  Exact transfers: {len(transferred)} params")
    print(f"  Partial transfers: {len(partial)}")
    for key, note in partial:
        print(f"    {key}: {note}")
    print(f"  Skipped: {len(skipped)}")
    for key, note in skipped:
        print(f"    {key}: {note}")

    return model


# ---------------------------------------------------------------------------
# 6. Training loop
# ---------------------------------------------------------------------------
def fit(epochs, model, loss_fn, opt, train_dl, valid_dl, device, save_path,
        patience=15, grad_clip=1.0, freeze_backbone_epochs=0, scheduler=None,
        warmup_epochs=0, on_save=None):
    best_val_loss = float("inf")
    epochs_no_improve = 0

    if scheduler is None:
        if warmup_epochs > 0 and epochs > warmup_epochs:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=0.01, total_iters=warmup_epochs,
            )
            cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=epochs - warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                opt, schedulers=[warmup_sched, cosine_sched],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.5, patience=5,
            )

    use_cosine = not isinstance(
        scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau,
    )

    for epoch in range(epochs):
        if epoch == freeze_backbone_epochs and freeze_backbone_epochs > 0:
            print(f"  -> Unfreezing backbone at epoch {epoch}")
            for param in model.parameters():
                param.requires_grad = True

        # -- Train --
        model.train()
        t_total, t_pt, t_ori, t_cls, t_int, t_n = 0, 0, 0, 0, 0, 0
        for img, prev_seq, target_seq, start_pt, level_ids, pad_mask in train_dl:
            img = img.to(device)
            prev_seq = prev_seq.to(device)
            target_seq = target_seq.to(device)
            start_pt = start_pt.to(device)
            level_ids = level_ids.to(device)
            pad_mask = pad_mask.to(device)

            point_pred, orient_pred, class_pred = model(
                img, prev_seq, start_pt, level_ids, pad_mask,
            )
            total, pt_l, ori_l, cls_l, int_l = loss_fn(
                point_pred, orient_pred, class_pred, target_seq, pad_mask,
            )

            total.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            opt.zero_grad()

            n = img.size(0)
            t_total += total.item() * n
            t_pt += pt_l.item() * n
            t_ori += ori_l.item() * n
            t_cls += cls_l.item() * n
            t_int += int_l.item() * n
            t_n += n

        t_total /= t_n
        t_pt /= t_n
        t_ori /= t_n
        t_cls /= t_n
        t_int /= t_n

        # -- Validate --
        model.eval()
        v_total, v_pt, v_ori, v_cls, v_int, v_n = 0, 0, 0, 0, 0, 0
        with torch.no_grad():
            for img, prev_seq, target_seq, start_pt, level_ids, pad_mask in valid_dl:
                img = img.to(device)
                prev_seq = prev_seq.to(device)
                target_seq = target_seq.to(device)
                start_pt = start_pt.to(device)
                level_ids = level_ids.to(device)
                pad_mask = pad_mask.to(device)

                point_pred, orient_pred, class_pred = model(
                    img, prev_seq, start_pt, level_ids, pad_mask,
                )
                total, pt_l, ori_l, cls_l, int_l = loss_fn(
                    point_pred, orient_pred, class_pred, target_seq, pad_mask,
                )

                n = img.size(0)
                v_total += total.item() * n
                v_pt += pt_l.item() * n
                v_ori += ori_l.item() * n
                v_cls += cls_l.item() * n
                v_int += int_l.item() * n
                v_n += n

        v_total /= v_n
        v_pt /= v_n
        v_ori /= v_n
        v_cls /= v_n
        v_int /= v_n

        if use_cosine:
            scheduler.step()
        else:
            scheduler.step(v_total)
        cur_lr = opt.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d} | "
            f"train={t_total:.4f} (pt={t_pt:.4f} ori={t_ori:.4f} cls={t_cls:.4f} int={t_int:.4f}) | "
            f"val={v_total:.4f} (pt={v_pt:.4f} ori={v_ori:.4f} cls={v_cls:.4f} int={v_int:.4f}) | "
            f"lr={cur_lr:.2e}",
            flush=True,
        )

        if v_total < best_val_loss:
            best_val_loss = v_total
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> saved best model ({save_path})")
            if on_save:
                on_save(save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"  Early stopping: no improvement for {patience} epochs "
                    f"(best={best_val_loss:.4f})"
                )
                break


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from PIL import Image

    parser = argparse.ArgumentParser(description="Train ContourOCRNet (V2)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--pages", type=int, default=20)
    parser.add_argument("--save-path", type=str, default="model_04.pth")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--page-width", type=int, default=2048)
    parser.add_argument("--page-height", type=int, default=2800)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rotate", action="store_true", default=True,
                        help="Enable rotated paragraph generation")
    parser.add_argument("--no-rotate", action="store_false", dest="rotate")

    # Loss weights
    parser.add_argument("--point-weight", type=float, default=1.0)
    parser.add_argument("--orient-weight", type=float, default=0.5)
    parser.add_argument("--class-weight", type=float, default=1.0)
    parser.add_argument("--intersect-weight", type=float, default=0.1)

    # Model architecture
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training phases
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to model_02.pth for backbone-only init")
    parser.add_argument("--pretrained-v03", type=str, default=None,
                        help="Path to model_03.pth for full weight transfer")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="Linear LR warmup epochs before cosine decay (0=use ReduceLROnPlateau)")

    # Real data
    parser.add_argument("--real-data", type=str, default=None,
                        help="Directory of annotated real images (from annotate_real.py)")
    parser.add_argument("--real-ratio", type=float, default=0.3,
                        help="Fraction of training samples from real data (default: 0.3)")

    # Resume / fine-tune
    parser.add_argument("--resume", type=str, default=None,
                        help="Load full model checkpoint (takes precedence over --pretrained)")
    parser.add_argument("--finetune-lr-factor", type=float, default=0.1,
                        help="LR multiplier when resuming with real data (default: 0.1)")

    # Inference mode
    parser.add_argument("--infer", type=str, default=None,
                        help="Run inference on image instead of training")
    parser.add_argument("--output", type=str, default="infer_04_output.png")
    parser.add_argument("--deploy", action="store_true",
                        help="Deploy weights to glyph-daemon on each checkpoint")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ContourOCRNet(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    ).to(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt)
        print(f"Resumed full checkpoint from {args.resume}")
    elif args.pretrained_v03:
        load_weights_from_v03(model, args.pretrained_v03, device)
    elif args.pretrained:
        load_backbone_from_v02(model, args.pretrained, device)

    # --- Inference mode ---
    if args.infer:
        if not args.resume:
            try:
                ckpt = torch.load(args.save_path, map_location=device, weights_only=True)
                model.load_state_dict(ckpt)
                print(f"Loaded model from {args.save_path}")
            except FileNotFoundError:
                print(f"Warning: {args.save_path} not found, using initialized weights")

        from infer_04 import run_inference
        run_inference(model, args.infer, args.output, device)
        sys.exit(0)

    # --- Training mode ---
    fonts = discover_fonts()
    print(f"Discovered {len(fonts)} fonts")
    if not fonts:
        print("No fonts found! Place .ttf/.otf/.ttc files in fonts/ directory.")
        sys.exit(1)

    print(f"Generating {args.pages} synthetic pages (rotate={args.rotate})...")
    synth_pages = []
    for i in range(args.pages):
        synth_pages.append(SyntheticPage(fonts, args.page_width, args.page_height,
                                         rotate_paragraphs=args.rotate))
        total_chars = sum(
            len(word["characters"])
            for para in synth_pages[-1].paragraphs
            for line in para["lines"]
            for word in line["words"]
        )
        print(f"  Page {i}: {len(synth_pages[-1].paragraphs)} paragraphs, {total_chars} chars")

    synth_ds = ContourSequenceDataset(pages=synth_pages, max_seq_len=args.max_seq_len)
    print(f"Synthetic dataset: {len(synth_ds)} sequences")

    # Load real annotated pages if provided
    real_ds = None
    if args.real_data:
        print(f"Loading real annotated pages from {args.real_data}...")
        real_pages = load_all_annotations(args.real_data)
        print(f"Loaded {len(real_pages)} real pages")
        if real_pages:
            real_ds = ContourSequenceDataset(pages=real_pages, max_seq_len=args.max_seq_len)
            print(f"Real dataset: {len(real_ds)} sequences")

    # Combine datasets
    if real_ds is not None and len(real_ds) > 0:
        combined_ds = ConcatDataset([synth_ds, real_ds])
        n_synth = len(synth_ds)
        n_real = len(real_ds)
        n_total = n_synth + n_real
        print(f"Combined dataset: {n_total} sequences ({n_synth} synth + {n_real} real)")

        # Weighted sampler: real_ratio controls effective mix
        r = args.real_ratio
        w_real = r / n_real
        w_synth = (1.0 - r) / n_synth
        weights = [w_synth] * n_synth + [w_real] * n_real

        val_size = max(1, int(n_total * args.val_split))
        train_size = n_total - val_size
        train_ds, val_ds = random_split(combined_ds, [train_size, val_size])

        # Build per-sample weights for the train subset
        train_weights = [weights[idx] for idx in train_ds.indices]
        sampler = WeightedRandomSampler(train_weights, num_samples=len(train_ds),
                                        replacement=True)
        print(f"Train: {train_size}, Val: {val_size}  (real_ratio={r:.0%})")
    else:
        val_size = max(1, int(len(synth_ds) * args.val_split))
        train_size = len(synth_ds) - val_size
        train_ds, val_ds = random_split(synth_ds, [train_size, val_size])
        sampler = None
        print(f"Train: {train_size}, Val: {val_size}  (synthetic only)")

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=2, pin_memory=True,
        collate_fn=contour_collate_fn,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
        collate_fn=contour_collate_fn,
    )

    # Freeze backbone for initial phase (skip when resuming)
    if args.freeze_backbone_epochs > 0 and not args.resume:
        print(f"Freezing backbone for {args.freeze_backbone_epochs} epochs")
        for name, param in model.named_parameters():
            if any(name.startswith(p) for p in
                   ("stem.", "layer1.", "layer2.", "layer3.", "layer4.")):
                param.requires_grad = False

    # LR scaling: lower LR when fine-tuning a resumed model on real data
    base_lr = args.lr
    if args.resume and args.real_data:
        base_lr = args.lr * args.finetune_lr_factor
        print(f"Fine-tune LR: {args.lr} * {args.finetune_lr_factor} = {base_lr:.2e}")

    # Differential learning rates
    backbone_params = []
    projection_params = []
    transformer_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(name.startswith(p) for p in
               ("stem.", "layer1.", "layer2.", "layer3.", "layer4.")):
            backbone_params.append(param)
        elif any(name.startswith(p) for p in
                 ("img_proj.", "det_proj.", "start_proj.", "pos_emb.", "level_emb.")):
            projection_params.append(param)
        else:
            transformer_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": base_lr * 0.01},
        {"params": projection_params, "lr": base_lr * 0.5},
        {"params": transformer_params, "lr": base_lr},
    ]
    param_groups = [g for g in param_groups if g["params"]]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-2)

    loss_fn = ContourLoss(
        point_weight=args.point_weight,
        orient_weight=args.orient_weight,
        class_weight=args.class_weight,
        intersect_weight=args.intersect_weight,
    ).to(device)

    # Deploy callback
    deploy_fn = None
    if args.deploy:
        from dotenv import load_dotenv
        load_dotenv()
        from deploy_weights import deploy
        deploy_fn = deploy
        print("Deploy to glyph-daemon: enabled")

    print(f"\n=== Training ContourOCRNet ({args.epochs} epochs) ===")
    fit(
        args.epochs, model, loss_fn, optimizer, train_dl, val_dl,
        device, args.save_path,
        patience=args.patience,
        grad_clip=args.grad_clip,
        freeze_backbone_epochs=(
            args.freeze_backbone_epochs if not args.resume else 0
        ),
        warmup_epochs=args.warmup_epochs,
        on_save=deploy_fn,
    )
    print("Done.")
