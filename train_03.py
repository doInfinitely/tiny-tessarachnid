"""
Training script for GPT-2 decoder over OCR backbone (RetinaOCRGPT).

Model: GPT-2 style transformer decoder with ResNet-18 image encoder.
The transformer attends to the full detection sequence, enabling language
modeling at the character level and better spatial reasoning.

Dataset: SequenceOCRDataset produces full sequences per (page, level, parent)
instead of individual steps. Supports both synthetic and real annotations.
"""

import argparse
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, random_split

from generate_training_data import (
    CLASS_NONE,
    CLASS_PARAGRAPH,
    CLASS_LINE,
    CLASS_WORD,
    CHAR_CLASS_OFFSET,
    NUM_CLASSES,
    RETINA_SIZE,
    PREV_BBOX_NONE,
    SyntheticPage,
    discover_fonts,
    scale_and_pad,
    bbox_to_retina,
    char_to_class,
    class_to_label,
)
from annotate_real import load_annotations
from train_02 import _remap_old_backbone_keys


# ---------------------------------------------------------------------------
# 1. TransformerBlock
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
        # Pre-norm self-attention with residual
        h = self.ln1(x)
        attn_out, _ = self.attn(
            h, h, h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        x = x + attn_out

        # Pre-norm FFN with residual
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# 2. RetinaOCRGPT
# ---------------------------------------------------------------------------
class RetinaOCRGPT(nn.Module):
    """GPT-2 decoder over a frozen ResNet-18 OCR backbone."""

    def __init__(self, d_model=128, n_heads=4, n_layers=4, d_ff=512,
                 max_seq_len=128, dropout=0.1):
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

        # --- Detection token embedding ---
        self.det_proj = nn.Linear(6, d_model)  # (x1,y1,x2,y2,class_id,is_handwritten)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.level_emb = nn.Embedding(4, d_model)  # 0=para, 1=line, 2=word, 3=char

        # --- GPT-2 transformer ---
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        # --- Output heads ---
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid(),
        )
        self.class_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES),
        )
        self.handwritten_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def _encode_image(self, img):
        """Encode image through frozen/unfrozen ResNet-18 backbone."""
        x = (img - self.img_mean) / self.img_std
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # (B, 512)
        return self.img_proj(x)     # (B, d_model)

    def forward(self, img, prev_seq, level_ids, padding_mask=None):
        """
        Args:
            img:          (B, 3, 1024, 1024) input image
            prev_seq:     (B, S, 6) detection sequence
                          (x1,y1,x2,y2,class_id,is_handwritten)
            level_ids:    (B,) level index (0=para, 1=line, 2=word, 3=char)
            padding_mask: (B, S) bool — True where padded

        Returns:
            bbox_pred:        (B, S, 4) in retina coords
            class_pred:       (B, S, NUM_CLASSES) logits
            handwritten_pred: (B, S, 1) logits (sigmoid → probability)
        """
        B, S, _ = prev_seq.shape

        # Image features: (B, d_model)
        img_feat = self._encode_image(img)

        # Normalize prev_seq: coords / 1024, class / NUM_CLASSES,
        # is_handwritten already 0/1 so pass through
        prev_norm = prev_seq.clone()
        prev_norm[:, :, :4] = prev_norm[:, :, :4] / RETINA_SIZE
        prev_norm[:, :, 4] = prev_norm[:, :, 4] / NUM_CLASSES
        # dim 5 (is_handwritten) is already 0/1

        # Detection token embeddings
        det_emb = self.det_proj(prev_norm)  # (B, S, d_model)

        # Add image features broadcast to all positions
        det_emb = det_emb + img_feat.unsqueeze(1)  # (B, S, d_model)

        # Positional embeddings
        positions = torch.arange(S, device=prev_seq.device)
        det_emb = det_emb + self.pos_emb(positions).unsqueeze(0)  # (B, S, d_model)

        # Level embeddings
        det_emb = det_emb + self.level_emb(level_ids).unsqueeze(1)  # (B, S, d_model)

        # Causal mask: (S, S), True = blocked
        causal_mask = torch.triu(
            torch.ones(S, S, device=prev_seq.device, dtype=torch.bool), diagonal=1,
        )

        # Run transformer blocks
        x = det_emb
        for block in self.blocks:
            x = block(x, attn_mask=causal_mask, key_padding_mask=padding_mask)
        x = self.ln_f(x)  # (B, S, d_model)

        # Output heads
        bbox_pred = self.bbox_head(x) * RETINA_SIZE    # (B, S, 4)
        class_pred = self.class_head(x)                 # (B, S, NUM_CLASSES)
        handwritten_pred = self.handwritten_head(x)     # (B, S, 1)

        return bbox_pred, class_pred, handwritten_pred


# ---------------------------------------------------------------------------
# 3. SequenceOCRDataset
# ---------------------------------------------------------------------------
class SequenceOCRDataset(Dataset):
    """Produces full detection sequences per (page, level, parent).

    Supports both synthetic SyntheticPage objects and real annotation dirs.
    4-level hierarchy: paragraphs → lines → words → characters.
    All targets are 6-dim: (x1, y1, x2, y2, class_id, is_handwritten).
    """

    def __init__(self, pages=None, real_dirs=None, max_seq_len=128):
        self.max_seq_len = max_seq_len
        self.pages = pages or []
        self.real_data = []  # list of (page_image, nested_dict)
        self.index = []

        if real_dirs:
            for d in real_dirs:
                try:
                    page_image, nested = load_annotations(d)
                    self.real_data.append((page_image, nested))
                except Exception as e:
                    print(f"Warning: failed to load {d}: {e}")

        self._build_index()

    def _build_index(self):
        # Synthetic pages
        for pi, page in enumerate(self.pages):
            self.index.append(("synthetic", pi, "para", ()))

            for pai, para in enumerate(page.paragraphs):
                self.index.append(("synthetic", pi, "line", (pai,)))

                for li, line in enumerate(para["lines"]):
                    self.index.append(("synthetic", pi, "word", (pai, li)))

                    for wi, word in enumerate(line["words"]):
                        self.index.append(("synthetic", pi, "char", (pai, li, wi)))

        # Real annotation pages
        for ri, (page_image, nested) in enumerate(self.real_data):
            paragraphs = nested.get("paragraphs", [])
            self.index.append(("real", ri, "para", ()))

            for pai, para in enumerate(paragraphs):
                self.index.append(("real", ri, "line", (pai,)))

                for li, line in enumerate(para.get("lines", [])):
                    self.index.append(("real", ri, "word", (pai, li)))

                    for wi, word in enumerate(line.get("words", [])):
                        self.index.append(("real", ri, "char", (pai, li, wi)))

    def __len__(self):
        return len(self.index)

    def _get_page_data(self, source_type, source_idx):
        """Return (page_image, paragraphs, bg_color) for either source type."""
        if source_type == "synthetic":
            page = self.pages[source_idx]
            return page.image, page.paragraphs, page.bg_color
        else:
            page_image, nested = self.real_data[source_idx]
            paragraphs = nested.get("paragraphs", [])
            from annotate_real import estimate_background_color
            bg_color = estimate_background_color(page_image)
            return page_image, paragraphs, bg_color

    def __getitem__(self, idx):
        source_type, source_idx, level, parent_ids = self.index[idx]
        page_image, paragraphs, bg_color = self._get_page_data(
            source_type, source_idx,
        )

        if level == "para":
            return self._make_para_sequence(
                page_image, paragraphs, bg_color,
            )
        elif level == "line":
            return self._make_line_sequence(
                page_image, paragraphs, bg_color, parent_ids[0],
            )
        elif level == "word":
            return self._make_word_sequence(
                page_image, paragraphs, bg_color,
                parent_ids[0], parent_ids[1],
            )
        else:
            return self._make_char_sequence(
                page_image, paragraphs, bg_color,
                parent_ids[0], parent_ids[1], parent_ids[2],
            )

    def _make_para_sequence(self, page_image, paragraphs, bg_color):
        retina, scale, ox, oy = scale_and_pad(page_image, bg_color)
        img_t = torch.from_numpy(np.array(retina)).permute(2, 0, 1).float() / 255.0

        prev_list = [PREV_BBOX_NONE]
        target_list = []

        for para in paragraphs:
            rb = bbox_to_retina(para["bbox"], scale, ox, oy)
            hw = int(para.get("is_handwritten", False))
            det = (rb[0], rb[1], rb[2], rb[3], CLASS_PARAGRAPH, hw)
            target_list.append(det)
            prev_list.append(det)

        target_list.append((0, 0, 0, 0, CLASS_NONE, 0))
        prev_list = prev_list[:len(target_list)]

        S = min(len(prev_list), self.max_seq_len)
        prev_list = prev_list[:S]
        target_list = target_list[:S]

        prev_seq = torch.tensor(prev_list, dtype=torch.float32)
        target_seq = torch.tensor(target_list, dtype=torch.float32)
        level_id = torch.tensor(0, dtype=torch.long)

        return img_t, prev_seq, target_seq, level_id, S

    def _make_line_sequence(self, page_image, paragraphs, bg_color, para_idx):
        para = paragraphs[para_idx]
        lines = para.get("lines", [])
        hw = int(para.get("is_handwritten", False))
        px1, py1, px2, py2 = para["bbox"]
        img = page_image.crop((px1, py1, px2, py2))
        retina, scale, ox, oy = scale_and_pad(img, bg_color)
        img_t = torch.from_numpy(np.array(retina)).permute(2, 0, 1).float() / 255.0

        prev_list = [PREV_BBOX_NONE]
        target_list = []

        for line in lines:
            lx1, ly1, lx2, ly2 = line["bbox"]
            local = (lx1 - px1, ly1 - py1, lx2 - px1, ly2 - py1)
            rb = bbox_to_retina(local, scale, ox, oy)
            det = (rb[0], rb[1], rb[2], rb[3], CLASS_LINE, hw)
            target_list.append(det)
            prev_list.append(det)

        target_list.append((0, 0, 0, 0, CLASS_NONE, 0))
        prev_list = prev_list[:len(target_list)]

        S = min(len(prev_list), self.max_seq_len)
        prev_list = prev_list[:S]
        target_list = target_list[:S]

        prev_seq = torch.tensor(prev_list, dtype=torch.float32)
        target_seq = torch.tensor(target_list, dtype=torch.float32)
        level_id = torch.tensor(1, dtype=torch.long)

        return img_t, prev_seq, target_seq, level_id, S

    def _make_word_sequence(self, page_image, paragraphs, bg_color,
                            para_idx, line_idx):
        para = paragraphs[para_idx]
        line = para.get("lines", [])[line_idx]
        words = line.get("words", [])
        hw = int(para.get("is_handwritten", False))
        lx1, ly1, lx2, ly2 = line["bbox"]
        img = page_image.crop((lx1, ly1, lx2, ly2))
        retina, scale, ox, oy = scale_and_pad(img, bg_color)
        img_t = torch.from_numpy(np.array(retina)).permute(2, 0, 1).float() / 255.0

        prev_list = [PREV_BBOX_NONE]
        target_list = []

        for word in words:
            wx1, wy1, wx2, wy2 = word["bbox"]
            local = (wx1 - lx1, wy1 - ly1, wx2 - lx1, wy2 - ly1)
            rb = bbox_to_retina(local, scale, ox, oy)
            det = (rb[0], rb[1], rb[2], rb[3], CLASS_WORD, hw)
            target_list.append(det)
            prev_list.append(det)

        target_list.append((0, 0, 0, 0, CLASS_NONE, 0))
        prev_list = prev_list[:len(target_list)]

        S = min(len(prev_list), self.max_seq_len)
        prev_list = prev_list[:S]
        target_list = target_list[:S]

        prev_seq = torch.tensor(prev_list, dtype=torch.float32)
        target_seq = torch.tensor(target_list, dtype=torch.float32)
        level_id = torch.tensor(2, dtype=torch.long)

        return img_t, prev_seq, target_seq, level_id, S

    def _make_char_sequence(self, page_image, paragraphs, bg_color,
                            para_idx, line_idx, word_idx):
        para = paragraphs[para_idx]
        word = para.get("lines", [])[line_idx].get("words", [])[word_idx]
        chars = word.get("characters", [])
        hw = int(para.get("is_handwritten", False))
        wx1, wy1, wx2, wy2 = word["bbox"]
        img = page_image.crop((wx1, wy1, wx2, wy2))
        retina, scale, ox, oy = scale_and_pad(img, bg_color)
        img_t = torch.from_numpy(np.array(retina)).permute(2, 0, 1).float() / 255.0

        prev_list = [PREV_BBOX_NONE]
        target_list = []

        for ch in chars:
            cx1, cy1, cx2, cy2 = ch["bbox"]
            local = (cx1 - wx1, cy1 - wy1, cx2 - wx1, cy2 - wy1)
            rb = bbox_to_retina(local, scale, ox, oy)
            class_id = char_to_class(ch["char"])
            det = (rb[0], rb[1], rb[2], rb[3], class_id, hw)
            target_list.append(det)
            prev_list.append(det)

        target_list.append((0, 0, 0, 0, CLASS_NONE, 0))
        prev_list = prev_list[:len(target_list)]

        S = min(len(prev_list), self.max_seq_len)
        prev_list = prev_list[:S]
        target_list = target_list[:S]

        prev_seq = torch.tensor(prev_list, dtype=torch.float32)
        target_seq = torch.tensor(target_list, dtype=torch.float32)
        level_id = torch.tensor(3, dtype=torch.long)

        return img_t, prev_seq, target_seq, level_id, S


def sequence_collate_fn(batch):
    """Pad sequences to max length in batch, create boolean padding mask."""
    imgs, prevs, targets, levels, lengths = zip(*batch)

    B = len(imgs)
    max_len = max(lengths)

    img_batch = torch.stack(imgs)
    level_batch = torch.stack(levels)

    prev_batch = torch.zeros(B, max_len, 6, dtype=torch.float32)
    target_batch = torch.zeros(B, max_len, 6, dtype=torch.float32)
    padding_mask = torch.ones(B, max_len, dtype=torch.bool)  # True = padded

    for i in range(B):
        S = lengths[i]
        prev_batch[i, :S] = prevs[i]
        target_batch[i, :S] = targets[i]
        padding_mask[i, :S] = False

    return img_batch, prev_batch, target_batch, level_batch, padding_mask


# ---------------------------------------------------------------------------
# 4. SequenceLoss
# ---------------------------------------------------------------------------
class SequenceLoss(nn.Module):
    """Loss for sequence predictions with padding mask."""

    def __init__(self, bbox_weight=1.0, class_weight=1.0, hw_weight=0.5):
        super().__init__()
        self.bbox_weight = bbox_weight
        self.class_weight = class_weight
        self.hw_weight = hw_weight
        self.cls_loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.bbox_loss_fn = nn.SmoothL1Loss(reduction="none")
        self.hw_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, bbox_pred, class_pred, handwritten_pred,
                target_seq, padding_mask):
        """
        Args:
            bbox_pred:        (B, S, 4) predicted bboxes
            class_pred:       (B, S, NUM_CLASSES) class logits
            handwritten_pred: (B, S, 1) handwritten logits
            target_seq:       (B, S, 6) target (x1,y1,x2,y2,class_id,is_handwritten)
            padding_mask:     (B, S) bool — True where padded
        """
        B, S, _ = class_pred.shape
        target_class = target_seq[:, :, 4].long()        # (B, S)
        target_bbox = target_seq[:, :, :4]                # (B, S, 4)
        target_hw = target_seq[:, :, 5]                   # (B, S)
        valid = ~padding_mask                              # (B, S) True where valid

        # Classification loss at valid positions
        cls_loss = self.cls_loss_fn(
            class_pred.reshape(B * S, -1),
            target_class.reshape(B * S),
        ).reshape(B, S)
        cls_loss = (cls_loss * valid.float()).sum() / valid.float().sum().clamp(min=1)

        # Bbox loss at valid non-NONE positions
        non_none = valid & (target_class != CLASS_NONE)  # (B, S)
        if non_none.any():
            pred_norm = bbox_pred / RETINA_SIZE
            tgt_norm = target_bbox / RETINA_SIZE
            bbox_loss = self.bbox_loss_fn(pred_norm, tgt_norm).mean(dim=-1)  # (B, S)
            bbox_loss = (bbox_loss * non_none.float()).sum() / non_none.float().sum().clamp(min=1)
        else:
            bbox_loss = torch.tensor(0.0, device=bbox_pred.device)

        # Handwritten BCE loss — only at paragraph-level positions
        para_mask = valid & (target_class == CLASS_PARAGRAPH)  # (B, S)
        if para_mask.any():
            hw_logits = handwritten_pred.squeeze(-1)  # (B, S)
            hw_loss = self.hw_loss_fn(hw_logits, target_hw)  # (B, S)
            hw_loss = (hw_loss * para_mask.float()).sum() / para_mask.float().sum().clamp(min=1)
        else:
            hw_loss = torch.tensor(0.0, device=bbox_pred.device)

        total = (self.bbox_weight * bbox_loss
                 + self.class_weight * cls_loss
                 + self.hw_weight * hw_loss)
        return total, bbox_loss, cls_loss, hw_loss


# ---------------------------------------------------------------------------
# 5. load_backbone_from_v02
# ---------------------------------------------------------------------------
def load_backbone_from_v02(model, checkpoint_path, device):
    """Load ResNet-18 backbone weights from model_02.pth into RetinaOCRGPT."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    ckpt = _remap_old_backbone_keys(ckpt)

    # Extract backbone keys: stem.*, layer1-4.*, img_mean, img_std
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
# 6. Training loop
# ---------------------------------------------------------------------------
def fit(epochs, model, loss_fn, opt, train_dl, valid_dl, device, save_path,
        patience=15, grad_clip=1.0, freeze_backbone_epochs=0, scheduler=None):
    best_val_loss = float("inf")
    epochs_no_improve = 0

    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=5,
        )

    for epoch in range(epochs):
        # Unfreeze backbone after freeze phase
        if epoch == freeze_backbone_epochs and freeze_backbone_epochs > 0:
            print(f"  -> Unfreezing backbone at epoch {epoch}")
            for name, param in model.named_parameters():
                param.requires_grad = True

        # -- Train --
        model.train()
        train_total, train_bbox, train_cls, train_hw, train_n = 0.0, 0.0, 0.0, 0.0, 0
        for img, prev_seq, target_seq, level_ids, padding_mask in train_dl:
            img = img.to(device)
            prev_seq = prev_seq.to(device)
            target_seq = target_seq.to(device)
            level_ids = level_ids.to(device)
            padding_mask = padding_mask.to(device)

            bbox_pred, class_pred, hw_pred = model(
                img, prev_seq, level_ids, padding_mask,
            )
            total, bbox_loss, cls_loss, hw_loss = loss_fn(
                bbox_pred, class_pred, hw_pred, target_seq, padding_mask,
            )

            total.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            opt.zero_grad()

            n = img.size(0)
            train_total += total.item() * n
            train_bbox += bbox_loss.item() * n
            train_cls += cls_loss.item() * n
            train_hw += hw_loss.item() * n
            train_n += n

        train_total /= train_n
        train_bbox /= train_n
        train_cls /= train_n
        train_hw /= train_n

        # -- Validate --
        model.eval()
        val_total, val_bbox, val_cls, val_hw, val_n = 0.0, 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for img, prev_seq, target_seq, level_ids, padding_mask in valid_dl:
                img = img.to(device)
                prev_seq = prev_seq.to(device)
                target_seq = target_seq.to(device)
                level_ids = level_ids.to(device)
                padding_mask = padding_mask.to(device)

                bbox_pred, class_pred, hw_pred = model(
                    img, prev_seq, level_ids, padding_mask,
                )
                total, bbox_loss, cls_loss, hw_loss = loss_fn(
                    bbox_pred, class_pred, hw_pred, target_seq, padding_mask,
                )

                n = img.size(0)
                val_total += total.item() * n
                val_bbox += bbox_loss.item() * n
                val_cls += cls_loss.item() * n
                val_hw += hw_loss.item() * n
                val_n += n

        val_total /= val_n
        val_bbox /= val_n
        val_cls /= val_n
        val_hw /= val_n

        scheduler.step(val_total)
        cur_lr = opt.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d} | "
            f"train loss={train_total:.4f} (bbox={train_bbox:.4f} cls={train_cls:.4f} hw={train_hw:.4f}) | "
            f"val loss={val_total:.4f} (bbox={val_bbox:.4f} cls={val_cls:.4f} hw={val_hw:.4f}) | "
            f"lr={cur_lr:.2e}",
            flush=True,
        )

        # Checkpoint best model
        if val_total < best_val_loss:
            best_val_loss = val_total
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> saved best model ({save_path})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"  Early stopping: val loss has not improved for "
                    f"{patience} epochs (best={best_val_loss:.4f})"
                )
                break


# ---------------------------------------------------------------------------
# 7. generate_sequence() inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_sequence(model, source_img, bg_color, level_id, device,
                      max_len=128):
    """Autoregressive sequence generation.

    Start with PREV_BBOX_NONE, run full history through transformer each step,
    take last position prediction. Stop on CLASS_NONE.

    Returns: [(source_bbox, class_id, is_handwritten), ...]
    where is_handwritten is a float probability (only meaningful for paragraphs).
    """
    from infer_02 import retina_to_source

    retina, scale, ox, oy = scale_and_pad(source_img, bg_color)
    img_t = (
        torch.from_numpy(np.array(retina))
        .permute(2, 0, 1).float().unsqueeze(0) / 255.0
    ).to(device)

    level_t = torch.tensor([level_id], dtype=torch.long, device=device)

    # Build sequence incrementally (6-dim)
    seq = [list(PREV_BBOX_NONE)]
    detections = []

    for _ in range(max_len):
        prev_t = torch.tensor([seq], dtype=torch.float32, device=device)
        bbox_pred, class_pred, hw_pred = model(img_t, prev_t, level_t)

        # Take last position
        class_id = class_pred[0, -1].argmax().item()
        if class_id == CLASS_NONE:
            break

        rb = bbox_pred[0, -1].tolist()
        hw_prob = torch.sigmoid(hw_pred[0, -1, 0]).item()
        source_bbox = retina_to_source(rb, scale, ox, oy)

        # Clamp to image bounds
        w, h = source_img.size
        source_bbox = (
            max(0, min(source_bbox[0], w - 1)),
            max(0, min(source_bbox[1], h - 1)),
            max(0, min(source_bbox[2], w)),
            max(0, min(source_bbox[3], h)),
        )
        if source_bbox[2] <= source_bbox[0] or source_bbox[3] <= source_bbox[1]:
            break

        hw_flag = 1.0 if hw_prob > 0.5 else 0.0
        detections.append((source_bbox, class_id, hw_prob))
        seq.append([rb[0], rb[1], rb[2], rb[3], class_id, hw_flag])

    return detections


def run_inference(model, image_path, output_path, device):
    """4-level hierarchical inference on a document image."""
    from infer_02 import class_to_char
    from PIL import ImageDraw, ImageFont

    page_img = Image.open(image_path).convert("RGB")
    from annotate_real import estimate_background_color
    bg_color = estimate_background_color(page_img)
    print(f"Image: {image_path} ({page_img.size[0]}x{page_img.size[1]}), bg={bg_color}")

    model.eval()

    # Level 0: detect paragraphs
    paragraphs = generate_sequence(model, page_img, bg_color, 0, device)
    print(f"Detected {len(paragraphs)} paragraphs")

    all_text = []
    all_lines_info = []
    all_words_info = []
    all_chars_info = []

    for pi, (para_bbox, _, hw_prob) in enumerate(paragraphs):
        hw_label = "handwritten" if hw_prob > 0.5 else "printed"
        print(f"  Paragraph {pi+1}: {hw_label} (p={hw_prob:.2f})")

        para_crop = page_img.crop(para_bbox)

        # Level 1: detect lines
        lines = generate_sequence(model, para_crop, bg_color, 1, device)
        print(f"    {len(lines)} lines")

        for li, (line_bbox, _, _) in enumerate(lines):
            line_crop = para_crop.crop(line_bbox)

            # Level 2: detect words
            words = generate_sequence(model, line_crop, bg_color, 2, device)

            line_text = []
            for wi, (word_bbox, _, _) in enumerate(words):
                word_crop = line_crop.crop(word_bbox)

                # Level 3: detect characters
                chars = generate_sequence(model, word_crop, bg_color, 3, device)

                word_text = []
                for char_bbox, class_id, _ in chars:
                    if class_id >= CHAR_CLASS_OFFSET:
                        word_text.append(class_to_char(class_id))
                    else:
                        word_text.append("?")
                    # Map char bbox to page coords
                    page_char = (
                        char_bbox[0] + word_bbox[0] + line_bbox[0] + para_bbox[0],
                        char_bbox[1] + word_bbox[1] + line_bbox[1] + para_bbox[1],
                        char_bbox[2] + word_bbox[0] + line_bbox[0] + para_bbox[0],
                        char_bbox[3] + word_bbox[1] + line_bbox[1] + para_bbox[1],
                    )
                    all_chars_info.append(page_char)

                line_text.append("".join(word_text))

                # Map word bbox to page coords
                page_word = (
                    word_bbox[0] + line_bbox[0] + para_bbox[0],
                    word_bbox[1] + line_bbox[1] + para_bbox[1],
                    word_bbox[2] + line_bbox[0] + para_bbox[0],
                    word_bbox[3] + line_bbox[1] + para_bbox[1],
                )
                all_words_info.append(page_word)

            text = " ".join(line_text)
            all_text.append((pi, li, text))

            # Map line bbox to page coords
            page_line = (
                line_bbox[0] + para_bbox[0],
                line_bbox[1] + para_bbox[1],
                line_bbox[2] + para_bbox[0],
                line_bbox[3] + para_bbox[1],
            )
            all_lines_info.append(page_line)

    # Print detected text
    print("\n--- Detected Text ---")
    if not all_text:
        print("  (no text detected)")
    current_para = -1
    for pi, li, text in all_text:
        if pi != current_para:
            current_para = pi
            hw_label = ("handwritten" if paragraphs[pi][2] > 0.5
                        else "printed")
            print(f"Paragraph {pi+1} [{hw_label}]:")
        print(f"  Line {li+1}: {text}")

    # Visualize
    vis = page_img.copy()
    draw = ImageDraw.Draw(vis)

    for para_bbox, _, hw_prob in paragraphs:
        color = (255, 0, 0)
        draw.rectangle(para_bbox, outline=color, width=3)
        hw_label = "HW" if hw_prob > 0.5 else "PR"
        draw.text((para_bbox[0], max(0, para_bbox[1] - 14)),
                  hw_label, fill=color)
    for lb in all_lines_info:
        draw.rectangle(lb, outline=(0, 0, 255), width=2)
    for wb in all_words_info:
        draw.rectangle(wb, outline=(255, 165, 0), width=2)  # orange
    for cb in all_chars_info:
        draw.rectangle(cb, outline=(0, 180, 0), width=1)

    vis.save(output_path)
    print(f"Visualization saved to {output_path}")


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from PIL import Image

    parser = argparse.ArgumentParser(description="Train RetinaOCRGPT")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--pages", type=int, default=20)
    parser.add_argument("--save-path", type=str, default="model_03.pth")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--bbox-weight", type=float, default=1.0)
    parser.add_argument("--class-weight", type=float, default=1.0)
    parser.add_argument("--hw-weight", type=float, default=0.5,
                        help="Handwritten classification loss weight")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--page-width", type=int, default=2048)
    parser.add_argument("--page-height", type=int, default=2800)
    parser.add_argument("--seed", type=int, default=None)

    # Model architecture
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training phases
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to model_02.pth for backbone init")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # Data sources
    parser.add_argument("--real-dirs", nargs="*", default=None,
                        help="Real annotation directories")

    # Inference mode
    parser.add_argument("--infer", type=str, default=None,
                        help="Run inference on image instead of training")
    parser.add_argument("--output", type=str, default="infer_03_output.png")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build model
    model = RetinaOCRGPT(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    ).to(device)

    # Load pretrained backbone
    if args.pretrained:
        load_backbone_from_v02(model, args.pretrained, device)

    # --- Inference mode ---
    if args.infer:
        ckpt_path = args.save_path
        if args.pretrained and not torch.cuda.is_available():
            # Try loading the v03 checkpoint
            pass
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt)
            print(f"Loaded model from {ckpt_path}")
        except FileNotFoundError:
            print(f"Warning: {ckpt_path} not found, using initialized weights")

        run_inference(model, args.infer, args.output, device)
        sys.exit(0)

    # --- Training mode ---
    # Discover fonts
    fonts = discover_fonts()
    print(f"Discovered {len(fonts)} fonts")
    if not fonts:
        print("No fonts found! Place .ttf/.otf/.ttc files in fonts/ directory.")
        sys.exit(1)

    # Generate synthetic pages
    print(f"Generating {args.pages} synthetic pages...")
    pages = []
    for i in range(args.pages):
        pages.append(SyntheticPage(fonts, args.page_width, args.page_height))
        total_chars = sum(
            len(line["characters"])
            for para in pages[-1].paragraphs
            for line in para["lines"]
        )
        print(f"  Page {i}: {len(pages[-1].paragraphs)} paragraphs, {total_chars} chars")

    # Build dataset
    dataset = SequenceOCRDataset(
        pages=pages,
        real_dirs=args.real_dirs,
        max_seq_len=args.max_seq_len,
    )
    print(f"Dataset size: {len(dataset)} sequences")

    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"Train: {train_size}, Val: {val_size}")

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
        collate_fn=sequence_collate_fn,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
        collate_fn=sequence_collate_fn,
    )

    # Freeze backbone for phase 1
    if args.freeze_backbone_epochs > 0:
        print(f"Freezing backbone for {args.freeze_backbone_epochs} epochs")
        for name, param in model.named_parameters():
            if any(name.startswith(p) for p in
                   ("stem.", "layer1.", "layer2.", "layer3.", "layer4.")):
                param.requires_grad = False

    # Optimizer with differential learning rates
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
                 ("img_proj.", "det_proj.", "pos_emb.", "level_emb.")):
            projection_params.append(param)
        else:
            transformer_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": args.lr * 0.01},
        {"params": projection_params, "lr": args.lr * 0.5},
        {"params": transformer_params, "lr": args.lr},
    ]
    # Filter out empty groups
    param_groups = [g for g in param_groups if g["params"]]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-2)

    # Loss
    loss_fn = SequenceLoss(
        bbox_weight=args.bbox_weight,
        class_weight=args.class_weight,
        hw_weight=args.hw_weight,
    ).to(device)

    # Train
    print(f"\n=== Training RetinaOCRGPT ({args.epochs} epochs) ===")
    fit(
        args.epochs, model, loss_fn, optimizer, train_dl, val_dl,
        device, args.save_path,
        patience=args.patience,
        grad_clip=args.grad_clip,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
    )
    print("Done.")
