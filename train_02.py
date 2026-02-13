"""
Training script for hierarchical retina-based OCR detection model.

Model: RetinaOCRNet — ResNet-18 backbone + dual head (bbox regression + classification).
Loss: SmoothL1 for bbox coords (non-NONE samples only) + CrossEntropy for 98-class tokens.
Dataset: RetinaOCRDataset from generate_training_data.py (teacher-forced, 3-tuple samples).
"""

import argparse
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, random_split

from generate_training_data import (
    CLASS_NONE,
    RETINA_SIZE,
    CharacterPretrainDataset,
    RetinaOCRDataset,
    SyntheticPage,
    discover_fonts,
)

NUM_CLASSES = 98


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class RetinaOCRNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Pretrained ResNet-18 backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.stem = nn.Sequential(
            resnet.conv1,    # (B,64,512,512)
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,  # (B,64,256,256)
        )
        self.layer1 = resnet.layer1  # (B,64,256,256)
        self.layer2 = resnet.layer2  # (B,128,128,128)
        self.layer3 = resnet.layer3  # (B,256,64,64)
        self.layer4 = resnet.layer4  # (B,512,32,32)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # ImageNet normalization constants
        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # Shared FC trunk: 512 (CNN) + 5 (prev_bbox) = 517
        self.fc_shared = nn.Sequential(
            nn.Linear(517, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Bbox head: predict (x1, y1, x2, y2) in retina coords
        self.bbox_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid(),
        )

        # Class head: 98-class logits
        self.class_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, img, prev_bbox):
        # ImageNet normalize
        x = (img - self.img_mean) / self.img_std

        # Backbone
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)           # (B,512,1,1)
        x = x.view(x.size(0), -1)  # (B,512)

        # Normalize prev_bbox: coords / RETINA_SIZE, class_id / NUM_CLASSES
        prev_norm = prev_bbox.clone()
        prev_norm[:, :4] = prev_norm[:, :4] / RETINA_SIZE
        prev_norm[:, 4] = prev_norm[:, 4] / NUM_CLASSES

        # Concatenate and pass through heads
        x = torch.cat([x, prev_norm], dim=1)  # (B, 517)
        x = self.fc_shared(x)                 # (B, 128)

        bbox_pred = self.bbox_head(x) * RETINA_SIZE  # (B, 4) in retina coords
        class_pred = self.class_head(x)               # (B, 98) raw logits

        return bbox_pred, class_pred


# ---------------------------------------------------------------------------
# Checkpoint compatibility
# ---------------------------------------------------------------------------
# Old backbone.N -> new named stages mapping:
#   backbone.0 -> stem.0 (conv1), backbone.1 -> stem.1 (bn1),
#   backbone.2 -> stem.2 (relu), backbone.3 -> stem.3 (maxpool),
#   backbone.4 -> layer1, backbone.5 -> layer2,
#   backbone.6 -> layer3, backbone.7 -> layer4
_OLD_BACKBONE_MAP = {
    "backbone.0.": "stem.0.",
    "backbone.1.": "stem.1.",
    "backbone.2.": "stem.2.",
    "backbone.3.": "stem.3.",
    "backbone.4.": "layer1.",
    "backbone.5.": "layer2.",
    "backbone.6.": "layer3.",
    "backbone.7.": "layer4.",
}


def _remap_old_backbone_keys(state_dict):
    """Remap old 'backbone.N.*' keys to new named stages if needed."""
    if not any(k.startswith("backbone.") for k in state_dict):
        return state_dict
    new_state = {}
    for k, v in state_dict.items():
        new_k = k
        for old_prefix, new_prefix in _OLD_BACKBONE_MAP.items():
            if k.startswith(old_prefix):
                new_k = new_prefix + k[len(old_prefix):]
                break
        new_state[new_k] = v
    return new_state


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class RetinaOCRLoss(nn.Module):
    def __init__(self, bbox_weight=1.0, class_weight=1.0):
        super().__init__()
        self.bbox_weight = bbox_weight
        self.class_weight = class_weight
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.bbox_loss_fn = nn.SmoothL1Loss()

    def forward(self, bbox_pred, class_pred, target):
        target_class = target[:, 4].long()
        class_loss = self.cls_loss_fn(class_pred, target_class)

        # Bbox loss only for non-NONE samples
        non_none = target_class != CLASS_NONE
        if non_none.any():
            pred_norm = bbox_pred[non_none] / RETINA_SIZE
            tgt_norm = target[non_none, :4] / RETINA_SIZE
            bbox_loss = self.bbox_loss_fn(pred_norm, tgt_norm)
        else:
            bbox_loss = torch.tensor(0.0, device=bbox_pred.device)

        total = self.bbox_weight * bbox_loss + self.class_weight * class_loss
        return total, bbox_loss, class_loss


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def loss_batch(model, loss_fn, img, prev, target, opt=None):
    bbox_pred, class_pred = model(img, prev)
    total, bbox_loss, class_loss = loss_fn(bbox_pred, class_pred, target)

    if opt is not None:
        total.backward()
        opt.step()
        opt.zero_grad()

    return total.item(), bbox_loss.item(), class_loss.item(), len(img)


def fit(epochs, model, loss_fn, opt, train_dl, valid_dl, device, save_path,
        patience=15):
    best_val_loss = float("inf")
    epochs_no_improve = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5,
    )

    for epoch in range(epochs):
        # -- Train --
        model.train()
        train_total, train_bbox, train_cls, train_n = 0.0, 0.0, 0.0, 0
        for img, prev, target in train_dl:
            img = img.to(device)
            prev = prev.to(device)
            target = target.to(device)
            t, b, c, n = loss_batch(model, loss_fn, img, prev, target, opt)
            train_total += t * n
            train_bbox += b * n
            train_cls += c * n
            train_n += n
        train_total /= train_n
        train_bbox /= train_n
        train_cls /= train_n

        # -- Validate --
        model.eval()
        val_total, val_bbox, val_cls, val_n = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for img, prev, target in valid_dl:
                img = img.to(device)
                prev = prev.to(device)
                target = target.to(device)
                t, b, c, n = loss_batch(model, loss_fn, img, prev, target)
                val_total += t * n
                val_bbox += b * n
                val_cls += c * n
                val_n += n
        val_total /= val_n
        val_bbox /= val_n
        val_cls /= val_n

        scheduler.step(val_total)
        cur_lr = opt.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d} | "
            f"train loss={train_total:.4f} (bbox={train_bbox:.4f} cls={train_cls:.4f}) | "
            f"val loss={val_total:.4f} (bbox={val_bbox:.4f} cls={val_cls:.4f}) | "
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
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RetinaOCRNet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pages", type=int, default=20)
    parser.add_argument("--save-path", type=str, default="model_02.pth")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--bbox-weight", type=float, default=1.0)
    parser.add_argument("--class-weight", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (epochs with no val improvement)")
    parser.add_argument("--page-width", type=int, default=2048)
    parser.add_argument("--page-height", type=int, default=2800)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pretrain-epochs", type=int, default=10,
                        help="Epochs of character pretraining (0 to skip)")
    parser.add_argument("--pretrain-samples", type=int, default=10000,
                        help="Number of single-character samples for pretraining")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Discover fonts
    fonts = discover_fonts()
    print(f"Discovered {len(fonts)} fonts")
    if not fonts:
        print("No fonts found! Place .ttf/.otf/.ttc files in fonts/ directory.")
        sys.exit(1)

    # Model
    model = RetinaOCRNet().to(device)

    # Optimizer with differential learning rates
    backbone_params = (
        list(model.stem.parameters())
        + list(model.layer1.parameters())
        + list(model.layer2.parameters())
        + list(model.layer3.parameters())
        + list(model.layer4.parameters())
    )
    head_params = (
        list(model.fc_shared.parameters())
        + list(model.bbox_head.parameters())
        + list(model.class_head.parameters())
    )

    # Loss
    loss_fn = RetinaOCRLoss(
        bbox_weight=args.bbox_weight, class_weight=args.class_weight,
    ).to(device)

    # ---- Phase 1: Character pretraining ----
    if args.pretrain_epochs > 0:
        print(f"\n=== Phase 1: Character pretraining ({args.pretrain_epochs} epochs) ===")
        print(f"Generating {args.pretrain_samples} character samples...")
        char_dataset = CharacterPretrainDataset(fonts, num_samples=args.pretrain_samples)
        print(f"Character dataset size: {len(char_dataset)} samples")

        char_val_size = int(len(char_dataset) * args.val_split)
        char_train_size = len(char_dataset) - char_val_size
        char_train_ds, char_val_ds = random_split(char_dataset,
                                                   [char_train_size, char_val_size])
        print(f"Char train: {char_train_size}, Char val: {char_val_size}")

        char_train_dl = DataLoader(
            char_train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=2, pin_memory=True,
        )
        char_val_dl = DataLoader(
            char_val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=2, pin_memory=True,
        )

        pretrain_opt = torch.optim.AdamW([
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": head_params, "lr": args.lr},
        ], weight_decay=1e-2)

        fit(args.pretrain_epochs, model, loss_fn, pretrain_opt,
            char_train_dl, char_val_dl, device, args.save_path,
            patience=args.pretrain_epochs)  # no early stopping during pretrain
        print("Character pretraining complete.\n")

    # ---- Phase 2: Full hierarchical training ----
    print(f"=== Phase 2: Full hierarchical training ({args.epochs} epochs) ===")

    # Generate pages
    print(f"Generating {args.pages} pages...")
    pages = []
    for i in range(args.pages):
        pages.append(SyntheticPage(fonts, args.page_width, args.page_height))
        total_chars = sum(
            len(line["characters"])
            for para in pages[-1].paragraphs
            for line in para["lines"]
        )
        print(f"  Page {i}: {len(pages[-1].paragraphs)} paragraphs, {total_chars} chars")

    # Build dataset and split
    dataset = RetinaOCRDataset(pages)
    print(f"Dataset size: {len(dataset)} samples")

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"Train: {train_size}, Val: {val_size}")

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # Fresh optimizer for phase 2
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=1e-2)

    # Train
    fit(args.epochs, model, loss_fn, optimizer, train_dl, val_dl, device, args.save_path,
        patience=args.patience)
    print("Done.")
