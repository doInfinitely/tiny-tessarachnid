"""
Training script for hierarchical retina-based OCR detection model.

Model: RetinaOCRNet — ResNet-18 backbone + dual head (bbox regression + classification).
Loss: SmoothL1 for bbox coords (non-NONE samples only) + CrossEntropy for 98-class tokens.
Dataset: RetinaOCRDataset from generate_training_data.py (teacher-forced, 3-tuple samples).
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision.models as models
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from generate_training_data import (
    CLASS_NONE,
    CLASS_PAGE,
    NUM_CLASSES,
    RETINA_SIZE,
    AugmentedSubset,
    CharacterPretrainDataset,
    RetinaOCRDataset,
    SyntheticPage,
    build_augmentation,
    discover_fonts,
)


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------
def setup_ddp():
    """Initialize distributed process group. Expects torchrun env vars."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
_BACKBONE_CONFIGS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, 512),
    "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT, 512),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
}


class RetinaOCRNet(nn.Module):
    def __init__(self, backbone="resnet18", num_classes=NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes

        factory, weights, feat_dim = _BACKBONE_CONFIGS[backbone]
        resnet = factory(weights=weights)
        self.feat_dim = feat_dim

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # ImageNet normalization constants
        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # Shared FC trunk: feat_dim (CNN) + 5 (prev_bbox)
        self.fc_shared = nn.Sequential(
            nn.Linear(feat_dim + 5, 256),
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

        # Class head: num_classes logits
        self.class_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
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
        x = self.pool(x)           # (B, feat_dim, 1, 1)
        x = x.view(x.size(0), -1)  # (B, feat_dim)

        # Normalize prev_bbox: coords / RETINA_SIZE, class_id / NUM_CLASSES
        # Trim to 5 dims (x1,y1,x2,y2,class_id) — dataset may supply extra dims
        prev_norm = prev_bbox[:, :5].clone()
        prev_norm[:, :4] = prev_norm[:, :4] / RETINA_SIZE
        prev_norm[:, 4] = prev_norm[:, 4] / self.num_classes

        # Concatenate and pass through heads
        x = torch.cat([x, prev_norm], dim=1)  # (B, feat_dim + 5)
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
    def __init__(self, bbox_weight=1.0, class_weight=1.0, label_smoothing=0.0):
        super().__init__()
        self.bbox_weight = bbox_weight
        self.class_weight = class_weight
        self.cls_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
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
def loss_batch(model, loss_fn, img, prev, target, opt=None,
               scaler=None, clip_grad=0.0):
    use_amp = scaler is not None

    with torch.amp.autocast("cuda", enabled=use_amp):
        bbox_pred, class_pred = model(img, prev)
        total, bbox_loss, class_loss = loss_fn(bbox_pred, class_pred, target)

    if opt is not None:
        if use_amp:
            scaler.scale(total).backward()
            if clip_grad > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(opt)
            scaler.update()
        else:
            total.backward()
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()
        opt.zero_grad()

    return total.item(), bbox_loss.item(), class_loss.item(), len(img)


def fit(epochs, model, loss_fn, opt, train_dl, valid_dl, device, save_path,
        patience=15, scaler=None, clip_grad=0.0, warmup_epochs=0):
    best_val_loss = float("inf")
    epochs_no_improve = 0
    use_ddp = dist.is_initialized()

    # Cosine annealing with optional linear warmup
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(epochs, 1),
        )

    for epoch in range(epochs):
        # Set epoch on distributed samplers for proper shuffling
        if use_ddp:
            if hasattr(train_dl.sampler, "set_epoch"):
                train_dl.sampler.set_epoch(epoch)

        # -- Train --
        model.train()
        train_total, train_bbox, train_cls, train_n = 0.0, 0.0, 0.0, 0
        for img, prev, target in train_dl:
            img = img.to(device)
            prev = prev.to(device)
            target = target.to(device)
            t, b, c, n = loss_batch(model, loss_fn, img, prev, target, opt,
                                    scaler=scaler, clip_grad=clip_grad)
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

        # Average val loss across ranks for consistent early stopping
        if use_ddp:
            val_tensor = torch.tensor([val_total, val_bbox, val_cls],
                                      device=device)
            dist.all_reduce(val_tensor, op=dist.ReduceOp.AVG)
            val_total, val_bbox, val_cls = val_tensor.tolist()

        scheduler.step()
        cur_lr = opt.param_groups[0]["lr"]

        if is_main_process():
            print(
                f"Epoch {epoch:3d} | "
                f"train loss={train_total:.4f} (bbox={train_bbox:.4f} cls={train_cls:.4f}) | "
                f"val loss={val_total:.4f} (bbox={val_bbox:.4f} cls={val_cls:.4f}) | "
                f"lr={cur_lr:.2e}",
                flush=True,
            )

        # Checkpoint best model (all ranks check, only rank 0 saves)
        if val_total < best_val_loss:
            best_val_loss = val_total
            epochs_no_improve = 0
            if is_main_process():
                raw_model = model.module if use_ddp else model
                torch.save(raw_model.state_dict(), save_path)
                print(f"  -> saved best model ({save_path})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if is_main_process():
                    print(
                        f"  Early stopping: val loss has not improved for "
                        f"{patience} epochs (best={best_val_loss:.4f})"
                    )
                break

        # Barrier to keep ranks in sync after each epoch
        if use_ddp:
            dist.barrier()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RetinaOCRNet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--pages", type=int, default=200)
    parser.add_argument("--save-path", type=str, default="model_02.pth")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--bbox-weight", type=float, default=1.0)
    parser.add_argument("--class-weight", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (epochs with no val improvement)")
    parser.add_argument("--page-width", type=int, default=2048)
    parser.add_argument("--page-height", type=int, default=2800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrain-epochs", type=int, default=10,
                        help="Epochs of character pretraining (0 to skip)")
    parser.add_argument("--pretrain-samples", type=int, default=50000,
                        help="Number of single-character samples for pretraining")
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet18", "resnet34", "resnet50"],
                        help="ResNet backbone variant")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Enable mixed-precision training")
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--clip-grad", type=float, default=1.0,
                        help="Max gradient norm (0 to disable)")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing for CrossEntropyLoss")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Linear LR warmup epochs before cosine decay")
    parser.add_argument("--augment", action="store_true", default=True,
                        help="Enable photometric augmentation on train set")
    parser.add_argument("--no-augment", action="store_false", dest="augment")
    args = parser.parse_args()

    # ---- DDP setup ----
    # Work around NCCL peer-access issues in some virtualized environments
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_SHM_DISABLE", "1")

    local_rank = setup_ddp()
    device = torch.device("cuda", local_rank)
    world_size = dist.get_world_size()

    if is_main_process():
        print(f"DDP: {world_size} GPUs, per-GPU batch size={args.batch_size}, "
              f"effective batch size={args.batch_size * world_size}")

    # Seed all ranks identically so data generation is deterministic
    data_seed = args.seed
    random.seed(data_seed)
    np.random.seed(data_seed)
    torch.manual_seed(data_seed)

    # Discover fonts
    fonts = discover_fonts()
    if is_main_process():
        print(f"Discovered {len(fonts)} fonts")
    if not fonts:
        if is_main_process():
            print("No fonts found! Place .ttf/.otf/.ttc files in fonts/ directory.")
        cleanup_ddp()
        sys.exit(1)

    # Model — find_unused_parameters needed because bbox_head gets no grad
    # on batches where all samples are CLASS_NONE (bbox loss is a constant).
    model = RetinaOCRNet(backbone=args.backbone).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if is_main_process():
        print(f"Backbone: {args.backbone} (feat_dim={model.module.feat_dim})")
        print(f"AMP: {args.amp}, Clip grad: {args.clip_grad}, "
              f"Label smoothing: {args.label_smoothing}")

    # GradScaler for mixed precision
    scaler = torch.amp.GradScaler("cuda") if args.amp else None

    # Optimizer with differential learning rates (reference via model.module)
    backbone_params = (
        list(model.module.stem.parameters())
        + list(model.module.layer1.parameters())
        + list(model.module.layer2.parameters())
        + list(model.module.layer3.parameters())
        + list(model.module.layer4.parameters())
    )
    head_params = (
        list(model.module.fc_shared.parameters())
        + list(model.module.bbox_head.parameters())
        + list(model.module.class_head.parameters())
    )

    # Loss
    loss_fn = RetinaOCRLoss(
        bbox_weight=args.bbox_weight, class_weight=args.class_weight,
        label_smoothing=args.label_smoothing,
    ).to(device)

    # Use more DataLoader workers to keep H100s fed
    num_workers = 4

    # ---- Phase 1: Character pretraining ----
    if args.pretrain_epochs > 0:
        if is_main_process():
            print(f"\n=== Phase 1: Character pretraining ({args.pretrain_epochs} epochs) ===")
            print(f"Generating {args.pretrain_samples} character samples...")
        char_dataset = CharacterPretrainDataset(fonts, num_samples=args.pretrain_samples)
        if is_main_process():
            print(f"Character dataset size: {len(char_dataset)} samples")

        char_val_size = int(len(char_dataset) * args.val_split)
        char_train_size = len(char_dataset) - char_val_size
        char_train_ds, char_val_ds = random_split(
            char_dataset, [char_train_size, char_val_size],
            generator=torch.Generator().manual_seed(data_seed),
        )
        if args.augment:
            char_train_ds = AugmentedSubset(char_train_ds, build_augmentation())
        if is_main_process():
            print(f"Char train: {char_train_size}, Char val: {char_val_size}"
                  f"{' (augmented)' if args.augment else ''}")

        char_train_sampler = DistributedSampler(char_train_ds, shuffle=True)
        char_val_sampler = DistributedSampler(char_val_ds, shuffle=False)

        char_train_dl = DataLoader(
            char_train_ds, batch_size=args.batch_size,
            sampler=char_train_sampler,
            num_workers=num_workers, pin_memory=True,
        )
        char_val_dl = DataLoader(
            char_val_ds, batch_size=args.batch_size,
            sampler=char_val_sampler,
            num_workers=num_workers, pin_memory=True,
        )

        pretrain_opt = torch.optim.AdamW([
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": head_params, "lr": args.lr},
        ], weight_decay=1e-2)

        fit(args.pretrain_epochs, model, loss_fn, pretrain_opt,
            char_train_dl, char_val_dl, device, args.save_path,
            patience=args.pretrain_epochs,
            scaler=scaler, clip_grad=args.clip_grad,
            warmup_epochs=args.warmup_epochs)  # no early stopping during pretrain
        if is_main_process():
            print("Character pretraining complete.\n")

    # ---- Phase 2: Full hierarchical training ----
    if is_main_process():
        print(f"=== Phase 2: Full hierarchical training ({args.epochs} epochs) ===")

    # Reset seed so all ranks generate identical pages
    random.seed(data_seed + 1)
    np.random.seed(data_seed + 1)

    # Generate pages
    if is_main_process():
        print(f"Generating {args.pages} pages...")
    pages = []
    for i in range(args.pages):
        pages.append(SyntheticPage(fonts, args.page_width, args.page_height))
        if is_main_process():
            total_chars = sum(
                len(word["characters"])
                for para in pages[-1].paragraphs
                for line in para["lines"]
                for word in line["words"]
            )
            print(f"  Page {i}: {len(pages[-1].paragraphs)} paragraphs, {total_chars} chars")

    # Build dataset and split
    dataset = RetinaOCRDataset(pages)
    if is_main_process():
        print(f"Dataset size: {len(dataset)} samples")

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(data_seed),
    )
    if args.augment:
        train_ds = AugmentedSubset(train_ds, build_augmentation())
    if is_main_process():
        print(f"Train: {train_size}, Val: {val_size}"
              f"{' (augmented)' if args.augment else ''}")

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=num_workers, pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=num_workers, pin_memory=True,
    )

    # Fresh optimizer for phase 2
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=1e-2)

    # Train
    fit(args.epochs, model, loss_fn, optimizer, train_dl, val_dl, device, args.save_path,
        patience=args.patience, scaler=scaler, clip_grad=args.clip_grad,
        warmup_epochs=args.warmup_epochs)

    cleanup_ddp()
    if is_main_process():
        print("Done.")
