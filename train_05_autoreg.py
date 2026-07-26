"""Autoregressive QuadOCRNet — emits ONE child quad per call, conditioned on
the previous child's quad coordinates + the level embedding.

Each call:  image (warped parent region @ 1024x1024)
          + level_id  (PAGE / PARAGRAPH / LINE / WORD / CHAR)
          + prev_quad (4x2 coords of the previous sibling, or a sentinel for
                       the first child)
        -> next_quad (4x2)
         + class_logits
         + objectness  (1.0 = "another child follows", 0.0 = end-of-sequence)

Reading order is in the original page coords (top-to-bottom, then left-to-
right). After choosing the parent region, the parent is cropped+warped to
1024x1024; children quads transform into the warped frame. Optional small
rotation/shear augmentation is applied on top of the warp.

At inference the cascade is:
    detect page -> warp the page quad to 1024x1024 rectangle ->
    repeatedly call with level=PARAGRAPH, prev_quad starts as sentinel, then
    each emitted quad becomes the prev_quad for the next call, until
    objectness < 0.5. For each paragraph, warp to rectangle and recurse for
    lines, etc.
"""
import os
import random
import sys

OLD = os.path.expanduser("~/Downloads/tiny-tessarachnid")
sys.path.insert(0, OLD)
HERE = os.path.dirname(os.path.abspath(__file__))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from generate_training_data import (
    CLASS_NONE, CLASS_PAGE, CLASS_PARAGRAPH, CLASS_LINE, CLASS_WORD,
    SyntheticPage, discover_fonts,
)
from augmentation import (
    sample_affine_params, apply_affine_image, apply_affine_points,
    estimate_bg_color,
)

RETINA_SIZE = 1024


# ---------------------------------------------------------------------------
# Homography helpers (mirrors cascade_infer.py — no circular import)
# ---------------------------------------------------------------------------
def _homography_matrix(src, dst):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    A, b = [], []
    for (x, y), (xp, yp) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -x * xp, -y * xp])
        A.append([0, 0, 0, x, y, 1, -x * yp, -y * yp])
        b.extend([xp, yp])
    h = np.linalg.solve(np.array(A), np.array(b))
    return np.array([[h[0], h[1], h[2]],
                     [h[3], h[4], h[5]],
                     [h[6], h[7], 1.0]])


def _apply_h(M, pts):
    """Apply 3×3 homography M to (N, 2) float points → (N, 2) float32."""
    pts = np.asarray(pts, dtype=np.float64)
    ones = np.ones((len(pts), 1))
    h = np.concatenate([pts, ones], axis=1) @ M.T
    return (h[:, :2] / h[:, 2:3]).astype(np.float32)


# ---------------------------------------------------------------------------
# Canonical corner ordering: 0=NW, 1=NE, 2=SE, 3=SW
# ---------------------------------------------------------------------------
def bbox_to_quad(x1, y1, x2, y2):
    return torch.tensor([
        [x1, y1], [x2, y1], [x2, y2], [x1, y2],
    ], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class AutoregQuadOCRNet(nn.Module):
    NUM_LEVELS = 5      # PAGE, PARAGRAPH, LINE, WORD, CHAR
    LEVELS = ["page", "paragraph", "line", "word", "char"]

    def __init__(self, num_classes, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # 4th input channel: binary mask of already-detected sibling quads.
        # Pretrained RGB weights are kept; mask channel starts at zero so the
        # model initially behaves as if the mask weren't there.
        conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            conv1.weight[:, :3] = resnet.conv1.weight
            conv1.weight[:, 3:] = 0.0
        self.stem = nn.Sequential(
            conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        D = 512
        self.level_emb = nn.Embedding(self.NUM_LEVELS, D)
        # learnable start token used when there is no previous sibling
        self.start_token = nn.Parameter(torch.randn(D) * 0.02)
        self.prev_mlp = nn.Sequential(
            nn.Linear(8, 256), nn.GELU(),
            nn.Linear(256, D), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(D + D + D, D), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(D, D), nn.GELU(), nn.Dropout(dropout),
        )
        self.quad_head = nn.Linear(D, 8)
        self.class_head = nn.Linear(D, num_classes)
        self.obj_head = nn.Linear(D, 1)

    def encode(self, img):
        if img.shape[1] == 3:
            mask = img.new_zeros(img.shape[0], 1, *img.shape[2:])
            img = torch.cat([img, mask], dim=1)
        rgb = (img[:, :3] - self.img_mean) / self.img_std
        x = torch.cat([rgb, img[:, 3:]], dim=1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.pool(x).flatten(1)             # (B, 512)

    def forward(self, img, level_ids, prev_quads, is_start):
        """
        img:        (B, 4, 1024, 1024) — RGB + detected-siblings mask
                    (a 3-channel input gets a zero mask appended)
        level_ids:  (B,) long
        prev_quads: (B, 4, 2) float (ignored where is_start)
        is_start:   (B,) bool

        Returns: pred_quad (B, 4, 2), cls_logits (B, num_classes),
                 obj_logit (B,)
        """
        feat = self.encode(img)                                # (B, D)
        lvl = self.level_emb(level_ids)                        # (B, D)
        prev_in = (prev_quads.flatten(1) / RETINA_SIZE).clamp(-1, 2)
        prev_ctx = self.prev_mlp(prev_in)                      # (B, D)
        # replace with start token where is_start
        start_expanded = self.start_token.unsqueeze(0).expand_as(prev_ctx)
        prev_ctx = torch.where(is_start.unsqueeze(-1),
                                start_expanded, prev_ctx)
        x = torch.cat([feat, lvl, prev_ctx], dim=-1)
        x = self.head(x)
        quad = torch.sigmoid(self.quad_head(x)).view(-1, 4, 2) * RETINA_SIZE
        cls = self.class_head(x)
        obj = self.obj_head(x).squeeze(-1)
        return quad, cls, obj


# ---------------------------------------------------------------------------
# Reading-order collection from a SyntheticPage
# ---------------------------------------------------------------------------
def _bbox_quad_class(b, class_id):
    return bbox_to_quad(*b), class_id


def collect_parents_and_children(page, level):
    """For the given level, return list of (parent_bbox_or_None, children).
    `children` is a list of (quad (4,2), class_id) in reading order.

    - level=page: one entry with parent=None (full page), children=[page quad]
    - level=paragraph: parent = full page bbox, children = paragraphs
    - level=line: one entry PER paragraph, children = that paragraph's lines
    - level=word: one entry per line
    - level=char: one entry per word
    """
    from generate_training_data import char_to_class
    out = []
    if level == "page":
        xs, ys = [], []
        for para in page.paragraphs:
            for line in para["lines"]:
                for w in line["words"]:
                    for ch in w["characters"]:
                        b = ch["bbox"]
                        xs += [b[0], b[2]]
                        ys += [b[1], b[3]]
        if xs:
            out.append((None, [_bbox_quad_class(
                (min(xs), min(ys), max(xs), max(ys)), CLASS_PAGE)]))
    elif level == "paragraph":
        children = []
        for p in page.paragraphs:
            if not p.get("bbox"):
                continue
            if "quad" in p:
                q = torch.from_numpy(
                    np.array(p["quad"], dtype=np.float32).reshape(4, 2))
            else:
                q = bbox_to_quad(*p["bbox"])
            children.append((q, CLASS_PARAGRAPH))
        if children:
            out.append((None, children))
    elif level == "line":
        for para in page.paragraphs:
            if not para.get("bbox"):
                continue
            if "quad" in para:
                parent = np.array(para["quad"], dtype=np.float32).reshape(4, 2)
            else:
                parent = np.array(para["bbox"], dtype=np.float32)
            kids = []
            for line in para["lines"]:
                if "quad" in line:
                    q = torch.from_numpy(
                        np.array(line["quad"], dtype=np.float32).reshape(4, 2))
                    kids.append((q, CLASS_LINE))
                elif line.get("bbox"):
                    kids.append(_bbox_quad_class(line["bbox"], CLASS_LINE))
            if kids:
                out.append((parent, kids))
    elif level == "word":
        for para in page.paragraphs:
            for line in para["lines"]:
                if not line.get("bbox"):
                    continue
                parent = np.array(line["bbox"], dtype=np.float32)
                kids = [_bbox_quad_class(w["bbox"], CLASS_WORD)
                        for w in line["words"] if w.get("bbox")]
                if kids:
                    out.append((parent, kids))
    elif level == "char":
        for para in page.paragraphs:
            for line in para["lines"]:
                for w in line["words"]:
                    if not w.get("bbox"):
                        continue
                    parent = np.array(w["bbox"], dtype=np.float32)
                    kids = [_bbox_quad_class(c["bbox"], char_to_class(c["char"]))
                            for c in w["characters"]]
                    if kids:
                        out.append((parent, kids))
    return out


# ---------------------------------------------------------------------------
# Warping a parent region to RETINA_SIZE x RETINA_SIZE
# ---------------------------------------------------------------------------
def warp_parent_to_retina(img_pil, parent, pad_frac=0.05):
    """Warp the parent region to RETINA_SIZE × RETINA_SIZE.

    `parent` is either:
      - (x1, y1, x2, y2) axis-aligned bbox  → padded crop + resize
      - list/array of 4 (x, y) corner pairs  → perspective warp (teacher-forcing GT quad)

    Returns (warped_pil, M) where M is the 3×3 forward homography mapping
    page coordinates → warped coordinates.  Use _apply_h(M, pts) to transform
    any ground-truth child quads into the warped frame.
    """
    parent = np.asarray(parent, dtype=np.float64)
    W, H = img_pil.size

    if parent.ndim == 1 and parent.shape[0] == 4:
        # Axis-aligned bbox: padded crop + resize (same result as perspective warp)
        x1, y1, x2, y2 = parent
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        if x2 - x1 < 1 or y2 - y1 < 1:
            raise ValueError(f"Degenerate bbox: {parent}")
        pw = (x2 - x1) * pad_frac; ph = (y2 - y1) * pad_frac
        x1 = max(0.0, x1 - pw); y1 = max(0.0, y1 - ph)
        x2 = min(W, x2 + pw); y2 = min(H, y2 + ph)
        cw = x2 - x1; ch = y2 - y1
        sx = RETINA_SIZE / cw; sy = RETINA_SIZE / ch
        M = np.array([[sx, 0, -x1 * sx],
                      [0, sy, -y1 * sy],
                      [0,  0,  1       ]], dtype=np.float64)
        arr = np.array(img_pil)
        cropped = arr[int(y1):int(y2), int(x1):int(x2)]
        warped_arr = cv2.resize(cropped, (RETINA_SIZE, RETINA_SIZE),
                                interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(warped_arr), M

    # Non-axis-aligned quad: perspective warp (matches cascade inference exactly)
    quad = parent.reshape(4, 2)
    dst = np.array([[0, 0], [RETINA_SIZE, 0],
                    [RETINA_SIZE, RETINA_SIZE], [0, RETINA_SIZE]], dtype=np.float64)
    M = _homography_matrix(quad, dst)
    arr = np.array(img_pil)
    warped_arr = cv2.warpPerspective(arr, M, (RETINA_SIZE, RETINA_SIZE),
                                     flags=cv2.INTER_CUBIC)
    return Image.fromarray(warped_arr), M


def jitter_quad(q, rel_sigma=0.05, min_sigma=2.0):
    """Perturb a (4, 2) quad tensor with correlated + per-corner Gaussian
    noise scaled to the quad's size. Simulates the imperfect predicted quads
    the model sees in its mask/prev inputs at inference (exposure bias fix).
    """
    side = (q.max(dim=0).values - q.min(dim=0).values).clamp(min=1.0)
    sigma = (side * rel_sigma).clamp(min=min_sigma)          # (2,)
    corner_noise = torch.randn(4, 2) * sigma * 0.5
    shift = torch.randn(1, 2) * sigma
    return q + corner_noise + shift


def rasterize_quads_mask(quads, size=None):
    """Rasterize a list of (4, 2) quads into a (1, size, size) float mask."""
    size = size or RETINA_SIZE
    mask = np.zeros((size, size), dtype=np.uint8)
    if quads:
        polys = [np.round(np.asarray(q, dtype=np.float64)).astype(np.int32)
                 for q in quads]
        cv2.fillPoly(mask, polys, 1)
    return torch.from_numpy(mask).unsqueeze(0).float()


def transform_quad_to_warped(quad_or_pts, M):
    """Apply homography M to a (4, 2) quad or raw (N, 2) points → (4, 2) tensor."""
    pts = np.asarray(quad_or_pts, dtype=np.float64).reshape(-1, 2)
    return torch.from_numpy(_apply_h(M, pts)).float()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class AutoregQuadDataset(Dataset):
    LEVELS = AutoregQuadOCRNet.LEVELS

    def __init__(self, fonts, num_samples=40_000, page_pool=None,
                 affine=True, shear_deg=10.0, rotation_deg=15.0):
        self.fonts = fonts
        self.num_samples = num_samples
        self.page_pool = page_pool
        self.affine = affine
        self.shear_deg = shear_deg
        self.rotation_deg = rotation_deg

    def __len__(self):
        return self.num_samples

    def _gen_page(self):
        if self.page_pool:
            return random.choice(self.page_pool)
        return SyntheticPage(self.fonts, RETINA_SIZE, RETINA_SIZE,
                             rotate_paragraphs=True, compute_contours=False)

    def __getitem__(self, idx):
        # Try up to a few times to find a level with non-empty content
        for _ in range(5):
            page = self._gen_page()
            level = random.choice(self.LEVELS)
            level_id = self.LEVELS.index(level)
            parent_groups = collect_parents_and_children(page, level)
            if parent_groups:
                break
        else:
            # fallback: dummy sample (end-of-seq with zero prev)
            return self._dummy(level_id=0)

        parent, children = random.choice(parent_groups)

        # Build the input image: perspective-warp GT parent quad → RETINA_SIZE
        if parent is None:
            # page / paragraph levels: whole page is already RETINA_SIZE × RETINA_SIZE
            img_pil = page.image
            M = np.eye(3, dtype=np.float64)
        else:
            img_pil, M = warp_parent_to_retina(page.image, parent)

        # Transform children quads into warped space using forward homography M
        child_quads = [(transform_quad_to_warped(q, M), c) for q, c in children]

        # Pick autoregressive emission position k in [0, N]
        N = len(child_quads)
        k = random.randint(0, N)
        if k == 0:
            prev_quad = torch.zeros(4, 2)
            is_start = True
        else:
            prev_quad = child_quads[k - 1][0]
            is_start = False
        if k < N:
            target_quad, target_cls = child_quads[k]
            target_obj = 1.0
        else:
            target_quad = torch.zeros(4, 2)
            target_cls = CLASS_NONE
            target_obj = 0.0

        # To tensor
        img_t = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float() / 255.0
        if img_t.shape[0] == 4:
            img_t = img_t[:3]
        bg = estimate_bg_color(img_t)

        # Teacher-forcing history: all siblings emitted before position k
        history = [q for q, _ in child_quads[:k]]

        # Affine augment on top of the warp
        if self.affine:
            params = sample_affine_params(
                rotation=self.rotation_deg, shear=self.shear_deg,
                scale_range=(0.9, 1.1))
            img_t = apply_affine_image(img_t, params, bg)
            target_quad = apply_affine_points(target_quad, params)
            if not is_start:
                prev_quad = apply_affine_points(prev_quad, params)
            history = [apply_affine_points(q, params) for q in history]

        # Jitter the autoregressive inputs (mask + prev) to match the
        # imperfect predictions the model will feed itself at inference.
        # Targets stay exact.
        history = [jitter_quad(q) for q in history]
        if not is_start:
            prev_quad = jitter_quad(prev_quad)

        # 4th channel: mask of already-detected siblings
        mask = rasterize_quads_mask([q.numpy() for q in history])
        img_t = torch.cat([img_t, mask], dim=0)

        return (img_t, level_id, prev_quad, bool(is_start),
                target_quad, int(target_cls), float(target_obj))

    def _dummy(self, level_id):
        img = torch.full((4, RETINA_SIZE, RETINA_SIZE), 1.0)
        img[3] = 0.0
        return (img, level_id, torch.zeros(4, 2), True,
                torch.zeros(4, 2), CLASS_NONE, 0.0)


def collate(batch):
    imgs, lvls, prev, starts, tq, tc, to = zip(*batch)
    return (torch.stack(imgs), torch.tensor(lvls, dtype=torch.long),
            torch.stack(prev), torch.tensor(starts, dtype=torch.bool),
            torch.stack(tq), torch.tensor(tc, dtype=torch.long),
            torch.tensor(to, dtype=torch.float32))


# ---------------------------------------------------------------------------
# Quad rectification
# ---------------------------------------------------------------------------
# Levels 0=page,1=paragraph predict arbitrary quads; levels 2+=line/word/char
# are axis-aligned within their rectified parent, so we constrain NE and SW
# to be derived from NW and SE.  rect_mode is a (B,) float in [0,1]:
#   0 → use all 4 predicted corners  (page / paragraph)
#   1 → NE and SW derived from NW+SE (line / word / char)
RECT_MODE_BY_LEVEL = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])  # page,para,line,word,char


def rectify_quad(quad, rect_mode):
    """Interpolate between 4-point quad and axis-aligned 2-point rect.

    quad:      (B, 4, 2) in NW, NE, SE, SW order
    rect_mode: (B,) float tensor, 0=free quad, 1=axis-aligned rect

    For rect_mode=1 the NE and SW corners are fully derived from NW and SE:
        NE = (x_SE, y_NW)   SW = (x_NW, y_SE)
    Gradient flows only through NW and SE at rect_mode=1; NE/SW predictions
    get no gradient there and the model learns to ignore those two heads.
    """
    rm = rect_mode.view(-1, 1)          # (B, 1) for broadcasting over xy
    nw = quad[:, 0]                     # (B, 2)
    ne = quad[:, 1]
    se = quad[:, 2]
    sw = quad[:, 3]
    rect_ne = torch.stack([se[:, 0], nw[:, 1]], dim=1)   # (x_SE, y_NW)
    rect_sw = torch.stack([nw[:, 0], se[:, 1]], dim=1)   # (x_NW, y_SE)
    eff_ne = (1.0 - rm) * ne + rm * rect_ne
    eff_sw = (1.0 - rm) * sw + rm * rect_sw
    return torch.stack([nw, eff_ne, se, eff_sw], dim=1)  # (B, 4, 2)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def autoreg_loss(pred_quad, pred_cls, pred_obj,
                 gt_quad, gt_cls, gt_obj,
                 level_ids,
                 quad_weight=0.05, cls_weight=1.0, obj_weight=1.0):
    """Single-quad loss with per-level rectification.

    Quad + class are only supervised on positive (obj=1) samples.
    rect_mode is derived from level_ids: 0 for page/paragraph, 1 for line+.
    """
    rect_mode = RECT_MODE_BY_LEVEL.to(pred_quad.device)[level_ids]  # (B,)
    pred_quad_r = rectify_quad(pred_quad, rect_mode)
    gt_quad_r   = rectify_quad(gt_quad,   rect_mode)

    obj_mask = (gt_obj > 0.5)
    n_pos = obj_mask.sum().clamp(min=1)
    if obj_mask.any():
        quad_l1 = F.smooth_l1_loss(
            pred_quad_r[obj_mask], gt_quad_r[obj_mask], reduction="sum") / (
            n_pos * 8.0)
        cls_loss = F.cross_entropy(pred_cls[obj_mask], gt_cls[obj_mask])
    else:
        quad_l1 = torch.zeros((), device=pred_quad.device)
        cls_loss = torch.zeros((), device=pred_quad.device)

    obj_loss = F.binary_cross_entropy_with_logits(pred_obj, gt_obj)

    total = quad_weight * quad_l1 + cls_weight * cls_loss + obj_weight * obj_loss
    return total, {"quad": quad_l1.detach().item(),
                   "cls": cls_loss.detach().item(),
                   "obj": obj_loss.detach().item()}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def build_page_pool(fonts, n_pages):
    print(f"Pre-rendering {n_pages} pages…")
    import time
    t0 = time.time()
    pool = [SyntheticPage(fonts, RETINA_SIZE, RETINA_SIZE,
                          rotate_paragraphs=True, compute_contours=False)
            for _ in range(n_pages)]
    print(f"  done in {time.time() - t0:.1f}s")
    return pool


def train(args):
    from generate_training_data import NUM_CLASSES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    fonts = discover_fonts()
    print(f"Fonts: {len(fonts)}")

    pool = build_page_pool(fonts, args.page_pool)
    ds_train = AutoregQuadDataset(fonts, num_samples=args.samples,
                                  page_pool=pool,
                                  shear_deg=args.shear,
                                  rotation_deg=args.rotation)
    ds_val = AutoregQuadDataset(fonts, num_samples=max(200, args.samples // 20),
                                page_pool=pool,
                                shear_deg=args.shear,
                                rotation_deg=args.rotation)

    pw = args.num_workers > 0
    train_loader = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=pw,
        multiprocessing_context="spawn" if pw else None,
        prefetch_factor=4 if pw else None,
        collate_fn=collate, drop_last=True,
    )
    val_loader = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        num_workers=max(1, args.num_workers // 2), pin_memory=True,
        persistent_workers=pw,
        multiprocessing_context="spawn" if pw else None,
        collate_fn=collate,
    )

    model = AutoregQuadOCRNet(NUM_CLASSES).to(device)
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(n_gpus)))
        print(f"DataParallel across {n_gpus} GPUs")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        tot, n = 0.0, 0
        comp = {"quad": 0.0, "cls": 0.0, "obj": 0.0}
        for imgs, lvls, prev, starts, tq, tc, to_ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            lvls = lvls.to(device, non_blocking=True)
            prev = prev.to(device, non_blocking=True)
            starts = starts.to(device, non_blocking=True)
            tq = tq.to(device, non_blocking=True)
            tc = tc.to(device, non_blocking=True)
            to_ = to_.to(device, non_blocking=True)
            optim.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pq, pc, po = model(imgs, lvls, prev, starts)
                loss, parts = autoreg_loss(pq, pc, po, tq, tc, to_, lvls)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()
            tot += loss.item() * imgs.size(0)
            n += imgs.size(0)
            for k, v in parts.items():
                comp[k] += v * imgs.size(0)
        train_loss = tot / max(n, 1)

        model.eval()
        v_tot, v_n = 0.0, 0
        v_parts = {"quad": 0.0, "cls": 0.0, "obj": 0.0}
        with torch.no_grad():
            for imgs, lvls, prev, starts, tq, tc, to_ in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                lvls = lvls.to(device, non_blocking=True)
                prev = prev.to(device, non_blocking=True)
                starts = starts.to(device, non_blocking=True)
                tq = tq.to(device, non_blocking=True)
                tc = tc.to(device, non_blocking=True)
                to_ = to_.to(device, non_blocking=True)
                pq, pc, po = model(imgs, lvls, prev, starts)
                loss, parts = autoreg_loss(pq, pc, po, tq, tc, to_, lvls)
                v_tot += loss.item() * imgs.size(0)
                v_n += imgs.size(0)
                for k, v in parts.items():
                    v_parts[k] += v * imgs.size(0)
        val_loss = v_tot / max(v_n, 1)
        sched.step()

        print(f"[ep {epoch + 1}/{args.epochs}] train={train_loss:.4f}  "
              f"val={val_loss:.4f}  "
              f"trn_quad={comp['quad']/n:.3f} "
              f"trn_cls={comp['cls']/n:.3f} trn_obj={comp['obj']/n:.3f}  "
              f"val_quad={v_parts['quad']/v_n:.3f} "
              f"val_cls={v_parts['cls']/v_n:.3f} val_obj={v_parts['obj']/v_n:.3f}  "
              f"lr={optim.param_groups[0]['lr']:.2e}", flush=True)

        if val_loss < best_val:
            best_val = val_loss
            raw = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({
                "model_state_dict": raw.state_dict(),
                "num_classes": NUM_CLASSES,
                "epoch": epoch + 1,
                "val_loss": val_loss,
            }, args.output)
            print(f"  -> saved best (val={val_loss:.4f}) to {args.output}",
                  flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="model_05_autoreg.pth")
    ap.add_argument("--samples", type=int, default=40_000)
    ap.add_argument("--page-pool", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--shear", type=float, default=10.0)
    ap.add_argument("--rotation", type=float, default=15.0)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
