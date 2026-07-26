"""Cascade inference for the autoregressive V5 detector.

Runs: page -> warp -> paragraphs -> warp -> lines -> warp -> words on a test
page. Each predicted trapezoid is perspective-warped to an axis-aligned 1024x
1024 rectangle before the next level's calls. At the end, every detected quad
is transformed back to original page coordinates via the inverse transform
stack and visualized on the input image.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser("~/Code/glyph-faerie"))
sys.path.insert(0, HERE)

import numpy as np
import torch
from PIL import Image, ImageDraw


def homography_matrix(src, dst):
    """3x3 forward homography M such that, in homogeneous coords,
    M @ [x_src, y_src, 1]^T  ~  [x_dst, y_dst, 1]^T.
    src, dst: (4, 2) arrays.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    A = []
    b = []
    for (x, y), (xp, yp) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -x * xp, -y * xp])
        A.append([0, 0, 0, x, y, 1, -x * yp, -y * yp])
        b.extend([xp, yp])
    h = np.linalg.solve(np.asarray(A), np.asarray(b))
    return np.array([[h[0], h[1], h[2]],
                     [h[3], h[4], h[5]],
                     [h[6], h[7], 1.0]], dtype=np.float64)


def pil_warp_perspective(pil, src_quad, out_size):
    """Warp pil so that `src_quad` (in pil coords) maps to the axis-aligned
    rectangle (0,0)-(out_size, out_size). Returns (warped_pil, M_forward).
    """
    dst = np.array([[0, 0], [out_size, 0],
                    [out_size, out_size], [0, out_size]], dtype=np.float64)
    M = homography_matrix(src_quad, dst)
    # PIL's PERSPECTIVE expects the 8 coeffs of the INVERSE map (dst -> src)
    M_inv = np.linalg.inv(M)
    M_inv = M_inv / M_inv[2, 2]
    coeffs = (M_inv[0, 0], M_inv[0, 1], M_inv[0, 2],
              M_inv[1, 0], M_inv[1, 1], M_inv[1, 2],
              M_inv[2, 0], M_inv[2, 1])
    warped = pil.transform((out_size, out_size), Image.PERSPECTIVE, coeffs,
                            Image.BICUBIC)
    return warped, M

from generate_training_data import (
    NUM_CLASSES, SyntheticPage, discover_fonts,
)
from train_05_autoreg import AutoregQuadOCRNet, RETINA_SIZE

LEVELS = ["page", "paragraph", "line", "word", "char"]
LEVEL_COLORS = {
    "page": (255, 0, 0),
    "paragraph": (0, 200, 0),
    "line": (0, 0, 255),
    "word": (255, 165, 0),
    "char": (200, 0, 200),
}


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = AutoregQuadOCRNet(ck["num_classes"]).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model


def img_to_tensor(pil, device):
    arr = np.array(pil.convert("RGB"))
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(device)


def to_canonical_rect(quad):
    """Sort the 4 quad points into NW/NE/SE/SW slots. The model is trained
    to output them already in this order; we just ensure it for safety.
    """
    pts = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    # sort by y then x to get top pair vs bottom pair
    s = pts[np.argsort(pts[:, 1])]
    top, bot = s[:2], s[2:]
    top = top[np.argsort(top[:, 0])]
    bot = bot[np.argsort(bot[:, 0])]
    nw, ne = top[0], top[1]
    sw, se = bot[0], bot[1]
    return np.stack([nw, ne, se, sw]).astype(np.float32)


def warp_to_rect(parent_pil, quad):
    """Perspective-warp the parent image so that `quad` maps to the axis-
    aligned RETINA_SIZE x RETINA_SIZE rectangle. Returns (warped_pil, M)
    where M is the 3x3 homography (parent -> warped).
    """
    src = to_canonical_rect(quad)
    return pil_warp_perspective(parent_pil, src, RETINA_SIZE)


def apply_h(M_inv, pts):
    """Apply 3x3 homography to (N, 2) points -> (N, 2)."""
    pts = np.asarray(pts, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1))
    h = np.concatenate([pts, ones], axis=1)            # (N, 3)
    out = h @ M_inv.T                                   # (N, 3)
    return (out[:, :2] / out[:, 2:3]).astype(np.float32)


def _quad_iou(a, b):
    """Axis-aligned IoU between two (4,2) quads via their bounding boxes."""
    ax1, ay1 = a[:, 0].min(), a[:, 1].min()
    ax2, ay2 = a[:, 0].max(), a[:, 1].max()
    bx1, by1 = b[:, 0].min(), b[:, 1].min()
    bx2, by2 = b[:, 0].max(), b[:, 1].max()
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


@torch.no_grad()
def emit_children(model, warped_pil, level, device,
                  max_children=64, obj_threshold=0.5, loop_iou=0.7):
    """Autoregressively emit children quads of one parent (the input pil
    is the warped parent at RETINA_SIZE x RETINA_SIZE). Returns list of
    (quad (4,2) in warped coords, class_id, confidence).

    Each emitted quad is painted into the mask channel (4th input channel)
    so the model sees what it has already detected. Stops when obj <
    obj_threshold OR the new prediction overlaps the previous by more than
    loop_iou (the model entered a fixed-point loop).
    """
    import cv2
    level_id = LEVELS.index(level)
    rgb_t = img_to_tensor(warped_pil, device)          # (1, 3, S, S)
    mask_np = np.zeros((RETINA_SIZE, RETINA_SIZE), dtype=np.uint8)
    lvl_t = torch.tensor([level_id], dtype=torch.long, device=device)
    quads = []
    prev = torch.zeros(1, 4, 2, device=device)
    prev_q_np = None
    is_start = torch.tensor([True], dtype=torch.bool, device=device)
    for _ in range(max_children):
        mask_t = torch.from_numpy(mask_np).float().to(device)
        img_t = torch.cat([rgb_t, mask_t.view(1, 1, *mask_t.shape)], dim=1)
        pq, pc, po = model(img_t, lvl_t, prev, is_start)
        obj_p = torch.sigmoid(po).item()
        if obj_p < obj_threshold:
            break
        q = pq.squeeze(0).cpu().numpy()
        if prev_q_np is not None and _quad_iou(q, prev_q_np) > loop_iou:
            break
        cls = pc.argmax(dim=-1).item()
        conf = torch.softmax(pc, dim=-1).max().item()
        quads.append((q, cls, conf, obj_p))
        cv2.fillPoly(mask_np, [np.round(q).astype(np.int32)], 1)
        prev = pq
        prev_q_np = q
        is_start = torch.tensor([False], dtype=torch.bool, device=device)
    return quads


def cascade(model, page_pil, device, max_depth=4, obj_threshold=0.5):
    """Run the cascade. Returns list of dicts:
        {level, quad_page_space (4,2), parent_idx, class_id, conf}
    """
    detections = []

    # Step 1: detect the page region from the whole page image.
    # Resize page to RETINA_SIZE first so the input matches training.
    W, H = page_pil.size
    page_resized = page_pil.resize((RETINA_SIZE, RETINA_SIZE), Image.BICUBIC)
    M_page_to_retina = np.array([[RETINA_SIZE / W, 0, 0],
                                  [0, RETINA_SIZE / H, 0],
                                  [0, 0, 1]], dtype=np.float64)
    page_kids = emit_children(model, page_resized, "page", device,
                              max_children=4, obj_threshold=obj_threshold)
    if not page_kids:
        return detections
    page_quad_retina = page_kids[0][0]
    page_quad_page = apply_h(np.linalg.inv(M_page_to_retina),
                              to_canonical_rect(page_quad_retina))
    detections.append({"level": "page", "quad": page_quad_page,
                       "parent_idx": -1, "class_id": page_kids[0][1],
                       "conf": page_kids[0][2]})

    # Step 2..N: recurse from the page region down to the requested depth.
    # Each frontier item carries: (parent_det_index, parent_pil, M_parent_to_page_inv)
    page_warped, M_page = warp_to_rect(page_pil, page_quad_page)
    M_page_inv = np.linalg.inv(M_page)
    frontier = [(len(detections) - 1, page_warped, M_page_inv)]

    next_level = {"page": "paragraph", "paragraph": "line",
                  "line": "word", "word": "char"}
    cur_level = "paragraph"
    for depth in range(max_depth):
        new_frontier = []
        for parent_idx, parent_pil, M_inv in frontier:
            kids = emit_children(model, parent_pil, cur_level, device,
                                  max_children=64,
                                  obj_threshold=obj_threshold)
            for q_warped, cls, conf, _ in kids:
                q_canon = to_canonical_rect(q_warped)
                if cur_level == "paragraph":
                    # Paragraph quads may be rotated/sheared — use the actual
                    # quad for perspective warp so next level sees upright text.
                    q_for_warp = q_canon
                    q_page = apply_h(M_inv, q_canon)
                else:
                    # Lines/words are trained as axis-aligned rects; snap so
                    # child warp is a clean rectangle.
                    xs, ys = q_canon[:, 0], q_canon[:, 1]
                    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                    q_for_warp = np.array([[x1, y1], [x2, y1],
                                           [x2, y2], [x1, y2]], dtype=np.float32)
                    q_page = apply_h(M_inv, q_for_warp)
                det_idx = len(detections)
                detections.append({"level": cur_level, "quad": q_page,
                                   "parent_idx": parent_idx,
                                   "class_id": cls, "conf": conf})
                child_warped, M_child_in_parent = warp_to_rect(
                    parent_pil, q_for_warp)
                M_child_to_page_inv = M_inv @ np.linalg.inv(
                    M_child_in_parent)
                new_frontier.append(
                    (det_idx, child_warped, M_child_to_page_inv))
        frontier = new_frontier
        if cur_level == "word":
            break
        cur_level = next_level[cur_level]
    return detections


def visualize(page_pil, detections, out_path):
    img = page_pil.copy().convert("RGB")
    d = ImageDraw.Draw(img)
    for det in detections:
        col = LEVEL_COLORS.get(det["level"], (128, 128, 128))
        q = det["quad"]
        pts = [(float(x), float(y)) for x, y in q]
        d.line(pts + [pts[0]], fill=col, width=2)
    img.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_05_autoreg.pth")
    ap.add_argument("--out", default="cascade_out.png")
    ap.add_argument("--obj-threshold", type=float, default=0.5)
    ap.add_argument("--max-depth", type=int, default=4,
                    help="0=page only, 1=paragraphs, 2=lines, 3=words, 4=chars")
    ap.add_argument("--page-img", default=None,
                    help="path to an existing image; if omitted, render one")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)
    print(f"Loaded {args.model} -> {device}")

    if args.page_img:
        page_pil = Image.open(args.page_img).convert("RGB")
    else:
        fonts = discover_fonts()
        print(f"Rendering a synthetic page (1024x1024) …")
        page = SyntheticPage(fonts, RETINA_SIZE, RETINA_SIZE)
        page_pil = page.image
    print(f"Page size: {page_pil.size}")

    dets = cascade(model, page_pil, device,
                   max_depth=args.max_depth,
                   obj_threshold=args.obj_threshold)
    print(f"\nDetections: {len(dets)}")
    by_lvl = {}
    for d in dets:
        by_lvl.setdefault(d["level"], 0)
        by_lvl[d["level"]] += 1
    for lvl in LEVELS:
        if lvl in by_lvl:
            print(f"  {lvl:>10s}: {by_lvl[lvl]}")

    visualize(page_pil, dets, args.out)
    print(f"\nVisualization saved to {args.out}")


if __name__ == "__main__":
    main()
