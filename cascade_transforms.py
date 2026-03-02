"""
Cascade rotation normalization and backward transform utilities.

Each level of the 5-level OCR hierarchy (PAGE -> PARAGRAPH -> LINE -> WORD ->
CHARACTER) crops a detected region and feeds it to the next level. This module
provides:

  - Forward: crop contour bbox, rotate about contour centroid (expand=True),
    scale_and_pad into the 1024x1024 retina.
  - Backward (lift): invert all transforms (un-pad -> un-scale -> un-rotate
    -> un-crop) to map result contours back to original image coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image

from generate_training_data import RETINA_SIZE, scale_and_pad, contour_to_retina


# ---------------------------------------------------------------------------
# CascadeTransform record
# ---------------------------------------------------------------------------

@dataclass
class CascadeTransform:
    """Records every parameter of a single cascade step so we can invert it."""
    crop_ox: float          # crop left x in parent coords
    crop_oy: float          # crop top y in parent coords
    angle_deg: float        # rotation angle (CCW positive)
    crop_w: float           # crop width (pre-rotation)
    crop_h: float           # crop height (pre-rotation)
    scale: float            # scale_and_pad scale factor
    pad_ox: float           # retina padding offset x
    pad_oy: float           # retina padding offset y
    center_x: float = 0.0  # rotation center x in crop-local coords
    center_y: float = 0.0  # rotation center y in crop-local coords


# ---------------------------------------------------------------------------
# Orientation detection
# ---------------------------------------------------------------------------

def compute_contour_orientation(contour: list[tuple]) -> float:
    """Compute the dominant text orientation from a contour polygon.

    Returns the angle in degrees of the longest edge, normalized to [-90, 90).
    0 means horizontal text, positive is CCW.
    """
    if len(contour) < 2:
        return 0.0
    best_len = 0.0
    best_angle = 0.0
    n = len(contour)
    for i in range(n):
        x1, y1 = contour[i]
        x2, y2 = contour[(i + 1) % n]
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length > best_len:
            best_len = length
            best_angle = math.degrees(math.atan2(dy, dx))
    # Normalize to [-90, 90)
    while best_angle >= 90:
        best_angle -= 180
    while best_angle < -90:
        best_angle += 180
    return best_angle


# ---------------------------------------------------------------------------
# Point rotation about an arbitrary center
# ---------------------------------------------------------------------------

def rotate_point(x: float, y: float, angle_deg: float,
                 cx: float, cy: float) -> tuple[float, float]:
    """Rotate a point CCW by angle_deg about (cx, cy)."""
    a = math.radians(angle_deg)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    dx = x - cx
    dy = y - cy
    rx = dx * cos_a - dy * sin_a + cx
    ry = dx * sin_a + dy * cos_a + cy
    return (rx, ry)


def inverse_rotate_point(rx: float, ry: float, angle_deg: float,
                         cx: float, cy: float) -> tuple[float, float]:
    """Inverse of rotate_point: un-rotate about (cx, cy)."""
    a = math.radians(angle_deg)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    dx = rx - cx
    dy = ry - cy
    x = dx * cos_a + dy * sin_a + cx
    y = -dx * sin_a + dy * cos_a + cy
    return (x, y)


def _expanded_bounds(crop_w: float, crop_h: float, angle_deg: float,
                     cx: float, cy: float) -> tuple[float, float, float, float]:
    """Compute AABB of the crop corners after rotation about (cx, cy).

    Returns (min_x, min_y, exp_w, exp_h) in the rotated frame.
    """
    corners = [(0, 0), (crop_w, 0), (crop_w, crop_h), (0, crop_h)]
    rot = [rotate_point(x, y, angle_deg, cx, cy) for x, y in corners]
    rxs = [p[0] for p in rot]
    rys = [p[1] for p in rot]
    min_x = min(rxs)
    min_y = min(rys)
    return (min_x, min_y, max(rxs) - min_x, max(rys) - min_y)


# ---------------------------------------------------------------------------
# Forward cascade step: crop -> rotate(expand=True) -> scale -> pad
# ---------------------------------------------------------------------------

def forward_cascade_step(
    parent_image: Image.Image,
    contour: list[tuple],
    bg_color: tuple[int, int, int],
) -> tuple[Image.Image, CascadeTransform]:
    """Crop contour bbox, rotate about centroid (expand=True), scale_and_pad.

    Steps:
      1. Crop contour AABB from parent image
      2. Rotate crop about contour centroid (expand=True via affine)
      3. scale_and_pad into RETINA_SIZE x RETINA_SIZE

    Returns (retina_pil, CascadeTransform).
    """
    pts = [(float(x), float(y)) for x, y in contour]

    # 1. AABB of contour + crop
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    bx1 = int(min(xs))
    by1 = int(min(ys))
    bx2 = int(math.ceil(max(xs)))
    by2 = int(math.ceil(max(ys)))

    if bx2 <= bx1 or by2 <= by1:
        retina, scale, pad_ox, pad_oy = scale_and_pad(
            Image.new("RGB", (1, 1), bg_color), bg_color)
        return retina, CascadeTransform(
            crop_ox=bx1, crop_oy=by1, angle_deg=0.0,
            crop_w=1, crop_h=1, scale=scale, pad_ox=pad_ox, pad_oy=pad_oy,
        )

    crop = parent_image.crop((bx1, by1, bx2, by2))
    crop_w, crop_h = crop.size

    # Contour centroid in crop-local coords
    cx = sum(xs) / len(xs) - bx1
    cy = sum(ys) / len(ys) - by1

    # 2. Rotate about centroid
    angle = compute_contour_orientation(contour)
    rot_angle = -angle
    if abs(rot_angle) < 0.5:
        rot_angle = 0.0
        rotated = crop
    else:
        # Compute expanded bounds after rotation about centroid
        min_x, min_y, exp_w, exp_h = _expanded_bounds(
            crop_w, crop_h, rot_angle, cx, cy)
        out_w = max(1, int(math.ceil(exp_w)))
        out_h = max(1, int(math.ceil(exp_h)))

        # Build affine: output (rotated) -> input (crop)
        # For each output pixel (ox, oy), find input pixel (ix, iy):
        #   world_x = ox + min_x   (shift output origin to world)
        #   un-rotate about (cx, cy) -> crop-local
        a = math.radians(rot_angle)
        cos_a = math.cos(a)
        sin_a = math.sin(a)
        # ix = cos_a*(ox+min_x-cx) + sin_a*(oy+min_y-cy) + cx
        # iy = -sin_a*(ox+min_x-cx) + cos_a*(oy+min_y-cy) + cy
        aff_a = cos_a
        aff_b = sin_a
        aff_c = (min_x - cx) * cos_a + (min_y - cy) * sin_a + cx
        aff_d = -sin_a
        aff_e = cos_a
        aff_f = -(min_x - cx) * sin_a + (min_y - cy) * cos_a + cy

        rotated = crop.transform(
            (out_w, out_h), Image.AFFINE,
            (aff_a, aff_b, aff_c, aff_d, aff_e, aff_f),
            resample=Image.BICUBIC, fillcolor=bg_color,
        )

    # 3. scale_and_pad
    retina, scale, pad_ox, pad_oy = scale_and_pad(rotated, bg_color)

    transform = CascadeTransform(
        crop_ox=bx1, crop_oy=by1,
        angle_deg=rot_angle,
        crop_w=crop_w, crop_h=crop_h,
        scale=scale, pad_ox=pad_ox, pad_oy=pad_oy,
        center_x=cx, center_y=cy,
    )
    return retina, transform


# ---------------------------------------------------------------------------
# Backward (lift): retina coords -> parent-image coords
# Order: un-pad -> un-scale -> un-rotate -> un-crop
# ---------------------------------------------------------------------------

def _retina_to_rotated(rx: float, ry: float,
                       t: CascadeTransform) -> tuple[float, float]:
    """Map retina coords to rotated-crop (expanded) coords."""
    # Un-pad
    x = rx - t.pad_ox
    y = ry - t.pad_oy
    # Un-scale
    x = x / t.scale
    y = y / t.scale
    # Shift to world coords of the expanded image
    if abs(t.angle_deg) >= 0.5:
        min_x, min_y, _, _ = _expanded_bounds(
            t.crop_w, t.crop_h, t.angle_deg, t.center_x, t.center_y)
        x = x + min_x
        y = y + min_y
    return (x, y)


def lift_point(rx: float, ry: float,
               transform: CascadeTransform) -> tuple[float, float]:
    """Invert a single retina-space point back to parent-image coordinates.

    Steps: un-pad -> un-scale -> un-rotate about centroid -> un-crop
    """
    t = transform
    # 1. Un-pad
    x = rx - t.pad_ox
    y = ry - t.pad_oy
    # 2. Un-scale
    x = x / t.scale
    y = y / t.scale
    # 3. Un-rotate about centroid (expanded coords -> crop-local coords)
    if abs(t.angle_deg) >= 0.5:
        min_x, min_y, _, _ = _expanded_bounds(
            t.crop_w, t.crop_h, t.angle_deg, t.center_x, t.center_y)
        # Map from expanded-image pixel coords to rotated world coords
        x = x + min_x
        y = y + min_y
        # Un-rotate about centroid
        x, y = inverse_rotate_point(x, y, t.angle_deg, t.center_x, t.center_y)
    # 4. Un-crop (crop-local -> parent coords)
    x = x + t.crop_ox
    y = y + t.crop_oy
    return (x, y)


def lift_contour(contour: list[tuple],
                 transform: CascadeTransform) -> list[tuple[int, int]]:
    """Lift all contour points from retina coords to parent-image coords."""
    return [(int(round(x)), int(round(y)))
            for x, y in (lift_point(px, py, transform)
                         for px, py in contour)]


# ---------------------------------------------------------------------------
# Forward contour mapping (for training): child contour -> retina coords
# Order: crop -> rotate about centroid -> scale+pad
# ---------------------------------------------------------------------------

def forward_contour(
    contour: list[tuple],
    transform: CascadeTransform,
) -> list[tuple[int, int]]:
    """Map a child contour (in page-global coords) through the forward transform
    into retina coordinates, for training target generation.

    Steps: crop-local -> rotate about centroid -> scale+pad
    """
    t = transform
    result = []
    for x, y in contour:
        # 1. Crop-local (parent -> crop coords)
        lx = x - t.crop_ox
        ly = y - t.crop_oy
        # 2. Rotate about centroid (crop-local -> rotated coords)
        if abs(t.angle_deg) >= 0.5:
            lx, ly = rotate_point(lx, ly, t.angle_deg, t.center_x, t.center_y)
            # Shift to expanded-image pixel coords
            min_x, min_y, _, _ = _expanded_bounds(
                t.crop_w, t.crop_h, t.angle_deg, t.center_x, t.center_y)
            lx = lx - min_x
            ly = ly - min_y
        # 3. Scale + pad
        rx = int(lx * t.scale + t.pad_ox)
        ry = int(ly * t.scale + t.pad_oy)
        result.append((rx, ry))
    return result
