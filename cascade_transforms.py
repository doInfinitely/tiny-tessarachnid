"""
Cascade rotation normalization and backward transform utilities.

Each level of the 5-level OCR hierarchy (PAGE -> PARAGRAPH -> LINE -> WORD ->
CHARACTER) crops a detected region and feeds it to the next level. This module
provides:

  - Forward: crop, rotate upright, scale_and_pad into the 1024x1024 retina.
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
    angle_deg: float        # rotation angle passed to PIL (CCW positive)
    crop_w: float           # crop width (= rotate canvas width, expand=False)
    crop_h: float           # crop height (= rotate canvas height)
    scale: float            # scale_and_pad scale factor
    pad_ox: float           # retina padding offset x
    pad_oy: float           # retina padding offset y


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
# Point rotation (PIL expand=False convention)
# ---------------------------------------------------------------------------

def rotate_point(x: float, y: float, angle_deg: float,
                 w: float, h: float) -> tuple[float, float]:
    """Forward-rotate a point using PIL Image.rotate(angle, expand=False).

    Rotates CCW by angle_deg around the image center. Canvas stays w x h.
    """
    a = math.radians(angle_deg)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    dx = x - w / 2.0
    dy = y - h / 2.0
    rx = dx * cos_a - dy * sin_a + w / 2.0
    ry = dx * sin_a + dy * cos_a + h / 2.0
    return (rx, ry)


def inverse_rotate_point(rx: float, ry: float, angle_deg: float,
                         w: float, h: float) -> tuple[float, float]:
    """Inverse of rotate_point: rotated-image coords back to original."""
    a = math.radians(angle_deg)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    dx = rx - w / 2.0
    dy = ry - h / 2.0
    x = dx * cos_a + dy * sin_a + w / 2.0
    y = -dx * sin_a + dy * cos_a + h / 2.0
    return (x, y)


def rotate_points(points: list[tuple], angle_deg: float,
                  w: float, h: float) -> list[tuple[float, float]]:
    """Batch-rotate points using PIL expand=False convention."""
    return [rotate_point(x, y, angle_deg, w, h) for x, y in points]


# ---------------------------------------------------------------------------
# Forward cascade step
# ---------------------------------------------------------------------------

def forward_cascade_step(
    parent_image: Image.Image,
    contour: list[tuple],
    bg_color: tuple[int, int, int],
) -> tuple[Image.Image, CascadeTransform]:
    """Crop, rotate to upright, and scale_and_pad into a retina image.

    Steps:
      1. Crop contour bbox from parent image
      2. Rotate in-place by -orientation (expand=False)
      3. scale_and_pad into RETINA_SIZE x RETINA_SIZE

    Returns (retina_pil, CascadeTransform).
    """
    # 1. Crop contour bbox
    xs = [p[0] for p in contour]
    ys = [p[1] for p in contour]
    cx1, cy1, cx2, cy2 = min(xs), min(ys), max(xs), max(ys)
    w, h = parent_image.size
    cx1 = max(0, int(cx1))
    cy1 = max(0, int(cy1))
    cx2 = min(w, int(cx2))
    cy2 = min(h, int(cy2))
    if cx2 <= cx1 or cy2 <= cy1:
        retina, scale, pad_ox, pad_oy = scale_and_pad(
            Image.new("RGB", (1, 1), bg_color), bg_color)
        return retina, CascadeTransform(
            crop_ox=cx1, crop_oy=cy1, angle_deg=0.0,
            crop_w=1, crop_h=1, scale=scale, pad_ox=pad_ox, pad_oy=pad_oy,
        )

    crop = parent_image.crop((cx1, cy1, cx2, cy2))
    crop_w, crop_h = crop.size

    # 2. Rotate in-place to straighten text
    angle = compute_contour_orientation(contour)
    rot_angle = -angle
    if abs(rot_angle) < 0.5:
        rotated = crop
        rot_angle = 0.0
    else:
        rotated = crop.rotate(rot_angle, resample=Image.BICUBIC,
                              expand=False, fillcolor=bg_color)

    # 3. scale_and_pad
    retina, scale, pad_ox, pad_oy = scale_and_pad(rotated, bg_color)

    transform = CascadeTransform(
        crop_ox=cx1, crop_oy=cy1,
        angle_deg=rot_angle,
        crop_w=crop_w, crop_h=crop_h,
        scale=scale, pad_ox=pad_ox, pad_oy=pad_oy,
    )
    return retina, transform


# ---------------------------------------------------------------------------
# Backward (lift): retina coords -> parent-image coords
# ---------------------------------------------------------------------------

def lift_point(rx: float, ry: float,
               transform: CascadeTransform) -> tuple[float, float]:
    """Invert a single retina-space point back to parent-image coordinates.

    Steps: un-pad -> un-scale -> un-rotate -> un-crop
    """
    t = transform
    # 1. Un-pad
    x = rx - t.pad_ox
    y = ry - t.pad_oy
    # 2. Un-scale
    x = x / t.scale
    y = y / t.scale
    # 3. Un-rotate
    if abs(t.angle_deg) >= 0.5:
        x, y = inverse_rotate_point(x, y, t.angle_deg, t.crop_w, t.crop_h)
    # 4. Un-crop
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
# ---------------------------------------------------------------------------

def forward_contour(
    contour: list[tuple],
    transform: CascadeTransform,
) -> list[tuple[int, int]]:
    """Map a child contour (in page-global coords) through the forward transform
    into retina coordinates, for training target generation.

    Steps: crop-local -> rotate -> scale+pad
    """
    t = transform
    result = []
    for x, y in contour:
        # To crop-local coords
        lx = x - t.crop_ox
        ly = y - t.crop_oy
        # Rotate in-place
        if abs(t.angle_deg) >= 0.5:
            lx, ly = rotate_point(lx, ly, t.angle_deg, t.crop_w, t.crop_h)
        # Scale + pad
        rx = int(lx * t.scale + t.pad_ox)
        ry = int(ly * t.scale + t.pad_oy)
        result.append((rx, ry))
    return result
