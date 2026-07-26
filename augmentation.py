"""
Training-time augmentation: composable pipeline of geometric and photometric
transforms with coordinate-aware application for bbox, sequence, and contour modes.

Provides ``AugmentationPipeline`` — a composable list of stages — and
``AugmentedDataset`` — a unified wrapper that applies the pipeline and
transforms coordinates accordingly.
"""

import io
import math
import random

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms

from generate_training_data import CLASS_NONE, RETINA_SIZE


# ---------------------------------------------------------------------------
# Geometric augmentation helpers
# ---------------------------------------------------------------------------

def estimate_bg_color(img_t: torch.Tensor) -> tuple[int, int, int]:
    """Estimate background colour from the 8-px border strip of a CHW [0,1] tensor."""
    c, h, w = img_t.shape
    border = 8
    strips = torch.cat([
        img_t[:, :border, :].reshape(c, -1),        # top
        img_t[:, -border:, :].reshape(c, -1),        # bottom
        img_t[:, border:-border, :border].reshape(c, -1),  # left
        img_t[:, border:-border, -border:].reshape(c, -1), # right
    ], dim=1)
    median = strips.median(dim=1).values  # (C,)
    return tuple(int(v * 255 + 0.5) for v in median.tolist())


def sample_affine_params(
    rotation: float = 5.0,
    scale_range: tuple[float, float] = (0.9, 1.1),
    translate: float = 20.0,
    shear: float = 3.0,
) -> dict:
    """Sample random affine parameters and compute the 3x3 forward matrix.

    Transform is centred on (RETINA_SIZE/2, RETINA_SIZE/2).
    Returns dict with forward_matrix (3x3), linear_part (2x2), is_perspective=False.
    """
    cx = cy = RETINA_SIZE / 2.0

    angle = random.uniform(-rotation, rotation)
    scale = random.uniform(*scale_range)
    tx = random.uniform(-translate, translate)
    ty = random.uniform(-translate, translate)
    shear_x = random.uniform(-shear, shear)
    shear_y = random.uniform(-shear, shear)

    rad = math.radians(angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    sx_rad, sy_rad = math.radians(shear_x), math.radians(shear_y)

    # Rotation + scale
    R = np.array([
        [cos_a * scale, -sin_a * scale],
        [sin_a * scale,  cos_a * scale],
    ])
    # Shear
    S = np.array([
        [1.0, math.tan(sx_rad)],
        [math.tan(sy_rad), 1.0],
    ])
    # Combined linear part
    L = R @ S  # 2x2

    # Full 3x3 forward matrix: translate to origin, apply L, translate back + shift
    M = np.eye(3)
    M[:2, :2] = L
    M[0, 2] = -cx * L[0, 0] - cy * L[0, 1] + cx + tx
    M[1, 2] = -cx * L[1, 0] - cy * L[1, 1] + cy + ty

    return {
        "angle": angle,
        "scale": scale,
        "tx": tx,
        "ty": ty,
        "shear_x": shear_x,
        "shear_y": shear_y,
        "forward_matrix": M,
        "linear_part": L,
        "is_perspective": False,
    }


def compose_affine(params_list: list[dict]) -> dict:
    """Compose multiple affine transforms via matrix multiplication.

    Returns a dict with ``forward_matrix`` and ``linear_part`` suitable for
    ``apply_affine_image`` / ``apply_affine_points`` etc.
    """
    M = np.eye(3)
    L = np.eye(2)
    for p in params_list:
        M = p["forward_matrix"] @ M
        L = p["linear_part"] @ L
    return {"forward_matrix": M, "linear_part": L}


def _compute_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Compute 3x3 homography from 4 point pairs via Direct Linear Transform.

    Parameters
    ----------
    src, dst : ndarray of shape (4, 2)

    Returns
    -------
    H : ndarray of shape (3, 3)
        Homography such that ``dst ~ H @ [src; 1]`` in homogeneous coords.
    """
    A = []
    for (xs, ys), (xd, yd) in zip(src, dst):
        A.append([-xs, -ys, -1, 0, 0, 0, xd * xs, xd * ys, xd])
        A.append([0, 0, 0, -xs, -ys, -1, yd * xs, yd * ys, yd])
    A = np.array(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


def sample_perspective_params(magnitude: float = 0.06) -> dict:
    """Sample a random perspective warp.

    Each image corner is displaced by up to ``magnitude * RETINA_SIZE`` pixels.
    Returns dict with ``forward_matrix`` (3x3) and ``is_perspective=True``.
    """
    sz = float(RETINA_SIZE)
    src = np.array([[0, 0], [sz, 0], [sz, sz], [0, sz]], dtype=np.float64)
    offsets = np.random.uniform(-magnitude * sz, magnitude * sz, size=(4, 2))
    dst = src + offsets
    H = _compute_homography(src, dst)
    return {"forward_matrix": H, "is_perspective": True}


def compose_transforms(params_list: list[dict]) -> dict:
    """Compose multiple geometric transforms (affine and/or perspective).

    Multiplies 3x3 forward matrices.  Tracks ``is_perspective`` — True if any
    input is perspective.  Only emits ``linear_part`` when purely affine.
    """
    M = np.eye(3)
    is_perspective = False
    for p in params_list:
        M = p["forward_matrix"] @ M
        if p.get("is_perspective", False):
            is_perspective = True
    result: dict = {"forward_matrix": M, "is_perspective": is_perspective}
    if not is_perspective:
        result["linear_part"] = M[:2, :2].copy()
    return result


# ---------------------------------------------------------------------------
# Affine coordinate transforms (backward-compatible)
# ---------------------------------------------------------------------------

def apply_affine_image(
    img_t: torch.Tensor,
    params: dict,
    bg_color: tuple[int, int, int],
) -> torch.Tensor:
    """Apply affine transform to a CHW [0,1] float tensor via PIL.

    Returns a new CHW [0,1] tensor.
    """
    img_pil = transforms.ToPILImage()(img_t)
    M = params["forward_matrix"]

    # PIL Image.transform expects the *inverse* 2x3 matrix (maps output→input)
    M_inv = np.linalg.inv(M)
    coeffs = (
        M_inv[0, 0], M_inv[0, 1], M_inv[0, 2],
        M_inv[1, 0], M_inv[1, 1], M_inv[1, 2],
    )
    img_pil = img_pil.transform(
        img_pil.size, Image.AFFINE, coeffs,
        resample=Image.BICUBIC, fillcolor=bg_color,
    )
    return transforms.ToTensor()(img_pil)


def apply_affine_points(points: torch.Tensor, params: dict) -> torch.Tensor:
    """Transform (N, 2) xy-points by the forward matrix, clamp to [0, RETINA_SIZE].

    Returns new (N, 2) tensor.
    """
    M = params["forward_matrix"]
    L = torch.tensor(M[:2, :2], dtype=points.dtype)
    t = torch.tensor(M[:2, 2], dtype=points.dtype)
    out = points @ L.T + t
    return out.clamp(0, RETINA_SIZE)


def apply_affine_bbox_6dim(tensor: torch.Tensor, params: dict) -> torch.Tensor:
    """Transform 6-dim bbox rows: (x1, y1, x2, y2, class_id, is_handwritten).

    For each non-CLASS_NONE row, transforms the 4 corners of the bbox through
    the affine, then recomputes the axis-aligned bounding box.
    Supports both 1-D (6,) and 2-D (N, 6) inputs.
    """
    single = tensor.dim() == 1
    if single:
        tensor = tensor.unsqueeze(0)

    out = tensor.clone()
    for i in range(tensor.shape[0]):
        class_id = int(tensor[i, 4].item())
        if class_id == CLASS_NONE:
            continue

        x1, y1, x2, y2 = tensor[i, :4]
        corners = torch.tensor([
            [x1, y1], [x2, y1], [x2, y2], [x1, y2],
        ], dtype=tensor.dtype)
        corners = apply_affine_points(corners, params)
        out[i, 0] = corners[:, 0].min()
        out[i, 1] = corners[:, 1].min()
        out[i, 2] = corners[:, 0].max()
        out[i, 3] = corners[:, 1].max()

    if single:
        out = out.squeeze(0)
    return out


def apply_affine_contour_5dim(tensor: torch.Tensor, params: dict) -> torch.Tensor:
    """Transform 5-dim contour rows: (x, y, dx, dy, class_id).

    For each non-CLASS_NONE row:
    - (x, y) are transformed as points
    - (dx, dy) are rotated by the 2x2 linear part and renormalized

    Supports both 1-D (5,) and 2-D (N, 5) inputs.
    """
    single = tensor.dim() == 1
    if single:
        tensor = tensor.unsqueeze(0)

    out = tensor.clone()
    L = torch.tensor(params["linear_part"], dtype=tensor.dtype)

    for i in range(tensor.shape[0]):
        class_id = int(tensor[i, 4].item())
        if class_id == CLASS_NONE:
            continue

        # Transform position
        pt = tensor[i, :2].unsqueeze(0)  # (1, 2)
        pt = apply_affine_points(pt, params)
        out[i, 0] = pt[0, 0]
        out[i, 1] = pt[0, 1]

        # Rotate orientation vector
        dv = tensor[i, 2:4]  # (dx, dy)
        dv_rot = L @ dv
        norm = dv_rot.norm()
        if norm > 1e-8:
            dv_rot = dv_rot / norm
        out[i, 2] = dv_rot[0]
        out[i, 3] = dv_rot[1]

    if single:
        out = out.squeeze(0)
    return out


# ---------------------------------------------------------------------------
# Unified geometric transforms (affine + perspective)
# ---------------------------------------------------------------------------

def _apply_projective_points(points: torch.Tensor, M: np.ndarray) -> torch.Tensor:
    """Apply projective transform to (N, 2) points with perspective divide.

    Returns (N, 2) clamped to [0, RETINA_SIZE].
    """
    N = points.shape[0]
    ones = torch.ones(N, 1, dtype=points.dtype)
    hom = torch.cat([points, ones], dim=1)  # (N, 3)
    M_t = torch.tensor(M, dtype=points.dtype)
    out_hom = hom @ M_t.T  # (N, 3)
    w = out_hom[:, 2:3].clamp(min=1e-8)
    out = out_hom[:, :2] / w
    return out.clamp(0, RETINA_SIZE)


def apply_geo_points(points: torch.Tensor, params: dict) -> torch.Tensor:
    """Transform (N, 2) points — dispatches affine vs projective."""
    if params.get("is_perspective", False):
        return _apply_projective_points(points, params["forward_matrix"])
    return apply_affine_points(points, params)


def apply_geo_image(
    img_t: torch.Tensor,
    params: dict,
    bg_color: tuple[int, int, int],
) -> torch.Tensor:
    """Apply geometric transform (affine or perspective) to a CHW [0,1] tensor."""
    img_pil = transforms.ToPILImage()(img_t)
    M = params["forward_matrix"]
    M_inv = np.linalg.inv(M)

    if params.get("is_perspective", False):
        # PIL PERSPECTIVE: 8 coefficients for inverse map, normalized so [2,2]=1
        s = M_inv[2, 2]
        coeffs = (
            M_inv[0, 0] / s, M_inv[0, 1] / s, M_inv[0, 2] / s,
            M_inv[1, 0] / s, M_inv[1, 1] / s, M_inv[1, 2] / s,
            M_inv[2, 0] / s, M_inv[2, 1] / s,
        )
        img_pil = img_pil.transform(
            img_pil.size, Image.PERSPECTIVE, coeffs,
            resample=Image.BICUBIC, fillcolor=bg_color,
        )
    else:
        coeffs = (
            M_inv[0, 0], M_inv[0, 1], M_inv[0, 2],
            M_inv[1, 0], M_inv[1, 1], M_inv[1, 2],
        )
        img_pil = img_pil.transform(
            img_pil.size, Image.AFFINE, coeffs,
            resample=Image.BICUBIC, fillcolor=bg_color,
        )
    return transforms.ToTensor()(img_pil)


def apply_geo_bbox_6dim(tensor: torch.Tensor, params: dict) -> torch.Tensor:
    """Transform 6-dim bbox rows using unified geo transforms.

    Supports both 1-D (6,) and 2-D (N, 6) inputs.
    """
    single = tensor.dim() == 1
    if single:
        tensor = tensor.unsqueeze(0)

    out = tensor.clone()
    for i in range(tensor.shape[0]):
        class_id = int(tensor[i, 4].item())
        if class_id == CLASS_NONE:
            continue

        x1, y1, x2, y2 = tensor[i, :4]
        corners = torch.tensor([
            [x1, y1], [x2, y1], [x2, y2], [x1, y2],
        ], dtype=tensor.dtype)
        corners = apply_geo_points(corners, params)
        out[i, 0] = corners[:, 0].min()
        out[i, 1] = corners[:, 1].min()
        out[i, 2] = corners[:, 0].max()
        out[i, 3] = corners[:, 1].max()

    if single:
        out = out.squeeze(0)
    return out


def _homography_jacobian(
    M: np.ndarray, x: float, y: float, xp: float, yp: float,
) -> np.ndarray:
    """2x2 Jacobian of homography M at source point (x,y) with output (xp,yp).

    J = (1/w) * [[h00 - xp*h20, h01 - xp*h21],
                  [h10 - yp*h20, h11 - yp*h21]]
    """
    w = M[2, 0] * x + M[2, 1] * y + M[2, 2]
    return (1.0 / w) * np.array([
        [M[0, 0] - xp * M[2, 0], M[0, 1] - xp * M[2, 1]],
        [M[1, 0] - yp * M[2, 0], M[1, 1] - yp * M[2, 1]],
    ])


def apply_geo_contour_5dim(tensor: torch.Tensor, params: dict) -> torch.Tensor:
    """Transform 5-dim contour rows using unified geo transforms.

    Positions via ``apply_geo_points``.  Directions via per-point Jacobian
    when perspective, constant ``linear_part`` when affine.
    Supports both 1-D (5,) and 2-D (N, 5) inputs.
    """
    single = tensor.dim() == 1
    if single:
        tensor = tensor.unsqueeze(0)

    out = tensor.clone()
    is_persp = params.get("is_perspective", False)
    M = params["forward_matrix"]

    if not is_persp:
        L_t = torch.tensor(params["linear_part"], dtype=tensor.dtype)

    for i in range(tensor.shape[0]):
        class_id = int(tensor[i, 4].item())
        if class_id == CLASS_NONE:
            continue

        # Original source point (before transform)
        x_src, y_src = tensor[i, 0].item(), tensor[i, 1].item()

        # Transform position
        pt = tensor[i, :2].unsqueeze(0)  # (1, 2)
        pt = apply_geo_points(pt, params)
        out[i, 0] = pt[0, 0]
        out[i, 1] = pt[0, 1]

        # Transform direction
        dv = tensor[i, 2:4]
        if is_persp:
            xp, yp = pt[0, 0].item(), pt[0, 1].item()
            J = _homography_jacobian(M, x_src, y_src, xp, yp)
            J_t = torch.tensor(J, dtype=tensor.dtype)
            dv_rot = J_t @ dv
        else:
            dv_rot = L_t @ dv

        norm = dv_rot.norm()
        if norm > 1e-8:
            dv_rot = dv_rot / norm
        out[i, 2] = dv_rot[0]
        out[i, 3] = dv_rot[1]

    if single:
        out = out.squeeze(0)
    return out


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

class AffineStage:
    """Geometric affine stage — composable with other geometric stages."""

    def __init__(
        self,
        p: float = 0.5,
        rotation: float = 5.0,
        scale_range: tuple[float, float] = (0.9, 1.1),
        translate: float = 20.0,
        shear: float = 3.0,
    ):
        self.p = p
        self.rotation = rotation
        self.scale_range = scale_range
        self.translate = translate
        self.shear = shear

    def sample(self) -> dict | None:
        """Sample affine params if triggered, else None."""
        if random.random() > self.p:
            return None
        return sample_affine_params(
            rotation=self.rotation,
            scale_range=self.scale_range,
            translate=self.translate,
            shear=self.shear,
        )


class PerspectiveStage:
    """Geometric perspective stage — composable with AffineStage."""

    def __init__(self, p: float = 0.4, magnitude: float = 0.06):
        self.p = p
        self.magnitude = magnitude

    def sample(self) -> dict | None:
        """Sample perspective params if triggered, else None."""
        if random.random() > self.p:
            return None
        return sample_perspective_params(magnitude=self.magnitude)


class ColorJitter:
    """Wraps torchvision ColorJitter with per-call probability."""

    def __init__(self, p: float = 0.8, brightness: float = 0.3,
                 contrast: float = 0.3, saturation: float = 0.3, hue: float = 0.1):
        self.p = p
        self._jitter = transforms.ColorJitter(
            brightness=brightness, contrast=contrast,
            saturation=saturation, hue=hue,
        )

    def __call__(self, img_pil: Image.Image, ctx: dict | None = None) -> Image.Image:
        if random.random() > self.p:
            return img_pil
        return self._jitter(img_pil)


class GaussianBlur:
    """Wraps torchvision GaussianBlur with per-call probability."""

    def __init__(self, p: float = 0.5, kernel_size: int = 3,
                 sigma: tuple[float, float] = (0.1, 1.5)):
        self.p = p
        self._blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def __call__(self, img_pil: Image.Image, ctx: dict | None = None) -> Image.Image:
        if random.random() > self.p:
            return img_pil
        return self._blur(img_pil)


class RandomGrayscale:
    """Wraps torchvision RandomGrayscale with per-call probability."""

    def __init__(self, p: float = 0.1):
        self.p = p
        self._gray = transforms.RandomGrayscale(p=1.0)

    def __call__(self, img_pil: Image.Image, ctx: dict | None = None) -> Image.Image:
        if random.random() > self.p:
            return img_pil
        return self._gray(img_pil)


class GaussianNoise:
    """Add Gaussian noise. Converts to numpy internally."""

    def __init__(self, p: float = 0.5, std_range: tuple[float, float] = (0.01, 0.05)):
        self.p = p
        self.std_range = std_range

    def __call__(self, img_pil: Image.Image, ctx: dict | None = None) -> Image.Image:
        if random.random() > self.p:
            return img_pil
        img = np.array(img_pil)
        std = random.uniform(*self.std_range) * 255.0
        noise = np.random.normal(0, std, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(img)


class SaltAndPepperNoise:
    """Random pixel flips to black or white."""

    def __init__(self, p: float = 0.3, prob_range: tuple[float, float] = (0.001, 0.01)):
        self.p = p
        self.prob_range = prob_range

    def __call__(self, img_pil: Image.Image, ctx: dict | None = None) -> Image.Image:
        if random.random() > self.p:
            return img_pil
        img = np.array(img_pil)
        prob = random.uniform(*self.prob_range)
        mask = np.random.random(img.shape[:2])
        img = img.copy()
        img[mask < prob / 2] = 0
        img[mask > 1 - prob / 2] = 255
        return Image.fromarray(img)


class JPEGCompression:
    """Simulate JPEG compression artifacts."""

    def __init__(self, p: float = 0.5, quality_range: tuple[int, int] = (30, 95)):
        self.p = p
        self.quality_range = quality_range

    def __call__(self, img_pil: Image.Image, ctx: dict | None = None) -> Image.Image:
        if random.random() > self.p:
            return img_pil
        quality = random.randint(*self.quality_range)
        buf = io.BytesIO()
        img_pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class MorphologicalDegradation:
    """Ink bleed (dilation) or thinning (erosion) via PIL min/max filter."""

    def __init__(self, p: float = 0.3, kernel_size: int = 3):
        self.p = p
        self.kernel_size = kernel_size

    def __call__(self, img_pil: Image.Image, ctx: dict | None = None) -> Image.Image:
        if random.random() > self.p:
            return img_pil
        if random.random() < 0.5:
            return img_pil.filter(ImageFilter.MinFilter(self.kernel_size))
        else:
            return img_pil.filter(ImageFilter.MaxFilter(self.kernel_size))


class RandomErasing:
    """Erase rectangular patches filled with estimated background colour."""

    def __init__(
        self,
        p: float = 0.3,
        count_range: tuple[int, int] = (1, 3),
        size_range: tuple[int, int] = (10, 60),
    ):
        self.p = p
        self.count_range = count_range
        self.size_range = size_range

    def __call__(self, img_pil: Image.Image, ctx: dict | None = None) -> Image.Image:
        if random.random() > self.p:
            return img_pil
        img = np.array(img_pil)
        h, w = img.shape[:2]
        bg_color = (ctx or {}).get("bg_color")
        if bg_color is None:
            bg_color = tuple(int(v) for v in np.median(img[:8, :, :].reshape(-1, 3), axis=0))
        out = img.copy()
        n = random.randint(*self.count_range)
        for _ in range(n):
            ew = random.randint(*self.size_range)
            eh = random.randint(*self.size_range)
            ex = random.randint(0, max(0, w - ew))
            ey = random.randint(0, max(0, h - eh))
            out[ey:ey + eh, ex:ex + ew] = bg_color
        return Image.fromarray(out)


class ShadowGradient:
    """Simulate directional lighting falloff via a linear gradient multiplier."""

    def __init__(self, p: float = 0.4,
                 intensity_range: tuple[float, float] = (0.15, 0.4)):
        self.p = p
        self.intensity_range = intensity_range

    def __call__(self, img_pil: Image.Image, ctx: dict | None = None) -> Image.Image:
        if random.random() > self.p:
            return img_pil
        intensity = random.uniform(*self.intensity_range)
        angle = random.uniform(0, 2 * math.pi)
        w, h = img_pil.size

        # 1-D vectors → outer-product gradient (avoids full meshgrid allocation)
        xs = np.linspace(0, 1, w, dtype=np.float32) * math.cos(angle)
        ys = np.linspace(0, 1, h, dtype=np.float32) * math.sin(angle)
        grad = ys[:, np.newaxis] + xs[np.newaxis, :]  # (h, w)
        gmin, gmax = grad.min(), grad.max()
        # Map to [(1 - intensity), 1.0] in-place
        grad = (1.0 - intensity) + intensity * (grad - gmin) / (gmax - gmin + 1e-8)

        img = np.array(img_pil, dtype=np.float32)
        img *= grad[:, :, np.newaxis]
        np.clip(img, 0, 255, out=img)
        return Image.fromarray(img.astype(np.uint8))


class PaperTexture:
    """Simulate paper grain via low-frequency additive noise."""

    def __init__(self, p: float = 0.4,
                 intensity_range: tuple[float, float] = (0.02, 0.08)):
        self.p = p
        self.intensity_range = intensity_range

    def __call__(self, img_pil: Image.Image, ctx: dict | None = None) -> Image.Image:
        if random.random() > self.p:
            return img_pil
        intensity = random.uniform(*self.intensity_range)
        w, h = img_pil.size

        # Small noise patch, upscale for smooth low-frequency variation
        small_w, small_h = 64, 64
        noise_small = np.random.randn(small_h, small_w).astype(np.float32)
        noise_pil = Image.fromarray(noise_small, mode="F")
        noise_pil = noise_pil.resize((w, h), Image.BILINEAR)
        noise = np.array(noise_pil) * intensity * 255.0

        img = np.array(img_pil, dtype=np.float32)
        img = img + noise[:, :, np.newaxis]
        img = np.clip(img, 0, 255).astype(np.uint8)
        return Image.fromarray(img)


class ResolutionJitter:
    """Simulate varied capture quality by downscale + upscale."""

    def __init__(self, p: float = 0.3,
                 scale_range: tuple[float, float] = (0.3, 0.8)):
        self.p = p
        self.scale_range = scale_range

    def __call__(self, img_pil: Image.Image, ctx: dict | None = None) -> Image.Image:
        if random.random() > self.p:
            return img_pil
        scale = random.uniform(*self.scale_range)
        w, h = img_pil.size
        small_w = max(1, int(w * scale))
        small_h = max(1, int(h * scale))
        img_small = img_pil.resize((small_w, small_h), Image.BILINEAR)
        return img_small.resize((w, h), Image.BILINEAR)


class MotionBlur:
    """Simulate camera shake via directional 1D blur kernel."""

    def __init__(self, p: float = 0.3,
                 kernel_size_range: tuple[int, int] = (3, 7),
                 angle_range: tuple[float, float] = (0, 360)):
        self.p = p
        self.kernel_size_range = kernel_size_range
        self.angle_range = angle_range

    def __call__(self, img_pil: Image.Image, ctx: dict | None = None) -> Image.Image:
        if random.random() > self.p:
            return img_pil

        # Pick odd kernel size
        ksize = random.randrange(
            self.kernel_size_range[0], self.kernel_size_range[1] + 1, 2,
        )
        if ksize % 2 == 0:
            ksize += 1

        angle = random.uniform(*self.angle_range)

        # Build motion blur kernel — line of 1s at given angle
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        center = ksize // 2
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        for i in range(ksize):
            offset = i - center
            x = int(round(center + offset * cos_a))
            y = int(round(center + offset * sin_a))
            if 0 <= x < ksize and 0 <= y < ksize:
                kernel[y, x] = 1.0

        if kernel.sum() == 0:
            kernel[center, center] = 1.0
        kernel /= kernel.sum()

        # Apply via FFT-based convolution (fully vectorized, no Python loops)
        img = np.array(img_pil, dtype=np.float32)
        h, w = img.shape[:2]
        # Pad kernel to image size and FFT
        k_pad = np.zeros((h, w), dtype=np.float32)
        ph, pw = ksize // 2, ksize // 2
        k_pad[:ksize, :ksize] = kernel
        # Roll so kernel center is at (0,0)
        k_pad = np.roll(np.roll(k_pad, -ph, axis=0), -pw, axis=1)
        k_fft = np.fft.rfft2(k_pad)
        # Convolve all 3 channels at once
        img_fft = np.fft.rfft2(img, axes=(0, 1))
        img_fft *= k_fft[:, :, np.newaxis]
        img = np.fft.irfft2(img_fft, s=(h, w), axes=(0, 1))
        np.clip(img, 0, 255, out=img)
        return Image.fromarray(img.astype(np.uint8))


# ---------------------------------------------------------------------------
# Composable pipeline
# ---------------------------------------------------------------------------

_GEO_STAGE_TYPES = (AffineStage, PerspectiveStage)


class AugmentationPipeline:
    """Composable augmentation pipeline.

    Geometric stages (``AffineStage``, ``PerspectiveStage``) are sampled and
    composed into a single 3x3 matrix before resampling the image once.
    Photometric stages run sequentially on the PIL image.

    Usage::

        pipeline = AugmentationPipeline([
            AffineStage(p=0.5, rotation=5),
            PerspectiveStage(p=0.4, magnitude=0.06),
            ColorJitter(p=0.8),
            GaussianNoise(p=0.5),
        ])
        img_t_out, geo_params = pipeline.apply(img_t)
    """

    def __init__(self, stages: list):
        self.geo_stages = [s for s in stages if isinstance(s, _GEO_STAGE_TYPES)]
        self.photo_stages = [s for s in stages if not isinstance(s, _GEO_STAGE_TYPES)]

    def apply(self, img_t: torch.Tensor) -> tuple[torch.Tensor, dict | None]:
        """Apply pipeline to a CHW [0,1] tensor.

        Returns ``(img_t, geo_params)`` where *geo_params* is ``None`` if no
        geometric transform was triggered.
        """
        bg_color = estimate_bg_color(img_t)

        # Sample and compose all geometric stages into a single transform
        geo_params = None
        if self.geo_stages:
            sampled = [s.sample() for s in self.geo_stages]
            sampled = [p for p in sampled if p is not None]
            if sampled:
                geo_params = compose_transforms(sampled)
                img_t = apply_geo_image(img_t, geo_params, bg_color)

        # Apply photometric stages sequentially
        if self.photo_stages:
            img_pil = transforms.ToPILImage()(img_t)
            ctx = {"bg_color": bg_color}
            for stage in self.photo_stages:
                img_pil = stage(img_pil, ctx=ctx)
            img_t = transforms.ToTensor()(img_pil)

        return img_t, geo_params


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def default_pipeline(geometric: bool = True) -> AugmentationPipeline:
    """Production synth-only training — balanced domain-gap augmentation."""
    stages: list = []
    if geometric:
        stages.append(AffineStage(p=0.5, rotation=5.0, scale_range=(0.9, 1.1),
                                  translate=20.0, shear=3.0))
        stages.append(PerspectiveStage(p=0.4, magnitude=0.06))
    stages.extend([
        ColorJitter(p=0.8, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        GaussianBlur(p=0.5, kernel_size=3, sigma=(0.1, 1.5)),
        MotionBlur(p=0.3, kernel_size_range=(3, 7)),
        RandomGrayscale(p=0.1),
        GaussianNoise(p=0.5, std_range=(0.01, 0.05)),
        SaltAndPepperNoise(p=0.3, prob_range=(0.001, 0.01)),
        JPEGCompression(p=0.5, quality_range=(30, 95)),
        MorphologicalDegradation(p=0.3, kernel_size=3),
        ShadowGradient(p=0.4, intensity_range=(0.15, 0.4)),
        PaperTexture(p=0.4, intensity_range=(0.02, 0.08)),
        ResolutionJitter(p=0.3, scale_range=(0.3, 0.8)),
        RandomErasing(p=0.3, count_range=(1, 3), size_range=(10, 60)),
    ])
    return AugmentationPipeline(stages)


def light_pipeline(geometric: bool = True) -> AugmentationPipeline:
    """Gentler augmentation — lower probabilities, tighter ranges (fine-tuning)."""
    stages: list = []
    if geometric:
        stages.append(AffineStage(p=0.3, rotation=3.0, scale_range=(0.95, 1.05),
                                  translate=10.0, shear=1.5))
        stages.append(PerspectiveStage(p=0.2, magnitude=0.03))
    stages.extend([
        ColorJitter(p=0.5, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        GaussianBlur(p=0.3, kernel_size=3, sigma=(0.1, 1.0)),
        MotionBlur(p=0.15, kernel_size_range=(3, 5)),
        RandomGrayscale(p=0.05),
        GaussianNoise(p=0.3, std_range=(0.005, 0.03)),
        SaltAndPepperNoise(p=0.15, prob_range=(0.0005, 0.005)),
        JPEGCompression(p=0.3, quality_range=(50, 95)),
        MorphologicalDegradation(p=0.15, kernel_size=3),
        ShadowGradient(p=0.2, intensity_range=(0.1, 0.25)),
        PaperTexture(p=0.2, intensity_range=(0.01, 0.04)),
        ResolutionJitter(p=0.15, scale_range=(0.5, 0.9)),
        RandomErasing(p=0.15, count_range=(1, 2), size_range=(10, 40)),
    ])
    return AugmentationPipeline(stages)


def heavy_pipeline(geometric: bool = True) -> AugmentationPipeline:
    """Aggressive augmentation — higher probabilities, wider ranges + extra jitter."""
    stages: list = []
    if geometric:
        stages.append(AffineStage(p=0.7, rotation=8.0, scale_range=(0.8, 1.2),
                                  translate=30.0, shear=5.0))
        stages.append(AffineStage(p=0.3, rotation=0, scale_range=(0.85, 1.15),
                                  translate=0, shear=0))
        stages.append(PerspectiveStage(p=0.5, magnitude=0.08))
    stages.extend([
        ColorJitter(p=0.9, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
        GaussianBlur(p=0.6, kernel_size=3, sigma=(0.1, 2.0)),
        MotionBlur(p=0.4, kernel_size_range=(3, 9)),
        RandomGrayscale(p=0.15),
        GaussianNoise(p=0.6, std_range=(0.01, 0.08)),
        SaltAndPepperNoise(p=0.4, prob_range=(0.001, 0.02)),
        JPEGCompression(p=0.6, quality_range=(20, 90)),
        MorphologicalDegradation(p=0.4, kernel_size=3),
        ShadowGradient(p=0.5, intensity_range=(0.2, 0.5)),
        PaperTexture(p=0.5, intensity_range=(0.03, 0.1)),
        ResolutionJitter(p=0.4, scale_range=(0.2, 0.7)),
        RandomErasing(p=0.4, count_range=(1, 4), size_range=(10, 80)),
    ])
    return AugmentationPipeline(stages)


# ---------------------------------------------------------------------------
# Unified wrapper
# ---------------------------------------------------------------------------

class AugmentedDataset(Dataset):
    """Wraps a dataset and applies augmentation via an ``AugmentationPipeline``.

    Parameters
    ----------
    subset : Dataset
        The underlying dataset to wrap.
    mode : str
        One of ``"bbox"`` (train_02), ``"sequence"`` (train_03),
        ``"contour"`` (train_04).
    pipeline : AugmentationPipeline | None
        Pipeline to apply.  Defaults to ``default_pipeline()``.
    """

    def __init__(
        self,
        subset: Dataset,
        mode: str,
        pipeline: AugmentationPipeline | None = None,
    ):
        assert mode in ("bbox", "sequence", "contour"), f"Unknown mode: {mode!r}"
        self.subset = subset
        self.mode = mode
        self.pipeline = pipeline or default_pipeline()

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        if self.mode == "bbox":
            return self._augment_bbox(self.subset[idx])
        elif self.mode == "sequence":
            return self._augment_sequence(self.subset[idx])
        else:
            return self._augment_contour(self.subset[idx])

    # -- bbox mode (train_02): (img_t, prev_t, target_t) --
    def _augment_bbox(self, sample):
        img_t, prev_t, target_t = sample
        img_t, geo_params = self.pipeline.apply(img_t)
        if geo_params is not None:
            prev_t = apply_geo_bbox_6dim(prev_t, geo_params)
            target_t = apply_geo_bbox_6dim(target_t, geo_params)
        return img_t, prev_t, target_t

    # -- sequence mode (train_03): (img_t, prev_seq, target_seq, level_id, S) --
    def _augment_sequence(self, sample):
        img_t, prev_seq, target_seq, level_id, S = sample
        img_t, geo_params = self.pipeline.apply(img_t)
        if geo_params is not None:
            prev_seq = apply_geo_bbox_6dim(prev_seq, geo_params)
            target_seq = apply_geo_bbox_6dim(target_seq, geo_params)
        return img_t, prev_seq, target_seq, level_id, S

    # -- contour mode (train_04): (img_t, prev_seq, target_seq, start_pt, level_t, S) --
    def _augment_contour(self, sample):
        img_t, prev_seq, target_seq, start_pt, level_t, S = sample
        img_t, geo_params = self.pipeline.apply(img_t)
        if geo_params is not None:
            prev_seq = apply_geo_contour_5dim(prev_seq, geo_params)
            target_seq = apply_geo_contour_5dim(target_seq, geo_params)
            sp = start_pt.unsqueeze(0)  # (1, 2)
            sp = apply_geo_points(sp, geo_params)
            start_pt = sp.squeeze(0)
        return img_t, prev_seq, target_seq, start_pt, level_t, S
