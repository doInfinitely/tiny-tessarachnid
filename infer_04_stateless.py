"""
Stateless contour tracing -- the mapper function.

Receives complete context (model, image tensor, level), returns detections,
retains nothing.  This is the pure-function boundary between the model and
the scaffolding.

    detections = trace_contours(model, img_t, level_id, device)
"""

import math
from dataclasses import dataclass

import torch

from generate_training_data import CLASS_NONE, PREV_CONTOUR_NONE
from infer_04_cached import decode_step, encode_image
from train_04 import ContourOCRNet


@dataclass
class TraceConfig:
    """Autoregressive tracing parameters."""
    max_points: int = 256
    closure_threshold: float = 15.0


@dataclass
class ContourDetection:
    """A single detected contour in retina coordinates."""
    retina_points: list   # list of (x, y) floats
    class_id: int


def load_model(weights_path, device):
    """Load a fresh ContourOCRNet from disk.  No side effects."""
    model = ContourOCRNet().to(device)
    ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    return model


@torch.no_grad()
def trace_contours(model, img_t, level_id, device, config=None):
    """Encode image ONCE, run autoregressive loop, return retina-space contours.

    Args:
        model: ContourOCRNet in eval mode.
        img_t: (1, 3, H, W) retina image tensor on device.
        level_id: Hierarchy level (0-4).
        device: torch device.
        config: TraceConfig (uses defaults if None).

    Returns:
        list[ContourDetection] in retina coordinates.
    """
    if config is None:
        config = TraceConfig()

    img_feat = encode_image(model, img_t)
    level_t = torch.tensor([level_id], dtype=torch.long, device=device)

    seq = [list(PREV_CONTOUR_NONE)]
    start_point = torch.zeros(1, 2, device=device)

    detections = []
    current_contour = []
    current_class = None

    for step in range(config.max_points):
        prev_t = torch.tensor([seq], dtype=torch.float32, device=device)
        point_pred, orient_pred, class_pred = decode_step(
            model, img_feat, prev_t, start_point, level_t,
        )

        class_id = class_pred[0, -1].argmax().item()
        if class_id == CLASS_NONE:
            if current_contour and current_class is not None:
                detections.append(ContourDetection(
                    retina_points=list(current_contour),
                    class_id=current_class,
                ))
            break

        px, py = point_pred[0, -1].tolist()
        dx, dy = orient_pred[0, -1].tolist()

        if current_class is None:
            current_class = class_id
            current_contour = [(px, py)]
            start_point = torch.tensor([[px, py]], device=device)
        else:
            current_contour.append((px, py))
            if len(current_contour) >= 3:
                sx, sy = current_contour[0]
                dist = math.hypot(px - sx, py - sy)
                if dist < config.closure_threshold:
                    detections.append(ContourDetection(
                        retina_points=list(current_contour),
                        class_id=current_class,
                    ))
                    current_contour = []
                    current_class = None

        seq.append([px, py, dx, dy, class_id])

    return detections
