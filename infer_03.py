"""
Inference script for RetinaOCRGPT (V3).

Generates a synthetic page (or loads a real image), then runs hierarchical
autoregressive detection. Adapts to the checkpoint format automatically
(older 5-dim/3-level/98-class checkpoints vs newer 6-dim/4-level/459-class).

Hierarchy levels:
  - 3-level (old): paragraphs → lines → characters
  - 4-level (new): paragraphs → lines → words → characters
"""

import argparse
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image, ImageDraw, ImageFont

from generate_training_data import (
    CHAR_CLASS_OFFSET,
    CLASS_NONE,
    CLASS_PARAGRAPH,
    CLASS_LINE,
    CLASS_WORD,
    CLASS_TO_CHAR,
    PREV_BBOX_NONE,
    RETINA_SIZE,
    SyntheticPage,
    discover_fonts,
    scale_and_pad,
)
from train_03 import TransformerBlock


# ---------------------------------------------------------------------------
# Compat model that auto-detects checkpoint format
# ---------------------------------------------------------------------------
class RetinaOCRGPTCompat(nn.Module):
    """RetinaOCRGPT that matches the checkpoint's architecture."""

    def __init__(self, d_model=128, n_heads=4, n_layers=4, d_ff=512,
                 max_seq_len=128, dropout=0.1,
                 det_dim=5, num_levels=3, num_classes=98,
                 has_handwritten_head=False):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.det_dim = det_dim
        self.has_handwritten_head = has_handwritten_head

        # Image encoder (ResNet-18)
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
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.img_proj = nn.Linear(512, d_model)

        # Detection token embedding
        self.det_proj = nn.Linear(det_dim, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.level_emb = nn.Embedding(num_levels, d_model)

        # Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        # Output heads
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 4), nn.Sigmoid())
        self.class_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, num_classes))

        if has_handwritten_head:
            self.handwritten_head = nn.Sequential(
                nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))

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

    def forward(self, img, prev_seq, level_ids, padding_mask=None):
        B, S, _ = prev_seq.shape

        img_feat = self._encode_image(img)

        prev_norm = prev_seq.clone()
        prev_norm[:, :, :4] = prev_norm[:, :, :4] / RETINA_SIZE
        if self.det_dim >= 5:
            from generate_training_data import NUM_CLASSES
            prev_norm[:, :, 4] = prev_norm[:, :, 4] / NUM_CLASSES

        det_emb = self.det_proj(prev_norm)
        det_emb = det_emb + img_feat.unsqueeze(1)

        positions = torch.arange(S, device=prev_seq.device)
        det_emb = det_emb + self.pos_emb(positions).unsqueeze(0)
        det_emb = det_emb + self.level_emb(level_ids).unsqueeze(1)

        causal_mask = torch.triu(
            torch.ones(S, S, device=prev_seq.device, dtype=torch.bool),
            diagonal=1)

        x = det_emb
        for block in self.blocks:
            x = block(x, attn_mask=causal_mask, key_padding_mask=padding_mask)
        x = self.ln_f(x)

        bbox_pred = self.bbox_head(x) * RETINA_SIZE
        class_pred = self.class_head(x)

        if self.has_handwritten_head:
            hw_pred = self.handwritten_head(x)
            return bbox_pred, class_pred, hw_pred

        return bbox_pred, class_pred


def detect_checkpoint_format(ckpt):
    """Detect model dimensions from checkpoint keys."""
    det_dim = ckpt["det_proj.weight"].shape[1]
    num_levels = ckpt["level_emb.weight"].shape[0]
    num_classes = ckpt["class_head.2.bias"].shape[0]
    has_hw = "handwritten_head.0.weight" in ckpt
    return det_dim, num_levels, num_classes, has_hw


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def retina_to_source(retina_bbox, scale, ox, oy):
    x1, y1, x2, y2 = retina_bbox
    return (
        int((x1 - ox) / scale),
        int((y1 - oy) / scale),
        int((x2 - ox) / scale),
        int((y2 - oy) / scale),
    )


def class_to_char(class_id):
    return CLASS_TO_CHAR.get(class_id, "?")


@torch.no_grad()
def generate_sequence(model, source_img, bg_color, level_id, device,
                      max_len=128):
    """Autoregressive sequence generation for the compat model."""
    retina, scale, ox, oy = scale_and_pad(source_img, bg_color)
    img_t = (
        torch.from_numpy(np.array(retina))
        .permute(2, 0, 1).float().unsqueeze(0) / 255.0
    ).to(device)

    level_t = torch.tensor([level_id], dtype=torch.long, device=device)

    # Build initial sequence token (match det_dim)
    det_dim = model.det_dim
    init_token = list(PREV_BBOX_NONE[:det_dim])
    seq = [init_token]
    detections = []

    for _ in range(max_len):
        prev_t = torch.tensor([seq], dtype=torch.float32, device=device)
        outputs = model(img_t, prev_t, level_t)

        bbox_pred = outputs[0]
        class_pred = outputs[1]
        hw_pred = outputs[2] if len(outputs) == 3 else None

        class_id = class_pred[0, -1].argmax().item()
        if class_id == CLASS_NONE:
            break

        rb = bbox_pred[0, -1].tolist()
        source_bbox = retina_to_source(rb, scale, ox, oy)

        # Clamp
        w, h = source_img.size
        source_bbox = (
            max(0, min(source_bbox[0], w - 1)),
            max(0, min(source_bbox[1], h - 1)),
            max(0, min(source_bbox[2], w)),
            max(0, min(source_bbox[3], h)),
        )
        if source_bbox[2] <= source_bbox[0] or source_bbox[3] <= source_bbox[1]:
            break

        hw_prob = 0.0
        if hw_pred is not None:
            hw_prob = torch.sigmoid(hw_pred[0, -1, 0]).item()

        detections.append((source_bbox, class_id, hw_prob))

        token = [rb[0], rb[1], rb[2], rb[3], class_id]
        if det_dim >= 6:
            token.append(1.0 if hw_prob > 0.5 else 0.0)
        seq.append(token[:det_dim])

    return detections


# ---------------------------------------------------------------------------
# Hierarchical inference
# ---------------------------------------------------------------------------
def run_hierarchical(model, page_img, bg_color, device, num_levels):
    """Run hierarchical inference, adapting to 3 or 4 levels."""
    model.eval()

    # Level 0: paragraphs
    paragraphs = generate_sequence(model, page_img, bg_color, 0, device)
    print(f"Detected {len(paragraphs)} paragraphs")

    all_text = []
    all_lines = []
    all_words = []
    all_chars = []

    for pi, (para_bbox, _, hw_prob) in enumerate(paragraphs):
        hw_label = "handwritten" if hw_prob > 0.5 else "printed"
        print(f"  Paragraph {pi+1}: {hw_label} (p={hw_prob:.2f})")
        para_crop = page_img.crop(para_bbox)

        # Level 1: lines
        lines = generate_sequence(model, para_crop, bg_color, 1, device)
        print(f"    {len(lines)} lines")

        for li, (line_bbox, _, _) in enumerate(lines):
            page_line = (
                line_bbox[0] + para_bbox[0], line_bbox[1] + para_bbox[1],
                line_bbox[2] + para_bbox[0], line_bbox[3] + para_bbox[1],
            )
            all_lines.append(page_line)
            line_crop = para_crop.crop(line_bbox)

            if num_levels == 4:
                # Level 2: words, then Level 3: chars in each word
                words = generate_sequence(
                    model, line_crop, bg_color, 2, device)
                line_text = []
                for wi, (word_bbox, _, _) in enumerate(words):
                    page_word = (
                        word_bbox[0] + line_bbox[0] + para_bbox[0],
                        word_bbox[1] + line_bbox[1] + para_bbox[1],
                        word_bbox[2] + line_bbox[0] + para_bbox[0],
                        word_bbox[3] + line_bbox[1] + para_bbox[1],
                    )
                    all_words.append(page_word)
                    word_crop = line_crop.crop(word_bbox)

                    chars = generate_sequence(
                        model, word_crop, bg_color, 3, device)
                    word_text = []
                    for char_bbox, class_id, _ in chars:
                        page_char = (
                            char_bbox[0] + word_bbox[0] + line_bbox[0] + para_bbox[0],
                            char_bbox[1] + word_bbox[1] + line_bbox[1] + para_bbox[1],
                            char_bbox[2] + word_bbox[0] + line_bbox[0] + para_bbox[0],
                            char_bbox[3] + word_bbox[1] + line_bbox[1] + para_bbox[1],
                        )
                        all_chars.append(page_char)
                        if class_id >= CHAR_CLASS_OFFSET:
                            word_text.append(class_to_char(class_id))
                        else:
                            word_text.append("?")
                    line_text.append("".join(word_text))
                all_text.append((pi, li, " ".join(line_text)))
            else:
                # 3-level: Level 2 is chars directly
                chars = generate_sequence(
                    model, line_crop, bg_color, 2, device)
                line_text = []
                for char_bbox, class_id, _ in chars:
                    page_char = (
                        char_bbox[0] + line_bbox[0] + para_bbox[0],
                        char_bbox[1] + line_bbox[1] + para_bbox[1],
                        char_bbox[2] + line_bbox[0] + para_bbox[0],
                        char_bbox[3] + line_bbox[1] + para_bbox[1],
                    )
                    all_chars.append(page_char)
                    if class_id >= CHAR_CLASS_OFFSET:
                        line_text.append(class_to_char(class_id))
                    else:
                        line_text.append("?")
                all_text.append((pi, li, "".join(line_text)))

    return paragraphs, all_text, all_lines, all_words, all_chars


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def visualize(page_img, paragraphs, all_text, all_lines, all_words, all_chars,
              output_path, page=None, scale=2):
    """Draw ground truth (if available) and predictions side by side."""
    w, h = page_img.size
    sw, sh = w * scale, h * scale
    page_scaled = page_img.resize((sw, sh), Image.NEAREST)
    s = scale

    try:
        font = ImageFont.truetype("fonts/Arial.ttf", 10 * scale)
        label_font = ImageFont.truetype("fonts/Arial.ttf", 14 * scale)
    except (OSError, IOError):
        font = ImageFont.load_default()
        label_font = font

    if page is not None:
        # Side-by-side: ground truth | predictions
        combined = Image.new("RGB", (sw * 2, sh), (255, 255, 255))
        combined.paste(page_scaled, (0, 0))
        combined.paste(page_scaled, (sw, 0))
        draw = ImageDraw.Draw(combined)

        # Left: ground truth
        for para in page.paragraphs:
            b = para["bbox"]
            draw.rectangle((b[0]*s, b[1]*s, b[2]*s, b[3]*s),
                           outline=(255, 0, 0), width=2*s)
            for line in para["lines"]:
                b = line["bbox"]
                draw.rectangle((b[0]*s, b[1]*s, b[2]*s, b[3]*s),
                               outline=(0, 0, 255), width=s)
                for word in line.get("words", []):
                    b = word["bbox"]
                    draw.rectangle((b[0]*s, b[1]*s, b[2]*s, b[3]*s),
                                   outline=(255, 165, 0), width=1)
                    for ch in word["characters"]:
                        b = ch["bbox"]
                        draw.rectangle((b[0]*s, b[1]*s, b[2]*s, b[3]*s),
                                       outline=(0, 180, 0), width=1)

        draw.text((10*s, 10*s), "GROUND TRUTH", fill=(0, 0, 0), font=label_font)
        ox = sw
    else:
        combined = Image.new("RGB", (sw, sh), (255, 255, 255))
        combined.paste(page_scaled, (0, 0))
        draw = ImageDraw.Draw(combined)
        ox = 0

    # Predictions
    for para_bbox, _, hw_prob in paragraphs:
        draw.rectangle(
            (para_bbox[0]*s + ox, para_bbox[1]*s,
             para_bbox[2]*s + ox, para_bbox[3]*s),
            outline=(255, 0, 0), width=2*s)
        label = "HW" if hw_prob > 0.5 else "PR"
        draw.text((para_bbox[0]*s + ox, max(0, para_bbox[1]*s - 14*s)),
                  label, fill=(255, 0, 0), font=font)

    for lb in all_lines:
        draw.rectangle(
            (lb[0]*s + ox, lb[1]*s, lb[2]*s + ox, lb[3]*s),
            outline=(0, 0, 255), width=s)

    for wb in all_words:
        draw.rectangle(
            (wb[0]*s + ox, wb[1]*s, wb[2]*s + ox, wb[3]*s),
            outline=(255, 165, 0), width=2)

    for cb in all_chars:
        draw.rectangle(
            (cb[0]*s + ox, cb[1]*s, cb[2]*s + ox, cb[3]*s),
            outline=(0, 180, 0), width=1)

    # Text labels on prediction lines
    font_size = 10 * scale
    for idx, (pi, li, text) in enumerate(all_text):
        if idx < len(all_lines):
            lb = all_lines[idx]
            tx = lb[0]*s + ox
            ty = max(0, lb[1]*s - font_size - 4)
            draw.text((tx+1, ty+1), text, fill=(0, 0, 0), font=font)
            draw.text((tx, ty), text, fill=(0, 180, 0), font=font)

    if page is not None:
        draw.text((ox + 10*s, 10*s), "PREDICTIONS",
                  fill=(0, 0, 0), font=label_font)

    combined.save(output_path)
    print(f"Visualization saved to {output_path} "
          f"({combined.size[0]}x{combined.size[1]})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="RetinaOCRGPT (V3) inference")
    parser.add_argument("--model-path", default="model_03.pth")
    parser.add_argument("--output", default="infer_03_output.png")
    parser.add_argument("--page-width", type=int, default=2048)
    parser.add_argument("--page-height", type=int, default=2800)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--image", type=str, default=None,
                        help="Path to real image (omit for synthetic)")

    # Model architecture (must match checkpoint)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=128)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint and detect format
    try:
        ckpt = torch.load(args.model_path, map_location=device,
                          weights_only=True)
        det_dim, num_levels, num_classes, has_hw = detect_checkpoint_format(ckpt)
        print(f"Checkpoint: det_dim={det_dim}, levels={num_levels}, "
              f"classes={num_classes}, handwritten_head={has_hw}")
    except FileNotFoundError:
        print(f"Warning: {args.model_path} not found, using defaults")
        ckpt = None
        det_dim, num_levels, num_classes, has_hw = 5, 3, 98, False

    model = RetinaOCRGPTCompat(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        det_dim=det_dim,
        num_levels=num_levels,
        num_classes=num_classes,
        has_handwritten_head=has_hw,
    ).to(device)

    if ckpt is not None:
        model.load_state_dict(ckpt, strict=False)
        print(f"Loaded model from {args.model_path}")

    model.eval()

    if args.image:
        page_img = Image.open(args.image).convert("RGB")
        from annotate_real import estimate_background_color
        bg_color = estimate_background_color(page_img)
        print(f"Image: {args.image} ({page_img.size[0]}x{page_img.size[1]}), "
              f"bg={bg_color}")

        results = run_hierarchical(model, page_img, bg_color, device,
                                   num_levels)
        paragraphs, all_text, all_lines, all_words, all_chars = results

        # Print detected text
        print("\n--- Detected Text ---")
        if not all_text:
            print("  (no text detected)")
        current_para = -1
        for pi, li, text in all_text:
            if pi != current_para:
                current_para = pi
                print(f"Paragraph {pi+1}:")
            print(f"  Line {li+1}: {text}")

        visualize(page_img, paragraphs, all_text, all_lines, all_words,
                  all_chars, args.output)
    else:
        # Generate synthetic page
        fonts = discover_fonts()
        if not fonts:
            print("No fonts found!")
            sys.exit(1)
        page = SyntheticPage(fonts, args.page_width, args.page_height)
        page_img = page.image
        bg_color = page.bg_color
        print(f"Generated page: {len(page.paragraphs)} paragraphs, "
              f"bg={bg_color}")

        results = run_hierarchical(model, page_img, bg_color, device,
                                   num_levels)
        paragraphs, all_text, all_lines, all_words, all_chars = results

        # Print detected text
        print("\n--- Detected Text ---")
        if not all_text:
            print("  (no text detected)")
        current_para = -1
        for pi, li, text in all_text:
            if pi != current_para:
                current_para = pi
                print(f"Paragraph {pi+1}:")
            print(f"  Line {li+1}: {text}")

        # Print ground truth
        print("\n--- Ground Truth ---")
        for pi, para in enumerate(page.paragraphs):
            hw_label = "handwritten" if para.get("is_handwritten") else "printed"
            print(f"Paragraph {pi+1} [{hw_label}]:")
            for li, line in enumerate(para["lines"]):
                gt_text = " ".join(
                    "".join(ch["char"] for ch in word["characters"])
                    for word in line["words"]
                )
                print(f"  Line {li+1}: {gt_text}")

        visualize(page_img, paragraphs, all_text, all_lines, all_words,
                  all_chars, args.output, page=page)


if __name__ == "__main__":
    main()
