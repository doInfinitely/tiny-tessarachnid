#!/usr/bin/env python3
"""Back-port detect_characters.py to tiny-tessarachnid.

Strategy:
- Use the 80-line partial recovery (recovered/detect_characters.py.partial)
  for the tiny-tessarachnid-specific top: docstring with Usage/CLI examples,
  imports including argparse/hashlib/ImageDraw/ImageFont, MEGA_BLOCK_THRESHOLD,
  Hangul helpers (decompose/compose_hangul), EmbeddingIndex class header.
- Reverse-port the rest from glyph-faerie/glyph_faerie/detection/detector.py,
  patching imports back to tiny-tessarachnid paths
  (glyph_faerie.detection.model → train_02_char,
   glyph_faerie.detection.blocks → create_lists).
- Use the partial's lines 230-239 (DetectorConfig with inline comments) instead
  of glyph-faerie's bare DetectorConfig, then continue with glyph-faerie's
  remaining fields (confidence_threshold, nms thresholds, batch size, pixel/edge
  thresholds).

Output: /home/remy/tiny-tessarachnid/detect_characters.py
"""

import re

PARTIAL = '/home/remy/tiny-tessarachnid/recovered/detect_characters.py.partial'
GLYPH = '/home/remy/glyph-faerie/glyph_faerie/detection/detector.py'

with open(PARTIAL) as f:
    partial = f.readlines()
with open(GLYPH) as f:
    glyph = f.readlines()


def partial_slice(start, end):
    """1-indexed inclusive slice of partial lines, with MISSING markers checked."""
    out = []
    for i in range(start - 1, end):
        ln = partial[i]
        if ln.startswith('# [MISSING LINE'):
            raise ValueError(f"partial line {i+1} is missing")
        out.append(ln)
    return ''.join(out)


def glyph_slice(start, end):
    """1-indexed inclusive slice of glyph-faerie lines."""
    return ''.join(glyph[start - 1:end])


def patch_imports(text):
    """Rewrite glyph-faerie-internal imports to tiny-tessarachnid imports."""
    text = text.replace(
        "from glyph_faerie.detection.model import HierarchicalCharNet, CHAR_INPUT_SIZE",
        "from train_02_char import HierarchicalCharNet, CHAR_INPUT_SIZE",
    )
    text = text.replace(
        "from glyph_faerie.detection.blocks import get_block_index",
        "from create_lists import get_block_index",
    )
    return text


parts = []

# Partial lines 1-39: docstring, imports, MEGA_BLOCK_THRESHOLD constant + comment header for Hangul
parts.append(partial_slice(1, 39))

# Partial lines 40-58: Hangul utils (decompose, compose) — tiny-tessarachnid-specific
parts.append(partial_slice(40, 58))

# Partial lines 59-80: EmbeddingIndex class header through `def save(self, path):`
parts.append(partial_slice(59, 80))

# Reverse-port glyph-faerie lines 44-89: rest of EmbeddingIndex class (save body, load, lookup_batch)
parts.append(glyph_slice(44, 89))

# DetectorConfig: use partial's comment-rich header (lines 230-239), then extend
# from glyph-faerie with the rest of the fields. The partial doesn't include
# the @dataclass decorator (it would have been on line 229, which is missing).
parts.append('\n')
parts.append('# ---------------------------------------------------------------------------\n')
parts.append('# Detector config\n')
parts.append('# ---------------------------------------------------------------------------\n\n')
parts.append('@dataclass\n')
parts.append(partial_slice(230, 239))
# Continue with the remaining DetectorConfig fields from glyph-faerie (lines 102-109)
parts.append('    confidence_threshold: float = 0.3\n')
parts.append('    nms_iou_threshold: float = 0.3\n')
parts.append('    max_batch_size: int = 256\n')
parts.append('    # Background rejection: minimum pixel std dev (0-1 scale) for a window\n')
parts.append('    # to be considered non-blank.  Blank patches (uniform color) have std~0.\n')
parts.append('    min_pixel_std: float = 0.10\n')
parts.append('    # Minimum edge density (fraction of edge pixels via Sobel) to keep a window\n')
parts.append('    min_edge_density: float = 0.06\n')

# Reverse-port glyph-faerie lines 112-700: CharacterDetector, load_detector,
# _extract_windows, _reject_blank_windows, detect_characters, detect_characters_raw,
# apply_block_priors, apply_nms. Patch imports as we go.
parts.append('\n\n')
remainder = glyph_slice(112, 700)
remainder = patch_imports(remainder)
parts.append(remainder)

# Add a logger module-level (glyph-faerie defines it at line 22)
# We need to insert this near the top, before any logger.* call. Easiest:
# inject it right after the imports block in the partial. The partial's last
# import is `from PIL import Image, ImageDraw, ImageFont` on line 30.
# Patch by inserting after MEGA_BLOCK_THRESHOLD definition.
text = ''.join(parts)

# Insert logger definition right after the MEGA_BLOCK_THRESHOLD line.
text = text.replace(
    'MEGA_BLOCK_THRESHOLD = 256\n',
    'MEGA_BLOCK_THRESHOLD = 256\n\n'
    'import logging\n'
    'logger = logging.getLogger(__name__)\n',
    1,
)

# Append a minimal CLI entrypoint mirroring the docstring's usage example.
text += '''


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_main():
    parser = argparse.ArgumentParser(
        description="Run HierarchicalCharNet detector on an image")
    parser.add_argument("image", help="Path to page image")
    parser.add_argument("--model", default="model_02_char.pth",
                        help="Path to model checkpoint (default: model_02_char.pth)")
    parser.add_argument("--embeddings", default=None,
                        help="Optional path to embedding index for mega-blocks")
    parser.add_argument("--visualize", action="store_true",
                        help="Save annotated image to <image>.detections.png")
    parser.add_argument("--output", default=None,
                        help="Save JSON detections to this path")
    parser.add_argument("--device", default=None,
                        help="cuda / cpu (auto-detect by default)")
    args = parser.parse_args()

    import json as _json
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    emb_idx = EmbeddingIndex.load(args.embeddings, device) if args.embeddings else None

    detector = load_detector(checkpoint, device, emb_idx)
    page = Image.open(args.image).convert("RGB")
    detections = detect_characters(page, detector)

    print(f"{len(detections)} detections")
    if args.output:
        with open(args.output, "w") as f:
            _json.dump(detections, f, indent=2, ensure_ascii=False)
        print(f"Wrote {args.output}")

    if args.visualize:
        out_path = os.path.splitext(args.image)[0] + ".detections.png"
        vis = page.copy()
        draw = ImageDraw.Draw(vis)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=1)
            draw.text((x1, max(0, y1 - 14)), det["char"], fill="red", font=font)
        vis.save(out_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    _cli_main()
'''

OUT = '/home/remy/tiny-tessarachnid/detect_characters.py'
with open(OUT, 'w') as f:
    f.write(text)
print(f"Wrote {OUT} ({len(text)} bytes, {text.count(chr(10))} lines)")

import py_compile
try:
    py_compile.compile(OUT, doraise=True)
    print("py_compile: OK")
except py_compile.PyCompileError as e:
    print(f"py_compile FAILED:\n{e}")
