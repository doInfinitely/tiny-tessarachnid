# train_02_char.py — playbook

Two-stage hierarchical Unicode character classifier. Trains a ResNet-18
backbone with one block-classifier head and per-block character heads.
Outputs `model_02_char.pth` consumed by `detect_characters.py` and
glyph-faerie's `glyph_faerie/detection/model.py`.

## Environment setup

```bash
cd tiny-tessarachnid
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch torchvision pillow numpy fonttools
# Optional: cryptography (only to decrypt deployed weights from Modal)
```

The session that produced this file ran on PyTorch 2.10.0+cu128. Any recent
torch with CUDA support should work. `torch.compile` is used when
`--compile` is passed, so PyTorch ≥ 2.0 is required.

## Font dependency

`get_unicode_chars()` in `generate_training_data.py` enumerates Unicode
codepoints by reading the cmap of every font in `tiny-tessarachnid/fonts/`
(via `fontTools`). A representative session run reported:

```
NFKC normalization: merged 2895 variant chars → 61876 unique chars remaining
Unicode chars: 61876, active blocks: 218
```

If your `fonts/` directory is empty or sparse, those numbers will be much
smaller and Stage A will train on a tiny set of blocks.  Drop Mac system
fonts + Google's Noto family in there for full coverage.

## Training from scratch

Full three-stage run (Stage A: block head, Stage B: char heads, Stage C:
joint fine-tune):

```bash
python train_02_char.py --batch-size 512 --ecological \
    --epochs-block 15 \
    --epochs-char 10 \
    --epochs-joint 50
```

Approximate timings observed on the original 2× RTX 3090 setup:

| Stage | Work | Time |
|---|---|---|
| A | data gen (43,400 samples) + 15 epochs of block head only | ~12 min |
| B | char heads, 50 spc × ~60k chars | 1–2 h |
| C | joint fine-tune, 50 epochs × 10k spc | 6–12 h |

Smoke-test the whole pipeline end-to-end in under an hour with smaller spc:

```bash
python train_02_char.py --batch-size 512 --ecological \
    --epochs-block 3 --epochs-char 3 --epochs-joint 3 \
    --joint-spc 1000
```

`--ecological` weights blocks by speaker population so common scripts
dominate (see `_BLOCK_SPEAKERS_M` in `train()`).  Without it every block is
weighted equally — fine for testing, bad for inference quality.

## Resuming + joint-only fine-tune

If you already have `model_02_char.pth` (from a previous run or by
decrypting `model_02_char.enc` from the Modal `glyph-weights` volume), skip
Stages A and B:

```bash
python train_02_char.py --batch-size 512 --ecological --resume-stage-c \
    --joint-spc 10000 --epochs-joint 50
```

This was the last command the recovery session was running before the dump.

## The "sliding window" experiment

The recovered session set up but never ran this Stage C variant.  Instead
of pre-rendered character crops, it trains on real sliding-window crops
extracted from synthetic pages using the same `_extract_windows` /
`_reject_blank_windows` code paths as inference. "Zero domain gap".

```bash
python train_02_char.py --batch-size 512 --ecological --resume-stage-c \
    --sliding-window --sw-pages 100 --epochs-joint 50
```

Motivation (from the session): the model reached 94.7% Latin block accuracy
and 80.1% Latin char accuracy on its own isolated renders, but only
19–24% class accuracy on the sliding-window crops produced at inference.
The diagnosis was a domain mismatch — and "lets make them the same" was
the user's instruction that prompted this dataset.

## Decrypting `model_02_char.pth` from the Modal volume

If the encrypted checkpoint is still on the Modal `glyph-weights` volume,
`recovered/restore_weights.py` pulls and decrypts it:

```bash
pip install cryptography modal
modal token set                  # if not already authenticated
export GLYPH_VAULT_KEY=<the Fernet key used by deploy_weights.py>

python recovered/restore_weights.py            # writes model_02_char.pth
python recovered/restore_weights.py --emb      # also pulls embedding index
```

`GLYPH_VAULT_KEY` was used by the deploy side; glyph-faerie's `vault.py`
calls the same value `FAERIE_VAULT_KEY` on the decryption side.

## Reconstructed regions to suspect first if something breaks

The current `train_02_char.py`, `create_lists.py` block-table appendix,
`generate_training_data.get_unicode_chars`, and `detect_characters.py` were
reconstructed from a partial Claude session recording — see
`recovered/RECOVERY_NOTES.md` for the full audit trail. Reconstructed
regions are wrapped in `# [RECONSTRUCTED ...]` / `# [/RECONSTRUCTED]`
markers. If something blows up at run time, suspect these first:

1. **`_render_char` / `_render_char_in_context`** (`train_02_char.py` lines
   ~269–340) — pure invention based on intent. If data gen prints
   "Rendered 0/N" or images are all-blank, the rendering is the culprit.
   Quick check:
   ```python
   from train_02_char import _render_char
   from PIL import ImageFont
   f = ImageFont.truetype("fonts/<some-font>.ttf", 32)
   img = _render_char("A", f, 128)   # should be a (3,128,128) uint8 tensor
   ```

2. **`HierarchicalLoss.forward`** (~735–762) — if Stage A loss is NaN or
   doesn't decrease, the per-block char-loss averaging is the suspect.

3. **`get_unicode_chars`** in `generate_training_data.py` — should print
   `NFKC normalization: merged ~2895 variant chars → ~61876 unique chars`.
   Substantially smaller numbers mean the font scan isn't finding what the
   session saw.

4. **`_pool_init` / `_render_task` / `_parallel_render`** (~353–396) — the
   worker signature was patched to pass `block_idx` instead of full sibling
   lists (siblings are stashed once per worker via the pool initializer).
   If workers sit idle or render workers crash on startup, this is where to
   look.

5. **`train()` setup + Stage A→B transition** (~865–930, 1023–1097) — bulk
   of the training scaffold. Errors here usually surface as missing
   attributes or wrong argument names.

If Stage A epoch 1 prints something like

```
[BlockDet] Epoch 1/15  loss=5.30  block=0.02/0.03  char=0.04/0.05  lr=0.000989
```

(matching the session's actual output), the major plumbing is correct.
