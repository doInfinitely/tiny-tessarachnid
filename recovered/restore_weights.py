"""Restore model_02_char.pth from the encrypted copy on the Modal volume.

The session deploy script (deploy_weights.py) encrypts each best checkpoint
with Fernet (AES-256) and uploads to Modal volume "glyph-weights" as
model_02_char.enc. This script reverses that: pulls the .enc down, decrypts
it in memory, writes the plaintext back to model_02_char.pth.

Usage:
  export GLYPH_VAULT_KEY=<the same Fernet key used by deploy_weights.py>
  python recovered/restore_weights.py

Optional flags:
  --volume   Modal volume name (default: glyph-weights)
  --enc      remote .enc filename (default: model_02_char.enc)
  --out      local output path (default: model_02_char.pth)

Also pulls model_02_char_emb_index.enc if --emb is set.

Prereqs on the machine you run this on:
  pip install cryptography modal
  modal token set  # if not already authenticated
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _modal_get(volume: str, remote: str, local: Path) -> None:
    cmd = ["modal", "volume", "get", "--force", volume, remote, str(local)]
    print(f"$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        sys.exit(f"modal volume get failed:\n{r.stderr.strip() or r.stdout.strip()}")


def _decrypt(enc_path: Path, out_path: Path, key: str) -> None:
    from cryptography.fernet import Fernet, InvalidToken
    fernet = Fernet(key.encode())
    ciphertext = enc_path.read_bytes()
    try:
        plaintext = fernet.decrypt(ciphertext)
    except InvalidToken:
        sys.exit(f"Decrypt failed for {enc_path} — wrong GLYPH_VAULT_KEY?")
    out_path.write_bytes(plaintext)
    print(f"  wrote {out_path} ({len(plaintext) / 1e6:.1f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--volume", default="glyph-weights")
    ap.add_argument("--enc", default="model_02_char.enc")
    ap.add_argument("--out", default="model_02_char.pth")
    ap.add_argument("--emb", action="store_true",
                    help="Also restore model_02_char_emb_index.enc")
    args = ap.parse_args()

    key = os.environ.get("GLYPH_VAULT_KEY")
    if not key:
        sys.exit("GLYPH_VAULT_KEY env var is not set")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _modal_get(args.volume, args.enc, tmp / args.enc)
        _decrypt(tmp / args.enc, Path(args.out), key)

        if args.emb:
            emb_remote = "model_02_char_emb_index.enc"
            emb_out = Path(args.out).with_name("model_02_char_emb_index.pth")
            _modal_get(args.volume, emb_remote, tmp / emb_remote)
            _decrypt(tmp / emb_remote, emb_out, key)


if __name__ == "__main__":
    main()
