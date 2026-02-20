"""
Deploy model weights to glyph-daemon.

Encrypts a .pth checkpoint with Fernet (AES-256) and copies it to the
glyph-daemon weights directory. The vault key is read from the
GLYPH_VAULT_KEY environment variable.
"""

import os
import sys
from pathlib import Path

GLYPH_WEIGHTS_DIR = Path.home() / "Code" / "glyph-deamon" / "weights"


def deploy(pth_path: str | Path) -> bool:
    """Encrypt and deploy a .pth file to glyph-daemon/weights/.

    Returns True on success, False on failure (prints warning, never raises).
    """
    pth_path = Path(pth_path)
    key = os.environ.get("GLYPH_VAULT_KEY")
    if not key:
        print("  [deploy] GLYPH_VAULT_KEY not set — skipping deploy", flush=True)
        return False

    try:
        from cryptography.fernet import Fernet

        fernet = Fernet(key.encode())
        plaintext = pth_path.read_bytes()
        ciphertext = fernet.encrypt(plaintext)

        GLYPH_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        enc_name = pth_path.stem + ".enc"
        dst = GLYPH_WEIGHTS_DIR / enc_name
        dst.write_bytes(ciphertext)

        size_mb = len(ciphertext) / (1024 * 1024)
        print(f"  [deploy] {pth_path.name} -> {dst} ({size_mb:.1f} MB)", flush=True)
        return True

    except ImportError:
        print(
            "  [deploy] cryptography package not installed — skipping deploy",
            flush=True,
        )
        return False
    except Exception as e:
        print(f"  [deploy] failed: {e}", flush=True)
        return False
