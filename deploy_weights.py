"""
Deploy model weights to Modal volume for GlyphWorker.

Encrypts a .pth checkpoint with Fernet (AES-256) and uploads it to the
Modal "glyph-weights" volume. The GlyphWorker loads weights from this
volume at container startup.

The vault key is read from the GLYPH_VAULT_KEY environment variable.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

MODAL_VOLUME = "glyph-weights"
MODAL_BIN = Path.home() / "Code" / "glyph-deamon" / ".venv" / "bin" / "modal"


def deploy(pth_path: str | Path) -> bool:
    """Encrypt and deploy a .pth file to the Modal volume.

    Returns True on success.
    """
    pth_path = Path(pth_path)
    key = os.environ.get("GLYPH_VAULT_KEY")
    if not key:
        print("  [deploy] GLYPH_VAULT_KEY not set — skipping deploy", flush=True)
        return False

    if not MODAL_BIN.exists():
        print("  [deploy] modal CLI not found — skipping", flush=True)
        return False

    try:
        from cryptography.fernet import Fernet

        fernet = Fernet(key.encode())
        plaintext = pth_path.read_bytes()
        ciphertext = fernet.encrypt(plaintext)
        enc_name = pth_path.stem + ".enc"
        size_mb = len(plaintext) / (1024 * 1024)

        print(f"  [deploy] {pth_path.name} ({size_mb:.1f} MB)", flush=True)

        with tempfile.NamedTemporaryFile(suffix=".enc", delete=False) as tmp:
            tmp.write(ciphertext)
            tmp_path = tmp.name

        result = subprocess.run(
            [str(MODAL_BIN), "volume", "put", "--force", MODAL_VOLUME, tmp_path, enc_name],
            capture_output=True, text=True, timeout=120,
        )
        os.unlink(tmp_path)

        if result.returncode == 0:
            print(f"  [deploy] -> {MODAL_VOLUME}/{enc_name} ({size_mb:.1f} MB)", flush=True)
            return True
        else:
            print(f"  [deploy] failed: {result.stderr.strip()}", flush=True)
            return False

    except ImportError:
        print(
            "  [deploy] cryptography package not installed — skipping deploy",
            flush=True,
        )
        return False
    except Exception as e:
        print(f"  [deploy] failed: {e}", flush=True)
        return False
