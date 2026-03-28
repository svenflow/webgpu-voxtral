#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["requests"]
# ///
"""
Upload extracted F16 weights to HuggingFace via git LFS.
Uses the HF commit API which supports cookie-based auth from Chrome.

This script uploads files using the HuggingFace commit API with multipart upload,
which handles LFS automatically.
"""

import hashlib
import json
import os
import sys
import base64
from pathlib import Path

REPO = "Svenflow/voxtral-tts-webgpu-f16"
WEIGHTS_DIR = Path(__file__).parent.parent / "models" / "weights"
API_BASE = f"https://huggingface.co/api/models/{REPO}"


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)  # 1MB chunks
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def upload_with_hf_cli(token: str):
    """Upload using the hf CLI."""
    import subprocess

    files = ["manifest.json", "backbone.bin", "fm.bin", "codec.bin"]
    for f in files:
        path = WEIGHTS_DIR / f
        if not path.exists():
            print(f"SKIP: {f} not found")
            continue
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"Uploading {f} ({size_mb:.1f} MB)...")

        result = subprocess.run(
            ["hf", "upload", REPO, str(path), f,
             "--token", token,
             "--commit-message", f"Add {f}"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr}")
        else:
            print(f"  OK: {result.stdout.strip()}")


def main():
    # Check for token
    token = os.environ.get("HF_TOKEN", "")

    if not token:
        # Try to read from hf CLI config
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            token = token_path.read_text().strip()

    if not token:
        print("No HF_TOKEN found. Please set HF_TOKEN or run 'hf auth login' first.")
        print(f"\nAlternative: manually upload files from {WEIGHTS_DIR} to")
        print(f"https://huggingface.co/{REPO}")
        sys.exit(1)

    print(f"Uploading to {REPO}")
    print(f"Weights dir: {WEIGHTS_DIR}")
    print()

    upload_with_hf_cli(token)
    print("\nDone!")


if __name__ == "__main__":
    main()
