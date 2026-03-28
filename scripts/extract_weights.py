#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "safetensors",
#   "numpy",
#   "torch",
#   "packaging",
# ]
# ///
"""
Phase 2: Extract weights from Voxtral TTS consolidated.safetensors

Converts BF16 → F16 and writes:
1. A single .bin file with all weights concatenated
2. A JSON manifest mapping tensor name → {offset, shape, dtype, bytes}

The TypeScript weight loader uses the manifest to create GPU buffers
via HTTP range requests (streaming) or from the full .bin file.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


def bf16_to_f16_bytes(tensor: torch.Tensor) -> bytes:
    """Convert a BF16 tensor to F16 bytes."""
    # BF16 → F32 → F16 (safest path, avoids precision issues)
    f16 = tensor.float().half()
    return f16.numpy().tobytes()


def f32_to_f16_bytes(tensor: torch.Tensor) -> bytes:
    """Convert an F32 tensor to F16 bytes."""
    f16 = tensor.half()
    return f16.numpy().tobytes()


def tensor_to_f16_bytes(tensor: torch.Tensor) -> bytes:
    """Convert any tensor to F16 bytes."""
    if tensor.dtype == torch.bfloat16:
        return bf16_to_f16_bytes(tensor)
    elif tensor.dtype == torch.float32:
        return f32_to_f16_bytes(tensor)
    elif tensor.dtype == torch.float16:
        return tensor.numpy().tobytes()
    else:
        raise ValueError(f"Unsupported dtype: {tensor.dtype}")


def classify_component(name: str) -> str:
    """Classify tensor into component for the manifest."""
    if name.startswith("acoustic_transformer."):
        return "fm"
    elif name.startswith("audio_tokenizer."):
        return "codec"
    elif name.startswith("layers.") or name.startswith("norm.") or name.startswith("mm_audio_embeddings."):
        return "backbone"
    else:
        return "other"


def extract_layer_info(name: str) -> dict:
    """Extract layer index and weight type from tensor name."""
    parts = name.split(".")
    info = {"component": classify_component(name)}

    # Extract layer index if present
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            info["layer"] = int(parts[i + 1])
            break
        if p.startswith("decoder_blocks") and i + 1 < len(parts) and parts[i + 1].isdigit():
            info["block"] = int(parts[i + 1])
            break

    return info


def main():
    parser = argparse.ArgumentParser(description="Extract Voxtral TTS weights to F16 binary")
    parser.add_argument("--model-dir", type=str, default="models/voxtral-tts",
                        help="Path to model directory")
    parser.add_argument("--output-dir", type=str, default="models/weights",
                        help="Output directory for .bin and manifest.json")
    parser.add_argument("--split-components", action="store_true",
                        help="Split into separate files per component")
    args = parser.parse_args()

    script_dir = Path(__file__).parent.parent
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = script_dir / model_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    st_file = model_dir / "consolidated.safetensors"
    if not st_file.exists():
        print(f"ERROR: {st_file} not found")
        sys.exit(1)

    print(f"Loading {st_file}...")
    tensors = load_file(str(st_file))
    print(f"Loaded {len(tensors)} tensors")

    # Sort tensors by component then name for better locality
    sorted_names = sorted(tensors.keys(), key=lambda n: (classify_component(n), n))

    manifest = {
        "format": "f16",
        "source": "mistralai/Voxtral-4B-TTS-2603",
        "total_tensors": len(sorted_names),
        "tensors": {},
    }

    if args.split_components:
        # Write separate .bin files per component
        components = {}
        for name in sorted_names:
            comp = classify_component(name)
            if comp not in components:
                components[comp] = {"names": [], "offset": 0}
            components[comp]["names"].append(name)

        for comp, info in components.items():
            bin_path = output_dir / f"{comp}.bin"
            offset = 0
            print(f"\nWriting {bin_path}...")

            with open(bin_path, "wb") as f:
                for name in info["names"]:
                    tensor = tensors[name]
                    data = tensor_to_f16_bytes(tensor)

                    # Align to 256 bytes for GPU buffer alignment
                    padding = (256 - (len(data) % 256)) % 256
                    padded_data = data + b'\x00' * padding

                    f.write(padded_data)

                    manifest["tensors"][name] = {
                        "file": f"{comp}.bin",
                        "offset": offset,
                        "size": len(data),
                        "padded_size": len(padded_data),
                        "shape": list(tensor.shape),
                        "dtype": "f16",
                        "original_dtype": str(tensor.dtype).replace("torch.", ""),
                        **extract_layer_info(name),
                    }

                    offset += len(padded_data)
                    print(f"  {name}: {list(tensor.shape)} ({len(data)} bytes)")

            total_mb = offset / 1e6
            print(f"  Total: {total_mb:.1f} MB")
            manifest[f"{comp}_total_bytes"] = offset

    else:
        # Write single consolidated .bin file
        bin_path = output_dir / "voxtral-tts-f16.bin"
        offset = 0
        print(f"\nWriting {bin_path}...")

        with open(bin_path, "wb") as f:
            for i, name in enumerate(sorted_names):
                tensor = tensors[name]
                data = tensor_to_f16_bytes(tensor)

                # Align to 256 bytes for GPU buffer alignment
                padding = (256 - (len(data) % 256)) % 256
                padded_data = data + b'\x00' * padding

                f.write(padded_data)

                manifest["tensors"][name] = {
                    "file": "voxtral-tts-f16.bin",
                    "offset": offset,
                    "size": len(data),
                    "padded_size": len(padded_data),
                    "shape": list(tensor.shape),
                    "dtype": "f16",
                    "original_dtype": str(tensor.dtype).replace("torch.", ""),
                    **extract_layer_info(name),
                }

                offset += len(padded_data)

                if (i + 1) % 50 == 0 or i == len(sorted_names) - 1:
                    print(f"  [{i+1}/{len(sorted_names)}] {offset / 1e9:.2f} GB written")

        manifest["total_bytes"] = offset
        print(f"\nTotal: {offset / 1e9:.2f} GB")

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}")

    # Summary
    comp_sizes = {}
    for name, info in manifest["tensors"].items():
        comp = info.get("component", "other")
        comp_sizes[comp] = comp_sizes.get(comp, 0) + info["size"]

    print("\n=== Component Sizes (F16) ===")
    total = 0
    for comp, size in sorted(comp_sizes.items()):
        print(f"  {comp}: {size / 1e9:.2f} GB")
        total += size
    print(f"  TOTAL: {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
