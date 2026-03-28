#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "safetensors",
#   "numpy",
#   "json5",
# ]
# ///
"""
Phase 1: Inspect Voxtral TTS model weights

Catalogs all weight tensors from consolidated.safetensors,
groups them by component (backbone, FM transformer, codec),
and documents shapes, dtypes, and parameter counts.
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

from safetensors import safe_open


def format_size(num_bytes: int) -> str:
    if num_bytes >= 1e9:
        return f"{num_bytes / 1e9:.2f} GB"
    elif num_bytes >= 1e6:
        return f"{num_bytes / 1e6:.2f} MB"
    elif num_bytes >= 1e3:
        return f"{num_bytes / 1e3:.2f} KB"
    return f"{num_bytes} B"


def format_params(count: int) -> str:
    if count >= 1e9:
        return f"{count / 1e9:.2f}B"
    elif count >= 1e6:
        return f"{count / 1e6:.1f}M"
    elif count >= 1e3:
        return f"{count / 1e3:.1f}K"
    return str(count)


def dtype_size(dtype_str: str) -> int:
    sizes = {
        "F32": 4, "F16": 2, "BF16": 2, "I32": 4, "I16": 2, "I8": 1,
        "U8": 1, "BOOL": 1, "F64": 8, "I64": 8,
    }
    return sizes.get(dtype_str, 4)


def classify_tensor(name: str) -> str:
    """Classify a tensor name into its component."""
    if name.startswith("backbone."):
        return "backbone"
    elif name.startswith("fm_transformer.") or name.startswith("flow_matching."):
        return "fm_transformer"
    elif name.startswith("codec.") or name.startswith("audio_codec."):
        return "codec"
    elif name.startswith("text_encoder.") or name.startswith("encoder."):
        return "text_encoder"
    else:
        # Try to infer from structure
        return "unknown"


def inspect_safetensors(model_path: Path) -> dict:
    """Inspect a safetensors file and return structured metadata."""
    st_files = list(model_path.glob("*.safetensors"))
    if not st_files:
        print(f"ERROR: No .safetensors files found in {model_path}")
        sys.exit(1)

    all_tensors = []

    for st_file in sorted(st_files):
        print(f"\nInspecting: {st_file.name}")
        with safe_open(str(st_file), framework="numpy") as f:
            metadata = f.metadata()
            if metadata:
                print(f"  Metadata: {json.dumps(dict(metadata), indent=2)}")

            keys = f.keys()
            print(f"  Tensors: {len(keys)}")

            for key in sorted(keys):
                # Use get_slice to avoid loading full tensor (handles BF16)
                sl = f.get_slice(key)
                shape = list(sl.get_shape())
                dtype_str = str(sl.get_dtype())
                num_params = 1
                for s in shape:
                    num_params *= s
                elem_size = dtype_size(dtype_str)
                num_bytes = num_params * elem_size

                all_tensors.append({
                    "name": key,
                    "shape": shape,
                    "dtype": dtype_str,
                    "params": num_params,
                    "bytes": num_bytes,
                    "file": st_file.name,
                    "component": classify_tensor(key),
                })

    return {"tensors": all_tensors, "files": [f.name for f in st_files]}


def print_report(data: dict):
    """Print a structured report of the model."""
    tensors = data["tensors"]

    # Group by component
    by_component = defaultdict(list)
    for t in tensors:
        by_component[t["component"]].append(t)

    print("\n" + "=" * 80)
    print("VOXTRAL TTS MODEL INSPECTION REPORT")
    print("=" * 80)

    total_params = sum(t["params"] for t in tensors)
    total_bytes = sum(t["bytes"] for t in tensors)
    print(f"\nTotal tensors: {len(tensors)}")
    print(f"Total parameters: {format_params(total_params)} ({total_params:,})")
    print(f"Total size: {format_size(total_bytes)}")

    # Per-component summary
    print(f"\n{'Component':<20} {'Tensors':>8} {'Params':>12} {'Size':>10}")
    print("-" * 55)
    for comp in sorted(by_component.keys()):
        comp_tensors = by_component[comp]
        comp_params = sum(t["params"] for t in comp_tensors)
        comp_bytes = sum(t["bytes"] for t in comp_tensors)
        print(f"{comp:<20} {len(comp_tensors):>8} {format_params(comp_params):>12} {format_size(comp_bytes):>10}")

    # Detailed tensor listing per component
    for comp in sorted(by_component.keys()):
        comp_tensors = by_component[comp]
        print(f"\n{'=' * 80}")
        print(f"COMPONENT: {comp}")
        print(f"{'=' * 80}")

        # Group by layer/block
        by_layer = defaultdict(list)
        for t in comp_tensors:
            # Extract layer number if present
            parts = t["name"].split(".")
            # Find the layer grouping
            layer_key = ".".join(parts[:3]) if len(parts) > 3 else t["name"]
            by_layer[layer_key].append(t)

        for layer_key in sorted(by_layer.keys()):
            layer_tensors = by_layer[layer_key]
            if len(layer_tensors) > 1:
                print(f"\n  {layer_key}:")
                for t in sorted(layer_tensors, key=lambda x: x["name"]):
                    short_name = t["name"][len(layer_key)+1:] if t["name"].startswith(layer_key + ".") else t["name"]
                    print(f"    {short_name:<40} {str(t['shape']):<25} {t['dtype']:<8} {format_params(t['params']):>8}")
            else:
                t = layer_tensors[0]
                print(f"  {t['name']:<55} {str(t['shape']):<25} {t['dtype']:<8} {format_params(t['params']):>8}")

    # Unique shapes analysis
    print(f"\n{'=' * 80}")
    print("UNIQUE SHAPES (for shader planning)")
    print(f"{'=' * 80}")
    shape_counts = defaultdict(int)
    for t in tensors:
        shape_counts[tuple(t["shape"])] += 1
    for shape, count in sorted(shape_counts.items(), key=lambda x: -x[1]):
        print(f"  {str(list(shape)):<30} × {count}")

    # Dtype analysis
    print(f"\n{'=' * 80}")
    print("DTYPE DISTRIBUTION")
    print(f"{'=' * 80}")
    dtype_counts = defaultdict(lambda: {"count": 0, "params": 0})
    for t in tensors:
        dtype_counts[t["dtype"]]["count"] += 1
        dtype_counts[t["dtype"]]["params"] += t["params"]
    for dtype, info in sorted(dtype_counts.items()):
        print(f"  {dtype:<10} {info['count']:>5} tensors  {format_params(info['params']):>10} params")


def main():
    parser = argparse.ArgumentParser(description="Inspect Voxtral TTS model weights")
    parser.add_argument("--model-dir", type=str, default="models/voxtral-tts",
                        help="Path to model directory containing .safetensors files")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of report")
    args = parser.parse_args()

    model_path = Path(args.model_dir)
    if not model_path.is_absolute():
        # Try relative to script directory first, then cwd
        script_dir = Path(__file__).parent.parent
        if (script_dir / model_path).exists():
            model_path = script_dir / model_path
        elif not model_path.exists():
            print(f"ERROR: Model directory not found: {model_path}")
            print(f"  Tried: {script_dir / args.model_dir}")
            print(f"  Tried: {Path.cwd() / args.model_dir}")
            sys.exit(1)

    data = inspect_safetensors(model_path)

    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print_report(data)


if __name__ == "__main__":
    main()
