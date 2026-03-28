#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "safetensors",
#   "numpy",
#   "packaging",
# ]
# ///
"""
Generate ground-truth reference activations for Voxtral TTS.

Runs a manual forward pass through the backbone and FM acoustic transformer
using raw PyTorch ops (no vllm/transformers), saving intermediate activations
as .npy files for validation of the WebGPU implementation.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "voxtral-tts"
WEIGHTS_PATH = MODEL_DIR / "consolidated.safetensors"
PARAMS_PATH = MODEL_DIR / "params.json"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / "activations"

# ---------------------------------------------------------------------------
# Hyperparameters (from params.json / ARCHITECTURE.md)
# ---------------------------------------------------------------------------
DIM = 3072
N_LAYERS = 26
HEAD_DIM = 128
N_HEADS = 32
N_KV_HEADS = 8
HIDDEN_DIM = 9216
ROPE_THETA = 1_000_000.0
NORM_EPS = 1e-5

FM_N_LAYERS = 3
FM_ROPE_THETA = 10_000.0
FM_N_STEPS = 8
FM_CFG_ALPHA = 1.2

ACOUSTIC_DIM = 36  # n_acoustic_codebook


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = NORM_EPS) -> torch.Tensor:
    """RMSNorm: x * rsqrt(mean(x^2) + eps) * weight"""
    dtype = x.dtype
    x = x.float()
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * rms).to(dtype) * weight


def precompute_freqs_cis(head_dim: int, seq_len: int, theta: float) -> torch.Tensor:
    """Precompute RoPE complex exponentials."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # [seq_len, head_dim/2]
    return torch.polar(torch.ones_like(freqs), freqs)  # [seq_len, head_dim/2]


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings (interleaved format).

    x: [batch, n_heads, seq_len, head_dim]
    freqs_cis: [seq_len, head_dim/2]
    """
    # Reshape x to pairs for complex multiplication
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # freqs_cis: [seq_len, head_dim/2] -> [1, 1, seq_len, head_dim/2]
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)
    x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return x_rotated.to(x.dtype)


def attention(
    x: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wo: torch.Tensor,
    freqs_cis: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """Grouped-query attention.

    x: [batch, seq_len, dim]
    """
    bsz, seqlen, _ = x.shape
    n_rep = N_HEADS // N_KV_HEADS

    q = x @ wq.T  # [bsz, seqlen, n_heads * head_dim]
    k = x @ wk.T  # [bsz, seqlen, n_kv_heads * head_dim]
    v = x @ wv.T  # [bsz, seqlen, n_kv_heads * head_dim]

    q = q.view(bsz, seqlen, N_HEADS, HEAD_DIM).transpose(1, 2)    # [bsz, n_heads, seqlen, head_dim]
    k = k.view(bsz, seqlen, N_KV_HEADS, HEAD_DIM).transpose(1, 2)  # [bsz, n_kv_heads, seqlen, head_dim]
    v = v.view(bsz, seqlen, N_KV_HEADS, HEAD_DIM).transpose(1, 2)  # [bsz, n_kv_heads, seqlen, head_dim]

    # Apply RoPE
    q = apply_rope(q, freqs_cis)
    k = apply_rope(k, freqs_cis)

    # Repeat KV heads for GQA
    k = k.repeat_interleave(n_rep, dim=1)  # [bsz, n_heads, seqlen, head_dim]
    v = v.repeat_interleave(n_rep, dim=1)

    # Scaled dot product attention
    scale = HEAD_DIM ** -0.5
    scores = (q @ k.transpose(-2, -1)) * scale  # [bsz, n_heads, seqlen, seqlen]

    if causal and seqlen > 1:
        mask = torch.triu(torch.full((seqlen, seqlen), float("-inf"), device=x.device), diagonal=1)
        scores = scores + mask

    attn = torch.softmax(scores.float(), dim=-1).to(x.dtype)
    out = attn @ v  # [bsz, n_heads, seqlen, head_dim]
    out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)  # [bsz, seqlen, n_heads * head_dim]

    return out @ wo.T


def swiglu(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
    """SwiGLU FFN: w2(silu(w1(x)) * w3(x))"""
    return (torch.nn.functional.silu(x @ w1.T) * (x @ w3.T)) @ w2.T


def sinusoidal_time_embedding(t: float, dim: int) -> torch.Tensor:
    """Sinusoidal positional embedding for timestep t in [0, 1]."""
    half_dim = dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = torch.tensor(t, dtype=torch.float32) * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)])  # [dim]
    return emb


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------
saved = {}


def save_activation(name: str, tensor: torch.Tensor) -> None:
    """Save activation as .npy and print summary."""
    arr = tensor.detach().float().cpu().numpy()
    path = OUTPUT_DIR / f"{name}.npy"
    np.save(path, arr)
    saved[name] = arr
    vals = arr.flatten()[:8]
    print(f"  {name}: shape={list(arr.shape)}, first8={vals}")


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------
def run_forward(weights: dict[str, torch.Tensor], seed: int = 42) -> dict[str, np.ndarray]:
    """Run a single deterministic forward pass and save activations."""
    torch.manual_seed(seed)
    saved.clear()
    device = torch.device("cpu")
    dtype = torch.float32

    # Convert all weights to float32
    w = {k: v.to(dtype=dtype, device=device) for k, v in weights.items()}

    # -----------------------------------------------------------------------
    # Backbone
    # -----------------------------------------------------------------------
    token_ids = torch.tensor([[1]], dtype=torch.long)  # BOS token
    embed = w["mm_audio_embeddings.tok_embeddings.weight"][token_ids]  # [1, 1, 3072]
    save_activation("backbone_embed", embed)

    # Precompute RoPE for backbone (seq_len=1)
    freqs_cis_backbone = precompute_freqs_cis(HEAD_DIM, 1, ROPE_THETA)

    h = embed.clone()
    for i in range(N_LAYERS):
        prefix = f"layers.{i}"

        # Attention norm
        h_normed = rms_norm(h, w[f"{prefix}.attention_norm.weight"])
        save_activation(f"backbone_layer{i}_attn_norm", h_normed)

        # Attention
        attn_out = attention(
            h_normed,
            w[f"{prefix}.attention.wq.weight"],
            w[f"{prefix}.attention.wk.weight"],
            w[f"{prefix}.attention.wv.weight"],
            w[f"{prefix}.attention.wo.weight"],
            freqs_cis_backbone,
            causal=True,
        )
        save_activation(f"backbone_layer{i}_attn_out", attn_out)
        h = h + attn_out

        # FFN norm
        h_normed = rms_norm(h, w[f"{prefix}.ffn_norm.weight"])
        save_activation(f"backbone_layer{i}_ffn_norm", h_normed)

        # FFN
        ffn_out = swiglu(h_normed, w[f"{prefix}.feed_forward.w1.weight"],
                         w[f"{prefix}.feed_forward.w2.weight"],
                         w[f"{prefix}.feed_forward.w3.weight"])
        h = h + ffn_out
        save_activation(f"backbone_layer{i}_ffn_out", h)

    # Final norm
    h = rms_norm(h, w["norm.weight"])
    save_activation("backbone_final_norm", h)

    # The hidden state passed to FM transformer
    hidden = h.squeeze(0)  # [1, 3072]
    save_activation("backbone_hidden", hidden)

    # -----------------------------------------------------------------------
    # FM Acoustic Transformer — semantic logits
    # -----------------------------------------------------------------------
    semantic_logits = hidden @ w["acoustic_transformer.semantic_codebook_output.weight"].T  # [1, 8320]
    save_activation("fm_semantic_logits", semantic_logits)

    # -----------------------------------------------------------------------
    # FM Acoustic Transformer — Euler ODE
    # -----------------------------------------------------------------------
    # Precompute RoPE for FM transformer (seq_len=3, bidirectional)
    freqs_cis_fm = precompute_freqs_cis(HEAD_DIM, 3, FM_ROPE_THETA)

    # LLM projection (conditioned)
    llm_proj_cond = hidden @ w["acoustic_transformer.llm_projection.weight"].T  # [1, 3072]
    # Unconditional: zero hidden
    llm_proj_uncond = torch.zeros_like(llm_proj_cond)

    # Initial noise
    torch.manual_seed(seed)  # Reset seed for reproducible noise
    x_t = torch.randn(1, ACOUSTIC_DIM, dtype=dtype)  # [1, 36]

    dt = 1.0 / FM_N_STEPS
    for step in range(FM_N_STEPS):
        t = step * dt

        # Time embedding
        time_emb = sinusoidal_time_embedding(t, DIM).unsqueeze(0).to(device)  # [1, 3072]
        time_proj = time_emb @ w["acoustic_transformer.time_projection.weight"].T  # [1, 3072]

        # Input projection of x_t
        x_t_proj = x_t @ w["acoustic_transformer.input_projection.weight"].T  # [1, 3072]

        # Build 3-token sequences for conditioned and unconditioned
        # Sequence: [x_t_proj, time_proj, llm_proj]
        seq_cond = torch.stack([x_t_proj.squeeze(0), time_proj.squeeze(0), llm_proj_cond.squeeze(0)]).unsqueeze(0)  # [1, 3, 3072]
        seq_uncond = torch.stack([x_t_proj.squeeze(0), time_proj.squeeze(0), llm_proj_uncond.squeeze(0)]).unsqueeze(0)

        # Run both through FM transformer layers
        for cond_idx, seq in enumerate([seq_cond, seq_uncond]):
            h_fm = seq.clone()
            for li in range(FM_N_LAYERS):
                pfx = f"acoustic_transformer.layers.{li}"

                h_fm_normed = rms_norm(h_fm, w[f"{pfx}.attention_norm.weight"])
                attn_out = attention(
                    h_fm_normed,
                    w[f"{pfx}.attention.wq.weight"],
                    w[f"{pfx}.attention.wk.weight"],
                    w[f"{pfx}.attention.wv.weight"],
                    w[f"{pfx}.attention.wo.weight"],
                    freqs_cis_fm,
                    causal=False,  # Bidirectional
                )
                h_fm = h_fm + attn_out

                h_fm_normed = rms_norm(h_fm, w[f"{pfx}.ffn_norm.weight"])
                ffn_out = swiglu(h_fm_normed, w[f"{pfx}.feed_forward.w1.weight"],
                                 w[f"{pfx}.feed_forward.w2.weight"],
                                 w[f"{pfx}.feed_forward.w3.weight"])
                h_fm = h_fm + ffn_out

            # Final norm
            h_fm = rms_norm(h_fm, w["acoustic_transformer.norm.weight"])

            # Extract position 0 -> acoustic velocity
            pos0 = h_fm[:, 0, :]  # [1, 3072]
            velocity = pos0 @ w["acoustic_transformer.acoustic_codebook_output.weight"].T  # [1, 36]

            if cond_idx == 0:
                v_cond = velocity
            else:
                v_uncond = velocity

        # CFG: v = alpha * v_cond + (1 - alpha) * v_uncond
        v = FM_CFG_ALPHA * v_cond + (1.0 - FM_CFG_ALPHA) * v_uncond
        save_activation(f"fm_step{step}_velocity", v)

        # Euler step
        x_t = x_t + v * dt

    # Quantize acoustic codes: clamp [-1,1], map to [0,20], offset by 2
    x_final = x_t.clamp(-1.0, 1.0)
    # Map from [-1, 1] to [0, 20]
    acoustic_codes = torch.round((x_final + 1.0) * 10.0).long()
    acoustic_codes = acoustic_codes.clamp(0, 20) + 2  # offset by 2
    save_activation("fm_acoustic_codes", acoustic_codes.float())

    return dict(saved)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Loading weights from {WEIGHTS_PATH} ...")
    if not WEIGHTS_PATH.exists():
        print(f"ERROR: Weights file not found at {WEIGHTS_PATH}", file=sys.stderr)
        sys.exit(1)

    weights = load_file(str(WEIGHTS_PATH))
    print(f"  Loaded {len(weights)} tensors")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Run 1
    print("\n=== Run 1 ===")
    results1 = run_forward(weights)

    # Run 2 (determinism check)
    print("\n=== Run 2 (determinism check) ===")
    results2 = run_forward(weights)

    # Verify bitwise equality
    print("\n=== Determinism Check ===")
    all_match = True
    for name in results1:
        if not np.array_equal(results1[name], results2[name]):
            print(f"  MISMATCH: {name}")
            all_match = False
    if all_match:
        print("  All activations are bitwise identical across runs.")
    else:
        print("  WARNING: Some activations differ between runs!", file=sys.stderr)
        sys.exit(1)

    total_files = len(list(OUTPUT_DIR.glob("*.npy")))
    print(f"\nDone. Saved {total_files} activation files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
