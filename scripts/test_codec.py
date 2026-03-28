#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "safetensors",
#   "numpy",
#   "soundfile",
#   "packaging",
# ]
# ///
"""
Test codec decoder by running it in PyTorch and comparing output.

Takes semantic + acoustic codes and decodes them through the model's
audio_tokenizer codec to produce reference audio.
"""

import sys
import json
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "voxtral-tts"
WEIGHTS_PATH = MODEL_DIR / "consolidated.safetensors"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / "activations"


def load_codec_weights(all_weights: dict) -> dict:
    """Extract codec decoder weights."""
    codec_w = {}
    for k, v in all_weights.items():
        if k.startswith("audio_tokenizer."):
            short = k[len("audio_tokenizer."):]
            codec_w[short] = v.float()
    return codec_w


def weight_norm_conv(weight: torch.Tensor, g: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Apply weight normalization: v * g / ||v||"""
    # Compute L2 norm over all dims except `dim`
    norms = torch.norm(weight.view(weight.shape[dim], -1) if dim == 0 else weight.transpose(0, dim).reshape(weight.shape[dim], -1), dim=1)
    # Reshape norms to broadcast with weight
    shape = [1] * weight.dim()
    shape[dim] = -1
    norms = norms.view(shape)
    return weight * (g / (norms + 1e-12))


def causal_conv1d(x: torch.Tensor, weight: torch.Tensor, g: torch.Tensor,
                  stride: int = 1) -> torch.Tensor:
    """Causal 1D convolution with weight norm. x: [B, C_in, T]"""
    # Apply weight normalization
    w_normed = weight_norm_conv(weight, g, dim=0)
    kernel = weight.shape[2]
    pad = (kernel - 1) * 1  # dilation=1
    x_padded = F.pad(x, (pad, 0))
    return F.conv1d(x_padded, w_normed, stride=stride)


def causal_conv_transpose1d(x: torch.Tensor, weight: torch.Tensor, g: torch.Tensor,
                            stride: int = 2) -> torch.Tensor:
    """Causal 1D transposed convolution with weight norm. x: [B, C_in, T]"""
    # Weight norm for ConvTranspose1d is on dim=1 (PyTorch convention is output channels for ConvTranspose)
    # Actually for ConvTranspose1d weight is [c_in, c_out, kernel], and weight_norm with dim=0 normalizes per c_in
    w_normed = weight_norm_conv(weight, g, dim=0)
    kernel = weight.shape[2]
    # Standard transposed conv
    out = F.conv_transpose1d(x, w_normed, stride=stride)
    # Trim to causal: remove trailing samples
    trim = kernel - stride
    if trim > 0:
        out = out[:, :, :-trim]
    return out


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 0.01) -> torch.Tensor:
    """RMSNorm"""
    rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x.float() * rms).to(x.dtype) * weight


def qk_norm(x: torch.Tensor, weight: torch.Tensor, n_heads: int, head_dim: int,
            eps: float = 1e-6) -> torch.Tensor:
    """QK normalization: per-head RMS norm. x: [B, T, n_heads * head_dim]"""
    B, T, _ = x.shape
    x = x.view(B, T, n_heads, head_dim)
    rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    w = weight.view(1, 1, n_heads, head_dim)
    return (x.float() * rms * w).to(x.dtype).view(B, T, -1)


def alibi_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    n_heads: int, head_dim: int, window: int) -> torch.Tensor:
    """ALiBi causal sliding window attention. q/k/v: [B, T, n_heads * head_dim]"""
    B, T, _ = q.shape
    q = q.view(B, T, n_heads, head_dim).transpose(1, 2)  # [B, H, T, D]
    k = k.view(B, T, n_heads, head_dim).transpose(1, 2)
    v = v.view(B, T, n_heads, head_dim).transpose(1, 2)

    scale = head_dim ** -0.5
    scores = (q @ k.transpose(-2, -1)) * scale  # [B, H, T, T]

    # ALiBi bias
    slopes = torch.pow(2.0, -8.0 * torch.arange(1, n_heads + 1, dtype=torch.float32, device=q.device) / n_heads)
    positions = torch.arange(T, dtype=torch.float32, device=q.device)
    bias = -(positions.unsqueeze(1) - positions.unsqueeze(0)).abs()  # [T, T]
    bias = bias.unsqueeze(0) * slopes.unsqueeze(-1).unsqueeze(-1)  # [H, T, T]
    scores = scores + bias.unsqueeze(0)

    # Causal mask
    causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=q.device), diagonal=1)
    scores = scores + causal_mask

    # Sliding window mask
    if window > 0:
        window_mask = torch.full((T, T), float("-inf"), device=q.device)
        for i in range(T):
            start = max(0, i - window + 1)
            window_mask[i, start:i+1] = 0
        scores = scores + window_mask

    attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    out = attn @ v  # [B, H, T, D]
    return out.transpose(1, 2).contiguous().view(B, T, -1)


def transformer_block(x: torch.Tensor, w: dict, prefix: str,
                      n_heads: int, head_dim: int, hidden_dim: int,
                      window: int, norm_eps: float, qk_norm_eps: float) -> torch.Tensor:
    """Single transformer block with layer scale."""
    dim = n_heads * head_dim

    # Attention
    residual = x
    h = rms_norm(x, w[f"{prefix}.attention_norm.weight"], norm_eps)

    q = h @ w[f"{prefix}.attention.wq.weight"].T
    k = h @ w[f"{prefix}.attention.wk.weight"].T
    v = h @ w[f"{prefix}.attention.wv.weight"].T

    q = qk_norm(q, w[f"{prefix}.attention.q_norm.weight"], n_heads, head_dim, qk_norm_eps)
    k = qk_norm(k, w[f"{prefix}.attention.k_norm.weight"], n_heads, head_dim, qk_norm_eps)

    attn_out = alibi_attention(q, k, v, n_heads, head_dim, window)
    attn_out = attn_out @ w[f"{prefix}.attention.wo.weight"].T

    # Layer scale + residual
    attn_scale = w.get(f"{prefix}.attention_scale.weight", w.get(f"{prefix}.attention_scale"))
    x = residual + attn_out * attn_scale.unsqueeze(0).unsqueeze(0)

    # FFN
    residual = x
    h = rms_norm(x, w[f"{prefix}.ffn_norm.weight"], norm_eps)

    gate = h @ w[f"{prefix}.feed_forward.w1.weight"].T
    up = h @ w[f"{prefix}.feed_forward.w3.weight"].T
    ffn_out = (F.silu(gate) * up) @ w[f"{prefix}.feed_forward.w2.weight"].T

    ffn_scale = w.get(f"{prefix}.ffn_scale.weight", w.get(f"{prefix}.ffn_scale"))
    x = residual + ffn_out * ffn_scale.unsqueeze(0).unsqueeze(0)

    return x


def codec_decode(semantic_codes: torch.Tensor, acoustic_codes: torch.Tensor,
                 w: dict) -> torch.Tensor:
    """
    Decode semantic + acoustic codes to audio waveform.

    semantic_codes: [T] long tensor
    acoustic_codes: [T, 36] long tensor
    Returns: [1, 1, T*1920] audio waveform
    """
    T = semantic_codes.shape[0]

    # 1. VQ lookup: semantic codes -> embeddings
    # Normalize codebook first
    emb_sum = w["quantizer.semantic_codebook.embedding_sum"]  # [8192, 256]
    usage = w["quantizer.semantic_codebook.cluster_usage"]  # [8192]
    codebook = emb_sum / usage.clamp(min=1e-5).unsqueeze(1)

    sem_embed = codebook[semantic_codes]  # [T, 256]

    # 2. FSQ dequant: acoustic codes -> floats
    # (code - 2) * 2 / 20 - 1
    ac_float = (acoustic_codes.float() - 2.0) * 2.0 / 20.0 - 1.0  # [T, 36]

    # 3. Concat
    codec_input = torch.cat([sem_embed, ac_float], dim=-1)  # [T, 292]

    # Convert to [B, C, T] for conv operations
    x = codec_input.unsqueeze(0).transpose(1, 2)  # [1, 292, T]

    # 4. Input conv: CausalConv1d(292->1024, k=3, s=1)
    conv0_w = w["decoder_blocks.0.conv.parametrizations.weight.original1"]  # [c_out, c_in, k]
    conv0_g = w["decoder_blocks.0.conv.parametrizations.weight.original0"]  # [c_out, 1, 1]
    x = causal_conv1d(x, conv0_w, conv0_g, stride=1)

    # 5. Decoder stages
    strides = [1, 2, 2, 2]  # from params.json decoder_convs_strides_str
    kernels = [3, 4, 4, 4]  # from params.json decoder_convs_kernels_str
    windows = [2, 4, 8, 16]  # base=16, halved per downsample, reversed for decoder
    n_heads = 8
    head_dim = 128
    hidden_dim = 4096
    norm_eps = 0.01
    qk_norm_eps = 1e-6

    # Blocks: 1(transformer), 2(conv), 3(transformer), 4(conv), 5(transformer), 6(conv), 7(transformer)
    stage_blocks = [(1, 2), (3, 4), (5, 6), (7, None)]

    for stage, (trans_block, conv_block) in enumerate(stage_blocks):
        # Transformer layers
        # Convert to [B, T, C] for transformer
        x_t = x.transpose(1, 2)  # [B, T, C]
        for li in range(2):
            prefix = f"decoder_blocks.{trans_block}.layers.{li}"
            x_t = transformer_block(x_t, w, prefix, n_heads, head_dim, hidden_dim,
                                    windows[stage], norm_eps, qk_norm_eps)
        x = x_t.transpose(1, 2)  # [B, C, T]

        # Conv transpose upsample
        if conv_block is not None and strides[stage + 1 if conv_block else stage] > 1:
            stride = strides[conv_block // 2]  # strides[1]=2, strides[2]=2, strides[3]=2
            kernel = kernels[conv_block // 2]
            conv_w = w[f"decoder_blocks.{conv_block}.conv.parametrizations.weight.original1"]
            conv_g = w[f"decoder_blocks.{conv_block}.conv.parametrizations.weight.original0"]
            x = causal_conv_transpose1d(x, conv_w, conv_g, stride=stride)

    # 6. Output conv: CausalConv1d(1024->240, k=7)
    out_w = w["output_proj.conv.parametrizations.weight.original1"]
    out_g = w["output_proj.conv.parametrizations.weight.original0"]
    x = causal_conv1d(x, out_w, out_g, stride=1)

    # 7. Reshape patches to waveform: [B, 240, T'] -> [B, 1, T'*240]
    B, patch_size, T_out = x.shape
    x = x.transpose(1, 2).reshape(B, 1, T_out * patch_size)

    return x


def main():
    print("Loading weights...")
    weights = load_file(str(WEIGHTS_PATH))

    codec_w = load_codec_weights(weights)
    print(f"  Loaded {len(codec_w)} codec tensors")

    # Test with simple codes
    T = 4
    torch.manual_seed(42)

    # Use some plausible semantic codes
    semantic_codes = torch.tensor([1789, 6124, 855, 10], dtype=torch.long)
    # Use neutral acoustic codes (12 = middle of [2,22] range)
    acoustic_codes = torch.full((T, 36), 12, dtype=torch.long)

    print(f"\nDecoding {T} frames...")
    print(f"  Semantic codes: {semantic_codes.tolist()}")

    audio = codec_decode(semantic_codes, acoustic_codes, codec_w)
    print(f"  Output shape: {audio.shape}")
    print(f"  Audio max: {audio.abs().max().item():.6f}")
    print(f"  Audio first 10: {audio[0, 0, :10].tolist()}")

    # Save as numpy
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "codec_ref_audio.npy", audio.detach().numpy())
    np.save(OUTPUT_DIR / "codec_ref_semantic_codes.npy", semantic_codes.numpy())
    np.save(OUTPUT_DIR / "codec_ref_acoustic_codes.npy", acoustic_codes.numpy())

    print(f"\nSaved to {OUTPUT_DIR}")

    # Also save intermediate activations for debugging
    # Re-run with saving intermediates
    print("\n=== Intermediate activations ===")

    # VQ lookup
    emb_sum = codec_w["quantizer.semantic_codebook.embedding_sum"]
    usage = codec_w["quantizer.semantic_codebook.cluster_usage"]
    codebook = emb_sum / usage.clamp(min=1e-5).unsqueeze(1)
    sem_embed = codebook[semantic_codes]
    print(f"  VQ embed shape: {sem_embed.shape}")
    print(f"  VQ embed first4: {sem_embed[0, :4].tolist()}")
    np.save(OUTPUT_DIR / "codec_ref_vq_embed.npy", sem_embed.detach().numpy())

    # FSQ dequant
    ac_float = (acoustic_codes.float() - 2.0) * 2.0 / 20.0 - 1.0
    print(f"  FSQ dequant: {ac_float[0, :4].tolist()}")

    # Concat
    codec_input = torch.cat([sem_embed, ac_float], dim=-1)
    print(f"  Concat shape: {codec_input.shape}")
    np.save(OUTPUT_DIR / "codec_ref_concat.npy", codec_input.detach().numpy())

    # Input conv output
    x = codec_input.unsqueeze(0).transpose(1, 2)
    conv0_w = codec_w["decoder_blocks.0.conv.parametrizations.weight.original1"]
    conv0_g = codec_w["decoder_blocks.0.conv.parametrizations.weight.original0"]
    x = causal_conv1d(x, conv0_w, conv0_g, stride=1)
    print(f"  After input conv: {x.shape}, max={x.abs().max().item():.6f}")
    np.save(OUTPUT_DIR / "codec_ref_after_input_conv.npy", x.detach().numpy())

    print("\nDone!")


if __name__ == "__main__":
    main()
