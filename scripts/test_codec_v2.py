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
Reference codec decoder using PyTorch native modules (nn.Conv1d, etc.)
for maximum correctness.

Takes real generated codes (from test_e2e.py) and decodes to audio,
saving intermediate activations for comparison with WebGPU.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "voxtral-tts"
WEIGHTS_PATH = MODEL_DIR / "consolidated.safetensors"
E2E_DIR = Path(__file__).resolve().parent.parent / "models" / "e2e_ref"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / "codec_ref"

# Codec hyperparams
DIM = 1024
HIDDEN_DIM = 4096
N_HEADS = 8
HEAD_DIM = 128
NORM_EPS = 0.01
QK_NORM_EPS = 1e-6
SEMANTIC_DIM = 256
N_ACOUSTIC = 36
PATCH_SIZE = 240


def load_codec_weights():
    """Load and extract codec weights."""
    all_w = load_file(str(WEIGHTS_PATH))
    w = {}
    for k, v in all_w.items():
        if k.startswith("audio_tokenizer."):
            short = k[len("audio_tokenizer."):]
            w[short] = v.float()
    return w


def vq_lookup(codes, w):
    """VQ codebook lookup with cluster_usage normalization."""
    emb_sum = w["quantizer.semantic_codebook.embedding_sum"]
    usage = w["quantizer.semantic_codebook.cluster_usage"]
    codebook = emb_sum / usage.clamp(min=1e-5).unsqueeze(1)
    return codebook[codes]  # [T, 256]


def fsq_dequant(codes, offset=2, levels=21):
    """FSQ dequantize: map integer codes back to [-1, 1]."""
    return (codes.float() - offset) * 2.0 / (levels - 1) - 1.0


def weight_norm_conv1d(x, weight_v, weight_g, stride=1, causal=True):
    """
    Causal Conv1d with weight normalization.
    x: [B, C_in, T]
    weight_v: [C_out, C_in, K] (original1)
    weight_g: [C_out, 1, 1] (original0)
    """
    # Apply weight norm: w = g * v / ||v||
    # norm over dims 1,2 for each output channel
    norms = torch.norm(weight_v.view(weight_v.shape[0], -1), dim=1, keepdim=True)
    norms = norms.unsqueeze(-1)  # [C_out, 1, 1]
    w = weight_g * weight_v / (norms + 1e-12)

    kernel = w.shape[2]
    if causal:
        pad = (kernel - 1)
        x = F.pad(x, (pad, 0))
    return F.conv1d(x, w, stride=stride)


def weight_norm_conv_transpose1d(x, weight_v, weight_g, stride=2, causal=True):
    """
    Causal ConvTranspose1d with weight normalization.
    x: [B, C_in, T]
    weight_v: [C_in, C_out, K] (ConvTranspose1d weight shape)
    weight_g: [C_in, 1, 1] (original0, weight_norm dim=0)
    """
    # Weight norm dim=0: normalize per first dim (c_in for ConvTranspose)
    norms = torch.norm(weight_v.view(weight_v.shape[0], -1), dim=1, keepdim=True)
    norms = norms.unsqueeze(-1)  # [C_in, 1, 1]
    w = weight_g * weight_v / (norms + 1e-12)

    kernel = w.shape[2]
    out = F.conv_transpose1d(x, w, stride=stride)
    # Causal: trim trailing samples
    trim = kernel - stride
    if trim > 0:
        out = out[:, :, :-trim]
    return out


def rms_norm(x, weight, eps=NORM_EPS):
    """RMS norm. x: [..., D]"""
    rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x.float() * rms).to(x.dtype) * weight


def qk_norm(x, weight, n_heads, head_dim, eps=QK_NORM_EPS):
    """Per-head RMS norm for QK normalization. x: [B, T, n_heads * head_dim]"""
    B, T, _ = x.shape
    x = x.view(B, T, n_heads, head_dim)
    rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    w = weight.view(1, 1, n_heads, head_dim)
    return (x.float() * rms * w).to(x.dtype).view(B, T, -1)


def alibi_attention(q, k, v, n_heads, head_dim, window):
    """ALiBi causal sliding-window attention."""
    B, T, _ = q.shape
    q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
    k = k.view(B, T, n_heads, head_dim).transpose(1, 2)
    v = v.view(B, T, n_heads, head_dim).transpose(1, 2)

    scale = head_dim ** -0.5
    scores = (q @ k.transpose(-2, -1)) * scale

    # ALiBi bias
    slopes = torch.pow(2.0, -8.0 * torch.arange(1, n_heads + 1, dtype=torch.float32) / n_heads)
    positions = torch.arange(T, dtype=torch.float32)
    bias = -(positions.unsqueeze(1) - positions.unsqueeze(0)).abs()
    bias = bias.unsqueeze(0) * slopes.unsqueeze(-1).unsqueeze(-1)
    scores = scores + bias.unsqueeze(0)

    # Causal mask
    causal_mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
    scores = scores + causal_mask

    # Sliding window mask
    if window > 0:
        window_mask = torch.full((T, T), float("-inf"))
        for i in range(T):
            start = max(0, i - window + 1)
            window_mask[i, start:i+1] = 0
        scores = scores + window_mask

    attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    out = attn @ v
    return out.transpose(1, 2).contiguous().view(B, T, -1)


def codec_transformer_block(x, w, prefix, n_heads, head_dim, hidden_dim, window):
    """Single codec transformer block."""
    dim = n_heads * head_dim

    # Attention
    residual = x
    h = rms_norm(x, w[f"{prefix}.attention_norm.weight"], NORM_EPS)
    q = h @ w[f"{prefix}.attention.wq.weight"].T
    k = h @ w[f"{prefix}.attention.wk.weight"].T
    v = h @ w[f"{prefix}.attention.wv.weight"].T
    q = qk_norm(q, w[f"{prefix}.attention.q_norm.weight"], n_heads, head_dim, QK_NORM_EPS)
    k = qk_norm(k, w[f"{prefix}.attention.k_norm.weight"], n_heads, head_dim, QK_NORM_EPS)
    attn_out = alibi_attention(q, k, v, n_heads, head_dim, window)
    attn_out = attn_out @ w[f"{prefix}.attention.wo.weight"].T
    attn_scale = w[f"{prefix}.attention_scale"]
    x = residual + attn_out * attn_scale.unsqueeze(0).unsqueeze(0)

    # FFN
    residual = x
    h = rms_norm(x, w[f"{prefix}.ffn_norm.weight"], NORM_EPS)
    gate = h @ w[f"{prefix}.feed_forward.w1.weight"].T
    up = h @ w[f"{prefix}.feed_forward.w3.weight"].T
    ffn_out = (F.silu(gate) * up) @ w[f"{prefix}.feed_forward.w2.weight"].T
    ffn_scale = w[f"{prefix}.ffn_scale"]
    x = residual + ffn_out * ffn_scale.unsqueeze(0).unsqueeze(0)

    return x


def codec_decode(semantic_codes, acoustic_codes, w, save_intermediates=True):
    """
    Full codec decode: codes → waveform.

    semantic_codes: [T] long tensor
    acoustic_codes: [T, 36] long tensor
    Returns: [T * 8 * 240] audio samples, dict of intermediates
    """
    T = semantic_codes.shape[0]
    intermediates = {}

    # 1. VQ lookup
    sem_embed = vq_lookup(semantic_codes, w)  # [T, 256]
    if save_intermediates:
        intermediates["vq_embed"] = sem_embed.detach().numpy()
        print(f"  VQ embed: shape={sem_embed.shape}, first4={sem_embed[0,:4].tolist()}")

    # 2. FSQ dequant
    ac_float = fsq_dequant(acoustic_codes)  # [T, 36]
    if save_intermediates:
        intermediates["fsq_dequant"] = ac_float.detach().numpy()

    # 3. Concat
    codec_input = torch.cat([sem_embed, ac_float], dim=-1)  # [T, 292]
    if save_intermediates:
        intermediates["concat"] = codec_input.detach().numpy()

    # Convert to [B, C, T] for convolutions
    x = codec_input.unsqueeze(0).transpose(1, 2)  # [1, 292, T]

    # 4. Input conv (block 0): CausalConv1d(292→1024, k=3, s=1)
    conv0_v = w["decoder_blocks.0.conv.parametrizations.weight.original1"]
    conv0_g = w["decoder_blocks.0.conv.parametrizations.weight.original0"]
    x = weight_norm_conv1d(x, conv0_v, conv0_g, stride=1)
    if save_intermediates:
        intermediates["after_input_conv"] = x.squeeze(0).transpose(0, 1).contiguous().detach().numpy()
        print(f"  After input conv: shape={x.shape}, max={x.abs().max():.6f}")

    # 5. Four decoder stages
    windows = [2, 4, 8, 16]
    stage_blocks = [
        (1, 2),   # transformer block 1, conv block 2 (stride 2)
        (3, 4),   # transformer block 3, conv block 4 (stride 2)
        (5, 6),   # transformer block 5, conv block 6 (stride 2)
        (7, None), # transformer block 7, no conv
    ]

    for stage_idx, (trans_block, conv_block) in enumerate(stage_blocks):
        # Transpose to [B, T, C] for transformer
        x_t = x.transpose(1, 2)  # [B, T, 1024]

        # 2 transformer layers
        for li in range(2):
            prefix = f"decoder_blocks.{trans_block}.layers.{li}"
            x_t = codec_transformer_block(x_t, w, prefix, N_HEADS, HEAD_DIM,
                                           HIDDEN_DIM, windows[stage_idx])

        x = x_t.transpose(1, 2)  # [B, 1024, T]

        if save_intermediates:
            intermediates[f"after_stage{stage_idx}_transformer"] = \
                x.squeeze(0).transpose(0, 1).contiguous().detach().numpy()
            print(f"  After stage {stage_idx} transformer: shape={x.shape}, "
                  f"max={x.abs().max():.6f}")

        # Conv transpose upsample (stages 0-2)
        if conv_block is not None:
            conv_v = w[f"decoder_blocks.{conv_block}.conv.parametrizations.weight.original1"]
            conv_g = w[f"decoder_blocks.{conv_block}.conv.parametrizations.weight.original0"]
            x = weight_norm_conv_transpose1d(x, conv_v, conv_g, stride=2)

            if save_intermediates:
                intermediates[f"after_stage{stage_idx}_conv_up"] = \
                    x.squeeze(0).transpose(0, 1).contiguous().detach().numpy()
                print(f"  After stage {stage_idx} conv up: shape={x.shape}, "
                      f"max={x.abs().max():.6f}")

    # 6. Output conv: CausalConv1d(1024→240, k=7, s=1)
    out_v = w["output_proj.conv.parametrizations.weight.original1"]
    out_g = w["output_proj.conv.parametrizations.weight.original0"]
    x = weight_norm_conv1d(x, out_v, out_g, stride=1)
    if save_intermediates:
        intermediates["after_output_conv"] = x.squeeze(0).transpose(0, 1).contiguous().detach().numpy()
        print(f"  After output conv: shape={x.shape}, max={x.abs().max():.6f}")

    # 7. Reshape to waveform: [B, 240, T'] → [B, 1, T'*240]
    B, patch_size, T_out = x.shape
    audio = x.transpose(1, 2).reshape(B, 1, T_out * patch_size)

    if save_intermediates:
        intermediates["audio"] = audio.squeeze().detach().numpy()
        print(f"  Audio: shape={audio.shape}, max={audio.abs().max():.6f}")

    return audio, intermediates


def main():
    print("Loading weights...")
    w = load_codec_weights()
    print(f"  Loaded {len(w)} codec tensors")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load codes from e2e reference
    sem_codes_f32 = np.load(E2E_DIR / "semantic_codes.npy")
    ac_codes_f32 = np.load(E2E_DIR / "acoustic_codes.npy")

    semantic_codes = torch.tensor(sem_codes_f32.astype(int), dtype=torch.long)
    acoustic_codes = torch.tensor(ac_codes_f32.astype(int), dtype=torch.long)

    T = semantic_codes.shape[0]
    print(f"\nDecoding {T} frames...")
    print(f"  Semantic codes: {semantic_codes.tolist()}")
    print(f"  Acoustic codes[0][:6]: {acoustic_codes[0, :6].tolist()}")

    audio, intermediates = codec_decode(semantic_codes, acoustic_codes, w)

    # Save intermediates
    for name, data in intermediates.items():
        np.save(OUTPUT_DIR / f"{name}.npy", data.astype(np.float32))

    # Save audio as WAV
    import soundfile as sf
    audio_np = audio.squeeze().detach().numpy()
    sf.write(str(OUTPUT_DIR / "reference_audio.wav"), audio_np, 24000)
    print(f"\nSaved {len(intermediates)} activation files and audio to {OUTPUT_DIR}")
    print(f"Audio: {len(audio_np)} samples = {len(audio_np)/24000:.2f}s")


if __name__ == "__main__":
    main()
