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
End-to-end Voxtral TTS reference: backbone + FM + codec in Python.

Generates 3 frames autoregressively, saving all intermediate activations
for comparison with the WebGPU implementation.

Usage: uv run scripts/test_e2e.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "voxtral-tts"
WEIGHTS_PATH = MODEL_DIR / "consolidated.safetensors"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / "e2e_ref"

# Backbone hyperparams
DIM = 3072
N_LAYERS = 26
HEAD_DIM = 128
N_HEADS = 32
N_KV_HEADS = 8
HIDDEN_DIM = 9216
ROPE_THETA = 1_000_000.0
NORM_EPS = 1e-5
VOCAB_SIZE = 131072

# FM hyperparams
FM_DIM = 3072
FM_N_LAYERS = 3
FM_N_HEADS = 32
FM_N_KV_HEADS = 8
FM_HIDDEN_DIM = 9216
FM_ROPE_THETA = 10_000.0
FM_N_STEPS = 8
FM_CFG_ALPHA = 1.2
ACOUSTIC_DIM = 36
SEMANTIC_VOCAB = 8320

# Token IDs
BOS = 1
EOS = 2
INST = 3
INST_END = 4
AUDIO = 24
OUTPUT_AUDIO = 26


def rms_norm(x, weight, eps=NORM_EPS):
    dtype = x.dtype
    x = x.float()
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * rms).to(dtype) * weight


def precompute_freqs_cis(head_dim, seq_len, theta):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x, freqs_cis):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)
    x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return x_rotated.to(x.dtype)


def apply_rope_single(x, freqs_cis_pos):
    """Apply RoPE for a single position. x: [n_heads, 1, head_dim], freqs: [head_dim/2]"""
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs_cis_pos.unsqueeze(0).unsqueeze(0)  # [1, 1, head_dim/2]
    x_rotated = torch.view_as_real(x_complex * freqs).flatten(-2)
    return x_rotated.to(x.dtype)


def sinusoidal_time_embedding(t, dim):
    half_dim = dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = torch.tensor(t, dtype=torch.float32) * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)])
    return emb


def swiglu(x, w1, w2, w3):
    return (F.silu(x @ w1.T) * (x @ w3.T)) @ w2.T


class BackboneKVCache:
    """KV cache for autoregressive backbone inference."""
    def __init__(self, max_len=2048):
        self.k_cache = [None] * N_LAYERS  # Will be [n_kv_heads, cur_len, head_dim]
        self.v_cache = [None] * N_LAYERS
        self.max_len = max_len

    def update(self, layer, k, v, pos):
        """Update KV cache for a layer. k,v: [1, n_kv_heads, 1, head_dim]"""
        k = k.squeeze(0)  # [n_kv_heads, 1, head_dim]
        v = v.squeeze(0)
        if self.k_cache[layer] is None:
            self.k_cache[layer] = k
            self.v_cache[layer] = v
        else:
            self.k_cache[layer] = torch.cat([self.k_cache[layer], k], dim=1)
            self.v_cache[layer] = torch.cat([self.v_cache[layer], v], dim=1)

    def get(self, layer):
        """Get full KV cache for a layer. Returns [n_kv_heads, cur_len, head_dim]"""
        return self.k_cache[layer], self.v_cache[layer]


def backbone_step(w, token_id, pos, kv_cache, use_audio_embed=False,
                  voice_embed=None, freqs_all=None, save_prefix=""):
    """
    Single autoregressive backbone step.
    Returns: normed hidden state [1, DIM]
    """
    activations = {}

    # Embedding
    if use_audio_embed:
        embed_table = w["mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight"]
    else:
        embed_table = w["mm_audio_embeddings.tok_embeddings.weight"]

    h = embed_table[token_id:token_id+1].unsqueeze(0)  # [1, 1, DIM]

    # Add voice embedding if provided (sum mode)
    if voice_embed is not None:
        h = h + voice_embed.unsqueeze(0).unsqueeze(0)  # [1, 1, DIM]

    if save_prefix:
        activations[f"{save_prefix}_embed"] = h.squeeze().detach().numpy()

    # Get RoPE for this position
    freqs_pos = freqs_all[pos:pos+1]  # [1, head_dim/2]

    for i in range(N_LAYERS):
        prefix = f"layers.{i}"

        # Attention norm
        h_normed = rms_norm(h, w[f"{prefix}.attention_norm.weight"])

        # Q/K/V projections
        q = h_normed @ w[f"{prefix}.attention.wq.weight"].T  # [1, 1, n_heads*head_dim]
        k = h_normed @ w[f"{prefix}.attention.wk.weight"].T  # [1, 1, n_kv_heads*head_dim]
        v = h_normed @ w[f"{prefix}.attention.wv.weight"].T

        q = q.view(1, 1, N_HEADS, HEAD_DIM).transpose(1, 2)      # [1, n_heads, 1, head_dim]
        k = k.view(1, 1, N_KV_HEADS, HEAD_DIM).transpose(1, 2)   # [1, n_kv_heads, 1, head_dim]
        v = v.view(1, 1, N_KV_HEADS, HEAD_DIM).transpose(1, 2)

        # RoPE
        q = apply_rope_single(q.squeeze(0), freqs_pos.squeeze(0)).unsqueeze(0)
        k = apply_rope_single(k.squeeze(0), freqs_pos.squeeze(0)).unsqueeze(0)

        # Update KV cache
        kv_cache.update(i, k, v, pos)
        k_full, v_full = kv_cache.get(i)  # [n_kv_heads, seq_len, head_dim]

        # Repeat KV for GQA
        n_rep = N_HEADS // N_KV_HEADS
        k_full_rep = k_full.repeat_interleave(n_rep, dim=0)  # [n_heads, seq_len, head_dim]
        v_full_rep = v_full.repeat_interleave(n_rep, dim=0)

        # Attention scores
        q_squeezed = q.squeeze(0)  # [n_heads, 1, head_dim]
        scale = HEAD_DIM ** -0.5
        scores = (q_squeezed @ k_full_rep.transpose(-2, -1)) * scale  # [n_heads, 1, seq_len]
        attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = attn @ v_full_rep  # [n_heads, 1, head_dim]
        out = out.transpose(0, 1).contiguous().view(1, 1, -1)  # [1, 1, n_heads*head_dim]

        # Output projection
        attn_out = out @ w[f"{prefix}.attention.wo.weight"].T
        h = h + attn_out

        # FFN
        h_normed = rms_norm(h, w[f"{prefix}.ffn_norm.weight"])
        ffn_out = swiglu(h_normed, w[f"{prefix}.feed_forward.w1.weight"],
                         w[f"{prefix}.feed_forward.w2.weight"],
                         w[f"{prefix}.feed_forward.w3.weight"])
        h = h + ffn_out

    # Final norm
    normed = rms_norm(h, w["norm.weight"])
    hidden = normed.squeeze(0)  # [1, DIM]

    if save_prefix:
        activations[f"{save_prefix}_hidden"] = hidden.squeeze().detach().numpy()

    return hidden, activations


def fm_forward(w, hidden, seed=42):
    """
    Full FM forward: semantic logits + 8-step Euler ODE with CFG.
    Returns: semantic_logits, acoustic_codes, activations dict
    """
    activations = {}

    # Semantic logits
    semantic_logits = hidden @ w["acoustic_transformer.semantic_codebook_output.weight"].T  # [1, 8320]
    activations["semantic_logits"] = semantic_logits.squeeze().detach().numpy()

    # LLM projection (conditioned)
    llm_proj_cond = hidden @ w["acoustic_transformer.llm_projection.weight"].T
    llm_proj_uncond = torch.zeros_like(llm_proj_cond)

    # Initial noise
    torch.manual_seed(seed)
    x_t = torch.randn(1, ACOUSTIC_DIM, dtype=torch.float32)
    activations["initial_noise"] = x_t.squeeze().detach().numpy()

    # Precompute RoPE for FM (seq_len=3)
    freqs_cis_fm = precompute_freqs_cis(HEAD_DIM, 3, FM_ROPE_THETA)

    dt = 1.0 / FM_N_STEPS
    for step in range(FM_N_STEPS):
        t = step * dt

        # Time embedding
        time_emb = sinusoidal_time_embedding(t, FM_DIM).unsqueeze(0)
        time_proj = time_emb @ w["acoustic_transformer.time_projection.weight"].T

        # Input projection
        x_t_proj = x_t @ w["acoustic_transformer.input_projection.weight"].T

        # Build 3-token sequences
        seq_cond = torch.stack([x_t_proj.squeeze(0), time_proj.squeeze(0),
                                llm_proj_cond.squeeze(0)]).unsqueeze(0)
        seq_uncond = torch.stack([x_t_proj.squeeze(0), time_proj.squeeze(0),
                                  llm_proj_uncond.squeeze(0)]).unsqueeze(0)

        for cond_idx, seq in enumerate([seq_cond, seq_uncond]):
            h_fm = seq.clone()  # [1, 3, DIM]

            for li in range(FM_N_LAYERS):
                pfx = f"acoustic_transformer.layers.{li}"

                # Attention
                h_fm_normed = rms_norm(h_fm, w[f"{pfx}.attention_norm.weight"])
                bsz, seqlen, _ = h_fm_normed.shape

                q = h_fm_normed @ w[f"{pfx}.attention.wq.weight"].T
                k = h_fm_normed @ w[f"{pfx}.attention.wk.weight"].T
                v = h_fm_normed @ w[f"{pfx}.attention.wv.weight"].T

                q = q.view(bsz, seqlen, FM_N_HEADS, HEAD_DIM).transpose(1, 2)
                k = k.view(bsz, seqlen, FM_N_KV_HEADS, HEAD_DIM).transpose(1, 2)
                v = v.view(bsz, seqlen, FM_N_KV_HEADS, HEAD_DIM).transpose(1, 2)

                q = apply_rope(q, freqs_cis_fm)
                k = apply_rope(k, freqs_cis_fm)

                n_rep = FM_N_HEADS // FM_N_KV_HEADS
                k = k.repeat_interleave(n_rep, dim=1)
                v = v.repeat_interleave(n_rep, dim=1)

                scale = HEAD_DIM ** -0.5
                scores = (q @ k.transpose(-2, -1)) * scale
                attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
                out = attn @ v
                out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
                attn_out = out @ w[f"{pfx}.attention.wo.weight"].T
                h_fm = h_fm + attn_out

                # FFN
                h_fm_normed = rms_norm(h_fm, w[f"{pfx}.ffn_norm.weight"])
                ffn_out = swiglu(h_fm_normed, w[f"{pfx}.feed_forward.w1.weight"],
                                 w[f"{pfx}.feed_forward.w2.weight"],
                                 w[f"{pfx}.feed_forward.w3.weight"])
                h_fm = h_fm + ffn_out

            # Final norm
            h_fm = rms_norm(h_fm, w["acoustic_transformer.norm.weight"])
            pos0 = h_fm[:, 0, :]
            velocity = pos0 @ w["acoustic_transformer.acoustic_codebook_output.weight"].T

            if cond_idx == 0:
                v_cond = velocity
            else:
                v_uncond = velocity

        # CFG combine
        v = FM_CFG_ALPHA * v_cond + (1.0 - FM_CFG_ALPHA) * v_uncond
        activations[f"step{step}_velocity"] = v.squeeze().detach().numpy()

        # Euler step
        x_t = x_t + v * dt

    activations["x_final"] = x_t.squeeze().detach().numpy()

    # FSQ quantize
    x_final = x_t.clamp(-1.0, 1.0)
    acoustic_codes = torch.round((x_final + 1.0) * 10.0).long()
    acoustic_codes = acoustic_codes.clamp(0, 20) + 2
    activations["acoustic_codes"] = acoustic_codes.float().squeeze().detach().numpy()

    return semantic_logits, acoustic_codes, activations


def main():
    print("Loading weights...")
    weights = load_file(str(WEIGHTS_PATH))
    w = {k: v.float() for k, v in weights.items()}
    print(f"  Loaded {len(w)} tensors")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build a minimal prompt: [BOS, INST, INST_END, OUTPUT_AUDIO]
    # (No voice embeddings, no text — simplest possible test)
    prompt_tokens = [BOS, INST, INST_END, OUTPUT_AUDIO]

    # Precompute RoPE for full sequence (prompt + generated)
    max_seq = len(prompt_tokens) + 10  # room for generated frames
    freqs_all = precompute_freqs_cis(HEAD_DIM, max_seq, ROPE_THETA)

    # Initialize KV cache
    kv_cache = BackboneKVCache()

    print(f"\n=== Phase 1: Prefill {len(prompt_tokens)} tokens ===")
    for i, tok in enumerate(prompt_tokens):
        print(f"  Token {i}: id={tok}, pos={i}")
        save_prefix = f"prefill_tok{i}" if i == len(prompt_tokens) - 1 else ""
        hidden, acts = backbone_step(w, tok, i, kv_cache, use_audio_embed=False,
                                      freqs_all=freqs_all, save_prefix=save_prefix)
        for k2, v2 in acts.items():
            np.save(OUTPUT_DIR / f"{k2}.npy", v2)

    print(f"  Final hidden (after OUTPUT_AUDIO) first 4: {hidden[0, :4].tolist()}")

    # Save the hidden state for frame 0
    np.save(OUTPUT_DIR / "frame0_backbone_hidden.npy", hidden.squeeze().detach().numpy())

    print(f"\n=== Phase 2: Generate 3 frames ===")
    pos = len(prompt_tokens)
    semantic_codes = []
    all_acoustic_codes = []

    for frame in range(3):
        print(f"\n--- Frame {frame} ---")

        if frame > 0:
            # Embed previous semantic code through backbone
            code = semantic_codes[-1]
            print(f"  Backbone step: audio code {code} at pos {pos}")
            hidden, acts = backbone_step(w, code, pos, kv_cache,
                                          use_audio_embed=True,
                                          freqs_all=freqs_all,
                                          save_prefix=f"frame{frame}_backbone")
            for k2, v2 in acts.items():
                np.save(OUTPUT_DIR / f"{k2}.npy", v2)
            pos += 1

        # FM forward
        print(f"  FM forward (seed=42)...")
        sem_logits, ac_codes, fm_acts = fm_forward(w, hidden, seed=42)

        # Save FM activations
        for k2, v2 in fm_acts.items():
            np.save(OUTPUT_DIR / f"frame{frame}_fm_{k2}.npy", v2)

        # Semantic code = argmax
        sem_code = sem_logits.squeeze().argmax().item()
        ac_list = ac_codes.squeeze().tolist()

        print(f"  Semantic logits top5: {sem_logits.squeeze().topk(5)}")
        print(f"  Semantic code (argmax): {sem_code}")
        print(f"  Acoustic codes first 6: {ac_list[:6]}")

        semantic_codes.append(sem_code)
        all_acoustic_codes.append(ac_list)

        np.save(OUTPUT_DIR / f"frame{frame}_backbone_hidden.npy",
                hidden.squeeze().detach().numpy())

    print(f"\n=== Results ===")
    print(f"  Semantic codes: {semantic_codes}")
    print(f"  Acoustic codes[0]: {all_acoustic_codes[0][:10]}")

    # Save codes as float32 (JS parseNpy reads as Float32Array)
    np.save(OUTPUT_DIR / "semantic_codes.npy", np.array(semantic_codes, dtype=np.float32))
    np.save(OUTPUT_DIR / "acoustic_codes.npy", np.array(all_acoustic_codes, dtype=np.float32))

    print(f"\nAll activations saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
