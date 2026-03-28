#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "safetensors",
#   "numpy",
#   "soundfile",
#   "packaging",
#   "mistral-common",
# ]
# ///
"""
Full E2E generation with voice embeddings and proper text tokenization.
Generates audio and saves it for comparison with WebGPU output.
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
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / "python_gen"

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
BEGIN_AUDIO = 25
OUTPUT_AUDIO = 26
AUDIO_TO_TEXT = 35   # [REPEAT_AUDIO_TEXT]
TEXT_TO_AUDIO = 36   # [NEXT_AUDIO_TEXT]


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


def apply_rope_single(x, freqs_cis_pos):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs_cis_pos.unsqueeze(0).unsqueeze(0)
    x_rotated = torch.view_as_real(x_complex * freqs).flatten(-2)
    return x_rotated.to(x.dtype)


def apply_rope(x, freqs_cis):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)
    x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return x_rotated.to(x.dtype)


def sinusoidal_time_embedding(t, dim):
    half_dim = dim // 2
    # Match vLLM-Omni and MLX: divide by half_dim (not half_dim - 1), (cos, sin) order
    inv_freq = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(half_dim, dtype=torch.float32) / half_dim)
    emb = torch.tensor(t, dtype=torch.float32) * inv_freq
    return torch.cat([torch.cos(emb), torch.sin(emb)])  # (cos, sin) order


def swiglu(x, w1, w2, w3):
    return (F.silu(x @ w1.T) * (x @ w3.T)) @ w2.T


class BackboneKVCache:
    def __init__(self, max_len=2048):
        self.k_cache = [None] * N_LAYERS
        self.v_cache = [None] * N_LAYERS

    def update(self, layer, k, v, pos):
        k = k.squeeze(0)
        v = v.squeeze(0)
        if self.k_cache[layer] is None:
            self.k_cache[layer] = k
            self.v_cache[layer] = v
        else:
            self.k_cache[layer] = torch.cat([self.k_cache[layer], k], dim=1)
            self.v_cache[layer] = torch.cat([self.v_cache[layer], v], dim=1)

    def get(self, layer):
        return self.k_cache[layer], self.v_cache[layer]


def backbone_step(w, token_id, pos, kv_cache, use_audio_embed=False,
                  voice_embed=None, freqs_all=None, precomputed_embed=None):
    if precomputed_embed is not None:
        # Use pre-computed embedding directly (e.g., multi-codebook sum)
        h = precomputed_embed
    else:
        if use_audio_embed:
            embed_table = w["mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight"]
        else:
            embed_table = w["mm_audio_embeddings.tok_embeddings.weight"]

        h = embed_table[token_id:token_id+1].unsqueeze(0)

        if voice_embed is not None:
            # REPLACE token embedding with voice embedding
            h = voice_embed.unsqueeze(0).unsqueeze(0)

    freqs_pos = freqs_all[pos:pos+1]

    for i in range(N_LAYERS):
        prefix = f"layers.{i}"
        h_normed = rms_norm(h, w[f"{prefix}.attention_norm.weight"])

        q = h_normed @ w[f"{prefix}.attention.wq.weight"].T
        k = h_normed @ w[f"{prefix}.attention.wk.weight"].T
        v = h_normed @ w[f"{prefix}.attention.wv.weight"].T

        q = q.view(1, 1, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(1, 1, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(1, 1, N_KV_HEADS, HEAD_DIM).transpose(1, 2)

        q = apply_rope_single(q.squeeze(0), freqs_pos.squeeze(0)).unsqueeze(0)
        k = apply_rope_single(k.squeeze(0), freqs_pos.squeeze(0)).unsqueeze(0)

        kv_cache.update(i, k, v, pos)
        k_full, v_full = kv_cache.get(i)

        n_rep = N_HEADS // N_KV_HEADS
        k_full_rep = k_full.repeat_interleave(n_rep, dim=0)
        v_full_rep = v_full.repeat_interleave(n_rep, dim=0)

        q_squeezed = q.squeeze(0)
        scale = HEAD_DIM ** -0.5
        scores = (q_squeezed @ k_full_rep.transpose(-2, -1)) * scale
        attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = attn @ v_full_rep
        out = out.transpose(0, 1).contiguous().view(1, 1, -1)
        attn_out = out @ w[f"{prefix}.attention.wo.weight"].T
        h = h + attn_out

        h_normed = rms_norm(h, w[f"{prefix}.ffn_norm.weight"])
        ffn_out = swiglu(h_normed, w[f"{prefix}.feed_forward.w1.weight"],
                         w[f"{prefix}.feed_forward.w2.weight"],
                         w[f"{prefix}.feed_forward.w3.weight"])
        h = h + ffn_out

    normed = rms_norm(h, w["norm.weight"])
    hidden = normed.squeeze(0)
    return hidden


def fm_forward(w, hidden):
    """FM forward with random noise (no seed pinning)."""
    # FM always runs in float32 for numerical stability
    hidden_f32 = hidden.float()
    semantic_logits = hidden_f32 @ w["acoustic_transformer.semantic_codebook_output.weight"].float().T
    llm_proj_cond = hidden_f32 @ w["acoustic_transformer.llm_projection.weight"].float().T
    llm_proj_uncond = torch.zeros_like(llm_proj_cond)

    x_t = torch.randn(1, ACOUSTIC_DIM, dtype=torch.float32)

    # Euler integration: linspace(0, 1, N_STEPS) gives N_STEPS-1 integration steps
    timesteps = [i / (FM_N_STEPS - 1) for i in range(FM_N_STEPS)]
    for step in range(FM_N_STEPS - 1):
        t = timesteps[step]
        dt = timesteps[step + 1] - t
        time_emb = sinusoidal_time_embedding(t, FM_DIM).unsqueeze(0)
        time_proj = time_emb @ w["acoustic_transformer.time_projection.weight"].float().T
        x_t_proj = x_t @ w["acoustic_transformer.input_projection.weight"].float().T

        seq_cond = torch.stack([x_t_proj.squeeze(0), time_proj.squeeze(0),
                                llm_proj_cond.squeeze(0)]).unsqueeze(0)
        seq_uncond = torch.stack([x_t_proj.squeeze(0), time_proj.squeeze(0),
                                  llm_proj_uncond.squeeze(0)]).unsqueeze(0)

        # FM weights accessor - always float32
        def fw(name):
            return w[name].float()

        for cond_idx, seq in enumerate([seq_cond, seq_uncond]):
            h_fm = seq.clone()

            for li in range(FM_N_LAYERS):
                pfx = f"acoustic_transformer.layers.{li}"
                h_fm_normed = rms_norm(h_fm, fw(f"{pfx}.attention_norm.weight"))
                bsz, seqlen, _ = h_fm_normed.shape

                q = h_fm_normed @ fw(f"{pfx}.attention.wq.weight").T
                k = h_fm_normed @ fw(f"{pfx}.attention.wk.weight").T
                v = h_fm_normed @ fw(f"{pfx}.attention.wv.weight").T

                q = q.view(bsz, seqlen, FM_N_HEADS, HEAD_DIM).transpose(1, 2)
                k = k.view(bsz, seqlen, FM_N_KV_HEADS, HEAD_DIM).transpose(1, 2)
                v = v.view(bsz, seqlen, FM_N_KV_HEADS, HEAD_DIM).transpose(1, 2)

                # NOTE: FM uses bidirectional attention WITHOUT RoPE (confirmed by MLX & vLLM-Omni)
                n_rep = FM_N_HEADS // FM_N_KV_HEADS
                k = k.repeat_interleave(n_rep, dim=1)
                v = v.repeat_interleave(n_rep, dim=1)

                scale = HEAD_DIM ** -0.5
                scores = (q @ k.transpose(-2, -1)) * scale
                attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
                out = attn @ v
                out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
                attn_out = out @ fw(f"{pfx}.attention.wo.weight").T
                h_fm = h_fm + attn_out

                h_fm_normed = rms_norm(h_fm, fw(f"{pfx}.ffn_norm.weight"))
                ffn_out = swiglu(h_fm_normed, fw(f"{pfx}.feed_forward.w1.weight"),
                                 fw(f"{pfx}.feed_forward.w2.weight"),
                                 fw(f"{pfx}.feed_forward.w3.weight"))
                h_fm = h_fm + ffn_out

            h_fm = rms_norm(h_fm, fw("acoustic_transformer.norm.weight"))
            pos0 = h_fm[:, 0, :]
            velocity = pos0 @ fw("acoustic_transformer.acoustic_codebook_output.weight").T

            if cond_idx == 0:
                v_cond = velocity
            else:
                v_uncond = velocity

        v = FM_CFG_ALPHA * v_cond + (1.0 - FM_CFG_ALPHA) * v_uncond
        x_t = x_t + v * dt

    x_final = x_t.clamp(-1.0, 1.0)
    acoustic_codes = torch.round((x_final + 1.0) * 10.0).long()
    acoustic_codes = acoustic_codes.clamp(0, 20) + 2

    return semantic_logits, acoustic_codes


def sample_top_p(logits, top_p=0.9, temperature=0.8):
    """Top-p sampling (same as WebGPU implementation)."""
    logits = logits / temperature
    probs = torch.softmax(logits.float(), dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum - sorted_probs > top_p
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum()
    idx = torch.multinomial(sorted_probs, 1)
    return sorted_indices[idx].item()


# Import codec decoder from test_codec_v2
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_codec_v2 import codec_decode, load_codec_weights, vq_lookup, fsq_dequant


def main():
    text = "The quick brown fox jumps over the lazy dog."
    voice = "neutral_female"
    max_frames = 100  # limit for speed
    use_argmax = True  # greedy

    print(f"Text: '{text}'")
    print(f"Voice: {voice}")
    print(f"Max frames: {max_frames}")

    USE_BF16 = True  # use original bf16 weights

    print("\nLoading weights...")
    weights = load_file(str(WEIGHTS_PATH))
    if USE_BF16:
        w = {k: v.bfloat16() if v.dtype == torch.bfloat16 else v.float() for k, v in weights.items()}
        # Actually keep all as bf16
        w = {k: v.bfloat16() for k, v in weights.items()}
    else:
        w = {k: v.float() for k, v in weights.items()}
    print(f"  Loaded {len(w)} tensors (dtype={'bf16' if USE_BF16 else 'f32'})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Known correct tokens from WebGPU TekkenTokenizer
    # "Hello world. This is a test." = [22177, 4304, 1046, 2409, 1395, 1261, 2688, 1046]
    TEXT_TOKENS = {
        "Hello world. This is a test.": [22177, 4304, 1046, 2409, 1395, 1261, 2688, 1046],
        "The quick brown fox jumps over the lazy dog.": [1784, 7586, 22980, 94137, 72993, 2136, 1278, 42757, 10575, 1046],
    }
    text_tokens = TEXT_TOKENS.get(text)
    if text_tokens is None:
        raise ValueError(f"No pre-computed tokens for text: '{text}'. Add to TEXT_TOKENS dict.")
    print(f"  Text tokens ({len(text_tokens)}): {text_tokens}")

    # Load voice embeddings
    voice_path = MODEL_DIR / "voice_embedding" / f"{voice}.pt"
    if voice_path.exists():
        voice_data = torch.load(voice_path, map_location="cpu", weights_only=True)
        if isinstance(voice_data, dict):
            voice_emb = voice_data.get("embeddings", voice_data.get("weight", list(voice_data.values())[0]))
        else:
            voice_emb = voice_data
        if USE_BF16:
            voice_emb = voice_emb.bfloat16()
        else:
            voice_emb = voice_emb.float()
        n_voice_tokens = voice_emb.shape[0]
        print(f"  Voice embeddings: {voice_emb.shape} ({n_voice_tokens} tokens)")
    else:
        print(f"  Warning: no voice embedding at {voice_path}")
        voice_emb = None
        n_voice_tokens = 0

    # Build prompt: [BOS, BEGIN_AUDIO, AUDIO×N, TEXT_TO_AUDIO, text_tokens, AUDIO_TO_TEXT, BEGIN_AUDIO]
    USE_VOICE = True
    tokens = [BOS, BEGIN_AUDIO]
    audio_token_start = len(tokens)
    if USE_VOICE:
        for i in range(n_voice_tokens):
            tokens.append(AUDIO)
    tokens.append(TEXT_TO_AUDIO)
    tokens.extend(text_tokens)
    tokens.append(AUDIO_TO_TEXT)
    tokens.append(BEGIN_AUDIO)
    print(f"  Total prompt tokens: {len(tokens)}")
    print(f"  Voice tokens: {n_voice_tokens} at positions [{audio_token_start}..{audio_token_start + n_voice_tokens - 1}]")

    # Precompute RoPE
    max_seq = len(tokens) + max_frames + 10
    freqs_all = precompute_freqs_cis(HEAD_DIM, max_seq, ROPE_THETA)

    # Initialize KV cache
    kv_cache = BackboneKVCache()

    # Batch Prefill (matches C code's tts_llm_prefill)
    print(f"\nBatch prefilling {len(tokens)} tokens...")
    seq_len = len(tokens)
    tok_emb_table = w["mm_audio_embeddings.tok_embeddings.weight"]

    # Build input embeddings for all tokens
    embeds = torch.zeros(1, seq_len, DIM, dtype=tok_emb_table.dtype)
    for i, tok in enumerate(tokens):
        if USE_VOICE and voice_emb is not None and audio_token_start <= i < audio_token_start + n_voice_tokens:
            # Voice embedding REPLACES token embedding
            embeds[0, i] = voice_emb[i - audio_token_start]
        else:
            embeds[0, i] = tok_emb_table[tok]

    # Process through all layers
    h = embeds  # [1, seq_len, DIM]
    freqs_batch = freqs_all[:seq_len]  # [seq_len, HEAD_DIM/2]

    for layer_i in range(N_LAYERS):
        prefix = f"layers.{layer_i}"
        h_normed = rms_norm(h, w[f"{prefix}.attention_norm.weight"])

        q = h_normed @ w[f"{prefix}.attention.wq.weight"].T  # [1, seq_len, q_dim]
        k = h_normed @ w[f"{prefix}.attention.wk.weight"].T  # [1, seq_len, kv_dim]
        v = h_normed @ w[f"{prefix}.attention.wv.weight"].T  # [1, seq_len, kv_dim]

        q = q.view(1, seq_len, N_HEADS, HEAD_DIM).transpose(1, 2)  # [1, N_HEADS, seq_len, HEAD_DIM]
        k = k.view(1, seq_len, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(1, seq_len, N_KV_HEADS, HEAD_DIM).transpose(1, 2)

        # Apply RoPE — need to handle batch shape [1, n_heads, seq_len, head_dim]
        # apply_rope expects [n_heads, seq_len, head_dim] with freqs [seq_len, head_dim/2]
        # but unsqueezes freqs to [1, 1, seq_len, head_dim/2] — fix by using correct broadcasting
        q_squeezed = q.squeeze(0)  # [N_HEADS, seq_len, HEAD_DIM]
        k_squeezed = k.squeeze(0)  # [N_KV_HEADS, seq_len, HEAD_DIM]
        # Convert to complex, apply rope, convert back
        q_complex = torch.view_as_complex(q_squeezed.float().reshape(N_HEADS, seq_len, -1, 2))
        k_complex = torch.view_as_complex(k_squeezed.float().reshape(N_KV_HEADS, seq_len, -1, 2))
        freqs_b = freqs_batch.unsqueeze(0)  # [1, seq_len, HEAD_DIM/2]
        q_rotated = torch.view_as_real(q_complex * freqs_b).flatten(-2).to(q.dtype)
        k_rotated = torch.view_as_real(k_complex * freqs_b).flatten(-2).to(k.dtype)
        q = q_rotated.unsqueeze(0)  # [1, N_HEADS, seq_len, HEAD_DIM]
        k = k_rotated.unsqueeze(0)  # [1, N_KV_HEADS, seq_len, HEAD_DIM]

        # Store K, V in cache for later decode steps
        kv_cache.k_cache[layer_i] = k.squeeze(0)  # [N_KV_HEADS, seq_len, HEAD_DIM]
        kv_cache.v_cache[layer_i] = v.squeeze(0)

        # GQA: repeat K, V for grouped query attention
        n_rep = N_HEADS // N_KV_HEADS
        k_rep = k.repeat_interleave(n_rep, dim=1)  # [1, N_HEADS, seq_len, HEAD_DIM]
        v_rep = v.repeat_interleave(n_rep, dim=1)

        # Causal attention
        scale = HEAD_DIM ** -0.5
        scores = (q @ k_rep.transpose(-2, -1)) * scale  # [1, N_HEADS, seq_len, seq_len]

        # Causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = attn @ v_rep  # [1, N_HEADS, seq_len, HEAD_DIM]
        out = out.transpose(1, 2).contiguous().view(1, seq_len, -1)  # [1, seq_len, q_dim]
        attn_out = out @ w[f"{prefix}.attention.wo.weight"].T
        h = h + attn_out

        h_normed = rms_norm(h, w[f"{prefix}.ffn_norm.weight"])
        ffn_out = swiglu(h_normed, w[f"{prefix}.feed_forward.w1.weight"],
                         w[f"{prefix}.feed_forward.w2.weight"],
                         w[f"{prefix}.feed_forward.w3.weight"])
        h = h + ffn_out

        if layer_i % 5 == 0:
            print(f"  Layer {layer_i}/{N_LAYERS}")

    # Final norm — take last token's hidden state
    normed = rms_norm(h, w["norm.weight"])

    # Debug: check text logits from the backbone to verify text conditioning
    text_logit_weight = w["mm_audio_embeddings.tok_embeddings.weight"]  # tied embeddings
    for check_pos in [220, 221, 222]:
        h_at_pos = normed[:, check_pos:check_pos+1, :]
        text_logits_at_pos = h_at_pos.squeeze() @ text_logit_weight.T
        top5 = torch.topk(text_logits_at_pos, 5)
        tok_at_pos = tokens[check_pos] if check_pos < len(tokens) else "?"
        tok_label = f"(input token at pos {check_pos} = {tok_at_pos})"
        next_tok = tokens[check_pos + 1] if check_pos + 1 < len(tokens) else "?"
        print(f"  Text logits @ pos {check_pos} {tok_label}: top5={top5.indices.tolist()}, expected next={next_tok}")

    hidden = normed[:, -1:, :]  # [1, 1, DIM] — last token
    hidden = hidden.squeeze(0)  # [1, DIM]
    print(f"  Batch prefill done. Hidden[0:4] = {hidden[0, :4].tolist()}")

    # Generate
    print(f"\nGenerating frames...")
    pos = len(tokens)
    semantic_codes = []
    all_acoustic_codes = []

    # MLX reference feeds AUDIO token (24) through TEXT embedding after prefill.
    # This produces the hidden state for frame 0.
    FEED_AUDIO_TOKEN = True
    if FEED_AUDIO_TOKEN:
        print(f"  Feeding AUDIO token (24) through TEXT embedding for first decode step...")
        hidden = backbone_step(w, AUDIO, pos, kv_cache,
                               use_audio_embed=False, freqs_all=freqs_all)
        pos += 1
        print(f"  Post-AUDIO hidden[0:4] = {hidden[0, :4].tolist()}")

    # Audio codebook embedding offsets (from voxtral-tts.c reference)
    # Codebook 0 (semantic): size 8194 (8192 + 2 special), offset 0
    # Codebooks 1-36 (acoustic): size 23 (21 + 2 special) each
    AUDIO_SPECIAL_COUNT = 2  # EMPTY=0, END=1
    SEMANTIC_CB_SIZE = 8192
    FSQ_LEVELS = 21
    NUM_CODEBOOKS = 37
    cb_offsets = [0]
    cb_offsets.append(SEMANTIC_CB_SIZE + AUDIO_SPECIAL_COUNT)  # 8194
    for i in range(2, NUM_CODEBOOKS + 1):
        cb_offsets.append(cb_offsets[-1] + FSQ_LEVELS + AUDIO_SPECIAL_COUNT)  # +23 each

    audio_embed_table = w["mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight"]

    MULTI_CB_SCALE = 1.0  # Try 1.0 (sum) or 1/37 (mean) or other scales

    def embed_audio_codes(sem_code, ac_codes_list):
        """Sum all 37 codebook embeddings for the next backbone input."""
        dtype = torch.bfloat16 if USE_BF16 else torch.float32
        result = torch.zeros(1, 1, DIM, dtype=dtype)
        # Semantic codebook (cb=0)
        idx = cb_offsets[0] + sem_code
        result += audio_embed_table[idx:idx+1].unsqueeze(0)
        # Acoustic codebooks (cb=1..36)
        for cb in range(36):
            ac_code = ac_codes_list[cb]
            idx = cb_offsets[cb + 1] + ac_code
            result += audio_embed_table[idx:idx+1].unsqueeze(0)
        return result * MULTI_CB_SCALE

    torch.manual_seed(42)  # for reproducibility

    USE_MULTI_CODEBOOK = True  # sum all 37 codebook embeddings (matches MLX/vLLM-Omni)

    for frame in range(max_frames):
        if frame > 0:
            if USE_MULTI_CODEBOOK:
                h = embed_audio_codes(semantic_codes[-1], all_acoustic_codes[-1])
                if frame <= 3:
                    print(f"  [DBG frame {frame}] embed norm={h.norm().item():.4f}, sem={semantic_codes[-1]}, ac[:3]={all_acoustic_codes[-1][:3]}")
                hidden = backbone_step(w, 0, pos, kv_cache,
                                       precomputed_embed=h, freqs_all=freqs_all)
            else:
                hidden = backbone_step(w, semantic_codes[-1], pos, kv_cache,
                                       use_audio_embed=True, freqs_all=freqs_all)
            if frame <= 3:
                print(f"  [DBG frame {frame}] hidden norm={hidden.norm().item():.4f}")
            pos += 1

        sem_logits, ac_codes = fm_forward(w, hidden)

        # Semantic logit masking (from vLLM-Omni and voxtral-tts.c):
        # - EMPTY_AUDIO (index 0) → -inf (should never be sampled)
        # - Padding beyond valid semantic vocab (indices >= 8194) → -inf
        sem_logits_masked = sem_logits.clone()
        sem_logits_masked[:, 0] = float('-inf')  # mask EMPTY_AUDIO
        valid_semantic_size = SEMANTIC_CB_SIZE + AUDIO_SPECIAL_COUNT  # 8194
        if sem_logits_masked.shape[-1] > valid_semantic_size:
            sem_logits_masked[:, valid_semantic_size:] = float('-inf')

        # Debug: show top logits
        if frame < 3:
            topk = torch.topk(sem_logits_masked.squeeze(), 5)
            print(f"  [LOGITS frame {frame}] top5 ids={topk.indices.tolist()}, vals={[round(v, 2) for v in topk.values.tolist()]}")
            print(f"  [LOGITS frame {frame}] logit[0]={sem_logits.squeeze()[0].item():.2f}, logit[1]={sem_logits.squeeze()[1].item():.2f}, logit[2]={sem_logits.squeeze()[2].item():.2f}")

        # Sample
        if use_argmax:
            sem_code = sem_logits_masked.squeeze().argmax().item()
        else:
            sem_code = sample_top_p(sem_logits_masked.squeeze(), top_p=0.9, temperature=0.8)

        # EOS: semantic code 1 = END_AUDIO (not text EOS=2)
        if sem_code == 1:
            print(f"  Frame {frame}: END_AUDIO!")
            break

        semantic_codes.append(sem_code)
        all_acoustic_codes.append(ac_codes.squeeze().tolist())

        if frame % 10 == 0:
            print(f"  Frame {frame}: semantic={sem_code}")

    print(f"\nGenerated {len(semantic_codes)} frames")
    print(f"  Semantic codes: {semantic_codes[:20]}...")

    # Codec decode
    if len(semantic_codes) > 0:
        print("\nDecoding with codec...")
        codec_w = {}
        for k, v in w.items():
            if k.startswith("audio_tokenizer."):
                codec_w[k[len("audio_tokenizer."):]] = v.float()

        sem_tensor = torch.tensor(semantic_codes, dtype=torch.long)
        ac_tensor = torch.tensor(all_acoustic_codes, dtype=torch.long)
        audio, intermediates = codec_decode(sem_tensor, ac_tensor, codec_w, save_intermediates=False)

        audio_np = audio.squeeze().detach().numpy()
        print(f"  Audio: {len(audio_np)} samples = {len(audio_np)/24000:.2f}s")
        print(f"  Max amplitude: {np.abs(audio_np).max():.6f}")

        # Save
        import soundfile as sf
        sf.write(str(OUTPUT_DIR / "python_gen_audio.wav"), audio_np, 24000)
        np.save(OUTPUT_DIR / "python_gen_semantic_codes.npy",
                np.array(semantic_codes, dtype=np.float32))
        print(f"\nSaved to {OUTPUT_DIR}")
    else:
        print("No frames generated!")


if __name__ == "__main__":
    main()
