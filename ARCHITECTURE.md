# Voxtral TTS WebGPU — Architecture Document

## Model: mistralai/Voxtral-4B-TTS-2603

- **Total params:** 4.00B (4,002,353,392)
- **Total size:** 8.00 GB (BF16) → ~4.0 GB as F16
- **Format:** Single consolidated.safetensors, BF16
- **386 tensors**

## Three Components

### 1. Backbone (Ministral 3B) — ~3.4B params

Standard Mistral transformer, autoregressive causal decoding.

| Parameter | Value |
|-----------|-------|
| dim | 3072 |
| n_layers | 26 |
| head_dim | 128 |
| hidden_dim (FFN) | 9216 |
| n_heads | 32 |
| n_kv_heads | 8 (GQA) |
| vocab_size | 131,072 |
| rope_theta | 1,000,000 |
| norm_eps | 1e-5 |
| tied_embeddings | yes (tok_embeddings = lm_head) |

**Weight naming:** `layers.{0-25}.{attention|feed_forward}.{wq|wk|wv|wo|w1|w2|w3}.weight`

Per-layer weights:
- `wq`: [4096, 3072] — 32 heads × 128 head_dim
- `wk`: [1024, 3072] — 8 KV heads × 128 head_dim
- `wv`: [1024, 3072]
- `wo`: [3072, 4096]
- `w1` (gate): [9216, 3072]
- `w2` (down): [3072, 9216]
- `w3` (up): [9216, 3072]
- `attention_norm.weight`: [3072]
- `ffn_norm.weight`: [3072]

Shared weights:
- `mm_audio_embeddings.tok_embeddings.weight`: [131072, 3072] — text embedding (= tied LM head)
- `mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight`: [9088, 3072] — audio token embeddings
- `norm.weight`: [3072] — final RMS norm

### 2. Flow Matching Acoustic Transformer — ~390M params

3-layer bidirectional transformer that converts backbone hidden state → acoustic codes via Euler ODE.

| Parameter | Value |
|-----------|-------|
| dim | 3072 |
| n_layers | 3 |
| head_dim | 128 |
| hidden_dim | 9216 |
| n_heads | 32 |
| n_kv_heads | 8 |
| NFE (ODE steps) | 8 (Euler) |
| CFG alpha | 1.2 |
| rope_theta | 10,000 |

**Weight naming:** `acoustic_transformer.layers.{0-2}.{attention|feed_forward}.{wq|wk|wv|wo|w1|w2|w3}.weight`

Additional weights:
- `acoustic_transformer.input_projection.weight`: [3072, 36] — projects x_t (acoustic noise) to dim
- `acoustic_transformer.llm_projection.weight`: [3072, 3072] — projects backbone hidden to dim
- `acoustic_transformer.time_projection.weight`: [3072, 3072] — projects time embedding to dim
- `acoustic_transformer.norm.weight`: [3072]
- `acoustic_transformer.semantic_codebook_output.weight`: [8320, 3072] — semantic logits (8192 + padding)
- `acoustic_transformer.acoustic_codebook_output.weight`: [36, 3072] — acoustic velocity output

**Forward pass per ODE step:**
1. `input_projection(x_t)` → [1, 3072]
2. `time_projection(sinusoidal_embed(t))` → [1, 3072]
3. `llm_projection(hidden)` → [1, 3072]
4. Concat → [3, 3072] (3-token sequence)
5. 3 bidirectional transformer layers (no causal mask)
6. Extract position 0 → `acoustic_codebook_output` → velocity [36]

**ODE integration (Euler):**
- x_0 ~ N(0, 1) shape [36]
- 8 steps from t=0 to t=1
- Each step: x_{t+dt} = x_t + v(x_t, t, h) * dt
- CFG: v = 1.2 * v_cond + (-0.2) * v_uncond (uncond = zero hidden state)
- Final: clamp [-1,1], quantize to [0,20], offset by 2

### 3. Neural Audio Codec Decoder — ~300M params

Causal transformer + conv decoder that converts (semantic, acoustic) codes → 24kHz audio.

| Parameter | Value |
|-----------|-------|
| dim | 1024 |
| hidden_dim | 4096 |
| head_dim | 128 |
| n_heads | 8 |
| n_kv_heads | 8 |
| semantic_codebook_size | 8192 |
| semantic_dim | 256 |
| acoustic_codebook_size | 21 (FSQ levels) |
| n_acoustic_codebook | 36 |
| sampling_rate | 24,000 Hz |
| frame_rate | 12.5 Hz |
| patch_size | 240 samples |
| attn_sliding_window | 16 |

**Decoder architecture (sequential blocks 0-7 + output_proj):**

```
Block 0: CausalConv1d(292 → 1024, k=3, s=1)     # input conv
Block 1: 2× TransformerBlock(1024)                 # attn + FFN, window=16
Block 2: CausalConvTranspose1d(1024 → 1024, k=4, s=2)  # 2x upsample
Block 3: 2× TransformerBlock(1024)                 # attn + FFN, window=8
Block 4: CausalConvTranspose1d(1024 → 1024, k=4, s=2)  # 2x upsample
Block 5: 2× TransformerBlock(1024)                 # attn + FFN, window=4
Block 6: CausalConvTranspose1d(1024 → 1024, k=4, s=2)  # 2x upsample
Block 7: 2× TransformerBlock(1024)                 # attn + FFN, window=2
Output:  CausalConv1d(1024 → 240, k=7)            # to patches
Reshape: (B, 240, T') → (B, 1, T'×240)            # unpatch to waveform
```

Total temporal upsample: 1 × 2 × 2 × 2 = 8x
Combined with patch_size=240: each frame → 1920 samples → 80ms at 24kHz

**Codec input construction:**
- Semantic codes → VQ lookup: [T, 1] → embedding_sum[8192, 256] → [T, 256]
- Acoustic codes → FSQ dequant: [T, 36] → rescale (code × 2/20 - 1) → [T, 36]
- Concatenate: [T, 292] → Conv1d input

**Special features:**
- Weight normalization on convolutions (parametrizations.weight.original0/original1)
- QK normalization (q_norm, k_norm per attention)
- Layer scale (attention_scale, ffn_scale per transformer block)
- ALiBi positional encoding (no RoPE in codec)
- Causal sliding window attention (halved at each downsample level)

## Data Flow Pipeline

```
Text → Tekken Tokenizer → token_ids
                              ↓
Voice embedding → replace [AUDIO] positions in embeddings
                              ↓
┌─────────────────────────────────────────────────────┐
│ BACKBONE (autoregressive, per token)                │
│                                                      │
│ embed → [layer 0..25: norm → GQA attn → norm → SwiGLU FFN] → final_norm → hidden │
└─────────────────┬───────────────────────────────────┘
                  ↓ hidden [3072]
┌─────────────────────────────────────────────────────┐
│ FM TRANSFORMER (per frame)                          │
│                                                      │
│ 1. semantic_logit = linear(hidden) → argmax → semantic_code │
│ 2. If semantic_code == END_AUDIO → stop              │
│ 3. x_0 ~ N(0,1) [36]                               │
│ 4. For t in linspace(0,1,8):                        │
│    - project [x_t, time(t), hidden] → [3, 3072]    │
│    - 3 bidirectional transformer layers             │
│    - v = CFG(v_cond, v_uncond, α=1.2)              │
│    - x_{t+dt} = x_t + v * dt                       │
│ 5. Quantize x_1 to [0,20] → 36 acoustic codes      │
│ 6. Output: [semantic_code, acoustic_codes] [37]     │
└─────────────────┬───────────────────────────────────┘
                  ↓ [T, 37] codes
┌─────────────────────────────────────────────────────┐
│ CODEC DECODER (per chunk, ~25 frames)               │
│                                                      │
│ semantic_codes → VQ lookup → [T, 256]               │
│ acoustic_codes → FSQ dequant → [T, 36]              │
│ concat → [T, 292]                                   │
│ → conv(292→1024) → [2×attn+conv↑2]×3 → 2×attn      │
│ → conv(1024→240) → unpatch → waveform [T×1920]      │
└─────────────────────────────────────────────────────┘
                  ↓
              24kHz audio
```

## GPU Memory Budget (F16)

| Component | Size |
|-----------|------|
| Backbone weights | ~3.4 GB |
| FM transformer weights | ~390 MB |
| Codec decoder weights | ~150 MB |
| Embedding table (131K × 3072) | ~400 MB |
| KV cache (26 layers, context) | Variable |
| **Total weights** | **~4.0 GB** |

## WebGPU Feasibility Notes

1. **4 GB weights in F16** fits in most discrete GPUs and Apple Silicon (M1+)
2. **Backbone is standard Mistral** — can reuse patterns from webgpu-qwen
3. **FM transformer is small** (3 layers) but runs 8× per frame with 2× batch for CFG
4. **Codec uses weight normalization and ALiBi** — need custom shaders
5. **BF16 → F16 conversion** needed (WebGPU has no native BF16 support)
6. **Benchmark result: 31ms/frame matmuls** (2.58x realtime headroom)
