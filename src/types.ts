/**
 * Voxtral TTS configuration interfaces
 *
 * Architecture: 3 components
 * 1. Backbone transformer (Ministral 3B base) — autoregressive text→semantic tokens
 * 2. Flow-matching acoustic transformer — semantic→acoustic latents (16 NFEs)
 * 3. Neural audio codec decoder — acoustic latents→24kHz waveform
 */

export interface BackboneConfig {
  dim: number;           // 3072
  n_layers: number;      // 26
  head_dim: number;      // 128
  hidden_dim: number;    // 9216 (FFN)
  n_heads: number;       // 32
  n_kv_heads: number;    // 8 (GQA)
  vocab_size: number;    // 131072
  rope_theta: number;    // 1000000
  norm_eps: number;      // 1e-5
}

export interface FMTransformerConfig {
  input_dim: number;     // 3072
  dim: number;           // 3072
  n_layers: number;      // 3
  head_dim: number;      // 128
  hidden_dim: number;    // 9216
  n_heads: number;       // 32
  n_kv_heads: number;    // 8 (GQA)
  nfe: number;           // 8 (Euler ODE steps)
  cfg_alpha: number;     // 1.2 (classifier-free guidance)
  rope_theta: number;    // 10000
  sigma: number;         // 1e-5
  sigma_max: number;     // 1.0
  n_acoustic_out: number; // 36 (FSQ dims)
  semantic_vocab: number; // 8320 (8192 + padding)
}

export interface CodecConfig {
  dim: number;           // 1024
  hidden_dim: number;    // 4096
  head_dim: number;      // 128
  n_heads: number;       // 8
  n_kv_heads: number;    // 8
  semantic_codebook_size: number; // 8192
  semantic_dim: number;  // 256
  n_acoustic_codebook: number;   // 36
  acoustic_codebook_size: number; // 21 (FSQ levels)
  sampling_rate: number; // 24000
  frame_rate: number;    // 12.5
  patch_size: number;    // 240
  decoder_stages: number; // 4
  decoder_layers_per_stage: number; // 2
  decoder_conv_strides: number[]; // [1, 2, 2, 2]
  decoder_conv_kernels: number[]; // [3, 4, 4, 4]
  attn_sliding_window: number; // 16
  norm_eps: number;      // 0.01 (much larger than backbone's 1e-5!)
  qk_norm_eps: number;   // 1e-6
  qk_norm: boolean;      // true
  layer_scale: boolean;  // true
  weight_norm_conv: boolean; // true
}

export interface VoxtralConfig {
  backbone: BackboneConfig;
  fm: FMTransformerConfig;
  codec: CodecConfig;
}

export const defaultConfig: VoxtralConfig = {
  backbone: {
    dim: 3072,
    n_layers: 26,
    head_dim: 128,
    hidden_dim: 9216,
    n_heads: 32,
    n_kv_heads: 8,
    vocab_size: 131072,
    rope_theta: 1_000_000,
    norm_eps: 1e-5,
  },
  fm: {
    input_dim: 3072,
    dim: 3072,
    n_layers: 3,
    head_dim: 128,
    hidden_dim: 9216,
    n_heads: 32,
    n_kv_heads: 8,
    nfe: 8,
    cfg_alpha: 1.2,
    rope_theta: 10_000,
    sigma: 1e-5,
    sigma_max: 1.0,
    n_acoustic_out: 36,
    semantic_vocab: 8320,
  },
  codec: {
    dim: 1024,
    hidden_dim: 4096,
    head_dim: 128,
    n_heads: 8,
    n_kv_heads: 8,
    semantic_codebook_size: 8192,
    semantic_dim: 256,
    n_acoustic_codebook: 36,
    acoustic_codebook_size: 21,
    sampling_rate: 24000,
    frame_rate: 12.5,
    patch_size: 240,
    decoder_stages: 4,
    decoder_layers_per_stage: 2,
    decoder_conv_strides: [1, 2, 2, 2],
    decoder_conv_kernels: [3, 4, 4, 4],
    attn_sliding_window: 16,
    norm_eps: 0.01,
    qk_norm_eps: 1e-6,
    qk_norm: true,
    layer_scale: true,
    weight_norm_conv: true,
  },
};

/** Matmul test case for phase 0a go/no-go */
export interface MatmulTestCase {
  name: string;
  M: number;  // rows of A (batch/sequence)
  K: number;  // inner dimension
  N: number;  // cols of B (output)
  count: number; // how many times per token/frame
  component: 'backbone' | 'fm' | 'codec';
}
