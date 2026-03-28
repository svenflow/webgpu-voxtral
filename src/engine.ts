/**
 * Voxtral TTS WebGPU Engine
 *
 * Orchestrates the 3-component forward pass:
 * 1. Backbone (Ministral 3B) — autoregressive text → hidden states
 * 2. FM Transformer — hidden state → semantic + acoustic codes via Euler ODE
 * 3. Codec Decoder — codes → 24kHz waveform
 *
 * Phase 5: Full forward pass with GPU compute dispatch
 */

import { VoxtralConfig, defaultConfig } from './types.js';
import {
  WeightManifest,
  ComponentWeights,
  loadManifest,
  loadComponentBulk,
  WeightLoadProgress,
  loadWeightsFromHF,
  HFLoadProgress,
  HF_VOXTRAL_URL,
} from './weights.js';
import * as shaders from './shaders.js';

// ---------------------------------------------------------------------------
// Buffer interfaces (unchanged from Phase 2)
// ---------------------------------------------------------------------------

/** Per-layer backbone weights */
interface BackboneLayerBuffers {
  attn_norm: GPUBuffer;     // [3072]
  wq: GPUBuffer;            // [4096, 3072]
  wk: GPUBuffer;            // [1024, 3072]
  wv: GPUBuffer;            // [1024, 3072]
  wo: GPUBuffer;            // [3072, 4096]
  ffn_norm: GPUBuffer;      // [3072]
  w1: GPUBuffer;            // [9216, 3072] (gate)
  w2: GPUBuffer;            // [3072, 9216] (down)
  w3: GPUBuffer;            // [9216, 3072] (up)
}

/** Per-layer FM transformer weights */
interface FMLayerBuffers {
  attn_norm: GPUBuffer;
  wq: GPUBuffer;
  wk: GPUBuffer;
  wv: GPUBuffer;
  wo: GPUBuffer;
  ffn_norm: GPUBuffer;
  w1: GPUBuffer;
  w2: GPUBuffer;
  w3: GPUBuffer;
}

/** Per-block codec decoder weights */
interface CodecTransformerLayerBuffers {
  attn_norm: GPUBuffer;     // [1024]
  q_norm: GPUBuffer;        // [1024]
  k_norm: GPUBuffer;        // [1024]
  wq: GPUBuffer;            // [1024, 1024]
  wk: GPUBuffer;            // [1024, 1024]
  wv: GPUBuffer;            // [1024, 1024]
  wo: GPUBuffer;            // [1024, 1024]
  attn_scale: GPUBuffer;    // [1024] (layer scale)
  ffn_norm: GPUBuffer;      // [1024]
  w1: GPUBuffer;            // [4096, 1024]
  w2: GPUBuffer;            // [1024, 4096]
  w3: GPUBuffer;            // [4096, 1024]
  ffn_scale: GPUBuffer;     // [1024] (layer scale)
}

/** All model weights organized by component */
interface ModelBuffers {
  // Backbone
  tok_embeddings: GPUBuffer;       // [131072, 3072]
  audio_embeddings: GPUBuffer;     // [9088, 3072]
  backbone_layers: BackboneLayerBuffers[];  // 26 layers
  final_norm: GPUBuffer;           // [3072]
  // (lm_head = tok_embeddings, tied)

  // FM Transformer
  fm_input_proj: GPUBuffer;        // [3072, 36]
  fm_llm_proj: GPUBuffer;          // [3072, 3072]
  fm_time_proj: GPUBuffer;         // [3072, 3072]
  fm_layers: FMLayerBuffers[];     // 3 layers
  fm_norm: GPUBuffer;              // [3072]
  fm_semantic_out: GPUBuffer;      // [8320, 3072]
  fm_acoustic_out: GPUBuffer;      // [36, 3072]

  // Codec Decoder
  codec_input_conv_w: GPUBuffer;      // [1024, 292, 3] (weight-normed)
  codec_input_conv_g: GPUBuffer;      // [1024, 1, 1] (weight norm scale)
  codec_stages: {
    transformer_layers: CodecTransformerLayerBuffers[];  // 2 per stage
    conv_w?: GPUBuffer;    // conv transpose weight (weight-normed)
    conv_g?: GPUBuffer;    // conv transpose gain g[c_in]
    conv_scale?: GPUBuffer; // precomputed scale[ci] = g[ci] / ||v[ci,:,:]|| (f32)
  }[];                              // 4 stages
  codec_output_conv_w: GPUBuffer;     // [240, 1024, 7]
  codec_output_conv_g: GPUBuffer;     // [240, 1, 1]
  codec_semantic_codebook: GPUBuffer; // [8192, 256]
  codec_cluster_usage: GPUBuffer; // [8192]
}

/** Working buffers for intermediate activations */
interface WorkBuffers {
  hidden: GPUBuffer;        // [3072] current hidden state (f32)
  residual: GPUBuffer;      // [3072] residual connection (f32)
  normed: GPUBuffer;        // [3072] after RMS norm (f32)

  // Attention
  q: GPUBuffer;             // [4096] = 32 heads x 128
  k: GPUBuffer;             // [1024] = 8 KV heads x 128
  v: GPUBuffer;             // [1024]
  attn_out: GPUBuffer;      // [4096]
  scores: GPUBuffer;        // [32 * max_seq] attention scores

  // FFN
  gate: GPUBuffer;          // [9216]
  up: GPUBuffer;            // [9216]
  down: GPUBuffer;          // [3072]

  // FM specific
  x_t: GPUBuffer;           // [36] acoustic noise state (f32)
  velocity: GPUBuffer;      // [36] predicted velocity (f32)
  v_uncond: GPUBuffer;      // [36] unconditional velocity for CFG (f32)
  time_embed: GPUBuffer;    // [3072] sinusoidal time embedding (f32)
  time_proj: GPUBuffer;     // [3072] projected time embedding (f32)
  x_t_proj: GPUBuffer;      // [3072] projected x_t for FM sequence (f32)
  fm_hidden: GPUBuffer;     // [3072] FM working hidden (f32)
  fm_residual: GPUBuffer;   // [3072] FM residual (f32)
  fm_normed: GPUBuffer;     // [3072] FM normed (f32)
  fm_q: GPUBuffer;          // [3 * n_heads * head_dim] FM query (f32)
  fm_k: GPUBuffer;          // [3 * n_kv_heads * head_dim] FM key (f32)
  fm_v: GPUBuffer;          // [3 * n_kv_heads * head_dim] FM value (f32)
  fm_attn_out: GPUBuffer;   // [3 * n_heads * head_dim] FM attn output (f32)
  fm_scores: GPUBuffer;     // [n_heads * 3 * 3] FM attention scores (f32)
  fm_seq: GPUBuffer;        // [3, 3072] concatenated FM input (f32)
  fm_gate: GPUBuffer;       // [3 * 9216] FM gate (f32)
  fm_up: GPUBuffer;         // [3 * 9216] FM up (f32)
  fm_down: GPUBuffer;       // [3 * 3072] FM down (f32)
  semantic_logits: GPUBuffer; // [8320] semantic codebook logits (f32)
  acoustic_out: GPUBuffer;  // [36] acoustic output from FM (f32)
  acoustic_codes: GPUBuffer; // [36] quantized acoustic codes (u32)

  // Semantic
  semantic_argmax: GPUBuffer; // [1] argmax of semantic_logits (u32)

  // Output
  logits: GPUBuffer;        // [131072] for LM head (f32)
  argmax_result: GPUBuffer; // [1] argmax index (u32)
}

/** KV cache for backbone attention */
interface KVCache {
  k: GPUBuffer;   // [max_seq, 1024] f32
  v: GPUBuffer;   // [max_seq, 1024] f32
}

/** All GPU compute pipelines */
interface Pipelines {
  matvecF16: GPUComputePipeline;
  matvecF16Chunked: GPUComputePipeline;
  matvecF16Offset: GPUComputePipeline;
  rmsNorm: GPUComputePipeline;
  rmsNormOffset: GPUComputePipeline;
  embeddingLookup: GPUComputePipeline;
  rope: GPUComputePipeline;
  ropeOffset: GPUComputePipeline;
  attnScore: GPUComputePipeline;
  softmax: GPUComputePipeline;
  attnValue: GPUComputePipeline;
  kvCacheWrite: GPUComputePipeline;
  swiGLU: GPUComputePipeline;
  addVectors: GPUComputePipeline;
  addVectorsOffset: GPUComputePipeline;
  addInPlace: GPUComputePipeline;
  addInPlaceOffset: GPUComputePipeline;
  copyBuffer: GPUComputePipeline;
  copyBufferOffset: GPUComputePipeline;
  timeEmbedding: GPUComputePipeline;
  eulerStep: GPUComputePipeline;
  cfgCombine: GPUComputePipeline;
  fsqQuantize: GPUComputePipeline;
  biAttnScore: GPUComputePipeline;
  biSoftmax: GPUComputePipeline;
  biAttnValue: GPUComputePipeline;
  swiGLUOffset: GPUComputePipeline;
  zeroFill: GPUComputePipeline;
  multiCodebookEmbed: GPUComputePipeline;
  // Codec
  vqLookup: GPUComputePipeline;
  fsqDequant: GPUComputePipeline;
  causalConv1d: GPUComputePipeline;
  causalConvTranspose1d: GPUComputePipeline;
  convTransposeNormScale: GPUComputePipeline;
  layerScale: GPUComputePipeline;
  alibiAttnScore: GPUComputePipeline;
  codecSoftmax: GPUComputePipeline;
  codecAttnValue: GPUComputePipeline;
  batchedMatvecF16: GPUComputePipeline;
  batchedRmsNorm: GPUComputePipeline;
  batchedSwiGLU: GPUComputePipeline;
  batchedAdd: GPUComputePipeline;
  batchedCopy: GPUComputePipeline;
  batchedLayerScale: GPUComputePipeline;
  qkNorm: GPUComputePipeline;
  concatCodecInput: GPUComputePipeline;
  argmax: GPUComputePipeline;
  normalizeCodebook: GPUComputePipeline;
}

export interface EngineOptions {
  config?: VoxtralConfig;
  maxSeqLen?: number;
  weightsUrl?: string;
  onProgress?: (msg: string) => void;
  onWeightProgress?: (progress: WeightLoadProgress) => void;
}

/** Result from a single TTS generation */
export interface TTSResult {
  /** Semantic token IDs generated by the backbone */
  semanticCodes: number[];
  /** Acoustic codes (36 per frame) generated by FM */
  acousticCodes: number[][];
  /** Raw audio samples at 24kHz */
  audio: Float32Array;
  /** Generation stats */
  stats: {
    backboneMs: number;
    fmMs: number;
    codecMs: number;
    totalMs: number;
    framesGenerated: number;
  };
}

// ---------------------------------------------------------------------------
// Helper: ceil division for workgroup dispatch
// ---------------------------------------------------------------------------
function cdiv(n: number, d: number): number {
  return Math.ceil(n / d);
}

/** Temperature + top-p (nucleus) sampling over logits. */
function sampleTopP(logits: Float32Array, topP: number, temperature: number): number {
  const n = logits.length;

  // Apply temperature
  const scaled = new Float32Array(n);
  for (let i = 0; i < n; i++) scaled[i] = logits[i] / temperature;

  // Softmax
  let maxV = -Infinity;
  for (let i = 0; i < n; i++) if (scaled[i] > maxV) maxV = scaled[i];
  let sumExp = 0;
  for (let i = 0; i < n; i++) {
    scaled[i] = Math.exp(scaled[i] - maxV);
    sumExp += scaled[i];
  }
  for (let i = 0; i < n; i++) scaled[i] /= sumExp;

  // Sort indices by descending probability
  const indices = Array.from({ length: n }, (_, i) => i);
  indices.sort((a, b) => scaled[b] - scaled[a]);

  // Accumulate top-p
  let cumSum = 0;
  let cutoff = n;
  for (let i = 0; i < n; i++) {
    cumSum += scaled[indices[i]];
    if (cumSum >= topP) {
      cutoff = i + 1;
      break;
    }
  }

  // Renormalize and sample
  let reSum = 0;
  for (let i = 0; i < cutoff; i++) reSum += scaled[indices[i]];
  const r = Math.random() * reSum;
  let acc = 0;
  for (let i = 0; i < cutoff; i++) {
    acc += scaled[indices[i]];
    if (acc >= r) return indices[i];
  }
  return indices[0];
}

export class VoxtralEngine {
  private device: GPUDevice | null = null;
  private config: VoxtralConfig;
  private maxSeqLen: number;
  private modelBuffers: ModelBuffers | null = null;
  private workBuffers: WorkBuffers | null = null;
  private pipelines: Pipelines | null = null;
  private kvCaches: KVCache[] = [];
  private position: number = 0;

  constructor(options: EngineOptions = {}) {
    this.config = options.config || defaultConfig;
    this.maxSeqLen = options.maxSeqLen || 4096;
  }

  // =========================================================================
  // Initialization
  // =========================================================================

  /**
   * Initialize WebGPU device, create work buffers and compile pipelines.
   */
  async init(): Promise<void> {
    if (!navigator.gpu) {
      throw new Error('WebGPU not supported');
    }
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
    });
    if (!adapter) throw new Error('No WebGPU adapter');

    const features: GPUFeatureName[] = [];
    if (adapter.features.has('shader-f16')) {
      features.push('shader-f16' as GPUFeatureName);
    }

    // Try 2GB limits first (desktop), fall back to adapter defaults (mobile)
    const desiredMaxBuffer = 2 * 1024 * 1024 * 1024; // 2GB
    const adapterMaxBuffer = adapter.limits.maxBufferSize;
    const adapterMaxStorage = adapter.limits.maxStorageBufferBindingSize;
    const useDesired = adapterMaxBuffer >= desiredMaxBuffer && adapterMaxStorage >= desiredMaxBuffer;

    this.device = await adapter.requestDevice({
      requiredFeatures: features,
      requiredLimits: {
        maxBufferSize: useDesired ? desiredMaxBuffer : adapterMaxBuffer,
        maxStorageBufferBindingSize: useDesired ? desiredMaxBuffer : adapterMaxStorage,
      },
    });

    this.createWorkBuffers();
    this.createKVCaches();
    this.createPipelines();
  }

  // =========================================================================
  // Pipeline creation
  // =========================================================================

  private createPipeline(code: string, label: string): GPUComputePipeline {
    const d = this.device!;
    const module = d.createShaderModule({ code, label });
    return d.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
      label,
    });
  }

  private createPipelines(): void {
    const p = (code: string, label: string) => this.createPipeline(code, label);
    this.pipelines = {
      // Core
      matvecF16: p(shaders.matvecF16, 'matvecF16'),
      matvecF16Chunked: p(shaders.matvecF16Chunked, 'matvecF16Chunked'),
      matvecF16Offset: p(shaders.matvecF16Offset, 'matvecF16Offset'),
      rmsNorm: p(shaders.rmsNorm, 'rmsNorm'),
      rmsNormOffset: p(shaders.rmsNormOffset, 'rmsNormOffset'),
      embeddingLookup: p(shaders.embeddingLookup, 'embeddingLookup'),
      rope: p(shaders.rope, 'rope'),
      ropeOffset: p(shaders.ropeOffset, 'ropeOffset'),
      attnScore: p(shaders.attnScore, 'attnScore'),
      softmax: p(shaders.softmax, 'softmax'),
      attnValue: p(shaders.attnValue, 'attnValue'),
      kvCacheWrite: p(shaders.kvCacheWrite, 'kvCacheWrite'),
      swiGLU: p(shaders.swiGLU, 'swiGLU'),
      addVectors: p(shaders.addVectors, 'addVectors'),
      addVectorsOffset: p(shaders.addVectorsOffset, 'addVectorsOffset'),
      addInPlace: p(shaders.addInPlace, 'addInPlace'),
      addInPlaceOffset: p(shaders.addInPlaceOffset, 'addInPlaceOffset'),
      copyBuffer: p(shaders.copyBuffer, 'copyBuffer'),
      copyBufferOffset: p(shaders.copyBufferOffset, 'copyBufferOffset'),
      // FM
      timeEmbedding: p(shaders.timeEmbedding, 'timeEmbedding'),
      eulerStep: p(shaders.eulerStep, 'eulerStep'),
      cfgCombine: p(shaders.cfgCombine, 'cfgCombine'),
      fsqQuantize: p(shaders.fsqQuantize, 'fsqQuantize'),
      biAttnScore: p(shaders.biAttnScore, 'biAttnScore'),
      biSoftmax: p(shaders.biSoftmax, 'biSoftmax'),
      biAttnValue: p(shaders.biAttnValue, 'biAttnValue'),
      swiGLUOffset: p(shaders.swiGLUOffset, 'swiGLUOffset'),
      zeroFill: p(shaders.zeroFill, 'zeroFill'),
      multiCodebookEmbed: p(shaders.multiCodebookEmbed, 'multiCodebookEmbed'),
      // Codec
      vqLookup: p(shaders.vqLookup, 'vqLookup'),
      fsqDequant: p(shaders.fsqDequant, 'fsqDequant'),
      causalConv1d: p(shaders.causalConv1d, 'causalConv1d'),
      causalConvTranspose1d: p(shaders.causalConvTranspose1d, 'causalConvTranspose1d'),
      convTransposeNormScale: p(shaders.convTransposeNormScale, 'convTransposeNormScale'),
      layerScale: p(shaders.layerScale, 'layerScale'),
      alibiAttnScore: p(shaders.alibiAttnScore, 'alibiAttnScore'),
      codecSoftmax: p(shaders.codecSoftmax, 'codecSoftmax'),
      codecAttnValue: p(shaders.codecAttnValue, 'codecAttnValue'),
      batchedMatvecF16: p(shaders.batchedMatvecF16, 'batchedMatvecF16'),
      batchedRmsNorm: p(shaders.batchedRmsNorm, 'batchedRmsNorm'),
      batchedSwiGLU: p(shaders.batchedSwiGLU, 'batchedSwiGLU'),
      batchedAdd: p(shaders.batchedAdd, 'batchedAdd'),
      batchedCopy: p(shaders.batchedCopy, 'batchedCopy'),
      batchedLayerScale: p(shaders.batchedLayerScale, 'batchedLayerScale'),
      qkNorm: p(shaders.qkNorm, 'qkNorm'),
      concatCodecInput: p(shaders.concatCodecInput, 'concatCodecInput'),
      argmax: p(shaders.argmax, 'argmax'),
      normalizeCodebook: p(shaders.normalizeCodebook, 'normalizeCodebook'),
    };
  }

  // =========================================================================
  // Uniform buffer helper
  // =========================================================================

  /** Create a uniform buffer from a Uint32Array / Float32Array of params. */
  private createUniform(data: ArrayBuffer): GPUBuffer {
    const d = this.device!;
    // WebGPU uniform buffers must be 16-byte aligned
    const alignedSize = Math.ceil(data.byteLength / 16) * 16;
    const buf = d.createBuffer({
      size: alignedSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint8Array(buf.getMappedRange()).set(new Uint8Array(data));
    buf.unmap();
    return buf;
  }

  /** Pack u32/f32 values into a uniform buffer. Values are written sequentially. */
  private packUniform(values: Array<{ u?: number; f?: number }>): GPUBuffer {
    const buf = new ArrayBuffer(values.length * 4);
    const u32 = new Uint32Array(buf);
    const f32 = new Float32Array(buf);
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (v.u !== undefined) u32[i] = v.u;
      else if (v.f !== undefined) f32[i] = v.f;
    }
    return this.createUniform(buf);
  }

  // =========================================================================
  // Work buffer creation
  // =========================================================================

  private createWorkBuffers(): void {
    const d = this.device!;
    const bb = this.config.backbone;
    const fm = this.config.fm;
    const STO = GPUBufferUsage.STORAGE;
    const SRC = GPUBufferUsage.COPY_SRC;
    const DST = GPUBufferUsage.COPY_DST;

    const buf = (size: number, label: string, extra = 0) =>
      d.createBuffer({ size, usage: STO | SRC | DST | extra, label });

    this.workBuffers = {
      // Backbone activations (f32)
      hidden: buf(bb.dim * 4, 'hidden'),
      residual: buf(bb.dim * 4, 'residual'),
      normed: buf(bb.dim * 4, 'normed'),

      q: buf(bb.n_heads * bb.head_dim * 4, 'q'),
      k: buf(bb.n_kv_heads * bb.head_dim * 4, 'k'),
      v: buf(bb.n_kv_heads * bb.head_dim * 4, 'v'),
      attn_out: buf(bb.n_heads * bb.head_dim * 4, 'attn_out'),
      scores: buf(bb.n_heads * this.maxSeqLen * 4, 'scores'),

      gate: buf(bb.hidden_dim * 4, 'gate'),
      up: buf(bb.hidden_dim * 4, 'up'),
      down: buf(bb.dim * 4, 'down'),

      // FM specific (f32)
      x_t: buf(fm.n_acoustic_out * 4, 'x_t'),
      velocity: buf(fm.n_acoustic_out * 4, 'velocity'),
      v_uncond: buf(fm.n_acoustic_out * 4, 'v_uncond'),
      time_embed: buf(fm.dim * 4, 'time_embed'),
      time_proj: buf(fm.dim * 4, 'time_proj'),
      x_t_proj: buf(fm.dim * 4, 'x_t_proj'),
      fm_hidden: buf(fm.dim * 4, 'fm_hidden'),
      fm_residual: buf(fm.dim * 4, 'fm_residual'),
      fm_normed: buf(fm.dim * 4, 'fm_normed'),
      fm_q: buf(3 * fm.n_heads * fm.head_dim * 4, 'fm_q'),
      fm_k: buf(3 * fm.n_kv_heads * fm.head_dim * 4, 'fm_k'),
      fm_v: buf(3 * fm.n_kv_heads * fm.head_dim * 4, 'fm_v'),
      fm_attn_out: buf(3 * fm.n_heads * fm.head_dim * 4, 'fm_attn_out'),
      fm_scores: buf(fm.n_heads * 3 * 3 * 4, 'fm_scores'),
      fm_seq: buf(3 * fm.dim * 4, 'fm_seq'),
      fm_gate: buf(3 * fm.hidden_dim * 4, 'fm_gate'),
      fm_up: buf(3 * fm.hidden_dim * 4, 'fm_up'),
      fm_down: buf(3 * fm.dim * 4, 'fm_down'),
      semantic_logits: buf(fm.semantic_vocab * 4, 'semantic_logits'),
      semantic_argmax: buf(4, 'semantic_argmax'),
      acoustic_out: buf(fm.n_acoustic_out * 4, 'acoustic_out'),
      acoustic_codes: buf(fm.n_acoustic_out * 4, 'acoustic_codes'),

      // Output
      logits: buf(bb.vocab_size * 4, 'logits'),
      argmax_result: buf(4, 'argmax_result'),
    };
  }

  private createKVCaches(): void {
    const d = this.device!;
    const cfg = this.config.backbone;
    const kvDim = cfg.n_kv_heads * cfg.head_dim; // 1024

    this.kvCaches = [];
    for (let i = 0; i < cfg.n_layers; i++) {
      this.kvCaches.push({
        k: d.createBuffer({
          size: this.maxSeqLen * kvDim * 4,  // f32
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          label: `kv_cache.${i}.k`,
        }),
        v: d.createBuffer({
          size: this.maxSeqLen * kvDim * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          label: `kv_cache.${i}.v`,
        }),
      });
    }
  }

  // =========================================================================
  // Weight loading (unchanged)
  // =========================================================================

  /**
   * Load model weights from pre-extracted F16 binary files (local/server).
   */
  async loadWeights(
    baseUrl: string,
    onProgress?: (progress: WeightLoadProgress) => void,
  ): Promise<void> {
    const d = this.device!;
    const manifest = await loadManifest(baseUrl);

    const log = onProgress || (() => {});

    log({ loaded: 0, total: 3, component: 'all', tensor: 'Loading backbone...' });
    const backbone = await loadComponentBulk(d, baseUrl, manifest, 'backbone', onProgress);

    log({ loaded: 1, total: 3, component: 'all', tensor: 'Loading FM transformer...' });
    const fm = await loadComponentBulk(d, baseUrl, manifest, 'fm', onProgress);

    log({ loaded: 2, total: 3, component: 'all', tensor: 'Loading codec decoder...' });
    const codec = await loadComponentBulk(d, baseUrl, manifest, 'codec', onProgress);

    this.modelBuffers = this.organizeWeights(backbone, fm, codec);

    log({ loaded: 3, total: 3, component: 'all', tensor: 'Done!' });
  }

  /**
   * Load model weights by streaming from HuggingFace CDN.
   *
   * Fetches BF16 tensors via HTTP range requests from the safetensors file,
   * converts to F16 in-browser, and caches in IndexedDB for instant reload.
   * Never holds more than one tensor in RAM at a time.
   */
  async loadWeightsFromHF(
    safetensorsUrl: string = HF_VOXTRAL_URL,
    onProgress?: (progress: HFLoadProgress) => void,
  ): Promise<void> {
    const d = this.device!;

    const { backbone, fm, codec } = await loadWeightsFromHF(
      d, safetensorsUrl, onProgress,
    );

    this.modelBuffers = this.organizeWeights(backbone, fm, codec);

    // Normalize VQ codebook: embedding_sum / max(cluster_usage, eps)
    await this.normalizeVQCodebook();

    // Precompute per-c_in weight norm scales for ConvTranspose1d layers
    await this.precomputeConvTransposeScales();
  }

  /**
   * Normalize the VQ semantic codebook in-place on the GPU.
   * The raw weights store embedding_sum and cluster_usage separately;
   * the actual codebook is embedding_sum / max(cluster_usage, eps).
   */
  private async normalizeVQCodebook(): Promise<void> {
    const d = this.device!;
    const P = this.pipelines!;
    const M = this.modelBuffers!;
    const codec = this.config.codec;

    const params = this.packUniform([
      { u: codec.semantic_codebook_size },  // 8192
      { u: codec.semantic_dim },            // 256
      { f: 1e-5 },                          // epsilon
    ]);

    const encoder = d.createCommandEncoder({ label: 'normalize_codebook' });
    const pass = encoder.beginComputePass({ label: 'normalize_codebook' });
    this.dispatch(pass, P.normalizeCodebook,
      [M.codec_semantic_codebook, M.codec_cluster_usage, params],
      [cdiv(codec.semantic_codebook_size * codec.semantic_dim / 2, 128)]);
    pass.end();
    d.queue.submit([encoder.finish()]);
    await d.queue.onSubmittedWorkDone();
  }

  /**
   * Precompute per-c_in weight norm scales for ConvTranspose1d layers.
   * For weight [c_in, c_out, kernel] with gain g[c_in]:
   *   scale[ci] = g[ci] / ||v[ci, :, :]||
   * This is needed because weight_norm dim=0 normalizes per first dim (c_in for ConvTranspose1d).
   */
  private async precomputeConvTransposeScales(): Promise<void> {
    const d = this.device!;
    const P = this.pipelines!;
    const M = this.modelBuffers!;
    const codec = this.config.codec;

    const encoder = d.createCommandEncoder({ label: 'precompute_conv_transpose_scales' });

    for (let stage = 0; stage < codec.decoder_stages; stage++) {
      const stageData = M.codec_stages[stage];
      if (!stageData.conv_w || !stageData.conv_g) continue;

      // Allocate scale buffer: [c_in] as f32
      const scaleBuf = d.createBuffer({
        size: codec.dim * 4,  // c_in = 1024, 4 bytes per f32
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        label: `codec_conv_transpose_scale_s${stage}`,
      });
      stageData.conv_scale = scaleBuf;

      const kernel = stage === 0 ? 4 : 4;  // all transpose convs have kernel=4
      const params = this.packUniform([
        { u: codec.dim },     // c_in = 1024
        { u: codec.dim },     // c_out = 1024
        { u: kernel },        // kernel
      ]);

      const pass = encoder.beginComputePass({ label: `conv_transpose_norm_scale_s${stage}` });
      this.dispatch(pass, P.convTransposeNormScale,
        [stageData.conv_w, stageData.conv_g, scaleBuf, params],
        [codec.dim]);  // one workgroup per c_in
      pass.end();
    }

    d.queue.submit([encoder.finish()]);
    await d.queue.onSubmittedWorkDone();
  }

  private organizeWeights(
    backbone: ComponentWeights,
    fm: ComponentWeights,
    codec: ComponentWeights,
  ): ModelBuffers {
    const get = (weights: ComponentWeights, name: string): GPUBuffer => {
      const buf = weights.buffers.get(name);
      if (!buf) throw new Error(`Missing weight: ${name}`);
      return buf;
    };

    const backboneLayers: BackboneLayerBuffers[] = [];
    for (let i = 0; i < this.config.backbone.n_layers; i++) {
      backboneLayers.push({
        attn_norm: get(backbone, `layers.${i}.attention_norm.weight`),
        wq: get(backbone, `layers.${i}.attention.wq.weight`),
        wk: get(backbone, `layers.${i}.attention.wk.weight`),
        wv: get(backbone, `layers.${i}.attention.wv.weight`),
        wo: get(backbone, `layers.${i}.attention.wo.weight`),
        ffn_norm: get(backbone, `layers.${i}.ffn_norm.weight`),
        w1: get(backbone, `layers.${i}.feed_forward.w1.weight`),
        w2: get(backbone, `layers.${i}.feed_forward.w2.weight`),
        w3: get(backbone, `layers.${i}.feed_forward.w3.weight`),
      });
    }

    const fmLayers: FMLayerBuffers[] = [];
    for (let i = 0; i < this.config.fm.n_layers; i++) {
      fmLayers.push({
        attn_norm: get(fm, `acoustic_transformer.layers.${i}.attention_norm.weight`),
        wq: get(fm, `acoustic_transformer.layers.${i}.attention.wq.weight`),
        wk: get(fm, `acoustic_transformer.layers.${i}.attention.wk.weight`),
        wv: get(fm, `acoustic_transformer.layers.${i}.attention.wv.weight`),
        wo: get(fm, `acoustic_transformer.layers.${i}.attention.wo.weight`),
        ffn_norm: get(fm, `acoustic_transformer.layers.${i}.ffn_norm.weight`),
        w1: get(fm, `acoustic_transformer.layers.${i}.feed_forward.w1.weight`),
        w2: get(fm, `acoustic_transformer.layers.${i}.feed_forward.w2.weight`),
        w3: get(fm, `acoustic_transformer.layers.${i}.feed_forward.w3.weight`),
      });
    }

    // Decoder block layout: block0=input_conv, block1=transformer, block2=conv_up,
    // block3=transformer, block4=conv_up, block5=transformer, block6=conv_up, block7=transformer
    // Each stage pairs a transformer with the conv_up that FOLLOWS it (except last stage).
    // Stage 0: transformer(block1) → conv(block2)
    // Stage 1: transformer(block3) → conv(block4)
    // Stage 2: transformer(block5) → conv(block6)
    // Stage 3: transformer(block7) → no conv
    const codecStages: ModelBuffers['codec_stages'] = [];
    for (let s = 0; s < 4; s++) {
      const tBlock = 1 + s * 2;  // 1, 3, 5, 7
      const cBlock = 2 + s * 2;  // 2, 4, 6, 8 (8 doesn't exist → no conv for stage 3)
      const hasConv = s < 3;     // stages 0-2 have conv_up, stage 3 does not
      codecStages.push({
        transformer_layers: this.getCodecTransformerLayers(codec, tBlock),
        ...(hasConv ? {
          conv_w: get(codec, `audio_tokenizer.decoder_blocks.${cBlock}.conv.parametrizations.weight.original1`),
          conv_g: get(codec, `audio_tokenizer.decoder_blocks.${cBlock}.conv.parametrizations.weight.original0`),
        } : {}),
      });
    }

    return {
      tok_embeddings: get(backbone, 'mm_audio_embeddings.tok_embeddings.weight'),
      audio_embeddings: get(backbone, 'mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight'),
      backbone_layers: backboneLayers,
      final_norm: get(backbone, 'norm.weight'),

      fm_input_proj: get(fm, 'acoustic_transformer.input_projection.weight'),
      fm_llm_proj: get(fm, 'acoustic_transformer.llm_projection.weight'),
      fm_time_proj: get(fm, 'acoustic_transformer.time_projection.weight'),
      fm_layers: fmLayers,
      fm_norm: get(fm, 'acoustic_transformer.norm.weight'),
      fm_semantic_out: get(fm, 'acoustic_transformer.semantic_codebook_output.weight'),
      fm_acoustic_out: get(fm, 'acoustic_transformer.acoustic_codebook_output.weight'),

      codec_input_conv_w: get(codec, 'audio_tokenizer.decoder_blocks.0.conv.parametrizations.weight.original1'),
      codec_input_conv_g: get(codec, 'audio_tokenizer.decoder_blocks.0.conv.parametrizations.weight.original0'),
      codec_stages: codecStages,
      codec_output_conv_w: get(codec, 'audio_tokenizer.output_proj.conv.parametrizations.weight.original1'),
      codec_output_conv_g: get(codec, 'audio_tokenizer.output_proj.conv.parametrizations.weight.original0'),
      codec_semantic_codebook: get(codec, 'audio_tokenizer.quantizer.semantic_codebook.embedding_sum'),
      codec_cluster_usage: get(codec, 'audio_tokenizer.quantizer.semantic_codebook.cluster_usage'),
    };
  }

  private getCodecTransformerLayers(
    codec: ComponentWeights,
    blockIdx: number,
  ): CodecTransformerLayerBuffers[] {
    const get = (name: string) => {
      const buf = codec.buffers.get(name);
      if (!buf) throw new Error(`Missing codec weight: ${name}`);
      return buf;
    };

    const layers: CodecTransformerLayerBuffers[] = [];
    for (let l = 0; l < 2; l++) {
      const prefix = `audio_tokenizer.decoder_blocks.${blockIdx}.layers.${l}`;
      layers.push({
        attn_norm: get(`${prefix}.attention_norm.weight`),
        q_norm: get(`${prefix}.attention.q_norm.weight`),
        k_norm: get(`${prefix}.attention.k_norm.weight`),
        wq: get(`${prefix}.attention.wq.weight`),
        wk: get(`${prefix}.attention.wk.weight`),
        wv: get(`${prefix}.attention.wv.weight`),
        wo: get(`${prefix}.attention.wo.weight`),
        attn_scale: get(`${prefix}.attention_scale`),
        ffn_norm: get(`${prefix}.ffn_norm.weight`),
        w1: get(`${prefix}.feed_forward.w1.weight`),
        w2: get(`${prefix}.feed_forward.w2.weight`),
        w3: get(`${prefix}.feed_forward.w3.weight`),
        ffn_scale: get(`${prefix}.ffn_scale`),
      });
    }
    return layers;
  }

  // =========================================================================
  // Dispatch helpers — record compute passes into a command encoder
  // =========================================================================

  /** Dispatch a compute shader with bindings and workgroup count. */
  private dispatch(
    pass: GPUComputePassEncoder,
    pipeline: GPUComputePipeline,
    bindings: GPUBuffer[],
    workgroups: [number, number?, number?],
  ): void {
    const entries: GPUBindGroupEntry[] = bindings.map((buffer, i) => ({
      binding: i,
      resource: { buffer },
    }));
    const bindGroup = this.device!.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries,
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(...workgroups);
  }

  // =========================================================================
  // BACKBONE: Single autoregressive token step
  // =========================================================================

  /**
   * Run one backbone token step: embed → 26 layers → norm → logits.
   * Uses KV cache for O(1) per-token cost at each layer.
   *
   * @param useAudioEmbedding - If true, look up from audio_embeddings instead of tok_embeddings.
   *                            Used for semantic code tokens during audio generation.
   * @param precomputedEmbedding - If provided, copy this buffer to hidden instead of embedding lookup.
   *                               Used for voice embeddings during prefill.
   */
  private backboneStep(
    encoder: GPUCommandEncoder,
    tokenId: number,
    useAudioEmbedding: boolean = false,
    precomputedEmbedding?: GPUBuffer,
  ): void {
    const P = this.pipelines!;
    const W = this.workBuffers!;
    const M = this.modelBuffers!;
    const bb = this.config.backbone;
    const pos = this.position;

    // Each major operation gets its own compute pass so that storage buffer
    // writes from one dispatch are visible to the next. WebGPU does NOT
    // guarantee memory visibility between dispatches within a single pass.
    let pass: GPUComputePassEncoder;

    // --- Embedding lookup ---
    pass = encoder.beginComputePass({ label: `embed_pos${pos}` });
    const embedParams = this.packUniform([
      { u: tokenId },
      { u: bb.dim },
    ]);
    const embedTable = useAudioEmbedding ? M.audio_embeddings : M.tok_embeddings;
    this.dispatch(pass, P.embeddingLookup,
      [embedTable, W.hidden, embedParams],
      [cdiv(bb.dim, 256)]);
    pass.end();

    // Voice embedding: REPLACE token embedding
    // input_embedding_concat_type "sum" refers to multi-codebook summing for autoregressive
    // audio tokens, NOT voice embedding handling. Voice .pt files contain pre-computed
    // embeddings that directly replace [AUDIO] token positions (confirmed by voxtral-tts.c).
    if (precomputedEmbedding) {
      pass = encoder.beginComputePass({ label: `voice_embed_pos${pos}` });
      const copyP = this.packUniform([{ u: bb.dim }]);
      this.dispatch(pass, P.copyBuffer,
        [precomputedEmbedding, W.hidden, copyP],
        [cdiv(bb.dim, 256)]);
      pass.end();
    }

    // --- 26 transformer layers ---
    // Each layer gets its own compute pass pair (attention + FFN) to ensure
    // storage buffer writes are visible to subsequent reads.
    for (let layer = 0; layer < bb.n_layers; layer++) {
      const L = M.backbone_layers[layer];
      const kv = this.kvCaches[layer];

      // --- Attention half ---
      pass = encoder.beginComputePass({ label: `layer${layer}_attn` });

      // 1. Save residual: residual = hidden
      const copyParams = this.packUniform([{ u: bb.dim }]);
      this.dispatch(pass, P.copyBuffer,
        [W.hidden, W.residual, copyParams],
        [cdiv(bb.dim, 256)]);

      // 2. RMS norm (attention)
      const normParams = this.packUniform([{ u: bb.dim }, { f: bb.norm_eps }]);
      this.dispatch(pass, P.rmsNorm,
        [W.hidden, L.attn_norm, W.normed, normParams],
        [1]);
      pass.end();

      pass = encoder.beginComputePass({ label: `layer${layer}_qkv` });

      // 3. Q/K/V projections
      const qParams = this.packUniform([{ u: bb.n_heads * bb.head_dim }, { u: bb.dim }]);
      this.dispatch(pass, P.matvecF16, [L.wq, W.normed, W.q, qParams],
        [bb.n_heads * bb.head_dim]);

      const kParams = this.packUniform([{ u: bb.n_kv_heads * bb.head_dim }, { u: bb.dim }]);
      this.dispatch(pass, P.matvecF16, [L.wk, W.normed, W.k, kParams],
        [bb.n_kv_heads * bb.head_dim]);

      this.dispatch(pass, P.matvecF16, [L.wv, W.normed, W.v, kParams],
        [bb.n_kv_heads * bb.head_dim]);
      pass.end();

      pass = encoder.beginComputePass({ label: `layer${layer}_rope_attn` });

      // 4. RoPE on Q and K
      const ropeQParams = this.packUniform([
        { u: bb.head_dim }, { u: pos }, { u: bb.n_heads }, { f: bb.rope_theta },
      ]);
      this.dispatch(pass, P.rope, [W.q, ropeQParams],
        [cdiv(bb.n_heads * bb.head_dim / 2, 64)]);

      const ropeKParams = this.packUniform([
        { u: bb.head_dim }, { u: pos }, { u: bb.n_kv_heads }, { f: bb.rope_theta },
      ]);
      this.dispatch(pass, P.rope, [W.k, ropeKParams],
        [cdiv(bb.n_kv_heads * bb.head_dim / 2, 64)]);

      // 5. Write K,V to cache
      const kvWriteParams = this.packUniform([
        { u: pos }, { u: bb.n_kv_heads * bb.head_dim },
      ]);
      this.dispatch(pass, P.kvCacheWrite,
        [W.k, W.v, kv.k, kv.v, kvWriteParams],
        [cdiv(bb.n_kv_heads * bb.head_dim, 256)]);

      // 6. Attention scores
      const seqLen = pos + 1;
      const kvRepeat = bb.n_heads / bb.n_kv_heads;
      const scoreParams = this.packUniform([
        { u: bb.n_heads }, { u: bb.n_kv_heads }, { u: bb.head_dim },
        { u: seqLen }, { u: kvRepeat },
      ]);
      this.dispatch(pass, P.attnScore,
        [W.q, kv.k, W.scores, scoreParams],
        [cdiv(bb.n_heads * seqLen, 64)]);
      pass.end();

      pass = encoder.beginComputePass({ label: `layer${layer}_attn_out` });

      // 7. Softmax
      const softmaxParams = this.packUniform([{ u: bb.n_heads }, { u: seqLen }]);
      this.dispatch(pass, P.softmax,
        [W.scores, softmaxParams],
        [bb.n_heads]);

      // 8. Attention value
      const valParams = this.packUniform([
        { u: bb.n_heads }, { u: bb.n_kv_heads }, { u: bb.head_dim },
        { u: seqLen }, { u: kvRepeat },
      ]);
      this.dispatch(pass, P.attnValue,
        [W.scores, kv.v, W.attn_out, valParams],
        [cdiv(bb.n_heads * bb.head_dim, 128)]);
      pass.end();

      pass = encoder.beginComputePass({ label: `layer${layer}_wo_res` });

      // 9. Output projection: hidden = Wo @ attn_out
      const woParams = this.packUniform([{ u: bb.dim }, { u: bb.n_heads * bb.head_dim }]);
      this.dispatch(pass, P.matvecF16, [L.wo, W.attn_out, W.hidden, woParams],
        [bb.dim]);
      pass.end();

      pass = encoder.beginComputePass({ label: `layer${layer}_res1` });

      // 10. Add residual: hidden += residual (in-place to avoid aliasing)
      const addParams = this.packUniform([{ u: bb.dim }]);
      this.dispatch(pass, P.addInPlace,
        [W.hidden, W.residual, addParams],
        [cdiv(bb.dim, 256)]);

      // 11. Save residual for FFN
      this.dispatch(pass, P.copyBuffer,
        [W.hidden, W.residual, copyParams],
        [cdiv(bb.dim, 256)]);

      // 12. RMS norm (FFN)
      const ffnNormParams = this.packUniform([{ u: bb.dim }, { f: bb.norm_eps }]);
      this.dispatch(pass, P.rmsNorm,
        [W.hidden, L.ffn_norm, W.normed, ffnNormParams],
        [1]);
      pass.end();

      // --- FFN half ---
      pass = encoder.beginComputePass({ label: `layer${layer}_ffn` });

      // 13. FFN: gate = w1 @ normed, up = w3 @ normed
      const gateParams = this.packUniform([{ u: bb.hidden_dim }, { u: bb.dim }]);
      this.dispatch(pass, P.matvecF16, [L.w1, W.normed, W.gate, gateParams],
        [bb.hidden_dim]);
      this.dispatch(pass, P.matvecF16, [L.w3, W.normed, W.up, gateParams],
        [bb.hidden_dim]);
      pass.end();

      pass = encoder.beginComputePass({ label: `layer${layer}_ffn_out` });

      // 14. SwiGLU: gate = silu(gate) * up (in-place)
      const swiParams = this.packUniform([{ u: bb.hidden_dim }]);
      this.dispatch(pass, P.swiGLU,
        [W.gate, W.up, swiParams],
        [cdiv(bb.hidden_dim, 256)]);

      // 15. Down projection: hidden = w2 @ gate
      const downParams = this.packUniform([{ u: bb.dim }, { u: bb.hidden_dim }]);
      this.dispatch(pass, P.matvecF16, [L.w2, W.gate, W.hidden, downParams],
        [bb.dim]);
      pass.end();

      pass = encoder.beginComputePass({ label: `layer${layer}_res2` });

      // 16. Add residual (in-place to avoid aliasing)
      this.dispatch(pass, P.addInPlace,
        [W.hidden, W.residual, addParams],
        [cdiv(bb.dim, 256)]);
      pass.end();
    }

    // --- Final norm ---
    pass = encoder.beginComputePass({ label: `final_norm` });
    const finalNormParams = this.packUniform([{ u: bb.dim }, { f: bb.norm_eps }]);
    this.dispatch(pass, P.rmsNorm,
      [W.hidden, M.final_norm, W.normed, finalNormParams],
      [1]);
    pass.end();

    // --- LM head (tied weights = tok_embeddings) ---
    // vocab_size (131072) exceeds maxComputeWorkgroupsPerDimension (65535),
    // so we dispatch in chunks using the 2D matvec shader with row_offset.
    pass = encoder.beginComputePass({ label: `lm_head` });
    {
      const chunkSize = 65535;
      for (let offset = 0; offset < bb.vocab_size; offset += chunkSize) {
        const rows = Math.min(chunkSize, bb.vocab_size - offset);
        const lmParams = this.packUniform([
          { u: rows }, { u: bb.dim }, { u: offset },
        ]);
        this.dispatch(pass, P.matvecF16Chunked,
          [M.tok_embeddings, W.normed, W.logits, lmParams],
          [rows]);
      }
    }
    pass.end();

    // --- Argmax ---
    pass = encoder.beginComputePass({ label: `argmax` });
    const argmaxParams = this.packUniform([{ u: bb.vocab_size }]);
    this.dispatch(pass, P.argmax,
      [W.logits, W.argmax_result, argmaxParams],
      [1]);
    pass.end();
  }

  // =========================================================================
  // FM TRANSFORMER: Flow-matching ODE with 7 Euler steps + CFG (linspace(0,1,8) → 7 intervals)
  // =========================================================================

  /**
   * Run FM transformer on the 3-token sequence with proper bidirectional attention.
   *
   * Sequence layout: fm_seq = [pos0: x_t_proj, pos1: time_proj, pos2: llm_proj]
   * Each position is [dim=3072] floats, total fm_seq = [3 * dim].
   *
   * All 3 positions go through shared weights with bidirectional attention.
   * The velocity output is extracted from position 0 (x_t) after final norm.
   */
  private fmTransformerPass(
    encoder: GPUCommandEncoder,
    velocityOut: GPUBuffer,
  ): void {
    const P = this.pipelines!;
    const W = this.workBuffers!;
    const M = this.modelBuffers!;
    const fm = this.config.fm;
    const dim = fm.dim;
    const seqLen = 3;
    const qDim = fm.n_heads * fm.head_dim;  // 4096
    const kvDim = fm.n_kv_heads * fm.head_dim;  // 1024
    const kvRepeat = fm.n_heads / fm.n_kv_heads;  // 4

    for (let layer = 0; layer < fm.n_layers; layer++) {
      const FL = M.fm_layers[layer];
      let pass: GPUComputePassEncoder;

      // --- Attention half ---
      pass = encoder.beginComputePass({ label: `fm_l${layer}_attn_prep` });

      // Save residual: fm_down = fm_seq (3 * dim)
      const copyAllParams = this.packUniform([{ u: seqLen * dim }, { u: 0 }, { u: 0 }]);
      this.dispatch(pass, P.copyBufferOffset,
        [W.fm_seq, W.fm_down, copyAllParams],
        [cdiv(seqLen * dim, 256)]);

      // RMSNorm each position independently → fm_gate
      for (let p = 0; p < seqLen; p++) {
        const off = p * dim;
        const normParams = this.packUniform([{ u: dim }, { f: 1e-5 }, { u: off }, { u: off }]);
        this.dispatch(pass, P.rmsNormOffset,
          [W.fm_seq, FL.attn_norm, W.fm_gate, normParams],
          [1]);
      }
      pass.end();

      // Q/K/V projections
      pass = encoder.beginComputePass({ label: `fm_l${layer}_qkv` });
      for (let p = 0; p < seqLen; p++) {
        const srcOff = p * dim;
        const qOff = p * qDim;
        const kvOff = p * kvDim;
        const qP = this.packUniform([{ u: qDim }, { u: dim }, { u: srcOff }, { u: qOff }]);
        this.dispatch(pass, P.matvecF16Offset, [FL.wq, W.fm_gate, W.fm_q, qP], [qDim]);
        const kP = this.packUniform([{ u: kvDim }, { u: dim }, { u: srcOff }, { u: kvOff }]);
        this.dispatch(pass, P.matvecF16Offset, [FL.wk, W.fm_gate, W.fm_k, kP], [kvDim]);
        this.dispatch(pass, P.matvecF16Offset, [FL.wv, W.fm_gate, W.fm_v, kP], [kvDim]);
      }
      pass.end();

      // NOTE: FM uses bidirectional attention WITHOUT RoPE (confirmed by MLX & vLLM-Omni)
      // No RoPE applied — the 3-element sequence [acoustic, time, llm] has no positional encoding.

      // Bidirectional attention
      pass = encoder.beginComputePass({ label: `fm_l${layer}_attn` });
      const biScoreParams = this.packUniform([
        { u: fm.n_heads }, { u: fm.n_kv_heads }, { u: fm.head_dim },
        { u: seqLen }, { u: kvRepeat },
      ]);
      this.dispatch(pass, P.biAttnScore,
        [W.fm_q, W.fm_k, W.fm_scores, biScoreParams],
        [cdiv(fm.n_heads * seqLen * seqLen, 64)]);
      pass.end();

      pass = encoder.beginComputePass({ label: `fm_l${layer}_attn_val` });
      const biSoftParams = this.packUniform([{ u: fm.n_heads }, { u: seqLen }]);
      this.dispatch(pass, P.biSoftmax,
        [W.fm_scores, biSoftParams],
        [cdiv(fm.n_heads * seqLen, 64)]);
      const biValParams = this.packUniform([
        { u: fm.n_heads }, { u: fm.n_kv_heads }, { u: fm.head_dim },
        { u: seqLen }, { u: kvRepeat },
      ]);
      this.dispatch(pass, P.biAttnValue,
        [W.fm_scores, W.fm_v, W.fm_attn_out, biValParams],
        [cdiv(seqLen * fm.n_heads * fm.head_dim, 64)]);
      pass.end();

      // Output projection → fm_seq + residual
      pass = encoder.beginComputePass({ label: `fm_l${layer}_wo_res` });
      for (let p = 0; p < seqLen; p++) {
        const srcOff = p * qDim;
        const dstOff = p * dim;
        const woP = this.packUniform([{ u: dim }, { u: qDim }, { u: srcOff }, { u: dstOff }]);
        this.dispatch(pass, P.matvecF16Offset,
          [FL.wo, W.fm_attn_out, W.fm_seq, woP], [dim]);
      }
      pass.end();

      pass = encoder.beginComputePass({ label: `fm_l${layer}_res1` });
      for (let p = 0; p < seqLen; p++) {
        const off = p * dim;
        const addP = this.packUniform([{ u: dim }, { u: off }, { u: off }]);
        this.dispatch(pass, P.addInPlaceOffset,
          [W.fm_seq, W.fm_down, addP],
          [cdiv(dim, 256)]);
      }

      // Save residual for FFN
      this.dispatch(pass, P.copyBufferOffset,
        [W.fm_seq, W.fm_down, copyAllParams],
        [cdiv(seqLen * dim, 256)]);
      pass.end();

      // --- FFN half ---
      pass = encoder.beginComputePass({ label: `fm_l${layer}_ffn` });
      for (let p2 = 0; p2 < seqLen; p2++) {
        const pOff = p2 * dim;
        const gOff = p2 * fm.hidden_dim;
        const normP2 = this.packUniform([{ u: dim }, { f: 1e-5 }, { u: pOff }, { u: 0 }]);
        this.dispatch(pass, P.rmsNormOffset,
          [W.fm_seq, FL.ffn_norm, W.fm_normed, normP2], [1]);
        const gateP2 = this.packUniform([{ u: fm.hidden_dim }, { u: dim }, { u: 0 }, { u: gOff }]);
        this.dispatch(pass, P.matvecF16Offset,
          [FL.w1, W.fm_normed, W.fm_gate, gateP2], [fm.hidden_dim]);
        this.dispatch(pass, P.matvecF16Offset,
          [FL.w3, W.fm_normed, W.fm_up, gateP2], [fm.hidden_dim]);
      }
      pass.end();

      pass = encoder.beginComputePass({ label: `fm_l${layer}_ffn_act` });
      for (let p2 = 0; p2 < seqLen; p2++) {
        const gOff = p2 * fm.hidden_dim;
        const swiP = this.packUniform([{ u: fm.hidden_dim }, { u: gOff }, { u: gOff }]);
        this.dispatch(pass, P.swiGLUOffset,
          [W.fm_gate, W.fm_up, swiP],
          [cdiv(fm.hidden_dim, 256)]);
      }
      pass.end();

      pass = encoder.beginComputePass({ label: `fm_l${layer}_ffn_down` });
      for (let p2 = 0; p2 < seqLen; p2++) {
        const gOff = p2 * fm.hidden_dim;
        const dOff = p2 * dim;
        const downP = this.packUniform([{ u: dim }, { u: fm.hidden_dim }, { u: gOff }, { u: dOff }]);
        this.dispatch(pass, P.matvecF16Offset,
          [FL.w2, W.fm_gate, W.fm_seq, downP], [dim]);
      }
      pass.end();

      // Add residual (in-place)
      pass = encoder.beginComputePass({ label: `fm_l${layer}_res2` });
      for (let p2 = 0; p2 < seqLen; p2++) {
        const off = p2 * dim;
        const addP = this.packUniform([{ u: dim }, { u: off }, { u: off }]);
        this.dispatch(pass, P.addInPlaceOffset,
          [W.fm_seq, W.fm_down, addP],
          [cdiv(dim, 256)]);
      }
      pass.end();
    }

    // Final norm on position 0 (x_t) → velocity output
    {
      const p = encoder.beginComputePass({ label: `fm_final_norm_vel` });
      const fmNormParams = this.packUniform([{ u: dim }, { f: 1e-5 }, { u: 0 }, { u: 0 }]);
      this.dispatch(p, P.rmsNormOffset,
        [W.fm_seq, M.fm_norm, W.fm_normed, fmNormParams],
        [1]);

      // velocity = acoustic_out @ fm_normed → [36]
      const acOutParams = this.packUniform([{ u: fm.n_acoustic_out }, { u: dim }]);
      this.dispatch(p, P.matvecF16,
        [M.fm_acoustic_out, W.fm_normed, velocityOut, acOutParams],
        [fm.n_acoustic_out]);
      p.end();
    }
  }

  /**
   * Full FM forward: semantic logits + 8-step Euler ODE with CFG.
   *
   * Input: backbone normed hidden state in W.normed
   * Output: W.semantic_logits, W.acoustic_codes
   *
   * Per architecture doc, the 3-token sequence is:
   *   [x_t_proj, time_proj, llm_proj]
   * Position 0 = x_t projected, 1 = time embedding projected, 2 = LLM hidden projected.
   *
   * CFG: run conditioned (real llm_hidden) and unconditioned (zero llm_hidden)
   * v = alpha * v_cond + (1 - alpha) * v_uncond
   */
  private fmForward(encoder: GPUCommandEncoder, debugNoise?: Float32Array): void {
    const P = this.pipelines!;
    const W = this.workBuffers!;
    const M = this.modelBuffers!;
    const fm = this.config.fm;
    const dim = fm.dim;

    let pass: GPUComputePassEncoder;

    // --- Initial projections ---
    pass = encoder.beginComputePass({ label: 'fm_init' });

    // Semantic head: semantic_logits = semantic_out @ normed
    const semParams = this.packUniform([{ u: fm.semantic_vocab }, { u: dim }]);
    this.dispatch(pass, P.matvecF16,
      [M.fm_semantic_out, W.normed, W.semantic_logits, semParams],
      [fm.semantic_vocab]);

    // Project LLM hidden for conditioned pass: llm_proj → temp
    const llmProjParams = this.packUniform([{ u: dim }, { u: dim }]);
    this.dispatch(pass, P.matvecF16,
      [M.fm_llm_proj, W.normed, W.fm_hidden, llmProjParams],
      [dim]);

    // Initialize x_t to random Gaussian noise (flow matching ODE starts from noise)
    {
      const noise = debugNoise ?? new Float32Array(fm.n_acoustic_out);
      if (!debugNoise) for (let i = 0; i < fm.n_acoustic_out; i++) {
        // Box-Muller transform for Gaussian noise
        const u1 = Math.random();
        const u2 = Math.random();
        noise[i] = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
      }
      this.device!.queue.writeBuffer(W.x_t, 0, noise as Float32Array<ArrayBuffer>);
    }
    pass.end();

    // Semantic argmax (needs barrier after semantic head)
    pass = encoder.beginComputePass({ label: 'fm_semantic_argmax' });
    const semArgmaxParams = this.packUniform([{ u: fm.semantic_vocab }]);
    this.dispatch(pass, P.argmax,
      [W.semantic_logits, W.semantic_argmax, semArgmaxParams],
      [1]);
    pass.end();

    // --- Euler integration: linspace(0, 1, nfe) gives nfe-1 steps ---
    // Matches vLLM-Omni/MLX: 8 timestep values, 7 integration steps, dt=1/7
    for (let step = 0; step < fm.nfe - 1; step++) {
      const t = step / (fm.nfe - 1);
      const dt = 1.0 / (fm.nfe - 1);

      // Step setup: time embedding + projections + sequence assembly
      pass = encoder.beginComputePass({ label: `fm_step${step}_prep` });

      // 1. Time embedding → time_embed
      const timeParams = this.packUniform([{ u: dim }, { f: t }]);
      this.dispatch(pass, P.timeEmbedding,
        [W.time_embed, timeParams],
        [cdiv(dim / 2, 256)]);
      pass.end();

      pass = encoder.beginComputePass({ label: `fm_step${step}_proj` });

      // 2. Project time → time_proj
      const timeProjParams = this.packUniform([{ u: dim }, { u: dim }]);
      this.dispatch(pass, P.matvecF16,
        [M.fm_time_proj, W.time_embed, W.time_proj, timeProjParams],
        [dim]);

      // 3. Project x_t → x_t_proj [dim] (saved for both cond and uncond passes)
      const inputProjParams = this.packUniform([{ u: dim }, { u: fm.n_acoustic_out }]);
      this.dispatch(pass, P.matvecF16,
        [M.fm_input_proj, W.x_t, W.x_t_proj, inputProjParams],
        [dim]);
      pass.end();

      // 4. Assemble conditioned sequence: fm_seq = [x_t_proj, time_proj, llm_proj]
      pass = encoder.beginComputePass({ label: `fm_step${step}_assemble` });
      const cp0 = this.packUniform([{ u: dim }, { u: 0 }, { u: 0 }]);
      this.dispatch(pass, P.copyBufferOffset,
        [W.x_t_proj, W.fm_seq, cp0],
        [cdiv(dim, 256)]);
      const cp1 = this.packUniform([{ u: dim }, { u: 0 }, { u: dim }]);
      this.dispatch(pass, P.copyBufferOffset,
        [W.time_proj, W.fm_seq, cp1],
        [cdiv(dim, 256)]);
      const cp2 = this.packUniform([{ u: dim }, { u: 0 }, { u: 2 * dim }]);
      this.dispatch(pass, P.copyBufferOffset,
        [W.fm_hidden, W.fm_seq, cp2],
        [cdiv(dim, 256)]);
      pass.end();

      // 5. Run FM transformer (conditioned) → velocity
      this.fmTransformerPass(encoder, W.velocity);

      // 6. Assemble unconditioned sequence: pos 2 = zeros
      pass = encoder.beginComputePass({ label: `fm_step${step}_uncond` });
      this.dispatch(pass, P.copyBufferOffset,
        [W.x_t_proj, W.fm_seq, cp0],
        [cdiv(dim, 256)]);
      this.dispatch(pass, P.copyBufferOffset,
        [W.time_proj, W.fm_seq, cp1],
        [cdiv(dim, 256)]);
      const zeroLlmParams = this.packUniform([{ u: dim }]);
      this.dispatch(pass, P.zeroFill, [W.fm_residual, zeroLlmParams],
        [cdiv(dim, 256)]);
      this.dispatch(pass, P.copyBufferOffset,
        [W.fm_residual, W.fm_seq, cp2],
        [cdiv(dim, 256)]);
      pass.end();

      // 7. Run FM transformer (unconditioned) → v_uncond
      this.fmTransformerPass(encoder, W.v_uncond);

      // 8. CFG combine + Euler step
      pass = encoder.beginComputePass({ label: `fm_step${step}_euler` });
      const cfgParams = this.packUniform([{ u: fm.n_acoustic_out }, { f: fm.cfg_alpha }]);
      this.dispatch(pass, P.cfgCombine,
        [W.velocity, W.v_uncond, cfgParams],
        [cdiv(fm.n_acoustic_out, 64)]);

      // 9. Euler step: x_t = x_t + velocity * dt
      const eulerParams = this.packUniform([{ u: fm.n_acoustic_out }, { f: dt }]);
      this.dispatch(pass, P.eulerStep,
        [W.x_t, W.velocity, eulerParams],
        [cdiv(fm.n_acoustic_out, 64)]);
      pass.end();
    }

    // --- FSQ quantize: x_t → acoustic codes ---
    pass = encoder.beginComputePass({ label: 'fm_fsq' });
    const fsqParams = this.packUniform([
      { u: fm.n_acoustic_out },
      { u: this.config.codec.acoustic_codebook_size },  // 21 levels
      { u: 2 },  // offset for special tokens
    ]);
    this.dispatch(pass, P.fsqQuantize,
      [W.x_t, W.acoustic_codes, fsqParams],
      [cdiv(fm.n_acoustic_out, 64)]);
    pass.end();
  }

  // =========================================================================
  // CODEC DECODER: codes → 24kHz waveform
  // =========================================================================

  /**
   * Decode semantic + acoustic codes into audio waveform.
   *
   * Pipeline:
   * 1. VQ lookup: semantic_codes → [T, 256]
   * 2. FSQ dequant: acoustic_codes → [T, 36]
   * 3. Concat → [T, 292]
   * 4. Input conv: CausalConv1d(292→1024, k=3, s=1) → [T, 1024]
   * 5. 4 stages: [2× transformer + optional conv upsample]
   * 6. Output conv: CausalConv1d(1024→240, k=7) → [T_out, 240]
   * 7. Reshape to waveform: [T_out × 240] samples
   *
   * @param semanticCodes - [T] semantic code IDs
   * @param acousticCodes - [T, 36] acoustic codes
   * @returns Audio samples at 24kHz
   */
  async codecDecode(
    semanticCodes: Uint32Array,
    acousticCodes: Uint32Array,  // flattened [T * 36]
  ): Promise<Float32Array> {
    const d = this.device!;
    const P = this.pipelines!;
    const M = this.modelBuffers!;
    const codec = this.config.codec;
    const T = semanticCodes.length;
    const codDim = codec.dim;  // 1024

    // Upload codes to GPU
    const semCodesBuf = this.uploadArray(semanticCodes);
    const acCodesBuf = this.uploadArray(acousticCodes);

    // Allocate codec working buffers (dynamic size based on T)
    const semEmbed = this.createGPUBuffer(T * codec.semantic_dim * 4, 'codec_sem_embed');
    const acFloat = this.createGPUBuffer(T * codec.n_acoustic_codebook * 4, 'codec_ac_float');
    const concatBuf = this.createGPUBuffer(T * 292 * 4, 'codec_concat');

    // Track temporal dimension through upsampling
    let curT = T;
    let curBuf = this.createGPUBuffer(curT * codDim * 4, 'codec_cur');
    let tmpBuf = this.createGPUBuffer(curT * codDim * 4, 'codec_tmp');  // Will be resized

    const encoder = d.createCommandEncoder({ label: 'codec_decode' });
    const toDestroy: GPUBuffer[] = [];  // defer destruction until after submit

    // Helper: each dispatch gets its own compute pass for proper barriers
    const dp = (pipeline: GPUComputePipeline, bindings: GPUBuffer[], workgroups: [number, number?, number?], label: string) => {
      const p = encoder.beginComputePass({ label });
      this.dispatch(p, pipeline, bindings, workgroups);
      p.end();
    };

    // 1. VQ lookup
    const vqParams = this.packUniform([{ u: T }, { u: codec.semantic_dim }]);
    dp(P.vqLookup,
      [semCodesBuf, M.codec_semantic_codebook, semEmbed, vqParams],
      [cdiv(T * codec.semantic_dim, 128)], 'codec_vq');

    // 2. FSQ dequant
    const fsqParams = this.packUniform([
      { u: T }, { u: codec.n_acoustic_codebook }, { u: codec.acoustic_codebook_size }, { u: 2 },
    ]);
    dp(P.fsqDequant,
      [acCodesBuf, acFloat, fsqParams],
      [cdiv(T * codec.n_acoustic_codebook, 64)], 'codec_fsq');

    // 3. Concat [T, 256] + [T, 36] → [T, 292]
    const concatParams = this.packUniform([{ u: T }, { u: codec.semantic_dim }, { u: codec.n_acoustic_codebook }]);
    dp(P.concatCodecInput,
      [semEmbed, acFloat, concatBuf, concatParams],
      [cdiv(T * 292, 256)], 'codec_concat');

    // 4. Input conv: CausalConv1d(292→1024, k=3, s=1)
    const inputConvParams = this.packUniform([
      { u: 292 }, { u: codDim }, { u: 3 }, { u: curT }, { u: 1 },
    ]);
    dp(P.causalConv1d,
      [concatBuf, M.codec_input_conv_w, M.codec_input_conv_g, curBuf, inputConvParams],
      [cdiv(codDim * curT, 64)], 'codec_input_conv');

    // 5. Four decoder stages
    // Strides/kernels per stage (stages 0-2 have conv_up with stride 2, stage 3 has none)
    const stageStrides = [2, 2, 2, 1];
    const stageKernels = [4, 4, 4, 3];  // kernel for stage 3 unused
    // Sliding windows: smallest at bottleneck (stage 0), doubling with upsampling
    // base=16, half_upon_downsampling → decoder reverses: 2, 4, 8, 16
    const windows = [2, 4, 8, 16];

    for (let stage = 0; stage < codec.decoder_stages; stage++) {
      const stageData = M.codec_stages[stage];

      // 2 transformer layers per stage
      for (let layerIdx = 0; layerIdx < codec.decoder_layers_per_stage; layerIdx++) {
        const TL = stageData.transformer_layers[layerIdx];

        // Allocate buffers for this layer's operations
        const layerBufSize = curT * codDim * 4;
        // Reuse tmpBuf, expand if needed
        if (tmpBuf.size < layerBufSize) {
          toDestroy.push(tmpBuf);
          tmpBuf = this.createGPUBuffer(layerBufSize, 'codec_tmp');
        }
        const normBuf = tmpBuf;

        // --- Attention sublayer ---
        // Save residual in normed-buf temp
        const totalElems = curT * codDim;
        const bCopyP = this.packUniform([{ u: totalElems }]);
        dp(P.batchedCopy, [curBuf, normBuf, bCopyP], [cdiv(totalElems, 256)],
          `codec_s${stage}_l${layerIdx}_copy_res`);

        // Batched RMSNorm
        const bNormP = this.packUniform([{ u: codDim }, { f: codec.norm_eps }, { u: curT }]);
        const attnNormed = this.createGPUBuffer(layerBufSize, 'codec_attn_normed');
        dp(P.batchedRmsNorm,
          [curBuf, TL.attn_norm, attnNormed, bNormP],
          [curT], `codec_s${stage}_l${layerIdx}_attn_norm`);

        // Q/K/V projections (batched)
        const qBuf = this.createGPUBuffer(curT * codDim * 4, 'codec_q');
        const kBuf = this.createGPUBuffer(curT * codDim * 4, 'codec_k');
        const vBuf = this.createGPUBuffer(curT * codDim * 4, 'codec_v');

        const bMvP = this.packUniform([{ u: codDim }, { u: codDim }, { u: curT }]);
        dp(P.batchedMatvecF16,
          [TL.wq, attnNormed, qBuf, bMvP], [codDim, curT],
          `codec_s${stage}_l${layerIdx}_qproj`);
        dp(P.batchedMatvecF16,
          [TL.wk, attnNormed, kBuf, bMvP], [codDim, curT],
          `codec_s${stage}_l${layerIdx}_kproj`);
        dp(P.batchedMatvecF16,
          [TL.wv, attnNormed, vBuf, bMvP], [codDim, curT],
          `codec_s${stage}_l${layerIdx}_vproj`);

        // QK normalization (in-place)
        const qkNP = this.packUniform([
          { u: codec.n_heads }, { u: codec.head_dim }, { u: curT }, { f: codec.qk_norm_eps },
        ]);
        dp(P.qkNorm, [qBuf, TL.q_norm, qkNP],
          [cdiv(curT * codec.n_heads, 128)], `codec_s${stage}_l${layerIdx}_qnorm`);
        dp(P.qkNorm, [kBuf, TL.k_norm, qkNP],
          [cdiv(curT * codec.n_heads, 128)], `codec_s${stage}_l${layerIdx}_knorm`);

        // ALiBi attention scores
        const scoresBuf = this.createGPUBuffer(codec.n_heads * curT * curT * 4, 'codec_scores');
        const alibiP = this.packUniform([
          { u: codec.n_heads }, { u: codec.head_dim }, { u: curT }, { u: windows[stage] },
        ]);
        dp(P.alibiAttnScore,
          [qBuf, kBuf, scoresBuf, alibiP],
          [cdiv(curT, 64), curT, codec.n_heads], `codec_s${stage}_l${layerIdx}_attn_score`);

        // Softmax
        const cSoftP = this.packUniform([{ u: codec.n_heads }, { u: curT }]);
        dp(P.codecSoftmax, [scoresBuf, cSoftP],
          [cdiv(codec.n_heads * curT, 64)], `codec_s${stage}_l${layerIdx}_softmax`);

        // Attention value
        const attnOutBuf = this.createGPUBuffer(curT * codDim * 4, 'codec_attn_out');
        const cValP = this.packUniform([{ u: codec.n_heads }, { u: codec.head_dim }, { u: curT }]);
        dp(P.codecAttnValue,
          [scoresBuf, vBuf, attnOutBuf, cValP],
          [cdiv(curT * codec.n_heads * codec.head_dim, 64)], `codec_s${stage}_l${layerIdx}_attn_val`);

        // Output projection
        const woBuf = this.createGPUBuffer(layerBufSize, 'codec_wo_out');
        dp(P.batchedMatvecF16,
          [TL.wo, attnOutBuf, woBuf, bMvP], [codDim, curT],
          `codec_s${stage}_l${layerIdx}_wo`);

        // Layer scale + residual: curBuf = woBuf * attn_scale + normBuf(residual)
        const lsP = this.packUniform([{ u: codDim }, { u: totalElems }]);
        dp(P.batchedLayerScale,
          [woBuf, TL.attn_scale, normBuf, curBuf, lsP],
          [cdiv(totalElems, 256)], `codec_s${stage}_l${layerIdx}_attn_res`);

        // --- FFN sublayer ---
        // Save residual
        dp(P.batchedCopy, [curBuf, normBuf, bCopyP], [cdiv(totalElems, 256)],
          `codec_s${stage}_l${layerIdx}_copy_ffn_res`);

        // Norm
        const ffnNormed = this.createGPUBuffer(layerBufSize, 'codec_ffn_normed');
        dp(P.batchedRmsNorm,
          [curBuf, TL.ffn_norm, ffnNormed, bNormP],
          [curT], `codec_s${stage}_l${layerIdx}_ffn_norm`);

        // Gate + Up projections
        const hiddenSize = curT * codec.hidden_dim;
        const gateBuf = this.createGPUBuffer(hiddenSize * 4, 'codec_gate');
        const upBuf = this.createGPUBuffer(hiddenSize * 4, 'codec_up');
        const ffnMvP = this.packUniform([{ u: codec.hidden_dim }, { u: codDim }, { u: curT }]);
        dp(P.batchedMatvecF16,
          [TL.w1, ffnNormed, gateBuf, ffnMvP], [codec.hidden_dim, curT],
          `codec_s${stage}_l${layerIdx}_gate`);
        dp(P.batchedMatvecF16,
          [TL.w3, ffnNormed, upBuf, ffnMvP], [codec.hidden_dim, curT],
          `codec_s${stage}_l${layerIdx}_up`);

        // SwiGLU (in-place)
        const swiP = this.packUniform([{ u: hiddenSize }]);
        dp(P.batchedSwiGLU,
          [gateBuf, upBuf, swiP],
          [cdiv(hiddenSize, 256)], `codec_s${stage}_l${layerIdx}_swiglu`);

        // Down projection
        const downBuf = this.createGPUBuffer(layerBufSize, 'codec_down');
        const downMvP = this.packUniform([{ u: codDim }, { u: codec.hidden_dim }, { u: curT }]);
        dp(P.batchedMatvecF16,
          [TL.w2, gateBuf, downBuf, downMvP], [codDim, curT],
          `codec_s${stage}_l${layerIdx}_down`);

        // Layer scale + residual
        dp(P.batchedLayerScale,
          [downBuf, TL.ffn_scale, normBuf, curBuf, lsP],
          [cdiv(totalElems, 256)], `codec_s${stage}_l${layerIdx}_ffn_res`);

        // Defer cleanup until after command buffer submission
        toDestroy.push(attnNormed, qBuf, kBuf, vBuf, scoresBuf,
          attnOutBuf, woBuf, ffnNormed, gateBuf, upBuf, downBuf);
      }

      // Conv transpose upsample (stages 0-2 have conv_up with stride 2)
      if (stageData.conv_w && stageData.conv_scale && stageStrides[stage] > 1) {
        const newT = curT * stageStrides[stage];
        const upsampledBuf = this.createGPUBuffer(newT * codDim * 4, 'codec_upsampled');

        const convP = this.packUniform([
          { u: codDim }, { u: codDim }, { u: stageKernels[stage] }, { u: newT }, { u: stageStrides[stage] },
        ]);
        dp(P.causalConvTranspose1d,
          [curBuf, stageData.conv_w, stageData.conv_scale, upsampledBuf, convP],
          [cdiv(codDim * newT, 64)], `codec_s${stage}_conv_up`);

        toDestroy.push(curBuf);
        curBuf = upsampledBuf;
        curT = newT;
      }
    }

    // 6. Output conv: CausalConv1d(1024→240, k=7)
    const outT = curT;
    const outBuf = this.createGPUBuffer(outT * codec.patch_size * 4, 'codec_output');
    const outConvP = this.packUniform([
      { u: codDim }, { u: codec.patch_size }, { u: 7 }, { u: outT }, { u: 1 },
    ]);
    dp(P.causalConv1d,
      [curBuf, M.codec_output_conv_w, M.codec_output_conv_g, outBuf, outConvP],
      [cdiv(codec.patch_size * outT, 64)], 'codec_output_conv');

    d.pushErrorScope('validation');
    d.queue.submit([encoder.finish()]);
    await d.queue.onSubmittedWorkDone();
    const codecErr = await d.popErrorScope();
    if (codecErr) {
      (globalThis as any).__codecError = codecErr.message;
    }

    // 7. Read back audio: [outT, 240] → flatten to waveform
    const totalSamples = outT * codec.patch_size;
    const audio = await this.readF32Array(outBuf, totalSamples);
    // Debug: expose stats on globalThis
    let nonZero = 0;
    for (let i = 0; i < Math.min(audio.length, 1000); i++) { if (audio[i] !== 0) nonZero++; }
    (globalThis as any).__codecDebug = {
      outT, patchSize: codec.patch_size, totalSamples, nonZero,
      first5: Array.from(audio.slice(0, 5)),
      curT,
    };

    // Cleanup all buffers
    for (const buf of toDestroy) buf.destroy();
    semCodesBuf.destroy();
    acCodesBuf.destroy();
    semEmbed.destroy();
    acFloat.destroy();
    concatBuf.destroy();
    curBuf.destroy();
    tmpBuf.destroy();
    outBuf.destroy();

    return audio;
  }

  /**
   * Debug codec decode: same as codecDecode but returns intermediates at each stage.
   * Batches operations into larger command buffers and only reads at stage boundaries.
   */
  async debugCodecDecode(
    semanticCodes: Uint32Array,
    acousticCodes: Uint32Array,  // flattened [T * 36]
  ): Promise<Record<string, Float32Array>> {
    const d = this.device!;
    const P = this.pipelines!;
    const M = this.modelBuffers!;
    const codec = this.config.codec;
    const T = semanticCodes.length;
    const codDim = codec.dim;  // 1024
    const intermediates: Record<string, Float32Array> = {};

    // Upload codes to GPU
    const semCodesBuf = this.uploadArray(semanticCodes);
    const acCodesBuf = this.uploadArray(acousticCodes);

    // Allocate codec working buffers
    const semEmbed = this.createGPUBuffer(T * codec.semantic_dim * 4, 'codec_sem_embed');
    const acFloat = this.createGPUBuffer(T * codec.n_acoustic_codebook * 4, 'codec_ac_float');
    const concatBuf = this.createGPUBuffer(T * 292 * 4, 'codec_concat');

    let curT = T;
    let curBuf = this.createGPUBuffer(curT * codDim * 4, 'codec_cur');
    let tmpBuf = this.createGPUBuffer(curT * codDim * 4, 'codec_tmp');

    const toDestroy: GPUBuffer[] = [];

    // Helper: each dispatch gets its own compute pass for proper barriers
    const dp = (encoder: GPUCommandEncoder, pipeline: GPUComputePipeline, bindings: GPUBuffer[], workgroups: [number, number?, number?], label: string) => {
      const p = encoder.beginComputePass({ label });
      this.dispatch(p, pipeline, bindings, workgroups);
      p.end();
    };

    // Phase 1: VQ + FSQ + Concat + Input conv (all in one submission)
    {
      const encoder = d.createCommandEncoder({ label: 'codec_phase1' });

      // 1. VQ lookup
      const vqParams = this.packUniform([{ u: T }, { u: codec.semantic_dim }]);
      dp(encoder, P.vqLookup,
        [semCodesBuf, M.codec_semantic_codebook, semEmbed, vqParams],
        [cdiv(T * codec.semantic_dim, 128)], 'codec_vq');

      // 2. FSQ dequant
      const fsqParams = this.packUniform([
        { u: T }, { u: codec.n_acoustic_codebook }, { u: codec.acoustic_codebook_size }, { u: 2 },
      ]);
      dp(encoder, P.fsqDequant,
        [acCodesBuf, acFloat, fsqParams],
        [cdiv(T * codec.n_acoustic_codebook, 64)], 'codec_fsq');

      // 3. Concat
      const concatParams = this.packUniform([{ u: T }, { u: codec.semantic_dim }, { u: codec.n_acoustic_codebook }]);
      dp(encoder, P.concatCodecInput,
        [semEmbed, acFloat, concatBuf, concatParams],
        [cdiv(T * 292, 256)], 'codec_concat');

      // 4. Input conv
      const inputConvParams = this.packUniform([
        { u: 292 }, { u: codDim }, { u: 3 }, { u: curT }, { u: 1 },
      ]);
      dp(encoder, P.causalConv1d,
        [concatBuf, M.codec_input_conv_w, M.codec_input_conv_g, curBuf, inputConvParams],
        [cdiv(codDim * curT, 64)], 'codec_input_conv');

      d.queue.submit([encoder.finish()]);
      await d.queue.onSubmittedWorkDone();
    }

    // Read phase 1 intermediates
    intermediates['vq_embed'] = await this.readF32Array(semEmbed, T * codec.semantic_dim);
    intermediates['fsq_dequant'] = await this.readF32Array(acFloat, T * codec.n_acoustic_codebook);
    intermediates['concat'] = await this.readF32Array(concatBuf, T * 292);
    intermediates['after_input_conv'] = await this.readF32Array(curBuf, curT * codDim);

    // 5. Four decoder stages — one submission per stage
    const stageStrides = [2, 2, 2, 1];
    const stageKernels = [4, 4, 4, 3];
    const windows = [2, 4, 8, 16];

    for (let stage = 0; stage < codec.decoder_stages; stage++) {
      const stageData = M.codec_stages[stage];
      const encoder = d.createCommandEncoder({ label: `codec_stage${stage}` });

      // 2 transformer layers per stage
      for (let layerIdx = 0; layerIdx < codec.decoder_layers_per_stage; layerIdx++) {
        const TL = stageData.transformer_layers[layerIdx];

        const layerBufSize = curT * codDim * 4;
        if (tmpBuf.size < layerBufSize) {
          toDestroy.push(tmpBuf);
          tmpBuf = this.createGPUBuffer(layerBufSize, 'codec_tmp');
        }
        const normBuf = tmpBuf;

        // --- Attention sublayer ---
        const totalElems = curT * codDim;
        const bCopyP = this.packUniform([{ u: totalElems }]);
        dp(encoder, P.batchedCopy, [curBuf, normBuf, bCopyP], [cdiv(totalElems, 256)],
          `codec_s${stage}_l${layerIdx}_copy_res`);

        const bNormP = this.packUniform([{ u: codDim }, { f: codec.norm_eps }, { u: curT }]);
        const attnNormed = this.createGPUBuffer(layerBufSize, 'codec_attn_normed');
        dp(encoder, P.batchedRmsNorm,
          [curBuf, TL.attn_norm, attnNormed, bNormP],
          [curT], `codec_s${stage}_l${layerIdx}_attn_norm`);

        const qBuf = this.createGPUBuffer(curT * codDim * 4, 'codec_q');
        const kBuf = this.createGPUBuffer(curT * codDim * 4, 'codec_k');
        const vBuf = this.createGPUBuffer(curT * codDim * 4, 'codec_v');

        const bMvP = this.packUniform([{ u: codDim }, { u: codDim }, { u: curT }]);
        dp(encoder, P.batchedMatvecF16,
          [TL.wq, attnNormed, qBuf, bMvP], [codDim, curT],
          `codec_s${stage}_l${layerIdx}_qproj`);
        dp(encoder, P.batchedMatvecF16,
          [TL.wk, attnNormed, kBuf, bMvP], [codDim, curT],
          `codec_s${stage}_l${layerIdx}_kproj`);
        dp(encoder, P.batchedMatvecF16,
          [TL.wv, attnNormed, vBuf, bMvP], [codDim, curT],
          `codec_s${stage}_l${layerIdx}_vproj`);

        const qkNP = this.packUniform([
          { u: codec.n_heads }, { u: codec.head_dim }, { u: curT }, { f: codec.qk_norm_eps },
        ]);
        dp(encoder, P.qkNorm, [qBuf, TL.q_norm, qkNP],
          [cdiv(curT * codec.n_heads, 128)], `codec_s${stage}_l${layerIdx}_qnorm`);
        dp(encoder, P.qkNorm, [kBuf, TL.k_norm, qkNP],
          [cdiv(curT * codec.n_heads, 128)], `codec_s${stage}_l${layerIdx}_knorm`);

        const scoresBuf = this.createGPUBuffer(codec.n_heads * curT * curT * 4, 'codec_scores');
        const alibiP = this.packUniform([
          { u: codec.n_heads }, { u: codec.head_dim }, { u: curT }, { u: windows[stage] },
        ]);
        dp(encoder, P.alibiAttnScore,
          [qBuf, kBuf, scoresBuf, alibiP],
          [cdiv(curT, 64), curT, codec.n_heads], `codec_s${stage}_l${layerIdx}_attn_score`);

        const cSoftP = this.packUniform([{ u: codec.n_heads }, { u: curT }]);
        dp(encoder, P.codecSoftmax, [scoresBuf, cSoftP],
          [cdiv(codec.n_heads * curT, 64)], `codec_s${stage}_l${layerIdx}_softmax`);

        const attnOutBuf = this.createGPUBuffer(curT * codDim * 4, 'codec_attn_out');
        const cValP = this.packUniform([{ u: codec.n_heads }, { u: codec.head_dim }, { u: curT }]);
        dp(encoder, P.codecAttnValue,
          [scoresBuf, vBuf, attnOutBuf, cValP],
          [cdiv(curT * codec.n_heads * codec.head_dim, 64)], `codec_s${stage}_l${layerIdx}_attn_val`);

        const woBuf = this.createGPUBuffer(layerBufSize, 'codec_wo_out');
        dp(encoder, P.batchedMatvecF16,
          [TL.wo, attnOutBuf, woBuf, bMvP], [codDim, curT],
          `codec_s${stage}_l${layerIdx}_wo`);

        const lsP = this.packUniform([{ u: codDim }, { u: totalElems }]);
        dp(encoder, P.batchedLayerScale,
          [woBuf, TL.attn_scale, normBuf, curBuf, lsP],
          [cdiv(totalElems, 256)], `codec_s${stage}_l${layerIdx}_attn_res`);

        // --- FFN sublayer ---
        dp(encoder, P.batchedCopy, [curBuf, normBuf, bCopyP], [cdiv(totalElems, 256)],
          `codec_s${stage}_l${layerIdx}_copy_ffn_res`);

        const ffnNormed = this.createGPUBuffer(layerBufSize, 'codec_ffn_normed');
        dp(encoder, P.batchedRmsNorm,
          [curBuf, TL.ffn_norm, ffnNormed, bNormP],
          [curT], `codec_s${stage}_l${layerIdx}_ffn_norm`);

        const hiddenSize = curT * codec.hidden_dim;
        const gateBuf = this.createGPUBuffer(hiddenSize * 4, 'codec_gate');
        const upBuf = this.createGPUBuffer(hiddenSize * 4, 'codec_up');
        const ffnMvP = this.packUniform([{ u: codec.hidden_dim }, { u: codDim }, { u: curT }]);
        dp(encoder, P.batchedMatvecF16,
          [TL.w1, ffnNormed, gateBuf, ffnMvP], [codec.hidden_dim, curT],
          `codec_s${stage}_l${layerIdx}_gate`);
        dp(encoder, P.batchedMatvecF16,
          [TL.w3, ffnNormed, upBuf, ffnMvP], [codec.hidden_dim, curT],
          `codec_s${stage}_l${layerIdx}_up`);

        const swiP = this.packUniform([{ u: hiddenSize }]);
        dp(encoder, P.batchedSwiGLU,
          [gateBuf, upBuf, swiP],
          [cdiv(hiddenSize, 256)], `codec_s${stage}_l${layerIdx}_swiglu`);

        const downBuf = this.createGPUBuffer(layerBufSize, 'codec_down');
        const downMvP = this.packUniform([{ u: codDim }, { u: codec.hidden_dim }, { u: curT }]);
        dp(encoder, P.batchedMatvecF16,
          [TL.w2, gateBuf, downBuf, downMvP], [codDim, curT],
          `codec_s${stage}_l${layerIdx}_down`);

        dp(encoder, P.batchedLayerScale,
          [downBuf, TL.ffn_scale, normBuf, curBuf, lsP],
          [cdiv(totalElems, 256)], `codec_s${stage}_l${layerIdx}_ffn_res`);

        toDestroy.push(attnNormed, qBuf, kBuf, vBuf, scoresBuf,
          attnOutBuf, woBuf, ffnNormed, gateBuf, upBuf, downBuf);
      }

      // Conv transpose upsample (stages 0-2)
      if (stageData.conv_w && stageData.conv_scale && stageStrides[stage] > 1) {
        const newT = curT * stageStrides[stage];
        const upsampledBuf = this.createGPUBuffer(newT * codDim * 4, 'codec_upsampled');

        const convP = this.packUniform([
          { u: codDim }, { u: codDim }, { u: stageKernels[stage] }, { u: newT }, { u: stageStrides[stage] },
        ]);
        dp(encoder, P.causalConvTranspose1d,
          [curBuf, stageData.conv_w, stageData.conv_scale, upsampledBuf, convP],
          [cdiv(codDim * newT, 64)], `codec_s${stage}_conv_up`);

        // Submit this stage
        d.queue.submit([encoder.finish()]);
        await d.queue.onSubmittedWorkDone();

        // Read intermediates
        intermediates[`after_stage${stage}_transformer`] = await this.readF32Array(curBuf, curT * codDim);

        toDestroy.push(curBuf);
        curBuf = upsampledBuf;
        curT = newT;

        intermediates[`after_stage${stage}_conv_up`] = await this.readF32Array(curBuf, curT * codDim);
      } else {
        // Stage 3: no conv, just submit and read
        d.queue.submit([encoder.finish()]);
        await d.queue.onSubmittedWorkDone();
        intermediates[`after_stage${stage}_transformer`] = await this.readF32Array(curBuf, curT * codDim);
      }
    }

    // 6. Output conv
    {
      const outT = curT;
      const outBuf = this.createGPUBuffer(outT * codec.patch_size * 4, 'codec_output');
      const outConvP = this.packUniform([
        { u: codDim }, { u: codec.patch_size }, { u: 7 }, { u: outT }, { u: 1 },
      ]);
      const encoder = d.createCommandEncoder({ label: 'codec_output' });
      dp(encoder, P.causalConv1d,
        [curBuf, M.codec_output_conv_w, M.codec_output_conv_g, outBuf, outConvP],
        [cdiv(codec.patch_size * outT, 64)], 'codec_output_conv');
      d.queue.submit([encoder.finish()]);
      await d.queue.onSubmittedWorkDone();

      intermediates['after_output_conv'] = await this.readF32Array(outBuf, outT * codec.patch_size);
      intermediates['audio'] = intermediates['after_output_conv'];
      toDestroy.push(outBuf);
    }

    // Cleanup
    for (const buf of toDestroy) buf.destroy();
    semCodesBuf.destroy();
    acCodesBuf.destroy();
    semEmbed.destroy();
    acFloat.destroy();
    concatBuf.destroy();
    curBuf.destroy();
    tmpBuf.destroy();

    return intermediates;
  }

  /** Upload a typed array to a GPU storage buffer. */
  private uploadArray(data: Uint32Array | Float32Array): GPUBuffer {
    const buf = this.device!.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    });
    if (data instanceof Uint32Array) {
      new Uint32Array(buf.getMappedRange()).set(data);
    } else {
      new Float32Array(buf.getMappedRange()).set(data);
    }
    buf.unmap();
    return buf;
  }

  /** Create a GPU buffer for intermediate computation. */
  private createGPUBuffer(byteSize: number, label: string): GPUBuffer {
    return this.device!.createBuffer({
      size: Math.max(byteSize, 4),  // min 4 bytes
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      label,
    });
  }

  // =========================================================================
  // Read back GPU buffer to CPU
  // =========================================================================

  private async readBuffer(src: GPUBuffer, byteSize: number): Promise<ArrayBuffer> {
    const d = this.device!;
    const staging = d.createBuffer({
      size: byteSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = d.createCommandEncoder();
    encoder.copyBufferToBuffer(src, 0, staging, 0, byteSize);
    d.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const data = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();
    return data;
  }

  private async readU32(src: GPUBuffer): Promise<number> {
    const data = await this.readBuffer(src, 4);
    return new Uint32Array(data)[0];
  }

  private async readF32Array(src: GPUBuffer, count: number): Promise<Float32Array> {
    const data = await this.readBuffer(src, count * 4);
    return new Float32Array(data);
  }

  private async readU32Array(src: GPUBuffer, count: number): Promise<Uint32Array> {
    const data = await this.readBuffer(src, count * 4);
    return new Uint32Array(data);
  }

  // =========================================================================
  // Public API
  // =========================================================================

  /**
   * Check if the engine is ready for inference.
   */
  get isReady(): boolean {
    return this.device !== null && this.modelBuffers !== null && this.pipelines !== null;
  }

  /**
   * Debug: read the first N values from a named work buffer.
   * Returns a Float32Array for inspection.
   */
  async debugRead(bufferName: string, count: number = 16): Promise<Float32Array> {
    const W = this.workBuffers!;
    const buf = (W as any)[bufferName] as GPUBuffer | undefined;
    if (!buf) throw new Error(`Unknown buffer: ${bufferName}. Available: ${Object.keys(W).join(', ')}`);
    return this.readF32Array(buf, count);
  }

  /**
   * Debug: run one backbone step and return intermediate activations.
   */
  async debugBackboneStep(tokenId: number): Promise<{
    hidden: Float32Array;
    normed: Float32Array;
    logits_first16: Float32Array;
    logits_max: number;
    argmax: number;
  }> {
    const encoder = this.device!.createCommandEncoder();
    this.backboneStep(encoder, tokenId);
    this.device!.queue.submit([encoder.finish()]);
    await this.device!.queue.onSubmittedWorkDone();
    this.position++;

    const W = this.workBuffers!;
    const hidden = await this.readF32Array(W.hidden, 16);
    const normed = await this.readF32Array(W.normed, 16);
    const logits_first16 = await this.readF32Array(W.logits, 16);
    const argmax = await this.readU32(W.argmax_result);

    // Find max logit
    const allLogits = await this.readF32Array(W.logits, 1024); // first 1024
    let maxLogit = -Infinity;
    for (let i = 0; i < allLogits.length; i++) {
      if (allLogits[i] > maxLogit) maxLogit = allLogits[i];
    }

    return { hidden, normed, logits_first16, logits_max: maxLogit, argmax };
  }

  /**
   * Debug: run backbone step layer-by-layer, reading activations after each layer.
   * Used for activation matching against reference .npy files.
   * Returns per-layer activations for comparison.
   */
  async debugBackboneLayerByLayer(tokenId: number): Promise<{
    embed: Float32Array;
    layers: Array<{
      attn_norm: Float32Array;
      attn_out: Float32Array;  // hidden state AFTER attention + residual
      ffn_norm: Float32Array;
      ffn_out: Float32Array;   // hidden state AFTER FFN + residual
    }>;
    final_norm: Float32Array;
    hidden: Float32Array;
  }> {
    const P = this.pipelines!;
    const W = this.workBuffers!;
    const M = this.modelBuffers!;
    const bb = this.config.backbone;
    const pos = this.position;
    const dim = bb.dim;
    let pass: GPUComputePassEncoder;

    // --- Embedding lookup ---
    {
      const encoder = this.device!.createCommandEncoder();
      pass = encoder.beginComputePass({ label: `debug_embed` });
      const embedParams = this.packUniform([{ u: tokenId }, { u: dim }]);
      this.dispatch(pass, P.embeddingLookup,
        [M.tok_embeddings, W.hidden, embedParams],
        [cdiv(dim, 256)]);
      pass.end();
      this.device!.queue.submit([encoder.finish()]);
      await this.device!.queue.onSubmittedWorkDone();
    }
    const embed = await this.readF32Array(W.hidden, dim);

    // --- Per-layer ---
    const layers: Array<{
      attn_norm: Float32Array;
      attn_out: Float32Array;
      ffn_norm: Float32Array;
      ffn_out: Float32Array;
    }> = [];

    for (let layer = 0; layer < bb.n_layers; layer++) {
      const L = M.backbone_layers[layer];
      const kv = this.kvCaches[layer];

      // Run attention half
      {
        const encoder = this.device!.createCommandEncoder();

        // Save residual + RMS norm
        pass = encoder.beginComputePass({ label: `debug_l${layer}_attn_prep` });
        const copyParams = this.packUniform([{ u: dim }]);
        this.dispatch(pass, P.copyBuffer, [W.hidden, W.residual, copyParams], [cdiv(dim, 256)]);
        const normParams = this.packUniform([{ u: dim }, { f: bb.norm_eps }]);
        this.dispatch(pass, P.rmsNorm, [W.hidden, L.attn_norm, W.normed, normParams], [1]);
        pass.end();

        this.device!.queue.submit([encoder.finish()]);
        await this.device!.queue.onSubmittedWorkDone();
      }
      const attn_norm = await this.readF32Array(W.normed, dim);

      // Q/K/V projections
      {
        const encoder = this.device!.createCommandEncoder();
        pass = encoder.beginComputePass({ label: `debug_l${layer}_qkv` });
        const qParams = this.packUniform([{ u: bb.n_heads * bb.head_dim }, { u: dim }]);
        this.dispatch(pass, P.matvecF16, [L.wq, W.normed, W.q, qParams], [bb.n_heads * bb.head_dim]);
        const kParams = this.packUniform([{ u: bb.n_kv_heads * bb.head_dim }, { u: dim }]);
        this.dispatch(pass, P.matvecF16, [L.wk, W.normed, W.k, kParams], [bb.n_kv_heads * bb.head_dim]);
        this.dispatch(pass, P.matvecF16, [L.wv, W.normed, W.v, kParams], [bb.n_kv_heads * bb.head_dim]);
        pass.end();

        // RoPE + KV cache + Attention
        pass = encoder.beginComputePass({ label: `debug_l${layer}_rope_attn` });
        const ropeQParams = this.packUniform([{ u: bb.head_dim }, { u: pos }, { u: bb.n_heads }, { f: bb.rope_theta }]);
        this.dispatch(pass, P.rope, [W.q, ropeQParams], [cdiv(bb.n_heads * bb.head_dim / 2, 64)]);
        const ropeKParams = this.packUniform([{ u: bb.head_dim }, { u: pos }, { u: bb.n_kv_heads }, { f: bb.rope_theta }]);
        this.dispatch(pass, P.rope, [W.k, ropeKParams], [cdiv(bb.n_kv_heads * bb.head_dim / 2, 64)]);
        const kvWriteParams = this.packUniform([{ u: pos }, { u: bb.n_kv_heads * bb.head_dim }]);
        this.dispatch(pass, P.kvCacheWrite, [W.k, W.v, kv.k, kv.v, kvWriteParams], [cdiv(bb.n_kv_heads * bb.head_dim, 256)]);
        const seqLen = pos + 1;
        const kvRepeat = bb.n_heads / bb.n_kv_heads;
        const scoreParams = this.packUniform([{ u: bb.n_heads }, { u: bb.n_kv_heads }, { u: bb.head_dim }, { u: seqLen }, { u: kvRepeat }]);
        this.dispatch(pass, P.attnScore, [W.q, kv.k, W.scores, scoreParams], [cdiv(bb.n_heads * seqLen, 64)]);
        pass.end();

        pass = encoder.beginComputePass({ label: `debug_l${layer}_attn_out` });
        const softmaxParams = this.packUniform([{ u: bb.n_heads }, { u: seqLen }]);
        this.dispatch(pass, P.softmax, [W.scores, softmaxParams], [bb.n_heads]);
        const valParams = this.packUniform([{ u: bb.n_heads }, { u: bb.n_kv_heads }, { u: bb.head_dim }, { u: seqLen }, { u: kvRepeat }]);
        this.dispatch(pass, P.attnValue, [W.scores, kv.v, W.attn_out, valParams], [cdiv(bb.n_heads * bb.head_dim, 128)]);
        pass.end();

        // Output projection + residual
        pass = encoder.beginComputePass({ label: `debug_l${layer}_wo` });
        const woParams = this.packUniform([{ u: dim }, { u: bb.n_heads * bb.head_dim }]);
        this.dispatch(pass, P.matvecF16, [L.wo, W.attn_out, W.hidden, woParams], [dim]);
        pass.end();

        pass = encoder.beginComputePass({ label: `debug_l${layer}_res1` });
        const addParams = this.packUniform([{ u: dim }]);
        this.dispatch(pass, P.addInPlace, [W.hidden, W.residual, addParams], [cdiv(dim, 256)]);
        pass.end();

        this.device!.queue.submit([encoder.finish()]);
        await this.device!.queue.onSubmittedWorkDone();
      }

      // Read attention output (Note: reference saves attn_out as just the Wo @ V part,
      // not the residual-added version. Check reference script for exact meaning.)
      // Actually, the reference saves attn_out = attention(...) which is Wo @ attn_weighted_v,
      // before residual add. But we've already added residual. So we read the hidden state
      // which is embed + attn_out at this point.
      // The reference saves: attn_out (no residual), then h = h + attn_out, then later ffn_out = h (with residual)
      // So attn_out from reference = just the Wo @ V output, not the residual sum.
      const attn_out_hidden = await this.readF32Array(W.hidden, dim);

      // Run FFN half
      {
        const encoder = this.device!.createCommandEncoder();

        // Save residual + FFN norm
        pass = encoder.beginComputePass({ label: `debug_l${layer}_ffn_prep` });
        const copyParams = this.packUniform([{ u: dim }]);
        this.dispatch(pass, P.copyBuffer, [W.hidden, W.residual, copyParams], [cdiv(dim, 256)]);
        const normParams = this.packUniform([{ u: dim }, { f: bb.norm_eps }]);
        this.dispatch(pass, P.rmsNorm, [W.hidden, L.ffn_norm, W.normed, normParams], [1]);
        pass.end();

        this.device!.queue.submit([encoder.finish()]);
        await this.device!.queue.onSubmittedWorkDone();
      }
      const ffn_norm = await this.readF32Array(W.normed, dim);

      // Gate/Up/SwiGLU/Down + residual
      {
        const encoder = this.device!.createCommandEncoder();

        pass = encoder.beginComputePass({ label: `debug_l${layer}_ffn` });
        const gateParams = this.packUniform([{ u: bb.hidden_dim }, { u: dim }]);
        this.dispatch(pass, P.matvecF16, [L.w1, W.normed, W.gate, gateParams], [bb.hidden_dim]);
        this.dispatch(pass, P.matvecF16, [L.w3, W.normed, W.up, gateParams], [bb.hidden_dim]);
        pass.end();

        pass = encoder.beginComputePass({ label: `debug_l${layer}_ffn_out` });
        const swiParams = this.packUniform([{ u: bb.hidden_dim }]);
        this.dispatch(pass, P.swiGLU, [W.gate, W.up, swiParams], [cdiv(bb.hidden_dim, 256)]);
        const downParams = this.packUniform([{ u: dim }, { u: bb.hidden_dim }]);
        this.dispatch(pass, P.matvecF16, [L.w2, W.gate, W.hidden, downParams], [dim]);
        pass.end();

        pass = encoder.beginComputePass({ label: `debug_l${layer}_res2` });
        const addParams = this.packUniform([{ u: dim }]);
        this.dispatch(pass, P.addInPlace, [W.hidden, W.residual, addParams], [cdiv(dim, 256)]);
        pass.end();

        this.device!.queue.submit([encoder.finish()]);
        await this.device!.queue.onSubmittedWorkDone();
      }
      const ffn_out = await this.readF32Array(W.hidden, dim);

      layers.push({ attn_norm, attn_out: attn_out_hidden, ffn_norm, ffn_out });
    }

    // --- Final norm ---
    {
      const encoder = this.device!.createCommandEncoder();
      pass = encoder.beginComputePass({ label: `debug_final_norm` });
      const normParams = this.packUniform([{ u: dim }, { f: bb.norm_eps }]);
      this.dispatch(pass, P.rmsNorm, [W.hidden, M.final_norm, W.normed, normParams], [1]);
      pass.end();
      this.device!.queue.submit([encoder.finish()]);
      await this.device!.queue.onSubmittedWorkDone();
    }
    const final_norm = await this.readF32Array(W.normed, dim);
    const hidden = await this.readF32Array(W.hidden, dim);

    this.position++;

    return { embed, layers, final_norm, hidden };
  }

  /**
   * Debug: run FM forward pass with deterministic noise and return per-step activations.
   * Must be called after a backbone step (uses normed buffer).
   */
  async debugFMForward(seed: number = 42): Promise<{
    semantic_logits: Float32Array;
    velocities: Float32Array[];
    acoustic_codes: Uint32Array;
    x_final: Float32Array;
  }> {
    const fm = this.config.fm;

    // Generate deterministic noise matching PyTorch reference
    // torch.manual_seed(42) → randn(1, 36) uses Box-Muller internally
    // We need the exact same noise. Pre-compute from Python and pass in.
    // For now, use the engine's normal fmForward but read activations.
    const encoder = this.device!.createCommandEncoder();
    this.fmForward(encoder);
    this.device!.queue.submit([encoder.finish()]);
    await this.device!.queue.onSubmittedWorkDone();

    const W = this.workBuffers!;
    const semantic_logits = await this.readF32Array(W.semantic_logits, fm.semantic_vocab);
    const acoustic_codes = await this.readU32Array(W.acoustic_codes, fm.n_acoustic_out);
    const x_final = await this.readF32Array(W.x_t, fm.n_acoustic_out);

    // Can't easily read per-step velocities since they're overwritten.
    // Return empty array for now — backbone matching is more critical.
    return { semantic_logits, velocities: [], acoustic_codes, x_final };
  }

  /**
   * Reset state for a new generation.
   */
  reset(): void {
    this.position = 0;
    // KV caches are implicitly reset since position tracking starts at 0.
    // Stale data at positions > 0 is never read because seqLen = position + 1.
  }

  /**
   * Run a single backbone step and return the predicted token ID.
   * Advances the position counter.
   */
  async backboneStepAndRead(tokenId: number, useAudioEmbedding: boolean = false): Promise<number> {
    const encoder = this.device!.createCommandEncoder();
    this.backboneStep(encoder, tokenId, useAudioEmbedding);
    this.device!.queue.submit([encoder.finish()]);
    await this.device!.queue.onSubmittedWorkDone();

    const nextToken = await this.readU32(this.workBuffers!.argmax_result);
    this.position++;
    return nextToken;
  }

  /**
   * Debug: Run backbone step, read normed hidden, and advance position.
   * Supports both text and audio embeddings.
   */
  async debugBackboneStepFull(tokenId: number, useAudioEmbedding: boolean = false): Promise<Float32Array> {
    const encoder = this.device!.createCommandEncoder();
    this.backboneStep(encoder, tokenId, useAudioEmbedding);
    this.device!.queue.submit([encoder.finish()]);
    await this.device!.queue.onSubmittedWorkDone();

    const normed = await this.readF32Array(this.workBuffers!.normed, this.config.backbone.dim);
    this.position++;
    return normed;
  }

  /**
   * Debug: Run FM forward with optional deterministic noise.
   * Returns semantic logits, acoustic codes, and x_final.
   */
  async debugFMStep(debugNoise?: Float32Array): Promise<{
    semantic_logits: Float32Array;
    acoustic_codes: Uint32Array;
    x_final: Float32Array;
  }> {
    const fm = this.config.fm;
    const encoder = this.device!.createCommandEncoder();
    this.fmForward(encoder, debugNoise);
    this.device!.queue.submit([encoder.finish()]);
    await this.device!.queue.onSubmittedWorkDone();

    const W = this.workBuffers!;
    return {
      semantic_logits: await this.readF32Array(W.semantic_logits, fm.semantic_vocab),
      acoustic_codes: await this.readU32Array(W.acoustic_codes, fm.n_acoustic_out),
      x_final: await this.readF32Array(W.x_t, fm.n_acoustic_out),
    };
  }

  /**
   * Run the FM transformer on the current hidden state.
   * Returns the 36 acoustic codes for one frame.
   */
  async fmStepAndRead(): Promise<Uint32Array> {
    const encoder = this.device!.createCommandEncoder();
    this.fmForward(encoder);
    this.device!.queue.submit([encoder.finish()]);
    await this.device!.queue.onSubmittedWorkDone();

    return this.readU32Array(
      this.workBuffers!.acoustic_codes,
      this.config.fm.n_acoustic_out,
    );
  }

  /**
   * Full TTS generation pipeline.
   *
   * Takes a pre-tokenized prompt (from TekkenTokenizer.buildTTSPrompt) and
   * generates audio frames until EOS or maxFrames.
   *
   * Pipeline per frame:
   * 1. Backbone autoregressive step → semantic code (argmax of semantic_logits)
   * 2. FM 7-step Euler ODE → 36 acoustic codes
   * 3. Collect codes for batch codec decoding at the end
   *
   * @param tokens - Full token sequence: BOS, BEGIN_AUDIO, AUDIO×N, TEXT_TO_AUDIO, text, AUDIO_TO_TEXT, BEGIN_AUDIO
   * @param audioTokenStart - Index where voice audio tokens start
   * @param audioTokenCount - Number of voice audio token positions
   * @param voiceEmbeddings - Pre-loaded voice embeddings [audioTokenCount, dim] as F32
   * @param maxFrames - Maximum number of audio frames to generate (default 500 = 40s)
   * @param onFrame - Optional callback per frame for streaming progress
   */
  async generate(
    tokens: number[],
    audioTokenStart: number,
    audioTokenCount: number,
    voiceEmbeddings: Float32Array | null,
    maxFrames: number = 500,
    onFrame?: (frame: number, semanticCode: number, acousticCodes: Uint32Array) => void,
  ): Promise<TTSResult> {
    if (!this.isReady) throw new Error('Engine not initialized. Call init() and loadWeights() first.');

    this.reset();
    const t0 = performance.now();

    // --- Phase 1: Prefill backbone with prompt tokens ---
    // Process all tokens: BOS, BEGIN_AUDIO, AUDIO×N, TEXT_TO_AUDIO, text, AUDIO_TO_TEXT, BEGIN_AUDIO
    // Voice embedding tokens ([AUDIO] x N) get REPLACED by voice embeddings.

    // Upload voice embeddings to GPU as per-position F32 buffers
    const voiceEmbedBuffers: GPUBuffer[] = [];
    if (voiceEmbeddings && audioTokenCount > 0) {
      const dim = this.config.backbone.dim;
      for (let j = 0; j < audioTokenCount; j++) {
        const buf = this.device!.createBuffer({
          size: dim * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
          mappedAtCreation: true,
        });
        new Float32Array(buf.getMappedRange()).set(
          voiceEmbeddings.subarray(j * dim, (j + 1) * dim)
        );
        buf.unmap();
        voiceEmbedBuffers.push(buf);
      }
    }

    for (let i = 0; i < tokens.length; i++) {
      const tokenId = tokens[i];
      const encoder = this.device!.createCommandEncoder();

      if (i >= audioTokenStart && i < audioTokenStart + audioTokenCount && voiceEmbedBuffers.length > 0) {
        // Voice embedding token — use pre-loaded voice embedding
        const voiceIdx = i - audioTokenStart;
        this.backboneStep(encoder, tokenId, false, voiceEmbedBuffers[voiceIdx]);
      } else {
        this.backboneStep(encoder, tokenId);
      }

      this.device!.queue.submit([encoder.finish()]);
      await this.device!.queue.onSubmittedWorkDone();
      this.position++;
    }

    const tPrefill = performance.now();

    // --- AUDIO token kick: feed AUDIO(24) through TEXT embedding to get first hidden state ---
    // Matches MLX/vLLM-Omni: after prefill, one backbone step with AUDIO token produces
    // the hidden state for frame 0.
    {
      const encoder = this.device!.createCommandEncoder();
      this.backboneStep(encoder, 24 /* AUDIO token */, false /* text embedding */);
      this.device!.queue.submit([encoder.finish()]);
      await this.device!.queue.onSubmittedWorkDone();
      this.position++;
    }

    // --- Phase 2: Autoregressive generation ---
    const semanticCodes: number[] = [];
    const acousticCodes: number[][] = [];
    const bb = this.config.backbone;
    const fm = this.config.fm;
    const P = this.pipelines!;
    const W = this.workBuffers!;
    const M = this.modelBuffers!;

    for (let frame = 0; frame < maxFrames; frame++) {
      if (frame > 0) {
        // Multi-codebook embedding: sum all 37 codebook embeddings (matches MLX/vLLM-Omni)
        // The semantic code is in W.semantic_argmax, acoustic codes in W.acoustic_codes
        const encoder = this.device!.createCommandEncoder();

        // Compute multi-codebook sum → W.hidden
        const pass = encoder.beginComputePass({ label: `multiCBEmbed_frame${frame}` });
        const mcParams = this.packUniform([
          { u: bb.dim },
          { u: 8194 },  // acoustic_base: semantic_codebook_size(8192) + 2 specials
          { u: 23 },    // acoustic_stride: acoustic_codebook_size(21) + 2 specials
          { u: 36 },    // n_acoustic codebooks
        ]);
        this.dispatch(pass, P.multiCodebookEmbed,
          [M.audio_embeddings, W.semantic_argmax, W.acoustic_codes, W.hidden, mcParams],
          [cdiv(bb.dim, 256)]);
        pass.end();

        // Run backbone step with precomputed multi-codebook embedding
        // We need to pass W.hidden as precomputedEmbedding — but backboneStep overwrites W.hidden
        // with the embedding lookup first. Since we already set W.hidden, use precomputedEmbedding=W.hidden.
        // Actually, backboneStep first does embedding lookup writing to W.hidden, THEN checks
        // precomputedEmbedding. So we need a temp buffer, or restructure.
        // Simplest: use backboneStep with precomputedEmbedding — it copies from the given buffer
        // to W.hidden, overwriting the lookup. So we need a separate buffer.

        // Copy W.hidden → temp buffer, then backboneStep with precomputed from temp
        // Actually, let's just do the backbone layers directly in the same encoder.
        // The backboneStep first writes embedding to W.hidden, then overwrites with precomputed.
        // So we can pass W.hidden as its own precomputed. But that's a race condition.
        // Better: write multi-codebook to a dedicated buffer.

        // Use W.fm_gate as a temp buffer (it's dim-sized and not in use during backbone)
        const copyPass = encoder.beginComputePass({ label: `mcb_copy_frame${frame}` });
        const copyP = this.packUniform([{ u: bb.dim }]);
        this.dispatch(copyPass, P.copyBuffer, [W.hidden, W.fm_gate, copyP], [cdiv(bb.dim, 256)]);
        copyPass.end();

        // Now run backbone step with the precomputed embedding from fm_gate
        this.backboneStep(encoder, 0 /* dummy token, ignored */, false, W.fm_gate);
        this.device!.queue.submit([encoder.finish()]);
        await this.device!.queue.onSubmittedWorkDone();
        this.position++;
      }

      // FM produces semantic logits + acoustic codes from backbone hidden state
      const fmEncoder = this.device!.createCommandEncoder();
      this.fmForward(fmEncoder);
      this.device!.queue.submit([fmEncoder.finish()]);
      await this.device!.queue.onSubmittedWorkDone();

      // Read semantic logits and apply masking
      const semLogits = await this.readF32Array(
        W.semantic_logits,
        fm.semantic_vocab,
      );

      // Semantic logit masking (matches MLX/vLLM-Omni):
      // - EMPTY_AUDIO (index 0) → -Infinity
      // - Padding beyond valid semantic vocab (indices >= 8194) → -Infinity
      semLogits[0] = -Infinity;
      const validSemanticSize = 8194; // 8192 codebook + 2 specials
      for (let i = validSemanticSize; i < semLogits.length; i++) {
        semLogits[i] = -Infinity;
      }

      const semanticCode = sampleTopP(semLogits, 0.9, 0.8);

      // EOS: semantic code 0 (EMPTY_AUDIO) or 1 (END_AUDIO)
      if (semanticCode <= 1) {
        break;
      }
      semanticCodes.push(semanticCode);

      // Write semantic code to GPU buffer for multi-codebook embedding in next iteration
      const semCodeBuf = new Uint32Array([semanticCode]);
      this.device!.queue.writeBuffer(W.semantic_argmax, 0, semCodeBuf);

      // Read acoustic codes (already computed by fmForward above, already on GPU for next step)
      const frameCodes = await this.readU32Array(
        W.acoustic_codes,
        fm.n_acoustic_out,
      );
      acousticCodes.push(Array.from(frameCodes));

      onFrame?.(frame, semanticCode, frameCodes);
    }

    const tGenerate = performance.now();

    // --- Phase 3: Codec decode (batch all frames) ---
    let audio: Float32Array;
    if (semanticCodes.length > 0) {
      const semCodesArr = new Uint32Array(semanticCodes);
      const acCodesFlat = new Uint32Array(acousticCodes.flat());
      audio = await this.codecDecode(semCodesArr, acCodesFlat);
    } else {
      audio = new Float32Array(0);
    }

    const tCodec = performance.now();

    return {
      semanticCodes,
      acousticCodes,
      audio,
      stats: {
        backboneMs: tPrefill - t0,
        fmMs: tGenerate - tPrefill,
        codecMs: tCodec - tGenerate,
        totalMs: tCodec - t0,
        framesGenerated: semanticCodes.length,
      },
    };
  }

  /**
   * Destroy all GPU resources.
   */
  destroy(): void {
    if (this.workBuffers) {
      for (const buf of Object.values(this.workBuffers)) {
        (buf as GPUBuffer).destroy();
      }
    }
    for (const cache of this.kvCaches) {
      cache.k.destroy();
      cache.v.destroy();
    }
    this.device?.destroy();
  }
}
