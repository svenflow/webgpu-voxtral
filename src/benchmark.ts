/**
 * Phase 0a: Go/No-Go Matmul Benchmark
 *
 * Tests the critical matrix multiply sizes for Voxtral TTS to determine
 * if real-time audio generation is feasible in the browser via WebGPU.
 *
 * Target: 12.5 frames/sec (80ms per frame budget)
 * Per frame: 26 backbone layers + 3 FM layers × 16 NFEs + codec decode
 *
 * If the bottleneck matmuls can't hit the timing budget, the whole project is dead.
 */

import { MatmulTestCase, defaultConfig } from './types.js';
import { matvecF32 } from './benchmark-shaders.js';

const WARMUP_ITERS = 5;
const BENCH_ITERS = 50;

/** All critical matmul sizes in Voxtral TTS */
function getTestCases(): MatmulTestCase[] {
  const cfg = defaultConfig;
  return [
    // === BACKBONE (per token, 26 layers) ===
    // QKV projection: [3072] → [4096+1024+1024] = [6144]
    // Q: n_heads * head_dim = 32*128 = 4096, K/V: n_kv_heads * head_dim = 8*128 = 1024
    {
      name: 'backbone.qkv_proj',
      M: cfg.backbone.n_heads * cfg.backbone.head_dim + 2 * cfg.backbone.n_kv_heads * cfg.backbone.head_dim,
      K: cfg.backbone.dim,
      N: 1,
      count: cfg.backbone.n_layers,
      component: 'backbone',
    },
    // Output projection: [4096] → [3072]
    {
      name: 'backbone.o_proj',
      M: cfg.backbone.dim,
      K: cfg.backbone.n_heads * cfg.backbone.head_dim,
      N: 1,
      count: cfg.backbone.n_layers,
      component: 'backbone',
    },
    // FFN gate+up: [3072] → [9216] (done twice: gate and up)
    {
      name: 'backbone.ffn_gate_up',
      M: cfg.backbone.hidden_dim * 2,  // gate + up fused
      K: cfg.backbone.dim,
      N: 1,
      count: cfg.backbone.n_layers,
      component: 'backbone',
    },
    // FFN down: [9216] → [3072]
    {
      name: 'backbone.ffn_down',
      M: cfg.backbone.dim,
      K: cfg.backbone.hidden_dim,
      N: 1,
      count: cfg.backbone.n_layers,
      component: 'backbone',
    },
    // LM head (tied embeddings): [3072] → [131072]
    {
      name: 'backbone.lm_head',
      M: cfg.backbone.vocab_size,
      K: cfg.backbone.dim,
      N: 1,
      count: 1,
      component: 'backbone',
    },

    // === FM TRANSFORMER (per frame, 3 layers × 16 NFEs = 48 passes) ===
    {
      name: 'fm.qkv_proj',
      M: cfg.fm.n_heads * cfg.fm.head_dim + 2 * cfg.fm.n_kv_heads * cfg.fm.head_dim,
      K: cfg.fm.dim,
      N: 1,
      count: cfg.fm.n_layers * cfg.fm.nfe * 2,  // ×2 for CFG (cond + uncond batched)
      component: 'fm',
    },
    {
      name: 'fm.o_proj',
      M: cfg.fm.dim,
      K: cfg.fm.n_heads * cfg.fm.head_dim,
      N: 1,
      count: cfg.fm.n_layers * cfg.fm.nfe * 2,  // ×2 for CFG (cond + uncond batched)
      component: 'fm',
    },
    {
      name: 'fm.ffn_gate_up',
      M: cfg.fm.hidden_dim * 2,
      K: cfg.fm.dim,
      N: 1,
      count: cfg.fm.n_layers * cfg.fm.nfe * 2,  // ×2 for CFG (cond + uncond batched)
      component: 'fm',
    },
    {
      name: 'fm.ffn_down',
      M: cfg.fm.dim,
      K: cfg.fm.hidden_dim,
      N: 1,
      count: cfg.fm.n_layers * cfg.fm.nfe * 2,  // ×2 for CFG (cond + uncond batched)
      component: 'fm',
    },

    // === CODEC DECODER (per frame, 4 stages × 2 layers = 8 layers) ===
    {
      name: 'codec.qkv_proj',
      M: cfg.codec.n_heads * cfg.codec.head_dim + 2 * cfg.codec.n_kv_heads * cfg.codec.head_dim,
      K: cfg.codec.dim,
      N: 1,
      count: cfg.codec.decoder_stages * cfg.codec.decoder_layers_per_stage,
      component: 'codec',
    },
    {
      name: 'codec.ffn_gate_up',
      M: cfg.codec.hidden_dim * 2,
      K: cfg.codec.dim,
      N: 1,
      count: cfg.codec.decoder_stages * cfg.codec.decoder_layers_per_stage,
      component: 'codec',
    },
    {
      name: 'codec.ffn_down',
      M: cfg.codec.dim,
      K: cfg.codec.hidden_dim,
      N: 1,
      count: cfg.codec.decoder_stages * cfg.codec.decoder_layers_per_stage,
      component: 'codec',
    },
  ];
}

interface BenchResult {
  name: string;
  M: number;
  K: number;
  avgMs: number;
  medianMs: number;
  minMs: number;
  count: number;
  totalMs: number;  // avgMs * count
  component: string;
  gflops: number;
}

async function createDevice(): Promise<GPUDevice> {
  if (!navigator.gpu) {
    throw new Error('WebGPU not supported in this browser');
  }
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
  });
  if (!adapter) {
    throw new Error('No WebGPU adapter found');
  }

  // Check feature support
  const hasF16 = adapter.features.has('shader-f16');
  const hasTimestamp = adapter.features.has('timestamp-query');

  const features: GPUFeatureName[] = [];
  if (hasF16) features.push('shader-f16' as GPUFeatureName);
  if (hasTimestamp) features.push('timestamp-query' as GPUFeatureName);

  const device = await adapter.requestDevice({
    requiredFeatures: features,
    requiredLimits: {
      maxBufferSize: 1024 * 1024 * 1024,  // 1GB
      maxStorageBufferBindingSize: 512 * 1024 * 1024,
    },
  });

  return device;
}

function createMatvecPipeline(
  device: GPUDevice,
  shaderCode: string,
): GPUComputePipeline {
  const module = device.createShaderModule({ code: shaderCode });
  return device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'main' },
  });
}

async function benchmarkMatvec(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  M: number,
  K: number,
  useF16: boolean,
): Promise<number[]> {
  const elemSize = useF16 ? 2 : 4;
  const bufSize = M * K * elemSize;

  // Check buffer size limits
  if (bufSize > device.limits.maxStorageBufferBindingSize) {
    // For very large matmuls (like lm_head 131072×3072), skip if too big
    return [-1]; // sentinel for "skipped"
  }

  // Create buffers
  const matrixBuf = device.createBuffer({
    size: bufSize,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  if (useF16) {
    const arr = new Uint16Array(matrixBuf.getMappedRange());
    for (let i = 0; i < arr.length; i++) {
      arr[i] = float32ToFloat16((Math.random() - 0.5) * 0.1);
    }
  } else {
    const arr = new Float32Array(matrixBuf.getMappedRange());
    for (let i = 0; i < arr.length; i++) {
      arr[i] = (Math.random() - 0.5) * 0.1;
    }
  }
  matrixBuf.unmap();

  const vecBuf = device.createBuffer({
    size: K * elemSize,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  if (useF16) {
    const arr = new Uint16Array(vecBuf.getMappedRange());
    for (let i = 0; i < arr.length; i++) {
      arr[i] = float32ToFloat16((Math.random() - 0.5) * 0.1);
    }
  } else {
    const arr = new Float32Array(vecBuf.getMappedRange());
    for (let i = 0; i < arr.length; i++) {
      arr[i] = (Math.random() - 0.5) * 0.1;
    }
  }
  vecBuf.unmap();

  const outBuf = device.createBuffer({
    size: M * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const paramBuf = device.createBuffer({
    size: 8,
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
  });
  new Uint32Array(paramBuf.getMappedRange()).set([M, K]);
  paramBuf.unmap();

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: matrixBuf } },
      { binding: 1, resource: { buffer: vecBuf } },
      { binding: 2, resource: { buffer: outBuf } },
      { binding: 3, resource: { buffer: paramBuf } },
    ],
  });

  const times: number[] = [];

  // Warmup: submit multiple dispatches and wait
  for (let i = 0; i < WARMUP_ITERS; i++) {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(M);
    pass.end();
    device.queue.submit([enc.finish()]);
  }
  await device.queue.onSubmittedWorkDone();

  // Benchmark using CPU-side timing with single dispatch per measurement
  // Submit one dispatch, wait for completion, measure wall time
  // This includes driver overhead but is reliable across all GPUs
  for (let i = 0; i < BENCH_ITERS; i++) {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(M);
    pass.end();

    const t0 = performance.now();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    const t1 = performance.now();
    times.push(t1 - t0);
  }

  // Cleanup
  matrixBuf.destroy();
  vecBuf.destroy();
  outBuf.destroy();
  paramBuf.destroy();

  return times;
}

function float32ToFloat16(val: number): number {
  const buf = new ArrayBuffer(4);
  new Float32Array(buf)[0] = val;
  const bits = new Uint32Array(buf)[0];
  const sign = (bits >> 16) & 0x8000;
  const exp = ((bits >> 23) & 0xff) - 127 + 15;
  const frac = (bits >> 13) & 0x3ff;
  if (exp <= 0) return sign;
  if (exp >= 31) return sign | 0x7c00;
  return sign | (exp << 10) | frac;
}

function median(arr: number[]): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export interface BenchmarkReport {
  device: string;
  hasF16: boolean;
  hasTimestamp: boolean;
  results: BenchResult[];
  summary: {
    backbone_total_ms: number;
    fm_total_ms: number;
    codec_total_ms: number;
    total_per_frame_ms: number;
    target_ms: number;
    feasible: boolean;
    realtime_factor: number;
  };
}

export async function runBenchmark(
  onProgress?: (msg: string) => void,
): Promise<BenchmarkReport> {
  const log = onProgress || console.log;

  log('Initializing WebGPU...');
  const device = await createDevice();
  const adapterInfo = device.adapterInfo;
  const deviceName = adapterInfo
    ? `${adapterInfo.vendor} ${adapterInfo.architecture} ${adapterInfo.device}`
    : 'Unknown GPU';
  const hasF16 = device.features.has('shader-f16');
  const hasTimestamp = device.features.has('timestamp-query');

  log(`GPU: ${deviceName}`);
  log(`F16 support: ${hasF16}`);
  log(`Timestamp queries: ${hasTimestamp}`);

  // Create pipeline
  const f32Pipeline = createMatvecPipeline(device, matvecF32);

  const testCases = getTestCases();
  const results: BenchResult[] = [];

  for (const tc of testCases) {
    log(`\nBenchmarking ${tc.name} [${tc.M} × ${tc.K}] × ${tc.count}...`);

    const times = await benchmarkMatvec(device, f32Pipeline, tc.M, tc.K, false);

    if (times[0] === -1) {
      log(`  SKIPPED — buffer too large (${((tc.M * tc.K * 4) / 1e9).toFixed(2)} GB)`);
      // Estimate from smaller matmuls later
      results.push({
        name: tc.name, M: tc.M, K: tc.K,
        avgMs: -1, medianMs: -1, minMs: -1,
        count: tc.count, totalMs: -1,
        component: tc.component, gflops: 0,
      });
      continue;
    }

    const avgMs = times.reduce((a, b) => a + b, 0) / times.length;
    const medMs = median(times);
    const minMs = Math.min(...times);
    const flops = 2 * tc.M * tc.K;
    const gflops = (flops / (medMs / 1000)) / 1e9;

    const result: BenchResult = {
      name: tc.name,
      M: tc.M,
      K: tc.K,
      avgMs,
      medianMs: medMs,
      minMs,
      count: tc.count,
      totalMs: medMs * tc.count,
      component: tc.component,
      gflops,
    };
    results.push(result);

    log(`  median: ${medMs.toFixed(3)}ms | total (×${tc.count}): ${result.totalMs.toFixed(2)}ms | ${gflops.toFixed(1)} GFLOPS`);
  }

  // Compute summary
  const backboneMs = results
    .filter(r => r.component === 'backbone')
    .reduce((sum, r) => sum + r.totalMs, 0);
  const fmMs = results
    .filter(r => r.component === 'fm')
    .reduce((sum, r) => sum + r.totalMs, 0);
  const codecMs = results
    .filter(r => r.component === 'codec')
    .reduce((sum, r) => sum + r.totalMs, 0);

  // Per audio frame: backbone generates 1 semantic token, FM runs 16 NFEs, codec decodes
  // But backbone is autoregressive per-token, so it runs once per frame
  const totalPerFrameMs = backboneMs + fmMs + codecMs;
  const targetMs = 80; // 12.5 fps = 80ms per frame
  const feasible = totalPerFrameMs < targetMs;
  const realtimeFactor = targetMs / totalPerFrameMs;

  const summary = {
    backbone_total_ms: backboneMs,
    fm_total_ms: fmMs,
    codec_total_ms: codecMs,
    total_per_frame_ms: totalPerFrameMs,
    target_ms: targetMs,
    feasible,
    realtime_factor: realtimeFactor,
  };

  log('\n========================================');
  log('VOXTRAL TTS — PHASE 0a GO/NO-GO RESULTS');
  log('========================================');
  log(`Backbone (26 layers):  ${backboneMs.toFixed(2)}ms`);
  log(`FM (3 layers × 16):   ${fmMs.toFixed(2)}ms`);
  log(`Codec (8 layers):     ${codecMs.toFixed(2)}ms`);
  log(`─────────────────────────────────────`);
  log(`Total per frame:      ${totalPerFrameMs.toFixed(2)}ms`);
  log(`Target (12.5 fps):    ${targetMs}ms`);
  log(`Realtime factor:      ${realtimeFactor.toFixed(2)}x`);
  log(`\nVERDICT: ${feasible ? '✅ GO — real-time TTS is feasible!' : '❌ NO-GO — too slow for real-time'}`);

  if (!feasible) {
    log(`\nNote: These are matmul-only times. Real inference adds ~30-50% overhead`);
    log(`for norms, activations, attention, sampling, etc.`);
    log(`\nFor feasibility, we need matmul total < ~55ms (leaving ~25ms for overhead).`);
  }

  device.destroy();

  return { device: deviceName, hasF16, hasTimestamp, results, summary };
}
