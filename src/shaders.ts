/**
 * WGSL compute shaders for Voxtral TTS
 *
 * All shaders operate on F16 weights and F32 activations.
 * Weights are stored as packed u32 (2× f16 per u32).
 * Activations are f32 for numerical stability during computation.
 */

// ============================================================================
// CORE: Matrix-vector multiply (F16 weights × F32 input → F32 output)
// ============================================================================

/**
 * F16 matrix-vector multiply: y[m] = sum_k(A[m,k] * x[k])
 * A is [M, K] stored as packed f16 (u32 pairs), x is f32, output y is f32.
 * This is THE critical shader — every linear layer uses this.
 *
 * Each workgroup handles one output row. 256 threads cooperate via
 * shared memory reduction over the K dimension.
 */
export const matvecF16 = /* wgsl */ `
struct Params {
  M: u32,
  K: u32,
}

@group(0) @binding(0) var<storage, read> matrix: array<u32>;   // packed f16 [M, K/2]
@group(0) @binding(1) var<storage, read> vector: array<f32>;   // [K]
@group(0) @binding(2) var<storage, read_write> output: array<f32>;  // [M]
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let row = wid.x;
  if (row >= params.M) { return; }
  let tid = lid.x;

  let K_packed = params.K / 2u;
  var sum: f32 = 0.0;

  // Each thread accumulates over a strided portion of K
  var pk = tid;
  let rowBase = row * K_packed;
  while (pk < K_packed) {
    let packed = matrix[rowBase + pk];
    let pair = unpack2x16float(packed);
    let k0 = pk * 2u;
    sum += f32(pair.x) * vector[k0] + f32(pair.y) * vector[k0 + 1u];
    pk += WG;
  }

  sdata[tid] = sum;
  workgroupBarrier();

  // Tree reduction
  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { sdata[tid] += sdata[tid + s]; }
    workgroupBarrier();
  }

  if (tid == 0u) {
    output[row] = sdata[0];
  }
}
`;

/**
 * Chunked F16 matvec with row_offset for dispatches exceeding 65535 workgroups.
 * Same as matvecF16 but reads/writes at (row_offset + wid.x) for both matrix and output.
 */
export const matvecF16Chunked = /* wgsl */ `
struct Params {
  M: u32,
  K: u32,
  row_offset: u32,
}

@group(0) @binding(0) var<storage, read> matrix: array<u32>;   // packed f16 [total_M, K/2]
@group(0) @binding(1) var<storage, read> vector: array<f32>;   // [K]
@group(0) @binding(2) var<storage, read_write> output: array<f32>;  // [total_M]
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let localRow = wid.x;
  if (localRow >= params.M) { return; }
  let globalRow = localRow + params.row_offset;
  let tid = lid.x;

  let K_packed = params.K / 2u;
  var sum: f32 = 0.0;

  var pk = tid;
  let rowBase = globalRow * K_packed;
  while (pk < K_packed) {
    let packed = matrix[rowBase + pk];
    let pair = unpack2x16float(packed);
    let k0 = pk * 2u;
    sum += f32(pair.x) * vector[k0] + f32(pair.y) * vector[k0 + 1u];
    pk += WG;
  }

  sdata[tid] = sum;
  workgroupBarrier();

  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { sdata[tid] += sdata[tid + s]; }
    workgroupBarrier();
  }

  if (tid == 0u) {
    output[globalRow] = sdata[0];
  }
}
`;

// ============================================================================
// CORE: RMS Normalization
// ============================================================================

/**
 * RMSNorm: y[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i]
 * Single workgroup reduction over dim.
 */
export const rmsNorm = /* wgsl */ `
struct Params {
  dim: u32,
  eps: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;  // packed f16
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;

  // Compute sum of squares
  var ss: f32 = 0.0;
  var i = tid;
  while (i < params.dim) {
    let v = input[i];
    ss += v * v;
    i += WG;
  }
  sdata[tid] = ss;
  workgroupBarrier();

  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { sdata[tid] += sdata[tid + s]; }
    workgroupBarrier();
  }

  let rms = 1.0 / sqrt(sdata[0] / f32(params.dim) + params.eps);

  // Scale and write output
  i = tid;
  while (i < params.dim) {
    let wPacked = weight[i / 2u];
    let wPair = unpack2x16float(wPacked);
    let w = select(f32(wPair.x), f32(wPair.y), (i & 1u) == 1u);
    output[i] = input[i] * rms * w;
    i += WG;
  }
}
`;

// ============================================================================
// CORE: Embedding Lookup
// ============================================================================

/**
 * Embedding lookup: output[i] = embedding_table[token_id, i]
 * Table is [vocab_size, dim] in packed f16.
 */
export const embeddingLookup = /* wgsl */ `
struct Params {
  token_id: u32,
  dim: u32,
}

@group(0) @binding(0) var<storage, read> table: array<u32>;  // packed f16 [vocab, dim/2]
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) { return; }

  let packedIdx = params.token_id * (params.dim / 2u) + i / 2u;
  let packed = table[packedIdx];
  let pair = unpack2x16float(packed);
  output[i] = select(f32(pair.x), f32(pair.y), (i & 1u) == 1u);
}
`;

// ============================================================================
// CORE: Rotary Position Embedding (RoPE)
// ============================================================================

/**
 * Apply RoPE to Q and K vectors in-place.
 * Interleaved format: pairs (x0, x1) at positions (2i, 2i+1) rotate together.
 * theta = base^(-2i/dim), angle = pos * theta
 */
export const rope = /* wgsl */ `
struct Params {
  dim: u32,       // head_dim (128)
  pos: u32,       // sequence position
  n_heads: u32,   // number of heads to process
  theta: f32,     // rope_theta (1e6 for backbone, 1e4 for FM)
}

@group(0) @binding(0) var<storage, read_write> qk: array<f32>;  // [n_heads, dim]
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total_pairs = params.n_heads * (params.dim / 2u);
  if (idx >= total_pairs) { return; }

  let head = idx / (params.dim / 2u);
  let pair = idx % (params.dim / 2u);

  let freq = 1.0 / pow(params.theta, f32(pair * 2u) / f32(params.dim));
  let angle = f32(params.pos) * freq;
  let cos_a = cos(angle);
  let sin_a = sin(angle);

  let base = head * params.dim + pair * 2u;
  let x0 = qk[base];
  let x1 = qk[base + 1u];

  qk[base]     = x0 * cos_a - x1 * sin_a;
  qk[base + 1u] = x0 * sin_a + x1 * cos_a;
}
`;

/**
 * Apply RoPE with an element offset into the buffer.
 * Used for FM transformer where Q/K for multiple positions are packed contiguously.
 * offset: element offset into the buffer (e.g., position * n_heads * head_dim)
 */
export const ropeOffset = /* wgsl */ `
struct Params {
  dim: u32,       // head_dim (128)
  pos: u32,       // sequence position for RoPE angle computation
  n_heads: u32,   // number of heads to process
  theta: f32,     // rope_theta
  offset: u32,    // element offset into qk buffer
}

@group(0) @binding(0) var<storage, read_write> qk: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total_pairs = params.n_heads * (params.dim / 2u);
  if (idx >= total_pairs) { return; }

  let head = idx / (params.dim / 2u);
  let pair = idx % (params.dim / 2u);

  let freq = 1.0 / pow(params.theta, f32(pair * 2u) / f32(params.dim));
  let angle = f32(params.pos) * freq;
  let cos_a = cos(angle);
  let sin_a = sin(angle);

  let base = params.offset + head * params.dim + pair * 2u;
  let x0 = qk[base];
  let x1 = qk[base + 1u];

  qk[base]     = x0 * cos_a - x1 * sin_a;
  qk[base + 1u] = x0 * sin_a + x1 * cos_a;
}
`;

// ============================================================================
// CORE: GQA Attention
// ============================================================================

/**
 * Attention score computation: scores[h, p] = sum_d(Q[h,d] * K[p, kv_h, d]) / sqrt(head_dim)
 * With causal masking: positions > current are masked to -inf.
 */
export const attnScore = /* wgsl */ `
struct Params {
  n_heads: u32,
  n_kv_heads: u32,
  head_dim: u32,
  seq_len: u32,      // current position + 1
  kv_repeat: u32,    // n_heads / n_kv_heads
}

@group(0) @binding(0) var<storage, read> q: array<f32>;         // [n_heads, head_dim]
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;   // [max_seq, n_kv_heads * head_dim]
@group(0) @binding(2) var<storage, read_write> scores: array<f32>; // [n_heads, seq_len]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.n_heads * params.seq_len;
  if (idx >= total) { return; }

  let h = idx / params.seq_len;
  let p = idx % params.seq_len;

  let kv_h = h / params.kv_repeat;
  let scale = 1.0 / sqrt(f32(params.head_dim));

  var dot: f32 = 0.0;
  let qBase = h * params.head_dim;
  let kBase = p * params.n_kv_heads * params.head_dim + kv_h * params.head_dim;

  for (var d: u32 = 0u; d < params.head_dim; d++) {
    dot += q[qBase + d] * k_cache[kBase + d];
  }

  scores[h * params.seq_len + p] = dot * scale;
}
`;

/**
 * Softmax over attention scores (per head).
 * In-place: scores[h, :seq_len] → probabilities
 */
export const softmax = /* wgsl */ `
struct Params {
  n_heads: u32,
  seq_len: u32,
}

@group(0) @binding(0) var<storage, read_write> scores: array<f32>; // [n_heads, seq_len]
@group(0) @binding(1) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let h = wid.x;
  if (h >= params.n_heads) { return; }
  let tid = lid.x;
  let base = h * params.seq_len;

  // Find max
  var maxVal: f32 = -1e30;
  var i = tid;
  while (i < params.seq_len) {
    maxVal = max(maxVal, scores[base + i]);
    i += WG;
  }
  sdata[tid] = maxVal;
  workgroupBarrier();
  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { sdata[tid] = max(sdata[tid], sdata[tid + s]); }
    workgroupBarrier();
  }
  let globalMax = sdata[0];

  // Compute exp and sum
  var expSum: f32 = 0.0;
  i = tid;
  while (i < params.seq_len) {
    let e = exp(scores[base + i] - globalMax);
    scores[base + i] = e;
    expSum += e;
    i += WG;
  }
  sdata[tid] = expSum;
  workgroupBarrier();
  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { sdata[tid] += sdata[tid + s]; }
    workgroupBarrier();
  }
  let totalSum = sdata[0];

  // Normalize
  i = tid;
  while (i < params.seq_len) {
    scores[base + i] /= totalSum;
    i += WG;
  }
}
`;

/**
 * Attention value multiply: out[h, d] = sum_p(scores[h, p] * V[p, kv_h, d])
 */
export const attnValue = /* wgsl */ `
struct Params {
  n_heads: u32,
  n_kv_heads: u32,
  head_dim: u32,
  seq_len: u32,
  kv_repeat: u32,
}

@group(0) @binding(0) var<storage, read> scores: array<f32>;     // [n_heads, seq_len]
@group(0) @binding(1) var<storage, read> v_cache: array<f32>;    // [max_seq, n_kv_heads * head_dim]
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // [n_heads * head_dim]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.n_heads * params.head_dim;
  if (idx >= total) { return; }

  let h = idx / params.head_dim;
  let d = idx % params.head_dim;
  let kv_h = h / params.kv_repeat;

  var sum: f32 = 0.0;
  for (var p: u32 = 0u; p < params.seq_len; p++) {
    let score = scores[h * params.seq_len + p];
    let vIdx = p * params.n_kv_heads * params.head_dim + kv_h * params.head_dim + d;
    sum += score * v_cache[vIdx];
  }

  output[idx] = sum;
}
`;

/**
 * Write K/V to cache at current position.
 */
export const kvCacheWrite = /* wgsl */ `
struct Params {
  pos: u32,
  kv_dim: u32,  // n_kv_heads * head_dim
}

@group(0) @binding(0) var<storage, read> k_new: array<f32>;    // [kv_dim]
@group(0) @binding(1) var<storage, read> v_new: array<f32>;    // [kv_dim]
@group(0) @binding(2) var<storage, read_write> k_cache: array<f32>; // [max_seq, kv_dim]
@group(0) @binding(3) var<storage, read_write> v_cache: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.kv_dim) { return; }

  let cacheIdx = params.pos * params.kv_dim + i;
  k_cache[cacheIdx] = k_new[i];
  v_cache[cacheIdx] = v_new[i];
}
`;

// ============================================================================
// CORE: SwiGLU FFN
// ============================================================================

/**
 * SwiGLU activation: out[i] = silu(gate[i]) * up[i]
 * silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 */
export const swiGLU = /* wgsl */ `
struct Params {
  dim: u32,
}

@group(0) @binding(0) var<storage, read_write> gate: array<f32>;   // [hidden_dim] from w1 (in-place output)
@group(0) @binding(1) var<storage, read> up: array<f32>;     // [hidden_dim] from w3
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) { return; }

  let g = gate[i];
  let silu = g / (1.0 + exp(-g));
  gate[i] = silu * up[i];
}
`;

// ============================================================================
// CORE: Element-wise operations
// ============================================================================

/**
 * Add residual: output[i] = a[i] + b[i]
 */
export const addVectors = /* wgsl */ `
struct Params {
  dim: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) { return; }
  output[i] = a[i] + b[i];
}
`;

/**
 * In-place add: a[i] += b[i]
 * Avoids aliasing violation when output buffer is the same as an input.
 */
export const addInPlace = /* wgsl */ `
struct Params {
  dim: u32,
}

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) { return; }
  a[i] = a[i] + b[i];
}
`;

/**
 * Copy buffer: dst[i] = src[i]
 */
export const copyBuffer = /* wgsl */ `
struct Params {
  size: u32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.size) { return; }
  dst[i] = src[i];
}
`;

// ============================================================================
// FM TRANSFORMER: Sinusoidal Time Embedding
// ============================================================================

/**
 * Sinusoidal time embedding for flow matching.
 * Generates positional encoding for timestep t ∈ [0, 1].
 * output[i] = cos(t * freq_i), output[half_dim + i] = sin(t * freq_i)
 * Then projected via time_projection weight.
 */
export const timeEmbedding = /* wgsl */ `
struct Params {
  dim: u32,
  t: f32,  // timestep value [0, 1]
}

@group(0) @binding(0) var<storage, read_write> output: array<f32>;  // [dim]
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let half_dim = params.dim / 2u;
  if (i >= half_dim) { return; }

  // Match vLLM-Omni/MLX: freq = exp(-log(10000) * i / half_dim), (cos, sin) order
  let freq = exp(-9.210340371976184 * f32(i) / f32(half_dim));
  let angle = params.t * freq;

  // Output layout: [cos_0, cos_1, ..., cos_{n-1}, sin_0, sin_1, ..., sin_{n-1}]
  output[i] = cos(angle);
  output[half_dim + i] = sin(angle);
}
`;

// ============================================================================
// FM TRANSFORMER: Euler ODE Step
// ============================================================================

/**
 * Euler ODE step: x_new[i] = x_old[i] + velocity[i] * dt
 */
export const eulerStep = /* wgsl */ `
struct Params {
  dim: u32,
  dt: f32,
}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;     // [dim] in-place
@group(0) @binding(1) var<storage, read> velocity: array<f32>;     // [dim]
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) { return; }
  x[i] = x[i] + velocity[i] * params.dt;
}
`;

/**
 * CFG combine: v[i] = alpha * v_cond[i] + (1 - alpha) * v_uncond[i]
 */
export const cfgCombine = /* wgsl */ `
struct Params {
  dim: u32,
  alpha: f32,
}

@group(0) @binding(0) var<storage, read_write> v_cond: array<f32>;  // in-place output
@group(0) @binding(1) var<storage, read> v_uncond: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) { return; }
  v_cond[i] = params.alpha * v_cond[i] + (1.0 - params.alpha) * v_uncond[i];
}
`;

/**
 * FSQ quantize: clamp to [-1,1], scale to [0, levels-1], round, offset by special_tokens
 */
export const fsqQuantize = /* wgsl */ `
struct Params {
  dim: u32,
  levels: u32,      // 21
  offset: u32,      // 2 (special tokens)
}

@group(0) @binding(0) var<storage, read> input: array<f32>;       // [dim] continuous
@group(0) @binding(1) var<storage, read_write> output: array<u32>; // [dim] quantized codes
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) { return; }

  let clamped = clamp(input[i], -1.0, 1.0);
  let scaled = (clamped + 1.0) * 0.5 * f32(params.levels - 1u);
  output[i] = u32(round(scaled)) + params.offset;
}
`;

// ============================================================================
// FM TRANSFORMER: Bidirectional Attention (no causal mask)
// ============================================================================

/**
 * Bidirectional attention scores for FM transformer.
 * 3-token sequence, no causal mask.
 * scores[h, i, j] = sum_d(Q[h,i,d] * K[h,j,d]) / sqrt(head_dim)
 */
export const biAttnScore = /* wgsl */ `
struct Params {
  n_heads: u32,
  n_kv_heads: u32,
  head_dim: u32,
  seq_len: u32,     // 3 for FM
  kv_repeat: u32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;   // [seq_len * n_heads * head_dim]
@group(0) @binding(1) var<storage, read> k: array<f32>;   // [seq_len * n_kv_heads * head_dim]
@group(0) @binding(2) var<storage, read_write> scores: array<f32>; // [n_heads * seq_len * seq_len]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.n_heads * params.seq_len * params.seq_len;
  if (idx >= total) { return; }

  let h = idx / (params.seq_len * params.seq_len);
  let rem = idx % (params.seq_len * params.seq_len);
  let qi = rem / params.seq_len;
  let ki = rem % params.seq_len;
  let kv_h = h / params.kv_repeat;

  let scale = 1.0 / sqrt(f32(params.head_dim));
  var dot: f32 = 0.0;

  let qBase = qi * params.n_heads * params.head_dim + h * params.head_dim;
  let kBase = ki * params.n_kv_heads * params.head_dim + kv_h * params.head_dim;

  for (var d: u32 = 0u; d < params.head_dim; d++) {
    dot += q[qBase + d] * k[kBase + d];
  }

  scores[idx] = dot * scale;
}
`;

/**
 * Bidirectional softmax (per head, per query position).
 */
export const biSoftmax = /* wgsl */ `
struct Params {
  n_heads: u32,
  seq_len: u32,  // 3
}

@group(0) @binding(0) var<storage, read_write> scores: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.n_heads * params.seq_len;
  if (idx >= total) { return; }

  let h = idx / params.seq_len;
  let qi = idx % params.seq_len;
  let base = h * params.seq_len * params.seq_len + qi * params.seq_len;

  // Find max
  var maxVal: f32 = -1e30;
  for (var j: u32 = 0u; j < params.seq_len; j++) {
    maxVal = max(maxVal, scores[base + j]);
  }

  // Exp and sum
  var expSum: f32 = 0.0;
  for (var j: u32 = 0u; j < params.seq_len; j++) {
    let e = exp(scores[base + j] - maxVal);
    scores[base + j] = e;
    expSum += e;
  }

  // Normalize
  for (var j: u32 = 0u; j < params.seq_len; j++) {
    scores[base + j] /= expSum;
  }
}
`;

/**
 * Bidirectional attention value multiply.
 * out[qi, h, d] = sum_ki(scores[h, qi, ki] * V[ki, kv_h, d])
 */
export const biAttnValue = /* wgsl */ `
struct Params {
  n_heads: u32,
  n_kv_heads: u32,
  head_dim: u32,
  seq_len: u32,
  kv_repeat: u32,
}

@group(0) @binding(0) var<storage, read> scores: array<f32>;
@group(0) @binding(1) var<storage, read> v: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.seq_len * params.n_heads * params.head_dim;
  if (idx >= total) { return; }

  let qi = idx / (params.n_heads * params.head_dim);
  let rem = idx % (params.n_heads * params.head_dim);
  let h = rem / params.head_dim;
  let d = rem % params.head_dim;
  let kv_h = h / params.kv_repeat;

  var sum: f32 = 0.0;
  let scoreBase = h * params.seq_len * params.seq_len + qi * params.seq_len;
  for (var ki: u32 = 0u; ki < params.seq_len; ki++) {
    let score = scores[scoreBase + ki];
    let vIdx = ki * params.n_kv_heads * params.head_dim + kv_h * params.head_dim + d;
    sum += score * v[vIdx];
  }

  output[idx] = sum;
}
`;

// ============================================================================
// CODEC: VQ Lookup + FSQ Dequantize
// ============================================================================

/**
 * VQ codebook lookup: output[t, d] = codebook[code[t], d]
 * codebook is [8192, 256] in packed f16
 */
export const vqLookup = /* wgsl */ `
struct Params {
  n_frames: u32,
  codebook_dim: u32,  // 256
}

@group(0) @binding(0) var<storage, read> codes: array<u32>;     // [n_frames] semantic codes
@group(0) @binding(1) var<storage, read> codebook: array<u32>;  // packed f16 [8192, 256/2]
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // [n_frames, 256]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.n_frames * params.codebook_dim;
  if (idx >= total) { return; }

  let t = idx / params.codebook_dim;
  let d = idx % params.codebook_dim;

  let code = codes[t];
  let packedIdx = code * (params.codebook_dim / 2u) + d / 2u;
  let packed = codebook[packedIdx];
  let pair = unpack2x16float(packed);
  output[idx] = select(f32(pair.x), f32(pair.y), (d & 1u) == 1u);
}
`;

/**
 * Normalize VQ codebook in-place: codebook[i] = embedding_sum[i] / max(cluster_usage[i], eps)
 * Both codebook and cluster_usage are packed f16.
 * codebook: [n_entries, dim] as packed u32 (f16 pairs)
 * cluster_usage: [n_entries] as packed u32 (f16 pairs)
 */
export const normalizeCodebook = /* wgsl */ `
struct Params {
  n_entries: u32,
  dim: u32,  // 256
  epsilon: f32,
}

@group(0) @binding(0) var<storage, read_write> codebook: array<u32>;  // packed f16 [n_entries, dim/2]
@group(0) @binding(1) var<storage, read> usage: array<u32>;           // packed f16 [n_entries/2]
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.n_entries * (params.dim / 2u);
  if (idx >= total) { return; }

  let entry = idx / (params.dim / 2u);

  // Read usage for this entry (packed f16)
  let usage_pair_idx = entry / 2u;
  let usage_packed = usage[usage_pair_idx];
  let usage_pair = unpack2x16float(usage_packed);
  let usage_val = max(select(f32(usage_pair.x), f32(usage_pair.y), (entry & 1u) == 1u), params.epsilon);

  // Read codebook pair, divide by usage, write back
  let packed = codebook[idx];
  let pair = unpack2x16float(packed);
  let x = f32(pair.x) / usage_val;
  let y = f32(pair.y) / usage_val;
  codebook[idx] = pack2x16float(vec2<f32>(x, y));
}
`;

/**
 * FSQ dequantize: output[t, d] = (code[t, d] - offset) * 2 / (levels - 1) - 1
 * Maps integer codes [0, levels-1] → float [-1, 1]
 */
export const fsqDequant = /* wgsl */ `
struct Params {
  n_frames: u32,
  n_codebook: u32,   // 36
  levels: u32,        // 21
  offset: u32,        // 2 (special tokens to subtract)
}

@group(0) @binding(0) var<storage, read> codes: array<u32>;       // [n_frames, 36]
@group(0) @binding(1) var<storage, read_write> output: array<f32>; // [n_frames, 36]
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.n_frames * params.n_codebook;
  if (idx >= total) { return; }

  let code = codes[idx] - params.offset;
  output[idx] = f32(code) * 2.0 / f32(params.levels - 1u) - 1.0;
}
`;

// ============================================================================
// CODEC: Causal Conv1d
// ============================================================================

/**
 * Causal Conv1d with weight normalization.
 * output[c_out, t] = sum_{c_in, k} weight[c_out, c_in, k] * input[c_in, t - (kernel-1) + k]
 * Causal: only uses past + current (left-padded with zeros).
 * Weight norm: effective_weight = g * (w / ||w||)
 */
export const causalConv1d = /* wgsl */ `
struct Params {
  c_in: u32,
  c_out: u32,
  kernel: u32,
  n_frames: u32,
  stride: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;    // [n_frames_in, c_in] (time-first)
@group(0) @binding(1) var<storage, read> weight: array<u32>;   // packed f16 [c_out, c_in, kernel]
@group(0) @binding(2) var<storage, read> g: array<u32>;        // packed f16 [c_out, 1, 1] (weight norm scale)
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // [n_frames_out, c_out] (time-first)
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let n_frames_out = params.n_frames;
  let total = params.c_out * n_frames_out;
  if (idx >= total) { return; }

  let co = idx / n_frames_out;
  let t_out = idx % n_frames_out;

  // Get weight norm scale
  let gPacked = g[co / 2u];
  let gPair = unpack2x16float(gPacked);
  let gVal = select(f32(gPair.x), f32(gPair.y), (co & 1u) == 1u);

  // Compute weight L2 norm for this output channel
  var wNorm: f32 = 0.0;
  let wBase = co * params.c_in * params.kernel;
  let wSize = params.c_in * params.kernel;
  let wPacked = wSize / 2u;
  for (var p: u32 = 0u; p < wPacked; p++) {
    let packed = weight[wBase / 2u + p];
    let pair = unpack2x16float(packed);
    wNorm += f32(pair.x) * f32(pair.x) + f32(pair.y) * f32(pair.y);
  }
  wNorm = 1.0 / sqrt(wNorm + 1e-12);

  // Convolution with causal padding — input is [T, C] layout
  var sum: f32 = 0.0;
  let pad = params.kernel - 1u;
  let n_frames_in = params.n_frames * params.stride;

  for (var ci: u32 = 0u; ci < params.c_in; ci++) {
    for (var k: u32 = 0u; k < params.kernel; k++) {
      let t_in = i32(t_out * params.stride) - i32(pad) + i32(k);
      if (t_in >= 0 && u32(t_in) < n_frames_in) {
        let wIdx = (co * params.c_in + ci) * params.kernel + k;
        let wPacked2 = weight[wIdx / 2u];
        let wPair2 = unpack2x16float(wPacked2);
        let w = select(f32(wPair2.x), f32(wPair2.y), (wIdx & 1u) == 1u);
        // Time-first input: input[t, ci] = input[t * c_in + ci]
        let x = input[u32(t_in) * params.c_in + ci];
        sum += (w * wNorm * gVal) * x;
      }
    }
  }

  // Time-first output: output[t_out, co] = output[t_out * c_out + co]
  output[t_out * params.c_out + co] = sum;
}
`;

/**
 * Causal ConvTranspose1d with weight normalization — [T, C] layout
 *
 * Used for upsampling in codec decoder (stride=2, kernel=4).
 * PyTorch ConvTranspose1d weight: [in_channels, out_channels, kernel_size]
 * output[t_out, co] = sum over ci, k: weight[ci,co,k] * g[co] / ||w[:,co,:]|| * input[(t_out-k)/stride, ci]
 *   where (t_out - k) >= 0 and (t_out - k) % stride == 0
 *
 * Causal padding: raw output = (T_in - 1) * stride + kernel, trim (kernel - stride) from right
 * So n_frames_out = T_in * stride
 */
/**
 * Precompute per-c_in weight norm scales for ConvTranspose1d.
 * For weight shape [c_in, c_out, kernel] with gain g[c_in]:
 *   scale[ci] = g[ci] / ||v[ci, :, :]||
 * One workgroup per c_in channel.
 */
export const convTransposeNormScale = /* wgsl */ `
struct Params {
  c_in: u32,
  c_out: u32,
  kernel: u32,
}

@group(0) @binding(0) var<storage, read> weight: array<u32>;   // packed f16 [c_in, c_out, kernel]
@group(0) @binding(1) var<storage, read> g: array<u32>;        // packed f16 [c_in]
@group(0) @binding(2) var<storage, read_write> scale: array<f32>; // [c_in] output
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let ci = wid.x;
  if (ci >= params.c_in) { return; }
  let tid = lid.x;

  // Compute ||v[ci, :, :]||^2 with parallel reduction
  let total_elems = params.c_out * params.kernel;
  let base = ci * params.c_out * params.kernel;
  var ss: f32 = 0.0;
  var idx = tid;
  while (idx < total_elems) {
    let wIdx = base + idx;
    let packed = weight[wIdx / 2u];
    let pair = unpack2x16float(packed);
    let w = select(f32(pair.x), f32(pair.y), (wIdx & 1u) == 1u);
    ss += w * w;
    idx += WG;
  }
  sdata[tid] = ss;
  workgroupBarrier();

  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { sdata[tid] += sdata[tid + s]; }
    workgroupBarrier();
  }

  if (tid == 0u) {
    let normInv = 1.0 / sqrt(sdata[0] + 1e-12);
    // Read g[ci]
    let gPacked = g[ci / 2u];
    let gPair = unpack2x16float(gPacked);
    let gVal = select(f32(gPair.x), f32(gPair.y), (ci & 1u) == 1u);
    scale[ci] = gVal * normInv;
  }
}
`;

/**
 * Causal ConvTranspose1d (upsampling) with precomputed per-c_in weight norm scales.
 * Weight layout: [c_in, c_out, kernel] — transposed conv (dim 0 = c_in).
 * Uses precomputed scale[ci] = g[ci] / ||v[ci, :, :]|| instead of raw g and on-the-fly norm.
 */
export const causalConvTranspose1d = /* wgsl */ `
struct Params {
  c_in: u32,
  c_out: u32,
  kernel: u32,
  n_frames_out: u32,  // T_in * stride
  stride: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;    // [T_in, c_in]
@group(0) @binding(1) var<storage, read> weight: array<u32>;   // packed f16 [c_in, c_out, kernel]
@group(0) @binding(2) var<storage, read> scale: array<f32>;    // [c_in] precomputed g[ci]/||v[ci,:,:]||
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // [n_frames_out, c_out]
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.c_out * params.n_frames_out;
  if (idx >= total) { return; }

  let co = idx / params.n_frames_out;
  let t_out = idx % params.n_frames_out;

  // Transposed convolution with per-c_in weight normalization
  var sum: f32 = 0.0;
  let n_frames_in = params.n_frames_out / params.stride;

  for (var k: u32 = 0u; k < params.kernel; k++) {
    let diff = i32(t_out) - i32(k);
    if (diff >= 0 && (u32(diff) % params.stride) == 0u) {
      let t_in = u32(diff) / params.stride;
      if (t_in < n_frames_in) {
        for (var ci: u32 = 0u; ci < params.c_in; ci++) {
          let wIdx = (ci * params.c_out + co) * params.kernel + k;
          let wPacked = weight[wIdx / 2u];
          let wPair = unpack2x16float(wPacked);
          let w = select(f32(wPair.x), f32(wPair.y), (wIdx & 1u) == 1u);
          let x = input[t_in * params.c_in + ci];
          sum += (w * scale[ci]) * x;
        }
      }
    }
  }

  output[t_out * params.c_out + co] = sum;
}
`;

// ============================================================================
// CODEC: Layer Scale
// ============================================================================

/**
 * Layer scale: output[i] = input[i] * scale[i % dim] + residual[i]
 */
export const layerScale = /* wgsl */ `
struct Params {
  dim: u32,
  n_frames: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<u32>;  // packed f16 [dim]
@group(0) @binding(2) var<storage, read> residual: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.dim * params.n_frames;
  if (idx >= total) { return; }

  let d = idx % params.dim;
  let sPacked = scale[d / 2u];
  let sPair = unpack2x16float(sPacked);
  let s = select(f32(sPair.x), f32(sPair.y), (d & 1u) == 1u);

  output[idx] = input[idx] * s + residual[idx];
}
`;

// ============================================================================
// OUTPUT: Argmax for sampling
// ============================================================================

/**
 * Argmax over a vector. Returns the index of the maximum value.
 * Two-pass: first reduce to 256 candidates, then find global max.
 */
export const argmax = /* wgsl */ `
struct Params {
  dim: u32,
}

struct MaxResult {
  value: f32,
  index: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<u32>;  // [1] index
@group(0) @binding(2) var<uniform> params: Params;

const WG: u32 = 256u;

struct SharedEntry {
  value: f32,
  index: u32,
}
var<workgroup> sdata: array<SharedEntry, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;

  // Each thread finds max over its portion
  var bestVal: f32 = -1e30;
  var bestIdx: u32 = 0u;

  var i = tid;
  while (i < params.dim) {
    let v = input[i];
    if (v > bestVal) {
      bestVal = v;
      bestIdx = i;
    }
    i += WG;
  }

  sdata[tid] = SharedEntry(bestVal, bestIdx);
  workgroupBarrier();

  // Reduction
  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (tid < s) {
      if (sdata[tid + s].value > sdata[tid].value) {
        sdata[tid] = sdata[tid + s];
      }
    }
    workgroupBarrier();
  }

  if (tid == 0u) {
    result[0] = sdata[0].index;
  }
}
`;

// ============================================================================
// OFFSET VARIANTS: For FM 3-token bidirectional processing
// ============================================================================

/**
 * F16 matvec with offset: y[dst_offset + m] = sum_k(A[m,k] * x[src_offset + k])
 * Used for processing individual positions in a multi-token sequence.
 */
export const matvecF16Offset = /* wgsl */ `
struct Params {
  M: u32,
  K: u32,
  src_offset: u32,
  dst_offset: u32,
}

@group(0) @binding(0) var<storage, read> matrix: array<u32>;
@group(0) @binding(1) var<storage, read> vector: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let row = wid.x;
  if (row >= params.M) { return; }
  let tid = lid.x;

  let K_packed = params.K / 2u;
  var sum: f32 = 0.0;

  var pk = tid;
  let rowBase = row * K_packed;
  while (pk < K_packed) {
    let packed = matrix[rowBase + pk];
    let pair = unpack2x16float(packed);
    let k0 = pk * 2u;
    sum += f32(pair.x) * vector[params.src_offset + k0]
         + f32(pair.y) * vector[params.src_offset + k0 + 1u];
    pk += WG;
  }

  sdata[tid] = sum;
  workgroupBarrier();

  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { sdata[tid] += sdata[tid + s]; }
    workgroupBarrier();
  }

  if (tid == 0u) {
    output[params.dst_offset + row] = sdata[0];
  }
}
`;

/**
 * RMSNorm with offset: reads/writes at position p in a sequence.
 * input[src_offset + i], weight[i], output[dst_offset + i]
 */
export const rmsNormOffset = /* wgsl */ `
struct Params {
  dim: u32,
  eps: f32,
  src_offset: u32,
  dst_offset: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;

  var ss: f32 = 0.0;
  var i = tid;
  while (i < params.dim) {
    let v = input[params.src_offset + i];
    ss += v * v;
    i += WG;
  }
  sdata[tid] = ss;
  workgroupBarrier();

  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { sdata[tid] += sdata[tid + s]; }
    workgroupBarrier();
  }

  let rms = 1.0 / sqrt(sdata[0] / f32(params.dim) + params.eps);

  i = tid;
  while (i < params.dim) {
    let wPacked = weight[i / 2u];
    let wPair = unpack2x16float(wPacked);
    let w = select(f32(wPair.x), f32(wPair.y), (i & 1u) == 1u);
    output[params.dst_offset + i] = input[params.src_offset + i] * rms * w;
    i += WG;
  }
}
`;

/**
 * Add vectors with offset: output[dst_offset + i] = a[off_a + i] + b[off_b + i]
 */
export const addVectorsOffset = /* wgsl */ `
struct Params {
  dim: u32,
  off_a: u32,
  off_b: u32,
  dst_offset: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) { return; }
  output[params.dst_offset + i] = a[params.off_a + i] + b[params.off_b + i];
}
`;

/**
 * In-place add with offsets: a[off_a + i] += b[off_b + i]
 * Avoids aliasing violation when output buffer is the same as input buffer a.
 */
export const addInPlaceOffset = /* wgsl */ `
struct Params {
  dim: u32,
  off_a: u32,
  off_b: u32,
}

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) { return; }
  a[params.off_a + i] = a[params.off_a + i] + b[params.off_b + i];
}
`;

/**
 * SwiGLU with offset (in-place): gate[off_g + i] = silu(gate[off_g + i]) * up[off_u + i]
 */
export const swiGLUOffset = /* wgsl */ `
struct Params {
  dim: u32,
  off_gate: u32,
  off_up: u32,
}

@group(0) @binding(0) var<storage, read_write> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) { return; }

  let g = gate[params.off_gate + i];
  let silu = g / (1.0 + exp(-g));
  gate[params.off_gate + i] = silu * up[params.off_up + i];
}
`;

/**
 * Copy with offsets: dst[dst_offset + i] = src[src_offset + i]
 */
export const copyBufferOffset = /* wgsl */ `
struct Params {
  size: u32,
  src_offset: u32,
  dst_offset: u32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.size) { return; }
  dst[params.dst_offset + i] = src[params.src_offset + i];
}
`;

// ============================================================================
// CODEC: ALiBi Attention Score (with sliding window)
// ============================================================================

/**
 * ALiBi attention scores with sliding window masking.
 * score[h, qi, ki] = sum_d(Q[qi,h,d] * K[ki,h,d]) / sqrt(head_dim) + alibi_bias
 * alibi_bias = -slope_h * |qi - ki|
 * Sliding window: mask positions outside window to -inf.
 *
 * Input layout: Q/K are [seq_len, n_heads, head_dim] (time-major for codec).
 */
export const alibiAttnScore = /* wgsl */ `
struct Params {
  n_heads: u32,
  head_dim: u32,
  seq_len: u32,
  window: u32,      // sliding window size (0 = full attention)
}

@group(0) @binding(0) var<storage, read> q: array<f32>;     // [seq_len, n_heads, head_dim]
@group(0) @binding(1) var<storage, read> k: array<f32>;     // [seq_len, n_heads, head_dim]
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;  // [n_heads, seq_len, seq_len]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // gid.x = ki_group (within seq_len), gid.y = qi, gid.z = head
  let ki = gid.x;
  let qi = gid.y;
  let h = gid.z;
  if (ki >= params.seq_len || qi >= params.seq_len || h >= params.n_heads) { return; }
  let idx = h * params.seq_len * params.seq_len + qi * params.seq_len + ki;

  // Causal mask: ki > qi means future — mask out
  if (ki > qi) {
    scores[idx] = -1e30;
    return;
  }

  // Sliding window mask
  if (params.window > 0u && qi - ki >= params.window) {
    scores[idx] = -1e30;
    return;
  }

  let scale = 1.0 / sqrt(f32(params.head_dim));

  // ALiBi slope: 2^(-8*h/n_heads) — geometric series from 2^(-8/n) to 2^(-8)
  let slope = pow(2.0, -8.0 * f32(h + 1u) / f32(params.n_heads));
  let alibi = -slope * f32(qi - ki);

  var dot: f32 = 0.0;
  let qBase = qi * params.n_heads * params.head_dim + h * params.head_dim;
  let kBase = ki * params.n_heads * params.head_dim + h * params.head_dim;

  for (var d: u32 = 0u; d < params.head_dim; d++) {
    dot += q[qBase + d] * k[kBase + d];
  }

  scores[idx] = dot * scale + alibi;
}
`;

/**
 * Codec attention softmax (per head, per query position).
 * Works on [n_heads, seq_len, seq_len] score array.
 */
export const codecSoftmax = /* wgsl */ `
struct Params {
  n_heads: u32,
  seq_len: u32,
}

@group(0) @binding(0) var<storage, read_write> scores: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.n_heads * params.seq_len;
  if (idx >= total) { return; }

  let h = idx / params.seq_len;
  let qi = idx % params.seq_len;
  let base = h * params.seq_len * params.seq_len + qi * params.seq_len;

  var maxVal: f32 = -1e30;
  for (var j: u32 = 0u; j < params.seq_len; j++) {
    maxVal = max(maxVal, scores[base + j]);
  }

  var expSum: f32 = 0.0;
  for (var j: u32 = 0u; j < params.seq_len; j++) {
    let e = exp(scores[base + j] - maxVal);
    scores[base + j] = e;
    expSum += e;
  }

  for (var j: u32 = 0u; j < params.seq_len; j++) {
    scores[base + j] /= expSum;
  }
}
`;

/**
 * Codec attention value multiply (time-major layout).
 * out[qi, h, d] = sum_ki(scores[h, qi, ki] * V[ki, h, d])
 */
export const codecAttnValue = /* wgsl */ `
struct Params {
  n_heads: u32,
  head_dim: u32,
  seq_len: u32,
}

@group(0) @binding(0) var<storage, read> scores: array<f32>;   // [n_heads, seq_len, seq_len]
@group(0) @binding(1) var<storage, read> v: array<f32>;        // [seq_len, n_heads, head_dim]
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // [seq_len, n_heads, head_dim]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.seq_len * params.n_heads * params.head_dim;
  if (idx >= total) { return; }

  let qi = idx / (params.n_heads * params.head_dim);
  let rem = idx % (params.n_heads * params.head_dim);
  let h = rem / params.head_dim;
  let d = rem % params.head_dim;

  var sum: f32 = 0.0;
  let scoreBase = h * params.seq_len * params.seq_len + qi * params.seq_len;
  for (var ki: u32 = 0u; ki < params.seq_len; ki++) {
    let score = scores[scoreBase + ki];
    let vIdx = ki * params.n_heads * params.head_dim + h * params.head_dim + d;
    sum += score * v[vIdx];
  }

  output[idx] = sum;
}
`;

// ============================================================================
// CODEC: Batched operations for multi-frame processing
// ============================================================================

/**
 * Batched matvec: for each frame t in [0, T), output[t*M + m] = sum_k(A[m,k] * input[t*K + k])
 * A is [M, K] packed f16, shared across all frames.
 */
export const batchedMatvecF16 = /* wgsl */ `
struct Params {
  M: u32,
  K: u32,
  T: u32,  // number of frames
}

@group(0) @binding(0) var<storage, read> matrix: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let row = wid.x;
  let frame = wid.y;
  if (row >= params.M || frame >= params.T) { return; }
  let tid = lid.x;

  let K_packed = params.K / 2u;
  var sum: f32 = 0.0;

  var pk = tid;
  let rowBase = row * K_packed;
  let inputBase = frame * params.K;
  while (pk < K_packed) {
    let packed = matrix[rowBase + pk];
    let pair = unpack2x16float(packed);
    let k0 = pk * 2u;
    sum += f32(pair.x) * input[inputBase + k0]
         + f32(pair.y) * input[inputBase + k0 + 1u];
    pk += WG;
  }

  sdata[tid] = sum;
  workgroupBarrier();

  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { sdata[tid] += sdata[tid + s]; }
    workgroupBarrier();
  }

  if (tid == 0u) {
    output[frame * params.M + row] = sdata[0];
  }
}
`;

/**
 * Batched RMSNorm: normalize each frame independently.
 * input/output are [T, dim], weight is [dim].
 */
export const batchedRmsNorm = /* wgsl */ `
struct Params {
  dim: u32,
  eps: f32,
  T: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let frame = wid.x;
  if (frame >= params.T) { return; }
  let tid = lid.x;
  let base = frame * params.dim;

  var ss: f32 = 0.0;
  var i = tid;
  while (i < params.dim) {
    let v = input[base + i];
    ss += v * v;
    i += WG;
  }
  sdata[tid] = ss;
  workgroupBarrier();

  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { sdata[tid] += sdata[tid + s]; }
    workgroupBarrier();
  }

  let rms = 1.0 / sqrt(sdata[0] / f32(params.dim) + params.eps);

  i = tid;
  while (i < params.dim) {
    let wPacked = weight[i / 2u];
    let wPair = unpack2x16float(wPacked);
    let w = select(f32(wPair.x), f32(wPair.y), (i & 1u) == 1u);
    output[base + i] = input[base + i] * rms * w;
    i += WG;
  }
}
`;

/**
 * Batched SwiGLU: out[i] = silu(gate[i]) * up[i] for all elements.
 */
export const batchedSwiGLU = /* wgsl */ `
struct Params {
  total: u32,  // T * hidden_dim
}

@group(0) @binding(0) var<storage, read_write> gate: array<f32>;  // in-place output
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.total) { return; }
  let g = gate[i];
  let silu = g / (1.0 + exp(-g));
  gate[i] = silu * up[i];
}
`;

/**
 * Batched add: output[i] = a[i] + b[i] for all elements.
 */
export const batchedAdd = /* wgsl */ `
struct Params {
  total: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.total) { return; }
  output[i] = a[i] + b[i];
}
`;

/**
 * Batched copy: dst[i] = src[i]
 */
export const batchedCopy = /* wgsl */ `
struct Params {
  total: u32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.total) { return; }
  dst[i] = src[i];
}
`;

/**
 * Batched layer scale with residual:
 * output[t*dim + d] = input[t*dim + d] * scale[d] + residual[t*dim + d]
 */
export const batchedLayerScale = /* wgsl */ `
struct Params {
  dim: u32,
  total: u32,  // T * dim
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<u32>;  // packed f16 [dim]
@group(0) @binding(2) var<storage, read> residual: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.total) { return; }

  let d = idx % params.dim;
  let sPacked = scale[d / 2u];
  let sPair = unpack2x16float(sPacked);
  let s = select(f32(sPair.x), f32(sPair.y), (d & 1u) == 1u);

  output[idx] = input[idx] * s + residual[idx];
}
`;

/**
 * QK normalization for codec: normalize Q and K per head.
 * Applies RMSNorm weight per-head to Q or K.
 * input: [seq_len, n_heads, head_dim], weight: [head_dim], output: same shape
 * Normalizes over head_dim dimension.
 */
export const qkNorm = /* wgsl */ `
struct Params {
  n_heads: u32,
  head_dim: u32,
  seq_len: u32,
  eps: f32,
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;  // [seq_len, n_heads * head_dim]
@group(0) @binding(1) var<storage, read> weight: array<u32>;  // packed f16 [n_heads * head_dim]
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.seq_len * params.n_heads;
  if (idx >= total) { return; }

  let base = idx * params.head_dim;
  // Head index for weight lookup: weight is [n_heads * head_dim], each head has head_dim elements
  let h = idx % params.n_heads;
  let wOffset = h * params.head_dim;

  // Compute RMS
  var ss: f32 = 0.0;
  for (var d: u32 = 0u; d < params.head_dim; d++) {
    let v = data[base + d];
    ss += v * v;
  }
  let rms = 1.0 / sqrt(ss / f32(params.head_dim) + params.eps);

  // Normalize with per-head weight (in-place)
  for (var d: u32 = 0u; d < params.head_dim; d++) {
    let wIdx = wOffset + d;
    let wPacked = weight[wIdx / 2u];
    let wPair = unpack2x16float(wPacked);
    let w = select(f32(wPair.x), f32(wPair.y), (wIdx & 1u) == 1u);
    data[base + d] = data[base + d] * rms * w;
  }
}
`;

/**
 * Concatenate semantic embeddings [T, 256] and acoustic floats [T, 36] → [T, 292]
 */
export const concatCodecInput = /* wgsl */ `
struct Params {
  T: u32,
  sem_dim: u32,   // 256
  ac_dim: u32,    // 36
}

@group(0) @binding(0) var<storage, read> semantic: array<f32>;  // [T, 256]
@group(0) @binding(1) var<storage, read> acoustic: array<f32>;  // [T, 36]
@group(0) @binding(2) var<storage, read_write> output: array<f32>;  // [T, 292]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let out_dim = params.sem_dim + params.ac_dim;
  let total = params.T * out_dim;
  if (idx >= total) { return; }

  let t = idx / out_dim;
  let d = idx % out_dim;

  if (d < params.sem_dim) {
    output[idx] = semantic[t * params.sem_dim + d];
  } else {
    output[idx] = acoustic[t * params.ac_dim + (d - params.sem_dim)];
  }
}
`;

/**
 * Multi-codebook embedding sum for autoregressive audio tokens.
 * Sums embeddings from 37 codebooks (1 semantic + 36 acoustic) into output.
 * Each codebook has its own offset in the global embedding table:
 *   CB 0 (semantic): offset 0, size 8194
 *   CB k (acoustic): offset 8194 + (k-1)*23, size 23
 * Matches vLLM-Omni/MLX multi_codebook_embedding.sum(dim=1).
 */
export const multiCodebookEmbed = /* wgsl */ `
struct Params {
  dim: u32,
  acoustic_base: u32,    // 8194 (semantic_codebook_size + 2 specials)
  acoustic_stride: u32,  // 23 (acoustic_codebook_size + 2 specials)
  n_acoustic: u32,       // 36
}

@group(0) @binding(0) var<storage, read> table: array<u32>;           // packed f16 [vocab, dim/2]
@group(0) @binding(1) var<storage, read> semantic_code: array<u32>;   // [1] semantic argmax
@group(0) @binding(2) var<storage, read> acoustic_codes: array<u32>;  // [36] FSQ codes with +2 offset
@group(0) @binding(3) var<storage, read_write> output: array<f32>;    // [dim]
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.dim) { return; }

  let half_i = i / 2u;
  let is_odd = (i & 1u) == 1u;
  var sum: f32 = 0.0;

  // Semantic codebook (offset 0)
  let sem_row = semantic_code[0];
  let sem_packed = table[sem_row * (params.dim / 2u) + half_i];
  let sem_pair = unpack2x16float(sem_packed);
  sum += select(f32(sem_pair.x), f32(sem_pair.y), is_odd);

  // 36 acoustic codebooks
  for (var k: u32 = 0u; k < params.n_acoustic; k++) {
    let ac_row = params.acoustic_base + k * params.acoustic_stride + acoustic_codes[k];
    let ac_packed = table[ac_row * (params.dim / 2u) + half_i];
    let ac_pair = unpack2x16float(ac_packed);
    sum += select(f32(ac_pair.x), f32(ac_pair.y), is_odd);
  }

  output[i] = sum;
}
`;

/**
 * Zero-fill a buffer.
 */
export const zeroFill = /* wgsl */ `
struct Params {
  total: u32,
}

@group(0) @binding(0) var<storage, read_write> buf: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.total) { return; }
  buf[i] = 0.0;
}
`;

// Keep benchmark shaders for backward compatibility
export { matvecF32, matmulF32 } from './benchmark-shaders.js';
