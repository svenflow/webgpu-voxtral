/**
 * Benchmark-only shaders (Phase 0a)
 * Kept separate from production shaders.
 */

export const matvecF32 = /* wgsl */ `
struct Params {
  M: u32,
  K: u32,
}

@group(0) @binding(0) var<storage, read> matrix: array<f32>;
@group(0) @binding(1) var<storage, read> vector: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const TILE_SIZE: u32 = 256u;
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let row = wid.x;
  if (row >= params.M) { return; }
  let tid = lid.x;
  var sum: f32 = 0.0;
  let num_tiles = (params.K + TILE_SIZE - 1u) / TILE_SIZE;
  for (var t: u32 = 0u; t < num_tiles; t++) {
    let k = t * TILE_SIZE + tid;
    if (k < params.K) {
      sum += matrix[row * params.K + k] * vector[k];
    }
  }
  sdata[tid] = sum;
  workgroupBarrier();
  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) { sdata[tid] += sdata[tid + stride]; }
    workgroupBarrier();
  }
  if (tid == 0u) { output[row] = sdata[0]; }
}
`;

export const matmulF32 = /* wgsl */ `
struct Params {
  M: u32,
  N: u32,
  K: u32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const TILE: u32 = 16u;
var<workgroup> tileA: array<f32, 256>;
var<workgroup> tileB: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = gid.y;
  let col = gid.x;
  let lr = lid.y;
  let lc = lid.x;
  var sum: f32 = 0.0;
  let num_tiles = (params.K + TILE - 1u) / TILE;
  for (var t: u32 = 0u; t < num_tiles; t++) {
    let aCol = t * TILE + lc;
    if (row < params.M && aCol < params.K) { tileA[lr * TILE + lc] = A[row * params.K + aCol]; }
    else { tileA[lr * TILE + lc] = 0.0; }
    let bRow = t * TILE + lr;
    if (bRow < params.K && col < params.N) { tileB[lr * TILE + lc] = B[bRow * params.N + col]; }
    else { tileB[lr * TILE + lc] = 0.0; }
    workgroupBarrier();
    for (var k: u32 = 0u; k < TILE; k++) { sum += tileA[lr * TILE + k] * tileB[k * TILE + lc]; }
    workgroupBarrier();
  }
  if (row < params.M && col < params.N) { C[row * params.N + col] = sum; }
}
`;
