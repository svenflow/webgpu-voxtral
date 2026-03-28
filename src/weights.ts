/**
 * Weight loader for Voxtral TTS
 *
 * Supports two loading strategies:
 * 1. Manifest-based: Load from pre-extracted F16 .bin files with manifest.json
 * 2. Direct safetensors: Parse consolidated.safetensors and convert BF16→F16 on the fly
 *
 * Both support streaming via HTTP range requests for progressive loading.
 */

import { VoxtralConfig, defaultConfig } from './types.js';

/** Weight manifest from extract_weights.py */
export interface WeightManifest {
  format: string;
  source: string;
  total_tensors: number;
  tensors: Record<string, TensorInfo>;
  total_bytes?: number;
}

export interface TensorInfo {
  file: string;
  offset: number;
  size: number;
  padded_size: number;
  shape: number[];
  dtype: string;
  component: string;
  layer?: number;
  block?: number;
}

/** Safetensors tensor entry */
interface SafetensorsTensorEntry {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

/** Safetensors header format */
type SafetensorsHeader = Record<string, SafetensorsTensorEntry | Record<string, string>>;

// BF16 dtype identifier in safetensors
const DTYPE_SIZES: Record<string, number> = {
  'F16': 2,
  'BF16': 2,
  'F32': 4,
  'F64': 8,
  'I8': 1,
  'I16': 2,
  'I32': 4,
  'I64': 8,
  'U8': 1,
  'U16': 2,
  'U32': 4,
  'BOOL': 1,
};

/**
 * Parse safetensors header from a URL.
 * Returns the header and data offset for range requests.
 */
export async function parseSafetensorsHeader(
  url: string,
): Promise<{ header: SafetensorsHeader; dataOffset: number }> {
  // First 8 bytes = little-endian u64 header length
  const headResp = await fetch(url, {
    headers: { Range: 'bytes=0-7' },
  });
  const headBuf = await headResp.arrayBuffer();
  const headerLen = Number(new DataView(headBuf).getBigUint64(0, true));

  // Read the JSON header
  const headerResp = await fetch(url, {
    headers: { Range: `bytes=8-${8 + headerLen - 1}` },
  });
  const headerText = await headerResp.text();
  const header: SafetensorsHeader = JSON.parse(headerText);

  return { header, dataOffset: 8 + headerLen };
}

/**
 * Convert BF16 bytes to F16 bytes in-place.
 *
 * BF16: [sign(1) | exp(8) | mantissa(7)]
 * F16:  [sign(1) | exp(5) | mantissa(10)]
 *
 * Conversion: preserve sign, remap exponent from bias-127 to bias-15,
 * extend mantissa from 7 to 10 bits (zero-fill low 3 bits).
 * Clamp overflows to F16 max, underflows to zero.
 */
export function convertBF16toF16(bf16Bytes: ArrayBuffer): ArrayBuffer {
  const input = new Uint16Array(bf16Bytes);
  const output = new Uint16Array(input.length);

  for (let i = 0; i < input.length; i++) {
    const bf16 = input[i];
    const sign = (bf16 >> 15) & 1;
    const exp = (bf16 >> 7) & 0xFF;   // 8-bit exponent, bias 127
    const mant = bf16 & 0x7F;          // 7-bit mantissa

    if (exp === 0xFF) {
      // Inf/NaN
      output[i] = (sign << 15) | 0x7C00 | (mant ? 0x0200 : 0); // preserve NaN vs Inf
    } else if (exp === 0) {
      // Zero or subnormal BF16 → zero in F16 (too small)
      output[i] = (sign << 15);
    } else {
      // Normal: remap exponent
      const unbiased = exp - 127;

      if (unbiased > 15) {
        // Overflow → F16 infinity
        output[i] = (sign << 15) | 0x7C00;
      } else if (unbiased < -14) {
        // Underflow → F16 subnormal or zero
        const shift = -14 - unbiased;
        if (shift > 10) {
          output[i] = (sign << 15); // too small
        } else {
          // Create F16 subnormal
          const f16Mant = ((0x80 | (mant << 1)) >> shift) >> 1;
          output[i] = (sign << 15) | (f16Mant & 0x03FF);
        }
      } else {
        // Normal range
        const f16Exp = unbiased + 15;
        // Extend 7-bit mantissa to 10 bits: shift left 3, zero-fill
        const f16Mant = mant << 3;
        output[i] = (sign << 15) | (f16Exp << 10) | (f16Mant & 0x03FF);
      }
    }
  }

  return output.buffer;
}

/**
 * Load weight manifest from a URL.
 */
export async function loadManifest(baseUrl: string): Promise<WeightManifest> {
  const resp = await fetch(`${baseUrl}/manifest.json`);
  if (!resp.ok) throw new Error(`Failed to load manifest: ${resp.status}`);
  return resp.json();
}

export interface WeightLoadProgress {
  loaded: number;
  total: number;
  component: string;
  tensor: string;
}

/**
 * Load a single tensor from a manifest-based binary file using range requests.
 */
export async function loadTensorFromManifest(
  baseUrl: string,
  manifest: WeightManifest,
  tensorName: string,
): Promise<Float32Array | Uint16Array> {
  const info = manifest.tensors[tensorName];
  if (!info) throw new Error(`Tensor not found: ${tensorName}`);

  const url = `${baseUrl}/${info.file}`;
  const resp = await fetch(url, {
    headers: { Range: `bytes=${info.offset}-${info.offset + info.size - 1}` },
  });

  const buf = await resp.arrayBuffer();

  if (info.dtype === 'f16') {
    return new Uint16Array(buf);
  } else {
    return new Float32Array(buf);
  }
}

/**
 * Load a tensor directly from safetensors, converting BF16→F16 on the fly.
 */
export async function loadTensorFromSafetensors(
  url: string,
  header: SafetensorsHeader,
  dataOffset: number,
  tensorName: string,
): Promise<Uint16Array> {
  const entry = header[tensorName] as SafetensorsTensorEntry | undefined;
  if (!entry || !('data_offsets' in entry)) {
    throw new Error(`Tensor not found in safetensors: ${tensorName}`);
  }

  const [start, end] = entry.data_offsets;
  const absStart = dataOffset + (start as number);
  const absEnd = dataOffset + (end as number) - 1;

  const resp = await fetch(url, {
    headers: { Range: `bytes=${absStart}-${absEnd}` },
  });
  const buf = await resp.arrayBuffer();

  if (entry.dtype === 'BF16') {
    const f16Buf = convertBF16toF16(buf);
    return new Uint16Array(f16Buf);
  } else if (entry.dtype === 'F16') {
    return new Uint16Array(buf);
  } else if (entry.dtype === 'F32') {
    // F32 → F16 conversion
    const f32 = new Float32Array(buf);
    const f16 = new Uint16Array(f32.length);
    for (let i = 0; i < f32.length; i++) {
      f16[i] = float32ToFloat16(f32[i]);
    }
    return f16;
  } else {
    throw new Error(`Unsupported dtype: ${entry.dtype}`);
  }
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

/**
 * Create a GPU storage buffer from F16 weight data.
 */
export function createWeightBuffer(
  device: GPUDevice,
  data: Uint16Array,
  label?: string,
): GPUBuffer {
  // Pad to 4-byte alignment for WebGPU
  const byteLength = data.byteLength;
  const paddedLength = Math.ceil(byteLength / 4) * 4;

  const buffer = device.createBuffer({
    size: paddedLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label,
    mappedAtCreation: true,
  });

  new Uint16Array(buffer.getMappedRange(0, data.byteLength)).set(data);
  buffer.unmap();

  return buffer;
}

/**
 * Weight loading orchestrator.
 *
 * Loads all weights for a specific component, creating GPU buffers.
 * Supports progressive loading with progress callbacks.
 */
export interface ComponentWeights {
  buffers: Map<string, GPUBuffer>;
  tensors: Map<string, { shape: number[]; buffer: GPUBuffer }>;
}

export async function loadComponentWeights(
  device: GPUDevice,
  baseUrl: string,
  manifest: WeightManifest,
  component: string,
  onProgress?: (progress: WeightLoadProgress) => void,
): Promise<ComponentWeights> {
  const tensorNames = Object.entries(manifest.tensors)
    .filter(([_, info]) => info.component === component)
    .map(([name, _]) => name);

  const buffers = new Map<string, GPUBuffer>();
  const tensors = new Map<string, { shape: number[]; buffer: GPUBuffer }>();

  const total = tensorNames.length;

  for (let i = 0; i < tensorNames.length; i++) {
    const name = tensorNames[i];
    const info = manifest.tensors[name];

    if (onProgress) {
      onProgress({
        loaded: i,
        total,
        component,
        tensor: name,
      });
    }

    const data = await loadTensorFromManifest(baseUrl, manifest, name) as Uint16Array;
    const buffer = createWeightBuffer(device, data, name);

    buffers.set(name, buffer);
    tensors.set(name, { shape: info.shape, buffer });
  }

  if (onProgress) {
    onProgress({ loaded: total, total, component, tensor: 'done' });
  }

  return { buffers, tensors };
}

/**
 * Bulk load: fetch entire component .bin file at once (faster than range requests).
 * Creates all GPU buffers from the single download.
 */
export async function loadComponentBulk(
  device: GPUDevice,
  baseUrl: string,
  manifest: WeightManifest,
  component: string,
  onProgress?: (progress: WeightLoadProgress) => void,
): Promise<ComponentWeights> {
  const tensorNames = Object.entries(manifest.tensors)
    .filter(([_, info]) => info.component === component)
    .map(([name, _]) => name);

  // Determine the .bin filename from the first tensor
  const firstInfo = manifest.tensors[tensorNames[0]];
  const binFile = firstInfo.file;

  if (onProgress) {
    onProgress({ loaded: 0, total: tensorNames.length, component, tensor: `downloading ${binFile}...` });
  }

  // Download entire .bin file
  const resp = await fetch(`${baseUrl}/${binFile}`);
  if (!resp.ok) throw new Error(`Failed to download ${binFile}: ${resp.status}`);
  const fullBuffer = await resp.arrayBuffer();

  const buffers = new Map<string, GPUBuffer>();
  const tensors = new Map<string, { shape: number[]; buffer: GPUBuffer }>();

  for (let i = 0; i < tensorNames.length; i++) {
    const name = tensorNames[i];
    const info = manifest.tensors[name];

    const slice = new Uint16Array(fullBuffer, info.offset, info.size / 2);
    const gpuBuf = createWeightBuffer(device, slice, name);

    buffers.set(name, gpuBuf);
    tensors.set(name, { shape: info.shape, buffer: gpuBuf });

    if (onProgress && (i % 20 === 0 || i === tensorNames.length - 1)) {
      onProgress({ loaded: i + 1, total: tensorNames.length, component, tensor: name });
    }
  }

  return { buffers, tensors };
}

// =========================================================================
// IndexedDB Cache for F16 weights
// =========================================================================

const IDB_NAME = 'voxtral-weights';
const IDB_VERSION = 1;
const IDB_STORE = 'tensors';

/** Open (or create) the IndexedDB for weight caching. */
function openWeightDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(IDB_NAME, IDB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(IDB_STORE)) {
        db.createObjectStore(IDB_STORE);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

/** Get a cached F16 tensor from IndexedDB. Returns null on miss. */
async function idbGet(db: IDBDatabase, key: string): Promise<ArrayBuffer | null> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, 'readonly');
    const store = tx.objectStore(IDB_STORE);
    const req = store.get(key);
    req.onsuccess = () => resolve(req.result ?? null);
    req.onerror = () => reject(req.error);
  });
}

/** Store an F16 tensor in IndexedDB. */
async function idbPut(db: IDBDatabase, key: string, data: ArrayBuffer): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, 'readwrite');
    const store = tx.objectStore(IDB_STORE);
    const req = store.put(data, key);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

/** Check how many tensors are already cached. */
async function idbCount(db: IDBDatabase): Promise<number> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, 'readonly');
    const store = tx.objectStore(IDB_STORE);
    const req = store.count();
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

/** Clear all cached weights. */
export async function clearWeightCache(): Promise<void> {
  const db = await openWeightDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, 'readwrite');
    const store = tx.objectStore(IDB_STORE);
    const req = store.clear();
    req.onsuccess = () => { db.close(); resolve(); };
    req.onerror = () => { db.close(); reject(req.error); };
  });
}

// =========================================================================
// HuggingFace CDN streaming loader with IndexedDB caching
// =========================================================================

/** Default HF CDN URL for Voxtral weights */
export const HF_VOXTRAL_URL =
  'https://huggingface.co/mistralai/Voxtral-4B-TTS-2603/resolve/main/consolidated.safetensors';

/** Progress info for HF streaming loader */
export interface HFLoadProgress {
  loaded: number;       // tensors loaded so far
  total: number;        // total tensors
  component: string;    // current component being loaded
  tensor: string;       // current tensor name
  cached: boolean;      // whether this tensor came from IDB cache
  bytesDownloaded: number;  // total bytes fetched from network
}

/**
 * Classify a safetensors tensor name into a component.
 * Mirrors the Python extract_weights.py logic.
 */
function classifyComponent(name: string): string {
  if (name.startsWith('acoustic_transformer.')) return 'fm';
  if (name.startsWith('audio_tokenizer.')) return 'codec';
  if (name.startsWith('layers.') || name.startsWith('norm.') || name.startsWith('mm_audio_embeddings.')) return 'backbone';
  return 'other';
}

/**
 * Load all model weights by streaming from HuggingFace CDN.
 *
 * For each tensor:
 * 1. Check IndexedDB cache — if hit, use cached F16 data directly
 * 2. If miss, fetch via HTTP range request from safetensors file
 * 3. Convert BF16 → F16 in JS
 * 4. Cache converted F16 in IndexedDB for next time
 * 5. Create GPU buffer, then release JS ArrayBuffer
 *
 * Never holds more than one tensor in RAM at a time.
 */
export async function loadWeightsFromHF(
  device: GPUDevice,
  safetensorsUrl: string = HF_VOXTRAL_URL,
  onProgress?: (progress: HFLoadProgress) => void,
): Promise<{
  backbone: ComponentWeights;
  fm: ComponentWeights;
  codec: ComponentWeights;
}> {
  // 1. Parse safetensors header (two small range requests)
  const { header, dataOffset } = await parseSafetensorsHeader(safetensorsUrl);

  // 2. Open IndexedDB
  const db = await openWeightDB();
  const cachedCount = await idbCount(db);

  // 3. Build tensor list grouped by component
  const tensorEntries: { name: string; entry: SafetensorsTensorEntry; component: string }[] = [];
  for (const [name, value] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const entry = value as SafetensorsTensorEntry;
    if (!entry.data_offsets) continue;
    const component = classifyComponent(name);
    if (component === 'other') continue;
    tensorEntries.push({ name, entry, component });
  }

  // Sort by component for better progress reporting
  const componentOrder: Record<string, number> = { backbone: 0, fm: 1, codec: 2 };
  tensorEntries.sort((a, b) => (componentOrder[a.component] ?? 9) - (componentOrder[b.component] ?? 9));

  const totalTensors = tensorEntries.length;
  let bytesDownloaded = 0;

  if (onProgress) {
    onProgress({
      loaded: 0, total: totalTensors, component: 'init',
      tensor: cachedCount > 0 ? `${cachedCount} tensors cached in IndexedDB` : 'Starting fresh download...',
      cached: false, bytesDownloaded: 0,
    });
  }

  // 4. Load each tensor: IDB cache → range request → convert → cache → GPU
  const results: Record<string, { buffers: Map<string, GPUBuffer>; tensors: Map<string, { shape: number[]; buffer: GPUBuffer }> }> = {
    backbone: { buffers: new Map(), tensors: new Map() },
    fm: { buffers: new Map(), tensors: new Map() },
    codec: { buffers: new Map(), tensors: new Map() },
  };

  // Version key for cache invalidation (based on safetensors header size + tensor count)
  const cacheVersion = `v1:${dataOffset}:${totalTensors}`;

  for (let i = 0; i < tensorEntries.length; i++) {
    const { name, entry, component } = tensorEntries[i];
    const cacheKey = `${cacheVersion}:${name}`;

    let f16Data: Uint16Array;
    let fromCache = false;

    // Try IDB cache first
    const cached = await idbGet(db, cacheKey);
    if (cached) {
      f16Data = new Uint16Array(cached);
      fromCache = true;
    } else {
      // Range request for this tensor
      const [start, end] = entry.data_offsets;
      const absStart = dataOffset + start;
      const absEnd = dataOffset + end - 1;
      const tensorBytes = end - start;

      const resp = await fetch(safetensorsUrl, {
        headers: { Range: `bytes=${absStart}-${absEnd}` },
      });
      if (!resp.ok && resp.status !== 206) {
        throw new Error(`Failed to fetch tensor ${name}: HTTP ${resp.status}`);
      }
      const buf = await resp.arrayBuffer();
      bytesDownloaded += tensorBytes;

      // Convert BF16 → F16
      if (entry.dtype === 'BF16') {
        const f16Buf = convertBF16toF16(buf);
        f16Data = new Uint16Array(f16Buf);
      } else if (entry.dtype === 'F16') {
        f16Data = new Uint16Array(buf);
      } else if (entry.dtype === 'F32') {
        const f32 = new Float32Array(buf);
        const f16 = new Uint16Array(f32.length);
        for (let j = 0; j < f32.length; j++) {
          f16[j] = float32ToFloat16(f32[j]);
        }
        f16Data = f16;
      } else {
        throw new Error(`Unsupported dtype for ${name}: ${entry.dtype}`);
      }

      // Cache in IndexedDB (store the underlying ArrayBuffer)
      await idbPut(db, cacheKey, f16Data.buffer as ArrayBuffer);
    }

    // Create GPU buffer
    const gpuBuf = createWeightBuffer(device, f16Data, name);
    const comp = results[component];
    comp.buffers.set(name, gpuBuf);
    comp.tensors.set(name, { shape: entry.shape, buffer: gpuBuf });

    // Release JS reference — GPU has its own copy
    // @ts-ignore - help GC
    f16Data = null!;

    if (onProgress) {
      onProgress({
        loaded: i + 1,
        total: totalTensors,
        component,
        tensor: name,
        cached: fromCache,
        bytesDownloaded,
      });
    }
  }

  db.close();

  return {
    backbone: results.backbone,
    fm: results.fm,
    codec: results.codec,
  };
}
