/**
 * Minimal .npy file reader for loading reference activations in tests.
 * Supports float32 and int32/uint32 arrays (little-endian).
 */

export interface NpyArray {
  dtype: string;
  shape: number[];
  data: Float32Array | Int32Array | Uint32Array;
}

/**
 * Parse a .npy file from an ArrayBuffer.
 * Supports NumPy format 1.0 and 2.0.
 */
export function parseNpy(buffer: ArrayBuffer): NpyArray {
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);

  // Magic: \x93NUMPY
  if (bytes[0] !== 0x93 || bytes[1] !== 0x4e || bytes[2] !== 0x55 ||
      bytes[3] !== 0x4d || bytes[4] !== 0x50 || bytes[5] !== 0x59) {
    throw new Error('Not a valid .npy file');
  }

  const major = bytes[6];
  const minor = bytes[7];

  let headerLen: number;
  let headerOffset: number;

  if (major === 1) {
    headerLen = view.getUint16(8, true);
    headerOffset = 10;
  } else if (major === 2) {
    headerLen = view.getUint32(8, true);
    headerOffset = 12;
  } else {
    throw new Error(`Unsupported npy version: ${major}.${minor}`);
  }

  const headerStr = new TextDecoder().decode(
    bytes.slice(headerOffset, headerOffset + headerLen)
  );

  // Parse the Python dict header, e.g.:
  // {'descr': '<f4', 'fortran_order': False, 'shape': (3072,), }
  const descrMatch = headerStr.match(/'descr'\s*:\s*'([^']+)'/);
  const shapeMatch = headerStr.match(/'shape'\s*:\s*\(([^)]*)\)/);
  const fortranMatch = headerStr.match(/'fortran_order'\s*:\s*(True|False)/);

  if (!descrMatch || !shapeMatch) {
    throw new Error(`Cannot parse npy header: ${headerStr}`);
  }

  const descr = descrMatch[1];
  const shapeStr = shapeMatch[1].trim();
  const shape = shapeStr === ''
    ? []
    : shapeStr.split(',').filter(s => s.trim() !== '').map(s => parseInt(s.trim()));

  const fortranOrder = fortranMatch ? fortranMatch[1] === 'True' : false;
  if (fortranOrder) {
    throw new Error('Fortran-order arrays not supported');
  }

  const dataOffset = headerOffset + headerLen;
  const dataBytes = buffer.slice(dataOffset);

  let data: Float32Array | Int32Array | Uint32Array;

  switch (descr) {
    case '<f4':
    case '=f4':
      data = new Float32Array(dataBytes);
      break;
    case '<f8':
    case '=f8': {
      // Convert float64 → float32
      const f64 = new Float64Array(dataBytes);
      data = new Float32Array(f64.length);
      for (let i = 0; i < f64.length; i++) data[i] = f64[i];
      break;
    }
    case '<i4':
    case '=i4':
      data = new Int32Array(dataBytes);
      break;
    case '<u4':
    case '=u4':
      data = new Uint32Array(dataBytes);
      break;
    case '<i8':
    case '=i8': {
      // Convert int64 → int32 (safe for our use case with small values)
      const i64 = new BigInt64Array(dataBytes);
      data = new Int32Array(i64.length);
      for (let i = 0; i < i64.length; i++) data[i] = Number(i64[i]);
      break;
    }
    default:
      throw new Error(`Unsupported dtype: ${descr}`);
  }

  return { dtype: descr, shape, data };
}

/**
 * Load a .npy file from a URL.
 */
export async function loadNpy(url: string): Promise<NpyArray> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to fetch ${url}: ${resp.status}`);
  const buffer = await resp.arrayBuffer();
  return parseNpy(buffer);
}

/**
 * Compare two arrays with allclose semantics.
 * Returns { passed, maxAbsDiff, maxRelDiff, mismatchCount }
 */
export function allclose(
  actual: Float32Array | Int32Array | Uint32Array,
  expected: Float32Array | Int32Array | Uint32Array,
  atol: number = 0.01,
  rtol: number = 0.01,
): {
  passed: boolean;
  maxAbsDiff: number;
  maxRelDiff: number;
  mismatchCount: number;
  totalCount: number;
} {
  if (actual.length !== expected.length) {
    return {
      passed: false,
      maxAbsDiff: Infinity,
      maxRelDiff: Infinity,
      mismatchCount: actual.length,
      totalCount: expected.length,
    };
  }

  let maxAbsDiff = 0;
  let maxRelDiff = 0;
  let mismatchCount = 0;

  for (let i = 0; i < actual.length; i++) {
    const a = actual[i];
    const e = expected[i];
    const absDiff = Math.abs(a - e);
    const relDiff = Math.abs(e) > 1e-8 ? absDiff / Math.abs(e) : 0;

    maxAbsDiff = Math.max(maxAbsDiff, absDiff);
    maxRelDiff = Math.max(maxRelDiff, relDiff);

    if (absDiff > atol + rtol * Math.abs(e)) {
      mismatchCount++;
    }
  }

  return {
    passed: mismatchCount === 0,
    maxAbsDiff,
    maxRelDiff,
    mismatchCount,
    totalCount: actual.length,
  };
}
