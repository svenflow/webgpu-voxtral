# voxtral-webgpu

[![npm](https://img.shields.io/npm/v/voxtral-webgpu)](https://www.npmjs.com/package/voxtral-webgpu)
[![license](https://img.shields.io/npm/l/voxtral-webgpu)](./LICENSE)

**Run Mistral's Voxtral 4B TTS in the browser via WebGPU. 20 voices, 9 languages, 24kHz audio. Zero dependencies.**

[**Live Demo**](https://svenflow.github.io/webgpu-voxtral/) | [npm](https://www.npmjs.com/package/voxtral-webgpu)

---

## Quick Start

```bash
npm install voxtral-webgpu
```

```typescript
import { VoxtralEngine, TekkenTokenizer } from 'voxtral-webgpu'

// Initialize engine and load weights
const engine = new VoxtralEngine({ maxSeqLen: 2048 })
await engine.init()
await engine.loadWeightsFromHF()

// Load tokenizer
const tokenizer = await TekkenTokenizer.load('/models/voxtral-tts/tekken.json')

// Build prompt and load voice embeddings
const { tokens, audioTokenStart, audioTokenCount } = tokenizer.buildTTSPrompt(
  'Hello from the browser!',
  'casual_female',
)
const voiceResp = await fetch('/models/voxtral-tts/voice_embedding_f32/casual_female.bin')
const voiceEmbeddings = new Float32Array(await voiceResp.arrayBuffer())

// Generate speech
const result = await engine.generate(
  tokens,
  audioTokenStart,
  audioTokenCount,
  voiceEmbeddings,
  500, // max frames
  (frame, semanticCode, acousticCodes) => {
    console.log(`Frame ${frame}: semantic=${semanticCode}`)
  },
)

// Play audio (24kHz mono)
const audioCtx = new AudioContext({ sampleRate: 24000 })
const buffer = audioCtx.createBuffer(1, result.audio.length, 24000)
buffer.getChannelData(0).set(result.audio)
const source = audioCtx.createBufferSource()
source.buffer = buffer
source.connect(audioCtx.destination)
source.start()
```

Create once, generate per request. Weights (~8GB BF16) are streamed from HuggingFace, converted to F16 on-the-fly, and cached in IndexedDB for instant reload.

## Benchmarks

[TODO: benchmark]

| Device | Backbone | FM | Codec | Total | RTF | Frames |
|--------|----------|----|-------|-------|-----|--------|
| Mac Mini M4 Pro, Chrome 134 | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| MacBook Air M2, Chrome 134 | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| NVIDIA RTX 4090, Chrome 134 | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |

[**Run this benchmark on your device →**](https://svenflow.github.io/webgpu-voxtral/)

## Features

- **~8GB weights** streamed from HuggingFace, converted BF16 → F16 in-browser, cached in IndexedDB
- **20 voices** across 9 languages (EN, FR, ES, DE, IT, PT, NL, AR, HI)
- **24kHz** mono audio output
- **50+ WGSL compute shaders** — pure WebGPU, no WASM, no ONNX Runtime
- **Zero dependencies** — single ES module, no runtime deps
- **Streaming progress** — per-frame callback during generation
- **Instant reload** — IndexedDB weight cache means first load is slow, subsequent loads are near-instant

## Install

```bash
npm install voxtral-webgpu
```

## API

### `VoxtralEngine`

The main engine class. Manages the WebGPU device, weight buffers, and compute pipelines.

#### `new VoxtralEngine(options?)`

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `config` | `VoxtralConfig` | `defaultConfig` | Model architecture config |
| `maxSeqLen` | `number` | `4096` | Maximum sequence length for KV cache |

#### `engine.init()`

Initialize the WebGPU device, allocate work buffers, and compile all compute pipelines. Must be called before loading weights.

Returns `Promise<void>`.

#### `engine.loadWeightsFromHF(safetensorsUrl?, onProgress?)`

Stream and load model weights from a safetensors file. Fetches BF16 tensors via HTTP range requests, converts to F16 in-browser, and caches each tensor in IndexedDB.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `safetensorsUrl` | `string` | HuggingFace CDN | URL to the consolidated safetensors file |
| `onProgress` | `(progress: HFLoadProgress) => void` | `undefined` | Progress callback per tensor |

Returns `Promise<void>`.

#### `engine.loadWeights(baseUrl, onProgress?)`

Load pre-converted weights from a local manifest (alternative to HuggingFace streaming).

| Parameter | Type | Description |
|-----------|------|-------------|
| `baseUrl` | `string` | Base URL containing the weight manifest and tensor files |
| `onProgress` | `(progress: WeightLoadProgress) => void` | Progress callback |

Returns `Promise<void>`.

#### `engine.generate(tokens, audioTokenStart, audioTokenCount, voiceEmbeddings, maxFrames?, onFrame?)`

Run the full 3-stage TTS pipeline: backbone prefill + autoregressive decode, FM flow-matching, and codec waveform synthesis.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tokens` | `number[]` | required | Token IDs from `tokenizer.buildTTSPrompt()` |
| `audioTokenStart` | `number` | required | Index where voice embedding tokens begin |
| `audioTokenCount` | `number` | required | Number of voice embedding tokens |
| `voiceEmbeddings` | `Float32Array \| null` | required | Pre-loaded voice embeddings `[audioTokenCount, 3072]` |
| `maxFrames` | `number` | `500` | Maximum audio frames to generate (500 = ~40s) |
| `onFrame` | `(frame, semanticCode, acousticCodes) => void` | `undefined` | Per-frame progress callback |

Returns `Promise<TTSResult>`:

```typescript
interface TTSResult {
  semanticCodes: number[]
  acousticCodes: number[][]
  audio: Float32Array       // 24kHz mono PCM
  stats: {
    backboneMs: number
    fmMs: number
    codecMs: number
    totalMs: number
    framesGenerated: number
  }
}
```

#### `engine.isReady`

`boolean` — `true` when the device, pipelines, and weights are all loaded.

#### `engine.destroy()`

Release all GPU resources (buffers, device).

### `TekkenTokenizer`

Mistral's Tekken BPE tokenizer (v7). Handles text encoding and TTS prompt construction.

#### `TekkenTokenizer.load(url)`

Load tokenizer data from a JSON file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `url` | `string` | URL to `tekken.json` |

Returns `Promise<TekkenTokenizer>`.

#### `tokenizer.buildTTSPrompt(text, voice)`

Build the full token sequence for TTS generation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `string` | Text to synthesize |
| `voice` | `string` | Voice name (e.g. `'casual_female'`, `'amandine'`) |

Returns `{ tokens: number[], audioTokenStart: number, audioTokenCount: number }`.

#### `tokenizer.voices`

`string[]` — list of available voice names.

### `TOKENS`

Special token ID constants:

```typescript
const TOKENS = {
  BOS: 1,
  EOS: 2,
  AUDIO: 24,
  BEGIN_AUDIO: 25,
  OUTPUT_AUDIO: 26,
  AUDIO_TO_TEXT: 35,
  TEXT_TO_AUDIO: 36,
  // ...
}
```

### `clearWeightCache()`

Delete all cached weight tensors from IndexedDB. Call this to force re-download on next load.

Returns `Promise<void>`.

## Requirements

| Browser | Status |
|---------|--------|
| Chrome 113+ | Supported |
| Edge 113+ | Supported |
| Safari 18+ (macOS) | Experimental |
| Firefox | Not supported (no WebGPU) |

**Hardware:** Requires a GPU with at least 8GB VRAM. Tested on Apple Silicon (M1+) and NVIDIA discrete GPUs. Integrated GPUs may not have enough memory.

**Note:** WebGPU is a desktop-first API. Mobile browsers generally lack the GPU memory required for a 4B parameter model.

## How It Works

```
Text + Voice → Backbone (Ministral 3B) → FM Transformer → Codec Decoder → 24kHz Audio
```

**1. Backbone (Ministral 3B LLM):** Takes the text tokens and voice embeddings as input. Runs autoregressive generation to produce one hidden state per audio frame. 26 transformer layers, 3072-dim, GQA with 32 heads / 8 KV heads.

**2. FM Transformer (Flow Matching):** Takes each hidden state and generates semantic + acoustic codes via an Euler ODE solver (8 steps). 3 transformer layers with classifier-free guidance. Produces 1 semantic code + 36 acoustic codes per frame.

**3. Codec Decoder (Mimi):** Converts the code sequence into a 24kHz waveform. 4-stage upsampling with transposed convolutions and transformer blocks. Each frame produces 1920 audio samples (80ms at 24kHz).

All three stages run entirely on the GPU via WebGPU compute shaders. No CPU-side inference, no WASM, no ONNX.

## Voices

| Language | Voices |
|----------|--------|
| English | casual_female, casual_male, cheerful_female, neutral_female, neutral_male |
| French | fr_female, fr_male |
| Spanish | es_female, es_male |
| German | de_female, de_male |
| Italian | it_female, it_male |
| Portuguese | pt_female, pt_male |
| Dutch | nl_female, nl_male |
| Hindi | hi_female, hi_male |
| Arabic | ar_male |

## Development

```bash
git clone https://github.com/svenflow/webgpu-voxtral.git
cd webgpu-voxtral
npm install
npm run dev    # Watch mode
npm run build  # Production build
```

## License

CC BY-NC — following Mistral's model license.
