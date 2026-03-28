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
import { Voxtral } from 'voxtral-webgpu'

// Load model (~8GB, cached in IndexedDB after first load)
const tts = await Voxtral.load()

// Generate speech
const { audio } = await tts.speak('Hello from the browser!', 'casual_female')

// Play it
const ctx = new AudioContext({ sampleRate: 24000 })
const buf = ctx.createBuffer(1, audio.length, 24000)
buf.getChannelData(0).set(audio)
const src = ctx.createBufferSource()
src.buffer = buf
src.connect(ctx.destination)
src.start()
```

That's it. Weights are streamed from HuggingFace, converted BF16 → F16 on the fly, and cached in IndexedDB. Second load is near-instant.

## API

### `Voxtral.load(options?)`

Initialize WebGPU and load model weights.

```typescript
const tts = await Voxtral.load({
  onProgress: (p) => console.log(`Loading: ${p.loaded}/${p.total} tensors`),
})
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `maxSeqLen` | `number` | `4096` | Max sequence length for KV cache |
| `weightsUrl` | `string` | HuggingFace CDN | URL to safetensors file |
| `modelsUrl` | `string` | HuggingFace | Base URL for tokenizer & voice files |
| `onProgress` | `(HFLoadProgress) => void` | — | Weight loading progress callback |

### `tts.speak(text, voice?, options?)`

Generate speech. Returns a `TTSResult` with `.audio` (Float32Array, 24kHz mono) and `.stats`.

```typescript
const { audio, stats } = await tts.speak('Bonjour le monde!', 'fr_female')
console.log(`${stats.framesGenerated} frames in ${stats.totalMs.toFixed(0)}ms`)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `string` | required | Text to synthesize |
| `voice` | `string` | `'casual_female'` | Voice name (see [Voices](#voices)) |
| `options.maxFrames` | `number` | `500` | Max frames (~40s of audio) |
| `options.onFrame` | `function` | — | Per-frame progress callback |

### `tts.voices`

Array of available voice names.

```typescript
console.log(tts.voices)
// ['casual_female', 'casual_male', 'fr_female', ...]
```

### `tts.destroy()`

Release all GPU resources.

### `clearWeightCache()`

Delete cached weights from IndexedDB to force re-download.

```typescript
import { clearWeightCache } from 'voxtral-webgpu'
await clearWeightCache()
```

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

## Requirements

| Browser | Status |
|---------|--------|
| Chrome 113+ | Supported |
| Edge 113+ | Supported |
| Safari 18+ (macOS) | Experimental |
| Firefox | Not supported (no WebGPU) |

**Hardware:** GPU with 8GB+ VRAM. Tested on Apple Silicon (M1+) and NVIDIA discrete GPUs.

## Advanced Usage

For fine-grained control over the engine, tokenizer, and weight loading:

```typescript
import { VoxtralEngine, TekkenTokenizer } from 'voxtral-webgpu'

const engine = new VoxtralEngine({ maxSeqLen: 2048 })
await engine.init()
await engine.loadWeightsFromHF(undefined, (p) => {
  console.log(`${p.loaded}/${p.total}`)
})

const tokenizer = await TekkenTokenizer.load(
  'https://huggingface.co/mistralai/Voxtral-4B-TTS-2603/resolve/main/tekken.json'
)

const { tokens, audioTokenStart, audioTokenCount } =
  tokenizer.buildTTSPrompt('Hello!', 'casual_female')

const voiceResp = await fetch(
  'https://huggingface.co/mistralai/Voxtral-4B-TTS-2603/resolve/main/voice_embedding_f32/casual_female.bin'
)
const voiceEmbeddings = new Float32Array(await voiceResp.arrayBuffer())

const result = await engine.generate(
  tokens, audioTokenStart, audioTokenCount,
  voiceEmbeddings, 500,
)
```

## How It Works

```
Text + Voice → Backbone (Ministral 3B) → FM Transformer → Codec Decoder → 24kHz Audio
```

1. **Backbone (Ministral 3B):** Autoregressive transformer generates one hidden state per audio frame. 26 layers, 3072-dim, GQA.
2. **FM Transformer:** Flow-matching ODE solver (8 steps) produces semantic + acoustic codes per frame. 3 layers with classifier-free guidance.
3. **Codec Decoder (Mimi):** Converts codes to 24kHz waveform via transposed convolutions. Each frame = 1920 samples (80ms).

All three stages run entirely on the GPU via 50+ WGSL compute shaders. No WASM, no ONNX.

## Benchmarks

Run the [live demo](https://svenflow.github.io/webgpu-voxtral/) to benchmark on your device.

## Development

```bash
git clone https://github.com/svenflow/webgpu-voxtral.git
cd webgpu-voxtral
npm install
npm run dev    # Watch mode
npm run build  # Production build
```

## License

CC BY-NC 4.0 — following Mistral's model license.
