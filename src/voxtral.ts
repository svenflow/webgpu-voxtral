/**
 * High-level Voxtral TTS API
 *
 * Simple interface: load() → speak(text, voice) → audio
 */

import { VoxtralEngine, TTSResult } from './engine.js';
import { TekkenTokenizer } from './tokenizer.js';
import { HFLoadProgress, HF_VOXTRAL_URL } from './weights.js';

/** Default base URL for tokenizer and voice embeddings on HuggingFace */
const HF_MODELS_BASE =
  'https://huggingface.co/mistralai/Voxtral-4B-TTS-2603/resolve/main';

/** Options for Voxtral.load() */
export interface VoxtralOptions {
  /** Max sequence length for KV cache (default 4096) */
  maxSeqLen?: number;
  /** URL to the consolidated.safetensors file (default: HuggingFace CDN) */
  weightsUrl?: string;
  /** Base URL for tokenizer and voice embeddings (default: HuggingFace) */
  modelsUrl?: string;
  /** Progress callback for weight loading */
  onProgress?: (progress: HFLoadProgress) => void;
}

/** Options for speak() */
export interface SpeakOptions {
  /** Maximum audio frames to generate (default 500 ≈ 40s of audio) */
  maxFrames?: number;
  /** Per-frame progress callback */
  onFrame?: (frame: number, semanticCode: number, acousticCodes: Uint32Array) => void;
}

/**
 * High-level Voxtral TTS interface.
 *
 * ```ts
 * const tts = await Voxtral.load()
 * const { audio } = await tts.speak("Hello from the browser!", "casual_female")
 * ```
 */
export class Voxtral {
  private engine: VoxtralEngine;
  private tokenizer: TekkenTokenizer;
  private modelsUrl: string;
  private voiceCache = new Map<string, Float32Array>();

  private constructor(
    engine: VoxtralEngine,
    tokenizer: TekkenTokenizer,
    modelsUrl: string,
  ) {
    this.engine = engine;
    this.tokenizer = tokenizer;
    this.modelsUrl = modelsUrl;
  }

  /**
   * Initialize Voxtral TTS.
   *
   * Sets up WebGPU, streams ~8GB of model weights (BF16 → F16, cached in
   * IndexedDB after first load), and loads the tokenizer.
   *
   * @example
   * ```ts
   * const tts = await Voxtral.load({
   *   onProgress: (p) => console.log(`${p.loaded}/${p.total} tensors`)
   * })
   * ```
   */
  static async load(options: VoxtralOptions = {}): Promise<Voxtral> {
    const modelsUrl = options.modelsUrl ?? HF_MODELS_BASE;

    const engine = new VoxtralEngine({ maxSeqLen: options.maxSeqLen });
    await engine.init();

    // Load weights and tokenizer in parallel
    const [, tokenizer] = await Promise.all([
      engine.loadWeightsFromHF(options.weightsUrl ?? HF_VOXTRAL_URL, options.onProgress),
      TekkenTokenizer.load(`${modelsUrl}/tekken.json`),
    ]);

    return new Voxtral(engine, tokenizer, modelsUrl);
  }

  /** Available voice names (e.g. "casual_female", "fr_male") */
  get voices(): string[] {
    return this.tokenizer.voices;
  }

  /**
   * Generate speech from text.
   *
   * @param text - Text to speak
   * @param voice - Voice name (default "casual_female"). See `.voices` for options.
   * @param options - Generation options (maxFrames, onFrame callback)
   * @returns TTSResult with `.audio` (Float32Array, 24kHz mono) and `.stats`
   *
   * @example
   * ```ts
   * const { audio, stats } = await tts.speak("Hello!", "neutral_male")
   * console.log(`Generated in ${stats.totalMs}ms`)
   *
   * // Play with Web Audio API
   * const ctx = new AudioContext({ sampleRate: 24000 })
   * const buf = ctx.createBuffer(1, audio.length, 24000)
   * buf.getChannelData(0).set(audio)
   * const src = ctx.createBufferSource()
   * src.buffer = buf
   * src.connect(ctx.destination)
   * src.start()
   * ```
   */
  async speak(
    text: string,
    voice = 'casual_female',
    options: SpeakOptions = {},
  ): Promise<TTSResult> {
    // Build token prompt
    const { tokens, audioTokenStart, audioTokenCount } =
      this.tokenizer.buildTTSPrompt(text, voice);

    // Load voice embeddings (cached after first fetch)
    let voiceEmbeddings = this.voiceCache.get(voice);
    if (!voiceEmbeddings) {
      const resp = await fetch(
        `${this.modelsUrl}/voice_embedding_f32/${voice}.bin`,
      );
      if (!resp.ok) {
        throw new Error(
          `Failed to load voice "${voice}": ${resp.status} ${resp.statusText}`,
        );
      }
      voiceEmbeddings = new Float32Array(await resp.arrayBuffer());
      this.voiceCache.set(voice, voiceEmbeddings);
    }

    return this.engine.generate(
      tokens,
      audioTokenStart,
      audioTokenCount,
      voiceEmbeddings,
      options.maxFrames ?? 500,
      options.onFrame,
    );
  }

  /** Release all GPU resources. */
  destroy(): void {
    this.engine.destroy();
    this.voiceCache.clear();
  }
}
