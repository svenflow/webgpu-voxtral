/**
 * voxtral-webgpu — Voxtral TTS in the browser via pure WebGPU
 *
 * Simple API:
 *   const tts = await Voxtral.load()
 *   const { audio } = await tts.speak("Hello!", "casual_female")
 *
 * Advanced API (full control):
 *   import { VoxtralEngine, TekkenTokenizer } from 'voxtral-webgpu'
 */

// ── High-level API (recommended) ──
export { Voxtral } from './voxtral.js';
export type { VoxtralOptions, SpeakOptions } from './voxtral.js';

// ── Advanced API ──
export { VoxtralEngine } from './engine.js';
export type { EngineOptions, TTSResult } from './engine.js';
export { TekkenTokenizer, TOKENS } from './tokenizer.js';

// ── Utilities ──
export { clearWeightCache, getWeightCacheInfo, HF_VOXTRAL_URL } from './weights.js';
export type { HFLoadProgress, WeightLoadProgress } from './weights.js';
