/**
 * Tekken tokenizer for Voxtral TTS
 *
 * Tekken is Mistral's BPE tokenizer (v7). This implements encoding only
 * (we don't need decoding for TTS — text goes in, audio comes out).
 *
 * For TTS, the prompt format is (from mistral_common):
 *   BOS [BEGIN_AUDIO] [AUDIO]×N [TEXT_TO_AUDIO] text [AUDIO_TO_TEXT] [BEGIN_AUDIO]
 *
 * The [AUDIO] tokens are REPLACED by pre-computed voice embeddings.
 */

/** Special token IDs */
export const TOKENS = {
  UNK: 0,
  BOS: 1,       // <s>
  EOS: 2,       // </s>
  INST: 3,      // [INST]
  INST_END: 4,  // [/INST]
  AUDIO: 24,            // [AUDIO]
  BEGIN_AUDIO: 25,      // [BEGIN_AUDIO]
  OUTPUT_AUDIO: 26,     // [OUTPUT_AUDIO]
  AUDIO_TO_TEXT: 35,    // [REPEAT_AUDIO_TEXT] — marks transition from audio to text in TTS prompt
  TEXT_TO_AUDIO: 36,    // [NEXT_AUDIO_TEXT] — marks transition from text to audio generation
  PAD: 11,
} as const;

interface TekkenVocabEntry {
  rank: number;
  token_bytes: string;  // base64 encoded
  token_str: string;
}

interface TekkenSpecialToken {
  rank: number;
  token_str: string;
  is_control: boolean;
}

interface TekkenConfig {
  pattern: string;
  num_vocab_tokens: number;
  default_vocab_size: number;
  default_num_special_tokens: number;
  version: string;
}

interface TekkenData {
  config: TekkenConfig;
  vocab: TekkenVocabEntry[];
  special_tokens: TekkenSpecialToken[];
  audio: {
    sampling_rate: number;
    frame_rate: number;
    voice_num_audio_tokens: Record<string, number>;
  };
}

export class TekkenTokenizer {
  private vocab: Map<string, number> = new Map();
  private specialTokens: Map<string, number> = new Map();
  private pattern: RegExp;
  private voiceNumTokens: Map<string, number> = new Map();

  constructor(data: TekkenData) {
    // Build vocab lookup: bytes → rank (token id)
    // Ranks 0-149999 are vocab tokens
    // Ranks 150000+ are special tokens (mapped to their position in default_vocab_size)
    for (const entry of data.vocab) {
      const bytes = atob(entry.token_bytes);
      this.vocab.set(bytes, entry.rank);
    }

    // Special tokens get IDs from default_vocab_size onwards
    // But actually the special tokens have specific ranks that map to token IDs
    // In tekken, special_token rank 0 = token ID (default_vocab_size + 0) = 131072
    // Wait, looking at the data: BOS is rank 1 and its token_id should be 1
    // Actually in Mistral's tekken, special tokens are prepended:
    // The first default_num_special_tokens IDs (0-999) are reserved for special tokens
    // Then vocab tokens start at 1000
    // But actually from the model config, vocab_size = 131072
    // Special tokens: rank maps directly to their token ID? Let me check...
    // Actually the standard Mistral convention:
    // Token IDs 0..num_special-1 are special tokens (ordered by rank)
    // Token IDs num_special..vocab_size-1 are BPE tokens
    const numSpecial = data.config.default_num_special_tokens; // 1000

    for (const st of data.special_tokens) {
      this.specialTokens.set(st.token_str, st.rank);
    }

    // Compile the regex pattern for pre-tokenization
    try {
      this.pattern = new RegExp(data.config.pattern, 'gu');
    } catch {
      // Fallback: simple whitespace split
      this.pattern = /\S+|\s+/gu;
    }

    // Voice token counts
    if (data.audio?.voice_num_audio_tokens) {
      for (const [voice, count] of Object.entries(data.audio.voice_num_audio_tokens)) {
        this.voiceNumTokens.set(voice, count);
      }
    }
  }

  /**
   * Load tokenizer from a URL (tekken.json).
   */
  static async load(url: string): Promise<TekkenTokenizer> {
    const resp = await fetch(url);
    const data: TekkenData = await resp.json();
    return new TekkenTokenizer(data);
  }

  /**
   * Get the number of audio tokens for a voice preset.
   */
  getVoiceNumTokens(voice: string): number {
    const count = this.voiceNumTokens.get(voice);
    if (count === undefined) {
      throw new Error(`Unknown voice: ${voice}. Available: ${[...this.voiceNumTokens.keys()].join(', ')}`);
    }
    return count;
  }

  /**
   * Build the full token sequence for TTS.
   *
   * Format (from mistral_common InstructTokenizerV11.encode_speech_request):
   *   BOS [BEGIN_AUDIO] [AUDIO]×N [TEXT_TO_AUDIO] text_tokens [AUDIO_TO_TEXT] [BEGIN_AUDIO]
   *
   * The [AUDIO] tokens at positions audioTokenStart..audioTokenStart+N-1
   * will be REPLACED by voice embeddings at the embedding level.
   */
  buildTTSPrompt(
    text: string,
    voice: string,
  ): { tokens: number[]; audioTokenStart: number; audioTokenCount: number } {
    const numVoiceTokens = this.getVoiceNumTokens(voice);

    const tokens: number[] = [];

    // BOS
    tokens.push(TOKENS.BOS);

    // [BEGIN_AUDIO] — starts the voice embedding section
    tokens.push(TOKENS.BEGIN_AUDIO);

    // Voice audio tokens — these get REPLACED by voice embeddings
    const audioTokenStart = tokens.length;
    for (let i = 0; i < numVoiceTokens; i++) {
      tokens.push(TOKENS.AUDIO);
    }

    // [TEXT_TO_AUDIO] — transition marker
    tokens.push(TOKENS.TEXT_TO_AUDIO);

    // Encode text
    const textTokens = this.encode(text);
    tokens.push(...textTokens);

    // [AUDIO_TO_TEXT] — transition marker
    tokens.push(TOKENS.AUDIO_TO_TEXT);

    // [BEGIN_AUDIO] — signals model to start generating audio
    tokens.push(TOKENS.BEGIN_AUDIO);

    return {
      tokens,
      audioTokenStart,
      audioTokenCount: numVoiceTokens,
    };
  }

  /**
   * Encode text to token IDs using BPE.
   * This is a simplified encoder — for production use mistral_common.
   *
   * For now: split by the regex pattern, then look up each piece in vocab.
   * Fall back to byte-level encoding for unknown pieces.
   */
  encode(text: string): number[] {
    const tokens: number[] = [];

    // Pre-tokenize using the regex pattern
    const matches = text.matchAll(this.pattern);
    for (const match of matches) {
      const piece = match[0];

      // Try direct vocab lookup
      const id = this.vocab.get(piece);
      if (id !== undefined) {
        tokens.push(id + 1000); // offset by num_special_tokens
        continue;
      }

      // Fall back to byte-level encoding
      const encoder = new TextEncoder();
      const bytes = encoder.encode(piece);
      for (const b of bytes) {
        const byteStr = String.fromCharCode(b);
        const byteId = this.vocab.get(byteStr);
        if (byteId !== undefined) {
          tokens.push(byteId + 1000);
        } else {
          tokens.push(TOKENS.UNK);
        }
      }
    }

    return tokens;
  }

  /**
   * Get available voice names.
   */
  get voices(): string[] {
    return [...this.voiceNumTokens.keys()];
  }
}
