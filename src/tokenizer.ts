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
  private bytesToRank: Map<string, number> = new Map();
  private specialTokens: Map<string, number> = new Map();
  private pattern: RegExp;
  private voiceNumTokens: Map<string, number> = new Map();
  private numSpecialTokens: number;

  constructor(data: TekkenData) {
    this.numSpecialTokens = data.config.default_num_special_tokens; // 1000

    // Build vocab lookup: bytes (as raw string) → rank
    // Ranks are ordered by merge priority (lower = more common)
    // Ranks 0-255 are individual bytes, 256+ are BPE merges
    for (const entry of data.vocab) {
      const bytes = atob(entry.token_bytes);
      this.bytesToRank.set(bytes, entry.rank);
    }

    // Special tokens
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
   * Encode text to token IDs using proper BPE merging.
   *
   * Algorithm:
   * 1. Pre-tokenize using regex pattern (splits into words/chunks)
   * 2. For each chunk, convert to bytes
   * 3. Iteratively merge the byte pair with the lowest vocab rank
   * 4. Map final merged tokens to IDs (rank + num_special_tokens)
   */
  encode(text: string): number[] {
    const tokens: number[] = [];

    // Pre-tokenize using the regex pattern
    const matches = text.matchAll(this.pattern);
    for (const match of matches) {
      const piece = match[0];
      const pieceTokens = this.bpeEncode(piece);
      tokens.push(...pieceTokens);
    }

    return tokens;
  }

  /**
   * BPE encode a single pre-tokenized piece.
   *
   * Starts with individual bytes and iteratively merges the pair
   * whose concatenation has the lowest rank in the vocab.
   */
  private bpeEncode(piece: string): number[] {
    // Convert to bytes as raw string chars
    const encoder = new TextEncoder();
    const bytes = encoder.encode(piece);

    // Start with individual byte strings
    let parts: string[] = [];
    for (const b of bytes) {
      parts.push(String.fromCharCode(b));
    }

    // If single byte, just return it
    if (parts.length <= 1) {
      const rank = this.bytesToRank.get(parts[0]);
      return [rank !== undefined ? rank + this.numSpecialTokens : TOKENS.UNK];
    }

    // Iteratively merge the pair with the lowest rank
    while (parts.length > 1) {
      // Find the pair whose merge has the lowest rank
      let bestRank = Infinity;
      let bestIdx = -1;

      for (let i = 0; i < parts.length - 1; i++) {
        const merged = parts[i] + parts[i + 1];
        const rank = this.bytesToRank.get(merged);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestIdx = i;
        }
      }

      // No more merges possible
      if (bestIdx === -1) break;

      // Apply the merge
      const newParts: string[] = [];
      for (let i = 0; i < parts.length; i++) {
        if (i === bestIdx) {
          newParts.push(parts[i] + parts[i + 1]);
          i++; // skip next
        } else {
          newParts.push(parts[i]);
        }
      }
      parts = newParts;
    }

    // Convert final parts to token IDs
    return parts.map(p => {
      const rank = this.bytesToRank.get(p);
      return rank !== undefined ? rank + this.numSpecialTokens : TOKENS.UNK;
    });
  }

  /**
   * Get available voice names.
   */
  get voices(): string[] {
    return [...this.voiceNumTokens.keys()];
  }
}
