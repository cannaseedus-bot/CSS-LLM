import fs from "node:fs/promises";

function hashTokens(tokens) {
  let h = 2166136261 >>> 0;
  for (const t of tokens) {
    h ^= t >>> 0;
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0;
}

function xorshift32(state) {
  let x = state >>> 0;
  x ^= (x << 13) >>> 0;
  x ^= x >>> 17;
  x ^= (x << 5) >>> 0;
  return x >>> 0;
}

export class MiniTensorShell {
  constructor(config = {}) {
    this.config = {
      modelName: "css-llm-mini",
      hiddenDim: 768,
      numLayers: 12,
      numHeads: 12,
      headDim: 64,
      ffnDim: 3072,
      vocabSize: 32000,
      contextLength: 1024,
      temperature: 0.8,
      topP: 0.95,
      ...config,
    };
    this.weights = null;
  }

  async loadWeights(scxPath) {
    const bytes = await fs.readFile(scxPath);
    const magic = bytes.subarray(0, 4).toString("utf8");
    if (magic !== "SCXM") {
      throw new Error("Invalid SCX header: expected SCXM");
    }

    this.weights = {
      path: scxPath,
      bytes: bytes.byteLength,
    };

    return this.weights;
  }

  async generate(prompt, { maxTokens = 50 } = {}) {
    if (!this.weights) {
      throw new Error("Weights must be loaded before generate().");
    }

    const inputTokens = Array.from(prompt).map((ch) => ch.charCodeAt(0) % this.config.vocabSize);
    let seed = hashTokens(inputTokens);
    const sampled = [];

    const steps = Math.min(maxTokens, this.config.contextLength);
    for (let i = 0; i < steps; i++) {
      seed = xorshift32(seed);
      sampled.push(seed % this.config.vocabSize);
    }

    return {
      prompt,
      inputTokens,
      outputTokens: sampled,
      seed,
      deterministic: true,
    };
  }
}
