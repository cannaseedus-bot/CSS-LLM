import { MiniTensorShell } from "./mini-tensor-shell.js";
import { loadScxHeader } from "./utils/scx_loader.js";
import { hashUint32 } from "./utils/hash.js";

export class TensorShell {
  constructor(config = {}) {
    this.core = new MiniTensorShell(config);
    this.loadedHeader = null;
  }

  async loadWeights(path) {
    this.loadedHeader = await loadScxHeader(path);
    await this.core.loadWeights(path);
    return this.loadedHeader;
  }

  async generate(prompt, options = {}) {
    const result = await this.core.generate(prompt, options);
    return {
      ...result,
      replayHash: hashUint32(result.outputTokens),
      scx: this.loadedHeader,
    };
  }
}
