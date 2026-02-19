import { BufferManager } from "./buffer-manager.js";
import { KVCache } from "./kv-cache.js";

export class TensorShell {
  constructor(config) {
    this.config = config;
    this.device = null;
    this.bufferManager = null;
    this.kvCache = new KVCache(8192);
  }

  async initialize() {
    if (!navigator.gpu) throw new Error("WebGPU required");
    const adapter = await navigator.gpu.requestAdapter();
    this.device = await adapter.requestDevice();
    this.bufferManager = new BufferManager(this.device);

    this.bufferManager.create(
      "input",
      this.config.maxBatchSize * this.config.maxSeqLen * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    );

    this.bufferManager.create(
      "output",
      this.config.maxBatchSize * this.config.maxSeqLen * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    );

    return this;
  }
}
