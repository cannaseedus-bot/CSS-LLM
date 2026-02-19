// tensor-shell-minimal.js
// Minimal 1B browser execution shell with CSS-governed config parsing.

export class MinimalTensorShell {
  constructor(governanceCSS) {
    this.config = this.parseGovernance(governanceCSS);
    this.device = null;
    this.buffers = new Map();
    this.pipelines = new Map();
    this.kvCache = new PagedCache(this.config);
  }

  parseGovernance(css) {
    const styles = css.match(/:root\s*{([^}]*)}/)?.[1] || "";
    const vars = {};

    styles.split(";").forEach((line) => {
      const match = line.match(/--([^:]+):\s*(.*)/);
      if (match) vars[match[1].trim()] = match[2].trim().replace(/^"|"$/g, "");
    });

    return {
      hiddenDim: parseInt(vars["hidden-dim"], 10),
      intermediateDim: parseInt(vars["intermediate-dim"], 10),
      numLayers: parseInt(vars["num-layers"], 10),
      numHeads: parseInt(vars["num-attention-heads"], 10),
      numKVHeads: parseInt(vars["num-kv-heads"], 10),
      headDim: parseInt(vars["head-dim"], 10),
      vocabSize: parseInt(vars["vocab-size"], 10),
      maxSeqLen: parseInt(vars["max-seq-len"], 10),
      maxBatchSize: parseInt(vars["max-batch-size"], 10),
      cachePageSize: parseInt(vars["cache-page-size"], 10),
      cacheMaxPages: parseInt(vars["cache-max-pages"], 10),
      workgroupSize: parseInt(vars["workgroup-size-x"], 10),
      temperature: parseFloat(vars["sampling-temperature"]),
      topK: parseInt(vars["sampling-top-k"], 10),
      topP: parseFloat(vars["sampling-top-p"]),
      attentionDispatch: vars["micronaut-attention-dispatch"],
      ffnDispatch: vars["micronaut-ffn-dispatch"],
    };
  }

  async initialize() {
    if (!navigator.gpu) throw new Error("WebGPU required");

    const adapter = await navigator.gpu.requestAdapter();
    this.device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: 2.5 * 1024 * 1024 * 1024,
        maxComputeWorkgroupSizeX: this.config.workgroupSize,
      },
    });

    this.createBuffers();
    await this.createPipelines();
    return this;
  }

  createBuffers() {
    const c = this.config;
    const weightSize = (c.hiddenDim * c.hiddenDim * c.numLayers * 4) / 8;

    this.buffers.set(
      "weights",
      this.device.createBuffer({
        size: weightSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      }),
    );

    const weightArray = new Uint32Array(this.buffers.get("weights").getMappedRange());
    weightArray.fill(0);
    this.buffers.get("weights").unmap();

    const kvSize = c.cacheMaxPages * c.cachePageSize * 2;
    this.buffers.set(
      "kv-cache",
      this.device.createBuffer({
        size: kvSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      }),
    );

    this.buffers.set(
      "inputs",
      this.device.createBuffer({
        size: c.maxBatchSize * c.maxSeqLen * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
    );

    this.buffers.set(
      "outputs",
      this.device.createBuffer({
        size: c.maxBatchSize * c.maxSeqLen * c.vocabSize * 2,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      }),
    );

    this.buffers.set(
      "config",
      this.device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
    );
  }

  async createPipelines() {
    const c = this.config;
    this.pipelines.set(
      "attention",
      await this.device.createComputePipelineAsync({
        layout: "auto",
        compute: {
          module: this.device.createShaderModule({
            code: `
              struct Config { layer: u32, scale: f32, seq_len: u32, batch_size: u32 }
              @group(0) @binding(0) var<storage> weights: array<u32>;
              @group(0) @binding(1) var<storage, read_write> kv_cache: array<u32>;
              @group(0) @binding(2) var<storage> inputs: array<u32>;
              @group(0) @binding(3) var<storage, read_write> outputs: array<f32>;
              @group(0) @binding(4) var<uniform> config: Config;
              var<workgroup> q_tile: array<f32, ${c.headDim}>;
              @compute @workgroup_size(${c.workgroupSize})
              fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
                if (id.y >= ${c.numHeads}u) { return; }
                if (lid.x < ${c.headDim}u) { q_tile[lid.x] = 0.0; }
                workgroupBarrier();
                let out_idx = id.x * ${c.hiddenDim}u + id.y * ${c.headDim}u + min(lid.x, ${c.headDim}u - 1u);
                outputs[out_idx] = q_tile[min(lid.x, ${c.headDim}u - 1u)];
              }
            `,
          }),
          entryPoint: "main",
        },
      }),
    );

    this.pipelines.set(
      "ffn",
      await this.device.createComputePipelineAsync({
        layout: "auto",
        compute: {
          module: this.device.createShaderModule({
            code: `
              @group(0) @binding(2) var<storage> inputs: array<f32>;
              @group(0) @binding(3) var<storage, read_write> outputs: array<f32>;
              @compute @workgroup_size(${c.workgroupSize})
              fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                outputs[id.x] = inputs[id.x];
              }
            `,
          }),
          entryPoint: "main",
        },
      }),
    );

    this.pipelines.set(
      "rmsnorm",
      await this.device.createComputePipelineAsync({
        layout: "auto",
        compute: {
          module: this.device.createShaderModule({
            code: `
              @group(0) @binding(2) var<storage> inputs: array<f32>;
              @group(0) @binding(3) var<storage, read_write> outputs: array<f32>;
              @compute @workgroup_size(${c.workgroupSize})
              fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                outputs[id.x] = inputs[id.x];
              }
            `,
          }),
          entryPoint: "main",
        },
      }),
    );
  }

  async forward(tokens) {
    const c = this.config;
    const batchSize = tokens.length;
    const seqLen = tokens[0].length;

    const configData = new Uint32Array([0, 0, seqLen, batchSize]);
    this.device.queue.writeBuffer(this.buffers.get("config"), 0, configData);
    this.device.queue.writeBuffer(this.buffers.get("inputs"), 0, new Uint32Array(tokens.flat()));

    const encoder = this.device.createCommandEncoder();

    for (let layer = 0; layer < c.numLayers; layer += 1) {
      configData[0] = layer;
      this.device.queue.writeBuffer(this.buffers.get("config"), 0, configData);

      const attnPass = encoder.beginComputePass();
      attnPass.setPipeline(this.pipelines.get("attention"));
      attnPass.setBindGroup(0, this.createBindGroup("attention"));
      attnPass.dispatchWorkgroups(batchSize * seqLen, c.numHeads);
      attnPass.end();

      const normPass = encoder.beginComputePass();
      normPass.setPipeline(this.pipelines.get("rmsnorm"));
      normPass.setBindGroup(0, this.createBindGroup("rmsnorm"));
      normPass.dispatchWorkgroups(batchSize * seqLen);
      normPass.end();

      const ffnPass = encoder.beginComputePass();
      ffnPass.setPipeline(this.pipelines.get("ffn"));
      ffnPass.setBindGroup(0, this.createBindGroup("ffn"));
      ffnPass.dispatchWorkgroups(batchSize * seqLen);
      ffnPass.end();
    }

    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    return this.readOutputs(batchSize, seqLen);
  }

  createBindGroup(pipelineKey) {
    return this.device.createBindGroup({
      layout: this.pipelines.get(pipelineKey).getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.buffers.get("weights") } },
        { binding: 1, resource: { buffer: this.buffers.get("kv-cache") } },
        { binding: 2, resource: { buffer: this.buffers.get("inputs") } },
        { binding: 3, resource: { buffer: this.buffers.get("outputs") } },
        { binding: 4, resource: { buffer: this.buffers.get("config") } },
      ],
    });
  }

  async readOutputs(batchSize, seqLen) {
    const readBuffer = this.device.createBuffer({
      size: batchSize * seqLen * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.buffers.get("outputs"), 0, readBuffer, 0, batchSize * seqLen * 4);
    this.device.queue.submit([encoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const data = new Uint32Array(readBuffer.getMappedRange());

    const result = [];
    for (let i = 0; i < batchSize; i += 1) {
      result.push(Array.from(data.slice(i * seqLen, (i + 1) * seqLen)));
    }
    readBuffer.unmap();
    return result;
  }
}

class PagedCache {
  constructor(config) {
    this.pageSize = config.cachePageSize;
    this.maxPages = config.cacheMaxPages;
    this.pages = new Map();
  }

  get(key) {
    const page = this.pages.get(key);
    if (!page) return null;
    page.lastAccess = Date.now();
    return this.decompress(page.data);
  }

  set(key, value) {
    if (this.pages.size >= this.maxPages) this.evict();
    this.pages.set(key, {
      data: this.compress(value),
      lastAccess: Date.now(),
    });
  }

  compress(values) {
    const max = Math.max(...values);
    const min = Math.min(...values);
    const scale = (max - min) / 255 || 1;
    return {
      data: new Uint8Array(values.map((v) => Math.round((v - min) / scale))),
      scale,
      min,
    };
  }

  decompress(page) {
    return Array.from(page.data, (v) => v * page.scale + page.min);
  }

  evict() {
    let oldest = Infinity;
    let oldestKey = null;
    for (const [key, page] of this.pages) {
      if (page.lastAccess < oldest) {
        oldest = page.lastAccess;
        oldestKey = key;
      }
    }
    if (oldestKey) this.pages.delete(oldestKey);
  }
}
