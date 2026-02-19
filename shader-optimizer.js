#!/usr/bin/env node

class ShaderOptimizer {
  constructor(device, config) {
    this.device = device;
    this.config = config;
    this.maxWorkgroupSize = device.limits.maxComputeWorkgroupSizeX;
  }

  async tuneAllKernels() {
    const attention = await this.tuneAttentionKernel();
    const ffn = await this.tuneFFNKernel();
    const norm = await this.tuneNormKernel();
    return { attention, ffn, norm };
  }

  async tuneAttentionKernel() {
    const candidates = [64, 128, 256].filter((x) => x <= this.maxWorkgroupSize).map((x) => ({ x, y: 1, z: 1 }));
    return this.benchmarkKernels("attention", candidates);
  }

  async tuneFFNKernel() {
    const candidates = [64, 128, 256, 512].filter((x) => x <= this.maxWorkgroupSize).map((x) => ({ x, y: 1, z: 1 }));
    return this.benchmarkKernels("ffn", candidates);
  }

  async tuneNormKernel() {
    const candidates = [64, 128, 256, 512].filter((x) => x <= this.maxWorkgroupSize).map((x) => ({ x, y: 1, z: 1 }));
    return this.benchmarkKernels("norm", candidates);
  }

  async benchmarkKernels(type, candidates) {
    const result = [];
    for (const cfg of candidates) {
      const pipeline = await this.createPipeline(this.generateKernel(type, cfg));
      const times = [];
      for (let i = 0; i < 5; i += 1) {
        times.push(await this.benchmarkPipeline(pipeline));
      }
      const avg = times.reduce((a, b) => a + b, 0) / times.length;
      result.push({ cfg, avg });
    }
    return result.sort((a, b) => a.avg - b.avg)[0].cfg;
  }

  generateKernel(type, cfg) {
    return `@compute @workgroup_size(${cfg.x}, ${cfg.y}, ${cfg.z}) fn main(@builtin(global_invocation_id) _id: vec3<u32>) {}`;
  }

  async createPipeline(shaderCode) {
    return this.device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: this.device.createShaderModule({ code: shaderCode }), entryPoint: "main" },
    });
  }

  async benchmarkPipeline(pipeline) {
    const buffer = this.device.createBuffer({
      size: 4096,
      usage: GPUBufferUsage.STORAGE,
    });

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer } }],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1, 1, 1);
    pass.end();

    const start = performance.now();
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    return (performance.now() - start) / 1000;
  }

  generateOptimizedCSS({ attention, ffn, norm }) {
    return `:root {
  --attention-workgroup-x: ${attention.x};
  --attention-workgroup-y: ${attention.y};
  --attention-workgroup-z: ${attention.z};
  --ffn-workgroup-x: ${ffn.x};
  --norm-workgroup-x: ${norm.x};
}`;
  }
}

export { ShaderOptimizer };
