class ReplayVerifier {
  constructor() {
    this.traces = new Map();
    this.hashes = new Map();
  }

  hashTensor(data, dtype = "float32") {
    let hash = 0x811c9dc5;
    const apply = (byte) => {
      hash ^= byte;
      hash += (hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24);
      hash >>>= 0;
    };

    if (dtype === "uint8") {
      for (let i = 0; i < data.length; i += 1) apply(data[i]);
    } else {
      const ctor = dtype === "int32" ? Int32Array : Float32Array;
      for (let i = 0; i < data.length; i += 1) {
        new Uint8Array(new ctor([data[i]]).buffer).forEach(apply);
      }
    }
    return hash.toString(16).padStart(8, "0");
  }

  async captureTrace(sessionId, step, inputs, outputs, metadata = {}) {
    const trace = {
      timestamp: Date.now(),
      step,
      inputHash: this.hashTensor(inputs, "int32"),
      outputHash: this.hashTensor(outputs, "float32"),
      metadata,
    };
    if (!this.traces.has(sessionId)) this.traces.set(sessionId, []);
    this.traces.get(sessionId).push(trace);
    this.hashes.set(`${sessionId}:${step}`, trace.outputHash);
    return trace;
  }

  verifyStep(sessionId, step, outputs) {
    const expectedHash = this.hashes.get(`${sessionId}:${step}`);
    if (!expectedHash) return { verified: false, error: "No trace found" };
    const actualHash = this.hashTensor(outputs, "float32");
    return { verified: actualHash === expectedHash, expectedHash, actualHash, step, sessionId };
  }

  exportTrace(sessionId) {
    const trace = this.traces.get(sessionId);
    if (!trace) return null;
    return { sessionId, steps: trace.length, hashes: trace.map((t) => ({ step: t.step, inputHash: t.inputHash, outputHash: t.outputHash, metadata: t.metadata })) };
  }

  importTrace(traceData) {
    for (const step of traceData.hashes) {
      this.hashes.set(`${traceData.sessionId}:${step.step}`, step.outputHash);
    }
    return traceData.sessionId;
  }
}

export { ReplayVerifier };
