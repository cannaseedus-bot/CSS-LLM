class MultiBackendRuntime {
  constructor(governance = {}) {
    this.governance = governance;
    this.backends = new Map();
    this.backend = null;

    this.registerBackend("webgpu", new WebGPUBackend(governance));
    this.registerBackend("wasm", new WASMBackend(governance));
    this.registerBackend("cpu", new CPUBackend(governance));
  }

  registerBackend(name, implementation) {
    this.backends.set(name, implementation);
  }

  async detectCapabilities() {
    const caps = new Map();

    if (globalThis.navigator?.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        caps.set("webgpu", { available: !!adapter });
      } catch {
        caps.set("webgpu", { available: false });
      }
    } else {
      caps.set("webgpu", { available: false });
    }

    caps.set("wasm", { available: typeof WebAssembly !== "undefined" });
    caps.set("cpu", { available: true });
    return caps;
  }

  async selectOptimalBackend() {
    const caps = await this.detectCapabilities();
    const preferred = this.governance["--bridge-protocol"]?.replace(/"/g, "");
    const order = [preferred, "webgpu", "wasm", "cpu"].filter(Boolean);

    for (const name of order) {
      if (caps.get(name)?.available) {
        this.backend = this.backends.get(name);
        await this.backend.initialize(caps.get(name));
        return name;
      }
    }

    throw new Error("No backend available");
  }

  async forward(tokens, params = {}) {
    if (!this.backend) await this.selectOptimalBackend();
    return this.backend.forward(tokens, params);
  }
}

class WebGPUBackend {
  async initialize() {}
  async forward(tokens) {
    return tokens;
  }
}

class WASMBackend {
  async initialize() {}
  async forward(tokens) {
    return tokens;
  }
}

class CPUBackend {
  async initialize() {}
  async forward(tokens) {
    return tokens;
  }
}

export { MultiBackendRuntime, WebGPUBackend, WASMBackend, CPUBackend };
