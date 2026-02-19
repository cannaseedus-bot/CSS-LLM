// enterprise-model-client.js
// Enterprise model loader that enforces server-required CSS contracts.

export class EnterpriseModelClient {
  constructor(governanceCSS, transport) {
    this.config = this.parse(governanceCSS);
    this.transport = transport;
    this.assertServerRequirements();
  }

  parse(css) {
    const styles = css.match(/:root\s*{([^}]*)}/)?.[1] || "";
    const vars = {};

    styles.split(";").forEach((line) => {
      const match = line.match(/--([^:]+):\s*(.*)/);
      if (!match) return;
      vars[match[1].trim()] = match[2].trim().replace(/^"|"$/g, "");
    });

    return {
      enterpriseMode: vars["enterprise-mode"] === "true",
      requiresServer: vars["requires-server"] === "true",
      standaloneBrowserAllowed: vars["standalone-browser-allowed"] === "true",
      endpoint: vars["control-plane-endpoint"],
      protocol: vars["control-plane-protocol"],
      timeoutMs: parseInt(vars["control-plane-timeout-ms"], 10) || 5000,
      tenantId: vars["control-plane-tenant-id"],
      modelId: vars["registry-model-id"],
      modelRevision: vars["registry-model-revision"],
      topology: vars["inference-topology"],
      contractHash: vars["enterprise-contract-hash"],
    };
  }

  assertServerRequirements() {
    if (!this.config.enterpriseMode) {
      throw new Error("Enterprise profile requires --enterprise-mode: true");
    }
    if (!this.config.requiresServer) {
      throw new Error("Enterprise profile must set --requires-server: true");
    }
    if (this.config.standaloneBrowserAllowed) {
      throw new Error("Enterprise profile cannot enable standalone browser execution");
    }
    if (!this.config.endpoint) {
      throw new Error("Missing --control-plane-endpoint");
    }
  }

  async initialize() {
    if (!this.transport || typeof this.transport.connect !== "function") {
      throw new Error("EnterpriseModelClient requires a server transport with connect()");
    }

    await this.transport.connect({
      endpoint: this.config.endpoint,
      protocol: this.config.protocol,
      timeoutMs: this.config.timeoutMs,
      tenantId: this.config.tenantId,
    });

    return this;
  }

  async generate(prompt, sampling = {}) {
    if (!this.transport || typeof this.transport.request !== "function") {
      throw new Error("EnterpriseModelClient requires transport.request()");
    }

    return this.transport.request("generate", {
      prompt,
      model: {
        id: this.config.modelId,
        revision: this.config.modelRevision,
        topology: this.config.topology,
        contractHash: this.config.contractHash,
      },
      sampling,
    });
  }
}
