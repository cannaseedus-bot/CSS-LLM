# CSS-LLM: Formal Governance Grammar + Runtime Formats

## Layers

1. **Governance layer (CSS):** strict contracts for model/runtime constraints.
2. **Bridge layer (JS):** CSS variable parsing into typed runtime config.
3. **Compute layer:** browser WebGPU shell or enterprise server execution.

## Included artifacts

- `specs/governance-grammar-v1.css`: baseline schema grammar.
- `models/model-1b-governance.css`: browser-first 1B profile.
- `src/tensor-shell-minimal.js`: browser shell (WebGPU).
- `examples/index.html`: browser demo loading the 1B profile.

## Enterprise server-required format

- `specs/enterprise-model-format-v1.css`: enterprise contract schema that explicitly requires a control plane/server.
- `models/model-enterprise-server-governance.css`: concrete enterprise profile with mTLS control plane, registry metadata, compliance routing, and SLO knobs.
- `src/enterprise-model-client.js`: runtime client enforcing `--requires-server: true` and routing generation requests through a server transport.

This enterprise format is designed for managed deployment and intentionally disallows standalone browser-only inference.

## Planned next steps

1. Weight converter (HF -> int4 packed format)
2. Shader auto-tuner for per-device workgroup tuning
3. Replay verifier for trace/input/output hash validation
4. WASM fallback backend
5. CSS model-zoo profiles
