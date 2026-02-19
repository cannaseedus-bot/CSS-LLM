# CSS-LLM: Formal Governance Grammar + Minimal 1B Shell

## Layers

1. **Governance layer (CSS):** a strict grammar for model/runtime constraints.
2. **Bridge layer (JS):** CSS variable parsing into typed runtime config.
3. **Compute layer (WebGPU):** buffers, pipelines, dispatch loops, cache lifecycle.

## Included artifacts

- `specs/governance-grammar-v1.css`: normative schema grammar.
- `models/model-1b-governance.css`: minimal viable 1B profile for browser execution.
- `src/tensor-shell-minimal.js`: browser shell with parser + WebGPU pipeline setup + paged cache.
- `examples/index.html`: end-to-end loading and invocation example.

## Baseline 1B profile targets

- Parameters: ~1B
- Quantized weights: int4
- Runtime target: WebGPU first, browser-safe batch/sequence bounds
- Determinism primitive: config hash in governance contract

## Planned next steps

1. Weight converter (HF -> int4 packed format)
2. Shader auto-tuner for per-device workgroup tuning
3. Replay verifier for trace/input/output hash validation
4. WASM fallback backend
5. CSS model-zoo profiles
