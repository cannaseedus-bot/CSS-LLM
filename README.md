# CSS-LLM Shell

CSS-LLM is a formal governance-driven runtime shell for browser and server-backed inference.

## What is implemented

- CLGS-1 grammar (`governance/css-grammar.ebnf`)
- JSON schema (`governance/css-llm.schema.json`)
- Governance validator + invariants (`governance/validate-config.js`, `governance/invariants.js`)
- Replay contract + verifier (`replay/replay-schema.json`, `shell/replay-engine.js`, `replay/verify.js`)
- Minimal shell modules (`shell/tensor-shell.js`, `shell/buffer-manager.js`, `shell/kv-cache.js`)
- Kernel set including deterministic INT4-focused scaffolds:
  - `kernels/rmsnorm.wgsl` (fused INT4 RMSNorm)
  - `kernels/flash_attention.wgsl` (streaming INT4 FlashAttention)
  - `kernels/moe_expert_int4.wgsl` (deterministic top-2 routing helper)
  - `kernels/matmul_int4.wgsl`, `kernels/rope.wgsl`, `kernels/swiglu.wgsl`, `kernels/softmax.wgsl`, `kernels/sampler.wgsl`
- CLIF-1 weight format docs/loader (`weights/quant-format.md`, `weights/weight-loader.js`)
- Build/packing/hash/fetch stubs (`tools/convert-hf-to-int4.py`, `tools/pack-weights.js`, `tools/hash-model.js`, `tools/fetch-weights.js`)
- Browser demo (`demo/index.html`, `demo/main.js`, `demo/governance.css`)

## Governance law

Unknown governance properties are rejected by the validator.
Runtime invariants are enforced before execution.

## GitHub artifact policy (binary vs Base64)

Short answer: **yes, GitHub allows Base64 text files**, but Base64 is only an encoding of binary and still inflates size (~33%).

Recommended policy:

- Keep real model binaries out of git history.
- Use object storage / releases for large weight blobs.
- Google Drive / Hugging Face links are valid external sources for real model files.
- Keep only small test fixtures or metadata in-repo.
- If needed for developer flows, commit small Base64 JSON artifacts.

`tools/pack-weights.js` emits a GitHub-safe Base64 artifact:

- default output: `weights/model-1b-int4.base64.json`

`tools/fetch-weights.js` can download external weights to local disk:

- `node tools/fetch-weights.js "https://..." weights/model.bin`

## Quick start

```bash
npm run check
npm run validate:demo
node tools/pack-weights.js
node tools/hash-model.js weights/model-1b-int4.base64.json
python3 -m http.server 8080
```

Open `http://localhost:8080/demo/index.html`.
