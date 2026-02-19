# CSS-LLM Shell

CSS-LLM is a formal governance-driven runtime shell for browser and server-backed inference.

## What is implemented

- CLGS-1 grammar (`governance/css-grammar.ebnf`)
- JSON schema (`governance/css-llm.schema.json`)
- Governance validator + invariants (`governance/validate-config.js`, `governance/invariants.js`)
- Replay contract + verifier (`replay/replay-schema.json`, `shell/replay-engine.js`, `replay/verify.js`)
- Replay proof envelope utilities (`shell/deterministic-replay.js`, `replay/proof-envelope.example.json`)
- Minimal shell modules (`shell/tensor-shell.js`, `shell/buffer-manager.js`, `shell/kv-cache.js`)
- Kernel set including deterministic INT4-focused scaffolds:
  - `kernels/fused_rmsnorm_matmul.wgsl`
  - `kernels/flashattention_int4.wgsl`
  - `kernels/swiglu_int4.wgsl`
  - `kernels/sampler_pcg32.wgsl`
  - `kernels/moe_cluster_int4.wgsl`
  - existing kernels: `kernels/rmsnorm.wgsl`, `kernels/flash_attention.wgsl`, `kernels/moe_expert_int4.wgsl`, `kernels/matmul_int4.wgsl`, `kernels/rope.wgsl`, `kernels/softmax.wgsl`, `kernels/swiglu.wgsl`, `kernels/sampler.wgsl`
- CLIF-1 weight format docs/loader (`weights/quant-format.md`, `weights/weight-loader.js`)
- Build/packing/hash/fetch/assemble stubs (`tools/convert-hf-to-int4.py`, `tools/pack-weights.js`, `tools/hash-model.js`, `tools/fetch-weights.js`, `tools/assemble-shards.js`)
- Browser demo (`demo/index.html`, `demo/main.js`, `demo/governance.css`)

## Deterministic 1B browser core (consolidated)

Implemented design blocks (checked phases):

- [x] Phase 1: Fused RMSNorm + residual + INT4 matmul kernel scaffold.
- [x] Phase 2: Fused INT4 SwiGLU kernel scaffold.
- [x] Phase 3: Deterministic PCG32 sampler primitive.
- [x] Phase 4: Replay proof envelope generation/verification.
- [x] Phase 5: Deterministic top-k MoE cluster kernel scaffold with 2-phase accumulation (no atomics).
- [x] Phase 6: A phase-structured 1B pipeline entrypoint in `TensorShell`.

## Artifact policy: sharded Base64 for GitHub, binary assembled locally

Base64 decodes to binary, so large models are represented as shards.

Recommended pattern:

- Host shard files externally (Google Drive / Hugging Face / object storage):
  - `css_llm_001.bin.base64.json`
  - `css_llm_002.bin.base64.json`
  - `css_llm_003.bin.base64.json`
  - ...
  - `css_llm_index.json`
- Download shard set locally.
- Assemble a local binary only on developer machine or runtime node.
- Never commit assembled model binaries to git.

## Shard workflow

```bash
# create local fixture shards
node tools/pack-weights.js weights/shards css_llm 3

# hash an index or shard artifact
node tools/hash-model.js weights/shards/css_llm_index.json

# assemble into local binary
node tools/assemble-shards.js weights/shards/css_llm_index.json weights/model.bin
```

## Quick start

```bash
npm run check
npm run validate:demo
python3 -m http.server 8080
```

Open `http://localhost:8080/demo/index.html`.


## To-do list

- [ ] Wire full end-to-end 1B inference through all kernel stages.
- [ ] Add deterministic regression fixtures for replay envelope verification.
- [ ] Expand shard tooling with integrity checks and resumable downloads.
- [ ] Add browser benchmark script for repeatable latency/memory reporting.
