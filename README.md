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
  - existing kernels: `kernels/rmsnorm.wgsl`, `kernels/flash_attention.wgsl`, `kernels/moe_expert_int4.wgsl`, `kernels/matmul_int4.wgsl` (tiled int4 decode + per-block scales), `kernels/rope.wgsl`, `kernels/softmax.wgsl`, `kernels/swiglu.wgsl`, `kernels/sampler.wgsl`
- CLIF-1 weight format docs/loader (`weights/quant-format.md`, `weights/weight-loader.js`)
- Mini profile artifacts for the 64M–120M browser tier:
  - `models/model-mini-governance.css`
  - `kernels/matmul_int4_mini.wgsl`
  - `kernels/flash_attention_lite.wgsl`
  - `src/mini-tensor-shell.js`
  - `weights/scx-mini-format.md`
- Build/packing/hash/fetch/assemble stubs (`tools/convert-hf-to-int4.py`, `tools/pack-weights.js`, `tools/hash-model.js`, `tools/fetch-weights.js`, `tools/assemble-shards.js`)
- Browser demo (`demo/index.html`, `demo/main.js`, `demo/governance.css`)
- Mini attention/runtime extras for practical deployment:
  - `src/kernels/attention_flash_lite.wgsl`
  - `src/utils/deterministic_sampler.js`
  - `demo/manifest.json`, `demo/service-worker.js`
  - `training/micro-config.yaml`, `training/micro_train.py`

## Deterministic 1B browser core (consolidated)

Implemented design blocks (checked phases):

1. Fused RMSNorm + residual + INT4 matmul kernel scaffold.
2. Fused INT4 SwiGLU kernel scaffold.
3. Deterministic PCG32 sampler primitive.
4. Replay proof envelope generation/verification.
5. Deterministic top-k MoE cluster kernel scaffold with split compute/reduce entrypoints, per-expert block scales, and 2-phase accumulation (no atomics).
6. A phase-structured 1B pipeline entrypoint in `TensorShell`.

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



## Mini 64M shippable workstream

- [x] Define mini governance + strict schema surfaces (`governance/model.css`, `governance/schema.json`).
- [x] Add runtime layout scaffolding for kernels, MoE routing, tokenizer, and deterministic utilities (`src/`).
- [x] Add SCX mini format docs + deterministic pack/unpack utilities (`scx/`).
- [x] Add a concrete 64M training/quantization recipe (`training/config.yaml`, `training/train.py`, `training/quantize.py`, `training/export_int4.py`).
- [x] Add a micro-training loop/profile for small-dataset iteration (`training/micro-config.yaml`, `training/micro_train.py`).
- [x] Add deterministic tokenizer + top-p sampler engine utilities (`src/tokenizer/tokenizer.js`, `src/utils/deterministic_sampler.js`, `src/utils/deterministic_rng.js`).
- [x] Add deployable PWA artifacts for offline demo boot (`demo/manifest.json`, `demo/service-worker.js`, `demo/icon-192.svg`).
- [x] Commit a deterministic mock `SCXM` artifact for integration wiring (`models/mini-64m-int4.scx.base64.json`).

## Mini build snippets

```bash
# validate JS surfaces and base checks
npm run check

# pack a deterministic SCX payload from JSON tensors
python3 scx/pack_scx.py /tmp/tensors.json /tmp/mini-64m-int4.scx

# inspect SCX header quickly
node -e "import('./scx/unpack_scx.js').then(m => console.log(m.unpackScx('models/mini-64m-int4.scx.base64.json')))"
```

## To-do list

- [ ] Wire full end-to-end 1B inference through all kernel stages.
- [x] Add mini 85M browser profile governance + deterministic shell scaffold.
- [ ] Add deterministic regression fixtures for replay envelope verification.
- [ ] Expand shard tooling with integrity checks and resumable downloads.
- [ ] Add browser benchmark script for repeatable latency/memory reporting.
