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
- CLIF-1 weight format docs/loader (`weights/quant-format.md`, `weights/weight-loader.js`)
- Build/packing/hash/fetch/assemble stubs (`tools/convert-hf-to-int4.py`, `tools/pack-weights.js`, `tools/hash-model.js`, `tools/fetch-weights.js`, `tools/assemble-shards.js`)
- CSS-LLM deterministic metadata pack (`models/CSS-LLM/*.json`, `models/CSS-LLM/merges.txt`, `specs/cssllm-metadata.schema.json`)
- Browser demo (`demo/index.html`, `demo/main.js`, `demo/governance.css`)

## Deterministic metadata pack

Added deterministic/replay metadata files under `models/CSS-LLM/`:

- `tokenizer.json`
- `codex.json`
- `vocab.json`
- `meta.json`
- `policy.json`
- `tools.json`
- `semantics.json`
- `geometry.json`
- `merges.txt` + `merges.cssllm.json`

Each JSON file is encoded in CSS-LLM metadata format (`css-llm/metadata-v1`) with canonicalization rules and a `payloadHash` for replay integrity.

Conversion and validation tools:

- `node tools/convert-to-cssllm-format.js <input.json> <output.json> [name]`
- `npm run validate:metadata`

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
node tools/pack-weights.js weights/shards css_llm 3
node tools/hash-model.js weights/shards/css_llm_index.json
node tools/assemble-shards.js weights/shards/css_llm_index.json weights/model.bin
```

## Quick start

```bash
npm run check
npm run validate:demo
npm run validate:metadata
python3 -m http.server 8080
```

Open `http://localhost:8080/demo/index.html`.
