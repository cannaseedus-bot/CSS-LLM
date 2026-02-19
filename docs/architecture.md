# CSS-LLM Shell Architecture

## Pipeline

1. Governance CSS is constrained by CLGS-1 grammar.
2. `validate-config.js` enforces allowed property registry.
3. `invariants.js` enforces hard model constraints.
4. Shell initializes GPU resources and kernels.
5. Replay engine emits deterministic replay records.
6. Replay verifier checks output hash against the record.

## Consolidated deterministic 1B core

The repository now includes explicit kernel files for the consolidated core:

- `kernels/fused_rmsnorm_matmul.wgsl`
- `kernels/flashattention_int4.wgsl`
- `kernels/swiglu_int4.wgsl`
- `kernels/sampler_pcg32.wgsl`

These are structured to support:

- deterministic reduction ordering
- INT4 decode-in-kernel
- f32 accumulation
- replayable generation with deterministic RNG

## Deterministic replay input set

- Governance CSS hash
- Model weights hash
- Kernel bundle hash
- Input token hash
- Sampling config + RNG seed
- Output token hash

## Replay proof envelope

Proof envelope is implemented in `shell/deterministic-replay.js` and modeled in
`replay/proof-envelope.example.json`.

`proofHash` is computed as:

`SHA256(governanceHash || weightHash || kernelHash || inputHash || rngSeed || outputHash)`

## Deterministic reduction law

To keep replay behavior stable, kernels follow these rules:

1. No atomics for floating-point reductions.
2. No unordered subgroup reduction primitives.
3. Fixed tree reduction ordering (`stride = N/2 ... 1`).
4. Fixed loop traversal order for sequence and feature dimensions.
5. No race-dependent accumulation paths.

## Artifact policy

- No large model binaries are committed to git/GitHub.
- Large model payloads are represented as sharded Base64 text artifacts.
- Canonical naming pattern:
  - `css_llm_001.bin.base64.json`
  - `css_llm_002.bin.base64.json`
  - `css_llm_003.bin.base64.json`
  - ... + `css_llm_index.json`
- Real shards are hosted externally (Google Drive, Hugging Face, object storage).
- `tools/fetch-weights.js` downloads index + shards.
- `tools/assemble-shards.js` reconstructs local runtime binary.

## Repository layout

- `governance/`: grammar, schema, law enforcement
- `shell/`: runtime modules
- `kernels/`: WGSL kernels
- `weights/`: CLIF-1 docs + loaders
- `tools/`: conversion/packing/hash/fetch/assemble utilities
- `replay/`: replay schema + verifier
- `demo/`: browser entrypoint
