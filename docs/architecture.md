# CSS-LLM Shell Architecture

## Pipeline

1. Governance CSS is constrained by CLGS-1 grammar.
2. `validate-config.js` enforces allowed property registry.
3. `invariants.js` enforces hard model constraints.
4. Shell initializes GPU resources and kernels.
5. Replay engine emits deterministic replay records.
6. Replay verifier checks output hash against the record.

## Deterministic replay input set

- Governance CSS hash
- Model weights hash
- Kernel bundle hash
- Input token hash
- Sampling config + RNG seed
- Output token hash

## Deterministic reduction law

To keep replay behavior stable, kernels follow these rules:

1. No atomics for floating-point reductions.
2. No unordered subgroup reduction primitives.
3. Fixed tree reduction ordering (`stride = N/2 ... 1`).
4. Fixed loop traversal order for sequence and feature dimensions.
5. No race-dependent accumulation paths.

The INT4 RMSNorm kernel applies this law directly with a workgroup-local
binary tree reduction over `partial[]`.

## Artifact policy

- No large binary model files are committed to git/GitHub.
- GitHub-safe Base64 JSON artifacts are acceptable for small fixtures/examples.
- Real model weights are hosted externally (e.g., Google Drive, Hugging Face, object storage).
- `tools/fetch-weights.js` is the local downloader for external HTTPS weight files.

## Repository layout

- `governance/`: grammar, schema, law enforcement
- `shell/`: runtime modules
- `kernels/`: WGSL kernels
- `weights/`: CLIF-1 docs + loaders
- `tools/`: conversion/packing/hash/fetch utilities
- `replay/`: replay schema + verifier
- `demo/`: browser entrypoint
