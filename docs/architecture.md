# CSS-LLM Shell Architecture

## Pipeline

1. Governance CSS is constrained by CLGS-1 grammar.
2. `validate-config.js` enforces allowed property registry.
3. `invariants.js` enforces hard model constraints.
4. Shell initializes GPU resources and kernels.
5. Replay engine emits deterministic replay records.
6. Replay verifier checks output hash against the record.

## Consolidated deterministic 1B core

The repository includes explicit kernel files for the consolidated core:

- `kernels/fused_rmsnorm_matmul.wgsl`
- `kernels/flashattention_int4.wgsl`
- `kernels/swiglu_int4.wgsl`
- `kernels/sampler_pcg32.wgsl`

These are structured to support deterministic reduction ordering, INT4 decode-in-kernel,
f32 accumulation, and replayable generation with deterministic RNG.

## Deterministic metadata plane

`models/CSS-LLM/` now provides deterministic metadata artifacts for replay and policy control:

- tokenizer, vocab, codex dispatch, meta, policy, tools, semantics, geometry
- raw merges (`merges.txt`) and canonical merges envelope (`merges.cssllm.json`)

All metadata JSON files use:

- `format: css-llm/metadata-v1`
- canonicalization contract (`utf-8`, sorted keys, no whitespace form, IEEE-754 f64 intent)
- SHA-256 `payloadHash` for deterministic verification

Validation:

- schema: `specs/cssllm-metadata.schema.json`
- verifier tool: `tools/validate-cssllm-metadata.js`

## Replay proof envelope

Proof envelope is implemented in `shell/deterministic-replay.js` and modeled in
`replay/proof-envelope.example.json`.

`proofHash` is computed as:

`SHA256(governanceHash || weightHash || kernelHash || inputHash || rngSeed || outputHash)`

## Artifact policy

- No large model binaries are committed to git/GitHub.
- Large model payloads are represented as sharded Base64 text artifacts.
- Real shards are hosted externally (Google Drive, Hugging Face, object storage).
- `tools/fetch-weights.js` downloads index + shards.
- `tools/assemble-shards.js` reconstructs local runtime binary.
