# SCX Mini Format (v1)

A compact INT4-focused payload for the CSS-LLM mini profile.

## Header

- `magic`: `SCXM`
- `version`: `1`
- `layers`: `12`
- `hidden`: `768`
- `dtype`: `int4`

## Tensor stream order

1. embedding weights
2. per-layer attention tensors (`q`, `k`, `v`, `o`)
3. per-layer FFN tensors (`up`, `down`)

All weights are packed INT4.

## Determinism contract

- fixed tensor order in file
- fixed decode order at runtime
- no atomics in reduction kernels
- deterministic sampling seed derived from input token hash
