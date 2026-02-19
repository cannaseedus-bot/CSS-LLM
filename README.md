# CSS-LLM

Production-oriented scaffold for a CSS-governed LLM runtime pipeline:

- `weight-converter.py`: HuggingFace model conversion to packed int4 blocks + governance CSS.
- `shader-optimizer.js`: WebGPU workgroup auto-tuning helper.
- `replay-verifier.js`: deterministic replay hash capture and verification.
- `multi-backend.js`: backend selector (`webgpu` → `wasm` → `cpu`).
- `model-zoo.css`: preconfigured model governance presets.
- `pipeline.sh`: end-to-end orchestration shell entrypoint.

## Quick start

```bash
python weight-converter.py meta-llama/Meta-Llama-3-8B --output ./models
```

```bash
./pipeline.sh meta-llama/Meta-Llama-3-8B ./models
```
