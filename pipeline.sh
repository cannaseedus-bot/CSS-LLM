#!/usr/bin/env bash
set -euo pipefail

MODEL_ID=${1:-meta-llama/Meta-Llama-3-8B}
OUT_DIR=${2:-./models}

python weight-converter.py "$MODEL_ID" --output "$OUT_DIR"

echo "Shader optimizer and replay verifier are intended for browser runtime integration."

tar -czf "$(basename "$MODEL_ID").csslm.tgz" -C "$OUT_DIR" "$(basename "$MODEL_ID")"
