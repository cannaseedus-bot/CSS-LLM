#!/usr/bin/env python3
"""CSS-LLM Weight Converter: HuggingFace -> int4 + CSS governance package."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM


class CSSWeightConverter:
    def __init__(self, model_id: str, output_dir: Path):
        self.model_id = model_id
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"📥 Loading {model_id}...")
        self.config = AutoConfig.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="cpu",
        )

        self.hidden_dim = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = getattr(self.config, "num_key_value_heads", self.num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.intermediate_dim = self.config.intermediate_size
        self.vocab_size = self.config.vocab_size

    @staticmethod
    def quantize_int4(tensor: torch.Tensor, block_size: int = 128) -> Tuple[np.ndarray, np.ndarray, int]:
        """Block-wise symmetric int4 quantization with nibble packing."""
        flat = tensor.detach().float().cpu().numpy().reshape(-1)
        orig_len = flat.shape[0]

        pad_len = (block_size - (orig_len % block_size)) % block_size
        if pad_len:
            flat = np.pad(flat, (0, pad_len))

        blocks = flat.reshape(-1, block_size)
        scales = np.maximum(np.abs(blocks).max(axis=1), 1e-8)
        normalized = blocks / scales[:, None]
        q = np.round(normalized * 7.0).clip(-8, 7).astype(np.int8)

        lo = (q[::2] & 0x0F).astype(np.uint8)
        hi = ((q[1::2] & 0x0F) << 4).astype(np.uint8)
        packed = lo | hi

        return packed, scales.astype(np.float32), orig_len

    @staticmethod
    def _sample_css(prefix: str, data: np.ndarray, limit: int = 16) -> Dict[str, float]:
        return {f"--{prefix}-{i}": float(int(v) / 255.0) for i, v in enumerate(data[:limit])}

    def _save_quant_blob(self, path: Path, packed: np.ndarray, scales: np.ndarray, orig_len: int):
        with open(path, "wb") as f:
            f.write(orig_len.to_bytes(8, "little"))
            f.write(len(scales).to_bytes(8, "little"))
            f.write(packed.tobytes())
            f.write(scales.astype(np.float16).tobytes())

    def convert_embeddings(self) -> Dict[str, float]:
        weight = self.model.get_input_embeddings().weight.data
        packed, scales, orig_len = self.quantize_int4(weight)
        self._save_quant_blob(self.output_dir / "embeddings.bin", packed, scales, orig_len)
        return self._sample_css("embed", packed)

    def convert_layers(self) -> Dict[str, float]:
        sampled: Dict[str, float] = {}
        offsets = []
        running = 0

        for layer_idx in range(self.num_layers):
            layer = self.model.model.layers[layer_idx]
            tensors = {
                "q": layer.self_attn.q_proj.weight.data,
                "k": layer.self_attn.k_proj.weight.data,
                "v": layer.self_attn.v_proj.weight.data,
                "o": layer.self_attn.o_proj.weight.data,
            }

            path = self.output_dir / f"layer_{layer_idx}.bin"
            with open(path, "wb") as f:
                for name, tensor in tensors.items():
                    packed, scales, orig_len = self.quantize_int4(tensor)
                    f.write(orig_len.to_bytes(8, "little"))
                    f.write(len(scales).to_bytes(8, "little"))
                    f.write(packed.tobytes())
                    f.write(scales.astype(np.float16).tobytes())
                    running += len(packed)
                    if layer_idx == 0 and name == "q":
                        sampled.update(self._sample_css("l0-q", packed))

            offsets.append(running)

        with open(self.output_dir / "layer_offsets.json", "w", encoding="utf-8") as f:
            json.dump(offsets, f)

        return sampled

    def convert_lm_head(self) -> Dict[str, float]:
        packed, scales, orig_len = self.quantize_int4(self.model.lm_head.weight.data)
        self._save_quant_blob(self.output_dir / "lm_head.bin", packed, scales, orig_len)
        return self._sample_css("lm-head", packed)

    def generate_css_governance(self, samples: Dict[str, float]) -> str:
        cfg = {
            "model_id": self.model_id,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
        }
        model_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()[:16]

        lines = [
            "/*! CSS-LLM Governance */",
            ":root {",
            f'  --model-family: "{self.model_id}";',
            f'  --model-name: "{self.model_id.split("/")[-1]}";',
            f'  --model-hash: "{model_hash}";',
            f"  --hidden-dim: {self.hidden_dim};",
            f"  --intermediate-dim: {self.intermediate_dim};",
            f"  --num-layers: {self.num_layers};",
            f"  --num-attention-heads: {self.num_heads};",
            f"  --num-kv-heads: {self.num_kv_heads};",
            f"  --head-dim: {self.head_dim};",
            f"  --vocab-size: {self.vocab_size};",
            '  --weight-precision: "int4";',
            '  --quant-scheme: "blockwise-symmetric";',
            '  --bridge-protocol: "webgpu";',
            '  --weight-base-url: "./";',
        ]
        for k, v in sorted(samples.items()):
            lines.append(f"  {k}: {v:.6f};")
        lines.append("}")
        return "\n".join(lines) + "\n"

    def create_manifest(self) -> Dict:
        return {
            "model_id": self.model_id,
            "format_version": "1.0",
            "quantization": "int4-blockwise",
            "architecture": {
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
                "intermediate_dim": self.intermediate_dim,
                "vocab_size": self.vocab_size,
            },
            "files": {
                "embeddings": "embeddings.bin",
                "layers": [f"layer_{i}.bin" for i in range(self.num_layers)],
                "lm_head": "lm_head.bin",
                "governance": "model.css",
                "offsets": "layer_offsets.json",
            },
        }

    def create_archive(self):
        archive = self.output_dir / f"{self.model_id.split('/')[-1]}.csslm"
        with tarfile.open(archive, "w:gz") as tar:
            for file in self.output_dir.glob("*"):
                if file != archive:
                    tar.add(file, arcname=file.name)

    def convert(self):
        emb = self.convert_embeddings()
        lay = self.convert_layers()
        head = self.convert_lm_head()

        css = self.generate_css_governance({**emb, **lay, **head})
        (self.output_dir / "model.css").write_text(css, encoding="utf-8")

        manifest = self.create_manifest()
        with open(self.output_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        self.create_archive()


def main():
    parser = argparse.ArgumentParser(description="Convert HF model to CSS-LLM format")
    parser.add_argument("model_id", type=str)
    parser.add_argument("--output", "-o", default="./csslm-models", type=str)
    args = parser.parse_args()

    out = Path(args.output) / args.model_id.split("/")[-1]
    CSSWeightConverter(args.model_id, out).convert()


if __name__ == "__main__":
    main()
