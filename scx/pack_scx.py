#!/usr/bin/env python3
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"SCXM"
VERSION = 1


def quantize_int4(tensor: np.ndarray):
    max_abs = float(np.max(np.abs(tensor))) if tensor.size else 0.0
    scale = max(max_abs / 7.0, 1e-8)
    quant = np.clip(np.round(tensor / scale), -8, 7).astype(np.int8)
    return quant, np.array([scale], dtype=np.float16)


def pack_int4(values: np.ndarray) -> bytes:
    flat = values.flatten().astype(np.int8)
    packed = bytearray((len(flat) + 1) // 2)
    for i in range(0, len(flat), 2):
        lo = int(flat[i]) & 0xF
        hi = int(flat[i + 1]) & 0xF if i + 1 < len(flat) else 0
        packed[i // 2] = lo | (hi << 4)
    return bytes(packed)


def write_scx(path: Path, tensors: dict):
    buf = bytearray()
    buf.extend(MAGIC)
    buf.extend(struct.pack("<I", VERSION))
    buf.extend(struct.pack("<I", len(tensors)))

    for name in sorted(tensors.keys()):
      tensor = np.array(tensors[name], dtype=np.float32)
      q, scales = quantize_int4(tensor)
      packed = pack_int4(q)

      name_bytes = name.encode("utf-8")
      buf.extend(struct.pack("<I", len(name_bytes)))
      buf.extend(name_bytes)
      buf.extend(struct.pack("<I", len(packed)))
      buf.extend(packed)
      scale_bytes = scales.tobytes()
      buf.extend(struct.pack("<I", len(scale_bytes) // 2))
      buf.extend(scale_bytes)

    digest = hashlib.sha256(buf).digest()
    buf.extend(digest)
    path.write_bytes(buf)


def main():
    parser = argparse.ArgumentParser(description="Pack SCX mini deterministic INT4 weights")
    parser.add_argument("input", help="Path to JSON tensor map")
    parser.add_argument("output", help="Path to .scx output")
    args = parser.parse_args()

    tensors = json.loads(Path(args.input).read_text())
    write_scx(Path(args.output), tensors)


if __name__ == "__main__":
    main()
