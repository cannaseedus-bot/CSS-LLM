#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np

from quantize import quantize_int4


def export_numpy_dict(npz_path: str, out_path: str):
    src = np.load(npz_path)
    payload = {}
    for key in src.files:
        q, scale = quantize_int4(src[key].astype(np.float32))
        payload[key] = {
            "shape": list(src[key].shape),
            "scale": float(scale),
            "q": q.flatten().tolist(),
        }
    Path(out_path).write_text(json.dumps(payload))


if __name__ == "__main__":
    raise SystemExit("Usage: import and call export_numpy_dict(npz, out_json)")
