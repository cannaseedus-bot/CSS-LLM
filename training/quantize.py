#!/usr/bin/env python3
import numpy as np


def quantize_int4(weights: np.ndarray):
    scale = max(float(np.max(np.abs(weights))) / 7.0, 1e-8)
    q = np.clip(np.round(weights / scale), -8, 7).astype(np.int8)
    return q, np.float16(scale)
