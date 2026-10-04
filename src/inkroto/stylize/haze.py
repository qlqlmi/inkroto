"""Aerial perspective: far things fade toward a flat gray."""

from __future__ import annotations

import numpy as np


def haze(
    lum: np.ndarray, depth: np.ndarray, start: float, density: float, value: float
) -> np.ndarray:
    """Blend luminance toward `value` where depth < start (1 = near, 0 = far)."""
    if start <= 0:
        return np.asarray(lum, dtype=np.float32)
    amount = np.clip((start - depth) / start, 0.0, 1.0) * density
    out = lum + (value - lum) * amount
    return np.clip(out, 0.0, 1.0).astype(np.float32)
