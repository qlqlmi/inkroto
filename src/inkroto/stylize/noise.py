"""Smooth random fields shared by hatching and line wobble."""

from __future__ import annotations

import math

import cv2
import numpy as np


def smooth_noise(shape: tuple[int, int], cell_px: float, rng: np.random.Generator) -> np.ndarray:
    """Low-frequency noise in -1..1: random values on a grid with `cell_px`
    spacing, upscaled to `shape` (H, W) with bicubic interpolation."""
    h, w = shape
    gh = max(2, math.ceil(h / cell_px) + 1)
    gw = max(2, math.ceil(w / cell_px) + 1)
    coarse = rng.uniform(-1.0, 1.0, size=(gh, gw)).astype(np.float32)
    field = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(field, -1.0, 1.0)
