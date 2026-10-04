"""Local contrast enhancement, so murky midtones keep readable shapes."""

from __future__ import annotations

import cv2
import numpy as np

from inkroto.io import from_uint16, to_uint16


def local_contrast(lum: np.ndarray, clip_limit: float, tiles: int) -> np.ndarray:
    """CLAHE on luminance (H, W) float32 0..1. Runs in 16 bit for precision."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tiles, tiles))
    return from_uint16(clahe.apply(to_uint16(lum)))
