"""Directional shading from depth normals, for hard, readable shadows."""

from __future__ import annotations

import cv2
import numpy as np

# A 3x3 Sobel kernel returns 8x the per-pixel derivative.
SOBEL3_GAIN = 8.0


def relight(
    lum: np.ndarray,
    depth: np.ndarray,
    direction: list[float],
    strength: float,
    normal_blur: float,
    normal_z: float,
) -> np.ndarray:
    """Shade luminance by the angle between depth normals and a light direction.

    `direction` is (x, y, z) with x to the right, y down, and z toward the viewer.
    Depth slopes are measured per image width, so the result does not depend on
    resolution. Returns lum * (1 - strength + strength * shade).
    """
    d = np.asarray(depth, dtype=np.float32)
    if normal_blur > 0:
        d = cv2.GaussianBlur(d, (0, 0), normal_blur)
    scale = np.float32(max(d.shape) / SOBEL3_GAIN)
    dx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT) * scale
    dy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT) * scale

    normals = np.stack([-dx, -dy, np.full_like(dx, normal_z)], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    light = np.asarray(direction, dtype=np.float32)
    light /= np.linalg.norm(light)
    shade = np.clip(normals @ light, 0.0, 1.0)

    out = lum * (1.0 - strength + strength * shade)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
