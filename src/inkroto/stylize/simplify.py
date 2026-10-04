"""Edge-preserving simplification into flat, paintable regions."""

from __future__ import annotations

import cv2
import numpy as np

from inkroto.io import from_uint8, to_uint8

METHODS = ("kuwahara", "meanshift", "bilateral")


def kuwahara(lum: np.ndarray, radius: int) -> np.ndarray:
    """Classic Kuwahara filter: mean of the least-varying of four quadrants."""
    lum = np.asarray(lum, dtype=np.float32)
    if radius < 1:
        return lum.copy()
    size = (radius + 1, radius + 1)
    sq = lum * lum
    # Anchor = where the output pixel sits inside the (r+1)x(r+1) window:
    # bottom-right (window above-left), bottom-left, top-right, top-left.
    anchors = [(radius, radius), (0, radius), (radius, 0), (0, 0)]
    means, variances = [], []
    for anchor in anchors:
        m = cv2.boxFilter(lum, -1, size, anchor=anchor, borderType=cv2.BORDER_REFLECT)
        m2 = cv2.boxFilter(sq, -1, size, anchor=anchor, borderType=cv2.BORDER_REFLECT)
        means.append(m)
        variances.append(m2 - m * m)
    means = np.stack(means)
    best = np.argmin(np.stack(variances), axis=0)
    return np.take_along_axis(means, best[None], axis=0)[0]


def meanshift(lum: np.ndarray, radius: int, sigma_color: float) -> np.ndarray:
    rgb = np.repeat(np.asarray(lum, dtype=np.float32)[..., None], 3, axis=2)
    out = cv2.pyrMeanShiftFiltering(to_uint8(rgb), radius, sigma_color * 255.0)
    return from_uint8(out[..., 0])


def bilateral(lum: np.ndarray, radius: int, sigma_color: float) -> np.ndarray:
    return cv2.bilateralFilter(
        np.asarray(lum, dtype=np.float32), 2 * radius + 1, sigma_color, radius
    )


def simplify(lum: np.ndarray, method: str, radius: int, sigma_color: float) -> np.ndarray:
    """Flatten texture while keeping edges. Returns (H, W) float32 0..1."""
    if method == "kuwahara":
        out = kuwahara(lum, radius)
    elif method == "meanshift":
        out = meanshift(lum, radius, sigma_color)
    elif method == "bilateral":
        out = bilateral(lum, radius, sigma_color)
    else:
        raise ValueError(f"unknown simplify method '{method}' (expected one of {METHODS})")
    return np.clip(out, 0.0, 1.0).astype(np.float32)
