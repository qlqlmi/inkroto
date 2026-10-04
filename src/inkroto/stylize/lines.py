"""Ink contour lines."""

from __future__ import annotations

import math

import cv2
import numpy as np

from inkroto.stylize.noise import smooth_noise


def xdog(
    lum: np.ndarray, sigma: float, k: float, p: float, epsilon: float, phi: float
) -> np.ndarray:
    """Extended difference of Gaussians. Returns (H, W) float32 0..1, 0 = line."""
    lum = np.asarray(lum, dtype=np.float32)
    g1 = cv2.GaussianBlur(lum, (0, 0), sigma)
    g2 = cv2.GaussianBlur(lum, (0, 0), k * sigma)
    d = (1.0 + p) * g1 - p * g2
    out = np.where(d >= epsilon, 1.0, 1.0 + np.tanh(phi * (d - epsilon)))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def denoise(lum: np.ndarray, strength: float, size_px: float) -> np.ndarray:
    """Edge-preserving (bilateral) smoothing that removes grain but keeps edges.
    `strength` is the luminance difference treated as noise; 0 returns the input."""
    lum = np.asarray(lum, dtype=np.float32)
    if strength <= 0 or size_px <= 0:
        return lum
    diameter = 2 * int(math.ceil(size_px)) + 1
    return cv2.bilateralFilter(lum, diameter, strength, size_px)


def xdog_edges(
    lum: np.ndarray, sigma: float, k: float, p: float, threshold: float, phi: float
) -> np.ndarray:
    """Contrast-only XDoG: ink on the dark side of edges, whatever their brightness.

    Plain `xdog` thresholds (1 + p) * G1 - p * G2, which is the luminance plus a
    sharpened edge term, so it misses edges in bright areas and inks flat dark
    areas. Here only the edge term E = (1 + p) * (G1 - G2) is used: a line where
    E < -threshold, softened by tanh. Returns (H, W) float32 0..1, 0 = line.
    """
    lum = np.asarray(lum, dtype=np.float32)
    g1 = cv2.GaussianBlur(lum, (0, 0), sigma)
    g2 = cv2.GaussianBlur(lum, (0, 0), k * sigma)
    edge = (1.0 + p) * (g1 - g2)
    out = np.where(edge >= -threshold, 1.0, 1.0 + np.tanh(phi * (edge + threshold)))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def wobble(
    line_img: np.ndarray, amount_px: float, rng: np.random.Generator, scale_px: float
) -> np.ndarray:
    """Displace the line layer by a smooth random field of at most `amount_px`,
    so outlines look drawn by hand. `scale_px` sets the size of the bends."""
    line_img = np.asarray(line_img, dtype=np.float32)
    if amount_px <= 0:
        return line_img.copy()
    h, w = line_img.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = xs + np.float32(amount_px) * smooth_noise((h, w), scale_px, rng)
    map_y = ys + np.float32(amount_px) * smooth_noise((h, w), scale_px, rng)
    out = cv2.remap(line_img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
