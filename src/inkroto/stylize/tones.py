"""Flat tone separation: black ink, one mid gray, white paper."""

from __future__ import annotations

import numpy as np

# Rec. 709 luma weights for R, G, B.
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """RGB (H, W, 3) float32 0..1 to grayscale (H, W) float32 0..1."""
    return np.clip(rgb @ LUMA_WEIGHTS, 0.0, 1.0).astype(np.float32)


def compute_thresholds(lum: np.ndarray, percentiles: list[float]) -> tuple[float, float]:
    """Luminance values at the two given percentiles of the histogram, as (t1, t2)."""
    if len(percentiles) != 2:
        raise ValueError(f"expected two percentiles, got {percentiles}")
    lo, hi = sorted(float(p) for p in percentiles)
    t1, t2 = np.percentile(lum, [lo, hi])
    return float(t1), float(t2)


def jitter_thresholds(
    thresholds: tuple[float, float], jitter: float, rng: np.random.Generator
) -> tuple[float, float]:
    """Each threshold plus its own Gaussian offset (std `jitter`), kept in order."""
    offsets = rng.normal(0.0, jitter, size=2)
    t1, t2 = sorted(float(t) + float(o) for t, o in zip(thresholds, offsets))
    return t1, t2


def quantize_tones(
    lum: np.ndarray, thresholds: tuple[float, float], mid_value: float
) -> tuple[np.ndarray, np.ndarray]:
    """0 below t1, mid_value from t1 to t2, 1 above. Returns (tones, mid_mask)."""
    t1, t2 = thresholds
    mid_mask = (lum >= t1) & (lum < t2)
    tones = np.where(lum < t1, 0.0, np.where(mid_mask, mid_value, 1.0))
    return tones.astype(np.float32), mid_mask


def apply_tones(
    lum: np.ndarray,
    thresholds: tuple[float, float],
    mid_value: float,
    jitter: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize luminance to 0, mid_value, and 1. Returns (tones, mid_mask).

    Each threshold gets its own Gaussian offset (std `jitter`), drawn once per
    call, so successive drawings of the same shot differ slightly.
    """
    return quantize_tones(lum, jitter_thresholds(thresholds, jitter, rng), mid_value)
