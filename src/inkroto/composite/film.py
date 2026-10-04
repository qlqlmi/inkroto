"""Film finish: weave, flicker, vignette, and grain.

Every part is a stateless function of the frame index and the seed, so any
frame can be finished on its own.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from inkroto.config import Config
from inkroto.stylize.tones import luminance

# Independent random streams, combined with the seed.
GRAIN_STREAM = 11
FLICKER_STREAM = 12
WEAVE_X_STREAM = 13
WEAVE_Y_STREAM = 14


def _knot(seed: int, stream: int, index: int) -> float:
    return float(np.random.default_rng([seed, stream, index]).uniform(-1.0, 1.0))


def smooth_noise_1d(frame_index: int, period: int, seed: int, stream: int) -> float:
    """Smooth noise in -1..1 over frames: random knots every `period` frames, eased
    between with a smoothstep. Computed directly from the frame index."""
    period = max(1, period)
    k, rem = divmod(frame_index, period)
    t = rem / period
    t = t * t * (3.0 - 2.0 * t)
    a, b = _knot(seed, stream, k), _knot(seed, stream, k + 1)
    return a + (b - a) * t


def flicker_gain(frame_index: int, amplitude: float, period: int, seed: int) -> float:
    """Global brightness gain, 1 +- amplitude, varying smoothly frame to frame."""
    return 1.0 + amplitude * smooth_noise_1d(frame_index, period, seed, FLICKER_STREAM)


def weave(frame_rgb: np.ndarray, frame_index: int, amount_px: float, period: int, seed: int) -> np.ndarray:
    """Sub-pixel translation of the whole frame, at most `amount_px` each way."""
    dx = amount_px * smooth_noise_1d(frame_index, period, seed, WEAVE_X_STREAM)
    dy = amount_px * smooth_noise_1d(frame_index, period, seed, WEAVE_Y_STREAM)
    h, w = frame_rgb.shape[:2]
    shift = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        frame_rgb, shift, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )


def vignette_gain(shape: tuple[int, int], strength: float, power: float) -> np.ndarray:
    """(H, W) gain: 1 at the center, 1 - strength at the corners."""
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xs - (w - 1) / 2.0) / ((w - 1) / 2.0 if w > 1 else 1.0)
    ny = (ys - (h - 1) / 2.0) / ((h - 1) / 2.0 if h > 1 else 1.0)
    r = np.sqrt(nx * nx + ny * ny) / np.float32(math.sqrt(2.0))  # 1 at the corners
    return (1.0 - strength * r**power).astype(np.float32)


def grain(frame_rgb: np.ndarray, frame_index: int, strength: float, seed: int) -> np.ndarray:
    """Per-frame Gaussian grain, the same on all channels, strongest in midtones."""
    rng = np.random.default_rng([seed, GRAIN_STREAM, frame_index])
    noise = rng.normal(0.0, strength, size=frame_rgb.shape[:2]).astype(np.float32)
    lum = luminance(np.clip(frame_rgb, 0.0, 1.0))
    midtone = 4.0 * lum * (1.0 - lum)  # 1 at mid gray, 0 at pure black and white
    return frame_rgb + (noise * midtone)[..., None]


def finish(frame_rgb: np.ndarray, frame_index: int, cfg: Config, seed: int) -> np.ndarray:
    """Weave, flicker, vignette, and grain, then clip to 0..1. Parts with zero
    strength are skipped, so all-zero strengths return the frame unchanged."""
    fc = cfg.film
    out = np.array(frame_rgb, dtype=np.float32, copy=True)
    if fc.weave_px > 0:
        out = weave(out, frame_index, fc.weave_px, fc.weave_frames, seed)
    if fc.flicker > 0:
        out = out * np.float32(flicker_gain(frame_index, fc.flicker, fc.flicker_frames, seed))
    if fc.vignette > 0:
        out = out * vignette_gain(out.shape[:2], fc.vignette, fc.vignette_power)[..., None]
    if fc.grain > 0:
        out = grain(out, frame_index, fc.grain, seed)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
