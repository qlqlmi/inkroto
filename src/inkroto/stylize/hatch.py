"""Hand-drawn hatching for the mid tone."""

from __future__ import annotations

import math

import numpy as np

from inkroto.stylize.noise import smooth_noise


def hatch_pattern(
    shape: tuple[int, int],
    spacing: float,
    angle: float,
    thickness: float,
    rng: np.random.Generator,
    jitter_px: float,
    thickness_jitter: float,
    noise_scale: float,
    segment: list[float],
    gap: list[float],
) -> np.ndarray:
    """Parallel uneven ink strokes at `angle` degrees. Returns (H, W) float32, 0 = ink.

    Stroke position and thickness wander with low-frequency noise, strokes are
    broken into segments of random length, and the whole pattern is shifted by a
    random offset, all drawn from `rng`.
    """
    h, w = shape
    theta = math.radians(angle)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    along = xs * math.cos(theta) + ys * math.sin(theta)    # position along a stroke
    across = -xs * math.sin(theta) + ys * math.cos(theta)  # position across strokes

    across = across + rng.uniform(0.0, spacing)
    along = along + rng.uniform(0.0, spacing)
    across = across + jitter_px * smooth_noise(shape, noise_scale, rng)
    width = thickness * (1.0 + thickness_jitter * smooth_noise(shape, noise_scale, rng))

    index = np.floor(across / spacing + 0.5)
    dist = np.abs(across - index * spacing)
    ink = np.clip(width / 2.0 + 0.5 - dist, 0.0, 1.0)  # anti-aliased stroke profile

    # Break each stroke into segments: a 1-D on/off track per stroke index.
    k = (index - index.min()).astype(np.int64)
    u0 = math.floor(float(along.min()))
    u = (np.floor(along) - u0).astype(np.int64)
    n_strokes, length = int(k.max()) + 1, int(u.max()) + 1
    seg_lo, seg_hi = sorted(segment)
    gap_lo, gap_hi = sorted(gap)
    n_pieces = math.ceil(length / max(seg_lo + gap_lo, 1.0)) + 2
    lengths = np.empty((n_strokes, 2 * n_pieces))
    lengths[:, 0::2] = rng.uniform(seg_lo, seg_hi, size=(n_strokes, n_pieces))
    lengths[:, 1::2] = rng.uniform(gap_lo, gap_hi, size=(n_strokes, n_pieces))
    starts = rng.uniform(0.0, seg_hi + gap_hi, size=(n_strokes, 1))
    edges = np.cumsum(lengths, axis=1) - starts
    positions = np.arange(length)
    track = np.empty((n_strokes, length), dtype=bool)
    for row in range(n_strokes):  # loop over strokes, not pixels
        track[row] = np.searchsorted(edges[row], positions, side="right") % 2 == 0

    ink = ink * track[k, u]
    return (1.0 - ink).astype(np.float32)


def apply_hatch(tones: np.ndarray, mid_mask: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """Replace the flat mid tone with white paper carrying the hatch strokes."""
    return np.where(mid_mask, pattern, tones).astype(np.float32)
