"""Held drawings: group output frames so each drawing lasts `hold` frames."""

from __future__ import annotations

import numpy as np


def group_frames(n_frames: int, hold: int) -> list[tuple[int, int]]:
    """Split frames 0..n_frames-1 into (start, end) ranges of `hold` frames.

    `end` is exclusive. The last group is shorter when n_frames is not a
    multiple of hold.
    """
    if hold < 1:
        raise ValueError(f"cadence.hold must be at least 1, got {hold}")
    if n_frames < 0:
        raise ValueError(f"n_frames must be non-negative, got {n_frames}")
    return [(start, min(start + hold, n_frames)) for start in range(0, n_frames, hold)]


def drawing_rng(seed: int, drawing_index: int) -> np.random.Generator:
    """Generator for one drawing, seeded from both values.

    A drawing's randomness depends only on the seed and its own index, never on
    how many frames or drawings came before it.
    """
    return np.random.default_rng([seed, drawing_index])
