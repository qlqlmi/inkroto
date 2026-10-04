"""Shot container and the motion backend protocol."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Protocol

import numpy as np

from inkroto.analyze import Analysis
from inkroto.config import Config

# compute(i) -> (frame RGB (H, W, 3), depth, matte, faces (each H, W)), all float32 0..1.
FrameFn = Callable[[int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]


class Shot:
    """A sequence of moving frames with per-frame depth, matte, and face mask.

    Frames are produced by `compute(i)`, so a backend can precompute arrays
    (see `from_arrays`) or render each frame lazily. The most recent frame is
    kept, so `frame(i)` followed by `analysis_at(i)` computes frame i once.
    A subclass may instead override `frame`, `analysis_at`, and `faces_at`
    separately and pass compute=None.
    """

    def __init__(self, n_frames: int, size: tuple[int, int], compute: FrameFn | None):
        if n_frames < 1:
            raise ValueError(f"a shot needs at least one frame, got {n_frames}")
        self.n_frames = n_frames
        self.width, self.height = size
        self._compute = compute
        self._cached: tuple[int, tuple[np.ndarray, ...]] | None = None

    @classmethod
    def from_arrays(
        cls,
        frames: np.ndarray,
        depths: np.ndarray,
        mattes: np.ndarray,
        faces: np.ndarray | None = None,
    ) -> Shot:
        n, h, w = frames.shape[:3]
        if faces is None:
            faces = np.zeros((n, h, w), np.float32)
        if any(a.shape != (n, h, w) for a in (depths, mattes, faces)):
            raise ValueError("frames, depths, mattes, and faces must agree on (N, H, W)")
        return cls(n, (w, h), lambda i: (frames[i], depths[i], mattes[i], faces[i]))

    def __len__(self) -> int:
        return self.n_frames

    def _get(self, i: int) -> tuple[np.ndarray, ...]:
        if not -self.n_frames <= i < self.n_frames:
            raise IndexError(f"frame {i} out of range for a {self.n_frames}-frame shot")
        i %= self.n_frames
        if self._cached is None or self._cached[0] != i:
            self._cached = (i, self._compute(i))
        return self._cached[1]

    def frame(self, i: int) -> np.ndarray:
        """RGB float32 (H, W, 3) for frame i."""
        return self._get(i)[0]

    def analysis_at(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """(depth, matte) for frame i, both float32 (H, W) 0..1."""
        _, depth, matte, _ = self._get(i)
        return depth, matte

    def faces_at(self, i: int) -> np.ndarray:
        """Soft face mask for frame i, float32 (H, W) 0..1."""
        return self._get(i)[3]

    @property
    def frames(self) -> FrameView:
        """Array-like (N, H, W, 3) view. Indexing renders on demand."""
        return FrameView(self)


class FrameView:
    """Lazy (N, H, W, 3) view over a Shot. `np.asarray(view)` materializes it."""

    def __init__(self, shot: Shot):
        self._shot = shot

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (len(self._shot), self._shot.height, self._shot.width, 3)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.float32)

    def __len__(self) -> int:
        return len(self._shot)

    def __getitem__(self, index: int | slice) -> np.ndarray:
        if isinstance(index, slice):
            return np.stack([self._shot.frame(i) for i in range(len(self))[index]])
        return self._shot.frame(index)

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(len(self)):
            yield self._shot.frame(i)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        arr = self[:]
        return arr if dtype is None else arr.astype(dtype)


class MotionBackend(Protocol):
    def build(self, image: np.ndarray, analysis: Analysis, cfg: Config) -> Shot: ...


def frame_count(cfg: Config) -> int:
    return max(1, int(round(cfg.output.duration_s * cfg.output.fps)))
