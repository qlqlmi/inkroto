"""Clip backend: an existing video (or gif) is the motion source.

Frames are resampled to the output fps, resized, and trimmed. Depth and matte
(and faces) are estimated lazily, only for the frame each drawing is made from (the middle
of its cadence group), and cached on disk per video and frame. Depth is smoothed
across drawings and normalized with the first drawing's range, so haze and
relight do not pump as the model's per-frame scale wanders. No parallax camera
move is added on top.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np

from inkroto import analyze as analyze_mod
from inkroto.analyze import Analysis, depth_range, normalize_raw_depth
from inkroto.cadence import group_frames
from inkroto.config import Config
from inkroto.io import FrameStack, file_sha1, read_video_resampled
from inkroto.motion.base import Shot, frame_count

# Finished (depth, matte) results kept in memory, most recent drawings first.
RESULT_CACHE_SIZE = 2


class ClipShot(Shot):
    def __init__(
        self, frames: FrameStack, video_hash: str, cfg: Config, cache_dir: str | Path | None
    ):
        super().__init__(len(frames), frames.size, compute=None)
        self._frames = frames
        self._cache_dir = cache_dir
        self._faces_enabled = cfg.stylize.faces.enabled
        self._face_min_score = cfg.stylize.faces.min_score
        self._alpha = float(cfg.motion.depth_smoothing)
        if not 0.0 < self._alpha <= 1.0:
            raise ValueError(f"motion.depth_smoothing must be in (0, 1], got {self._alpha}")
        groups = group_frames(len(frames), cfg.cadence.hold)
        self.drawing_frames = [start + (end - start) // 2 for start, end in groups]
        self._drawing_of = np.repeat(np.arange(len(groups)), [end - start for start, end in groups])
        w, h = frames.size
        self._key = f"clip:{video_hash}:{cfg.output.fps}fps:{w}x{h}"
        # Running state: smoothed raw depth up to drawing `_ema_index`.
        self._ema_index = -1
        self._ema: np.ndarray | None = None
        self._range: tuple[float, float] | None = None
        self._results: OrderedDict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()

    def _check_index(self, i: int) -> int:
        if not -self.n_frames <= i < self.n_frames:
            raise IndexError(f"frame {i} out of range for a {self.n_frames}-frame shot")
        return i % self.n_frames

    def frame(self, i: int) -> np.ndarray:
        return self._frames[self._check_index(i)]

    def analysis_at(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """(depth, matte) of the drawing frame `i` belongs to.

        Each drawing is analyzed once, from its middle frame. Depth is an
        exponential moving average over drawings, normalized with the first
        drawing's range for the whole shot.
        """
        depth, matte, _ = self._drawing_analysis(i)
        return depth, matte

    def faces_at(self, i: int) -> np.ndarray:
        """Face mask of the drawing frame `i` belongs to (zeros if faces are off)."""
        return self._drawing_analysis(i)[2]

    def _drawing_analysis(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        k = int(self._drawing_of[self._check_index(i)])
        if k in self._results:
            self._results.move_to_end(k)
            return self._results[k]
        if k <= self._ema_index:  # going back: replay from the start (disk cache)
            self._ema_index, self._ema = -1, None
        matte = faces = None
        while self._ema_index < k:
            j = self._ema_index + 1
            source = self.drawing_frames[j]
            raw, matte, faces = analyze_mod.analyze_raw(
                self._frames[source], key=f"{self._key}:frame{source}",
                cache_dir=self._cache_dir, faces=self._faces_enabled,
                face_min_score=self._face_min_score,
            )
            if j == 0:
                self._range = depth_range(raw)
                self._ema = raw
            else:
                self._ema = self._alpha * raw + (1.0 - self._alpha) * self._ema
            self._ema_index = j
        depth = normalize_raw_depth(self._ema, *self._range)
        if faces is None:
            faces = np.zeros_like(depth)
        self._results[k] = (depth, matte, faces)
        while len(self._results) > RESULT_CACHE_SIZE:
            self._results.popitem(last=False)
        return self._results[k]


class ClipBackend:
    def __init__(self, cache_dir: str | Path | None = ".cache"):
        self.cache_dir = cache_dir

    def build(self, image: np.ndarray | None, analysis: Analysis | None, cfg: Config) -> Shot:
        """The photo and its analysis are not used: the clip supplies both."""
        path = cfg.motion.clip_path
        if not path:
            raise ValueError("motion.backend is 'clip' but motion.clip_path is not set")
        frames = read_video_resampled(path, cfg.output.fps, cfg.output.width, frame_count(cfg))
        return ClipShot(frames, file_sha1(path), cfg, self.cache_dir)
