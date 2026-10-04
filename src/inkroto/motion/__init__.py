"""Camera and subject motion backends."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from inkroto.motion.base import MotionBackend, Shot, frame_count
from inkroto.motion.clip import ClipBackend, ClipShot
from inkroto.motion.parallax import ParallaxBackend

# name in cfg.motion.backend -> factory taking cache_dir
BACKENDS: dict[str, Callable[..., MotionBackend]] = {
    "parallax": ParallaxBackend,
    "clip": ClipBackend,
}
# Named but not built yet.
PLANNED_BACKENDS: tuple[str, ...] = ()


def register_backend(name: str, factory: Callable[..., MotionBackend]) -> None:
    BACKENDS[name] = factory


def get_backend(name: str, cache_dir: str | Path | None = ".cache") -> MotionBackend:
    """The motion backend registered under `name`."""
    if name in BACKENDS:
        return BACKENDS[name](cache_dir=cache_dir)
    if name in PLANNED_BACKENDS:
        raise NotImplementedError(f"the {name} motion backend is not implemented yet")
    raise ValueError(
        f"unknown motion backend '{name}' (available: {', '.join(sorted(BACKENDS))})"
    )


__all__ = [
    "BACKENDS", "ClipBackend", "ClipShot", "MotionBackend", "ParallaxBackend", "Shot", "frame_count",
    "get_backend", "register_backend",
]
