"""Orchestration: photo in, video out.

    load -> analyze -> motion backend -> group frames
         -> per drawing: average, stylize, color
         -> per frame:   atmosphere, film finish
         -> write
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from tqdm import tqdm

from inkroto.analyze import Analysis, analyze
from inkroto.cadence import drawing_rng, group_frames
from inkroto.composite.atmosphere import apply_atmosphere
from inkroto.composite.color import apply_color
from inkroto.composite.film import finish
from inkroto.config import Config, config_to_dict
from inkroto.io import (
    VideoWriter, is_video, load_image, save_debug_sheet, save_image, save_yaml,
)
from inkroto.motion import Shot, get_backend
from inkroto.stylize.pencil import tint_paper
from inkroto.stylize.pipeline import ShotState, stylize_drawing

DEBUG_SHEET_COLUMNS = 4


@dataclass
class Drawing:
    """One held drawing and the per-frame inputs that go with it."""

    image: np.ndarray  # RGB (H, W, 3): ink plus color accents
    depth: np.ndarray  # depth of the group's middle frame, for atmosphere
    frames: range      # output frames this drawing is held for


def build_shot(
    photo: np.ndarray | None,
    analysis: Analysis | None,
    cfg: Config,
    cache_dir: str | Path | None = ".cache",
) -> Shot:
    """Motion stage: the backend named by cfg.motion.backend."""
    return get_backend(cfg.motion.backend, cache_dir=cache_dir).build(photo, analysis, cfg)


def with_input(cfg: Config, input_path: str | Path) -> Config:
    """A video or gif INPUT becomes the clip backend's source."""
    if not is_video(input_path):
        return cfg
    motion = replace(cfg.motion, backend="clip", clip_path=str(input_path))
    return replace(cfg, motion=motion)


def average_group(
    shot: Shot, frames: range
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average the group's frames into one RGB image (removes motion shimmer);
    depth, matte, and faces come from the group's middle frame."""
    total = np.zeros((shot.height, shot.width, 3), np.float64)
    for i in frames:
        total += shot.frame(i)
    rgb = (total / len(frames)).astype(np.float32)
    middle = frames[len(frames) // 2]
    depth, matte = shot.analysis_at(middle)
    return rgb, depth, matte, shot.faces_at(middle)


def make_drawings(
    shot: Shot, cfg: Config, debug: dict[str, np.ndarray] | None = None
) -> Iterator[Drawing]:
    """Per drawing: average the group, stylize once, add color.

    Thresholds lock on the first drawing; each drawing has its own rng, so tone
    jitter, hatching, and wobble change once per drawing and never within a hold.
    """
    state = ShotState()
    for index, (start, end) in enumerate(group_frames(len(shot), cfg.cadence.hold)):
        frames = range(start, end)
        rgb, depth, matte, faces = average_group(shot, frames)
        stages = debug if index == 0 else None
        ink, lines = stylize_drawing(
            rgb, depth, matte, cfg, state, drawing_rng(cfg.seed, index), debug=stages, faces=faces
        )
        # Color comes from the same averaged image, so it holds with the drawing.
        image = apply_color(ink, rgb, cfg.color, lines)
        if cfg.stylize.medium == "pencil":
            image = tint_paper(image, cfg.stylize.pencil)
        if stages is not None:
            stages["color"] = image
        yield Drawing(image, depth, frames)


def finish_frame(drawing: Drawing, frame_index: int, cfg: Config) -> np.ndarray:
    """Per output frame: atmosphere, then the film finish (always last)."""
    frame = apply_atmosphere(drawing.image, drawing.depth, frame_index, cfg, cfg.seed)
    return finish(frame, frame_index, cfg, cfg.seed)


def render_frames(
    photo: np.ndarray | None,
    analysis: Analysis | None,
    cfg: Config,
    debug: dict[str, np.ndarray] | None = None,
    cache_dir: str | Path | None = ".cache",
) -> Iterator[np.ndarray]:
    """Yield every finished output frame as RGB (H, W, 3) float32 0..1.

    If `debug` is a dict, the first drawing's stages are recorded in it.
    """
    shot = build_shot(photo, analysis, cfg, cache_dir)
    for drawing in make_drawings(shot, cfg, debug):
        for i in drawing.frames:
            yield finish_frame(drawing, i, cfg)


def debug_frame_indices(n_frames: int) -> list[int]:
    """First, middle, and last frame, without repeats."""
    return sorted({0, (n_frames - 1) // 2, n_frames - 1})


def render(
    input_path: str | Path,
    output_path: str | Path,
    cfg: Config,
    cache_dir: str | Path | None = ".cache",
    debug_dir: str | Path | None = None,
) -> Path:
    """Render `input_path` to a video at `output_path`. Returns the output path.

    A photo gets the configured motion backend (parallax by default). A video or
    gif input is used as the motion source through the clip backend.

    Also writes the resolved config next to the video as `<name>.config.yaml`.
    With `debug_dir`, saves `<name>_stages.png` (every stage of the first drawing)
    and the first, middle, and last finished frames as PNGs.
    """
    output_path = Path(output_path)
    cfg = with_input(cfg, input_path)
    if cfg.motion.backend == "clip":
        photo, analysis = None, None  # the clip brings its own frames and analysis
    else:
        photo = load_image(input_path, cfg.output.width)   # load
        fc = cfg.stylize.faces
        analysis = analyze(photo, cache_dir, faces=fc.enabled, face_min_score=fc.min_score)
    shot = build_shot(photo, analysis, cfg, cache_dir)      # motion
    width, height = shot.width, shot.height
    n_frames = len(shot)
    stages: dict[str, np.ndarray] | None = {} if debug_dir is not None else None
    keep = set(debug_frame_indices(n_frames)) if debug_dir is not None else set()

    progress = tqdm(total=n_frames, desc=output_path.name, unit="frame")
    with VideoWriter(output_path, cfg.output.fps, (width, height)) as writer:
        for drawing in make_drawings(shot, cfg, stages):    # per drawing
            if stages and drawing.frames.start == 0:
                save_debug_sheet(
                    stages, Path(debug_dir) / f"{output_path.stem}_stages.png",
                    cols=DEBUG_SHEET_COLUMNS,
                )
            for i in drawing.frames:                         # per frame
                frame = finish_frame(drawing, i, cfg)
                writer.write(frame)                          # write
                if i in keep:
                    save_image(Path(debug_dir) / f"{output_path.stem}_frame_{i:04d}.png", frame)
                progress.update(1)
    progress.close()

    save_yaml(config_to_dict(cfg), config_path_for(output_path))
    return output_path


def config_path_for(output_path: str | Path) -> Path:
    """`clip_001.mp4` -> `clip_001.config.yaml`."""
    return Path(output_path).with_suffix(".config.yaml")
