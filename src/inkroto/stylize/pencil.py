"""Pencil medium: graphite lines and shading on textured paper.

Used when `stylize.medium` is "pencil". The same contours and tone regions as
the ink look are rendered as graphite: gray, sketchy, pressure-varied lines
drawn in several passes; shading instead of solid black; and a paper grain that
breaks up the graphite. The paper is generated once per shot from the seed, so
it stays still while the drawing redraws.
"""

from __future__ import annotations

import cv2
import numpy as np

from inkroto.config import PencilConfig
from inkroto.stylize.lines import wobble
from inkroto.stylize.noise import smooth_noise

# Random stream for the paper texture, combined with the seed.
PAPER_STREAM = 21
# The paper's coarse fibre layer is this many times larger than its grain,
# and contributes this share of the texture.
PAPER_FIBRE_SCALE = 4.0
PAPER_FIBRE_SHARE = 0.3


def paper_texture(shape: tuple[int, int], cfg: PencilConfig, seed: int) -> np.ndarray:
    """Paper grain, (H, W) float32 in 0..1: 1 = a raised fibre that catches graphite."""
    rng = np.random.default_rng([seed, PAPER_STREAM])
    grain = smooth_noise(shape, cfg.paper_scale, rng)
    fibre = smooth_noise(shape, cfg.paper_scale * PAPER_FIBRE_SCALE, rng)
    mixed = (1.0 - PAPER_FIBRE_SHARE) * grain + PAPER_FIBRE_SHARE * fibre
    return np.clip(0.5 + 0.5 * mixed, 0.0, 1.0).astype(np.float32)


def _tooth(darkness: np.ndarray, paper: np.ndarray, amount: float) -> np.ndarray:
    """Graphite only sticks fully on raised grain: scale darkness by the paper."""
    return darkness * ((1.0 - amount) + amount * paper)


def graphite_lines(
    lines: np.ndarray, paper: np.ndarray, cfg: PencilConfig, rng: np.random.Generator,
    wobble_scale: float,
) -> np.ndarray:
    """Turn an ink line layer (0 = line) into pencil lines (H, W) float32, 1 = paper.

    Each of `passes` redraws the lines with its own small wobble, and the passes
    are overlaid, so contours look sketched over more than once. Darkness tops out
    at `line_value`, varies along strokes with `pressure`, has gaps where
    `break_rate` says, and is broken up by the paper grain.
    """
    shape = lines.shape
    darkness = np.zeros(shape, np.float32)
    for _ in range(max(1, cfg.passes)):
        drawn = wobble(lines, cfg.pass_offset_px, rng, wobble_scale) if cfg.passes > 1 else lines
        darkness = np.maximum(darkness, 1.0 - drawn)
    darkness *= np.float32(1.0 - cfg.line_value)

    if cfg.pressure > 0:
        pressure = 0.5 + 0.5 * smooth_noise(shape, cfg.pressure_scale, rng)
        darkness *= (1.0 - cfg.pressure) + cfg.pressure * pressure
    if cfg.break_rate > 0:
        gaps = smooth_noise(shape, cfg.break_scale, rng)
        darkness *= gaps > np.quantile(gaps, cfg.break_rate)
    darkness = _tooth(darkness, paper, cfg.paper_tooth)
    return np.clip(1.0 - darkness, 0.0, 1.0).astype(np.float32)


def graphite_shading(
    dark_mask: np.ndarray,
    mid_mask: np.ndarray,
    strokes: np.ndarray | None,
    paper: np.ndarray,
    cfg: PencilConfig,
) -> np.ndarray:
    """Shading (H, W) float32, 1 = paper, from the tone regions.

    Shadows get the dark graphite value instead of solid black, the mid tone a
    light even layer, and hatch strokes (0 = stroke) darken the mid tone toward
    the shadow value. The paper grain shows through, and `smudge` softens it all.
    """
    dark_value, mid_value = cfg.shade_values
    shade = np.ones(dark_mask.shape, np.float32)
    shade[mid_mask] = mid_value
    if strokes is not None:
        stroke = (1.0 - strokes) * mid_mask
        shade = shade - stroke * np.float32(mid_value - dark_value)
    shade[dark_mask] = dark_value
    darkness = _tooth(1.0 - shade, paper, cfg.paper_tooth)
    if cfg.smudge > 0:
        darkness = cv2.GaussianBlur(darkness, (0, 0), cfg.smudge)
    return np.clip(1.0 - darkness, 0.0, 1.0).astype(np.float32)


def paper_finish(drawing: np.ndarray, paper: np.ndarray, cfg: PencilConfig) -> np.ndarray:
    """Let the paper texture show faintly everywhere, including blank paper."""
    out = drawing * (1.0 - np.float32(cfg.paper_grain) * (1.0 - paper))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def tint_paper(rgb: np.ndarray, cfg: PencilConfig) -> np.ndarray:
    """Warm the white of the page (and the graphite with it)."""
    return (rgb * np.asarray(cfg.paper_tint, np.float32)).astype(np.float32)
