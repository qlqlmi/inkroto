"""Atmosphere over the drawing: drifting smoke, rising embers, falling ash.

Every function is stateless: the overlay for a frame is computed directly from
the frame index and the seed, so any frame can be rendered on its own. These
effects change on every output frame, while the drawing underneath holds.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import cv2
import numpy as np

from inkroto.config import AshConfig, Config, EmbersConfig, SmokeConfig

# Independent random streams per effect, combined with the seed.
SMOKE_STREAM = 1
EMBERS_STREAM = 2
ASH_STREAM = 3

# Each noise octave has twice the cells and this fraction of the amplitude.
NOISE_PERSISTENCE = 0.5
# Cells of wrap-around padding used to make upscaled noise tile seamlessly.
NOISE_PAD_CELLS = 2
# Particles respawn at a new spot each lifetime, stepped by golden-ratio offsets.
GOLDEN = 0.6180339887498949
# A soft dot is drawn out to this many radii.
DOT_EXTENT = 3.0
# A 3x3 Sobel kernel returns 8x the per-pixel derivative.
SOBEL3_GAIN = 8.0

SMOKE_STYLES = ("soft", "flat")


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _tileable_upscale(small: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bicubic upscale of `small` to `shape` (H, W) that wraps around at the edges."""
    h, w = shape
    cy, cx = small.shape
    sx, sy = w / cx, h / cy
    padded = np.pad(small, NOISE_PAD_CELLS, mode="wrap")
    size = (round((cx + 2 * NOISE_PAD_CELLS) * sx), round((cy + 2 * NOISE_PAD_CELLS) * sy))
    big = cv2.resize(padded, size, interpolation=cv2.INTER_CUBIC)
    ox, oy = round(NOISE_PAD_CELLS * sx), round(NOISE_PAD_CELLS * sy)
    return big[oy : oy + h, ox : ox + w]


def fractal_noise(
    shape: tuple[int, int], octaves: int, rng: np.random.Generator, base_cells: int
) -> np.ndarray:
    """Tileable multi-octave value noise, (H, W) float32 in 0..1.

    Octave o is a (base_cells * 2**o)-wide grid of random values upscaled with
    bicubic interpolation; octaves are summed with halving amplitude.
    """
    h, w = shape
    total = np.zeros((h, w), np.float32)
    amplitude, norm = 1.0, 0.0
    for octave in range(octaves):
        cx = max(1, base_cells * 2**octave)
        cy = max(1, round(cx * h / w))
        small = rng.uniform(-1.0, 1.0, size=(cy, cx)).astype(np.float32)
        total += np.float32(amplitude) * _tileable_upscale(small, shape)
        norm += amplitude
        amplitude *= NOISE_PERSISTENCE
    return np.clip(total / np.float32(norm) * 0.5 + 0.5, 0.0, 1.0).astype(np.float32)


@functools.lru_cache(maxsize=32)
def _smoke_texture(
    h: int, w: int, octaves: int, base_cells: int, seed: int, layer: int
) -> np.ndarray:
    # A pure function of its arguments, cached so each frame does not rebuild it.
    rng = np.random.default_rng([seed, SMOKE_STREAM, layer])
    tex = fractal_noise((h, w), octaves, rng, base_cells)
    tex.setflags(write=False)
    return tex


def smoke(
    frame_rgb: np.ndarray, depth: np.ndarray, frame_index: int, cfg: Config, seed: int
) -> np.ndarray:
    """Layers of drifting smoke tucked into the scene by depth. Nearer layers have
    larger wisps and scroll faster.

    Smoke darkens the paper toward gray with a multiply blend, so it shows on
    white areas while black ink stays black and keeps its definition. Style
    "soft" gives soft-edged wisps; "flat" gives hard-edged flat-tone shapes.
    """
    sc: SmokeConfig = cfg.atmosphere.smoke
    out = np.array(frame_rgb, dtype=np.float32, copy=True)
    if sc.opacity <= 0 or sc.layers < 1:
        return out
    if sc.style not in SMOKE_STYLES:
        raise ValueError(f"unknown smoke.style '{sc.style}' (expected one of {SMOKE_STYLES})")
    h, w = depth.shape
    direction = np.asarray(sc.direction, np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    layer_depths = np.linspace(sc.depths[0], sc.depths[1], sc.layers)

    for layer, layer_depth in enumerate(layer_depths):
        cells = max(1, round(sc.base_cells / sc.scale_step**layer))
        tex = _smoke_texture(h, w, sc.octaves, cells, seed, layer)
        travel = sc.speed * sc.speed_step**layer * frame_index
        ox = (travel * direction[0] * w) % w
        oy = (travel * direction[1] * h) % h
        shift = np.float32([[1, 0, -ox], [0, 1, -oy]])  # dst(x) = src(x - offset)
        moved = cv2.warpAffine(
            tex, shift, (w, h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_WRAP,
        )
        if sc.style == "flat":
            wisps = _flat_shape(moved, (sc.density[0] + sc.density[1]) / 2.0)
        else:
            wisps = _smoothstep(sc.density[0], sc.density[1], moved)
        # Only pixels farther than the layer are covered; a soft edge of `softness`.
        cover = _smoothstep(0.0, sc.softness, layer_depth - depth)
        amount = (sc.opacity * wisps * cover)[..., None]
        out = out * (1.0 - amount * np.float32(1.0 - sc.value))  # multiply toward gray
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _flat_shape(field: np.ndarray, level: float) -> np.ndarray:
    """Hard-edged region where `field` exceeds `level`, anti-aliased over one pixel."""
    gx = cv2.Sobel(field, cv2.CV_32F, 1, 0, ksize=3) / np.float32(SOBEL3_GAIN)
    gy = cv2.Sobel(field, cv2.CV_32F, 0, 1, ksize=3) / np.float32(SOBEL3_GAIN)
    slope = np.sqrt(gx * gx + gy * gy) + np.float32(1e-6)
    return np.clip((field - np.float32(level)) / slope + 0.5, 0.0, 1.0)


@dataclass(frozen=True)
class _Particles:
    x: np.ndarray
    y: np.ndarray
    radius: np.ndarray
    fade: np.ndarray


def particles(
    shape: tuple[int, int],
    t: float,
    seed: int,
    stream: int,
    count: int,
    speed: list[float],
    sway: list[float],
    sway_hz: list[float],
    size: list[float],
    life: list[float],
    rise: bool,
) -> _Particles:
    """Closed-form particle state at time `t` seconds.

    Every particle's spawn spot, speed, sway, phase, size, and lifetime come from
    the seed and its id. Age loops over the lifetime, and each new life starts at
    a new spot. Fade is sin(pi * age / life), so particles fade in and out.
    """
    h, w = shape
    rng = np.random.default_rng([seed, stream])
    u = rng.random((count, 3))
    v = rng.uniform(speed[0], speed[1], count)
    amp = rng.uniform(sway[0], sway[1], count)
    hz = rng.uniform(sway_hz[0], sway_hz[1], count)
    phase = rng.uniform(0.0, 2.0 * math.pi, count)
    radius = rng.uniform(size[0], size[1], count)
    lifetime = rng.uniform(life[0], life[1], count)

    clock = t + u[:, 2] * lifetime
    cycle = np.floor(clock / lifetime)
    age = clock - cycle * lifetime
    x0 = np.mod(u[:, 0] + cycle * GOLDEN, 1.0) * w
    y0 = np.mod(u[:, 1] + cycle * GOLDEN * GOLDEN, 1.0) * h
    travel = v * h * age
    y = y0 - travel if rise else y0 + travel
    x = x0 + amp * w * np.sin(2.0 * math.pi * hz * t + phase)
    fade = np.sin(math.pi * age / lifetime)
    return _Particles(x, y, radius, fade)


def _splat(shape: tuple[int, int], p: _Particles, spread: float = 1.0) -> np.ndarray:
    """Soft Gaussian dots of radius `spread` * particle radius, (H, W) float32;
    overlapping dots add up."""
    h, w = shape
    glow = np.zeros((h, w), np.float32)
    for x, y, r, a in zip(p.x, p.y, p.radius * spread, p.fade):
        reach = int(math.ceil(DOT_EXTENT * r))
        x0, x1 = max(0, int(x) - reach), min(w, int(x) + reach + 2)
        y0, y1 = max(0, int(y) - reach), min(h, int(y) + reach + 2)
        if a <= 0 or x0 >= x1 or y0 >= y1:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d2 = (xs - np.float32(x)) ** 2 + (ys - np.float32(y)) ** 2
        glow[y0:y1, x0:x1] += np.float32(a) * np.exp(-d2 / np.float32(2.0 * r * r))
    return glow


def embers(frame_rgb: np.ndarray, frame_index: int, cfg: Config, seed: int) -> np.ndarray:
    """Rising sparks with a short motion streak.

    Each spark has a solid core, normal-blended so it shows on white paper, and
    an additive glow that shows over ink. The core is accent orange; with color
    disabled it is white with a thin ink rim so it still reads on paper.
    """
    ec: EmbersConfig = cfg.atmosphere.embers
    out = np.array(frame_rgb, dtype=np.float32, copy=True)
    if ec.count < 1:
        return out
    shape = out.shape[:2]
    t = frame_index / cfg.output.fps
    core = np.zeros(shape, np.float32)
    halo = np.zeros(shape, np.float32)
    rim = np.zeros(shape, np.float32)
    samples = max(1, ec.trail)
    for k in range(samples):  # streak: the same sparks a moment earlier, fading out
        p = particles(
            shape, t - k * ec.trail_s, seed, EMBERS_STREAM, ec.count,
            ec.speed, ec.sway, ec.sway_hz, ec.size, ec.life, rise=True,
        )
        tail = 1.0 - k / samples
        p = _Particles(p.x, p.y, p.radius * (1.0 - 0.5 * k / samples), p.fade * tail)
        core += _splat(shape, p)
        if k == 0 and cfg.color.enabled:
            halo += _splat(shape, p, spread=ec.glow_size)
        elif not cfg.color.enabled:
            rim += _splat(shape, p, spread=ec.rim_size)
    core = np.clip(core * np.float32(ec.brightness), 0.0, 1.0)[..., None]
    halo = halo[..., None]

    if cfg.color.enabled:
        tint = np.asarray(ec.color, np.float32)
        out = out + np.float32(ec.glow) * halo * tint  # additive glow over ink
    else:
        tint = np.ones(3, np.float32)
        edge = np.clip(rim * np.float32(ec.rim), 0.0, 1.0)[..., None]
        out = out * (1.0 - edge)  # thin ink rim so a white spark reads on paper
    out = out * (1.0 - core) + tint * core  # solid core over paper and ink
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def ash(frame_rgb: np.ndarray, frame_index: int, cfg: Config, seed: int) -> np.ndarray:
    """Slowly falling gray flakes, normal-blended."""
    ac: AshConfig = cfg.atmosphere.ash
    out = np.array(frame_rgb, dtype=np.float32, copy=True)
    if ac.count < 1:
        return out
    shape = out.shape[:2]
    p = particles(
        shape, frame_index / cfg.output.fps, seed, ASH_STREAM, ac.count,
        ac.speed, ac.sway, ac.sway_hz, ac.size, ac.life, rise=False,
    )
    alpha = (np.clip(_splat(shape, p), 0.0, 1.0) * np.float32(ac.opacity))[..., None]
    out = out * (1.0 - alpha) + np.float32(ac.value) * alpha
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def apply_atmosphere(
    frame_rgb: np.ndarray, depth: np.ndarray, frame_index: int, cfg: Config, seed: int
) -> np.ndarray:
    """Smoke, then ash, then embers, each when enabled."""
    out = frame_rgb
    at = cfg.atmosphere
    if at.smoke.enabled:
        out = smoke(out, depth, frame_index, cfg, seed)
    if at.ash.enabled:
        out = ash(out, frame_index, cfg, seed)
    if at.embers.enabled:
        out = embers(out, frame_index, cfg, seed)
    return np.asarray(out, dtype=np.float32)
