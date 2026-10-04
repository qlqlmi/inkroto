"""Turn one moving frame into one ink drawing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from inkroto.config import Config, FacesConfig, HatchConfig, StylizeConfig
from inkroto.stylize.cleanup import cleanup
from inkroto.stylize.contrast import local_contrast
from inkroto.stylize.hatch import apply_hatch, hatch_pattern
from inkroto.stylize.haze import haze
from inkroto.stylize.lines import denoise, wobble, xdog, xdog_edges
from inkroto.stylize.pencil import graphite_lines, graphite_shading, paper_finish, paper_texture
from inkroto.stylize.relight import relight
from inkroto.stylize.simplify import simplify
from inkroto.stylize.tones import (
    compute_thresholds, jitter_thresholds, luminance, quantize_tones,
)

LINE_SOURCES = ("relit", "hazed", "simplified")
LINE_MODES = ("edges", "level")
MEDIUMS = ("ink", "pencil")
# Face mask values at or above this count as face for tone decisions.
FACE_MASK_LEVEL = 0.5


@dataclass
class ShotState:
    """Values locked on the first drawing of a shot so tones do not drift."""

    thresholds: tuple[float, float] | None = None
    paper: np.ndarray | None = None  # pencil medium: the same page under every drawing


def stylize_drawing(
    rgb: np.ndarray,
    depth: np.ndarray,
    matte: np.ndarray,
    cfg: Config,
    state: ShotState,
    rng: np.random.Generator,
    debug: dict[str, np.ndarray] | None = None,
    faces: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """RGB frame plus its depth and matte to a grayscale drawing (H, W), 0 = ink.

    Returns (drawing, lines): the final drawing and the line layer on its own,
    so color can be painted under the lines later.
    If `debug` is a dict, every intermediate stage is added to it in order.
    `faces` is an optional soft face mask; with stylize.faces enabled, faces get a
    stronger simplify, only two tones, and no hatching, while lines are unchanged
    and everything outside the mask is exactly as without faces.
    """
    scfg = cfg.stylize

    def keep(name: str, img: np.ndarray) -> None:
        if debug is not None:
            debug[name] = img

    keep("input", rgb)
    keep("depth", depth)
    keep("matte", matte)

    lum = luminance(rgb)
    keep("luminance", lum)

    cc = scfg.contrast
    if cc.enabled:
        lum = local_contrast(lum, cc.clip_limit, cc.tiles)
        keep("contrast", lum)

    rc = scfg.relight
    if rc.enabled:
        lum = relight(lum, depth, rc.direction, rc.strength, rc.normal_blur, rc.normal_z)
        keep("relight", lum)
    line_sources = {"relit": lum}

    hc = scfg.haze
    if hc.enabled:
        lum = haze(lum, depth, hc.start, hc.density, hc.value)
        keep("haze", lum)
    line_sources["hazed"] = lum

    sc = scfg.simplify
    detail = lum
    lum = simplify(lum, sc.method, sc.radius, sc.sigma_color)
    keep(f"simplify ({sc.method})", lum)
    line_sources["simplified"] = lum

    if state.thresholds is None:
        state.thresholds = compute_thresholds(lum, scfg.tones.percentiles)
    thresholds = jitter_thresholds(state.thresholds, scfg.tones.jitter, rng)
    tones, _ = quantize_tones(lum, thresholds, scfg.tones.mid_value)
    keep("tones", tones)

    clc = scfg.cleanup
    tones = cleanup(tones, clc.min_area, clc.morph)
    keep("cleanup", tones)
    regions = tones

    if scfg.medium not in MEDIUMS:
        raise ValueError(f"unknown stylize.medium '{scfg.medium}' (expected one of {MEDIUMS})")
    hc = scfg.hatch
    if hc.enabled:
        tones = _hatch(tones, lum, state.thresholds, scfg.tones.mid_value, hc, rng)
        keep("hatch", tones)

    fc = scfg.faces
    if fc.enabled and faces is not None and faces.max() >= FACE_MASK_LEVEL:
        # Swapped in after hatching, so the random draws (and everything outside
        # the faces) are exactly as without faces.
        keep("faces", faces)
        face = faces >= FACE_MASK_LEVEL
        face_tones = _face_tones(detail, lum, faces, thresholds, scfg, fc)
        regions = np.where(face, face_tones, regions)
        tones = np.where(face, face_tones, tones).astype(np.float32)
        keep("faces applied", tones)

    lc = scfg.lines
    if lc.source not in LINE_SOURCES:
        raise ValueError(f"unknown lines.source '{lc.source}' (expected one of {LINE_SOURCES})")
    # Tracing lines before haze and simplify keeps contours that those stages
    # flatten; haze and simplify still shape the tones.
    if lc.mode not in LINE_MODES:
        raise ValueError(f"unknown lines.mode '{lc.mode}' (expected one of {LINE_MODES})")
    line_lum = denoise(line_sources[lc.source], lc.denoise, lc.denoise_px)
    if lc.mode == "edges":
        lines = xdog_edges(line_lum, lc.sigma, lc.k, lc.p, lc.edge_threshold, lc.phi)
    else:
        lines = xdog(line_lum, lc.sigma, lc.k, lc.p, lc.epsilon, lc.phi)
    keep(f"lines (from {lc.source})", lines)
    if lc.wobble_px > 0:
        lines = wobble(lines, lc.wobble_px, rng, lc.wobble_scale)
        keep("lines wobbled", lines)

    if scfg.medium == "pencil":
        pc = scfg.pencil
        if state.paper is None or state.paper.shape != lines.shape:
            state.paper = paper_texture(lines.shape, pc, cfg.seed)
        mid_value = np.float32(scfg.tones.mid_value)
        strokes = tones if hc.enabled else None
        tones = graphite_shading(regions == 0.0, regions == mid_value, strokes, state.paper, pc)
        keep("graphite shading", tones)
        lines = graphite_lines(lines, state.paper, pc, rng, lc.wobble_scale)
        keep("graphite lines", lines)

    drawing = np.minimum(tones, lines).astype(np.float32)
    if scfg.medium == "pencil":
        drawing = paper_finish(drawing, state.paper, scfg.pencil)
    keep("drawing", drawing)
    return drawing, lines


def _hatch(
    tones: np.ndarray,
    lum: np.ndarray,
    thresholds: tuple[float, float],
    mid_value: float,
    hc: HatchConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Hatch the mid tone; its darker half also gets strokes at angle + 90."""
    mid_mask = tones == np.float32(mid_value)
    if not mid_mask.any():
        return tones

    def layer(angle: float) -> np.ndarray:
        return hatch_pattern(
            tones.shape, hc.spacing, angle, hc.thickness, rng,
            hc.jitter_px, hc.thickness_jitter, hc.noise_scale, hc.segment, hc.gap,
        )

    pattern = layer(hc.angle)
    dark_mid = mid_mask & (lum < (thresholds[0] + thresholds[1]) / 2.0)
    if dark_mid.any():
        pattern = np.where(dark_mid, np.minimum(pattern, layer(hc.angle + 90.0)), pattern)
    return apply_hatch(tones, mid_mask, pattern)


def _face_tones(
    detail: np.ndarray,
    lum: np.ndarray,
    faces: np.ndarray,
    thresholds: tuple[float, float],
    scfg: StylizeConfig,
    fc: FacesConfig,
) -> np.ndarray:
    """Tones for face areas: a stronger simplify blended in by the soft mask, the
    same thresholds, and (with two_tone) the mid tone collapsed into white."""
    sc = scfg.simplify
    strong = simplify(detail, sc.method, fc.simplify_radius, sc.sigma_color)
    blended = lum * (1.0 - faces) + strong * faces
    mid_value = scfg.tones.mid_value
    tones, _ = quantize_tones(blended, thresholds, mid_value)
    tones = cleanup(tones, scfg.cleanup.min_area, scfg.cleanup.morph)
    if fc.two_tone:
        tones = np.where(tones == np.float32(mid_value), 1.0, tones).astype(np.float32)
    return tones
