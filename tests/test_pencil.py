from pathlib import Path

import numpy as np
import pytest

from inkroto.analyze import Analysis
from inkroto.config import load_config
from inkroto.render import render_frames
from inkroto.stylize.lines import denoise
from inkroto.stylize.pencil import graphite_lines, paper_texture
from inkroto.stylize.pipeline import ShotState, stylize_drawing

PRESET = Path(__file__).resolve().parents[1] / "configs" / "pencil.yaml"
NO_ATMOSPHERE_OR_FILM = [
    "atmosphere.smoke.enabled=false", "atmosphere.embers.enabled=false",
    "film.grain=0", "film.vignette=0", "film.flicker=0", "film.weave_px=0",
]


def _pencil(*overrides):
    return load_config(overrides=["stylize.medium=pencil", *overrides])


def test_preset_loads():
    cfg = load_config(PRESET)
    assert cfg.stylize.medium == "pencil"
    assert cfg.color.enabled is False


def test_pencil_has_no_pure_black(photo, fake_depth, fake_matte):
    cfg = _pencil()
    pc = cfg.stylize.pencil
    drawing, lines = stylize_drawing(
        photo, fake_depth, fake_matte, cfg, ShotState(), np.random.default_rng(0)
    )
    assert drawing.dtype == np.float32 and drawing.max() <= 1.0
    assert drawing.min() >= pc.line_value * (1.0 - pc.paper_grain) - 1e-6
    assert (drawing < 0.6).mean() > 0.01  # but there is graphite on the page
    ink, _ = stylize_drawing(
        photo, fake_depth, fake_matte, load_config(overrides=["stylize.medium=ink"]),
        ShotState(), np.random.default_rng(0),
    )
    assert ink.min() == 0.0  # the ink look still has solid black


def test_paper_holds_still_while_drawings_change(photo, fake_depth, fake_matte):
    cfg = _pencil()
    state = ShotState()
    a, _ = stylize_drawing(photo, fake_depth, fake_matte, cfg, state, np.random.default_rng(1))
    paper = state.paper.copy()
    b, _ = stylize_drawing(photo, fake_depth, fake_matte, cfg, state, np.random.default_rng(2))
    np.testing.assert_array_equal(state.paper, paper)
    np.testing.assert_array_equal(paper, paper_texture(paper.shape, cfg.stylize.pencil, cfg.seed))
    assert not np.array_equal(a, b)  # the drawing boils


def test_plain_graphite_lines_are_gray_ink_lines():
    cfg = _pencil(
        "stylize.pencil.passes=1", "stylize.pencil.pressure=0", "stylize.pencil.break_rate=0",
        "stylize.pencil.paper_tooth=0",
    )
    pc = cfg.stylize.pencil
    rng = np.random.default_rng(0)
    lines = rng.random((40, 60), dtype=np.float32)
    paper = paper_texture(lines.shape, pc, 1)
    out = graphite_lines(lines, paper, pc, np.random.default_rng(0), 32)
    np.testing.assert_allclose(out, 1.0 - (1.0 - lines) * (1.0 - pc.line_value), atol=1e-6)


def test_sketch_passes_and_breaks_change_lines():
    pc = _pencil().stylize.pencil
    lines = np.ones((600, 200), np.float32)  # a long stroke, so gaps can show
    lines[:, 100] = 0.0
    paper = paper_texture(lines.shape, pc, 1)
    out = graphite_lines(lines, paper, pc, np.random.default_rng(0), 32)
    drawn = (out < 0.9)[:, 92:109]
    assert drawn.any(axis=1).mean() < 0.97      # gaps along the stroke
    assert drawn.any(axis=1).mean() > 0.6       # but mostly drawn
    assert drawn.sum(axis=1).max() > 1          # overlapping passes widen it


def test_pencil_render_is_deterministic_and_warm(photo, fake_depth, fake_matte):
    cfg = load_config(PRESET, ["output.fps=12", "output.duration_s=0.5", *NO_ATMOSPHERE_OR_FILM])
    analysis = Analysis(fake_depth, fake_matte)
    a = list(render_frames(photo, analysis, cfg))
    b = list(render_frames(photo, analysis, cfg))
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    page = a[0][a[0].mean(axis=2) > 0.85]
    assert page[:, 0].mean() > page[:, 2].mean()  # paper tint is warm, not neutral
    np.testing.assert_array_equal(a[0], a[1])      # still held on twos


def test_unknown_medium_raises(photo, fake_depth, fake_matte):
    cfg = load_config(overrides=["stylize.medium=charcoal"])
    with pytest.raises(ValueError, match="stylize.medium"):
        stylize_drawing(photo, fake_depth, fake_matte, cfg, ShotState(), np.random.default_rng(0))


def test_denoise_removes_grain_lines_and_keeps_edges():
    from inkroto.stylize.lines import xdog_edges

    lc = load_config().stylize.lines
    rng = np.random.default_rng(0)
    img = np.full((80, 120), 0.7, np.float32)
    img[:, 60:] = 0.4
    img = np.clip(img + rng.normal(0, 0.012, img.shape), 0, 1).astype(np.float32)

    def ink(src, cols):
        return (xdog_edges(src, lc.sigma, lc.k, lc.p, lc.edge_threshold, lc.phi)[:, cols] < 0.5).mean()

    clean = denoise(img, lc.denoise, lc.denoise_px)
    flat = slice(5, 50)
    assert ink(clean, flat) < 0.5 * ink(img, flat)       # grain no longer drawn
    assert ink(clean, slice(56, 66)) > 0.05              # the real edge still is
    np.testing.assert_array_equal(denoise(img, 0.0, 2.0), img)
