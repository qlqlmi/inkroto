import numpy as np
import pytest

from inkroto.composite.atmosphere import (
    apply_atmosphere, ash, embers, fractal_noise, particles, smoke, EMBERS_STREAM,
)
from inkroto.config import load_config

SEED = 1234


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    return rng.random((120, 160, 3), dtype=np.float32)


@pytest.fixture
def depth():
    # Far (0) on the left, near (1) on the right.
    return np.tile(np.linspace(0.0, 1.0, 160, dtype=np.float32), (120, 1))


def _cfg(*overrides):
    return load_config(overrides=["atmosphere.ash.enabled=true", *overrides])


def test_fractal_noise_range_and_tiling():
    n = fractal_noise((96, 128), 4, np.random.default_rng(0), 3)
    assert n.shape == (96, 128) and n.dtype == np.float32
    assert n.min() >= 0.0 and n.max() <= 1.0 and n.std() > 0.05
    step = np.abs(np.diff(n, axis=1)).mean()
    assert np.abs(n[:, 0] - n[:, -1]).mean() < 3 * step  # wraps left to right
    assert np.abs(n[0] - n[-1]).mean() < 3 * np.abs(np.diff(n, axis=0)).mean()
    np.testing.assert_array_equal(n, fractal_noise((96, 128), 4, np.random.default_rng(0), 3))


@pytest.mark.parametrize("effect", ["smoke", "ash", "embers", "all"])
def test_same_frame_and_seed_identical(frame, depth, effect):
    cfg = _cfg()
    run = {
        "smoke": lambda i, s: smoke(frame, depth, i, cfg, s),
        "ash": lambda i, s: ash(frame, i, cfg, s),
        "embers": lambda i, s: embers(frame, i, cfg, s),
        "all": lambda i, s: apply_atmosphere(frame, depth, i, cfg, s),
    }[effect]
    np.testing.assert_array_equal(run(17, SEED), run(17, SEED))
    assert not np.array_equal(run(17, SEED), run(18, SEED))
    assert not np.array_equal(run(17, SEED), run(17, SEED + 1))


def test_frames_render_independently(frame, depth):
    cfg = _cfg()
    alone = apply_atmosphere(frame, depth, 40, cfg, SEED)
    for i in range(40):  # rendering earlier frames first changes nothing
        apply_atmosphere(frame, depth, i, cfg, SEED)
    np.testing.assert_array_equal(alone, apply_atmosphere(frame, depth, 40, cfg, SEED))


def test_smoke_opacity_zero_is_identity(frame, depth):
    out = smoke(frame, depth, 5, _cfg("atmosphere.smoke.opacity=0"), SEED)
    np.testing.assert_array_equal(out, frame)


def test_smoke_spares_pixels_nearer_than_every_layer(frame, depth):
    cfg = _cfg("atmosphere.smoke.opacity=1.0")
    nearest = cfg.atmosphere.smoke.depths[1]
    near = depth >= nearest
    changed = np.zeros(depth.shape, bool)
    for i in range(0, 60, 7):
        out = smoke(frame, depth, i, cfg, SEED)
        assert np.array_equal(out[near], frame[near])
        changed |= np.any(out != frame, axis=2)
    assert changed[depth < 0.5].mean() > 0.3  # far pixels do get smoke


def test_smoke_drifts_smoothly(depth):
    cfg = _cfg("atmosphere.smoke.opacity=1.0")
    base = np.ones((120, 160, 3), np.float32)
    a, b = smoke(base, depth, 10, cfg, SEED), smoke(base, depth, 11, cfg, SEED)
    assert 0 < np.abs(a - b).mean() < 0.1


def test_embers_count_zero_is_identity(frame):
    out = embers(frame, 9, _cfg("atmosphere.embers.count=0"), SEED)
    np.testing.assert_array_equal(out, frame)
    out = ash(frame, 9, _cfg("atmosphere.ash.count=0"), SEED)
    np.testing.assert_array_equal(out, frame)


def test_ember_color_follows_color_switch():
    black = np.zeros((120, 160, 3), np.float32)
    lit = embers(black, 30, _cfg("atmosphere.embers.count=300"), SEED)
    assert lit[..., 0].sum() > 1.5 * lit[..., 2].sum()  # orange
    white = embers(black, 30, _cfg("atmosphere.embers.count=300", "color.enabled=false"), SEED)
    np.testing.assert_allclose(white[..., 0], white[..., 2])


def test_particles_rise_and_loop():
    kw = dict(seed=SEED, stream=EMBERS_STREAM, count=50, speed=[0.1, 0.2], sway=[0.0, 0.0],
              sway_hz=[0.5, 0.5], size=[1.0, 1.0], life=[2.0, 3.0])
    a = particles((100, 100), 0.50, rise=True, **kw)
    b = particles((100, 100), 0.55, rise=True, **kw)
    assert (b.y < a.y).mean() > 0.8  # all rise, except the few that just respawned
    assert np.all((a.fade >= 0) & (a.fade <= 1))
    later = particles((100, 100), 3.0 * 7, rise=True, **kw)  # many lifetimes later
    assert np.all(np.isfinite(later.x)) and np.all((later.fade >= 0) & (later.fade <= 1))


@pytest.mark.parametrize("i", [0, 13, 95])
def test_outputs_stay_in_range(depth, i):
    for base in (np.zeros((120, 160, 3), np.float32), np.ones((120, 160, 3), np.float32)):
        out = apply_atmosphere(base, depth, i, _cfg("atmosphere.smoke.opacity=1.0",
                                                     "atmosphere.embers.count=500"), SEED)
        assert out.dtype == np.float32
        assert out.min() >= 0.0 and out.max() <= 1.0


def test_embers_show_on_white_paper():
    paper = np.ones((120, 160, 3), np.float32)
    col = embers(paper, 30, _cfg("atmosphere.embers.count=300"), SEED)
    orange = (col[..., 0] > 0.9) & (col[..., 2] < 0.5)
    assert orange.sum() > 50
    bw = embers(paper, 30, _cfg("atmosphere.embers.count=300", "color.enabled=false"), SEED)
    assert (bw.max(axis=2) < 0.5).sum() > 20  # ink rims
    np.testing.assert_allclose(bw[..., 0], bw[..., 2])


def test_ember_streak_trails_below_the_spark():
    spark = ["atmosphere.embers.count=1", "atmosphere.embers.sway=[0,0]", "atmosphere.embers.glow=0",
             "atmosphere.embers.speed=[0.5,0.5]", "atmosphere.embers.life=[10,10]"]
    cfg = _cfg(*spark)
    one = _cfg(*spark, "atmosphere.embers.trail=1")
    black = np.zeros((200, 200, 3), np.float32)
    streak = embers(black, 30, cfg, SEED)[..., 0] > 0.05
    dot = embers(black, 30, one, SEED)[..., 0] > 0.05
    assert streak.sum() > dot.sum()
    ys_streak, ys_dot = np.nonzero(streak)[0], np.nonzero(dot)[0]
    assert ys_streak.max() > ys_dot.max()  # the tail hangs below a rising spark



@pytest.mark.parametrize("style", ["soft", "flat"])
def test_smoke_shows_on_paper_and_keeps_ink_black(depth, style):
    cfg = _cfg(f"atmosphere.smoke.style={style}")
    paper = np.ones((120, 160, 3), np.float32)
    paper[:, ::10] = 0.0  # vertical ink lines
    out = smoke(paper, depth, 0, cfg, SEED)
    ink = paper == 0.0
    np.testing.assert_array_equal(out[ink], 0.0)        # ink stays black
    assert np.all(out <= paper)                          # smoke only darkens paper
    far_paper = (~ink[..., 0]) & (depth < cfg.atmosphere.smoke.depths[0])
    assert (out[..., 0][far_paper] < 0.97).mean() > 0.2  # visible on white


def test_flat_smoke_has_hard_edges(depth):
    paper = np.ones((120, 160, 3), np.float32)
    flat = smoke(paper, np.zeros_like(depth), 0, _cfg("atmosphere.smoke.style=flat"), SEED)[..., 0]
    soft = smoke(paper, np.zeros_like(depth), 0, _cfg("atmosphere.smoke.style=soft"), SEED)[..., 0]
    levels = lambda a: len(np.unique(np.round(a, 3)))
    assert levels(flat) < levels(soft)


def test_unknown_smoke_style_raises(frame, depth):
    with pytest.raises(ValueError, match="smoke.style"):
        smoke(frame, depth, 0, _cfg("atmosphere.smoke.style=fog"), SEED)
