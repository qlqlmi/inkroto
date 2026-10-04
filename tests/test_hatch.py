import numpy as np
import pytest

from inkroto.config import load_config
from inkroto.stylize.hatch import apply_hatch, hatch_pattern
from inkroto.stylize.lines import wobble
from inkroto.stylize.pipeline import ShotState, stylize_drawing


@pytest.fixture
def hc():
    return load_config().stylize.hatch


def _pattern(hc, seed, shape=(120, 160), angle=None):
    return hatch_pattern(
        shape, hc.spacing, hc.angle if angle is None else angle, hc.thickness,
        np.random.default_rng(seed),
        hc.jitter_px, hc.thickness_jitter, hc.noise_scale, hc.segment, hc.gap,
    )


def test_pattern_is_ink_strokes_on_paper(hc):
    p = _pattern(hc, 0)
    assert p.shape == (120, 160) and p.dtype == np.float32
    assert p.min() >= 0.0 and p.max() <= 1.0
    ink = (p < 0.5).mean()
    assert 0.05 < ink < 0.5  # strokes, mostly paper
    assert (p == 1.0).mean() > 0.4


def test_hatch_only_changes_mid_pixels(hc):
    mid = 0.55
    rng = np.random.default_rng(3)
    tones = rng.choice(np.array([0.0, mid, 1.0], np.float32), size=(120, 160))
    mid_mask = tones == np.float32(mid)
    out = apply_hatch(tones, mid_mask, _pattern(hc, 0))
    np.testing.assert_array_equal(out[~mid_mask], tones[~mid_mask])
    assert not np.any(out[mid_mask] == np.float32(mid))  # flat gray is gone


def test_pipeline_hatch_only_touches_mid_tone(photo, fake_depth, fake_matte):
    cfg = load_config(overrides=["stylize.lines.wobble_px=0"])
    st = {}
    stylize_drawing(photo, fake_depth, fake_matte, cfg, ShotState(), np.random.default_rng(0), debug=st)
    before, after = st["cleanup"], st["hatch"]
    mid_mask = before == np.float32(cfg.stylize.tones.mid_value)
    assert mid_mask.any()
    np.testing.assert_array_equal(after[~mid_mask], before[~mid_mask])
    assert not np.array_equal(after[mid_mask], before[mid_mask])


def test_pattern_seeds(hc):
    np.testing.assert_array_equal(_pattern(hc, 1), _pattern(hc, 1))
    assert not np.array_equal(_pattern(hc, 1), _pattern(hc, 2))


def test_pattern_follows_angle(hc):
    # Horizontal strokes vary down a column far more than along a row.
    p = _pattern(hc, 0, angle=0.0, shape=(200, 200)) < 0.5
    row_changes = np.abs(np.diff(p.astype(int), axis=1)).sum()
    col_changes = np.abs(np.diff(p.astype(int), axis=0)).sum()
    assert col_changes > 3 * row_changes


def test_dark_mid_band_is_cross_hatched(hc):
    one = _pattern(hc, 0)
    rng = np.random.default_rng(0)
    from inkroto.stylize.pipeline import _hatch

    lum = np.tile(np.linspace(0.3, 0.6, 160, dtype=np.float32), (120, 1))
    tones = np.full((120, 160), 0.55, np.float32)
    out = _hatch(tones, lum, (0.3, 0.6), 0.55, hc, rng)
    dark_ink = (out[:, :70] < 0.5).mean()
    light_ink = (out[:, 90:] < 0.5).mean()
    assert dark_ink > 1.4 * light_ink
    assert light_ink == pytest.approx((one < 0.5).mean(), abs=0.1)


def _line_positions(img, axis):
    return np.argmin(img, axis=axis).astype(np.float64)


def test_wobble_zero_is_identity():
    img = np.ones((64, 64), np.float32)
    img[:, 30] = 0.0
    out = wobble(img, 0.0, np.random.default_rng(0), 32)
    np.testing.assert_array_equal(out, img)


def test_wobble_displacement_is_small_and_nonzero():
    cfg = load_config().stylize.lines
    h, w, x0, y0 = 200, 240, 120, 100
    vertical = np.ones((h, w), np.float32)
    vertical[:, x0] = 0.0
    horizontal = np.ones((h, w), np.float32)
    horizontal[y0, :] = 0.0

    out_v = wobble(vertical, cfg.wobble_px, np.random.default_rng(5), cfg.wobble_scale)
    out_h = wobble(horizontal, cfg.wobble_px, np.random.default_rng(5), cfg.wobble_scale)
    dx = np.abs(_line_positions(out_v, axis=1) - x0)
    dy = np.abs(_line_positions(out_h, axis=0) - y0)
    for d in (dx, dy):
        assert 0.0 < d.mean() < 3.0
    assert out_v.min() < 0.5  # the line survives
    np.testing.assert_array_equal(
        out_v, wobble(vertical, cfg.wobble_px, np.random.default_rng(5), cfg.wobble_scale)
    )
