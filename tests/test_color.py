import cv2
import numpy as np
import pytest

from conftest import RED, RECT
from inkroto.analyze import Analysis
from inkroto.composite.color import accent_layer, apply_color, color_mask
from inkroto.config import load_config
from inkroto.render import render_frames


@pytest.fixture
def cc():
    return load_config().color


def _mask(rgb, cc, ranges=None):
    return color_mask(
        rgb, ranges or cc.hue_ranges, cc.min_sat, cc.min_val, cc.morph, cc.min_area, cc.feather
    )


def _hsv_patch(hue_cv, sat=0.9, val=0.8, size=(40, 40)):
    hsv = np.zeros((*size, 3), np.float32)
    hsv[..., 0] = hue_cv * 2.0  # degrees
    hsv[..., 1] = sat
    hsv[..., 2] = val
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def test_gray_source_gives_empty_mask(cc):
    ramp = np.tile(np.linspace(0, 1, 120, dtype=np.float32), (80, 1))
    gray = np.repeat(ramp[..., None], 3, axis=2)
    mask = _mask(gray, cc)
    assert mask.shape == (80, 120) and mask.dtype == np.float32
    assert not mask.any()


def test_hue_ranges_wrapping_zero(cc):
    canvas = np.full((80, 200, 3), 0.5, np.float32)
    canvas[20:60, 20:60] = _hsv_patch(4)     # red just above 0
    canvas[20:60, 80:120] = _hsv_patch(175)  # red just below 180
    canvas[20:60, 140:180] = _hsv_patch(60)  # green: not an accent
    mask = _mask(canvas, cc, [[0, 12], [168, 180]])
    assert mask[40, 40] > 0.9
    assert mask[40, 100] > 0.9
    assert mask[40, 160] == 0.0
    assert mask[5, 5] == 0.0


def test_mask_drops_specks_and_low_saturation(cc):
    canvas = np.full((80, 120, 3), 0.5, np.float32)
    canvas[10:13, 10:13] = _hsv_patch(4, size=(3, 3))          # tiny speck
    canvas[30:70, 60:100] = _hsv_patch(4, sat=0.3)             # dull red
    mask = _mask(canvas, cc)
    assert not mask.any()


def test_accent_layer_is_flat_paint(cc):
    rng = np.random.default_rng(0)
    src = np.clip(_hsv_patch(5) + rng.normal(0, 0.05, (40, 40, 3)), 0, 1).astype(np.float32)
    paint = accent_layer(src, cc.accent_saturation, cc.accent_values, cc.accent_split)
    hsv = cv2.cvtColor(paint, cv2.COLOR_RGB2HSV)
    assert len(np.unique(np.round(hsv[..., 2], 4))) <= 2  # two value levels
    assert np.allclose(hsv[..., 1], cc.accent_saturation, atol=1e-3)
    assert (paint[..., 0] > 2 * paint[..., 1]).all()


def test_red_patch_stays_red(photo, cc):
    ink = np.full(photo.shape[:2], 0.3, np.float32)
    lines = np.ones(photo.shape[:2], np.float32)
    out = apply_color(ink, photo, cc, lines)
    assert out.shape == photo.shape and out.dtype == np.float32
    x0, x1, y0, y1 = RED
    core = out[y0 + 5 : y1 - 5, x0 + 5 : x1 - 5]
    assert (core[..., 0] > 2 * core[..., 1]).all()
    assert (core[..., 0] > 2 * core[..., 2]).all()
    # Pixels away from the patch stay neutral gray.
    gx0, gx1, gy0, gy1 = RECT
    assert np.allclose(out[gy0 + 5, gx0 + 5], out[gy0 + 5, gx0 + 5].mean(), atol=1e-6)


def test_lines_stay_on_top_of_color(photo, cc):
    ink = np.full(photo.shape[:2], 0.3, np.float32)
    lines = np.ones(photo.shape[:2], np.float32)
    x0, x1, y0, y1 = RED
    lines[(y0 + y1) // 2, x0:x1] = 0.0
    out = apply_color(ink, photo, cc, lines)
    assert out[(y0 + y1) // 2, x0 + 5 : x1 - 5].max() == 0.0


def test_color_disabled_is_gray(photo):
    cc = load_config(overrides=["color.enabled=false"]).color
    ink = np.random.default_rng(0).random(photo.shape[:2], dtype=np.float32)
    out = apply_color(ink, photo, cc, np.ones_like(ink))
    for c in range(3):
        np.testing.assert_array_equal(out[..., c], ink)


def test_render_keeps_red_patch_red(photo, fake_depth, fake_matte):
    cfg = load_config(overrides=[
        "output.fps=12", "output.duration_s=0.5", "stylize.medium=ink",
        "atmosphere.smoke.enabled=false", "atmosphere.embers.enabled=false", "atmosphere.ash.enabled=false",
        "film.grain=0", "film.vignette=0", "film.flicker=0", "film.weave_px=0",
    ])
    frames = list(render_frames(photo, Analysis(fake_depth, fake_matte), cfg))
    x0, x1, y0, y1 = RED
    for f in frames:
        core = f[y0 + 8 : y1 - 8, x0 + 8 : x1 - 8]
        assert core[..., 0].mean() > 2 * core[..., 1].mean()
    # Everything outside accents is neutral.
    sky = frames[0][:40]
    assert np.abs(sky[..., 0] - sky[..., 1]).max() < 1e-6
    # Color holds with the drawing.
    np.testing.assert_array_equal(frames[0], frames[1])
