import numpy as np
import pytest
from typer.testing import CliRunner

from inkroto.cli import app
from inkroto.config import load_config
from inkroto.stylize.cleanup import cleanup
from inkroto.stylize.haze import haze
from inkroto.stylize.relight import relight
from inkroto.stylize.simplify import kuwahara, simplify


@pytest.fixture
def cfg():
    return load_config()


def _dome(size=128):
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    c = (size - 1) / 2
    r2 = ((xs - c) ** 2 + (ys - c) ** 2) / (size * 0.4) ** 2
    return np.sqrt(np.clip(1.0 - r2, 0.0, 1.0)).astype(np.float32)  # 1 = near at center


def test_relight_lights_the_side_facing_the_light(cfg):
    rc = cfg.stylize.relight
    depth = _dome()
    lum = np.full_like(depth, 0.8)
    left_light = [-1.0, 0.0, 0.5]
    out = relight(lum, depth, left_light, rc.strength, rc.normal_blur, rc.normal_z)
    assert out.shape == lum.shape and out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    mid = depth.shape[0] // 2
    left, right = out[mid, 30:50].mean(), out[mid, 78:98].mean()
    assert left > right + 0.1

    # Default light comes from the upper left: the upper-left flank wins.
    out = relight(lum, depth, rc.direction, rc.strength, rc.normal_blur, rc.normal_z)
    assert out[35:50, 35:50].mean() > out[78:93, 78:93].mean()


def test_relight_strength_zero_is_identity(cfg):
    rc = cfg.stylize.relight
    lum = np.random.default_rng(0).random((64, 64), dtype=np.float32)
    out = relight(lum, _dome(64), rc.direction, 0.0, rc.normal_blur, rc.normal_z)
    np.testing.assert_allclose(out, lum, atol=1e-6)


def test_haze_only_touches_far_pixels():
    start, density, value = 0.55, 0.8, 0.75
    lum = np.random.default_rng(0).random((50, 100), dtype=np.float32) * 0.5
    depth = np.tile(np.linspace(0.0, 1.0, 100, dtype=np.float32), (50, 1))
    out = haze(lum, depth, start, density, value)
    near = depth >= start
    np.testing.assert_array_equal(out[near], lum[near])
    far = depth < start
    assert np.all(np.abs(out[far] - value) <= np.abs(lum[far] - value) + 1e-6)
    farthest = depth == 0.0
    np.testing.assert_allclose(
        out[farthest], lum[farthest] + (value - lum[farthest]) * density, atol=1e-6
    )


def test_kuwahara_flattens_noise_and_keeps_edges(cfg):
    radius = cfg.stylize.simplify.radius
    rng = np.random.default_rng(0)
    img = np.full((80, 80), 0.2, np.float32)
    img[:, 40:] = 0.8
    noisy = np.clip(img + rng.normal(0, 0.05, img.shape).astype(np.float32), 0, 1)

    out = kuwahara(noisy, radius)
    assert out.shape == noisy.shape and out.dtype == np.float32
    flat = (slice(10, 70), slice(5, 30))
    assert out[flat].var() < 0.5 * noisy[flat].var()

    contrast = lambda a: a[:, 41:44].mean() - a[:, 36:39].mean()
    assert contrast(out) > 0.8 * contrast(noisy)


@pytest.mark.parametrize("method", ["kuwahara", "meanshift", "bilateral"])
def test_simplify_methods(photo, method):
    lum = photo.mean(axis=2).astype(np.float32)
    out = simplify(lum, method, 5, 0.1)
    assert out.shape == lum.shape and out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_simplify_unknown_method():
    with pytest.raises(ValueError, match="unknown simplify method"):
        simplify(np.zeros((8, 8), np.float32), "blur", 3, 0.1)


def test_cleanup_removes_small_blob_keeps_large(cfg):
    min_area = cfg.stylize.cleanup.min_area
    mid = 0.45
    tones = np.full((80, 80), mid, np.float32)
    tones[10:13, 10:13] = 0.0          # 9 px black speck, below min_area
    tones[40:60, 40:60] = 0.0          # 400 px black region
    tones[20:22, 60:62] = 1.0          # 4 px white speck
    tones[60:75, 5:20] = 1.0           # 225 px white region
    out = cleanup(tones, min_area, cfg.stylize.cleanup.morph)

    assert np.all(out[10:13, 10:13] == mid)
    assert np.all(out[20:22, 60:62] == mid)
    assert np.all(out[42:58, 42:58] == 0.0)
    assert np.all(out[62:73, 7:18] == 1.0)
    assert set(np.unique(out).tolist()) <= {0.0, np.float32(mid).item(), 1.0}


def test_cleanup_speck_takes_surrounding_tone():
    tones = np.zeros((40, 40), np.float32)
    tones[18:20, 18:20] = 1.0  # white speck inside black
    out = cleanup(tones, 24, 0)
    assert np.all(out == 0.0)


def test_cli_debug_sheet(tmp_path, photo_path, fake_estimators):
    out = tmp_path / "out.mp4"
    debug = tmp_path / "debug"
    result = CliRunner().invoke(
        app,
        [
            "render", str(photo_path), "-o", str(out), "--cache-dir", str(tmp_path / "c"),
            "--debug", str(debug), "--set", "output.width=320", "--set", "output.duration_s=0.25",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (debug / "out_001_stages.png").is_file()


def _thin_lines_image():
    img = np.full((120, 160, 3), 0.85, np.float32)
    for x in range(20, 140, 12):
        img[:, x : x + 2] = 0.1  # 2 px ink lines, thinner than the simplify window
    return img


def test_early_line_sources_keep_thin_lines():
    from inkroto.stylize.pipeline import ShotState, stylize_drawing

    img = _thin_lines_image()
    depth = np.full(img.shape[:2], 0.8, np.float32)
    matte = np.zeros(img.shape[:2], np.float32)
    base = ["stylize.simplify.radius=5", "stylize.relight.enabled=false", "stylize.haze.enabled=false",
            "stylize.lines.mode=level", "stylize.lines.epsilon=0.15"]

    def debug_lines(source):
        cfg = load_config(overrides=base + [f"stylize.lines.source={source}"])
        stages = {}
        stylize_drawing(img, depth, matte, cfg, ShotState(), np.random.default_rng(0), debug=stages)
        return stages[f"lines (from {source})"]

    columns = slice(20, 140)
    ink = {s: (debug_lines(s)[:, columns] < 0.5).mean() for s in ("relit", "hazed", "simplified")}
    assert ink["relit"] > 2 * ink["simplified"]
    assert ink["hazed"] > 2 * ink["simplified"]


def test_unknown_line_source_raises(photo, fake_depth, fake_matte):
    from inkroto.stylize.pipeline import ShotState, stylize_drawing

    cfg = load_config(overrides=["stylize.lines.source=edges"])
    with pytest.raises(ValueError, match="lines.source"):
        stylize_drawing(photo, fake_depth, fake_matte, cfg, ShotState(), np.random.default_rng(0))



def test_lines_traced_before_haze_keep_far_contours():
    from inkroto.stylize.pipeline import ShotState, stylize_drawing

    img = np.full((80, 120, 3), 0.3, np.float32)
    img[:, 60:] = 0.5  # an edge in the far distance; heavy haze washes it out
    depth = np.zeros(img.shape[:2], np.float32)
    matte = np.zeros(img.shape[:2], np.float32)

    def edge_ink(source):
        cfg = load_config(overrides=[
            f"stylize.lines.source={source}", "stylize.haze.density=0.9",
            "stylize.relight.enabled=false", "stylize.contrast.enabled=false",
        ])
        stages = {}
        stylize_drawing(img, depth, matte, cfg, ShotState(), np.random.default_rng(0), debug=stages)
        return stages[f"lines (from {source})"][:, 54:66].min()

    assert edge_ink("relit") < 0.2
    assert edge_ink("hazed") > edge_ink("relit")


def test_local_contrast_lifts_murky_detail():
    from inkroto.stylize.contrast import local_contrast

    rng = np.random.default_rng(0)
    lum = np.full((128, 128), 0.3, np.float32)
    lum[:, 64:] = 0.8
    lum[:, :64] += rng.normal(0, 0.01, (128, 64)).astype(np.float32)  # faint texture
    out = local_contrast(lum, 2.0, 8)
    assert out.shape == lum.shape and out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out[:, 8:56].std() > 1.5 * lum[:, 8:56].std()



def _step(lo, hi):
    img = np.full((40, 80), lo, np.float32)
    img[:, 40:] = hi
    return img


@pytest.mark.parametrize("lo,hi", [(0.1, 0.3), (0.4, 0.6), (0.6, 0.8), (0.75, 0.95)])
def test_edge_lines_ink_edges_at_any_brightness(lo, hi):
    from inkroto.stylize.lines import xdog_edges

    lc = load_config().stylize.lines
    out = xdog_edges(_step(lo, hi), lc.sigma, lc.k, lc.p, lc.edge_threshold, lc.phi)
    assert out.dtype == np.float32 and 0.0 <= out.min() and out.max() <= 1.0
    assert out[:, 35:41].min() < 0.2           # a line on the dark side of the edge
    assert np.all(out[:, :30] == 1.0)          # flat areas stay paper, dark or not
    assert np.all(out[:, 50:] == 1.0)


def test_level_lines_miss_bright_edges_that_edge_lines_find():
    from inkroto.stylize.lines import xdog, xdog_edges

    lc = load_config().stylize.lines
    bright = _step(0.6, 0.8)
    level = xdog(bright, lc.sigma, lc.k, lc.p, lc.epsilon, lc.phi)
    edges = xdog_edges(bright, lc.sigma, lc.k, lc.p, lc.edge_threshold, lc.phi)
    assert level.min() > 0.9 and edges.min() < 0.2


def test_edge_threshold_ignores_faint_texture():
    from inkroto.stylize.lines import xdog_edges

    lc = load_config().stylize.lines
    rng = np.random.default_rng(0)
    faint = np.clip(0.5 + rng.normal(0, 0.01, (60, 60)), 0, 1).astype(np.float32)
    out = xdog_edges(faint, lc.sigma, lc.k, lc.p, lc.edge_threshold, lc.phi)
    assert (out < 0.5).mean() < 0.01


def test_unknown_line_mode_raises(photo, fake_depth, fake_matte):
    from inkroto.stylize.pipeline import ShotState, stylize_drawing

    cfg = load_config(overrides=["stylize.lines.mode=sketch"])
    with pytest.raises(ValueError, match="lines.mode"):
        stylize_drawing(photo, fake_depth, fake_matte, cfg, ShotState(), np.random.default_rng(0))
