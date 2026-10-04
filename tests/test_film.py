import numpy as np
import pytest

from inkroto.composite.film import (
    finish, flicker_gain, smooth_noise_1d, vignette_gain, weave,
)
from inkroto.config import load_config

SEED = 1234


@pytest.fixture
def frame():
    return np.random.default_rng(0).random((90, 120, 3), dtype=np.float32)


def _cfg(*overrides):
    return load_config(overrides=list(overrides))


ZERO = ["film.grain=0", "film.vignette=0", "film.flicker=0", "film.weave_px=0"]


def test_zero_strengths_are_identity(frame):
    out = finish(frame, 7, _cfg(*ZERO), SEED)
    np.testing.assert_array_equal(out, frame)


@pytest.mark.parametrize("i", [0, 5, 47])
def test_output_in_range(i):
    cfg = _cfg("film.grain=0.5", "film.flicker=0.5", "film.vignette=1.0", "film.weave_px=3")
    for base in (np.zeros((90, 120, 3), np.float32), np.ones((90, 120, 3), np.float32),
                 np.full((90, 120, 3), 0.5, np.float32)):
        out = finish(base, i, cfg, SEED)
        assert out.dtype == np.float32 and out.shape == base.shape
        assert out.min() >= 0.0 and out.max() <= 1.0


def test_deterministic_per_frame_and_seed(frame):
    cfg = _cfg()
    a = finish(frame, 12, cfg, SEED)
    np.testing.assert_array_equal(a, finish(frame, 12, cfg, SEED))
    assert not np.array_equal(a, finish(frame, 13, cfg, SEED))
    assert not np.array_equal(a, finish(frame, 12, cfg, SEED + 1))
    for i in range(12):  # rendering other frames first changes nothing
        finish(frame, i, cfg, SEED)
    np.testing.assert_array_equal(a, finish(frame, 12, cfg, SEED))


def test_flicker_is_smooth_and_bounded():
    fc = _cfg().film
    gains = np.array([flicker_gain(i, fc.flicker, fc.flicker_frames, SEED) for i in range(500)])
    assert np.all(np.abs(np.diff(gains)) < fc.flicker)
    assert np.all(np.abs(gains - 1.0) <= fc.flicker + 1e-9)
    assert gains.std() > fc.flicker / 10  # it does flicker


def test_smooth_noise_hits_knots_and_stays_in_range():
    values = [smooth_noise_1d(i, 4, SEED, 99) for i in range(200)]
    assert min(values) >= -1.0 and max(values) <= 1.0
    assert smooth_noise_1d(8, 4, SEED, 99) == smooth_noise_1d(8, 4, SEED, 99)


def test_grain_is_neutral_and_spares_black_and_white():
    cfg = _cfg("film.vignette=0", "film.flicker=0", "film.weave_px=0", "film.grain=0.1")
    img = np.zeros((60, 90, 3), np.float32)
    img[:, 30:60] = 0.5
    img[:, 60:] = 1.0
    out = finish(img, 3, cfg, SEED)
    np.testing.assert_array_equal(out[..., 0], out[..., 1])  # same on every channel
    np.testing.assert_array_equal(out[..., 1], out[..., 2])
    np.testing.assert_array_equal(out[:, :30], 0.0)
    np.testing.assert_array_equal(out[:, 60:], 1.0)
    assert out[:, 30:60].std() > 0.05


def test_vignette_darkens_corners_only():
    g = vignette_gain((90, 120), 0.35, 2.0)
    assert g[45, 60] > 0.99
    assert g[0, 0] == pytest.approx(0.65, abs=0.01)
    assert g[0, 0] < g[20, 30] < g[45, 60]


def test_weave_is_subpixel_and_changes_per_frame():
    img = np.zeros((60, 80, 3), np.float32)
    img[:, 40] = 1.0
    centers = []
    for i in range(30):
        out = weave(img, i, 0.4, 3, SEED)[..., 0]
        cols = np.arange(80)
        centers.append((out.sum(axis=0) * cols).sum() / out.sum())
    centers = np.array(centers)
    assert np.all(np.abs(centers - 40) <= 0.4 + 1e-3)
    assert len(np.unique(np.round(centers, 4))) > 10
