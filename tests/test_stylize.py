import numpy as np
import pytest

from inkroto.config import load_config
from inkroto.stylize.lines import xdog
from inkroto.stylize.pipeline import ShotState, stylize_drawing
from inkroto.stylize.tones import apply_tones, compute_thresholds


@pytest.fixture
def cfg():
    return load_config()


def test_tones_only_three_values(photo, cfg):
    tc = cfg.stylize.tones
    lum = photo.mean(axis=2)
    thresholds = compute_thresholds(lum, tc.percentiles)
    tones, mid_mask = apply_tones(lum, thresholds, tc.mid_value, tc.jitter, np.random.default_rng(1))
    assert tones.dtype == np.float32 and tones.shape == lum.shape
    values = set(np.unique(tones).tolist())
    assert values <= {0.0, np.float32(tc.mid_value).item(), 1.0}
    assert len(values) == 3
    assert mid_mask.dtype == bool
    np.testing.assert_array_equal(tones[mid_mask], np.float32(tc.mid_value))


def test_tone_coverage_matches_percentiles():
    lum = np.random.default_rng(0).random((400, 500), dtype=np.float32)
    thresholds = compute_thresholds(lum, [30, 70])
    tones, mid_mask = apply_tones(lum, thresholds, 0.45, 0.0, np.random.default_rng(0))
    black = (tones == 0.0).mean()
    mid = mid_mask.mean()
    white = (tones == 1.0).mean()
    assert black == pytest.approx(0.30, abs=0.01)
    assert mid == pytest.approx(0.40, abs=0.01)
    assert white == pytest.approx(0.30, abs=0.01)


def test_tone_jitter_uses_rng():
    lum = np.random.default_rng(0).random((200, 200), dtype=np.float32)
    t = compute_thresholds(lum, [30, 70])
    a, _ = apply_tones(lum, t, 0.45, 0.05, np.random.default_rng(1))
    b, _ = apply_tones(lum, t, 0.45, 0.05, np.random.default_rng(1))
    c, _ = apply_tones(lum, t, 0.45, 0.05, np.random.default_rng(2))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_xdog_step_edge(cfg):
    lc = cfg.stylize.lines
    edge_x = 50
    lum = np.full((60, 100), 0.6, np.float32)
    lum[:, edge_x:] = 1.0
    out = xdog(lum, lc.sigma, lc.k, lc.p, lc.epsilon, lc.phi)
    assert out.shape == lum.shape and out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0

    reach = int(np.ceil(3 * lc.k * lc.sigma))
    near = out[:, edge_x - reach : edge_x + reach]
    assert near.min() < 0.1  # a dark line along the edge
    far = np.ones(out.shape[1], bool)
    far[edge_x - 2 * reach : edge_x + 2 * reach] = False
    assert np.all(out[:, far] == 1.0)  # white paper away from it
    # the line runs the full height of the edge
    assert np.all(out[:, edge_x - reach : edge_x + reach].min(axis=1) < 0.1)


def test_shot_state_locks_thresholds(photo, fake_depth, fake_matte, cfg):
    state = ShotState()
    assert state.thresholds is None
    stylize_drawing(photo, fake_depth, fake_matte, cfg, state, np.random.default_rng(0))
    locked = state.thresholds
    assert locked is not None and locked[0] < locked[1]

    brighter = np.clip(photo * 0.5 + 0.5, 0.0, 1.0)
    out, lines = stylize_drawing(brighter, fake_depth, fake_matte, cfg, state, np.random.default_rng(1))
    assert state.thresholds == locked
    for arr in (out, lines):
        assert arr.shape == fake_depth.shape and arr.dtype == np.float32
        assert arr.min() >= 0.0 and arr.max() <= 1.0
    assert np.all(out <= lines + 1e-6)  # lines are always on top of the drawing


def test_stylize_is_deterministic(photo, fake_depth, fake_matte, cfg):
    run = lambda: stylize_drawing(
        photo, fake_depth, fake_matte, cfg, ShotState(), np.random.default_rng(7)
    )
    (a, la), (b, lb) = run(), run()
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(la, lb)
