import numpy as np
import pytest
from typer.testing import CliRunner

from inkroto.analyze import Analysis
from inkroto.cadence import drawing_rng, group_frames
from inkroto.cli import app
from inkroto.config import load_config
from inkroto.render import render_frames


@pytest.fixture
def analysis(fake_depth, fake_matte):
    return Analysis(depth=fake_depth, matte=fake_matte)


# Atmosphere and film change every output frame; switch them off to see the held drawing.
NO_ATMOSPHERE = [
    "atmosphere.smoke.enabled=false", "atmosphere.embers.enabled=false", "atmosphere.ash.enabled=false",
    "film.grain=0", "film.vignette=0", "film.flicker=0", "film.weave_px=0",
]


def _frames(photo, analysis, *overrides, atmosphere=False):
    base = ["output.fps=12", "output.duration_s=1.0"] + ([] if atmosphere else NO_ATMOSPHERE)
    cfg = load_config(overrides=base + list(overrides))
    return list(render_frames(photo, analysis, cfg))


def test_group_frames():
    assert group_frames(6, 2) == [(0, 2), (2, 4), (4, 6)]
    assert group_frames(7, 3) == [(0, 3), (3, 6), (6, 7)]
    assert group_frames(3, 1) == [(0, 1), (1, 2), (2, 3)]
    assert group_frames(0, 2) == []
    with pytest.raises(ValueError, match="hold"):
        group_frames(4, 0)


def test_drawing_rng_depends_only_on_seed_and_index():
    a = drawing_rng(1234, 5).random(4)
    _ = [drawing_rng(1234, i).random(100) for i in range(5)]  # earlier drawings
    np.testing.assert_array_equal(a, drawing_rng(1234, 5).random(4))
    assert not np.array_equal(a, drawing_rng(1234, 6).random(4))
    assert not np.array_equal(a, drawing_rng(99, 5).random(4))


def test_hold_two_repeats_each_drawing(photo, analysis):
    frames = _frames(photo, analysis, "cadence.hold=2")
    assert len(frames) == 12
    for k in range(0, 12, 2):
        np.testing.assert_array_equal(frames[k], frames[k + 1])
    for k in range(0, 10, 2):
        assert not np.array_equal(frames[k], frames[k + 2])  # consecutive drawings differ


def test_atmosphere_moves_within_a_hold(photo, analysis):
    held = _frames(photo, analysis, "cadence.hold=2")
    live = _frames(photo, analysis, "cadence.hold=2", atmosphere=True)
    np.testing.assert_array_equal(held[0], held[1])
    assert not np.array_equal(live[0], live[1])  # smoke and embers move every frame
    # Same seed, same frames.
    again = _frames(photo, analysis, "cadence.hold=2", atmosphere=True)
    for a, b in zip(live, again):
        np.testing.assert_array_equal(a, b)


def test_hold_one(photo, analysis):
    frames = _frames(photo, analysis, "cadence.hold=1")
    assert len(frames) == 12
    for a, b in zip(frames, frames[1:]):
        assert not np.array_equal(a, b)


def test_uneven_hold_keeps_frame_count(photo, analysis):
    frames = _frames(photo, analysis, "cadence.hold=5")
    assert len(frames) == 12
    np.testing.assert_array_equal(frames[10], frames[11])  # short last group
    assert not np.array_equal(frames[9], frames[10])


def test_frames_are_valid_drawings(photo, analysis):
    for f in _frames(photo, analysis)[:2]:
        assert f.shape == photo.shape and f.dtype == np.float32
        assert f.min() >= 0.0 and f.max() <= 1.0


def _render(tmp_path, photo_path, name, seed):
    out = tmp_path / f"{name}.mp4"
    result = CliRunner().invoke(app, [
        "render", str(photo_path), "-o", str(out), "--cache-dir", str(tmp_path / "cache"),
        "--set", "output.width=160", "--set", "output.duration_s=0.5", "--set", f"seed={seed}",
    ])
    assert result.exit_code == 0, result.output
    return (tmp_path / f"{name}_001.mp4").read_bytes()


def test_full_render_is_deterministic(tmp_path, photo_path, fake_estimators):
    a = _render(tmp_path, photo_path, "a", 7)
    b = _render(tmp_path, photo_path, "b", 7)
    c = _render(tmp_path, photo_path, "c", 8)
    assert a == b
    assert a != c
