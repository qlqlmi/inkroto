import numpy as np
import pytest

from conftest import RECT
from inkroto.analyze import Analysis
from inkroto.config import load_config
from inkroto.motion import ParallaxBackend, Shot, get_backend
from inkroto.motion.parallax import background_plate


@pytest.fixture
def analysis(fake_depth, fake_matte):
    return Analysis(depth=fake_depth, matte=fake_matte)


def _cfg(*overrides):
    return load_config(overrides=["output.fps=12", "output.duration_s=1.0", *overrides])


def _red_centroid_x(frame):
    red = (frame[..., 0] > 0.6) & (frame[..., 1] < 0.3) & (frame[..., 2] < 0.3)
    assert red.any()
    return float(np.nonzero(red)[1].mean())


def _matte_centroid_x(matte):
    xs = np.arange(matte.shape[1], dtype=np.float64)
    return float((matte.sum(axis=0) * xs).sum() / matte.sum())


@pytest.mark.parametrize("duration,fps", [(1.0, 12), (0.5, 24), (2.0, 10)])
def test_frame_count(photo, analysis, duration, fps):
    cfg = load_config(overrides=[f"output.fps={fps}", f"output.duration_s={duration}"])
    shot = ParallaxBackend().build(photo, analysis, cfg)
    assert len(shot) == round(duration * fps)
    assert shot.frames.shape == (len(shot), 240, 320, 3)


def test_static_camera_is_identity(photo, analysis):
    cfg = _cfg("motion.pan=[0, 0]", "motion.zoom=1.0")
    shot = ParallaxBackend().build(photo, analysis, cfg)
    for frame in shot.frames:
        assert frame.shape == photo.shape and frame.dtype == np.float32
        assert np.abs(frame - photo).max() < 1e-3


def test_foreground_moves_more_than_background(photo, analysis):
    # Fake depth: rectangle 0.95 (near); the red patch sits on the ground near 0.5,
    # which is the focus depth, so it should barely move.
    cfg = _cfg("motion.pan=[0.1, 0]", "motion.zoom=1.0", "motion.focus=0.5")
    shot = ParallaxBackend().build(photo, analysis, cfg)
    first_depth, first_matte = shot.analysis_at(0)
    last_depth, last_matte = shot.analysis_at(len(shot) - 1)
    fg_move = abs(_matte_centroid_x(last_matte) - _matte_centroid_x(first_matte))
    bg_move = abs(_red_centroid_x(shot.frame(-1)) - _red_centroid_x(shot.frame(0)))
    assert fg_move > 10
    assert fg_move > 3 * bg_move

    # The rectangle's pixels follow its matte.
    x0, x1, y0, y1 = RECT
    row = (y0 + y1) // 2
    bright = np.nonzero(shot.frame(-1)[row, :, 0] > 0.85)[0]
    assert abs(bright.mean() - _matte_centroid_x(last_matte)) < 2


def test_warped_analysis_shape_dtype_range(photo, analysis):
    cfg = _cfg()  # default pan and zoom
    shot = ParallaxBackend().build(photo, analysis, cfg)
    for i in (0, len(shot) // 2, len(shot) - 1):
        depth, matte = shot.analysis_at(i)
        for arr in (depth, matte):
            assert arr.shape == (240, 320)
            assert arr.dtype == np.float32
            assert arr.min() >= 0.0 and arr.max() <= 1.0
        frame = shot.frame(i)
        assert frame.dtype == np.float32 and 0.0 <= frame.min() and frame.max() <= 1.0


def test_zoom_pushes_in(photo, analysis):
    cfg = _cfg("motion.pan=[0, 0]", "motion.zoom=1.2")
    shot = ParallaxBackend().build(photo, analysis, cfg)
    width = lambda m: (m.sum(axis=0) > 0.5 * m.max(axis=0).max()).sum()
    assert width(shot.analysis_at(-1)[1]) > width(shot.analysis_at(0)[1])


def test_background_plate_removes_subject(photo, fake_depth, fake_matte):
    cfg = load_config()
    plate, plate_depth = background_plate(photo, fake_depth, fake_matte, cfg.motion)
    x0, x1, y0, y1 = RECT
    inside = plate[y0 + 10 : y1 - 10, x0 + 10 : x1 - 10]
    assert inside.mean() < 0.5  # bright rectangle replaced by dark ground / sky
    assert plate_depth[y0 + 10 : y1 - 10, x0 + 10 : x1 - 10].mean() < 0.8
    far = np.ones_like(fake_matte, bool)
    far[y0 - 20 : y1 + 20, x0 - 20 : x1 + 20] = False
    np.testing.assert_array_equal(plate[far], photo[far])


def test_shot_from_arrays_and_backend_registry():
    frames = np.zeros((3, 4, 6, 3), np.float32)
    maps = np.zeros((3, 4, 6), np.float32)
    shot = Shot.from_arrays(frames, maps, maps)
    assert len(shot) == 3 and np.asarray(shot.frames).shape == (3, 4, 6, 3)
    assert isinstance(get_backend("parallax"), ParallaxBackend)
    with pytest.raises(ValueError):
        get_backend("nope")
