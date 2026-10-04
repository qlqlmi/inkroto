"""Clip backend: a synthetic video written by the test, fake estimators."""

import subprocess

import imageio_ffmpeg
import numpy as np
import pytest
from typer.testing import CliRunner

from inkroto import analyze as analyze_mod
from inkroto.cli import app
from inkroto.config import load_config
from inkroto.io import VideoWriter, read_video
from inkroto.motion import ClipBackend
from inkroto.render import render_frames

SRC_FPS, SRC_W, SRC_H, SRC_FRAMES = 10, 240, 180, 30  # a 3 s source clip


def _write_clip(path, photo):
    """The fixture photo with a bright bar sliding right, one step per frame."""
    import cv2

    base = cv2.resize(photo, (SRC_W, SRC_H), interpolation=cv2.INTER_AREA)
    with VideoWriter(path, SRC_FPS, (SRC_W, SRC_H)) as w:
        for i in range(SRC_FRAMES):
            f = base.copy()
            f[20:40, 5 + i * 7 : 15 + i * 7] = 1.0
            w.write(f)
    return path


@pytest.fixture
def clip_path(tmp_path, photo):
    return _write_clip(tmp_path / "clip.mp4", photo)


def _cfg(clip, *overrides):
    return load_config(overrides=[
        "motion.backend=clip", f"motion.clip_path={clip}", "output.fps=12", "output.width=160",
        *overrides,
    ])


def test_frame_count_and_size(clip_path, tmp_path, fake_estimators):
    cfg = _cfg(clip_path, "output.duration_s=2.0")
    shot = ClipBackend(cache_dir=tmp_path / "cache").build(None, None, cfg)
    assert len(shot) == 24                      # trimmed: 2 s at 12 fps
    assert (shot.width, shot.height) == (160, 120)
    f = shot.frame(5)
    assert f.shape == (120, 160, 3) and f.dtype == np.float32
    assert 0.0 <= f.min() and f.max() <= 1.0


def test_short_clip_is_not_padded(clip_path, tmp_path, fake_estimators):
    cfg = _cfg(clip_path, "output.duration_s=10.0")
    shot = ClipBackend(cache_dir=tmp_path / "cache").build(None, None, cfg)
    assert len(shot) == 36                      # the whole 3 s source at 12 fps


def test_even_dimensions(clip_path, tmp_path, fake_estimators):
    cfg = _cfg(clip_path, "output.width=101", "output.duration_s=0.5")
    shot = ClipBackend(cache_dir=tmp_path / "cache").build(None, None, cfg)
    assert shot.width % 2 == 0 and shot.height % 2 == 0


def test_nearest_frame_resampling(clip_path, tmp_path, fake_estimators):
    cfg = _cfg(clip_path, "output.duration_s=2.0", "output.width=240", "output.fps=24")
    shot = ClipBackend(cache_dir=tmp_path / "cache").build(None, None, cfg)
    for k in range(len(shot)):
        bar_x = int(np.argmax(shot.frame(k)[30, :, 1] > 0.95))
        source = int(np.floor(k * SRC_FPS / 24 + 0.5))
        assert abs(bar_x - (5 + source * 7)) <= 1


def test_estimators_run_once_per_drawing(clip_path, tmp_path, fake_estimators):
    cfg = _cfg(clip_path, "output.duration_s=2.0", "cadence.hold=3")
    frames = list(render_frames(None, None, cfg, cache_dir=tmp_path / "cache"))  # clip ignores photo/analysis
    assert len(frames) == 24
    drawings = 8                                 # 24 frames on threes
    assert fake_estimators == {"depth": drawings, "matte": drawings}


def test_analysis_comes_from_middle_frames(clip_path, tmp_path, fake_estimators, monkeypatch):
    seen = []
    real = analyze_mod.analyze_raw

    def spy(image, key, cache_dir=".cache", faces=True, face_min_score=0.75):
        seen.append(key.rsplit("frame", 1)[-1])
        return real(image, key, cache_dir, faces, face_min_score)

    monkeypatch.setattr(analyze_mod, "analyze_raw", spy)
    cfg = _cfg(clip_path, "output.duration_s=1.0", "cadence.hold=4")
    shot = ClipBackend(cache_dir=tmp_path / "cache").build(None, None, cfg)
    for i in range(len(shot)):  # ask for every frame: still one analysis per drawing
        shot.analysis_at(i)
    assert seen == ["2", "6", "10"]


def test_second_render_reads_the_cache(clip_path, tmp_path, fake_estimators):
    cfg = _cfg(clip_path, "output.duration_s=1.0")
    cache = tmp_path / "cache"
    shot = ClipBackend(cache_dir=cache).build(None, None, cfg)
    first = [shot.analysis_at(i)[0] for i in range(len(shot))]
    calls_after_first = dict(fake_estimators)
    assert calls_after_first["depth"] == 6
    assert len(list(cache.glob("frame-*.npz"))) == 6

    again = ClipBackend(cache_dir=cache).build(None, None, cfg)
    second = [again.analysis_at(i)[0] for i in range(len(again))]
    assert fake_estimators == calls_after_first   # no new estimator calls
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_depth_normalized_with_first_drawing_range(clip_path, tmp_path, fake_estimators, monkeypatch):
    # The raw scale doubles every drawing; with per-frame normalization depth
    # would look identical, with a shared range later drawings saturate.
    calls = {"n": 0}

    def drifting_raw(image):
        calls["n"] += 1
        ramp = np.tile(np.linspace(0, 1, image.shape[1], dtype=np.float32), (image.shape[0], 1))
        return ramp * (2.0 ** (calls["n"] - 1))

    monkeypatch.setattr(analyze_mod, "estimate_depth_raw", drifting_raw)
    cfg = _cfg(clip_path, "output.duration_s=1.0", "motion.depth_smoothing=1.0")
    shot = ClipBackend(cache_dir=None).build(None, None, cfg)
    d0 = shot.analysis_at(0)[0]
    d1 = shot.analysis_at(2)[0]
    assert d0.dtype == np.float32 and 0.0 <= d0.min() and d1.max() <= 1.0
    assert d1.mean() > d0.mean() + 0.1


def test_depth_smoothing_follows_changes_gradually(clip_path, tmp_path, fake_estimators, monkeypatch):
    calls = {"n": 0}

    def step_raw(image):
        calls["n"] += 1
        ramp = np.tile(np.linspace(0, 1, image.shape[1], dtype=np.float32), (image.shape[0], 1))
        return ramp if calls["n"] == 1 else ramp + 0.5  # a sudden jump after drawing 0

    monkeypatch.setattr(analyze_mod, "estimate_depth_raw", step_raw)
    smooth = ClipBackend(cache_dir=None).build(
        None, None, _cfg(clip_path, "output.duration_s=1.0", "motion.depth_smoothing=0.5"))
    calls["n"] = 0
    raw = ClipBackend(cache_dir=None).build(
        None, None, _cfg(clip_path, "output.duration_s=1.0", "motion.depth_smoothing=1.0"))
    calls["n"] = 0
    s = [smooth.analysis_at(i)[0].mean() for i in (0, 2, 4)]
    calls["n"] = 0
    r = [raw.analysis_at(i)[0].mean() for i in (0, 2, 4)]
    assert s[0] == pytest.approx(r[0])
    assert s[0] < s[1] < r[1]          # eases toward the new depth
    assert s[1] < s[2]


def test_missing_clip_path(tmp_path):
    cfg = load_config(overrides=["motion.backend=clip"])
    with pytest.raises(ValueError, match="clip_path"):
        ClipBackend(cache_dir=None).build(None, None, cfg)


@pytest.mark.parametrize("suffix", [".mp4", ".gif"])
def test_render_video_input_end_to_end(tmp_path, photo, fake_estimators, suffix):
    clip = _write_clip(tmp_path / "src.mp4", photo)
    if suffix == ".gif":
        gif = tmp_path / "src.gif"
        subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-i", str(clip), str(gif)],
            check=True,
        )
        clip = gif
    result = CliRunner().invoke(app, [
        "render", str(clip), "-o", str(tmp_path / "out.mp4"), "--cache-dir", str(tmp_path / "c"),
        "--set", "output.duration_s=1.0", "--set", "output.fps=12", "--set", "output.width=160",
    ])
    assert result.exit_code == 0, result.output
    frames, fps = read_video(tmp_path / "out_001.mp4")
    assert len(frames) == 12 and fps == 12
    assert frames[0].shape == (120, 160, 3)
    assert fake_estimators["depth"] == 6        # once per drawing, on twos
    saved = load_config(tmp_path / "out_001.config.yaml")
    assert saved.motion.backend == "clip" and saved.motion.clip_path == str(clip)

    # Second render: everything comes from the cache.
    result = CliRunner().invoke(app, [
        "render", str(clip), "-o", str(tmp_path / "out.mp4"), "--cache-dir", str(tmp_path / "c"),
        "--set", "output.duration_s=1.0", "--set", "output.fps=12", "--set", "output.width=160",
    ])
    assert result.exit_code == 0, result.output
    assert fake_estimators["depth"] == 6
