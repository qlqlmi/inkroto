"""Full pipeline through the CLI, with the fixture photo and fake analysis."""

import numpy as np
import yaml
from typer.testing import CliRunner

from inkroto.cli import app
from inkroto.config import load_config
from inkroto.io import read_video, save_image
from inkroto.motion import BACKENDS, get_backend

import pytest


def _run(*args):
    return CliRunner().invoke(app, [str(a) for a in args])


def test_render_end_to_end(tmp_path, photo_path, fake_estimators):
    debug = tmp_path / "debug"
    result = _run(
        "render", photo_path, "-o", tmp_path / "clip.mp4", "--cache-dir", tmp_path / "cache",
        "--debug", debug, "--set", "output.duration_s=0.5", "--set", "output.fps=12",
    )
    assert result.exit_code == 0, result.output
    out = tmp_path / "clip_001.mp4"
    assert out.is_file()

    frames, fps = read_video(out)
    assert len(frames) == 6 and fps == 12
    h, w = frames[0].shape[:2]
    assert h % 2 == 0 and w % 2 == 0
    assert not all(np.array_equal(frames[0], f) for f in frames[1:])

    saved = yaml.safe_load((tmp_path / "clip_001.config.yaml").read_text())
    assert saved["output"]["fps"] == 12 and saved["output"]["duration_s"] == 0.5
    assert saved["stylize"]["hatch"]["angle"] == load_config().stylize.hatch.angle

    names = sorted(p.name for p in debug.iterdir())
    assert names == [
        "clip_001_frame_0000.png", "clip_001_frame_0002.png", "clip_001_frame_0005.png",
        "clip_001_stages.png",
    ]


def test_saved_config_round_trips(tmp_path, photo_path, fake_estimators):
    result = _run(
        "render", photo_path, "-o", tmp_path / "a.mp4", "--cache-dir", tmp_path / "cache",
        "--set", "output.duration_s=0.25", "--set", "seed=99", "--set", "stylize.hatch.angle=60",
    )
    assert result.exit_code == 0, result.output
    cfg = load_config(tmp_path / "a_001.config.yaml")
    assert cfg.seed == 99 and cfg.stylize.hatch.angle == 60


def test_batch(tmp_path, photo, fake_estimators):
    src = tmp_path / "photos"
    src.mkdir()
    save_image(src / "alpha.jpg", photo)
    save_image(src / "beta.PNG", photo)
    (src / "beta.yaml").write_text("output:\n  width: 160\n")
    (src / "broken.jpeg").write_bytes(b"not an image")
    (src / "notes.txt").write_text("ignored")
    out = tmp_path / "clips"

    result = _run(
        "batch", src, "-o", out, "--cache-dir", tmp_path / "cache",
        "--set", "output.duration_s=0.25", "--set", "output.fps=12",
    )
    assert result.exit_code == 1  # one image failed
    assert "FAILED broken.jpeg" in result.output
    assert "done: 2 rendered, 1 failed" in result.output
    assert sorted(p.name for p in out.glob("*.mp4")) == ["alpha_001.mp4", "beta_001.mp4"]

    alpha, _ = read_video(out / "alpha_001.mp4")
    beta, _ = read_video(out / "beta_001.mp4")
    assert alpha[0].shape[1] == 1280            # default width
    assert beta[0].shape[1] == 160              # from beta.yaml
    assert len(alpha) == len(beta) == 3         # --set applies to all
    assert (out / "beta_001.config.yaml").is_file()


def test_batch_set_wins_over_sidecar(tmp_path, photo, fake_estimators):
    src = tmp_path / "photos"
    src.mkdir()
    save_image(src / "one.png", photo)
    (src / "one.yaml").write_text("output:\n  width: 160\n")
    result = _run(
        "batch", src, "-o", tmp_path / "out", "--cache-dir", tmp_path / "cache",
        "--set", "output.duration_s=0.1", "--set", "output.width=96",
    )
    assert result.exit_code == 0, result.output
    frames, _ = read_video(tmp_path / "out" / "one_001.mp4")
    assert frames[0].shape[1] == 96


def test_batch_bad_sidecar_is_one_failure(tmp_path, photo, fake_estimators):
    src = tmp_path / "photos"
    src.mkdir()
    save_image(src / "good.png", photo)
    save_image(src / "bad.png", photo)
    (src / "bad.yaml").write_text("output:\n  widht: 160\n")
    result = _run(
        "batch", src, "-o", tmp_path / "out", "--cache-dir", tmp_path / "cache",
        "--set", "output.duration_s=0.1", "--set", "output.width=96",
    )
    assert result.exit_code == 1
    assert "output.widht" in result.output
    assert (tmp_path / "out" / "good_001.mp4").is_file()


def test_batch_empty_folder(tmp_path):
    (tmp_path / "empty").mkdir()
    result = _run("batch", tmp_path / "empty", "-o", tmp_path / "out")
    assert result.exit_code == 1
    assert "no jpg" in result.output


def test_backend_registry():
    assert {"parallax", "clip"} <= set(BACKENDS)
    with pytest.raises(ValueError, match="available: clip, parallax"):
        get_backend("warp")


def test_batch_same_name_different_extension(tmp_path, photo, fake_estimators):
    src = tmp_path / "photos"
    src.mkdir()
    save_image(src / "shot.jpg", photo)
    save_image(src / "shot.png", photo)
    result = _run(
        "batch", src, "-o", tmp_path / "out", "--cache-dir", tmp_path / "cache",
        "--set", "output.duration_s=0.1", "--set", "output.width=96",
    )
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in (tmp_path / "out").glob("*.mp4")) == [
        "shot_jpg_001.mp4", "shot_png_001.mp4",
    ]
