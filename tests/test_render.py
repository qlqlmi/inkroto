from typer.testing import CliRunner

from inkroto.cli import app
from inkroto.io import read_video


def test_cli_render(tmp_path, photo_path, fake_estimators):
    out = tmp_path / "out.mp4"
    result = CliRunner().invoke(
        app,
        [
            "render", str(photo_path), "-o", str(out), "--cache-dir", str(tmp_path / "cache"),
            "--set", "output.width=160", "--set", "output.duration_s=0.5",
            "--set", "atmosphere.embers.enabled=false",
        ],
    )
    assert result.exit_code == 0, result.output
    assert not out.exists()
    frames, fps = read_video(tmp_path / "out_001.mp4")
    assert len(frames) == 12
    assert frames[0].shape == (120, 160, 3)
    # Away from color accents the video is gray: channels match up to codec noise.
    sky = frames[0][:30]
    assert abs(sky[..., 0] - sky[..., 1]).max() < 0.05


def test_cli_unknown_key(tmp_path, photo_path):
    result = CliRunner().invoke(
        app, ["render", str(photo_path), "-o", str(tmp_path / "x.mp4"), "--set", "nope=1"]
    )
    assert result.exit_code != 0
    assert "unknown config key" in result.output


def test_cli_render_numbers_outputs(tmp_path, photo_path, fake_estimators):
    args = [
        "render", str(photo_path), "-o", str(tmp_path / "clip.mp4"),
        "--cache-dir", str(tmp_path / "cache"),
        "--set", "output.width=96", "--set", "output.duration_s=0.1",
    ]
    for _ in range(2):
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, result.output
    assert sorted(p.name for p in tmp_path.glob("clip*.mp4")) == ["clip_001.mp4", "clip_002.mp4"]
    assert "clip_002.mp4" in result.output
