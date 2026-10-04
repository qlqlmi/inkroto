import pytest

from inkroto.config import Config, ConfigError, load_config


def test_default_config_loads():
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.seed == 1234
    assert cfg.output.fps == 24
    assert cfg.output.width == 1280
    assert cfg.motion.pan == [0.03, 0.0]
    assert cfg.motion.clip_path is None
    assert cfg.stylize.lines.p == 20.0
    assert cfg.color.hue_ranges == [[0, 12], [168, 180], [12, 28]]
    assert cfg.atmosphere.ash.enabled is False


def test_overrides_apply():
    cfg = load_config(
        overrides=[
            "output.fps=12",
            "stylize.hatch.enabled=false",
            "motion.pan=[0.05, 0.01]",
            "motion.clip_path=clips/a.mp4",
            "film.grain=0",
        ]
    )
    assert cfg.output.fps == 12
    assert cfg.stylize.hatch.enabled is False
    assert cfg.motion.pan == [0.05, 0.01]
    assert cfg.motion.clip_path == "clips/a.mp4"
    assert cfg.film.grain == 0.0 and isinstance(cfg.film.grain, float)


def test_user_yaml_merges(tmp_path):
    user = tmp_path / "user.yaml"
    user.write_text("output:\n  fps: 30\nstylize:\n  haze:\n    density: 0.5\n")
    cfg = load_config(user, ["output.fps=8"])
    assert cfg.output.fps == 8  # overrides win over the user file
    assert cfg.stylize.haze.density == 0.5
    assert cfg.output.width == 1280  # untouched defaults survive


def test_unknown_override_key_raises():
    with pytest.raises(ConfigError, match="unknown config key 'output.fpz'"):
        load_config(overrides=["output.fpz=12"])


def test_unknown_user_yaml_key_raises(tmp_path):
    user = tmp_path / "user.yaml"
    user.write_text("stylize:\n  bogus: 1\n")
    with pytest.raises(ConfigError, match="stylize.bogus"):
        load_config(user)


@pytest.mark.parametrize(
    "override",
    ["output.fps=fast", "stylize.hatch.enabled=1", "output.fps", "stylize=3"],
)
def test_bad_overrides_raise(override):
    with pytest.raises(ConfigError):
        load_config(overrides=[override])
