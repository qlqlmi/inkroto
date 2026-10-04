import numpy as np
import pytest
from typer.testing import CliRunner

from inkroto import analyze as analyze_mod
from inkroto.analyze import Analysis, analyze, normalize_depth
from inkroto.cli import app
from inkroto.io import load_image


def _assert_valid(result, shape):
    assert isinstance(result, Analysis)
    for arr in (result.depth, result.matte):
        assert arr.shape == shape
        assert arr.dtype == np.float32
        assert arr.min() >= 0.0 and arr.max() <= 1.0


def test_analyze_shapes_and_cache(tmp_path, photo, fake_depth, fake_matte, fake_estimators):
    first = analyze(photo, cache_dir=tmp_path)
    _assert_valid(first, photo.shape[:2])
    assert fake_estimators == {"depth": 1, "matte": 1}
    assert len(list(tmp_path.glob("*.npz"))) == 1

    second = analyze(photo, cache_dir=tmp_path)
    assert fake_estimators == {"depth": 1, "matte": 1}  # served from cache
    _assert_valid(second, photo.shape[:2])
    np.testing.assert_array_equal(second.depth, fake_depth)
    np.testing.assert_array_equal(second.matte, fake_matte)


def test_cache_key_depends_on_image(tmp_path, photo, fake_estimators):
    analyze(photo, cache_dir=tmp_path)
    changed = photo.copy()
    changed[0, 0, 0] = 1.0 - changed[0, 0, 0]
    analyze(changed, cache_dir=tmp_path)
    assert fake_estimators["depth"] == 2
    assert len(list(tmp_path.glob("*.npz"))) == 2


def test_corrupt_cache_is_recomputed(tmp_path, photo, fake_estimators):
    analyze(photo, cache_dir=tmp_path)
    (cached,) = tmp_path.glob("*.npz")
    cached.write_bytes(b"not an npz")
    result = analyze(photo, cache_dir=tmp_path)
    _assert_valid(result, photo.shape[:2])
    assert fake_estimators["depth"] == 2


def test_no_cache_dir(tmp_path, photo, fake_estimators):
    analyze(photo, cache_dir=None)
    analyze(photo, cache_dir=None)
    assert fake_estimators["depth"] == 2


def test_normalize_depth_resizes_and_clips():
    raw = np.linspace(0.0, 50.0, 30 * 40, dtype=np.float32).reshape(30, 40)
    depth = normalize_depth(raw, (80, 60))
    assert depth.shape == (60, 80)
    assert depth.dtype == np.float32
    assert depth.min() == 0.0 and depth.max() == 1.0
    sign = -1 if analyze_mod.DEPTH_MODEL_INVERTED else 1
    assert sign * (depth[-1].mean() - depth[0].mean()) > 0


def test_missing_ml_extra_message(monkeypatch):
    def fail(name):
        raise ImportError(name)

    monkeypatch.setattr(analyze_mod.importlib, "import_module", fail)
    with pytest.raises(analyze_mod.MissingMLDependencyError, match=r'pip install -e "\.\[ml\]"'):
        analyze_mod._require("torch")


def test_cli_analyze_sheet(tmp_path, photo_path, fake_estimators, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "sheet.png"
    result = CliRunner().invoke(
        app, ["analyze", str(photo_path), "-o", str(out), "--set", "output.width=320"]
    )
    assert result.exit_code == 0, result.output
    sheet = load_image(out, 960)
    h, w, _ = sheet.shape
    assert w > h * 2  # three panels in one row
    assert (tmp_path / ".cache").is_dir()


@pytest.mark.slow
def test_real_models(tmp_path, photo):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("rembg")
    result = analyze(photo, cache_dir=tmp_path)
    _assert_valid(result, photo.shape[:2])
    assert result.depth.std() > 0.05
    # The ground at the bottom of the frame is nearer than the sky at the top.
    assert result.depth[-40:].mean() > result.depth[:40].mean()
