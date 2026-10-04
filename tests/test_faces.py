import numpy as np
import pytest

from inkroto import analyze as analyze_mod
from inkroto.analyze import Analysis, analyze, face_mask
from inkroto.config import load_config
from inkroto.motion import ParallaxBackend, Shot
from inkroto.stylize.pipeline import FACE_MASK_LEVEL, ShotState, stylize_drawing


def _draw(photo, depth, matte, faces, *overrides, seed=0):
    cfg = load_config(overrides=["stylize.medium=ink", *overrides])
    st = {}
    out, lines = stylize_drawing(
        photo, depth, matte, cfg, ShotState(), np.random.default_rng(seed), debug=st, faces=faces
    )
    return out, lines, st, cfg


def test_face_has_no_mid_tone_and_no_hatch(photo, fake_depth, fake_matte, fake_faces):
    _, _, st, cfg = _draw(photo, fake_depth, fake_matte, fake_faces)
    face = fake_faces >= FACE_MASK_LEVEL
    hatched = st["hatch"][face]
    assert np.any((hatched > 0.0) & (hatched < 1.0))  # the area was hatched before
    tones = st["faces applied"]
    mid = np.float32(cfg.stylize.tones.mid_value)
    assert set(np.unique(tones[face]).tolist()) <= {0.0, 1.0}  # two tones only
    assert not np.any(tones[face] == mid)
    # Hatching only ever produces values strictly between 0 and 1 from the
    # pattern's anti-aliased strokes; none of those inside the face.
    assert not np.any((tones[face] > 0.0) & (tones[face] < 1.0))
    # Outside the face, hatching is still there.
    assert np.any((tones[~face] > 0.0) & (tones[~face] < 1.0))


def test_outside_the_face_is_unchanged(photo, fake_depth, fake_matte, fake_faces):
    with_faces, lines_a, _, _ = _draw(photo, fake_depth, fake_matte, fake_faces)
    without, lines_b, _, _ = _draw(photo, fake_depth, fake_matte, None)
    disabled, _, _, _ = _draw(photo, fake_depth, fake_matte, fake_faces, "stylize.faces.enabled=false")
    outside = fake_faces < FACE_MASK_LEVEL
    np.testing.assert_array_equal(with_faces[outside], without[outside])
    np.testing.assert_array_equal(disabled, without)
    np.testing.assert_array_equal(lines_a, lines_b)  # the line layer is untouched
    assert not np.array_equal(with_faces, without)


def test_two_tone_off_keeps_mid_but_still_no_hatch(photo, fake_depth, fake_matte, fake_faces):
    _, _, st, cfg = _draw(photo, fake_depth, fake_matte, fake_faces, "stylize.faces.two_tone=false")
    face = fake_faces >= FACE_MASK_LEVEL
    values = set(np.unique(st["faces applied"][face]).tolist())
    assert values <= {0.0, np.float32(cfg.stylize.tones.mid_value).item(), 1.0}


def test_faces_in_pencil_medium(photo, fake_depth, fake_matte, fake_faces):
    cfg = load_config(overrides=["stylize.medium=pencil"])
    a, _ = stylize_drawing(photo, fake_depth, fake_matte, cfg, ShotState(),
                           np.random.default_rng(0), faces=fake_faces)
    b, _ = stylize_drawing(photo, fake_depth, fake_matte, cfg, ShotState(),
                           np.random.default_rng(0), faces=None)
    assert a.shape == b.shape and a.dtype == np.float32
    assert not np.array_equal(a, b)


def test_face_mask_shape():
    mask = face_mask((100, 120), np.array([[40, 20, 30, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9]]))
    assert mask.dtype == np.float32 and mask.shape == (100, 120)
    assert mask[40, 55] == 1.0            # center
    assert mask[0, 0] == 0.0 and mask[99, 119] == 0.0
    assert 0.0 < mask[40, 55 + 17] < 1.0  # soft edge just outside the box
    assert face_mask((10, 10), np.zeros((0, 15))).max() == 0.0


BOXES = np.array([[60, 70, 50, 56, 0.92], [200, 40, 30, 34, 0.6]], np.float32)  # x, y, w, h, score


def test_analyze_includes_and_caches_faces(tmp_path, photo, fake_estimators, monkeypatch):
    calls = {"n": 0}

    def detect(image, model_dir=".cache"):
        calls["n"] += 1
        return BOXES.copy()

    monkeypatch.setattr(analyze_mod, "detect_face_boxes", detect)
    a = analyze(photo, cache_dir=tmp_path)
    np.testing.assert_array_equal(a.faces, face_mask(photo.shape[:2], BOXES[:1]))  # 0.6 < 0.75
    b = analyze(photo, cache_dir=tmp_path)
    np.testing.assert_array_equal(b.faces, a.faces)
    assert calls["n"] == 1 and fake_estimators["depth"] == 1
    assert analyze(photo, cache_dir=tmp_path, faces=False).faces is None

    # A different cut-off reuses the cached boxes: no new detection.
    low = analyze(photo, cache_dir=tmp_path, face_min_score=0.5)
    np.testing.assert_array_equal(low.faces, face_mask(photo.shape[:2], BOXES))
    assert calls["n"] == 1


def test_old_cache_gets_faces_without_rerunning_models(tmp_path, photo, fake_estimators, monkeypatch):
    analyze(photo, cache_dir=tmp_path, faces=False)        # a cache entry without faces
    monkeypatch.setattr(analyze_mod, "detect_face_boxes", lambda image, model_dir=".cache": BOXES)
    a = analyze(photo, cache_dir=tmp_path)
    assert a.faces.max() == 1.0
    assert fake_estimators == {"depth": 1, "matte": 1}


def test_missing_face_model_warns_and_continues(tmp_path, photo, fake_estimators, monkeypatch):
    def unavailable(image, model_dir=".cache"):
        raise analyze_mod.FaceModelUnavailable("offline")

    monkeypatch.setattr(analyze_mod, "detect_face_boxes", unavailable)
    with pytest.warns(UserWarning, match="without face handling"):
        a = analyze(photo, cache_dir=tmp_path)
    assert a.faces is None


def test_parallax_moves_faces_with_their_layer(photo, fake_depth, fake_matte, fake_faces):
    from conftest import RECT

    x0, x1, y0, y1 = RECT
    subject_face = np.zeros_like(fake_matte)
    subject_face[y0 + 10 : y0 + 50, x0 + 15 : x1 - 15] = 1.0
    cfg = load_config(overrides=["output.fps=12", "output.duration_s=1.0", "motion.pan=[0.1,0]", "motion.zoom=1.0"])
    cx = lambda m: (m.sum(axis=0) * np.arange(m.shape[1])).sum() / m.sum()

    shot = ParallaxBackend().build(photo, Analysis(fake_depth, fake_matte, subject_face), cfg)
    last = len(shot) - 1
    face_move = cx(shot.faces_at(last)) - cx(shot.faces_at(0))
    matte_move = cx(shot.analysis_at(last)[1]) - cx(shot.analysis_at(0)[1])
    assert abs(face_move - matte_move) < 1.0          # a face on the subject moves with it

    shot = ParallaxBackend().build(photo, Analysis(fake_depth, fake_matte, fake_faces), cfg)
    bg_move = cx(shot.faces_at(last)) - cx(shot.faces_at(0))
    # A face in the far background moves with the background: the other way
    # from the near subject, since focus sits between them.
    assert np.sign(bg_move) == -np.sign(matte_move) and abs(bg_move) > 1.0
    f = shot.faces_at(0)
    assert f.dtype == np.float32 and 0.0 <= f.min() and f.max() <= 1.0


def test_shot_without_faces_reports_zeros():
    frames = np.zeros((2, 4, 6, 3), np.float32)
    maps = np.zeros((2, 4, 6), np.float32)
    assert Shot.from_arrays(frames, maps, maps).faces_at(1).max() == 0.0


@pytest.mark.slow
def test_real_face_detector_finds_a_face(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from pathlib import Path
    from inkroto.io import load_image

    factory = Path(__file__).resolve().parents[1] / "photo4.jpg"
    if not factory.is_file():
        pytest.skip("photo4.jpg not present")
    image = load_image(factory, 1280)
    boxes = analyze_mod.detect_face_boxes(image, model_dir=tmp_path)
    assert boxes.shape[1] == 5 and len(boxes) >= 3
    mask = analyze_mod.detect_faces(image, model_dir=tmp_path)
    assert mask.dtype == np.float32 and mask.max() == 1.0 and 0.001 < mask.mean() < 0.2
