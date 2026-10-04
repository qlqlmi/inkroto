"""Synthetic fixtures. No binary files are committed."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from inkroto import analyze as analyze_mod
from inkroto.io import save_image

W, H = 320, 240
HORIZON = 96                      # sky above, ground below
RECT = (150, 230, 60, 180)        # x0, x1, y0, y1 of the bright foreground rectangle
RED = (40, 90, 150, 200)          # x0, x1, y0, y1 of the saturated red patch


@pytest.fixture
def photo() -> np.ndarray:
    """320x240 RGB float32: sky gradient, dark textured ground, bright rect, red patch."""
    rng = np.random.default_rng(0)
    img = np.zeros((H, W, 3), np.float32)

    t = np.linspace(0.0, 1.0, HORIZON, dtype=np.float32)[:, None]
    top = np.array([0.45, 0.6, 0.85], np.float32)
    bottom = np.array([0.85, 0.88, 0.92], np.float32)
    img[:HORIZON] = top + (bottom - top) * t[..., None]

    ground_h = H - HORIZON
    noise = rng.random((ground_h, W), dtype=np.float32)
    stripes = 0.5 + 0.5 * np.sin(np.arange(W, dtype=np.float32) / 3.0)[None, :]
    base = 0.08 + 0.12 * noise + 0.05 * stripes
    img[HORIZON:] = base[..., None] * np.array([1.0, 0.9, 0.8], np.float32)

    x0, x1, y0, y1 = RECT
    img[y0:y1, x0:x1] = 0.92

    x0, x1, y0, y1 = RED
    img[y0:y1, x0:x1] = (0.9, 0.08, 0.05)
    return np.clip(img, 0.0, 1.0)


@pytest.fixture
def fake_depth() -> np.ndarray:
    """Depth matching `photo`: sky far (0), ground ramps nearer downwards, rect near."""
    depth = np.zeros((H, W), np.float32)
    depth[HORIZON:] = np.linspace(0.3, 0.7, H - HORIZON, dtype=np.float32)[:, None]
    x0, x1, y0, y1 = RECT
    depth[y0:y1, x0:x1] = 0.95
    return depth


@pytest.fixture
def fake_matte() -> np.ndarray:
    """Matte matching `photo`: the foreground rectangle only."""
    matte = np.zeros((H, W), np.float32)
    x0, x1, y0, y1 = RECT
    matte[y0:y1, x0:x1] = 1.0
    return matte


@pytest.fixture
def photo_path(tmp_path, photo):
    path = tmp_path / "photo.png"
    save_image(path, photo)
    return path


@pytest.fixture
def fake_estimators(monkeypatch, fake_depth, fake_matte):
    """Replace the ML estimators with the fake depth and matte, resized to the input.

    Returns a dict counting how often each estimator was called.
    """
    calls = {"depth": 0, "matte": 0}

    def resized(arr, image):
        h, w = image.shape[:2]
        if arr.shape == (h, w):
            return arr.copy()
        return cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)

    def depth(image):
        calls["depth"] += 1
        return resized(fake_depth, image)

    def matte(image):
        calls["matte"] += 1
        return resized(fake_matte, image)

    def depth_raw(image):
        # Like the real model: an arbitrary scale and offset, larger = nearer.
        calls["depth"] += 1
        return resized(fake_depth, image) * 7.0 + 3.0

    monkeypatch.setattr(analyze_mod, "estimate_depth", depth)
    monkeypatch.setattr(analyze_mod, "estimate_depth_raw", depth_raw)
    monkeypatch.setattr(analyze_mod, "estimate_matte", matte)
    # Offline: no face model download. No faces unless a test patches this itself.
    monkeypatch.setattr(
        analyze_mod, "detect_face_boxes",
        lambda image, model_dir=".cache": np.zeros((0, 5), np.float32),
    )
    return calls


@pytest.fixture
def fake_faces():
    """A soft elliptical "face" straddling the horizon, over sky and textured
    ground, so it covers mid tones and hatching."""
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xs - 90) / 45) ** 2 + ((ys - HORIZON) / 40) ** 2)
    return np.clip((1.1 - r) / 0.2, 0.0, 1.0).astype(np.float32)
