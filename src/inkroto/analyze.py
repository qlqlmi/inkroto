"""Depth, subject matte, and face estimation, with an on-disk cache.

Heavy ML imports (torch, transformers, rembg) are lazy and live only here.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import os
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from inkroto.io import from_uint8, to_bgr_uint8, to_uint8

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
MATTE_MODEL = "u2net"

# Depth Anything predicts relative inverse depth (disparity), so larger values are
# nearer, which already matches the project convention (1 = near). Set this to True
# if the model output turns out to be far = large.
# The direction MUST be verified by eye with the debug sheet
# (`inkroto analyze photo.jpg -o sheet.png`): near objects must be bright.
DEPTH_MODEL_INVERTED = False

# Robust normalization range for raw depth, in percentiles.
DEPTH_LOW_PERCENTILE = 2.0
DEPTH_HIGH_PERCENTILE = 98.0

# Bump when the cached arrays change meaning, to invalidate old cache files.
CACHE_VERSION = 1

ML_INSTALL_HINT = 'pip install -e ".[ml]"'

# Face detection: OpenCV's YuNet, downloaded to the cache folder on first use.
FACE_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
FACE_MODEL_FILE = "face_detection_yunet_2023mar.onnx"
FACE_MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
FACE_DOWNLOAD_TIMEOUT_S = 30
# Detection runs at these scales (half size finds close-ups, full size small
# faces) and keeps every box above a low floor with its score; the score cut-off
# that decides what counts as a face is the config's stylize.faces.min_score.
FACE_SCALES = (0.5, 1.0)
FACE_SCORE_FLOOR = 0.3
FACE_NMS_THRESHOLD = 0.3
FACE_TOP_K = 50
# The mask is an ellipse inside the detected box, grown by this factor, with a
# soft edge this fraction of the face size wide.
FACE_EXPAND = 1.15
FACE_FEATHER = 0.15


class MissingMLDependencyError(ImportError):
    """Raised when the optional `ml` extra is not installed."""


@dataclass
class Analysis:
    depth: np.ndarray  # (H, W) float32 0..1, 1 = near
    matte: np.ndarray  # (H, W) float32 0..1, 1 = subject
    faces: np.ndarray | None = None  # (H, W) float32 0..1, 1 = face; None = not computed

    def faces_or_zeros(self) -> np.ndarray:
        return self.faces if self.faces is not None else np.zeros_like(self.depth)


def _require(module: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingMLDependencyError(
            f"'{module}' is not installed. Depth and matte estimation need the ml "
            f"extra: run  {ML_INSTALL_HINT}"
        ) from exc


def _device() -> str:
    torch = _require("torch")
    return "cuda" if torch.cuda.is_available() else "cpu"


@functools.lru_cache(maxsize=None)
def _depth_pipeline(device: str) -> Any:
    transformers = _require("transformers")
    return transformers.pipeline("depth-estimation", model=DEPTH_MODEL, device=device)


@functools.lru_cache(maxsize=None)
def _matte_session(device: str) -> Any:
    rembg = _require("rembg")
    providers = ["CPUExecutionProvider"]
    if device == "cuda":
        providers.insert(0, "CUDAExecutionProvider")
    return rembg.new_session(MATTE_MODEL, providers=providers)


def _to_pil(image: np.ndarray) -> Any:
    pil_image = _require("PIL.Image")
    return pil_image.fromarray(to_uint8(image))


def depth_range(raw: np.ndarray) -> tuple[float, float]:
    """Robust (low, high) of raw depth, from the configured percentiles."""
    lo, hi = np.percentile(raw, [DEPTH_LOW_PERCENTILE, DEPTH_HIGH_PERCENTILE])
    return float(lo), float(hi)


def normalize_raw_depth(raw: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map raw depth (larger = nearer) to 0..1 with a fixed (lo, hi) range."""
    depth = (np.asarray(raw, np.float32) - lo) / max(hi - lo, float(np.finfo(np.float32).eps))
    return np.clip(depth, 0.0, 1.0).astype(np.float32)


def _orient(raw: np.ndarray) -> np.ndarray:
    """Make larger values nearer, whatever the model's convention."""
    return -raw if DEPTH_MODEL_INVERTED else raw


def normalize_depth(raw: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize raw model depth to `size` (width, height), normalize to 0..1, 1 = near."""
    raw = np.asarray(raw, dtype=np.float32)
    resized = _orient(cv2.resize(raw, size, interpolation=cv2.INTER_CUBIC))
    return normalize_raw_depth(resized, *depth_range(resized))


def estimate_depth_raw(image: np.ndarray) -> np.ndarray:
    """Unnormalized model depth at the image size, (H, W) float32, larger = nearer.

    Values are only comparable between frames of one shot; normalize them with a
    shared range (see `depth_range`) so depth does not pump over time.
    """
    pipe = _depth_pipeline(_device())
    result = pipe(_to_pil(image))
    raw = result["predicted_depth"].squeeze().detach().cpu().numpy().astype(np.float32)
    height, width = image.shape[:2]
    return _orient(cv2.resize(raw, (width, height), interpolation=cv2.INTER_CUBIC))


def estimate_depth(image: np.ndarray) -> np.ndarray:
    """RGB float32 (H, W, 3) to depth float32 (H, W), 0..1, 1 = near."""
    raw = estimate_depth_raw(image)
    return normalize_raw_depth(raw, *depth_range(raw))


def estimate_matte(image: np.ndarray) -> np.ndarray:
    """RGB float32 (H, W, 3) to subject matte float32 (H, W), 0..1, 1 = subject."""
    rembg = _require("rembg")
    mask = rembg.remove(_to_pil(image), session=_matte_session(_device()), only_mask=True)
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    height, width = image.shape[:2]
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    return from_uint8(mask)


class FaceModelUnavailable(RuntimeError):
    """The YuNet model could not be downloaded or loaded."""


def face_model_path(model_dir: str | Path) -> Path:
    """The YuNet model in `model_dir`, downloaded and checked on first use."""
    path = Path(model_dir) / FACE_MODEL_FILE
    if path.is_file() and _sha256(path) == FACE_MODEL_SHA256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.download")
    try:
        with urllib.request.urlopen(FACE_MODEL_URL, timeout=FACE_DOWNLOAD_TIMEOUT_S) as r:
            tmp.write_bytes(r.read())
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise FaceModelUnavailable(f"could not download the face model: {exc}") from exc
    if _sha256(tmp) != FACE_MODEL_SHA256:
        tmp.unlink(missing_ok=True)
        raise FaceModelUnavailable("downloaded face model does not match its checksum")
    os.replace(tmp, path)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@functools.lru_cache(maxsize=4)
def _face_detector(model: str) -> Any:
    # OpenCV 5 warns "setPreferableTarget ... not supported by the new graph
    # engine" while building the detector; it is harmless, so silence it here only.
    logging = cv2.utils.logging
    previous = logging.getLogLevel()
    logging.setLogLevel(logging.LOG_LEVEL_ERROR)
    try:
        return cv2.FaceDetectorYN.create(
            model, "", (320, 320), FACE_SCORE_FLOOR, FACE_NMS_THRESHOLD, FACE_TOP_K
        )
    finally:
        logging.setLogLevel(previous)


def face_mask(shape: tuple[int, int], boxes: np.ndarray) -> np.ndarray:
    """Soft elliptical mask (H, W) float32 0..1 from face boxes (x, y, w, h, ...)."""
    h, w = shape
    mask = np.zeros((h, w), np.float32)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    for box in boxes:
        x, y, bw, bh = (float(v) for v in box[:4])
        if bw <= 0 or bh <= 0:
            continue
        cx, cy = x + bw / 2.0, y + bh / 2.0
        ax, ay = bw / 2.0 * FACE_EXPAND, bh / 2.0 * FACE_EXPAND
        r = np.sqrt(((xs - cx) / ax) ** 2 + ((ys - cy) / ay) ** 2)  # 1 on the ellipse
        soft = np.clip((1.0 + FACE_FEATHER / 2.0 - r) / FACE_FEATHER, 0.0, 1.0)
        mask = np.maximum(mask, soft * soft * (3.0 - 2.0 * soft))
    return mask


def detect_face_boxes(image: np.ndarray, model_dir: str | Path = ".cache") -> np.ndarray:
    """Face boxes for an RGB float32 image as an (N, 5) float32 array of
    (x, y, w, h, score) in image pixels, merged over FACE_SCALES."""
    detector = _face_detector(str(face_model_path(model_dir)))
    bgr = to_bgr_uint8(image)
    h, w = image.shape[:2]
    found = []
    for scale in FACE_SCALES:
        sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        scaled = bgr if (sw, sh) == (w, h) else cv2.resize(bgr, (sw, sh), interpolation=interp)
        detector.setInputSize((sw, sh))
        _, boxes = detector.detect(scaled)
        if boxes is not None:
            found.append(np.column_stack([boxes[:, :4] / scale, boxes[:, 14]]))
    if not found:
        return np.zeros((0, 5), np.float32)
    boxes = np.concatenate(found).astype(np.float32)
    keep = cv2.dnn.NMSBoxes(
        boxes[:, :4].tolist(), boxes[:, 4].tolist(), FACE_SCORE_FLOOR, FACE_NMS_THRESHOLD
    )
    return boxes[np.asarray(keep, dtype=int).reshape(-1)]


def detect_faces(
    image: np.ndarray, model_dir: str | Path = ".cache", min_score: float = 0.75
) -> np.ndarray:
    """Soft elliptical face mask for an RGB float32 image, (H, W) float32 0..1."""
    boxes = detect_face_boxes(image, model_dir)
    return face_mask(image.shape[:2], boxes[boxes[:, 4] >= min_score])


def _face_boxes_or_none(image: np.ndarray, model_dir: str | Path | None) -> np.ndarray | None:
    """Face boxes, or None (with a warning) if the face model is unavailable."""
    try:
        return np.asarray(detect_face_boxes(image, model_dir or ".cache"), np.float32).reshape(-1, 5)
    except FaceModelUnavailable as exc:
        warnings.warn(f"{exc}; rendering without face handling", stacklevel=3)
        return None


def cache_key(image: np.ndarray) -> str:
    """sha1 of the image bytes, shape, and the model ids that produced the analysis."""
    h = hashlib.sha1()
    arr = np.ascontiguousarray(image, dtype=np.float32)
    h.update(repr(arr.shape).encode())
    h.update(arr.tobytes())
    meta = f"{DEPTH_MODEL}|{MATTE_MODEL}|inv={DEPTH_MODEL_INVERTED}|v{CACHE_VERSION}"
    h.update(meta.encode())
    return h.hexdigest()


def _check(name: str, arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape != shape:
        raise ValueError(f"{name} has shape {arr.shape}, expected {shape}")
    return np.clip(arr, 0.0, 1.0)


def _load_cached(path: Path | None) -> dict[str, np.ndarray] | None:
    if path is None or not path.is_file():
        return None
    try:
        with np.load(path) as data:
            return {k: np.asarray(data[k]) for k in data.files}
    except (OSError, ValueError):
        return None  # unreadable cache file: recompute and overwrite


def _save_cached(path: Path | None, arrays: dict[str, np.ndarray]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, path)


def _cached_analysis(
    image: np.ndarray,
    path: Path | None,
    depth_key: str,
    estimate: Any,
    faces: bool,
    cache_dir: str | Path | None,
    face_min_score: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """(depth-like array, matte, face mask) from the cache or the estimators.

    Face boxes are cached with their scores, so the mask for any `face_min_score`
    is built without detecting again. A cache file without face boxes is
    completed without re-running the depth and matte models.
    """
    shape = image.shape[:2]
    data = _load_cached(path)
    try:
        depth = np.asarray(data[depth_key], np.float32)
        matte = _check("cached matte", data["matte"], shape)
        if depth.shape != shape:
            raise ValueError("stale shape")
    except (TypeError, KeyError, ValueError):
        data = None
    if data is None:
        depth = np.asarray(estimate(image), np.float32)
        if depth.shape != shape:
            raise ValueError(f"{depth_key} has shape {depth.shape}, expected {shape}")
        matte = _check("matte", estimate_matte(image), shape)
        data = {depth_key: depth, "matte": matte}
        dirty = True
    else:
        dirty = False

    face_mask_ = None
    if faces:
        boxes = data.get("face_boxes")
        if boxes is None or boxes.ndim != 2 or boxes.shape[1] != 5:
            boxes = _face_boxes_or_none(image, cache_dir)
            if boxes is not None:
                data["face_boxes"] = boxes
                dirty = True
        if boxes is not None:
            face_mask_ = face_mask(shape, boxes[boxes[:, 4] >= face_min_score])
    if dirty:
        data.pop("faces", None)  # masks from an earlier format
        _save_cached(path, data)
    return depth, matte, face_mask_


def analyze(
    image: np.ndarray,
    cache_dir: str | Path | None = ".cache",
    faces: bool = True,
    face_min_score: float = 0.75,
) -> Analysis:
    """Depth, matte, and (if `faces`) a face mask for an RGB float32 image,
    cached as .npz in `cache_dir`. Pass cache_dir=None to disable caching."""
    path = None if cache_dir is None else Path(cache_dir) / f"analysis-{cache_key(image)}.npz"
    depth, matte, face_mask_ = _cached_analysis(
        image, path, "depth", estimate_depth, faces, cache_dir, face_min_score
    )
    return Analysis(depth=_check("depth", depth, image.shape[:2]), matte=matte, faces=face_mask_)


def analyze_raw(
    image: np.ndarray,
    key: str,
    cache_dir: str | Path | None = ".cache",
    faces: bool = True,
    face_min_score: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """(raw depth, matte, faces) for one frame of a shot, cached under `key`.

    Raw depth is unnormalized (larger = nearer), so a caller can normalize a whole
    shot with one range. `key` identifies the frame, e.g. a video hash and frame
    index; the model ids are added to it. Faces is None when `faces` is False or
    the face model is unavailable.
    """
    path = None
    if cache_dir is not None:
        meta = f"{key}|{DEPTH_MODEL}|{MATTE_MODEL}|inv={DEPTH_MODEL_INVERTED}|v{CACHE_VERSION}"
        path = Path(cache_dir) / f"frame-{hashlib.sha1(meta.encode()).hexdigest()}.npz"
    return _cached_analysis(
        image, path, "raw_depth", estimate_depth_raw, faces, cache_dir, face_min_score
    )
