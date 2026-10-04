"""Two-layer depth parallax: a slow push-in and drift from one photo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from inkroto.analyze import Analysis
from inkroto.config import Config, MotionConfig
from inkroto.io import from_uint16, to_uint16
from inkroto.motion.base import Shot, frame_count

# The background plate is inpainted at this fraction of full resolution.
INPAINT_SCALE = 0.5


def smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def background_plate(
    image: np.ndarray, depth: np.ndarray, matte: np.ndarray, mcfg: MotionConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Image and depth with the subject removed by TELEA inpainting.

    The subject mask is dilated before inpainting so the fill is taken from clean
    background beyond the subject's edge. Only pixels where the matte is nonzero are
    replaced; everything else keeps its original value, so a still camera
    reproduces the input exactly.
    """
    mask = (matte > mcfg.matte_threshold).astype(np.uint8)
    if mcfg.dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * mcfg.dilate_px + 1, 2 * mcfg.dilate_px + 1)
        )
        mask = cv2.dilate(mask, kernel)
    if not mask.any():
        return image.copy(), depth.copy()

    h, w = depth.shape
    small = (max(1, round(w * INPAINT_SCALE)), max(1, round(h * INPAINT_SCALE)))
    small_mask = (cv2.resize(mask.astype(np.float32), small, interpolation=cv2.INTER_AREA) > 0)
    small_mask = small_mask.astype(np.uint8)

    def fill(channel: np.ndarray) -> np.ndarray:
        # Inpaint in 16 bit: OpenCV's float32 TELEA overshoots far outside 0..1.
        low = cv2.resize(channel, small, interpolation=cv2.INTER_AREA)
        low = cv2.inpaint(to_uint16(low), small_mask, mcfg.inpaint_radius, cv2.INPAINT_TELEA)
        return cv2.resize(from_uint16(low), (w, h), interpolation=cv2.INTER_CUBIC)

    keep = (mask == 0) | (matte <= 0.0)
    plate = np.stack([fill(np.ascontiguousarray(image[..., c])) for c in range(3)], axis=-1)
    plate = np.where(keep[..., None], image, np.clip(plate, 0.0, 1.0))
    plate_depth = np.where(keep, depth, np.clip(fill(depth), 0.0, 1.0))
    return plate.astype(np.float32), plate_depth.astype(np.float32)


def foreground_layer_depth(depth: np.ndarray, matte: np.ndarray, threshold: float) -> np.ndarray:
    """Depth that drives the foreground warp: subject depth inside the matte, and the
    mean subject depth outside it, so the subject moves as one piece without tearing
    at its silhouette."""
    subject = matte > threshold
    if not subject.any():
        return depth
    fill = float(depth[subject].mean())
    return np.where(subject, depth, fill).astype(np.float32)


@dataclass
class _Layers:
    image: np.ndarray
    depth: np.ndarray
    matte: np.ndarray
    faces: np.ndarray
    plate: np.ndarray
    plate_depth: np.ndarray
    fg_drive: np.ndarray  # blurred depth that drives the foreground map
    bg_drive: np.ndarray  # blurred depth that drives the background map


def sampling_map(
    drive: np.ndarray, offset: tuple[float, float], zoom: float, focus: float
) -> tuple[np.ndarray, np.ndarray]:
    """Backward map for cv2.remap: where each output pixel samples the source."""
    h, w = drive.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    rel = drive - np.float32(focus)
    sx = xs + np.float32(offset[0] * w) * rel
    sy = ys + np.float32(offset[1] * h) * rel
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    inv = np.float32(1.0 / zoom)
    map_x = np.float32(cx) + (sx - np.float32(cx)) * inv
    map_y = np.float32(cy) + (sy - np.float32(cy)) * inv
    return map_x, map_y


def _warp(arr: np.ndarray, maps: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return cv2.remap(arr, maps[0], maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


class ParallaxBackend:
    def __init__(self, cache_dir: str | Path | None = None):
        pass  # the photo's analysis arrives already computed

    def build(self, image: np.ndarray, analysis: Analysis, cfg: Config) -> Shot:
        mcfg = cfg.motion
        depth, matte = analysis.depth, analysis.matte
        h, w = depth.shape
        if image.shape[:2] != (h, w) or matte.shape != (h, w):
            raise ValueError("image, depth, and matte must share (H, W)")

        plate, plate_depth = background_plate(image, depth, matte, mcfg)
        fg_drive = foreground_layer_depth(depth, matte, mcfg.matte_threshold)
        if mcfg.depth_blur > 0:
            fg_drive = cv2.GaussianBlur(fg_drive, (0, 0), mcfg.depth_blur)
            bg_drive = cv2.GaussianBlur(plate_depth, (0, 0), mcfg.depth_blur)
        else:
            bg_drive = plate_depth
        faces = analysis.faces_or_zeros()
        layers = _Layers(image, depth, matte, faces, plate, plate_depth, fg_drive, bg_drive)

        n = frame_count(cfg)
        pan = (float(mcfg.pan[0]), float(mcfg.pan[1]))

        def compute(i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            t = i / (n - 1) if n > 1 else 0.0
            s = smoothstep(t)
            offset = (pan[0] * (s - 0.5), pan[1] * (s - 0.5))
            zoom = 1.0 + (mcfg.zoom - 1.0) * s
            return self._frame(layers, offset, zoom, mcfg.focus)

        return Shot(n, (w, h), compute)

    @staticmethod
    def _frame(
        layers: _Layers, offset: tuple[float, float], zoom: float, focus: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        fg_map = sampling_map(layers.fg_drive, offset, zoom, focus)
        bg_map = sampling_map(layers.bg_drive, offset, zoom, focus)

        fg = _warp(layers.image, fg_map)
        fg_matte = np.clip(_warp(layers.matte, fg_map), 0.0, 1.0)
        fg_depth = _warp(layers.depth, fg_map)
        bg = _warp(layers.plate, bg_map)
        bg_depth = _warp(layers.plate_depth, bg_map)

        alpha = fg_matte[..., None]
        frame = fg * alpha + bg * (1.0 - alpha)
        depth = fg_depth * fg_matte + bg_depth * (1.0 - fg_matte)
        # Faces on the subject move with it; faces in the background move with
        # the plate (and the subject's faces are left out of the plate's copy).
        fg_faces = _warp(layers.faces, fg_map)
        bg_faces = _warp(layers.faces * (1.0 - layers.matte), bg_map)
        faces = fg_faces * fg_matte + bg_faces * (1.0 - fg_matte)
        return (
            np.clip(frame, 0.0, 1.0).astype(np.float32),
            np.clip(depth, 0.0, 1.0).astype(np.float32),
            fg_matte.astype(np.float32),
            np.clip(faces, 0.0, 1.0).astype(np.float32),
        )
