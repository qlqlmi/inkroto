"""Selective color: a few flat paint accents (reds, oranges) over the ink."""

from __future__ import annotations

import cv2
import numpy as np

from inkroto.config import ColorConfig

# cv2.cvtColor on float32 gives hue in degrees (0..360); config hues use the
# OpenCV 8-bit convention (0..180).
HUE_DEGREES_PER_UNIT = 2.0


def color_mask(
    rgb: np.ndarray,
    hue_ranges: list[list[int]],
    min_sat: float,
    min_val: float,
    morph: int,
    min_area: int,
    feather: float,
) -> np.ndarray:
    """Soft (H, W) float32 0..1 mask of saturated pixels whose hue is in any range.

    Ranges are inclusive [lo, hi] in OpenCV hue units, so a red that wraps around
    zero is written as two ranges, [0, 12] and [168, 180].
    """
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.float32), cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0] / HUE_DEGREES_PER_UNIT
    in_hue = np.zeros(hue.shape, bool)
    for lo, hi in hue_ranges:
        in_hue |= (hue >= lo) & (hue <= hi)
    mask = (in_hue & (hsv[..., 1] > min_sat) & (hsv[..., 2] > min_val)).astype(np.uint8)

    if morph > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if min_area > 1 and mask.any():
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] >= min_area
        keep[0] = False
        mask = keep[labels].astype(np.uint8)

    soft = mask.astype(np.float32)
    if feather > 0 and soft.any():
        soft = cv2.GaussianBlur(soft, (0, 0), feather)
    return np.clip(soft, 0.0, 1.0)


def accent_layer(
    rgb: np.ndarray, saturation: float, values: list[float], split: float
) -> np.ndarray:
    """Flat paint version of the source: same hue, high saturation, and value
    quantized to two levels."""
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.float32), cv2.COLOR_RGB2HSV)
    dark, light = values
    hsv[..., 1] = saturation
    hsv[..., 2] = np.where(hsv[..., 2] < split, dark, light)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def apply_color(
    ink_gray: np.ndarray, rgb_src: np.ndarray, cfg: ColorConfig, lines: np.ndarray
) -> np.ndarray:
    """RGB frame: ink blended toward flat paint where the mask is set, with the
    line layer multiplied back on top so black lines stay black."""
    ink = np.repeat(np.asarray(ink_gray, np.float32)[..., None], 3, axis=2)
    if not cfg.enabled:
        return ink
    mask = color_mask(
        rgb_src, cfg.hue_ranges, cfg.min_sat, cfg.min_val, cfg.morph, cfg.min_area, cfg.feather
    )
    if not mask.any():
        return ink
    paint = accent_layer(rgb_src, cfg.accent_saturation, cfg.accent_values, cfg.accent_split)
    alpha = (mask * cfg.strength)[..., None]
    out = (ink * (1.0 - alpha) + paint * alpha) * np.asarray(lines, np.float32)[..., None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)
