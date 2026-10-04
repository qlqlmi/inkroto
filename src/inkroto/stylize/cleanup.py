"""Remove speckle from the tone map so regions read as clean shapes."""

from __future__ import annotations

import cv2
import numpy as np

# Pixel connectivity for connected components.
CONNECTIVITY = 8


def _fill_from_nearest(tones: np.ndarray, holes: np.ndarray) -> np.ndarray:
    """Give every pixel in `holes` the tone of the nearest pixel outside it."""
    if not holes.any() or holes.all():
        return tones
    # Zero pixels (the kept ones) each get a label in raster order; every hole
    # pixel receives the label of its nearest kept pixel.
    _, labels = cv2.distanceTransformWithLabels(
        holes.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5, labelType=cv2.DIST_LABEL_PIXEL
    )
    kept_values = tones[~holes]  # raster order, matching the labels
    out = tones.copy()
    out[holes] = kept_values[labels[holes] - 1]
    return out


def _remove_small(tones: np.ndarray, region: np.ndarray, min_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        region.astype(np.uint8), connectivity=CONNECTIVITY
    )
    small = stats[:, cv2.CC_STAT_AREA] < min_area
    small[0] = False  # label 0 is everything outside the region
    if not small.any():
        return tones
    return _fill_from_nearest(tones, small[labels])


def cleanup(tones: np.ndarray, min_area: int, morph: int) -> np.ndarray:
    """Drop black and white specks smaller than `min_area` pixels, then smooth
    region outlines with a morphological open and close.

    Removed specks take the tone that surrounds them. Values stay within the
    tones already present.
    """
    out = np.asarray(tones, dtype=np.float32)
    if min_area > 1:
        out = _remove_small(out, out == 0.0, min_area)
        out = _remove_small(out, out == 1.0, min_area)
    if morph > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, borderType=cv2.BORDER_REFLECT)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_REFLECT)
    return out.astype(np.float32)
