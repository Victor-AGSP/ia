"""Colour descriptors of the segmented fruit."""

from __future__ import annotations

import cv2
import numpy as np

from . import config


def pixel_colours(image: np.ndarray) -> dict[str, np.ndarray]:
    """Per-pixel colour representations on interpretable scales.

    R, G, B in [0, 255]; H in degrees on (-180, 180] so that red sits around 0 and green
    around +100 without the 0/360 wrap; S, V in [0, 1]; CIELAB a* (green-red) and
    b* (blue-yellow) on their usual scale (about -128..127).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    lab = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    hue = hsv[..., 0] * 2.0
    return {
        "R": image[..., 0].astype(np.float32),
        "G": image[..., 1].astype(np.float32),
        "B": image[..., 2].astype(np.float32),
        "H": np.where(hue > 180.0, hue - 360.0, hue),
        "S": hsv[..., 1] / 255.0,
        "V": hsv[..., 2] / 255.0,
        "a*": lab[..., 1],
        "b*": lab[..., 2],
    }


def circular_mean_deg(angles_deg: np.ndarray) -> float:
    """Mean direction of angles; an arithmetic mean of 5 deg and 355 deg would give 180 deg."""
    rad = np.deg2rad(angles_deg)
    return float(np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())))


def extract_features(image: np.ndarray, mask: np.ndarray, linear_hue: bool = False) -> dict[str, float]:
    """Mean descriptors over the fruit pixels only.

    ``linear_hue=True`` reproduces the naive arithmetic mean of H in [0, 360) and exists
    only for the ablation that quantifies why the circular mean is required.
    An empty mask (segmentation failure) falls back to the whole image and is flagged so
    the failure stays visible in the feature table instead of producing NaNs.
    """
    empty = not mask.any()
    region = np.ones(image.shape[:2], dtype=bool) if empty else mask
    colours = pixel_colours(image)
    feats = {name: float(colours[name][region].mean()) for name in config.FEATURES if name != "H"}
    hue = colours["H"][region]
    feats["H"] = float(np.mod(hue, 360.0).mean()) if linear_hue else circular_mean_deg(hue)
    feats["empty_mask"] = empty
    return {k: feats[k] for k in [*config.FEATURES, "empty_mask"]}
