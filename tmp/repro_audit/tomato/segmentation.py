"""K-Means fruit/background segmentation, fruit-cluster selection and Jaccard evaluation."""

from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import ndimage as ndi
from sklearn.cluster import KMeans

from . import config


def kmeans_labels(image: np.ndarray, combination: str, seed: int = config.SEED):
    """Cluster the pixels of one image using only the channels in ``combination``.

    Centroids are fitted on a uniform random sample of pixels (fixed seed) and every
    pixel is then assigned to its nearest centroid. Returns the label image and the
    number of Lloyd iterations of the retained run (convergence check).
    """
    data = image.reshape(-1, 3)[:, [config.CHANNELS[c] for c in combination]].astype(np.float32)
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(data), size=min(config.PIXEL_SAMPLE, len(data)), replace=False)
    km = KMeans(random_state=seed, **config.KMEANS_PARAMS).fit(data[sample])
    return km.predict(data).reshape(image.shape[:2]), int(km.n_iter_)


def cluster_descriptors(labels: np.ndarray, image: np.ndarray) -> list[dict]:
    """Mask-free descriptors of each cluster used to decide which one is the fruit."""
    h, w = labels.shape
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    out = []
    for cluster in range(config.KMEANS_PARAMS["n_clusters"]):
        mask = labels == cluster
        n_cc, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        if n_cc > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest = 1 + int(np.argmax(areas))
            coherence = float(areas.max()) / max(1.0, float(mask.sum()))
            cx, cy = centroids[largest]
        else:
            coherence, cx, cy = 0.0, w / 2, h / 2
        border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
        out.append({
            "area": float(mask.mean()),
            "coherence": coherence,
            # 1 at the image centre, 0 at a corner (distance normalised by sqrt(0.5)).
            "centrality": 1.0 - min(1.0, math.hypot(cx / w - 0.5, cy / h - 0.5) / math.sqrt(0.5)),
            "saturation": float(hsv[..., 1][mask].mean() / 255.0) if mask.any() else 0.0,
            "value": float(hsv[..., 2][mask].mean() / 255.0) if mask.any() else 0.0,
            "border": float(border.mean()),
        })
    return out


def heuristic_score(d: dict) -> float:
    """Object score: bright/saturated, compact, central cluster that avoids the border.

    The area term is a Gaussian prior centred on the typical annotated fruit area.
    """
    area_prior = math.exp(-((d["area"] - 0.38) / 0.32) ** 2)
    return (0.10 * d["coherence"] + 0.15 * d["centrality"] + 0.30 * d["saturation"]
            + 0.50 * d["value"] + 0.10 * area_prior - 0.10 * d["border"])


SELECTION_RULES = {
    "heuristic": lambda ds: int(np.argmax([heuristic_score(d) for d in ds])),
    "min_border": lambda ds: int(np.argmin([d["border"] for d in ds])),
    "max_saturation": lambda ds: int(np.argmax([d["saturation"] for d in ds])),
}
SELECTED_RULE = "heuristic"


def postprocess(mask: np.ndarray) -> np.ndarray:
    """Opening + closing, hole filling and removal of components < 1 % of the image."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.MORPH_KERNEL, config.MORPH_KERNEL))
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    m = ndi.binary_fill_holes(m).astype(np.uint8)
    n_cc, cc, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = [c for c in range(1, n_cc) if stats[c, cv2.CC_STAT_AREA] >= config.MIN_COMPONENT_FRACTION * m.size]
    return np.isin(cc, keep)


def jaccard(pred: np.ndarray, ref: np.ndarray) -> float:
    union = np.logical_or(pred, ref).sum()
    return float(np.logical_and(pred, ref).sum() / union) if union else 1.0


def segment_dataset(records: list[dict], split: np.ndarray, seed: int = config.SEED):
    """Run K-Means for every image and combination.

    Returns a per-image table (Jaccard for every selection rule, the oracle cluster,
    and the post-processed mask) and the selected raw masks keyed by (index, combination).
    The reference mask is used only to compute Jaccard, never to build a mask.
    """
    rows, masks = [], {}
    for i, rec in enumerate(records):
        ref = rec["reference_mask"]
        for combo in config.COMBINATIONS:
            labels, n_iter = kmeans_labels(rec["image"], combo, seed)
            descriptors = cluster_descriptors(labels, rec["image"])
            cluster_j = [jaccard(labels == c, ref) for c in range(len(descriptors))]
            row = {
                "index": i, "name": rec["name"], "class": rec["class_name"], "split": split[i],
                "combination": combo, "n_iter": n_iter,
                "jaccard_oracle_cluster": max(cluster_j),
            }
            for rule, select in SELECTION_RULES.items():
                row[f"jaccard_{rule}"] = cluster_j[select(descriptors)]
            chosen = labels == SELECTION_RULES[SELECTED_RULE](descriptors)
            row["jaccard"] = row[f"jaccard_{SELECTED_RULE}"]
            row["jaccard_post"] = jaccard(postprocess(chosen), ref)
            masks[(i, combo)] = chosen
            rows.append(row)
    return rows, masks
