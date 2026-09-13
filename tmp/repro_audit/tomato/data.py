"""Dataset discovery, duplicate control and image-level stratified partition."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from . import config

CLASSES = (("unripe", 0), ("ripe", 1))
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_data_root(path: Path | None) -> Path:
    """Return the dataset root (a folder containing ``ripe/`` and ``unripe/``)."""
    candidates = [path] if path is not None else [Path("data"), Path("dataset")]
    for candidate in candidates:
        if all((candidate / name / "images").is_dir() for name, _ in CLASSES):
            return candidate
    raise FileNotFoundError(f"No dataset with ripe/ and unripe/ found in {candidates}")


def resize_pair(image: np.ndarray, mask: np.ndarray, max_side: int = config.MAX_SIDE):
    """Downscale image and mask to the same size, preserving aspect ratio."""
    scale = min(1.0, max_side / max(image.shape[:2]))
    if scale == 1.0:
        return image, mask
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    image_r = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    mask_r = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    return image_r, mask_r


def load_records(data_root: Path) -> tuple[list[dict], list[dict]]:
    """Load every image/mask pair and drop byte-identical duplicate images.

    Exact duplicates would otherwise be able to fall on both sides of the partition
    (information leakage) or be counted twice. A duplicate with a conflicting label is an
    annotation error and aborts the run.
    """
    records: list[dict] = []
    duplicates: list[dict] = []
    seen: dict[str, dict] = {}
    for class_name, label in CLASSES:
        image_dir = data_root / class_name / "images"
        mask_dir = data_root / class_name / "masks"
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            digest = hashlib.md5(image_path.read_bytes()).hexdigest()
            if digest in seen:
                kept = seen[digest]
                if kept["label"] != label:
                    raise ValueError(f"{image_path.name} duplicates {kept['name']} with another label")
                duplicates.append({"removed": image_path.name, "kept": kept["name"], "class": class_name})
                continue
            mask_path = mask_dir / f"{image_path.stem}_mask.png"
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")
            pil_image = Image.open(image_path)
            original_size, bands = pil_image.size, len(pil_image.getbands())
            image = np.asarray(pil_image.convert("RGB"))  # RGBA -> RGB drops alpha
            mask_img = Image.open(mask_path)
            if mask_img.size != original_size:
                raise ValueError(f"Mask size {mask_img.size} != image size {original_size} for {image_path.name}")
            mask = np.asarray(mask_img.convert("L")) > 0
            image, mask = resize_pair(image, mask)
            record = {
                "name": image_path.name,
                "class_name": class_name,
                "label": label,
                "width": original_size[0],
                "height": original_size[1],
                "channels": bands,
                "image": image,
                "reference_mask": mask,
            }
            seen[digest] = record
            records.append(record)
    return records, duplicates


def split_records(labels: np.ndarray, seed: int = config.SEED) -> np.ndarray:
    """Stratified image-level split with integer sizes closest to 60/20/20."""
    n = len(labels)
    n_val = round(n * config.SPLIT_FRACTIONS[1])
    n_test = round(n * config.SPLIT_FRACTIONS[2])
    idx = np.arange(n)
    dev_idx, test_idx = train_test_split(idx, test_size=n_test, stratify=labels, random_state=seed)
    train_idx, val_idx = train_test_split(
        dev_idx, test_size=n_val, stratify=labels[dev_idx], random_state=seed
    )
    split = np.empty(n, dtype=object)
    split[train_idx], split[val_idx], split[test_idx] = "train", "validation", "test"
    return split.astype(str)
