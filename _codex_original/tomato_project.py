"""Reproducible solution for Project 1: tomato segmentation and maturity classification.

The script reads the local dataset, creates an image-level train/validation/test split,
evaluates K-Means segmentation for all non-empty RGB channel combinations, extracts
RGB/HSV fruit descriptors, trains Gaussian Naive Bayes models with analysis-based
selection, greedy SFS and PCA, and writes tables/figures used by the LaTeX report.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


SEED = 20260913
MAX_SIDE = 256
PIXEL_SAMPLE = 12000
CHANNELS = {"R": 0, "G": 1, "B": 2}
COMBINATIONS = ["R", "G", "B", "RG", "RB", "GB", "RGB"]
FEATURES = ["R", "G", "B", "H", "S", "V"]


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def read_rgb(path: Path) -> np.ndarray:
    """Read RGB/RGBA image and discard alpha if present."""
    arr = np.asarray(Image.open(path).convert("RGB"))
    return arr


def resize_pair(image: np.ndarray, mask: np.ndarray, max_side: int = MAX_SIDE):
    scale = min(1.0, max_side / max(image.shape[:2]))
    if scale == 1:
        return image, mask
    size = (max(1, int(round(image.shape[1] * scale))), max(1, int(round(image.shape[0] * scale))))
    image_r = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    mask_r = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    return image_r, mask_r


def load_records(data_root: Path) -> list[dict]:
    records = []
    for label_name, label in [("unripe", 0), ("ripe", 1)]:
        image_dir = data_root / label_name / "images"
        mask_dir = data_root / label_name / "masks"
        for image_path in sorted(image_dir.glob("*")):
            mask_path = mask_dir / f"{image_path.stem}_mask.png"
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")
            image = read_rgb(image_path)
            mask = np.asarray(Image.open(mask_path).convert("L")) > 0
            image, mask = resize_pair(image, mask)
            records.append(
                {
                    "name": image_path.name,
                    "stem": image_path.stem,
                    "class_name": label_name,
                    "label": label,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "image": image,
                    "reference_mask": mask,
                }
            )
    return records


def split_records(records: list[dict], seed: int = SEED):
    idx = np.arange(len(records))
    y = np.array([r["label"] for r in records])
    dev_idx, test_idx = train_test_split(idx, test_size=0.20, random_state=seed, stratify=y)
    dev_y = y[dev_idx]
    train_idx, val_idx = train_test_split(
        dev_idx, test_size=0.25, random_state=seed, stratify=dev_y
    )
    split = {}
    for i in train_idx:
        split[int(i)] = "train"
    for i in val_idx:
        split[int(i)] = "validation"
    for i in test_idx:
        split[int(i)] = "test"
    return split


def foreground_cluster(labels: np.ndarray, image: np.ndarray) -> int:
    """Choose the tomato cluster without using the reference mask.

    Tomatoes are expected to form a compact object near the image center. The score
    combines connected-component coherence, centrality, saturation, and a mild area
    prior. The reference annotation is used only afterwards for Jaccard evaluation.
    """
    h, w = labels.shape
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    yy, xx = np.mgrid[0:h, 0:w]
    scores = []
    for cluster in (0, 1):
        mask = labels == cluster
        area = float(mask.mean())
        n_cc, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        if n_cc > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest = float(areas.max())
            coherence = largest / max(1.0, float(mask.sum()))
            largest_idx = 1 + int(np.argmax(areas))
            cx, cy = centroids[largest_idx]
        else:
            coherence, cx, cy = 0.0, w / 2, h / 2
        centrality = 1.0 - min(1.0, math.hypot((cx / max(w, 1)) - 0.5, (cy / max(h, 1)) - 0.5) / 0.7072)
        saturation = float(hsv[..., 1][mask].mean() / 255.0) if mask.any() else 0.0
        value = float(hsv[..., 2][mask].mean() / 255.0) if mask.any() else 0.0
        area_prior = math.exp(-((area - 0.38) / 0.32) ** 2)
        # Penalize a cluster that is mostly on the border, a common background pattern.
        border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
        border_fraction = float(border.mean())
        # Bright, saturated fruit is preferred over dark foliage or pale backgrounds.
        # The area prior prevents the full background from winning when it is compact.
        score = (
            0.10 * coherence
            + 0.15 * centrality
            + 0.30 * saturation
            + 0.50 * value
            + 0.10 * area_prior
            - 0.10 * border_fraction
        )
        scores.append(score)
    return int(np.argmax(scores))


def segment_image(image: np.ndarray, combination: str, seed: int = SEED):
    data = image.reshape(-1, 3)[:, [CHANNELS[c] for c in combination]].astype(np.float32)
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(data), size=min(PIXEL_SAMPLE, len(data)), replace=False)
    km = KMeans(
        n_clusters=2,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=seed,
    )
    km.fit(data[sample_idx])
    labels = km.predict(data).reshape(image.shape[:2])
    chosen = foreground_cluster(labels, image)
    return labels == chosen


def jaccard(pred: np.ndarray, ref: np.ndarray) -> float:
    intersection = np.logical_and(pred, ref).sum()
    union = np.logical_or(pred, ref).sum()
    return float(intersection / union) if union else 1.0


def extract_features(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if not mask.any():
        mask = np.ones(image.shape[:2], dtype=bool)
    rgb = image[mask].astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[mask].astype(np.float32)
    return {
        "R": float(rgb[:, 0].mean()),
        "G": float(rgb[:, 1].mean()),
        "B": float(rgb[:, 2].mean()),
        "H": float(hsv[:, 0].mean() * 2.0),
        "S": float(hsv[:, 1].mean() / 255.0),
        "V": float(hsv[:, 2].mean() / 255.0),
    }


def evaluate_segmentation(records: list[dict], split: dict[int, str], out_dir: Path):
    rows = []
    masks = {}
    for i, record in enumerate(records):
        for combo in COMBINATIONS:
            pred = segment_image(record["image"], combo, SEED)
            masks[(i, combo)] = pred
            rows.append(
                {
                    "index": i,
                    "name": record["name"],
                    "class": record["class_name"],
                    "split": split[i],
                    "combination": combo,
                    "jaccard": jaccard(pred, record["reference_mask"]),
                }
            )
    df = pd.DataFrame(rows)
    development = df[df["split"].isin(["train", "validation"])]
    summary = (
        df.groupby(["combination", "split"], as_index=False)["jaccard"].agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "jaccard_mean", "std": "jaccard_std"})
    )
    dev_means = development.groupby("combination")["jaccard"].mean().sort_values(ascending=False)
    best_combo = str(dev_means.index[0])
    df.to_csv(out_dir / "segmentation_per_image.csv", index=False)
    summary.to_csv(out_dir / "segmentation_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ordered = COMBINATIONS
    means = [development.loc[development.combination == c, "jaccard"].mean() for c in ordered]
    stds = [development.loc[development.combination == c, "jaccard"].std() for c in ordered]
    ax.bar(ordered, means, yerr=stds, capsize=3, color="#b33b3b")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Jaccard medio (desarrollo)")
    ax.set_xlabel("Canales usados por K-Means")
    ax.set_title("Segmentación: comparación de las siete combinaciones")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "segmentation_jaccard.png", dpi=180)
    plt.close(fig)
    return df, summary, masks, best_combo


def make_segmentation_examples(records, masks, best_combo: str, out_dir: Path):
    selected = [
        next(i for i, r in enumerate(records) if r["class_name"] == "ripe"),
        next(i for i, r in enumerate(records) if r["class_name"] == "unripe"),
    ]
    fig, axes = plt.subplots(len(selected), 4, figsize=(11, 5.6))
    for row, i in enumerate(selected):
        rec = records[i]
        axes[row, 0].imshow(rec["image"])
        axes[row, 0].set_title(f"Imagen ({rec['class_name']})")
        axes[row, 1].imshow(rec["reference_mask"], cmap="gray")
        axes[row, 1].set_title("Máscara de referencia")
        axes[row, 2].imshow(masks[(i, best_combo)], cmap="gray")
        axes[row, 2].set_title(f"K-Means {best_combo}")
        overlay = rec["image"].copy()
        pred = masks[(i, best_combo)]
        overlay[~pred] = (overlay[~pred] * 0.25).astype(np.uint8)
        axes[row, 3].imshow(overlay)
        axes[row, 3].set_title("Superposición")
        for col in range(4):
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "segmentation_examples.png", dpi=180)
    plt.close(fig)


def build_feature_table(records, masks, best_combo: str, split: dict[int, str], out_dir: Path):
    rows = []
    for i, rec in enumerate(records):
        feats = extract_features(rec["image"], masks[(i, best_combo)])
        row = {"index": i, "name": rec["name"], "class": rec["class_name"], "label": rec["label"], "split": split[i]}
        row.update(feats)
        row["segmented_area"] = float(masks[(i, best_combo)].mean())
        row["reference_area"] = float(rec["reference_mask"].mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "features.csv", index=False)
    return df


def exploratory_plot(features: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for ax, feature in zip(axes.ravel(), FEATURES):
        data = [features.loc[features["class"] == c, feature] for c in ["unripe", "ripe"]]
        ax.boxplot(data, tick_labels=["Inmaduro", "Maduro"], patch_artist=True,
                   boxprops={"facecolor": "#f4a261"}, medianprops={"color": "#1d3557"})
        ax.set_title(feature)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Características del fruto segmentado por clase", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def standardize_fit(X_fit, X_other):
    scaler = StandardScaler().fit(X_fit)
    return scaler, scaler.transform(X_fit), scaler.transform(X_other)


def cv_auc_for_features(X: np.ndarray, y: np.ndarray, feature_idx: list[int], seed: int = SEED) -> float:
    if len(feature_idx) == 0:
        return 0.5
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []
    for train, test in skf.split(X, y):
        scaler = StandardScaler().fit(X[train][:, feature_idx])
        clf = GaussianNB().fit(scaler.transform(X[train][:, feature_idx]), y[train])
        scores.append(roc_auc_score(y[test], clf.predict_proba(scaler.transform(X[test][:, feature_idx]))[:, 1]))
    return float(np.mean(scores))


def greedy_sfs(X: np.ndarray, y: np.ndarray, names: list[str], seed: int = SEED):
    remaining = list(range(X.shape[1]))
    selected: list[int] = []
    trace = []
    current = 0.5
    while remaining:
        candidates = []
        for idx in remaining:
            score = cv_auc_for_features(X, y, selected + [idx], seed)
            candidates.append((score, idx))
        score, idx = max(candidates, key=lambda x: (x[0], -x[1]))
        if not selected or score >= current - 0.01:
            selected.append(idx)
            remaining.remove(idx)
            current = score
            trace.append({"step": len(selected), "feature": names[idx], "cv_auc": score})
        else:
            break
        if len(selected) >= min(4, X.shape[1]):
            break
    return selected, pd.DataFrame(trace)


def youden_threshold(y_true: np.ndarray, scores: np.ndarray):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    finite = np.isfinite(thresholds)
    j = tpr - fpr
    j[~finite] = -np.inf
    pos = int(np.argmax(j))
    threshold = float(thresholds[pos])
    return threshold, float(j[pos]), fpr, tpr, thresholds


def metric_row(y_true, scores, threshold, model_name, split_name):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    specificity = tn / max(1, tn + fp)
    return {
        "model": model_name,
        "split": split_name,
        "threshold": float(threshold),
        "auc": float(roc_auc_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "sensitivity": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def fit_bayes_model(X: np.ndarray, y: np.ndarray, train_idx, val_idx, test_idx, feature_idx, name: str):
    # The threshold is selected only on validation scores from a model fit on train.
    scaler = StandardScaler().fit(X[train_idx][:, feature_idx])
    clf = GaussianNB().fit(scaler.transform(X[train_idx][:, feature_idx]), y[train_idx])
    val_scores = clf.predict_proba(scaler.transform(X[val_idx][:, feature_idx]))[:, 1]
    threshold, youden, *_ = youden_threshold(y[val_idx], val_scores)
    val_metrics = metric_row(y[val_idx], val_scores, threshold, name, "validation")
    # Refit only after all decisions are fixed, using train + validation.
    dev_idx = np.concatenate([train_idx, val_idx])
    scaler_final = StandardScaler().fit(X[dev_idx][:, feature_idx])
    clf_final = GaussianNB().fit(scaler_final.transform(X[dev_idx][:, feature_idx]), y[dev_idx])
    test_scores = clf_final.predict_proba(scaler_final.transform(X[test_idx][:, feature_idx]))[:, 1]
    test_metrics = metric_row(y[test_idx], test_scores, threshold, name, "test")
    fpr, tpr, _ = roc_curve(y[test_idx], test_scores)
    return {
        "name": name,
        "feature_idx": feature_idx,
        "threshold": threshold,
        "youden": youden,
        "val_scores": val_scores,
        "test_scores": test_scores,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "fpr": fpr,
        "tpr": tpr,
    }


def fit_pca_model(X: np.ndarray, y: np.ndarray, train_idx, val_idx, test_idx, n_components: int, name: str):
    scaler = StandardScaler().fit(X[train_idx])
    pca = PCA().fit(scaler.transform(X[train_idx]))
    pca_selected = PCA(n_components=n_components).fit(scaler.transform(X[train_idx]))
    clf = GaussianNB().fit(pca_selected.transform(scaler.transform(X[train_idx])), y[train_idx])
    val_scores = clf.predict_proba(pca_selected.transform(scaler.transform(X[val_idx])))[:, 1]
    threshold, youden, *_ = youden_threshold(y[val_idx], val_scores)
    val_metrics = metric_row(y[val_idx], val_scores, threshold, name, "validation")
    dev_idx = np.concatenate([train_idx, val_idx])
    scaler_final = StandardScaler().fit(X[dev_idx])
    pca_final = PCA(n_components=n_components).fit(scaler_final.transform(X[dev_idx]))
    clf_final = GaussianNB().fit(pca_final.transform(scaler_final.transform(X[dev_idx])), y[dev_idx])
    test_scores = clf_final.predict_proba(pca_final.transform(scaler_final.transform(X[test_idx])))[:, 1]
    test_metrics = metric_row(y[test_idx], test_scores, threshold, name, "test")
    fpr, tpr, _ = roc_curve(y[test_idx], test_scores)
    return {
        "name": name,
        "threshold": threshold,
        "youden": youden,
        "val_scores": val_scores,
        "test_scores": test_scores,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "fpr": fpr,
        "tpr": tpr,
        "pca": pca,
        "explained_variance": pca.explained_variance_.tolist(),
        "explained_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_ratio": np.cumsum(pca.explained_variance_ratio_).tolist(),
    }


def plot_roc(models, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for model in models:
        metrics = model["test_metrics"]
        ax.plot(model["fpr"], model["tpr"], lw=2, label=f"{model['name']} (AUC={metrics['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="Tasa de falsos positivos", ylabel="Sensibilidad", title="ROC en el conjunto de test")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_test.png", dpi=180)
    plt.close(fig)


def plot_pca(pca_model, out_dir: Path):
    ratio = np.array(pca_model["explained_ratio"])
    cumulative = np.array(pca_model["cumulative_ratio"])
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    x = np.arange(1, len(ratio) + 1)
    ax1.bar(x, ratio * 100, color="#457b9d", alpha=0.85, label="Individual")
    ax1.set_xlabel("Componente principal")
    ax1.set_ylabel("Varianza explicada (%)")
    ax1.set_xticks(x)
    ax2 = ax1.twinx()
    ax2.plot(x, cumulative * 100, "o-", color="#e63946", label="Acumulada")
    ax2.axhline(95, color="#e63946", ls="--", lw=1)
    ax2.set_ylabel("Varianza acumulada (%)")
    ax2.set_ylim(0, 105)
    ax1.set_title("PCA ajustado sobre entrenamiento")
    fig.tight_layout()
    fig.savefig(out_dir / "pca_variance.png", dpi=180)
    plt.close(fig)


def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def dataset_and_exploration_summary(records, split: dict[int, str], out_dir: Path):
    rows = []
    pixel_rows = []
    for i, rec in enumerate(records):
        original = Image.open(rec["image_path"])
        channels = len(original.getbands())
        rows.append(
            {
                "index": i,
                "name": rec["name"],
                "class": rec["class_name"],
                "split": split[i],
                "height": int(original.height),
                "width": int(original.width),
                "channels": channels,
                "reference_area": float(rec["reference_mask"].mean()),
            }
        )
        if split[i] == "test":
            continue
        hsv = cv2.cvtColor(rec["image"], cv2.COLOR_RGB2HSV).astype(np.float32)
        for region, sel in [("fruit", rec["reference_mask"]), ("background", ~rec["reference_mask"])]:
            rgb = rec["image"][sel].astype(np.float32)
            hs = hsv[sel]
            vals = {
                "R": rgb[:, 0], "G": rgb[:, 1], "B": rgb[:, 2],
                "H": hs[:, 0] * 2.0, "S": hs[:, 1] / 255.0, "V": hs[:, 2] / 255.0,
            }
            for feature, values in vals.items():
                pixel_rows.append(
                    {
                        "class": rec["class_name"], "region": region, "feature": feature,
                        "mean": float(values.mean()), "std": float(values.std()),
                    }
                )
    dataset_df = pd.DataFrame(rows)
    dataset_df.to_csv(out_dir / "dataset_summary.csv", index=False)
    pixel_df = pd.DataFrame(pixel_rows)
    pixel_summary = pixel_df.groupby(["class", "region", "feature"], as_index=False)[["mean", "std"]].mean()
    pixel_summary.to_csv(out_dir / "exploratory_pixel_summary.csv", index=False)
    return dataset_df, pixel_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("dataset"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    seed_everything()
    args.out.mkdir(parents=True, exist_ok=True)
    records = load_records(args.data)
    split = split_records(records)
    dataset_df, pixel_summary = dataset_and_exploration_summary(records, split, args.out)
    split_df = pd.DataFrame([{"index": i, "name": r["name"], "class": r["class_name"], "split": split[i]} for i, r in enumerate(records)])
    split_df.to_csv(args.out / "split.csv", index=False)

    seg_df, seg_summary, masks, best_combo = evaluate_segmentation(records, split, args.out)
    make_segmentation_examples(records, masks, best_combo, args.out)
    features = build_feature_table(records, masks, best_combo, split, args.out)
    exploratory_plot(features[features["split"] != "test"], args.out)

    train_idx = np.array([i for i in range(len(records)) if split[i] == "train"])
    val_idx = np.array([i for i in range(len(records)) if split[i] == "validation"])
    test_idx = np.array([i for i in range(len(records)) if split[i] == "test"])
    X = features[FEATURES].to_numpy(dtype=float)
    y = features["label"].to_numpy(dtype=int)

    # Analysis-based subset: the large class separation in R/G/H is visible in EDA.
    analysis_features = ["R", "G", "H"]
    analysis_idx = [FEATURES.index(f) for f in analysis_features]
    sfs_idx, sfs_trace = greedy_sfs(X[train_idx], y[train_idx], FEATURES)
    sfs_features = [FEATURES[i] for i in sfs_idx]
    sfs_trace.to_csv(args.out / "sfs_trace.csv", index=False)
    sfs_model = fit_bayes_model(X, y, train_idx, val_idx, test_idx, sfs_idx, "Bayes + SFS")
    analysis_model = fit_bayes_model(X, y, train_idx, val_idx, test_idx, analysis_idx, "Bayes + análisis")

    pca_probe = PCA().fit(StandardScaler().fit_transform(X[train_idx]))
    cumulative = np.cumsum(pca_probe.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, 0.95) + 1)
    pca_model = fit_pca_model(X, y, train_idx, val_idx, test_idx, n_components, "PCA + Bayes")
    models = [analysis_model, sfs_model, pca_model]
    plot_roc(models, args.out)
    plot_pca(pca_model, args.out)

    metrics = pd.DataFrame([m["test_metrics"] for m in models] + [m["val_metrics"] for m in models])
    metrics.to_csv(args.out / "classification_metrics.csv", index=False)

    # Per-image integration analysis uses the selected model's test probabilities.
    best_test = sfs_model
    test_scores = best_test["test_scores"]
    test_order = test_idx
    test_j = np.array([seg_df[(seg_df.index >= 0) & (seg_df["index"] == i) & (seg_df["combination"] == best_combo)]["jaccard"].iloc[0] for i in test_order])
    test_correct = (test_scores >= best_test["threshold"]).astype(int) == y[test_order]
    rho, pvalue = spearmanr(test_j, test_correct.astype(int)) if len(np.unique(test_correct)) > 1 else (np.nan, np.nan)
    integration = {
        "best_segmentation": best_combo,
        "best_segmentation_development_jaccard": float(seg_df[(seg_df["split"].isin(["train", "validation"])) & (seg_df["combination"] == best_combo)]["jaccard"].mean()),
        "best_segmentation_test_jaccard": float(seg_df[(seg_df["split"] == "test") & (seg_df["combination"] == best_combo)]["jaccard"].mean()),
        "test_mean_jaccard_correct": float(test_j[test_correct].mean()) if test_correct.any() else np.nan,
        "test_mean_jaccard_incorrect": float(test_j[~test_correct].mean()) if (~test_correct).any() else np.nan,
        "test_spearman_jaccard_correct": float(rho) if np.isfinite(rho) else None,
        "test_spearman_pvalue": float(pvalue) if np.isfinite(pvalue) else None,
    }

    split_counts = split_df.groupby(["split", "class"]).size().unstack(fill_value=0).to_dict()
    report = {
        "seed": SEED,
        "max_side": MAX_SIDE,
        "pixel_sample": PIXEL_SAMPLE,
        "n_images": len(records),
        "dataset_dimensions": {
            "height_min": int(dataset_df["height"].min()),
            "height_max": int(dataset_df["height"].max()),
            "width_min": int(dataset_df["width"].min()),
            "width_max": int(dataset_df["width"].max()),
            "channel_counts": {str(k): int(v) for k, v in dataset_df["channels"].value_counts().items()},
        },
        "class_counts": {k: int(v) for k, v in split_df["class"].value_counts().items()},
        "split_counts": split_counts,
        "best_segmentation": best_combo,
        "segmentation_summary": seg_summary.to_dict(orient="records"),
        "analysis_features": analysis_features,
        "sfs_features": sfs_features,
        "sfs_trace": sfs_trace.to_dict(orient="records"),
        "pca_n_components": n_components,
        "pca_explained_variance": pca_model["explained_variance"],
        "pca_explained_ratio": pca_model["explained_ratio"],
        "pca_cumulative_ratio": pca_model["cumulative_ratio"],
        "classification_metrics": metrics.to_dict(orient="records"),
        "integration": integration,
    }
    save_json(report, args.out / "report_data.json")
    print(json.dumps({"best_segmentation": best_combo, "analysis_features": analysis_features, "sfs_features": sfs_features, "pca_n_components": n_components, "metrics": [m["test_metrics"] for m in models]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
