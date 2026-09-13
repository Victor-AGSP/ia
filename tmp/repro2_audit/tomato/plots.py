"""Figures used by the report and the presentation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .features import pixel_colours
from .segmentation import postprocess

RIPE, UNRIPE, FRUIT, BACK = "#c0392b", "#5a8f29", "#d35400", "#7f8c8d"
STRATEGY_COLORS = {"Bayes + análisis": "#c0392b", "Bayes + SFS": "#2c3e50", "PCA + Bayes": "#e67e22"}
AXIS_LABEL = {"R": "R", "G": "G", "B": "B", "H": "H (°)", "S": "S", "V": "V", "a*": "a*", "b*": "b*"}
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10})


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def dataset_montage(records, split, out: Path, per_class: int = 4):
    fig, axes = plt.subplots(2, per_class * 2, figsize=(14, 3.8))
    for row, (cls, title) in enumerate([("ripe", "Maduro"), ("unripe", "Inmaduro")]):
        idx = [i for i, r in enumerate(records) if r["class_name"] == cls and split[i] == "train"][:per_class]
        for k, i in enumerate(idx):
            axes[row, 2 * k].imshow(records[i]["image"])
            axes[row, 2 * k + 1].imshow(records[i]["reference_mask"], cmap="gray")
            axes[row, 2 * k].set_title(f"{title}", fontsize=9)
            axes[row, 2 * k + 1].set_title("máscara", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    _save(fig, out / "dataset_montage.png")


def sample_pixels(records, indices, seed):
    """Balanced pixel sample per image and region (fruit/background) from reference masks."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in indices:
        rec = records[i]
        colours = pixel_colours(rec["image"])
        for region, sel in (("fruto", rec["reference_mask"]), ("fondo", ~rec["reference_mask"])):
            flat = np.flatnonzero(sel)
            if len(flat) == 0:
                continue
            take = rng.choice(flat, size=min(config.EDA_PIXELS_PER_REGION, len(flat)), replace=False)
            df = pd.DataFrame({k: v.reshape(-1)[take] for k, v in colours.items()})
            df["region"], df["class"] = region, rec["class_name"]
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def pixel_histograms(pixels: pd.DataFrame, out: Path):
    ranges = {"R": (0, 255), "G": (0, 255), "B": (0, 255), "H": (-180, 180), "S": (0, 1), "V": (0, 1),
              "a*": (-60, 80), "b*": (-40, 90)}
    specs = [
        ("pixel_hist_fruit_background.png", "Píxeles de entrenamiento: fruto vs fondo",
         [("fruto", pixels[pixels.region == "fruto"], FRUIT), ("fondo", pixels[pixels.region == "fondo"], BACK)]),
        ("pixel_hist_ripe_unripe.png", "Píxeles de fruto (máscara de referencia): maduro vs inmaduro",
         [("maduro", pixels[(pixels.region == "fruto") & (pixels["class"] == "ripe")], RIPE),
          ("inmaduro", pixels[(pixels.region == "fruto") & (pixels["class"] == "unripe")], UNRIPE)]),
    ]
    for fname, title, groups in specs:
        fig, axes = plt.subplots(2, 4, figsize=(13, 5.4))
        for ax, feat in zip(axes.ravel(), config.FEATURES):
            bins = np.linspace(*ranges[feat], 61)
            for label, df, color in groups:
                ax.hist(np.clip(df[feat], *ranges[feat]), bins=bins, density=True, histtype="stepfilled",
                        alpha=0.35, color=color, label=label)
                ax.hist(np.clip(df[feat], *ranges[feat]), bins=bins, density=True, histtype="step", color=color)
            ax.set_title(AXIS_LABEL[feat])
            ax.set_yticks([])
        axes[0, 0].legend(frameon=False)
        fig.suptitle(title, y=1.0)
        _save(fig, out / fname)


def feature_distributions(features: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6))
    rng = np.random.default_rng(config.SEED)
    for ax, feat in zip(axes.ravel(), config.FEATURES):
        data = [features.loc[features["class"] == c, feat].to_numpy() for c in ("unripe", "ripe")]
        ax.boxplot(data, tick_labels=["Inmaduro", "Maduro"], widths=0.5, showfliers=False,
                   medianprops={"color": "black"})
        for k, (vals, color) in enumerate(zip(data, (UNRIPE, RIPE))):
            ax.scatter(k + 1 + rng.uniform(-0.12, 0.12, len(vals)), vals, s=14, color=color, alpha=0.8, zorder=3)
        ax.set_title(AXIS_LABEL[feat])
    fig.suptitle("Descriptores del fruto segmentado (K-Means) — imágenes de entrenamiento", y=1.0)
    _save(fig, out / "feature_distributions.png")


def correlation_heatmap(corr: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [AXIS_LABEL[c] for c in corr.columns]
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Correlación de Pearson (entrenamiento)")
    _save(fig, out / "feature_correlation.png")


def segmentation_bars(seg: pd.DataFrame, out: Path):
    dev = seg[seg.split != "test"]
    g = dev.groupby("combination")[["jaccard", "jaccard_post", "jaccard_oracle_cluster"]].mean().loc[config.COMBINATIONS]
    sd = dev.groupby("combination")["jaccard"].std().loc[config.COMBINATIONS]
    x = np.arange(len(config.COMBINATIONS))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - 0.27, g["jaccard"], 0.27, yerr=sd, capsize=3, color="#c0392b", label="K-Means (regla automática)")
    ax.bar(x, g["jaccard_post"], 0.27, color="#e67e22", label="+ post-procesado morfológico")
    ax.bar(x + 0.27, g["jaccard_oracle_cluster"], 0.27, color="#bdc3c7", label="cota: mejor cluster (oráculo)")
    for k, v in enumerate(g["jaccard"]):
        ax.text(k - 0.27, 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=7, color="white", rotation=90)
    ax.set_xticks(x, config.COMBINATIONS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Jaccard medio (desarrollo)")
    ax.set_xlabel("Canales usados por K-Means")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3)
    _save(fig, out / "segmentation_jaccard.png")

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.boxplot([dev.loc[dev.combination == c, "jaccard"] for c in config.COMBINATIONS],
               tick_labels=config.COMBINATIONS, medianprops={"color": "black"})
    ax.set_ylabel("Jaccard por imagen (desarrollo)")
    ax.set_ylim(0, 1)
    _save(fig, out / "segmentation_jaccard_boxplot.png")


def segmentation_examples(records, masks, seg: pd.DataFrame, best: str, out: Path):
    """Best, median and worst development image for the selected combination."""
    dev = seg[(seg.split != "test") & (seg.combination == best)].sort_values("jaccard")
    picks = [("peor", dev.iloc[0]), ("mediana", dev.iloc[len(dev) // 2]), ("mejor", dev.iloc[-1])]
    fig, axes = plt.subplots(3, 5, figsize=(13, 7.6))
    for row, (tag, r) in enumerate(picks):
        rec, raw = records[int(r["index"])], masks[(int(r["index"]), best)]
        post = postprocess(raw)
        overlay = rec["image"].copy()
        overlay[~post] = (overlay[~post] * 0.25).astype(np.uint8)
        panels = [(rec["image"], f"{tag}: {rec['class_name']}"), (rec["reference_mask"], "referencia"),
                  (raw, f"K-Means {best}  J={r['jaccard']:.2f}"), (post, f"post-proc.  J={r['jaccard_post']:.2f}"),
                  (overlay, "superposición")]
        for col, (img, title) in enumerate(panels):
            axes[row, col].imshow(img, cmap="gray" if img.ndim == 2 else None)
            axes[row, col].set_title(title, fontsize=9)
            axes[row, col].axis("off")
    _save(fig, out / "segmentation_examples.png")


def segmentation_all_combinations(records, masks, seg: pd.DataFrame, out: Path):
    dev = seg[seg.split != "test"]
    per_image = dev.groupby("index")["jaccard"].mean().sort_values()
    picks = [int(per_image.index[len(per_image) // 3]), int(per_image.index[(2 * len(per_image)) // 3])]
    fig, axes = plt.subplots(len(picks), 9, figsize=(17, 4.2))
    for row, i in enumerate(picks):
        axes[row, 0].imshow(records[i]["image"])
        axes[row, 0].set_title(records[i]["class_name"], fontsize=9)
        axes[row, 1].imshow(records[i]["reference_mask"], cmap="gray")
        axes[row, 1].set_title("referencia", fontsize=9)
        for k, combo in enumerate(config.COMBINATIONS):
            j = seg[(seg["index"] == i) & (seg.combination == combo)]["jaccard"].iloc[0]
            axes[row, k + 2].imshow(masks[(i, combo)], cmap="gray")
            axes[row, k + 2].set_title(f"{combo}  J={j:.2f}", fontsize=9)
        for ax in axes[row]:
            ax.axis("off")
    _save(fig, out / "segmentation_all_combinations.png")


def sfs_trace(trace: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.errorbar(trace["step"], trace["cv_auc"], yerr=trace["cv_auc_std"], marker="o", capsize=3, color="#2c3e50")
    for _, r in trace.iterrows():
        ax.annotate("+" + AXIS_LABEL[r["added"]], (r["step"], r["cv_auc"]), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8)
    k = int(trace.loc[trace["selected"], "step"].iloc[0])
    ax.axvline(k, color="#c0392b", ls="--", lw=1)
    ax.set_xlabel("Número de características")
    ax.set_ylabel("AUC CV (5×10, entrenamiento)")
    ax.set_title("Traza de SFS")
    _save(fig, out / "sfs_trace.png")


def threshold_sweeps(results, out: Path):
    fig, axes = plt.subplots(1, len(results), figsize=(13, 3.6), sharey=True)
    for ax, res in zip(axes, results):
        sw = res["sweep"]
        ax.plot(sw["threshold"], sw["sensitivity"], drawstyle="steps-post", label="sensibilidad", color=RIPE)
        ax.plot(sw["threshold"], sw["specificity"], drawstyle="steps-post", label="especificidad", color=UNRIPE)
        ax.plot(sw["threshold"], sw["accuracy"], drawstyle="steps-post", label="exactitud", color="#2c3e50", ls=":")
        ax.axvline(res["threshold"], color="black", ls="--", lw=1)
        ax.set_title(f"{res['name']}\nln θ* = {res['threshold']:.2f}", fontsize=9)
        ax.set_xlabel("umbral sobre ln Λ(x)")
        ax.set_xscale("symlog", linthresh=1.0)
    axes[0].set_ylabel("validación")
    axes[0].legend(frameon=False, fontsize=8)
    _save(fig, out / "threshold_sweep.png")


def roc_curves(results, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, part in zip(axes, ("val", "test")):
        for res in results:
            fpr, tpr = res[f"{part}_roc"]
            m = res[f"{part}_metrics"]
            color = STRATEGY_COLORS[res["name"]]
            ax.plot(fpr, tpr, lw=2, color=color, label=f"{res['name']} (AUC={m['auc']:.3f})")
            ax.plot(1 - m["specificity"], m["sensitivity"], "o", ms=9, mfc="white", mec=color, mew=2)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set(xlabel="1 − especificidad (FPR)", ylabel="sensibilidad (TPR)",
               title="Validación (umbral elegido aquí)" if part == "val" else "Test (umbral fijo de validación)")
        ax.legend(loc="lower right", fontsize=8, frameon=False)
    _save(fig, out / "roc_curves.png")


def confusion_matrices(results, out: Path):
    fig, axes = plt.subplots(1, len(results), figsize=(10, 3.2))
    for ax, res in zip(axes, results):
        m = res["test_metrics"]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Reds", vmin=0, vmax=cm.sum())
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
        ax.set_xticks([0, 1], ["pred. inmaduro", "pred. maduro"], fontsize=8)
        ax.set_yticks([0, 1], ["inmaduro", "maduro"], fontsize=8)
        ax.set_title(res["name"], fontsize=10)
    _save(fig, out / "confusion_test.png")


def pca_variance(spectrum: dict, out: Path):
    ratio, cum = np.array(spectrum["ratio"]), np.array(spectrum["cumulative"])
    x = np.arange(1, len(ratio) + 1)
    fig, ax1 = plt.subplots(figsize=(7.5, 4))
    ax1.bar(x, ratio * 100, color="#e67e22", label="individual")
    for xi, ev in zip(x, spectrum["eigenvalues"]):
        ax1.text(xi, ratio[xi - 1] * 100 + 1, f"λ={ev:.2f}", ha="center", fontsize=7)
    ax1.set_xlabel("Componente principal")
    ax1.set_ylabel("Varianza explicada (%)")
    ax1.set_xticks(x)
    ax2 = ax1.twinx()
    ax2.plot(x, cum * 100, "o-", color="#2c3e50")
    ax2.axhline(config.PCA_VARIANCE_TARGET * 100, color="#2c3e50", ls="--", lw=1)
    ax2.axvline(spectrum["n_components"], color="#c0392b", ls=":", lw=1.5)
    ax2.set_ylabel("Varianza acumulada (%)")
    ax2.set_ylim(0, 105)
    ax2.spines["right"].set_visible(True)
    ax1.set_title(f"PCA sobre descriptores estandarizados (entrenamiento) → p = {spectrum['n_components']}")
    _save(fig, out / "pca_variance.png")

    load = spectrum["loadings"].iloc[: max(3, spectrum["n_components"])]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    im = ax.imshow(load.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(load.shape[1]), [AXIS_LABEL[c] for c in load.columns])
    ax.set_yticks(range(load.shape[0]), load.index)
    for i in range(load.shape[0]):
        for j in range(load.shape[1]):
            ax.text(j, i, f"{load.iat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title("Cargas (loadings) de las componentes")
    _save(fig, out / "pca_loadings.png")


def repeated_splits(rep: pd.DataFrame, out: Path):
    """Mean test accuracy (± SD) and pooled error counts over the repeated splits.

    Most splits end with perfect accuracy, so box plots collapse onto 1.0; means and
    pooled FP/FN counts show the differences that do exist.
    """
    order = list(STRATEGY_COLORS)
    sources = [("kmeans", "K-Means (sistema)"), ("reference_test_swap", "ideal solo en test, mismo modelo"),
               ("reference_fixed", "ideal, diseño fijo"), ("kmeans_linear_hue", "K-Means + H aritmético")]
    shades = {"kmeans": 0.9, "reference_test_swap": 0.3, "reference_fixed": 0.55, "kmeans_linear_hue": 0.6}
    hatches = {"kmeans": "", "reference_test_swap": "", "reference_fixed": "..", "kmeans_linear_hue": "//"}
    n_repeats = rep["repeat"].nunique()
    n_pred = int(rep[(rep.mask_source == "kmeans") & (rep.strategy == order[0])][["tn", "fp", "fn", "tp"]].to_numpy().sum())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(order))
    w = 0.2
    for k, (src, label) in enumerate(sources):
        sub = rep[rep.mask_source == src].groupby("strategy")
        mean, sd = sub["accuracy"].mean().loc[order], sub["accuracy"].std().loc[order]
        errors = (sub["fp"].sum() + sub["fn"].sum()).loc[order]
        colors = [STRATEGY_COLORS[s] for s in order]
        axes[0].bar(x + (k - 1.5) * w, mean, w, yerr=sd, capsize=3, color=colors, alpha=shades[src],
                    hatch=hatches[src], edgecolor="black", linewidth=0.5, label=label)
        bars = axes[1].bar(x + (k - 1.5) * w, errors, w, color=colors, alpha=shades[src], hatch=hatches[src],
                           edgecolor="black", linewidth=0.5, label=label)
        axes[1].bar_label(bars, fontsize=8, padding=2)
    axes[0].set_ylim(0.7, 1.05)
    axes[0].set_ylabel("Exactitud en test (media ± DE)")
    axes[1].set_ylabel(f"Errores acumulados (FP+FN) de {n_pred} predicciones")
    for ax in axes:
        ax.set_xticks(x, order, fontsize=9)
    handles = [matplotlib.patches.Patch(facecolor="#888888", alpha=shades[s], hatch=hatches[s], edgecolor="black",
                                        label=l) for s, l in sources]
    axes[1].legend(handles=handles, frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"{n_repeats} particiones 60/20/20 independientes (protocolo completo en cada una)", y=1.0)
    _save(fig, out / "repeated_splits.png")


def integration_scatter(per_image: pd.DataFrame, strategy_name: str, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = per_image["class"].map({"ripe": RIPE, "unripe": UNRIPE})
    axes[0].scatter(per_image["jaccard"], per_image["feature_shift"], c=colors, s=22)
    axes[0].set_xlabel("Jaccard de la máscara K-Means")
    axes[0].set_ylabel("‖x(K-Means) − x(referencia)‖ (z-score)")
    axes[0].set_title("Peor máscara → descriptores más contaminados")
    test = per_image[per_image.split == "test"]
    marker_ok = test["correct_all"]
    axes[1].scatter(test.loc[marker_ok, "jaccard"], test.loc[marker_ok, "llr_best"], c=colors[test.index][marker_ok],
                    s=40, marker="o")
    axes[1].scatter(test.loc[~marker_ok, "jaccard"], test.loc[~marker_ok, "llr_best"], c=colors[test.index][~marker_ok],
                    s=70, marker="X")
    for marker, label in (("o", "acierto (3 estrategias)"), ("X", "≥1 estrategia falla")):
        axes[1].scatter([], [], c="#555555", marker=marker, label=label)
    axes[1].scatter([], [], c=RIPE, marker="s", label="maduro")
    axes[1].scatter([], [], c=UNRIPE, marker="s", label="inmaduro")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[1].set_xlabel("Jaccard de la máscara K-Means")
    axes[1].set_ylabel(f"ln Λ(x) — {strategy_name}")
    axes[1].set_title("Imágenes de test")
    axes[1].legend(frameon=False, fontsize=8)
    _save(fig, out / "integration.png")
