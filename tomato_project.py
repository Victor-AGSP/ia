"""Proyecto 1 (INFO1185): K-Means tomato segmentation and Bayesian ripeness classification.

Pipeline (every decision uses train/validation only; the test images never feed a decision):

1. Load images + reference masks, remove byte-identical duplicates, stratified image-level
   split 60/20/20 with a fixed seed.
2. Exploratory analysis on training pixels (fruit vs background, ripe vs unripe).
3. K-Means (k=2) for R, G, B, RG, RB, GB, RGB; automatic fruit-cluster selection;
   Jaccard against the reference mask; combination and post-processing chosen on
   development (train + validation) Jaccard.
4. Fruit descriptors inside the estimated mask: mean R, G, B, circular mean H, S, V, a*, b*.
5. Three Gaussian Bayes classifiers with a likelihood-ratio decision rule:
   analysis-based subset, SFS subset, and PCA components. Threshold by Youden on validation.
6. Final test evaluation with bootstrap CIs, plus declared complementary analyses that do
   not modify the system: a controlled mask ablation and a repeated-split study that
   re-runs the whole protocol (segmentation choice included) on additional partitions.

Usage:  python tomato_project.py [--data data] [--out results] [--quick]
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from tomato import config, plots, reporting
from tomato.classification import design_strategies, evaluate_strategy, pca_transform
from tomato.data import load_records, resolve_data_root, split_records
from tomato.features import circular_mean_deg, extract_features
from tomato.segmentation import postprocess, segment_dataset

F = config.FEATURES


def feature_matrix(records, masks: dict[int, np.ndarray], linear_hue: bool = False) -> pd.DataFrame:
    rows = []
    for i, rec in enumerate(records):
        row = {"index": i, "name": rec["name"], "class": rec["class_name"], "label": rec["label"]}
        row.update(extract_features(rec["image"], masks[i], linear_hue=linear_hue))
        rows.append(row)
    return pd.DataFrame(rows)


def indices(split: np.ndarray):
    return (np.flatnonzero(split == "train"), np.flatnonzero(split == "validation"),
            np.flatnonzero(split == "test"))


def evaluate_with_design(design, X, y, split, X_test=None, with_ci=False):
    """Evaluate the strategies of an existing design (same feature subsets / number of PCs)."""
    tr, va, te = indices(split)
    return [evaluate_strategy(name, feats, fit, X, y, tr, va, te, with_ci=with_ci, X_test=X_test)
            for name, (feats, fit) in design["specs"].items()]


def run_all_strategies(X, y, split, seed=config.SEED, with_ci=False):
    tr, _, _ = indices(split)
    design = design_strategies(X[tr], y[tr], F, seed)
    return design, evaluate_with_design(design, X, y, split, with_ci=with_ci)


def choose_segmentation(seg: pd.DataFrame, split: np.ndarray) -> tuple[str, bool]:
    """Best combination by mean development Jaccard, and whether post-processing helps it."""
    dev = seg[np.isin(seg["index"], np.flatnonzero(split != "test"))]
    means = dev.groupby("combination")[["jaccard", "jaccard_post"]].mean()
    combo = str(means["jaccard"].idxmax())
    return combo, bool(means.loc[combo, "jaccard_post"] > means.loc[combo, "jaccard"])


def paired_changes(base, other, y, test_idx):
    """Per strategy, test errors of ``base`` that ``other`` corrects and errors it introduces."""
    out = []
    for b, o in zip(base, other):
        wrong_b = (b["test_scores"] >= b["threshold"]).astype(int) != y[test_idx]
        wrong_o = (o["test_scores"] >= o["threshold"]).astype(int) != y[test_idx]
        out.append({"corrected": int(np.sum(wrong_b & ~wrong_o)), "introduced": int(np.sum(~wrong_b & wrong_o))})
    return out


def variant_rows(variants: dict, base, y, test_idx, extra: dict) -> list[dict]:
    rows = []
    for source, res in variants.items():
        for r, change in zip(res, paired_changes(base, res, y, test_idx)):
            rows.append({**extra, "mask_source": source, "strategy": r["name"],
                         "features": ",".join(r["features"]), **change, **r["test_metrics"]})
    return rows


def summarise_segmentation(seg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for combo in config.COMBINATIONS:
        s = seg[seg.combination == combo]
        dev, test = s[s.split != "test"], s[s.split == "test"]
        rows.append({
            "combination": combo,
            "train_mean": s[s.split == "train"]["jaccard"].mean(),
            "val_mean": s[s.split == "validation"]["jaccard"].mean(),
            "dev_mean": dev["jaccard"].mean(), "dev_std": dev["jaccard"].std(),
            "dev_median": dev["jaccard"].median(),
            "dev_post_mean": dev["jaccard_post"].mean(),
            "dev_oracle_mean": dev["jaccard_oracle_cluster"].mean(),
            "dev_heuristic": dev["jaccard_heuristic"].mean(),
            "dev_min_border": dev["jaccard_min_border"].mean(),
            "dev_max_saturation": dev["jaccard_max_saturation"].mean(),
            "dev_frac_below_03": (dev["jaccard"] < 0.3).mean(),
            "test_mean": test["jaccard"].mean(), "test_std": test["jaccard"].std(),
            "test_post_mean": test["jaccard_post"].mean(),
            "iter_mean": s["n_iter"].mean(), "iter_max": s["n_iter"].max(),
        })
    return pd.DataFrame(rows)


def pixel_summary(pixels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cls, region), g in pixels.groupby(["class", "region"]):
        row = {"class": cls, "region": region}
        for feat in F:
            row[feat] = circular_mean_deg(g[feat].to_numpy()) if feat == "H" else float(g[feat].mean())
            row[f"{feat}_std"] = float(g[feat].std())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=None, help="dataset root (default: data/ or dataset/)")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--quick", action="store_true", help="3 repeated splits instead of 30 (smoke test)")
    args = parser.parse_args()
    n_repeats = 3 if args.quick else config.N_REPEATED_SPLITS
    t0 = time.time()
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # 1. Data --------------------------------------------------------------------------
    root = resolve_data_root(args.data)
    records, duplicates = load_records(root)
    labels = np.array([r["label"] for r in records])
    y = labels
    split = split_records(labels)
    tr, va, te = indices(split)
    dataset_df = pd.DataFrame([{
        "index": i, "name": r["name"], "class": r["class_name"], "split": split[i], "width": r["width"],
        "height": r["height"], "channels": r["channels"], "reference_area": float(r["reference_mask"].mean()),
    } for i, r in enumerate(records)])
    dataset_df.to_csv(out / "dataset_split.csv", index=False)
    pd.DataFrame(duplicates).to_csv(out / "duplicates_removed.csv", index=False)
    print(f"[data] {len(records)} images ({len(duplicates)} duplicates removed) from {root}")

    # 2. Exploratory analysis (training pixels only) -----------------------------------
    pixels = plots.sample_pixels(records, tr, config.SEED)
    pix_summary = pixel_summary(pixels)
    pix_summary.to_csv(out / "eda_pixel_summary.csv", index=False)
    plots.pixel_histograms(pixels, out)
    plots.dataset_montage(records, split, out)

    # 3. Segmentation -----------------------------------------------------------------
    seg_rows, masks = segment_dataset(records, split)
    seg = pd.DataFrame(seg_rows)
    seg.to_csv(out / "segmentation_per_image.csv", index=False)
    seg_summary = summarise_segmentation(seg)
    seg_summary.to_csv(out / "segmentation_summary.csv", index=False)
    ranked = seg_summary.sort_values("dev_mean", ascending=False).reset_index(drop=True)
    best, use_post = choose_segmentation(seg, split)
    best_row = seg_summary.set_index("combination").loc[best]
    plots.segmentation_bars(seg, out)
    plots.segmentation_examples(records, masks, seg, best, out)
    plots.segmentation_all_combinations(records, masks, seg, out)
    print(f"[seg] best={best} dev J={best_row['dev_mean']:.3f} post={best_row['dev_post_mean']:.3f} use_post={use_post}")

    # 4. Features ---------------------------------------------------------------------
    cache: dict[tuple, pd.DataFrame] = {}

    def kmeans_features(combo: str, post: bool, linear_hue: bool = False) -> pd.DataFrame:
        key = (combo, post, linear_hue)
        if key not in cache:
            chosen = {i: postprocess(masks[(i, combo)]) if post else masks[(i, combo)] for i in range(len(records))}
            cache[key] = feature_matrix(records, chosen, linear_hue)
        return cache[key]

    feats = {
        "kmeans": kmeans_features(best, use_post),
        "kmeans_raw": kmeans_features(best, False),
        "reference": feature_matrix(records, {i: r["reference_mask"] for i, r in enumerate(records)}),
        # Ablation only: naive arithmetic hue mean (the 0/360 wrap turns red into spurious hues).
        "kmeans_linear_hue": kmeans_features(best, use_post, linear_hue=True),
    }
    for name, df in feats.items():
        df.assign(split=split).to_csv(out / f"features_{name}.csv", index=False)
    X = feats["kmeans"][F].to_numpy(float)
    Xref = feats["reference"][F].to_numpy(float)
    plots.feature_distributions(feats["kmeans"].iloc[tr], out)

    # 5. Classification (main split) --------------------------------------------------
    design, results = run_all_strategies(X, y, split, with_ci=True)
    design["separability"].to_csv(out / "separability_train.csv", index=False)
    design["trace"].to_csv(out / "sfs_trace.csv", index=False)
    design["spectrum"]["loadings"].to_csv(out / "pca_loadings.csv")
    plots.correlation_heatmap(design["corr"], out)
    plots.sfs_trace(design["trace"], out)
    plots.threshold_sweeps(results, out)
    plots.roc_curves(results, out)
    plots.confusion_matrices(results, out)
    plots.pca_variance(design["spectrum"], out)
    for res in results:
        res["sweep"].to_csv(out / f"threshold_sweep_{res['name'].replace(' + ', '_').replace(' ', '_')}.csv", index=False)
    metrics = pd.DataFrame([{"strategy": r["name"], "split": part, "features": ",".join(r["features"]),
                             **r[f"{part}_metrics"]} for part in ("val", "test") for r in results])
    metrics.to_csv(out / "classification_metrics.csv", index=False)

    # Supplementary diagnostic (validation only, not used for any decision).
    pca_by_p = []
    for p in range(1, len(F) + 1):
        r = evaluate_strategy(f"PCA{p}", [], pca_transform(p), X, y, tr, va, te, evaluate_test=False)
        pca_by_p.append({"p": p, "cumulative": design["spectrum"]["cumulative"][p - 1],
                         "val_auc": r["val_metrics"]["auc"], "val_accuracy": r["val_metrics"]["accuracy"]})
    pca_by_p = pd.DataFrame(pca_by_p)
    pca_by_p.to_csv(out / "pca_validation_by_p.csv", index=False)
    print("[clf] " + " | ".join(f"{r['name']}: {r['features']} test AUC={r['test_metrics']['auc']:.3f} "
                                  f"acc={r['test_metrics']['accuracy']:.3f}" for r in results))

    # 6. Segmentation quality vs classification (controlled ablation) ------------------
    # reference_test_swap: same features, model and threshold; only the test descriptors
    #   come from the reference mask -> isolates the effect of test-time segmentation.
    # reference_fixed: same feature subsets / number of PCs, model and threshold refitted
    #   on reference-mask descriptors -> isolates mask quality with the selection fixed.
    # reference_redesign: selection repeated on reference descriptors (not a controlled
    #   comparison: mask, selection, model and threshold all change).
    variants = {
        "kmeans": results,
        "kmeans_raw_fixed": evaluate_with_design(design, feats["kmeans_raw"][F].to_numpy(float), y, split),
        "reference_test_swap": evaluate_with_design(design, X, y, split, X_test=Xref),
        "reference_fixed": evaluate_with_design(design, Xref, y, split),
        "reference_redesign": run_all_strategies(Xref, y, split)[1],
        "kmeans_linear_hue": run_all_strategies(feats["kmeans_linear_hue"][F].to_numpy(float), y, split)[1],
    }
    mask_effect = pd.DataFrame(variant_rows(variants, results, y, te, {}))
    mask_effect.to_csv(out / "mask_effect_test.csv", index=False)

    sd = Xref[tr].std(axis=0, ddof=1)
    final_j = seg[seg.combination == best]["jaccard_post" if use_post else "jaccard"].to_numpy()
    shift = np.linalg.norm((X - Xref) / sd, axis=1)
    rho, p_rho = spearmanr(final_j, shift)
    # Reference strategy for the per-image plot: best validation AUC (never test).
    ref_res = max(results, key=lambda r: (r["val_metrics"]["auc"], r["val_metrics"]["balanced_accuracy"]))
    per_image = pd.DataFrame({"name": [r["name"] for r in records], "class": [r["class_name"] for r in records],
                              "split": split, "jaccard": final_j, "feature_shift": shift})
    llr_best = np.full(len(records), np.nan)
    llr_best[te] = ref_res["test_scores"]
    per_image["llr_best"] = llr_best
    correct = np.ones(len(te), dtype=bool)
    for k, r in enumerate(results):
        ok = (r["test_scores"] >= r["threshold"]).astype(int) == y[te]
        per_image.loc[te, f"ok_{k}"] = ok
        correct &= ok
    per_image["correct_all"] = False
    per_image.loc[te, "correct_all"] = correct
    per_image.to_csv(out / "integration_per_image.csv", index=False)
    plots.integration_scatter(per_image, ref_res["name"], out)
    test_images = per_image.iloc[te].sort_values("jaccard")

    # 7. Robustness: the whole protocol on repeated additional splits -----------------
    rep_rows = []
    for r in range(n_repeats):
        seed_r = config.SEED + 1 + r
        s = split_records(labels, seed=seed_r)
        combo_r, post_r = choose_segmentation(seg, s)
        Xk_r = kmeans_features(combo_r, post_r)[F].to_numpy(float)
        design_r, res_k = run_all_strategies(Xk_r, y, s, seed=seed_r)
        variants_r = {
            "kmeans": res_k,
            "reference_test_swap": evaluate_with_design(design_r, Xk_r, y, s, X_test=Xref),
            "reference_fixed": evaluate_with_design(design_r, Xref, y, s),
            "reference_redesign": run_all_strategies(Xref, y, s, seed=seed_r)[1],
            "kmeans_linear_hue": run_all_strategies(
                kmeans_features(combo_r, post_r, linear_hue=True)[F].to_numpy(float), y, s, seed=seed_r)[1],
        }
        rep_rows += variant_rows(variants_r, res_k, y, np.flatnonzero(s == "test"),
                                 {"repeat": r, "combination": combo_r, "postprocess": post_r})
    rep = pd.DataFrame(rep_rows)
    rep.to_csv(out / "repeated_splits.csv", index=False)
    grouped = rep.groupby(["mask_source", "strategy"], sort=False)
    rep_summary = grouped[["auc", "accuracy", "sensitivity", "specificity"]].agg(["mean", "std"])
    rep_summary.columns = [f"{a}_{b}" for a, b in rep_summary.columns]
    rep_summary = rep_summary.join(grouped[["fp", "fn", "corrected", "introduced"]].sum()).reset_index()
    rep_summary.to_csv(out / "repeated_splits_summary.csv", index=False)
    # Subsets are counted as sets: "H,G,R" and "G,H,R" are the same selection.
    km_rep = rep[rep.mask_source == "kmeans"].assign(
        features=lambda d: d["features"].map(lambda f: ",".join(sorted(f.split(",")))))
    selection_freq = km_rep.groupby("strategy")["features"].value_counts().rename("count").reset_index()
    selection_freq.to_csv(out / "repeated_selection_frequency.csv", index=False)
    seg_choice = rep.drop_duplicates("repeat")[["repeat", "combination", "postprocess"]]
    seg_choice.to_csv(out / "repeated_segmentation_choice.csv", index=False)
    plots.repeated_splits(rep, out)

    # 8. Report artefacts --------------------------------------------------------------
    split_counts = {cls: {s: int(dataset_df[(dataset_df["class"] == cls) & (dataset_df.split == s)].shape[0])
                          for s in ("train", "validation", "test")} for cls in ("ripe", "unripe")}
    ctx = {
        "n_images_raw": len(records) + len(duplicates), "n_images": len(records), "duplicates": duplicates,
        "class_counts": dataset_df["class"].value_counts().to_dict(), "split_counts": split_counts,
        "n_split": {s: int((split == s).sum()) for s in ("train", "validation", "test")},
        "pixel_summary": pix_summary, "seg_summary": seg_summary, "best_combo": best,
        "best_dev_j": best_row["dev_mean"], "best_test_j": best_row["test_mean"],
        "best_dev_post_j": best_row["dev_post_mean"], "best_test_post_j": best_row["test_post_mean"],
        "best_dev_oracle_j": best_row["dev_oracle_mean"],
        "second_combo": ranked.loc[1, "combination"], "second_dev_j": ranked.loc[1, "dev_mean"],
        "worst_combo": ranked.iloc[-1]["combination"], "worst_dev_j": ranked.iloc[-1]["dev_mean"],
        "use_post": use_post, "separability": design["separability"], "corr": design["corr"],
        "analysis_features": design["specs"]["Bayes + análisis"][0], "sfs_features": design["specs"]["Bayes + SFS"][0],
        "sfs_trace": design["trace"], "spectrum": design["spectrum"], "results": results, "pca_by_p": pca_by_p,
        "mask_effect": mask_effect, "repeated_summary": rep_summary, "test_images": test_images,
        "repeated": rep, "selection_freq": selection_freq, "seg_choice": seg_choice,
        "iter_max": int(seg["n_iter"].max()), "test_low_j_correct": int(((per_image.iloc[te]["jaccard"] < 0.2)
                                                                           & per_image.iloc[te]["correct_all"]).sum()),
        "rho_shift": float(rho), "rho_shift_p": float(p_rho), "ref_strategy": ref_res["name"],
    }
    reporting.write_tables(out, ctx)
    reporting.write_macros(out, ctx)
    summary = {
        "seed": config.SEED, "config": {k: getattr(config, k) for k in dir(config) if k.isupper()},
        "data_root": str(root), "n_images_raw": ctx["n_images_raw"], "n_images": len(records),
        "duplicates": duplicates, "class_counts": ctx["class_counts"], "split_counts": split_counts,
        "dataset_dimensions": {"height": [int(dataset_df.height.min()), int(dataset_df.height.max())],
                               "width": [int(dataset_df.width.min()), int(dataset_df.width.max())],
                               "channels": dataset_df.channels.value_counts().to_dict()},
        "reference_area_by_class": dataset_df.groupby("class")["reference_area"].mean().to_dict(),
        "pixel_summary": pix_summary, "segmentation_summary": seg_summary, "best_segmentation": best,
        "use_postprocessing": use_post, "separability": design["separability"],
        "analysis_features": ctx["analysis_features"], "sfs_features": ctx["sfs_features"],
        "sfs_trace": design["trace"],
        "pca": {k: v for k, v in design["spectrum"].items() if k != "loadings"},
        "pca_loadings": design["spectrum"]["loadings"].reset_index().to_dict(orient="records"),
        "pca_validation_by_p": pca_by_p,
        "strategies": [{k: r[k] for k in ("name", "features", "threshold", "youden", "n_tied_thresholds",
                                          "val_metrics", "test_metrics", "test_ci")} for r in results],
        "mask_effect_test": mask_effect, "repeated_splits_summary": rep_summary,
        "repeated_selection_frequency": selection_freq, "repeated_segmentation_choice": seg_choice,
        "integration": {"spearman_jaccard_vs_feature_shift": float(rho), "p_value": float(p_rho),
                        "reference_strategy": ref_res["name"], "test_images": test_images},
        "runtime_seconds": round(time.time() - t0, 1),
    }
    reporting.write_json(out / "report_data.json", summary)
    print(f"[done] {time.time() - t0:.1f}s -> {out.resolve()}")


if __name__ == "__main__":
    main()
