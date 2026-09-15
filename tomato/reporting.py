"""LaTeX tables/macros and JSON generated from the results, so the report cannot go stale."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

TEX_FEATURE = {"R": r"$\bar R$", "G": r"$\bar G$", "B": r"$\bar B$", "H": r"$\bar H$", "S": r"$\bar S$",
               "V": r"$\bar V$", "a*": r"$\bar a^*$", "b*": r"$\bar b^*$"}
TEX_PIXEL = {"R": "R", "G": "G", "B": "B", "H": "H", "S": "S", "V": "V", "a*": "$a^*$", "b*": "$b^*$"}
SOURCE_ES = {
    "kmeans": "K-Means (sistema)",
    "kmeans_raw_fixed": "K-Means sin post-proc., diseño fijo",
    "reference_test_swap": "Ideal solo en test, mismo modelo",
    "reference_fixed": "Ideal, diseño fijo",
    "reference_redesign": "Ideal, rediseño completo",
    "kmeans_linear_hue": r"K-Means, $\bar H$ aritmético",
}
MASK_EFFECT_ORDER = ("kmeans", "kmeans_raw_fixed", "reference_test_swap", "reference_fixed",
                     "reference_redesign", "kmeans_linear_hue")
REPEATED_MACROS = (("kmeans", "Km"), ("reference_test_swap", "Swap"), ("reference_fixed", "Fix"),
                   ("reference_redesign", "Ref"), ("kmeans_linear_hue", "Lin"))
CLASS_ES = {"ripe": "Maduro", "unripe": "Inmaduro"}
SPLIT_ES = {"train": "Entrenamiento", "validation": "Validación", "test": "Test"}


def f(x, d: int = 3) -> str:
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def tex_features(names) -> str:
    return ", ".join(TEX_FEATURE.get(n, n) for n in names)


def escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%")


def _table(path: Path, colspec: str, header: str, rows: list[str], size: str = r"\small"):
    body = "\n".join(rows)
    path.write_text(
        f"{size}\n\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n",
        encoding="utf-8")


def write_tables(out: Path, ctx: dict):
    tex = out / "tex"
    tex.mkdir(exist_ok=True)

    # Dataset split
    counts = ctx["split_counts"]
    rows = []
    for cls in ("ripe", "unripe"):
        c = [counts[cls].get(s, 0) for s in ("train", "validation", "test")]
        rows.append(f"{CLASS_ES[cls]} & {c[0]} & {c[1]} & {c[2]} & {sum(c)} \\\\")
    tot = [sum(counts[k].get(s, 0) for k in counts) for s in ("train", "validation", "test")]
    n = sum(tot)
    rows += [r"\midrule", f"Total & {tot[0]} & {tot[1]} & {tot[2]} & {n} \\\\",
             f"Porcentaje & {100 * tot[0] / n:.1f}\\% & {100 * tot[1] / n:.1f}\\% & {100 * tot[2] / n:.1f}\\% & 100\\% \\\\"]
    _table(tex / "split.tex", "lrrr|r", r"Clase & Entrenamiento & Validación & Test & Total", rows)

    # Duplicates
    rows = [f"\\texttt{{{escape(d['removed'])}}} & \\texttt{{{escape(d['kept'])}}} & {CLASS_ES[d['class']]} \\\\"
            for d in ctx["duplicates"]] or [r"\multicolumn{3}{c}{sin duplicados} \\"]
    _table(tex / "duplicates.tex", "llr", r"Eliminada & Se conserva & Clase", rows)

    # Pixel summary
    ps = ctx["pixel_summary"]
    rows = []
    for cls in ("ripe", "unripe"):
        for region in ("fruto", "fondo"):
            r = ps[(ps["class"] == cls) & (ps["region"] == region)].iloc[0]
            vals = " & ".join(f(r[k], 3 if k in "SV" else 1) for k in config.FEATURES)
            rows.append(f"{CLASS_ES[cls]} & {region} & {vals} \\\\")
    _table(tex / "pixel_summary.tex", "ll" + "r" * len(config.FEATURES),
           "Clase & Región & " + " & ".join(TEX_PIXEL[k] for k in config.FEATURES), rows, r"\footnotesize")

    # Segmentation per combination
    seg = ctx["seg_summary"]
    best = ctx["best_combo"]
    rows = []
    for _, r in seg.iterrows():
        dev = f"{f(r['dev_mean'])} $\\pm$ {f(r['dev_std'])}"
        if r["combination"] == best:
            dev = f"\\textbf{{{f(r['dev_mean'])}}} $\\pm$ {f(r['dev_std'])}"
        rows.append(f"{r['combination']} & {f(r['train_mean'])} & {f(r['val_mean'])} & {dev} & {f(r['dev_post_mean'])} & "
                    f"{f(r['dev_oracle_mean'])} & {f(r['test_mean'])} $\\pm$ {f(r['test_std'])} & "
                    f"{f(r['test_post_mean'])} & {r['iter_mean']:.1f} / {int(r['iter_max'])} \\\\")
    _table(tex / "segmentation.tex", "lrrrrrrrr",
           r"Canales & Entr. & Valid. & Desarrollo & +Post & Oráculo & Test & Test +Post & Iter. (media/máx)",
           rows, r"\footnotesize")

    rows = [f"{r['combination']} & {f(r['dev_heuristic'])} & {f(r['dev_min_border'])} & "
            f"{f(r['dev_max_saturation'])} & {f(r['dev_oracle_mean'])} \\\\" for _, r in seg.iterrows()]
    _table(tex / "cluster_rules.tex", "lrrrr",
           r"Canales & Puntaje objeto & Mín. borde & Máx. saturación & Oráculo", rows, r"\footnotesize")

    # Separability
    sep = ctx["separability"]
    rows = []
    for _, r in sep.iterrows():
        mark = r"\checkmark" if r["feature"] in ctx["analysis_features"] else ""
        d = 3 if r["feature"] in ("S", "V") else 1
        rows.append(f"{TEX_FEATURE[r['feature']]} & {f(r['mean_ripe'], d)} $\\pm$ {f(r['std_ripe'], d)} & "
                    f"{f(r['mean_unripe'], d)} $\\pm$ {f(r['std_unripe'], d)} & {f(r['cohen_d'], 2)} & "
                    f"{f(r['auc_univariate'])} & {mark} \\\\")
    _table(tex / "separability.tex", "lrrrrc",
           r"Descriptor & Maduro & Inmaduro & $d$ de Cohen & AUC univ. & Selección", rows, r"\footnotesize")

    # Correlation among the selected features
    corr = ctx["corr"]
    rows = [TEX_FEATURE[a] + " & " + " & ".join(f(corr.loc[a, b], 2) for b in config.FEATURES) + r" \\"
            for a in config.FEATURES]
    _table(tex / "correlation.tex", "l" + "r" * len(config.FEATURES),
           " & " + " & ".join(TEX_FEATURE[b] for b in config.FEATURES), rows, r"\scriptsize")

    # SFS
    rows = [f"{int(r['step'])} & {TEX_FEATURE[r['added']]} & {f(r['cv_auc'])} $\\pm$ {f(r['cv_auc_std'])} & "
            f"{r'\checkmark' if r['selected'] else ''} \\\\" for _, r in ctx["sfs_trace"].iterrows()]
    _table(tex / "sfs.tex", "clrc", r"Paso & Se añade & AUC CV (media $\pm$ DE) & Tamaño elegido", rows)

    # Metrics
    rows = []
    for part, label in (("val", "Validación"), ("test", "Test")):
        for res in ctx["results"]:
            m = res[f"{part}_metrics"]
            rows.append(f"{res['name']} & {label} & {m['threshold']:.2f} & {f(m['auc'])} & {f(m['accuracy'])} & "
                        f"{f(m['sensitivity'])} & {f(m['specificity'])} & {f(m['f1'])} & "
                        f"{m['tp']}/{m['fn']}/{m['tn']}/{m['fp']} \\\\")
        if part == "val":
            rows.append(r"\midrule")
    _table(tex / "metrics.tex", "llrrrrrrc",
           r"Estrategia & Conjunto & $\ln\theta^*$ & AUC & Exact. & Sens. & Espec. & F1 & VP/FN/VN/FP", rows,
           r"\footnotesize")

    rows = []
    for res in ctx["results"]:
        ci = res["test_ci"]
        rows.append(f"{res['name']} & " + " & ".join(f"[{f(ci[k][0], 2)}, {f(ci[k][1], 2)}]"
                                                      for k in ("auc", "accuracy", "sensitivity", "specificity")) + r" \\")
    _table(tex / "bootstrap.tex", "lrrrr", r"Estrategia & AUC & Exactitud & Sensibilidad & Especificidad", rows,
           r"\footnotesize")

    rows = [f"{res['name']} & {tex_features(res['features'])} & {res['threshold']:.2f} & {f(res['youden'], 3)} & "
            f"{res['n_tied_thresholds']} \\\\" for res in ctx["results"]]
    _table(tex / "operating_points.tex", "llrrc",
           r"Estrategia & Entradas & $\ln\theta^*$ & $J$ en validación & Umbrales empatados", rows, r"\footnotesize")

    # PCA
    sp = ctx["spectrum"]
    rows = [f"PC{i + 1} & {f(ev)} & {100 * r:.2f}\\% & {100 * c:.2f}\\% \\\\"
            for i, (ev, r, c) in enumerate(zip(sp["eigenvalues"], sp["ratio"], sp["cumulative"]))]
    _table(tex / "pca.tex", "crrr", r"Componente & Valor propio $\lambda$ & Varianza & Acumulada", rows)

    rows = [f"{int(r['p'])} & {100 * r['cumulative']:.1f}\\% & {f(r['val_auc'])} & {f(r['val_accuracy'])} \\\\"
            for _, r in ctx["pca_by_p"].iterrows()]
    _table(tex / "pca_by_p.tex", "crrr", r"$p$ & Var. acumulada & AUC valid. & Exact. valid.", rows, r"\footnotesize")

    # Integration (main split)
    src_es = SOURCE_ES
    rows = []
    for src in MASK_EFFECT_ORDER:
        sub = ctx["mask_effect"][ctx["mask_effect"].mask_source == src]
        cells = " & ".join(f"{f(r['auc'])} / {f(r['accuracy'])}" for _, r in sub.iterrows())
        rows.append(f"{src_es[src]} & {cells} \\\\")
    _table(tex / "mask_effect.tex", "lrrr",
           "Variante & " + " & ".join(r["name"] for r in ctx["results"]), rows,
           r"\footnotesize")

    rep = ctx["repeated_summary"]
    rows, previous = [], None
    for _, r in rep.iterrows():
        if previous is not None and r["mask_source"] != previous:
            rows.append(r"\midrule")
        previous = r["mask_source"]
        rows.append(f"{src_es[r['mask_source']]} & {r['strategy']} & {f(r['auc_mean'])} $\\pm$ {f(r['auc_std'])} & "
                    f"{f(r['accuracy_mean'])} $\\pm$ {f(r['accuracy_std'])} & {int(r['fp'] + r['fn'])} & "
                    f"{int(r['corrected'])} / {int(r['introduced'])} \\\\")
    _table(tex / "repeated.tex", "llrrrr",
           r"Variante & Estrategia & AUC test & Exactitud test & Errores & Corrige / introduce", rows,
           r"\footnotesize")

    rows = []
    for _, r in ctx["test_images"].iterrows():
        ok = " & ".join(r"\checkmark" if r[f"ok_{k}"] else r"$\times$" for k in range(len(ctx["results"])))
        rows.append(f"\\texttt{{{escape(r['name'])}}} & {CLASS_ES[r['class']]} & {f(r['jaccard'])} & "
                    f"{f(r['feature_shift'], 2)} & {ok} \\\\")
    _table(tex / "test_images.tex", "llrrccc",
           r"Imagen & Clase & Jaccard & Desplaz. $\mathbf{x}$ & " + " & ".join(
               r["name"].replace("Bayes + ", "B+").replace("PCA + Bayes", "PCA+B") for r in ctx["results"]),
           rows, r"\scriptsize")


def write_macros(out: Path, ctx: dict):
    res = {r["name"]: r for r in ctx["results"]}
    macros = {
        "NImagesRaw": ctx["n_images_raw"], "NImages": ctx["n_images"], "NDuplicates": len(ctx["duplicates"]),
        "NRipe": ctx["class_counts"]["ripe"], "NUnripe": ctx["class_counts"]["unripe"],
        "NTrain": ctx["n_split"]["train"], "NVal": ctx["n_split"]["validation"], "NTest": ctx["n_split"]["test"],
        "Seed": config.SEED, "MaxSide": config.MAX_SIDE, "PixelSample": config.PIXEL_SAMPLE,
        "BestCombo": ctx["best_combo"], "BestDevJ": f(ctx["best_dev_j"]), "BestTestJ": f(ctx["best_test_j"]),
        "BestDevPostJ": f(ctx["best_dev_post_j"]), "BestTestPostJ": f(ctx["best_test_post_j"]),
        "BestDevOracleJ": f(ctx["best_dev_oracle_j"]),
        "SecondCombo": ctx["second_combo"], "SecondDevJ": f(ctx["second_dev_j"]),
        "WorstCombo": ctx["worst_combo"], "WorstDevJ": f(ctx["worst_dev_j"]),
        "UsePost": "sí" if ctx["use_post"] else "no",
        "AnalysisFeatures": tex_features(ctx["analysis_features"]),
        "SFSFeatures": tex_features(ctx["sfs_features"]),
        "PCAn": ctx["spectrum"]["n_components"],
        "PCAcum": f"{100 * ctx['spectrum']['cumulative'][ctx['spectrum']['n_components'] - 1]:.1f}",
        "RhoShift": f(ctx["rho_shift"], 2), "RhoShiftP": f"{ctx['rho_shift_p']:.1e}",
        "NRepeats": config.N_REPEATED_SPLITS, "NBoot": config.N_BOOTSTRAP,
        "RefStrategy": ctx["ref_strategy"],
    }
    for key, name in (("An", "Bayes + análisis"), ("Sfs", "Bayes + SFS"), ("Pca", "PCA + Bayes")):
        for part in ("val", "test"):
            m = res[name][f"{part}_metrics"]
            P = part.capitalize()
            macros[f"{key}{P}AUC"] = f(m["auc"])
            macros[f"{key}{P}Acc"] = f(m["accuracy"])
            macros[f"{key}{P}Sens"] = f(m["sensitivity"])
            macros[f"{key}{P}Spec"] = f(m["specificity"])
        macros[f"{key}Thr"] = f"{res[name]['threshold']:.2f}"
        rep = ctx["repeated_summary"]
        for src, S in REPEATED_MACROS:
            r = rep[(rep.strategy == name) & (rep.mask_source == src)].iloc[0]
            macros[f"{key}Rep{S}AUC"] = f"{f(r['auc_mean'])} $\\pm$ {f(r['auc_std'])}"
            macros[f"{key}Rep{S}Acc"] = f"{f(r['accuracy_mean'])} $\\pm$ {f(r['accuracy_std'])}"
            macros[f"{key}Rep{S}Err"] = int(r["fp"] + r["fn"])
            macros[f"{key}Rep{S}Corr"] = int(r["corrected"])
            macros[f"{key}Rep{S}New"] = int(r["introduced"])
    choice = ctx["seg_choice"]
    counts = choice["combination"].value_counts()
    macros["RepComboFreq"] = ", ".join(f"{c} en {n}" for c, n in counts.items())
    macros["RepPostCount"] = int(choice["postprocess"].sum())
    macros["AuthorLine"] = f"{config.AUTHORS}\\\\" if config.AUTHORS else ""
    macros["NRepPredictions"] = int(ctx["repeated"].query("mask_source == 'kmeans' and strategy == 'Bayes + SFS'")
                                    [["tn", "fp", "fn", "tp"]].to_numpy().sum())
    # Subset frequencies are counted as sets (selection_freq is already order-independent).
    freq = ctx["selection_freq"]
    canon = lambda feats: ",".join(sorted(feats))
    for key, name, main in (("An", "Bayes + análisis", ctx["analysis_features"]),
                            ("Sfs", "Bayes + SFS", ctx["sfs_features"])):
        g = freq[freq.strategy == name].sort_values("count", ascending=False)
        macros[f"{key}DistinctSubsets"] = len(g)
        macros[f"{key}TopSubset"] = tex_features(g.iloc[0]["features"].split(","))
        macros[f"{key}TopSubsetCount"] = int(g.iloc[0]["count"])
        macros[f"{key}MainSubsetCount"] = int(g[g.features == canon(main)]["count"].sum())
    sfs = freq[freq.strategy == "Bayes + SFS"]
    macros["SfsOnlyHCount"] = int(sfs[sfs.features == "H"]["count"].sum())
    macros["SfsWithHCount"] = int(sfs[sfs.features.str.split(",").map(lambda fs: "H" in fs)]["count"].sum())
    # Paired comparison of test accuracy per repeated split (wins / ties / losses of PCA).
    acc = ctx["repeated"].query("mask_source == 'kmeans'").pivot(index="repeat", columns="strategy", values="accuracy")
    for key, name in (("An", "Bayes + análisis"), ("Sfs", "Bayes + SFS")):
        d = acc["PCA + Bayes"] - acc[name]
        macros[f"PcaVs{key}"] = f"{int((d > 0).sum())}/{int((d == 0).sum())}/{int((d < 0).sum())}"
    # Theoretical threshold ln(theta) = 0 on validation (diagnostic only).
    labels = {"Bayes + análisis": "análisis", "Bayes + SFS": "SFS", "PCA + Bayes": "PCA"}
    zero = [f"{labels[r['name']]} ({r['val_metrics_zero']['fp']} FP, {r['val_metrics_zero']['fn']} FN)"
            for r in ctx["results"] if r["val_metrics_zero"]["fp"] + r["val_metrics_zero"]["fn"] > 0]
    ok = [labels[r["name"]] for r in ctx["results"] if r["val_metrics_zero"]["fp"] + r["val_metrics_zero"]["fn"] == 0]
    macros["ZeroThrErrors"] = ", ".join(zero) if zero else "ninguna estrategia"
    macros["ZeroThrOk"] = ", ".join(ok) if ok else "ninguna"
    seg = ctx["seg_summary"]
    macros["IterMax"] = ctx["iter_max"]
    macros["HeurBestCount"] = int((seg["dev_heuristic"] >= seg[["dev_min_border", "dev_max_saturation"]].max(axis=1)).sum())
    macros["PostDevBetterCount"] = int((seg["dev_post_mean"] > seg["dev_mean"]).sum())
    macros["TestLowJCorrect"] = ctx["test_low_j_correct"]
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    (out / "tex" / "values.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def write_json(path: Path, obj):
    path.write_text(json.dumps(to_jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")
