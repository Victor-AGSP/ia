"""Feature selection, Gaussian Bayes likelihood-ratio classifiers and operating-point selection."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from . import config

# Label convention: positive class 1 = ripe ("maduro"), negative class 0 = unripe.


# --------------------------------------------------------------------------- Bayes rule
def log_likelihood_ratio(clf: GaussianNB, Z: np.ndarray) -> np.ndarray:
    """ln Lambda(x) = ln p(x | ripe) - ln p(x | unripe) under the Gaussian naive model.

    ``predict_joint_log_proba`` returns ln P(w) + ln p(x | w); the priors are removed so
    the score is a pure likelihood ratio and the prior/cost information lives entirely in
    the threshold: decide ripe  <=>  ln Lambda(x) >= ln theta.
    """
    assert list(clf.classes_) == [0, 1]
    joint = clf.predict_joint_log_proba(Z)
    log_prior = np.log(clf.class_prior_)
    return (joint[:, 1] - log_prior[1]) - (joint[:, 0] - log_prior[0])


def confusion_counts(y: np.ndarray, scores: np.ndarray, threshold: float):
    pred = scores >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    fp = int(np.sum(pred & (y == 0)))
    return tn, fp, fn, tp


def metrics_at(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    tn, fp, fn, tp = confusion_counts(y, scores, threshold)
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    prec = tp / max(1, tp + fp)
    return {
        "threshold": float(threshold),
        "auc": float(roc_auc_score(y, scores)),
        "accuracy": (tp + tn) / len(y),
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": (sens + spec) / 2,
        "precision": prec,
        "f1": 2 * prec * sens / max(1e-12, prec + sens),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    """Midpoints between consecutive distinct scores, plus one threshold beyond each end.

    Midpoints transfer better to unseen data than thresholds sitting exactly on a
    validation score.
    """
    s = np.unique(scores)
    return np.concatenate(([s[0] - 1.0], (s[:-1] + s[1:]) / 2.0, [s[-1] + 1.0]))


def threshold_sweep(y: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    rows = [metrics_at(y, scores, t) for t in candidate_thresholds(scores)]
    df = pd.DataFrame(rows)
    df["fpr"] = 1 - df["specificity"]
    df["youden"] = df["sensitivity"] - df["fpr"]
    return df


def youden_operating_point(y: np.ndarray, scores: np.ndarray) -> dict:
    """Threshold maximising J = sensitivity + specificity - 1.

    With a small validation set several thresholds tie; the centre of the tied set is
    taken so the decision boundary is not placed right against a validation sample.
    """
    sweep = threshold_sweep(y, scores)
    best = np.flatnonzero(np.isclose(sweep["youden"], sweep["youden"].max()))
    pos = int(best[len(best) // 2])
    return {
        "threshold": float(sweep.loc[pos, "threshold"]),
        "youden": float(sweep.loc[pos, "youden"]),
        "n_tied_thresholds": int(len(best)),
        "sweep": sweep,
    }


def bootstrap_ci(y: np.ndarray, scores: np.ndarray, threshold: float, seed: int = config.SEED) -> dict:
    """Stratified percentile bootstrap (95 %) for test metrics at a fixed threshold."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    samples = {k: [] for k in ("auc", "accuracy", "sensitivity", "specificity")}
    for _ in range(config.N_BOOTSTRAP):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        m = metrics_at(y[idx], scores[idx], threshold)
        for k in samples:
            samples[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in samples.items()}


# --------------------------------------------------------------------- feature analysis
def separability_table(X: np.ndarray, y: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Per-feature class statistics on the training images."""
    rows = []
    for j, name in enumerate(names):
        ripe, unripe = X[y == 1, j], X[y == 0, j]
        pooled = np.sqrt(((len(ripe) - 1) * ripe.var(ddof=1) + (len(unripe) - 1) * unripe.var(ddof=1))
                         / (len(ripe) + len(unripe) - 2))
        auc = roc_auc_score(y, X[:, j])
        rows.append({
            "feature": name,
            "mean_ripe": ripe.mean(), "std_ripe": ripe.std(ddof=1),
            "mean_unripe": unripe.mean(), "std_unripe": unripe.std(ddof=1),
            "cohen_d": (ripe.mean() - unripe.mean()) / pooled,
            "auc_univariate": max(auc, 1 - auc),
            "direction": "mayor en maduro" if auc >= 0.5 else "mayor en inmaduro",
        })
    return pd.DataFrame(rows)


def analysis_based_selection(table: pd.DataFrame, corr: pd.DataFrame) -> list[str]:
    """Rank features by |Cohen d| and keep the strongest ones that are not redundant.

    A candidate is skipped when |Pearson r| with an already selected feature exceeds
    ANALYSIS_MAX_ABS_CORR: two almost collinear features violate the naive independence
    assumption and double count the same evidence.
    """
    selected: list[str] = []
    for name in table.assign(abs_d=table["cohen_d"].abs()).sort_values("abs_d", ascending=False)["feature"]:
        if all(abs(corr.loc[name, s]) <= config.ANALYSIS_MAX_ABS_CORR for s in selected):
            selected.append(name)
        if len(selected) == config.ANALYSIS_MAX_FEATURES:
            break
    return selected


def cv_auc(X: np.ndarray, y: np.ndarray, seed: int = config.SEED) -> tuple[float, float]:
    cv = RepeatedStratifiedKFold(n_splits=config.SFS_FOLDS, n_repeats=config.SFS_REPEATS, random_state=seed)
    scores = []
    for tr, te in cv.split(X, y):
        clf = GaussianNB().fit(X[tr], y[tr])
        scores.append(roc_auc_score(y[te], log_likelihood_ratio(clf, X[te])))
    return float(np.mean(scores)), float(np.std(scores))


def sequential_forward_selection(X: np.ndarray, y: np.ndarray, names: list[str], seed: int = config.SEED):
    """Wrapper SFS with repeated stratified CV-AUC of the same Bayes classifier.

    The greedy search runs until every feature is added so the whole trace is visible;
    the final subset is the smallest prefix whose CV-AUC is within SFS_TOLERANCE of the
    best prefix (parsimony when the criterion saturates).
    """
    remaining, selected, trace = list(range(X.shape[1])), [], []
    while remaining:
        scored = [(*cv_auc(X[:, selected + [j]], y, seed), j) for j in remaining]
        mean, std, best_j = max(scored, key=lambda t: (t[0], -t[2]))
        selected.append(best_j)
        remaining.remove(best_j)
        trace.append({"step": len(selected), "added": names[best_j],
                      "subset": ",".join(names[k] for k in selected), "cv_auc": mean, "cv_auc_std": std})
    trace_df = pd.DataFrame(trace)
    best = trace_df["cv_auc"].max()
    k = int(trace_df.index[trace_df["cv_auc"] >= best - config.SFS_TOLERANCE][0]) + 1
    trace_df["selected"] = trace_df["step"] == k
    return [names[j] for j in selected[:k]], trace_df


# --------------------------------------------------------------------------- strategies
Transform = Callable[[np.ndarray], np.ndarray]


def subset_transform(feature_idx: list[int]) -> Callable[[np.ndarray], Transform]:
    # Gaussian naive Bayes is (up to GaussianNB's var_smoothing) invariant to per-feature
    # affine scaling, so no scaler.
    return lambda X_fit: (lambda X: X[:, feature_idx])


def pca_transform(n_components: int) -> Callable[[np.ndarray], Transform]:
    def fit(X_fit: np.ndarray) -> Transform:
        scaler = StandardScaler().fit(X_fit)
        pca = PCA(n_components=n_components, svd_solver="full").fit(scaler.transform(X_fit))
        return lambda X: pca.transform(scaler.transform(X))
    return fit


def pca_spectrum(X_train: np.ndarray, names: list[str]) -> dict:
    scaler = StandardScaler().fit(X_train)
    pca = PCA(svd_solver="full").fit(scaler.transform(X_train))
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    return {
        "eigenvalues": pca.explained_variance_.tolist(),
        "ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative": cumulative.tolist(),
        "n_components": int(np.searchsorted(cumulative, config.PCA_VARIANCE_TARGET) + 1),
        "loadings": pd.DataFrame(pca.components_, columns=names,
                                 index=[f"PC{i + 1}" for i in range(len(names))]),
    }


def design_strategies(X_train: np.ndarray, y_train: np.ndarray, names: list[str], seed: int = config.SEED) -> dict:
    """All design decisions, taken with training images only."""
    table = separability_table(X_train, y_train, names)
    corr = pd.DataFrame(X_train, columns=names).corr()
    analysis = analysis_based_selection(table, corr)
    sfs, trace = sequential_forward_selection(X_train, y_train, names, seed)
    spectrum = pca_spectrum(X_train, names)
    return {
        "separability": table, "corr": corr, "trace": trace, "spectrum": spectrum,
        "specs": {
            "Bayes + análisis": (analysis, subset_transform([names.index(f) for f in analysis])),
            "Bayes + SFS": (sfs, subset_transform([names.index(f) for f in sfs])),
            "PCA + Bayes": ([f"PC{i + 1}" for i in range(spectrum["n_components"])],
                            pca_transform(spectrum["n_components"])),
        },
    }


def evaluate_strategy(name, features, fit_transform, X, y, train_idx, val_idx, test_idx,
                      with_ci: bool = False, X_test: np.ndarray | None = None,
                      evaluate_test: bool = True) -> dict:
    """Fit on train, choose the Youden threshold on validation, then score the test images.

    The same fitted model produces validation and test scores, so the threshold chosen on
    validation is applied to exactly the score function it was chosen for.
    ``X_test`` replaces only the test descriptors (e.g. extracted with reference masks)
    while transform, model and threshold stay those learned from ``X``.
    ``evaluate_test=False`` never touches the test images (validation-only diagnostics).
    """
    transform = fit_transform(X[train_idx])
    clf = GaussianNB().fit(transform(X[train_idx]), y[train_idx])
    val_scores = log_likelihood_ratio(clf, transform(X[val_idx]))
    op = youden_operating_point(y[val_idx], val_scores)
    result = {
        "name": name, "features": features, "threshold": op["threshold"], "youden": op["youden"],
        "n_tied_thresholds": op["n_tied_thresholds"], "sweep": op["sweep"],
        "val_scores": val_scores,
        "val_metrics": metrics_at(y[val_idx], val_scores, op["threshold"]),
        "val_metrics_zero": metrics_at(y[val_idx], val_scores, 0.0),
        "val_roc": roc_curve(y[val_idx], val_scores)[:2],
        "gnb_theta": clf.theta_.tolist(), "gnb_var": clf.var_.tolist(),
    }
    if evaluate_test:
        X_eval = X if X_test is None else X_test
        test_scores = log_likelihood_ratio(clf, transform(X_eval[test_idx]))
        result.update({
            "test_scores": test_scores,
            "test_metrics": metrics_at(y[test_idx], test_scores, op["threshold"]),
            "test_roc": roc_curve(y[test_idx], test_scores)[:2],
        })
        if with_ci:
            result["test_ci"] = bootstrap_ci(y[test_idx], test_scores, op["threshold"])
    return result
