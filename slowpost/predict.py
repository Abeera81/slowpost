"""Time-ordered backtest: TabPFN against the most-common-label baseline."""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from .features import build_matrices
from .labels import apply_labels, fit_thresholds
from .profiles import Profile

MAX_TRAIN = 1000  # CPU limit for TabPFN
MAX_TEST = 1500  # predicting 1,500 rows takes about 160 s on this laptop
MIN_TRAIN = 30


def time_split(df: pd.DataFrame, profile: Profile, test_frac: float = 0.2,
               max_test: int = MAX_TEST, max_train: int = MAX_TRAIN, seed: int = 0
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test = the newest posts. Train = a random sample (fixed seed) of up to `max_train`
    posts drawn from ALL posts older than the test window, so every weekday and hour is covered.
    Nothing in train is newer than anything in test (the cut is on the timestamp, so ties
    cannot straddle it)."""
    df = df.sort_values(profile.time_col, kind="stable").reset_index(drop=True)
    n_test = min(max_test, max(1, math.ceil(len(df) * test_frac)))
    t0 = df[profile.time_col].iloc[len(df) - n_test]
    test = df[df[profile.time_col] >= t0]
    pool = df[df[profile.time_col] < t0]
    if len(pool) < MIN_TRAIN:
        raise ValueError(f"Only {len(pool)} posts are older than the test window; "
                         f"need at least {MIN_TRAIN} to train.")
    train = pool.sample(n=min(max_train, len(pool)), random_state=seed)
    train = train.sort_values(profile.time_col, kind="stable")
    return train, test


def _auc(y_true: np.ndarray, proba: np.ndarray, classes: list[str]) -> float:
    try:
        if len(classes) == 2:
            return float(roc_auc_score(y_true == classes[1], proba[:, 1]))
        return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro",
                                   labels=classes))
    except ValueError:  # a class is missing from this (re)sample
        return float("nan")


def evaluate(y_true: np.ndarray, proba: np.ndarray, classes: list[str], baseline_label: str,
             n_boot: int = 1000, seed: int = 0) -> dict:
    """Metrics for the model and the baseline, with bootstrap 95% intervals on the gaps."""
    y_true = np.asarray(y_true)
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    base_pred = np.full(len(y_true), baseline_label)

    def gap_ba(idx: np.ndarray) -> float:
        return (balanced_accuracy_score(y_true[idx], pred[idx])
                - balanced_accuracy_score(y_true[idx], base_pred[idx]))

    everything = np.arange(len(y_true))
    rng = np.random.default_rng(seed)
    ba_gaps, aucs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        ba_gaps.append(gap_ba(idx))
        aucs.append(_auc(y_true[idx], proba[idx], classes))
    ba_lo, ba_hi = np.percentile(ba_gaps, [2.5, 97.5])
    auc_lo, auc_hi = np.nanpercentile(aucs, [2.5, 97.5])
    res = {
        "n_test": len(y_true),
        "baseline_label": baseline_label,
        "accuracy": float(accuracy_score(y_true, pred)),
        "baseline_accuracy": float(accuracy_score(y_true, base_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "baseline_balanced_accuracy": float(balanced_accuracy_score(y_true, base_pred)),
        "ba_gap": float(gap_ba(everything)),
        "ba_gap_ci": (float(ba_lo), float(ba_hi)),
        "auc": _auc(y_true, proba, classes),
        "auc_ci": (float(auc_lo), float(auc_hi)),
        "chance_auc": 0.5,
        "predicted_mix": pd.Series(pred).value_counts(normalize=True).round(3).to_dict(),
        "test_mix": pd.Series(y_true).value_counts(normalize=True).round(3).to_dict(),
    }
    res["verdict"] = verdict(res)
    return res


def verdict(res: dict) -> str:
    """One plain sentence. 'Beats' needs an interval fully above chance."""
    ba_better = res["ba_gap_ci"][0] > 0
    auc_better = res["auc_ci"][0] > res["chance_auc"]
    ba_worse = res["ba_gap_ci"][1] < 0
    if ba_better:
        return ("Better than the baseline: the model separates posts by label more than "
                "chance would, on this test window.")
    if auc_better:
        return ("Some signal: the model ranks posts better than chance (ROC AUC above 0.5), "
                "but at the default cutoff its labels are no better than the baseline's, "
                "so it cannot yet flag individual posts reliably.")
    if ba_worse:
        return "Worse than the baseline on this test window."
    return ("No better than the baseline: the model is not separating posts by label any "
            "better than guessing, so treat its predictions as noise.")


def backtest(df: pd.DataFrame, profile: Profile, seed: int = 0, **split_kwargs) -> dict:
    """Run the whole backtest. Imports TabPFN here so the rest of the module stays light."""
    from tabpfn import TabPFNClassifier

    train, test = time_split(df, profile, seed=seed, **split_kwargs)
    thresholds = fit_thresholds(train["outcome"], profile)  # training rows only
    y_train = apply_labels(train["outcome"], thresholds, profile)
    y_test = apply_labels(test["outcome"], thresholds, profile)
    if y_train.nunique() < 2:
        raise ValueError("The training posts all have the same label; nothing to learn from.")
    x_train, x_test, cat_idx = build_matrices(train, test, profile)

    clf = TabPFNClassifier(device="cpu", categorical_features_indices=cat_idx)
    clf.fit(x_train, y_train.values)
    t0 = time.perf_counter()
    proba = clf.predict_proba(x_test)
    seconds = time.perf_counter() - t0

    res = evaluate(y_test.values, proba, [str(c) for c in clf.classes_],
                   y_train.mode().iloc[0], seed=seed)
    res.update({
        "profile": profile.name,
        "n_train": len(train),
        "thresholds": thresholds,
        "train_range": (str(train[profile.time_col].min()), str(train[profile.time_col].max())),
        "test_range": (str(test[profile.time_col].min()), str(test[profile.time_col].max())),
        "train_mix": y_train.value_counts(normalize=True).round(3).to_dict(),
        "predict_seconds": round(seconds, 1),
    })
    return res
