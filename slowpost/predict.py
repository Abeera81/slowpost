"""Time-ordered backtest: TabPFN against the most-common-label baseline."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from .features import build_matrices
from .labels import apply_labels, fit_thresholds
from .profiles import Profile

MAX_TRAIN = 1000  # CPU limit for TabPFN
MAX_TEST = 2000  # predicting 1,500 rows took about 160 s on this laptop
TEST_DAYS = 7
MIN_TEST = 100  # fewer posts than this in the newest week -> use the newest 20% instead
MIN_TRAIN = 30


def time_split(df: pd.DataFrame, profile: Profile, test_days: int = TEST_DAYS,
               max_test: int = MAX_TEST, max_train: int = MAX_TRAIN, seed: int = 0
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test = a random sample (fixed seed) of up to `max_test` posts from the newest
    `test_days` days. Train = a random sample (fixed seed) of up to `max_train` posts from ALL
    older posts, so every weekday and hour is covered. Nothing in train is newer than anything
    in test (the cut is on the timestamp, so ties cannot straddle it).

    A small history (fewer than MIN_TEST posts in the newest week) falls back to the newest
    20% of posts as the test window. `test.attrs["window"]` says which rule was used."""
    col = profile.time_col
    df = df.sort_values(col, kind="stable").reset_index(drop=True)
    cut = df[col].iloc[-1] - pd.Timedelta(days=test_days)
    window = f"newest {test_days} days"
    if (df[col] > cut).sum() < MIN_TEST:
        n_recent = max(1, math.ceil(len(df) * 0.2))
        cut = df[col].iloc[len(df) - n_recent - 1] if n_recent < len(df) else df[col].iloc[0]
        window = "newest 20% of posts (the newest week was too small)"
    recent, pool = df[df[col] > cut], df[df[col] <= cut]
    if len(pool) < MIN_TRAIN:
        raise ValueError(f"Only {len(pool)} posts are older than the test window; "
                         f"need at least {MIN_TRAIN} to train.")
    test = recent.sample(n=min(max_test, len(recent)), random_state=seed)
    test = test.sort_values(col, kind="stable")
    test.attrs["window"] = f"{window}; {len(recent)} posts, {len(test)} sampled"
    train = pool.sample(n=min(max_train, len(pool)), random_state=seed)
    train = train.sort_values(col, kind="stable")
    return train, test


def _auc(y_true: np.ndarray, proba: np.ndarray, classes: list[str]) -> float:
    try:
        if len(classes) == 2:
            return float(roc_auc_score(y_true == classes[1], proba[:, 1]))
        return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro",
                                   labels=classes))
    except ValueError:  # a class is missing from this (re)sample
        return float("nan")


def predict_labels(proba: np.ndarray, classes: list[str], priors: np.ndarray) -> np.ndarray:
    """Binary: 'yes' (the second class) when P(yes) is above its training share.
    More classes: pick the class with the highest probability relative to its training share.
    `priors` comes from training rows only."""
    if len(classes) == 2:
        return np.where(proba[:, 1] > priors[1], classes[1], classes[0])
    return np.asarray(classes)[(proba / priors).argmax(axis=1)]


LIFT_FRACTIONS = {"top10": 0.10, "top20": 0.20}


def _top_share(y_pos: np.ndarray, score: np.ndarray, frac: float) -> float:
    """Share of actual positives among the `frac` highest-scored posts."""
    k = max(1, math.ceil(frac * len(score)))
    top = np.argsort(-score, kind="stable")[:k]
    return float(y_pos[top].mean())


def evaluate(y_true: np.ndarray, proba: np.ndarray, classes: list[str], baseline_label: str,
             priors: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """Metrics for the model and the baseline, with bootstrap 95% intervals on the gaps."""
    y_true = np.asarray(y_true)
    pred = predict_labels(proba, classes, np.asarray(priors))
    base_pred = np.full(len(y_true), baseline_label)

    def gap_ba(idx: np.ndarray) -> float:
        return (balanced_accuracy_score(y_true[idx], pred[idx])
                - balanced_accuracy_score(y_true[idx], base_pred[idx]))

    everything = np.arange(len(y_true))
    rng = np.random.default_rng(seed)
    ba_gaps, aucs = [], []
    binary = len(classes) == 2
    if binary:  # lift: do the posts the model ranks highest really get reactions more often?
        y_pos, score = y_true == classes[1], proba[:, 1]
        boot_top = {k: [] for k in LIFT_FRACTIONS}
        boot_overall = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        ba_gaps.append(gap_ba(idx))
        aucs.append(_auc(y_true[idx], proba[idx], classes))
        if binary:
            boot_overall.append(float(y_pos[idx].mean()))
            for name, frac in LIFT_FRACTIONS.items():
                boot_top[name].append(_top_share(y_pos[idx], score[idx], frac))
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
        "cutoff": float(priors[1]) if len(classes) == 2 else None,  # P(yes) must exceed this
        "predicted_mix": pd.Series(pred).value_counts(normalize=True).round(3).to_dict(),
        "test_mix": pd.Series(y_true).value_counts(normalize=True).round(3).to_dict(),
    }
    if binary:
        overall = float(y_pos.mean())
        o_lo, o_hi = np.percentile(boot_overall, [2.5, 97.5])
        res["overall_share"] = overall
        res["overall_share_ci"] = (float(o_lo), float(o_hi))
        res["lift"] = {}
        for name, frac in LIFT_FRACTIONS.items():
            share = _top_share(y_pos, score, frac)
            lo, hi = np.percentile(boot_top[name], [2.5, 97.5])
            ratio = np.array(boot_top[name]) / np.array(boot_overall)
            r_lo, r_hi = np.percentile(ratio, [2.5, 97.5])
            res["lift"][name] = {
                "top_fraction": frac, "n_top": max(1, math.ceil(frac * len(y_true))),
                "share": share, "share_ci": (float(lo), float(hi)),
                "lift": share / overall if overall else float("nan"),
                "lift_ci": (float(r_lo), float(r_hi)),
            }
    res["verdict"] = verdict(res)
    return res


def verdict(res: dict) -> str:
    """One plain sentence from both measures. Each needs a 95% interval fully above chance."""
    ba_better = res["ba_gap_ci"][0] > 0
    auc_better = res["auc_ci"][0] > res["chance_auc"]
    if ba_better and auc_better:
        return ("Better than the baseline: the model ranks posts better than chance and its "
                "labels beat the baseline's, on this test window.")
    if auc_better:
        return ("Some signal: the model ranks posts better than chance (ROC AUC above 0.5), "
                "but its labels are not clearly better than the baseline's, so individual "
                "flags are unreliable.")
    if ba_better:
        return ("Some signal: the model's labels beat the baseline's, but its ranking "
                "(ROC AUC) is not clearly above chance, so treat this with caution.")
    if res["ba_gap_ci"][1] < 0:
        return "Worse than the baseline on this test window."
    return ("No better than the baseline: the model is not separating posts by label any "
            "better than guessing, so treat its predictions as noise.")


@dataclass
class Fitted:
    """A TabPFN model fitted on the training sample, plus what is needed to score new rows.
    Call `release()` to free memory before loading Gemma."""
    clf: object
    profile: Profile
    train: pd.DataFrame
    test: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    thresholds: tuple[float, ...]
    classes: list[str]
    priors: np.ndarray  # training label shares

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Probabilities for rows that have the profile's feature columns."""
        _, x, _ = build_matrices(self.train, features, self.profile)
        return self.clf.predict_proba(x)

    def release(self) -> None:
        import gc

        self.clf = None
        gc.collect()


def fit_model(df: pd.DataFrame, profile: Profile, seed: int = 0, **split_kwargs) -> Fitted:
    """Split by time, learn label thresholds from training rows only, and fit TabPFN.
    Imports TabPFN here so the rest of the module stays light."""
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
    classes = [str(c) for c in clf.classes_]
    priors = y_train.value_counts(normalize=True).reindex(classes).to_numpy()  # training only
    return Fitted(clf, profile, train, test, x_test, y_train, y_test, thresholds, classes, priors)


def score_groups(y_pos: np.ndarray, p_yes: np.ndarray, cutoff: float, strong: float) -> dict:
    """Measured share with reactions in each score group of the test window."""
    groups = {
        "strong": p_yes >= strong,
        "weak": p_yes < cutoff,
    }
    groups["okay"] = ~(groups["strong"] | groups["weak"])
    return {k: {"n": int(m.sum()), "share_with_reactions": float(y_pos[m].mean()) if m.any() else None}
            for k, m in groups.items()}


def run_backtest(df: pd.DataFrame, profile: Profile, seed: int = 0, **split_kwargs
                 ) -> tuple[dict, Fitted]:
    """Backtest result plus the fitted model (so scoring can reuse it)."""
    model = fit_model(df, profile, seed=seed, **split_kwargs)
    t0 = time.perf_counter()
    proba = model.clf.predict_proba(model.x_test)
    seconds = time.perf_counter() - t0
    train, test = model.train, model.test
    res = evaluate(model.y_test.values, proba, model.classes, model.y_train.mode().iloc[0],
                   model.priors, seed=seed)
    res.update({
        "profile": profile.name,
        "n_train": len(train),
        "positive": model.classes[1] if len(model.classes) == 2 else None,
        "thresholds": model.thresholds,
        "train_range": (str(train[profile.time_col].min()), str(train[profile.time_col].max())),
        "test_range": (str(test[profile.time_col].min()), str(test[profile.time_col].max())),
        "train_mix": model.y_train.value_counts(normalize=True).round(3).to_dict(),
        "test_window": test.attrs.get("window", ""),
        "predict_seconds": round(seconds, 1),
    })
    if len(model.classes) == 2:
        p_yes = proba[:, 1]
        k = max(1, math.ceil(LIFT_FRACTIONS["top20"] * len(p_yes)))
        # Strong = at or above the P(yes) of the model's top-20% test posts, but never below
        # the cutoff. Weak = below the training-share cutoff.
        res["strong_threshold"] = float(max(np.sort(p_yes)[::-1][k - 1], res["cutoff"]))
        res["score_groups"] = score_groups(
            (model.y_test.values == model.classes[1]), p_yes, res["cutoff"],
            res["strong_threshold"])
    return res, model


def backtest(df: pd.DataFrame, profile: Profile, seed: int = 0, **split_kwargs) -> dict:
    """Run the whole backtest and return only the result."""
    return run_backtest(df, profile, seed=seed, **split_kwargs)[0]
