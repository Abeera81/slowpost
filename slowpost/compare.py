"""Does TabPFN beat simpler models on the same split? Logistic regression and gradient boosting
are trained on the identical training sample and scored on the identical test posts.

Nothing here is tuned: the baselines use fixed, ordinary settings chosen before looking at any
result (see docs/analysis-log.md). Intervals are 95% bootstrap intervals; differences between
models use a paired bootstrap on the same test posts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import build_matrices
from .predict import MAX_TEST, MAX_TRAIN, Fitted, _auc, _top_share, evaluate, fit_on
from .profiles import Profile

NAMES = {"majority": "Most common outcome", "logistic": "Logistic regression",
         "gradient_boosting": "Gradient boosting", "tabpfn": "TabPFN"}


def make_baselines(cat_idx: list[int], n_cols: int, seed: int = 0) -> dict:
    """Logistic regression (scaled numbers + one-hot categories) and gradient boosting
    (scikit-learn's HistGradientBoosting, default settings, categories handled natively)."""
    numeric = [i for i in range(n_cols) if i not in cat_idx]
    parts = [("num", StandardScaler(), numeric)]
    if cat_idx:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_idx))
    logistic = make_pipeline(ColumnTransformer(parts), LogisticRegression(max_iter=1000))
    cat_mask = [i in cat_idx for i in range(n_cols)]
    boosting = HistGradientBoostingClassifier(
        categorical_features=cat_mask if cat_idx else None, random_state=seed)
    return {"logistic": logistic, "gradient_boosting": boosting}


def paired_diff(y_true: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray, classes: list[str],
                n_boot: int = 1000, seed: int = 0) -> dict:
    """A minus B on the same test posts, resampled together: ROC AUC and (binary) the share of
    real positives among the model's top 10%."""
    rng = np.random.default_rng(seed)
    binary = len(classes) == 2
    y_pos = (y_true == classes[1]) if binary else None
    auc_d, top_d = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        auc_d.append(_auc(y_true[idx], proba_a[idx], classes) - _auc(y_true[idx], proba_b[idx], classes))
        if binary:
            top_d.append(_top_share(y_pos[idx], proba_a[idx, 1], 0.1)
                         - _top_share(y_pos[idx], proba_b[idx, 1], 0.1))
    out = {"auc_diff": _auc(y_true, proba_a, classes) - _auc(y_true, proba_b, classes),
           "auc_diff_ci": tuple(map(float, np.nanpercentile(auc_d, [2.5, 97.5])))}
    if binary:
        out["top10_diff"] = _top_share(y_pos, proba_a[:, 1], 0.1) - _top_share(y_pos, proba_b[:, 1], 0.1)
        out["top10_diff_ci"] = tuple(map(float, np.percentile(top_d, [2.5, 97.5])))
    return out


def compact(res: dict) -> dict:
    """The parts of an evaluate() result shown in the comparison table."""
    row = {"balanced_accuracy": res["balanced_accuracy"], "ba_ci": res["ba_ci"],
           "ba_gap": res["ba_gap"], "ba_gap_ci": res["ba_gap_ci"],
           "auc": res["auc"], "auc_ci": res["auc_ci"], "n_test": res["n_test"]}
    for name, v in (res.get("lift") or {}).items():
        row[name] = {"share": v["share"], "share_ci": v["share_ci"], "lift": v["lift"],
                     "lift_ci": v["lift_ci"], "n_top": v["n_top"]}
    return row


def winner_text(rows: dict, diffs: dict) -> str:
    """Plain verdict on whether TabPFN wins, from the paired ROC AUC differences, one clause
    per simpler model. A higher point estimate with an interval that includes zero is not a win."""
    if not diffs:
        return ""
    clear, unclear, worse = [], [], []
    for key, d in diffs.items():
        lo, hi = d["auc_diff_ci"]
        text = f"{NAMES[key].lower()} (AUC {d['auc_diff']:+.3f}, paired 95% interval {lo:+.3f} to {hi:+.3f})"
        (clear if lo > 0 else worse if hi < 0 else unclear).append(text)
    parts = []
    if clear:
        parts.append("TabPFN ranks posts clearly better than " + " and ".join(clear))
    if unclear:
        parts.append(("TabPFN is" if not parts else "it is") + " not clearly different from "
                     + " and ".join(unclear))
    if worse:
        parts.append("TabPFN ranks posts worse than " + " and ".join(worse))
    text = "; ".join(parts) + "."
    if unclear or worse:
        text += " So TabPFN does not clearly win here: a simple model does about as well."
    higher_ba = [NAMES[k] for k in diffs if rows[k]["balanced_accuracy"] > rows["tabpfn"]["balanced_accuracy"]]
    if higher_ba:
        text += (f" {' and '.join(higher_ba)} has the higher balanced accuracy "
                 f"({max(rows[k]['balanced_accuracy'] for k in diffs):.3f} vs "
                 f"{rows['tabpfn']['balanced_accuracy']:.3f}).")
    return text


def compare_models(model: Fitted, tabpfn_proba: np.ndarray, tabpfn_res: dict, seed: int = 0) -> dict:
    """Majority baseline, logistic regression, gradient boosting and TabPFN on the identical
    split. `tabpfn_res` is the evaluate() result already computed for TabPFN."""
    x_train, x_test, cat_idx = build_matrices(model.train, model.test, model.profile)
    y_train, y_test = model.y_train.to_numpy(), model.y_test.to_numpy()
    rows = {"tabpfn": compact(tabpfn_res)}
    probas = {"tabpfn": tabpfn_proba}
    baseline_label = tabpfn_res["baseline_label"]
    for key, clf in make_baselines(cat_idx, x_train.shape[1], seed).items():
        clf.fit(x_train, y_train)
        proba = clf.predict_proba(x_test)
        if [str(c) for c in clf.classes_] != model.classes:
            raise ValueError("Class order differs between models.")
        probas[key] = proba
        rows[key] = compact(evaluate(y_test, proba, model.classes, baseline_label, model.priors,
                                     seed=seed))
    # The most common outcome: a constant score has no ranking (AUC 0.5, lift 1.0).
    majority = {"balanced_accuracy": tabpfn_res["baseline_balanced_accuracy"], "ba_ci": None,
                "auc": 0.5, "auc_ci": None, "n_test": tabpfn_res["n_test"]}
    if tabpfn_res.get("lift"):
        majority.update({name: {"share": tabpfn_res["overall_share"], "lift": 1.0, "lift_ci": None,
                                "share_ci": tabpfn_res["overall_share_ci"], "n_top": v["n_top"]}
                         for name, v in tabpfn_res["lift"].items()})
    ordered = {"majority": majority, **{k: rows[k] for k in ("logistic", "gradient_boosting", "tabpfn")}}
    diffs = {k: paired_diff(y_test, probas["tabpfn"], probas[k], model.classes, seed=seed)
             for k in ("logistic", "gradient_boosting")}
    return {"models": ordered, "tabpfn_minus": diffs, "winner": winner_text(ordered, diffs),
            "settings": {"logistic": "StandardScaler on numbers, one-hot on categories, "
                                     "LogisticRegression(max_iter=1000)",
                         "gradient_boosting": "HistGradientBoostingClassifier, default settings, "
                                              "categories handled natively",
                         "seed": seed}}


def random_split(df: pd.DataFrame, profile: Profile, seed: int = 0) -> dict:
    """TabPFN on a random 80/20 split of all posts (same size caps as the temporal split).
    Shown only to compare: it lets the model train on posts from the same weeks it is tested on."""
    order = np.random.default_rng(seed).permutation(len(df))
    cut = int(0.8 * len(df))
    pool_train, pool_test = df.iloc[order[:cut]], df.iloc[order[cut:]]
    train = pool_train.sample(n=min(MAX_TRAIN, len(pool_train)), random_state=seed)
    test = pool_test.sample(n=min(MAX_TEST, len(pool_test)), random_state=seed)
    model = fit_on(train, test, profile)
    proba = model.clf.predict_proba(model.x_test)
    res = evaluate(model.y_test.to_numpy(), proba, model.classes, model.y_train.mode().iloc[0],
                   model.priors, seed=seed)
    model.release()
    out = compact(res)
    out.update({"n_train": len(train), "n_test": len(test),
                "test_range": (str(test[profile.time_col].min()), str(test[profile.time_col].max())),
                "overall_share": res.get("overall_share")})
    return out

