"""Baseline comparison: the simple models run on a given split; no TabPFN needed."""
from __future__ import annotations

import numpy as np

from slowpost.compare import make_baselines, paired_diff, winner_text


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.normal(size=n), rng.integers(0, 4, n), rng.integers(0, 2, n)]).astype(float)
    logit = 1.5 * x[:, 0] + (x[:, 1] == 2)
    y = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "yes", "no")
    return x, y


def test_baselines_fit_and_rank_better_than_chance():
    from sklearn.metrics import roc_auc_score

    x, y = _data()
    for name, clf in make_baselines(cat_idx=[1], n_cols=3).items():
        clf.fit(x[:300], y[:300])
        assert [str(c) for c in clf.classes_] == ["no", "yes"], name
        assert roc_auc_score(y[300:] == "yes", clf.predict_proba(x[300:])[:, 1]) > 0.65, name


def test_paired_diff_is_zero_for_identical_scores():
    x, y = _data()
    p = np.column_stack([1 - x[:, 0] / 10 - 0.5, x[:, 0] / 10 + 0.5])
    d = paired_diff(y, p, p, ["no", "yes"], n_boot=50)
    assert d["auc_diff"] == 0 and d["auc_diff_ci"] == (0.0, 0.0)


def test_winner_text_says_when_tabpfn_does_not_win():
    rows = {"tabpfn": {"auc": 0.58, "balanced_accuracy": 0.54},
            "logistic": {"auc": 0.60, "balanced_accuracy": 0.56},
            "gradient_boosting": {"auc": 0.50, "balanced_accuracy": 0.51}}
    diffs = {"logistic": {"auc_diff": -0.02, "auc_diff_ci": (-0.04, 0.01)},
             "gradient_boosting": {"auc_diff": 0.08, "auc_diff_ci": (0.05, 0.11)}}
    text = winner_text(rows, diffs)
    assert "clearly better than gradient boosting" in text
    assert "not clearly different from logistic regression" in text
    assert "does not clearly win" in text and "higher balanced accuracy" in text
    clear = {k: {"auc_diff": 0.05, "auc_diff_ci": (0.02, 0.08)} for k in diffs}
    text = winner_text({**rows, "logistic": {"auc": 0.5, "balanced_accuracy": 0.5}}, clear)
    assert "does not clearly win" not in text and "clearly better than" in text
    worse = {"logistic": {"auc_diff": -0.05, "auc_diff_ci": (-0.08, -0.02)}}
    assert "ranks posts worse than" in winner_text(rows, worse)


def test_calibration_bins_are_equal_count_and_match_rates():
    from slowpost.predict import calibration_bins

    p = np.linspace(0.05, 0.6, 400)
    y = np.random.default_rng(1).random(400) < p
    bins = calibration_bins(y, p, n_bins=8)
    assert len(bins) == 8 and all(b["n"] == 50 for b in bins)
    assert bins[0]["mean_predicted"] < bins[-1]["mean_predicted"]
    assert sum(b["positives"] for b in bins) == int(y.sum())
    assert all(b["actual_ci"][0] <= b["actual"] <= b["actual_ci"][1] for b in bins)
