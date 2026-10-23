"""Label rules. Thresholds are learned from training rows only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .profiles import Profile


def fit_thresholds(train_outcome: pd.Series, profile: Profile) -> tuple[float, ...]:
    """Cut points between labels, computed from training rows only."""
    if profile.label_rule == "thirds":
        lo, hi = train_outcome.quantile([1 / 3, 2 / 3])
        return (float(lo), float(hi))
    if profile.label_rule == "min_value":
        return (float(profile.min_value),)  # fixed rule: nothing is learned from the data
    raise ValueError(f"Unknown label rule {profile.label_rule}")


def apply_labels(values: pd.Series, thresholds: tuple[float, ...], profile: Profile) -> pd.Series:
    names = profile.label_names
    if profile.label_rule == "thirds":
        lo, hi = thresholds
        labels = np.where(values <= lo, names[0], np.where(values <= hi, names[1], names[2]))
    else:
        labels = np.where(values >= thresholds[0], names[1], names[0])
    return pd.Series(labels, index=values.index)
