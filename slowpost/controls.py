"""Controlled checks on the DEV history: does a gap survive when we hold another feature fixed?

Cover image and reading time are related (long posts more often have covers), so a raw gap in one
could be a gap in the other. Two descriptive checks, both fixed before looking at the results:

1. The cover-image gap (cover vs no cover) inside each reading-time band.
2. The reading-time gap ("long" = at or above the median reading time, vs shorter) with a cover
   and without one.

Each reports per-stratum rates, a 95% interval for the difference, and a Mantel-Haenszel pooled
odds ratio next to the crude (unadjusted) one. Associations only: nothing here says why.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

BANDS = [-1, 2, 4, 7, 1e9]
BAND_LABELS = ["0-2 min", "3-4 min", "5-7 min", "8+ min"]
MIN_CELL = 20  # a group smaller than this is shown but not counted as evidence
Z = 1.96


def _rate_diff(yes1: int, n1: int, yes0: int, n0: int) -> dict:
    """Rates of the two groups and their difference (group 1 minus group 0) with a 95% interval."""
    p1, p0 = yes1 / n1 if n1 else float("nan"), yes0 / n0 if n0 else float("nan")
    diff = p1 - p0
    se = math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0) if n1 and n0 else float("nan")
    return {"n1": n1, "rate1": p1, "n0": n0, "rate0": p0, "diff": diff,
            "diff_ci": (diff - Z * se, diff + Z * se),
            "enough": bool(n1 >= MIN_CELL and n0 >= MIN_CELL)}


def _odds_ratios(tables: list[tuple[int, int, int, int]]) -> dict:
    """tables: (a, b, c, d) = exposed yes/no, unexposed yes/no. Crude and Mantel-Haenszel ORs."""
    a, b, c, d = (sum(t[i] for t in tables) for i in range(4))
    crude = (a * d) / (b * c) if b * c else float("nan")
    r_sum = s_sum = 0.0
    for a_, b_, c_, d_ in tables:
        n = a_ + b_ + c_ + d_
        if n:
            r_sum += a_ * d_ / n
            s_sum += b_ * c_ / n
    if not r_sum or not s_sum:
        return {"crude_or": crude, "mh_or": float("nan"), "mh_or_ci": (float("nan"),) * 2}
    var = 0.0
    for a_, b_, c_, d_ in tables:
        n = a_ + b_ + c_ + d_
        if not n:
            continue
        p_, q_, r_, s_ = (a_ + d_) / n, (b_ + c_) / n, a_ * d_ / n, b_ * c_ / n
        var += p_ * r_ / (2 * r_sum ** 2) + (p_ * s_ + q_ * r_) / (2 * r_sum * s_sum) \
            + q_ * s_ / (2 * s_sum ** 2)
    mh = r_sum / s_sum
    se = math.sqrt(var)
    return {"crude_or": crude, "mh_or": mh,
            "mh_or_ci": (mh * math.exp(-Z * se), mh * math.exp(Z * se))}


def _stratified(df: pd.DataFrame, exposed: pd.Series, strata: pd.Series) -> dict:
    """Group-1 (exposed) vs group-0 gap inside each stratum, plus pooled odds ratios."""
    yes = (df["outcome"] >= 1).to_numpy()
    rows, tables = [], []
    for name in strata.cat.categories if hasattr(strata, "cat") else sorted(strata.unique()):
        m = (strata == name).to_numpy()
        e = exposed.to_numpy().astype(bool)
        a, b = int((yes & m & e).sum()), int((~yes & m & e).sum())
        c, d = int((yes & m & ~e).sum()), int((~yes & m & ~e).sum())
        rows.append({"stratum": str(name), **_rate_diff(a, a + b, c, c + d)})
        tables.append((a, b, c, d))
    crude = _rate_diff(int((yes & exposed.to_numpy().astype(bool)).sum()),
                       int(exposed.sum()), int((yes & ~exposed.to_numpy().astype(bool)).sum()),
                       int((~exposed.to_numpy().astype(bool)).sum()))
    return {"strata": rows, "crude": crude, **_odds_ratios(tables)}


def _holds(strata: list[dict]) -> str:
    use = [s for s in strata if s["enough"]]
    pos = [s for s in use if s["diff_ci"][0] > 0]
    neg = [s for s in use if s["diff_ci"][1] < 0]
    same_sign = sum(1 for s in use if s["diff"] > 0)
    if not use:
        return "Too few posts in every group to say."
    if len(pos) == len(use):
        return f"Holds: the gap is clearly positive in all {len(use)} groups."
    if neg and pos:
        return "Mixed: the gap is clearly positive in some groups and clearly negative in others."
    if pos:
        return (f"Partly holds: clearly positive in {len(pos)} of {len(use)} groups "
                f"(positive in {same_sign}), not clear in the rest.")
    if same_sign == len(use):
        return (f"Same direction in all {len(use)} groups, but no group shows a clear gap on its own "
                "(the 95% intervals include zero).")
    return "Does not hold: the gap is not clearly present inside the groups."


def controlled_checks(df: pd.DataFrame) -> dict:
    """df: the DEV history with outcome (reactions), has_cover_image and reading_time_minutes."""
    bands = pd.cut(df["reading_time_minutes"], BANDS, labels=BAND_LABELS)
    cover = df["has_cover_image"].astype(int) == 1
    median = float(df["reading_time_minutes"].median())
    long_read = df["reading_time_minutes"] >= median
    cover_in_bands = _stratified(df, cover, bands)
    cover_in_bands["statement"] = _holds(cover_in_bands["strata"])
    cover_strata = pd.Series(np.where(cover, "with a cover", "without a cover"), index=df.index)
    time_by_cover = _stratified(df, long_read, cover_strata)
    time_by_cover["statement"] = _holds(time_by_cover["strata"])
    return {
        "outcome": "any reactions (reactions >= 1)",
        "median_reading_time": median,
        "cover_gap_within_reading_time": cover_in_bands,
        "reading_time_gap_by_cover": time_by_cover,
        "note": ("Associations in the history only. Reading time and cover reflect current values "
                 "(authors can edit them after publishing)."),
    }
