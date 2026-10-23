"""Signal summary: does any feature change how likely a post is to have the top label?

Descriptive, not a backtest. Labels are whatever the caller passes in (usually thresholds from
all rows shown), so use it to look for patterns, not to measure accuracy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from .features import groupings
from .profiles import Profile

ALPHA = 0.05
MIN_GROUP = 20  # smaller groups are shown but left out of the test


def signal_summary(df: pd.DataFrame, labels: pd.Series, profile: Profile) -> dict:
    """For each feature: per-group counts and the share of top-label posts, plus a chi-square
    p-value corrected for the number of features tested (Bonferroni).

    'Top label' = the most common label overall. Returns
    {"top_label", "overall_share", "features": {name: {"table", "p", "p_adj", "clear"}}}.
    """
    top = labels.value_counts().index[0]
    is_top = (labels == top).astype(int)
    overall = float(is_top.mean())
    groups = groupings(df, profile)
    out: dict[str, dict] = {}
    for name, g in groups.items():
        g = g.astype(str)
        counts = pd.DataFrame({"g": g, "top": is_top.values})
        table = counts.groupby("g")["top"].agg(n="size", top_share="mean")
        table["gap"] = table["top_share"] - overall
        big = table.index[table["n"] >= MIN_GROUP]
        p = _chi_square(g[g.isin(big)], labels[g.isin(big)])
        out[name] = {"table": table.round(3), "p": p}
    k = max(len(out), 1)
    for v in out.values():
        v["p_adj"] = min(1.0, v["p"] * k)
        v["clear"] = bool(v["p_adj"] < ALPHA)
    return {"top_label": top, "overall_share": overall, "features": out}


def _chi_square(groups: pd.Series, labels: pd.Series) -> float:
    ct = pd.crosstab(groups.values, labels.values)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 1.0
    return float(chi2_contingency(ct)[1])


def describe(summary: dict) -> str:
    """Plain-language line for the UI and the console."""
    clear = [n for n, v in summary["features"].items() if v["clear"]]
    if not clear:
        return ("No feature shows a clear difference (after correcting for testing several "
                "features at once). Any pattern you see is probably chance.")
    return "Clear differences (corrected p < 0.05) for: " + ", ".join(clear) + "."


def best_p(summary: dict) -> float:
    return float(np.nanmin([v["p_adj"] for v in summary["features"].values()]))


# What-if feature name -> the signal-summary group that tests it.
SIGNAL_FOR_FEATURE = {
    "has_cover_image": "has cover image",
    "hour": "hour (4h blocks, UTC)",
    "weekday": "weekday",
    "main_tag": "main tag",
    "reading_time_minutes": "reading time",
    "title_has_question": "title has question",
    "title_has_number": "title has number",
}


def supported_features(summary: dict) -> set[str]:
    """What-if features whose signal-summary group shows a clear difference after correction."""
    clear = {n for n, v in summary["features"].items() if v["clear"]}
    return {f for f, group in SIGNAL_FOR_FEATURE.items() if group in clear}


def to_jsonable(summary: dict) -> dict:
    """Plain dicts and lists, for saving next to the backtest."""
    return {
        "top_label": summary["top_label"], "overall_share": summary["overall_share"],
        "text": describe(summary),
        "features": {
            name: {"p": v["p"], "p_adj": v["p_adj"], "clear": v["clear"],
                   "groups": [{"group": str(g), "n": int(r["n"]), "top_share": float(r["top_share"]),
                               "gap": float(r["gap"])} for g, r in v["table"].iterrows()]}
            for name, v in summary["features"].items()},
    }
