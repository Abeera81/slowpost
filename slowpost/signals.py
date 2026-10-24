"""Signal summary: does any feature change how likely a post is to have the top label?

Descriptive, not a backtest. Labels are whatever the caller passes in (usually thresholds from
all rows shown), so use it to look for patterns, not to measure accuracy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from .features import _blocks, groupings
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


_DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _group_of(feature: str, value: object) -> tuple[str, str] | None:
    """(signal-summary group name, group value) that holds a feature value, as in `groupings`."""
    if feature in ("has_cover_image", "title_has_question", "title_has_number"):
        return SIGNAL_FOR_FEATURE[feature], "yes" if int(value) else "no"
    if feature == "weekday":
        return "weekday", _DAY_NAMES[int(value)]
    if feature == "hour":
        return "hour (4h blocks, UTC)", str(_blocks(pd.Series([int(value)])).iloc[0])
    if feature == "reading_time_minutes":
        v = int(value)
        band = "0-2 min" if v <= 2 else "3-4 min" if v <= 4 else "5-7 min" if v <= 7 else "8+ min"
        return "reading time", band
    if feature == "main_tag":
        return "main tag", str(value)
    return None


def observed_rates(summary: dict, feature: str, feats: dict, overrides: dict) -> dict | None:
    """Share of past posts that got reactions in the group of the changed value ("new") and in
    the group of the draft's current value ("old"). None when the history has no such group."""
    if feature not in overrides:
        return None
    new, old = _group_of(feature, overrides[feature]), _group_of(feature, feats[feature])
    if new is None or old is None or new[0] not in summary["features"]:
        return None
    table = summary["features"][new[0]]["table"]
    names = [new[1], old[1]]
    if feature == "main_tag":  # tags outside the top 10 are grouped as "other"
        names = [n if n in table.index else "other" for n in names]
    if any(n not in table.index for n in names):
        return None
    top_is_none = "no_reaction" in str(summary["top_label"])  # DEV: the top label is "no_reactions"

    def rate(group: str) -> float:
        share = float(table.loc[group, "top_share"])
        return 1 - share if top_is_none else share

    return {"new": rate(names[0]), "old": rate(names[1]),
            "n_new": int(table.loc[names[0], "n"]), "n_old": int(table.loc[names[1], "n"])}
