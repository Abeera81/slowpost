"""Features known before posting, computed by code, plus the groups used in signal summaries."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .profiles import Profile

_DIGIT = re.compile(r"\d")


def split_tags(value: object) -> list[str]:
    """tag_list is stored as 'webdev,python'. Accepts a list or a string too."""
    if isinstance(value, (list, tuple)):
        return [str(t).strip() for t in value if str(t).strip()]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [t.strip() for t in str(value).split(",") if t.strip()]


def dev_features(df: pd.DataFrame) -> pd.DataFrame:
    tags = df["tag_list"].map(split_tags)
    title = df["title"].fillna("").astype(str)
    out = df.copy()
    out["weekday"] = df["post_datetime"].dt.day_of_week  # 0 = Monday, UTC
    out["hour"] = df["post_datetime"].dt.hour  # UTC
    out["reading_time_minutes"] = pd.to_numeric(df["reading_time_minutes"], errors="coerce")
    out["n_tags"] = tags.map(len)
    out["main_tag"] = tags.map(lambda t: t[0] if t else "none")
    out["has_cover_image"] = df["has_cover_image"].map(
        lambda v: 1 if str(v).strip().lower() in ("true", "1", "1.0") else 0)
    out["title_length"] = title.str.len()
    out["title_has_question"] = title.str.contains("?", regex=False).astype(int)
    out["title_has_number"] = title.map(lambda s: int(bool(_DIGIT.search(s))))
    return out


def instagram_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("post_hour", "caption_length", "hashtags_count", "has_call_to_action"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def add_features(df: pd.DataFrame, profile: Profile) -> pd.DataFrame:
    if profile.name == "dev_posts":
        return dev_features(df)
    if profile.name == "instagram_kaggle":
        return instagram_features(df)
    raise ValueError(f"No feature rules for profile {profile.name}")


def build_matrices(train: pd.DataFrame, test: pd.DataFrame, profile: Profile
                   ) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """Numeric matrices for the model. Category lists come from the training rows only;
    anything unseen (or outside the training top-N) becomes 'other'.
    Returns (x_train, x_test, indices of categorical columns)."""
    cols = profile.features
    x_train, x_test = train[cols].copy(), test[cols].copy()
    for col in profile.categorical_features:
        counts = train[col].astype(str).value_counts()
        keep = list(counts.index[: profile.top_categories.get(col, len(counts))])
        codes = {v: i for i, v in enumerate([*keep, "other"])}
        for x in (x_train, x_test):
            x[col] = x[col].astype(str).map(lambda v, c=codes: c.get(v, c["other"]))
    cat_idx = [cols.index(c) for c in profile.categorical_features]
    return x_train.astype(float), x_test.astype(float), cat_idx


def _blocks(hours: pd.Series) -> pd.Series:
    return pd.cut(hours, [-1, 3, 7, 11, 15, 19, 23],
                  labels=["00-03", "04-07", "08-11", "12-15", "16-19", "20-23"])


def _quartiles(values: pd.Series) -> pd.Series:
    return pd.qcut(values, 4, duplicates="drop").astype(str)


def groupings(df: pd.DataFrame, profile: Profile) -> dict[str, pd.Series]:
    """Features grouped into readable buckets for the signal summary."""
    if profile.name == "dev_posts":
        top = df["main_tag"].value_counts().index[:10]
        days = dict(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]))
        yes_no = {1: "yes", 0: "no"}
        return {
            "weekday": df["weekday"].map(days),
            "hour (4h blocks, UTC)": _blocks(df["hour"]),
            "reading time": pd.cut(df["reading_time_minutes"], [-1, 2, 4, 7, 1e9],
                                   labels=["0-2 min", "3-4 min", "5-7 min", "8+ min"]),
            "number of tags": df["n_tags"].map(lambda n: str(n) if n < 4 else "4+"),
            "main tag": df["main_tag"].where(df["main_tag"].isin(top), "other"),
            "has cover image": df["has_cover_image"].map(yes_no),
            "title length": _quartiles(df["title_length"]),
            "title has question": df["title_has_question"].map(yes_no),
            "title has number": df["title_has_number"].map(yes_no),
        }
    if profile.name == "instagram_kaggle":
        return {
            "media_type": df["media_type"],
            "content_category": df["content_category"],
            "day_of_week": df["day_of_week"],
            "post_hour (4h blocks)": _blocks(df["post_hour"]),
            "has_call_to_action": df["has_call_to_action"].map({1: "yes", 0: "no"}),
            "hashtags_count (4 bins)": _quartiles(df["hashtags_count"]),
            "caption_length (4 bins)": _quartiles(df["caption_length"]),
        }
    raise ValueError(f"No groupings for profile {profile.name}")
