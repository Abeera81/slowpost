"""Load a post history through a source profile."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .features import add_features
from .profiles import Profile


class MissingColumnsError(ValueError):
    """The file lacks columns the profile needs. The message names them."""


def load_posts(source: str | Path | pd.DataFrame, profile: Profile,
               where: dict[str, object] | None = None) -> pd.DataFrame:
    """Read a CSV (or DataFrame), map columns, compute outcome and features, sort by time.

    `where` filters raw rows by column value before mapping (e.g. {"account_id": 15}).
    """
    raw = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    for col, value in (where or {}).items():
        if col not in raw.columns:
            raise MissingColumnsError(f"Missing column: {col}")
        raw = raw[raw[col] == value]
    missing = [c for c in profile.required_columns if c not in raw.columns]
    if missing:
        raise MissingColumnsError(f"Missing column(s): {', '.join(missing)}")

    df = raw[profile.required_columns].rename(columns=profile.column_map).copy()
    df[profile.time_col] = pd.to_datetime(df[profile.time_col], errors="coerce", utc=True)
    df = df.dropna(subset=[profile.time_col])
    df["outcome"] = outcome(df, profile)
    df = df.dropna(subset=["outcome"])
    df = df.sort_values(profile.time_col, kind="stable").reset_index(drop=True)
    return add_features(df, profile)


def outcome(df: pd.DataFrame, profile: Profile) -> pd.Series:
    """The number we try to predict. NaN rows are dropped by the caller."""
    if profile.name == "instagram_kaggle":
        # Computed, not read: the CSV's own engagement_rate is ignored.
        reach = df["reach"].where(df["reach"] > 0)
        return (df["likes"] + df["comments"] + df["shares"] + df["saves"]) / reach
    if profile.name == "dev_posts":
        return pd.to_numeric(df["reactions"], errors="coerce")
    raise ValueError(f"No outcome rule for profile {profile.name}")
