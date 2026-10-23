"""Throwaway signal check: do the pre-posting features move engagement at all?

Usage: python scripts/signal_check.py [--account 15] [--test-rows 1500]
Prints to the console only; nothing is written to disk.

Part 1: per-feature engagement and label mix, for one account and for all accounts pooled.
        Labels here are thirds of that scope's whole history (descriptive only, not a backtest).
        A Kruskal-Wallis p-value says whether the groups differ more than chance would allow.
Part 2: time-ordered backtest on the pooled data, TabPFN vs most-common-label baseline.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import kruskal

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "Instagram_Analytics.csv"
FEATURES = [
    "media_type", "content_category", "day_of_week", "post_hour",
    "caption_length", "hashtags_count", "has_call_to_action",
]
CATEGORICAL = ["media_type", "content_category", "day_of_week"]
LABELS = ["below", "around", "above"]


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV, parse_dates=["post_datetime"])
    df = df[df["reach"] > 0].sort_values("post_datetime", kind="stable").reset_index(drop=True)
    df["engagement"] = (df["likes"] + df["comments"] + df["shares"] + df["saves"]) / df["reach"]
    return df


def to_label(values: pd.Series, lo: float, hi: float) -> pd.Series:
    return pd.Series(np.where(values <= lo, "below", np.where(values <= hi, "around", "above")),
                     index=values.index)


def groupings(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "media_type": df["media_type"],
        "content_category": df["content_category"],
        "day_of_week": df["day_of_week"],
        "post_hour (4h blocks)": pd.cut(df["post_hour"], [-1, 3, 7, 11, 15, 19, 23],
                                        labels=["00-03", "04-07", "08-11", "12-15", "16-19", "20-23"]),
        "has_call_to_action": df["has_call_to_action"],
        "hashtags_count (4 bins)": pd.qcut(df["hashtags_count"], 4, duplicates="drop"),
    }


def describe(df: pd.DataFrame, name: str) -> list[tuple[str, float]]:
    lo, hi = df["engagement"].quantile([1 / 3, 2 / 3])
    df = df.assign(label=to_label(df["engagement"], lo, hi))
    print(f"\n{'=' * 78}\n{name}: {len(df)} posts, mean engagement {df['engagement'].mean():.4f}, "
          f"thirds at {lo:.4f} / {hi:.4f}\n{'=' * 78}")
    pvalues = []
    for feat, groups in groupings(df).items():
        g = df.groupby(groups, observed=True)
        table = pd.DataFrame({
            "n": g.size(),
            "mean_eng": g["engagement"].mean().round(4),
            "%below": (g["label"].apply(lambda s: (s == "below").mean()) * 100).round(1),
            "%around": (g["label"].apply(lambda s: (s == "around").mean()) * 100).round(1),
            "%above": (g["label"].apply(lambda s: (s == "above").mean()) * 100).round(1),
        })
        samples = [grp["engagement"].values for _, grp in g if len(grp) > 1]
        p = kruskal(*samples).pvalue if len(samples) > 1 else float("nan")
        spread = table["mean_eng"].max() - table["mean_eng"].min()
        print(f"\n{feat}   [Kruskal-Wallis p = {p:.3g}; best-vs-worst group gap = "
              f"{spread:.4f} = {spread / df['engagement'].mean() * 100:.1f}% of the mean]")
        print(table.to_string())
        pvalues.append((feat, p))
    caption_corr = df[["caption_length", "engagement"]].corr(method="spearman").iloc[0, 1]
    print(f"\ncaption_length vs engagement, Spearman correlation: {caption_corr:+.3f}")
    return pvalues


def pooled_backtest(df: pd.DataFrame, test_rows: int) -> None:
    from tabpfn import TabPFNClassifier

    # Oldest 80% is the training pool, newest 20% is the test pool (as in the Tech Design).
    # Train on the most recent 1,000 pool rows (CPU cap); test on the first `test_rows` after the cut.
    cut = int(len(df) * 0.8)
    train, test = df.iloc[cut - 1000: cut], df.iloc[cut: cut + test_rows]
    lo, hi = train["engagement"].quantile([1 / 3, 2 / 3])  # training rows only
    y_train, y_test = to_label(train["engagement"], lo, hi), to_label(test["engagement"], lo, hi)

    both = pd.concat([train, test])[FEATURES].copy()
    for col in CATEGORICAL:
        both[col] = both[col].astype("category").cat.codes
    both = both.astype(float)
    x_train, x_test = both.iloc[: len(train)], both.iloc[len(train):]

    print(f"\n{'=' * 78}\nPOOLED BACKTEST (all accounts, time-ordered)\n{'=' * 78}")
    print(f"train: {len(train)} rows {train['post_datetime'].min()} to {train['post_datetime'].max()}")
    print(f"test:  {len(test)} rows {test['post_datetime'].min()} to {test['post_datetime'].max()}")
    print("train label mix:", y_train.value_counts(normalize=True).round(3).to_dict())
    print("test  label mix:", y_test.value_counts(normalize=True).round(3).to_dict())

    clf = TabPFNClassifier(device="cpu")
    clf.fit(x_train, y_train)
    t0 = time.perf_counter()
    pred = clf.predict(x_test)
    print(f"predict time: {time.perf_counter() - t0:.0f}s")

    base = y_train.mode().iloc[0]
    acc, base_acc = float((pred == y_test.values).mean()), float((y_test == base).mean())
    se = np.sqrt(base_acc * (1 - base_acc) / len(test))
    print(f"TabPFN accuracy:   {acc:.3f}")
    print(f"Baseline ('{base}'): {base_acc:.3f}")
    print(f"Difference: {acc - base_acc:+.3f} (one standard error of an accuracy at this test size "
          f"is about {se:.3f}); chance for 3 balanced labels is 0.333")
    print("TabPFN predicted-label mix:", pd.Series(pred).value_counts(normalize=True).round(3).to_dict())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", type=int, default=15)
    parser.add_argument("--test-rows", type=int, default=1500)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")

    df = load()
    results = {
        f"account {args.account}": describe(df[df["account_id"] == args.account], f"ACCOUNT {args.account}"),
        "pooled": describe(df, "ALL ACCOUNTS POOLED"),
    }
    print(f"\n{'=' * 78}\nKruskal-Wallis p-values (small = groups differ more than chance)\n{'=' * 78}")
    summary = pd.DataFrame({k: dict(v) for k, v in results.items()}).map(lambda p: f"{p:.3g}")
    print(summary.to_string())

    pooled_backtest(df, args.test_rows)


if __name__ == "__main__":
    main()
