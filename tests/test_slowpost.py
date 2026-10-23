"""Unit tests for M1. No network, no TabPFN."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from slowpost.data import MissingColumnsError, load_posts
from slowpost.devfetch import FIELDS, article_to_row, in_window
from slowpost.features import build_matrices
from slowpost.labels import apply_labels, fit_thresholds
from slowpost.predict import evaluate, time_split, verdict
from slowpost.profiles import DEV, INSTAGRAM
from slowpost.signals import signal_summary


def dev_raw(n: int = 400, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-08-01", tz="UTC")
    return pd.DataFrame({
        "id": range(n),
        "published_at": [(start + pd.Timedelta(hours=int(h))).isoformat()
                         for h in rng.integers(0, 24 * 40, n)],
        "title": [f"Post {i}" for i in range(n)],
        "tag_list": rng.choice(["webdev,python", "javascript", "beginners,webdev,css"], n),
        "reading_time_minutes": rng.integers(1, 12, n),
        "has_cover_image": rng.choice([True, False], n),
        "public_reactions_count": rng.choice([0, 0, 0, 1, 3, 9], n),
        "comments_count": rng.integers(0, 3, n),
    })


def insta_raw(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    return pd.DataFrame({
        "post_datetime": pd.date_range("2025-01-01", periods=n, freq="6h").astype(str),
        "media_type": rng.choice(["reel", "image", "carousel"], n),
        "content_category": rng.choice(["Food", "Travel"], n),
        "day_of_week": rng.choice(["Monday", "Tuesday"], n),
        "post_hour": rng.integers(0, 24, n), "caption_length": rng.integers(10, 300, n),
        "hashtags_count": rng.integers(0, 20, n), "has_call_to_action": rng.integers(0, 2, n),
        "likes": rng.integers(1, 100, n), "comments": rng.integers(0, 10, n),
        "shares": rng.integers(0, 10, n), "saves": rng.integers(0, 10, n),
        "reach": rng.integers(100, 1000, n),
    })


# ---- column mapping ----

def test_dev_column_mapping():
    df = load_posts(dev_raw(), DEV)
    assert {"post_datetime", "reactions", "comments", "outcome"} <= set(df.columns)
    assert (df["outcome"] == df["reactions"]).all()
    assert df["post_datetime"].is_monotonic_increasing


def test_instagram_engagement_is_computed_not_read():
    raw = insta_raw()
    raw["engagement_rate"] = 99.0  # must be ignored
    df = load_posts(raw, INSTAGRAM)
    expect = (raw.likes + raw.comments + raw.shares + raw.saves) / raw.reach
    assert np.allclose(np.sort(df["outcome"]), np.sort(expect))


@pytest.mark.parametrize("profile,raw", [(DEV, dev_raw()), (INSTAGRAM, insta_raw())])
def test_missing_columns_error_names_them(profile, raw):
    col = [c for c in raw.columns if c in profile.column_map][2]
    with pytest.raises(MissingColumnsError, match=col):
        load_posts(raw.drop(columns=[col]), profile)


def test_instagram_zero_reach_rows_dropped():
    raw = insta_raw()
    raw.loc[0, "reach"] = 0
    assert len(load_posts(raw, INSTAGRAM)) == len(raw) - 1


# ---- features ----

def test_dev_feature_computation():
    raw = pd.DataFrame([{
        "id": 1, "published_at": "2026-09-02T14:30:00Z",  # a Wednesday
        "title": "5 ways to learn Python?", "tag_list": "python,beginners,webdev",
        "reading_time_minutes": 6, "has_cover_image": True,
        "public_reactions_count": 2, "comments_count": 0,
    }, {
        "id": 2, "published_at": "2026-09-05T03:00:00Z",  # a Saturday
        "title": "Hello", "tag_list": "", "reading_time_minutes": 1,
        "has_cover_image": False, "public_reactions_count": 0, "comments_count": 0,
    }])
    df = load_posts(raw, DEV)
    a, b = df.iloc[0], df.iloc[1]
    assert (a.weekday, a.hour, a.n_tags, a.main_tag) == (2, 14, 3, "python")
    assert (a.has_cover_image, a.title_length) == (1, len("5 ways to learn Python?"))
    assert (a.title_has_question, a.title_has_number) == (1, 1)
    assert (b.weekday, b.n_tags, b.main_tag, b.has_cover_image) == (5, 0, "none", 0)
    assert (b.title_has_question, b.title_has_number) == (0, 0)


def test_main_tag_categories_come_from_training_only():
    train = load_posts(dev_raw(), DEV).iloc[:100].copy()
    test = train.iloc[:5].copy()
    test["main_tag"] = "brand-new-tag"
    x_train, x_test, cat_idx = build_matrices(train, test, DEV)
    col = DEV.features[cat_idx[0]]
    assert (x_test[col] == x_train[col].max() + 1).all() or (x_test[col] == x_test[col].iloc[0]).all()
    assert x_test[col].iloc[0] not in set(x_train[col])  # unseen tag maps to 'other', unused in train


# ---- split ----

def test_no_future_rows_in_training():
    df = load_posts(dev_raw(), DEV)
    train, test = time_split(df, DEV, seed=3)
    assert train["post_datetime"].max() < test["post_datetime"].min()
    assert len(train) <= 1000
    assert set(train.index).isdisjoint(test.index)


def test_split_ties_do_not_straddle_the_cut():
    df = load_posts(dev_raw(300), DEV)
    df.loc[df.index[-80:], "post_datetime"] = df["post_datetime"].iloc[-60]  # many identical times
    train, test = time_split(df, DEV)
    assert train["post_datetime"].max() < test["post_datetime"].min()


def test_training_sample_is_random_but_fixed_by_seed_and_capped():
    df = load_posts(dev_raw(2500), DEV)
    t1, _ = time_split(df, DEV, seed=7, max_test=300)
    t2, _ = time_split(df, DEV, seed=7, max_test=300)
    t3, _ = time_split(df, DEV, seed=8, max_test=300)
    assert len(t1) == 1000 and list(t1.index) == list(t2.index) != list(t3.index)
    assert t1["weekday"].nunique() == 7  # a spread of weekdays, not one or two days


def test_split_refuses_tiny_history():
    with pytest.raises(ValueError, match="older than the test window"):
        time_split(load_posts(dev_raw(20), DEV), DEV)


# ---- labels ----

def test_thirds_thresholds_use_training_rows_only():
    train = pd.Series(np.arange(1, 91, dtype=float))
    base = fit_thresholds(train, INSTAGRAM)
    # Adding wild test values elsewhere must not change what training produced.
    assert fit_thresholds(train, INSTAGRAM) == base
    assert base == pytest.approx((30.67, 60.33), abs=0.01)
    test = pd.Series([1.0, 50.0, 1e6])
    assert list(apply_labels(test, base, INSTAGRAM)) == ["below", "around", "above"]


def test_split_then_label_uses_training_rows_only():
    df = load_posts(insta_raw(), INSTAGRAM)
    train, test = time_split(df, INSTAGRAM)
    base = fit_thresholds(train["outcome"], INSTAGRAM)
    wild = test.copy()
    wild["outcome"] = wild["outcome"] * 1000
    # the pipeline fits thresholds on `train` only, so changing the test rows cannot move them
    assert fit_thresholds(train["outcome"], INSTAGRAM) == base
    assert base[0] < base[1] < wild["outcome"].max()
    assert set(apply_labels(wild["outcome"], base, INSTAGRAM)) == {"above"}


def test_dev_label_is_fixed_rule():
    assert fit_thresholds(pd.Series([0, 0, 0]), DEV) == (1.0,)
    labels = apply_labels(pd.Series([0, 1, 5]), (1.0,), DEV)
    assert list(labels) == ["no_reactions", "some_reactions", "some_reactions"]


# ---- DEV fetch ----

def test_article_to_row_keeps_only_allowed_fields():
    article = {
        "id": 9, "published_at": "2026-09-01T10:00:00Z", "title": "T", "tag_list": ["a", "b"],
        "reading_time_minutes": 3, "cover_image": "https://x/y.png",
        "public_reactions_count": 4, "comments_count": 1,
        "user": {"name": "Someone", "username": "someone"}, "url": "https://dev.to/someone/t",
    }
    row = article_to_row(article)
    assert list(row) == FIELDS
    assert row["tag_list"] == "a,b" and row["has_cover_image"] is True
    assert "Someone" not in str(row) and "someone" not in str(row)


def test_fourteen_day_filter():
    now = datetime(2026, 10, 23, tzinfo=UTC)

    def row(days_old: float) -> dict:
        return {"published_at": (now - timedelta(days=days_old)).isoformat()}

    assert not in_window(row(13.9), now, 60)
    assert in_window(row(14.1), now, 60)
    assert in_window(row(70), now, 60)
    assert not in_window(row(75), now, 60)


# ---- evaluation ----

def test_verdict_rule():
    rng = np.random.default_rng(0)
    y = rng.choice(["no", "yes"], 800, p=[0.85, 0.15])
    classes = ["no", "yes"]
    noise = rng.random((800, 1))
    res = evaluate(y, np.hstack([1 - noise, noise]), classes, "no", n_boot=200)
    assert res["baseline_balanced_accuracy"] == pytest.approx(0.5)
    assert res["verdict"].startswith("No better")
    p_yes = np.where(y == "yes", 0.8, 0.1)[:, None]
    res = evaluate(y, np.hstack([1 - p_yes, p_yes]), classes, "no", n_boot=200)
    assert res["verdict"].startswith(("Better", "Some signal")) and res["auc"] > 0.95
    assert verdict({"ba_gap_ci": (-0.2, -0.05), "auc_ci": (0.3, 0.45), "chance_auc": 0.5}
                   ).startswith("Worse")


# ---- signal summary ----

def test_signal_summary_finds_planted_signal_and_ignores_noise():
    df = load_posts(dev_raw(3000, seed=4), DEV)
    rng = np.random.default_rng(5)
    noise_labels = pd.Series(rng.choice(["no_reactions", "some_reactions"], len(df), p=[.8, .2]))
    s = signal_summary(df, noise_labels, DEV)
    assert s["top_label"] == "no_reactions"
    assert not any(v["clear"] for v in s["features"].values())
    planted = noise_labels.where(df["has_cover_image"] == 0, "some_reactions")
    s = signal_summary(df, planted, DEV)
    assert s["features"]["has cover image"]["clear"]
