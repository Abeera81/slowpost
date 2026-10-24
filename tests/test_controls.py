"""Controlled checks: a gap that is only confounding vanishes inside strata."""
from __future__ import annotations

import numpy as np
import pandas as pd

from slowpost.controls import _odds_ratios, controlled_checks


def _frame(n=20000, seed=0, cover_effect=0.0):
    rng = np.random.default_rng(seed)
    reading = rng.integers(1, 15, n)
    cover = (rng.random(n) < np.clip(reading / 14, 0.05, 0.95)).astype(int)  # long posts have covers
    p = 0.05 + 0.02 * reading + cover_effect * cover
    reactions = (rng.random(n) < p).astype(int)
    return pd.DataFrame({"outcome": reactions, "has_cover_image": cover,
                         "reading_time_minutes": reading})


def test_cover_gap_is_confounded_by_reading_time():
    out = controlled_checks(_frame(cover_effect=0.0))["cover_gap_within_reading_time"]
    assert out["crude_or"] > 1.5  # looks like a cover effect...
    assert 0.8 < out["mh_or"] < 1.25  # ...but it is reading time


def test_a_real_cover_gap_holds_inside_bands():
    out = controlled_checks(_frame(cover_effect=0.15))["cover_gap_within_reading_time"]
    assert out["mh_or"] > 1.5 and out["statement"].startswith(("Holds", "Partly holds"))


def test_odds_ratio_of_a_known_table():
    r = _odds_ratios([(20, 80, 10, 90)])
    assert abs(r["crude_or"] - 2.25) < 1e-9 and abs(r["mh_or"] - 2.25) < 1e-9
