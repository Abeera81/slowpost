"""Fetch public DEV posts for a few tags into data/dev_posts.csv (git-ignored).

Usage: python scripts/fetch_dev_posts.py [--days 60] [--max-pages 60]
Keeps posts 14 to (14 + days) days old. Stores no author names or usernames.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from slowpost.devfetch import FIELDS, fetch_tag, utc_now

TAGS = ["webdev", "python", "javascript", "beginners"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60, help="window length beyond the 14-day minimum")
    parser.add_argument("--max-pages", type=int, default=60, help="page cap per tag")
    parser.add_argument("--out", default=str(ROOT / "data" / "dev_posts.csv"))
    args = parser.parse_args()

    now = utc_now()
    rows: list[dict] = []
    for tag in TAGS:
        print(f"Fetching tag {tag}...")
        rows.extend(fetch_tag(tag, now, args.days, args.max_pages))
    df = pd.DataFrame(rows, columns=FIELDS).drop_duplicates("id")
    df = df.sort_values("published_at").reset_index(drop=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} unique posts to {out}")
    if len(df):
        yes = (df["public_reactions_count"] >= 1).mean()
        print(f"Range {df['published_at'].min()} to {df['published_at'].max()}; "
              f"share with at least 1 reaction: {yes:.3f}")


if __name__ == "__main__":
    main()
