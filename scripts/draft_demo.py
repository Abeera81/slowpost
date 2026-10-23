"""Draft DEV posts with Gemma and show the features computed by code.

Usage: python scripts/draft_demo.py --brief "..." [--n 5]
Uses data/dev_posts.csv (if present) only for computed summaries; no post text goes to the model.
Prints to the console; nothing is written to disk.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from slowpost.data import load_posts
from slowpost.drafts import DEFAULT_N, OllamaUnavailable, generate_drafts
from slowpost.profiles import DEV


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brief", required=True)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    args = parser.parse_args()

    csv = ROOT / "data" / "dev_posts.csv"
    history = load_posts(csv, DEV) if csv.exists() else None
    t0 = time.perf_counter()
    try:
        results = generate_drafts(args.brief, args.n, history)
    except (ValueError, OllamaUnavailable) as e:
        sys.exit(str(e))
    total = time.perf_counter() - t0

    for i, r in enumerate(results, 1):
        print(f"\n--- Draft {i} (angle: {r['angle']}; {r['seconds']} s) ---")
        if r["draft"] is None:
            print(r["error"])
            continue
        d = r["draft"]
        print(f"Title:   {d.title}\nTags:    {', '.join(d.tags)}\nCover:   {d.cover_idea}")
        print("Outline:\n" + "\n".join(f"  {k}. {p}" for k, p in enumerate(d.outline, 1)))
        print(f"Reading time: {d.reading_time_minutes} min; suggested {d.day} {d.hour:02d}:00 UTC")
        print(f"Features (computed by code): {r['features']}")
    ok = sum(r["draft"] is not None for r in results)
    print(f"\n{ok}/{len(results)} drafts valid; total {total:.0f} s")


if __name__ == "__main__":
    main()
