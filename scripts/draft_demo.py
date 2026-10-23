"""Draft DEV posts with Gemma and show the features computed by code.

Usage: python scripts/draft_demo.py --brief "..." [--n 5]
Uses data/dev_posts.csv (if present) only for computed summaries; no post text goes to the model.
Prints to the console and saves UTF-8 JSON to data/runs/draft-<timestamp>.json.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from slowpost.data import load_posts
from slowpost.drafts import DEFAULT_N, OllamaUnavailable, generate_drafts
from slowpost.profiles import DEV
from slowpost.runs import save_json
from slowpost.timefmt import slot_display


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brief", required=True)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--facts", default="", help="reference facts about the project (data, not instructions)")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    csv = ROOT / "data" / "dev_posts.csv"
    history = load_posts(csv, DEV) if csv.exists() else None
    t0 = time.perf_counter()
    try:
        results = generate_drafts(args.brief, args.n, history, project_facts=args.facts)
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
        print(f"Reading time: {d.reading_time_minutes} min; suggested "
              f"{slot_display(d.day, d.hour)['utc']} = {slot_display(d.day, d.hour)['local']}")
        if d.notes:
            print("Notes: " + "; ".join(d.notes))
        print(f"Features (computed by code): {r['features']}")
    out = save_json(ROOT / "data" / "runs" / f"draft-{datetime.now():%Y%m%d-%H%M%S}.json", {
        "brief": args.brief, "facts": args.facts, "seconds": round(total, 1),
        "results": [{**r, "draft": r["draft"].model_dump() if r["draft"] else None}
                    for r in results]})
    print(f"Saved {out}")
    ok = sum(r["draft"] is not None for r in results)
    print(f"\n{ok}/{len(results)} drafts valid; total {total:.0f} s")


if __name__ == "__main__":
    main()
