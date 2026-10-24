"""Headless overnight run for DEV posts: draft, backtest, score, rewrite weak drafts, save.

Usage: python scripts/run_overnight.py --brief "..." [--facts "..."] [--n 5]
       python scripts/run_overnight.py --job data/scheduled/<task>.json   (used by the scheduled task)
Needs data/dev_posts.csv (python scripts/fetch_dev_posts.py) and Ollama running.
Writes data/runs/<timestamp>/{backtest.json,drafts.json,plan.csv,log.txt} (UTF-8, git-ignored).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from slowpost.drafts import DEFAULT_N, OllamaUnavailable
from slowpost.pipeline import run_overnight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brief")
    parser.add_argument("--job", help="JSON file with brief, facts, n and csv (a scheduled run)")
    parser.add_argument("--facts", default="", help="reference facts about the project (data, not instructions)")
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", default=str(ROOT / "data" / "dev_posts.csv"))
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")  # TABPFN_TOKEN lives here; never printed
    runs = ROOT / "data" / "runs"
    if args.job:  # a scheduled run: same files as a run started from the app, so the app shows it
        job = json.loads(Path(args.job).read_text(encoding="utf-8"))
        args.brief, args.facts = job["brief"], job.get("facts", "")
        args.n, args.csv = int(job.get("n", args.n)), job.get("csv", args.csv)
        runs.mkdir(parents=True, exist_ok=True)
        sys.stdout = open(runs / "_current.log", "w", encoding="utf-8", buffering=1)  # noqa: SIM115
        sys.stderr = sys.stdout
        (runs / "_current.json").write_text(json.dumps({
            "pid": os.getpid(), "started": time.time(),
            "started_text": datetime.now().strftime("%H:%M:%S")}), encoding="utf-8")
    if not args.brief or not args.brief.strip():
        sys.exit("Please give a brief (--brief or --job).")
    if not Path(args.csv).exists():
        sys.exit(f"Missing {args.csv}. Run: python scripts/fetch_dev_posts.py")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        run_overnight(args.brief, args.facts, Path(args.csv), runs,
                      n=args.n, seed=args.seed)
    except (ValueError, OllamaUnavailable, RuntimeError) as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
