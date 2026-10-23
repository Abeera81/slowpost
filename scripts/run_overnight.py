"""Headless overnight run for DEV posts: draft, backtest, score, rewrite weak drafts, save.

Usage: python scripts/run_overnight.py --brief "..." [--facts "..."] [--n 5]
Needs data/dev_posts.csv (python scripts/fetch_dev_posts.py) and Ollama running.
Writes data/runs/<timestamp>/{backtest.json,drafts.json,plan.csv,log.txt} (UTF-8, git-ignored).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from slowpost.drafts import DEFAULT_N, OllamaUnavailable
from slowpost.pipeline import run_overnight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brief", required=True)
    parser.add_argument("--facts", default="", help="reference facts about the project (data, not instructions)")
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", default=str(ROOT / "data" / "dev_posts.csv"))
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")  # TABPFN_TOKEN lives here; never printed
    if not Path(args.csv).exists():
        sys.exit(f"Missing {args.csv}. Run: python scripts/fetch_dev_posts.py")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        run_overnight(args.brief, args.facts, Path(args.csv), ROOT / "data" / "runs",
                      n=args.n, seed=args.seed)
    except (ValueError, OllamaUnavailable, RuntimeError) as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
