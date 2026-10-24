"""Run the time-ordered backtest and signal summary for one data source.

Usage:
  python scripts/run_backtest.py --source dev
  python scripts/run_backtest.py --source instagram          # sample/sample_posts.csv
  python scripts/run_backtest.py --source instagram --csv data/Instagram_Analytics.csv --account 15
Prints to the console. With --save, also writes data/backtests/<source>.json (git-ignored).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from slowpost.compare import NAMES
from slowpost.controls import controlled_checks
from slowpost.data import load_posts
from slowpost.labels import apply_labels, fit_thresholds
from slowpost.predict import backtest
from slowpost.profiles import DEV, INSTAGRAM
from slowpost.runs import save_json
from slowpost.signals import describe, signal_summary, supported_features, to_jsonable

SOURCES = {
    "dev": (DEV, ROOT / "data" / "dev_posts.csv"),
    "instagram": (INSTAGRAM, ROOT / "sample" / "sample_posts.csv"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--csv")
    parser.add_argument("--account", type=int, help="Instagram only: keep one account_id")
    parser.add_argument("--save", action="store_true", help="write data/backtests/<source>.json for the app")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")  # TABPFN_TOKEN lives here; never printed

    profile, default_csv = SOURCES[args.source]
    csv = Path(args.csv) if args.csv else default_csv
    where = {"account_id": args.account} if args.account else None
    df = load_posts(csv, profile, where=where)
    print(f"{profile.title}: {len(df)} usable posts, {df[profile.time_col].min()} "
          f"to {df[profile.time_col].max()}")

    # Signal summary: descriptive, labels from all rows.
    labels = apply_labels(df["outcome"], fit_thresholds(df["outcome"], profile), profile)
    print("\nLabel mix (all rows):", labels.value_counts(normalize=True).round(3).to_dict())
    summary = signal_summary(df, labels, profile)
    print(f"Top label: {summary['top_label']} ({summary['overall_share']:.1%} of posts)")
    for name, v in summary["features"].items():
        print(f"  {name:<26} p = {v['p']:.3g}, corrected {v['p_adj']:.3g}"
              f"{'  <- clear' if v['clear'] else ''}")
    print(describe(summary))

    print("\nBacktest (TabPFN, local, CPU)...")
    r = backtest(df, profile, compare=True)
    print(f"train: {r['n_train']} posts sampled from {r['train_range'][0]} to {r['train_range'][1]}")
    print(f"test:  {r['n_test']} posts, {r['test_window']}, "
          f"{r['test_range'][0]} to {r['test_range'][1]}")
    if r["cutoff"] is not None:
        print(f"cutoff (training share of '{r['positive']}'): predict it when P > {r['cutoff']:.3f}")
    print(f"label thresholds (training rows only): {r['thresholds']}")
    print(f"train mix {r['train_mix']}; test mix {r['test_mix']}")
    print(f"TabPFN predicted mix {r['predicted_mix']} ({r['predict_seconds']} s)")
    print(f"Baseline always says '{r['baseline_label']}'")
    print(f"balanced accuracy at that cutoff: TabPFN {r['balanced_accuracy']:.3f} vs baseline "
          f"{r['baseline_balanced_accuracy']:.3f}; gap {r['ba_gap']:+.3f} "
          f"(95% CI {r['ba_gap_ci'][0]:+.3f} to {r['ba_gap_ci'][1]:+.3f})")
    print(f"ROC AUC: {r['auc']:.3f} (95% CI {r['auc_ci'][0]:.3f} to {r['auc_ci'][1]:.3f}; "
          f"0.5 = chance)")
    for name, v in r.get("lift", {}).items():
        print(f"lift, model's {name[3:]}% highest-ranked posts (n={v['n_top']}): "
              f"{v['share']:.1%} got reactions (95% CI {v['share_ci'][0]:.1%} to "
              f"{v['share_ci'][1]:.1%}) vs {r['overall_share']:.1%} overall in the test window "
              f"(CI {r['overall_share_ci'][0]:.1%} to {r['overall_share_ci'][1]:.1%}); "
              f"lift {v['lift']:.2f}x (CI {v['lift_ci'][0]:.2f} to {v['lift_ci'][1]:.2f})")
    print(f"raw accuracy (reference only): TabPFN {r['accuracy']:.3f} vs baseline "
          f"{r['baseline_accuracy']:.3f}")
    cmp = r.get("compare")
    if cmp:
        print("\nSame split, four models (balanced accuracy / ROC AUC / lift in the top 10%):")
        for key, row in cmp["models"].items():
            top = row.get("top10")
            print(f"  {NAMES[key]:<22} BA {row['balanced_accuracy']:.3f}  AUC {row['auc']:.3f}"
                  + (f"  top-10% lift {top['lift']:.2f}x" if top else ""))
        for key, d in cmp["tabpfn_minus"].items():
            print(f"  TabPFN minus {NAMES[key]}: AUC {d['auc_diff']:+.3f} "
                  f"(paired 95% CI {d['auc_diff_ci'][0]:+.3f} to {d['auc_diff_ci'][1]:+.3f})")
        print(cmp["winner"])
        rs = cmp["random_split"]
        print(f"TabPFN on a random 80/20 split (for comparison only): BA "
              f"{rs['balanced_accuracy']:.3f}, AUC {rs['auc']:.3f}")
    print(f"\nVERDICT: {r['verdict']}")
    if args.save:
        out = save_json(ROOT / "data" / "backtests" / f"{args.source}.json", {
            "source": args.source, "title": profile.title, "n_posts": len(df),
            "first_post": str(df[profile.time_col].min()), "last_post": str(df[profile.time_col].max()),
            "label_mix": labels.value_counts(normalize=True).round(3).to_dict(),
            "result": r, "signals": to_jsonable(summary),
            "controls": controlled_checks(df) if profile.name == "dev_posts" else None,
            "supported_features": sorted(supported_features(summary)) if profile.name == "dev_posts" else []})
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
