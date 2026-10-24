"""Overnight run for the DEV source: draft -> backtest + score -> rewrite weak -> rescore -> files.

Gemma and TabPFN are never in memory together: Gemma unloads itself (keep_alive=0), and the
fitted TabPFN model is released before the rewrite step.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import psutil

from .data import load_posts
from .drafts import (
    DEFAULT_N,
    Chat,
    draft_features,
    generate_drafts,
    history_facts,
    ollama_chat,
)
from .labels import apply_labels, fit_thresholds
from .plan import plan_csv
from .predict import fit_model, run_backtest
from .profiles import DEV
from .runs import save_json
from .score import (
    evidence_text,
    label_for,
    p_yes,
    rewrite_draft,
    score_drafts,
    text_change,
)
from .signals import signal_summary, supported_features, to_jsonable
from .timefmt import slot_display


def _free_gb() -> float:
    return round(psutil.virtual_memory().available / 1e9, 2)


def run_overnight(brief: str, project_facts: str, history_csv: Path, out_root: Path,
                  n: int = DEFAULT_N, seed: int = 0, chat: Chat = ollama_chat,
                  log: Callable[[str], None] = print) -> Path:
    """Run everything and write data/runs/<timestamp>/. Returns the run folder."""
    run_dir = out_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    timings: dict[str, float] = {}
    t_start = time.perf_counter()

    def say(msg: str) -> None:
        stamp = f"[{time.perf_counter() - t_start:6.0f}s] {msg}"
        lines.append(stamp)
        log(stamp)

    def timed(name: str, t0: float) -> None:
        timings[name] = round(time.perf_counter() - t0, 1)
        say(f"{name}: {timings[name]} s (free RAM {_free_gb()} GB)")

    say(f"Run folder: {run_dir}")
    history = load_posts(history_csv, DEV)
    facts = history_facts(history)
    say(f"History: {len(history)} DEV posts. Free RAM {_free_gb()} GB.")

    # 1. Gemma drafts (then Gemma unloads itself).
    t0 = time.perf_counter()
    results = generate_drafts(brief, n, history, chat, project_facts,
                              on_draft=lambda done, total: say(f"Drafted {done}/{total}"))
    timed("1 drafting (Gemma)", t0)
    drafts = [r["draft"] for r in results if r["draft"]]
    if not drafts:
        raise RuntimeError("Gemma could not produce any valid draft. See log.txt.")

    # 2. TabPFN: backtest, thresholds, score drafts and what-ifs.
    t0 = time.perf_counter()
    res, model = run_backtest(history, DEV, seed=seed)
    save_json(run_dir / "backtest.json", res)
    timed("2a backtest (TabPFN fit + predict test window)", t0)
    say(f"Backtest verdict: {res['verdict']}")
    cutoff, strong = res["cutoff"], res["strong_threshold"]
    labels = apply_labels(history["outcome"], fit_thresholds(history["outcome"], DEV), DEV)
    summary = signal_summary(history, labels, DEV)
    supported = supported_features(summary)
    save_json(run_dir / "backtest.json", {**res, "signals": to_jsonable(summary)})
    say("Features with a clear difference in the signal summary: "
        + (", ".join(sorted(supported)) or "none"))
    t0 = time.perf_counter()
    scored = score_drafts(model, drafts, facts, cutoff, strong, supported)
    timed("2b scoring drafts + what-ifs (TabPFN)", t0)
    model.release()  # free TabPFN before Gemma loads again

    # 3. Gemma rewrites weak drafts once, if a title change is expected to help.
    t0 = time.perf_counter()
    rewrites: dict[int, dict] = {}
    for i, (d, s) in enumerate(zip(drafts, scored, strict=True)):
        if s["label"] != "Weak":
            continue
        change = text_change(s)
        if change is None:
            rewrites[i] = {"status": "skipped: no change backed by the data is expected "
                                     "to improve the score"}
            continue
        new, status = rewrite_draft(d, change, chat, project_facts)
        rewrites[i] = {"status": status, "change": change["text"], "draft": new}
    timed("3 rewriting weak drafts (Gemma)", t0)

    # 4. Rescore originals and rewrites with a freshly fitted model (same sample and seed).
    done = {i: r for i, r in rewrites.items() if r.get("draft")}
    if done:
        t0 = time.perf_counter()
        model = fit_model(history, DEV, seed=seed)
        rows = []
        for i, r in done.items():
            rows += [draft_features(drafts[i]), draft_features(r["draft"])]
        probs = p_yes(model, rows)
        for k, (i, r) in enumerate(done.items()):
            r["p_before"], r["p_after"] = float(probs[2 * k]), float(probs[2 * k + 1])
            r["label_after"] = label_for(r["p_after"], cutoff, strong)
            r["improved"] = r["p_after"] - r["p_before"] >= 0.01
        model.release()
        timed("4 rescoring rewrites (TabPFN)", t0)

    # Save.
    evidence = evidence_text(res)
    items = []
    for i, (d, s) in enumerate(zip(drafts, scored, strict=True)):
        r = rewrites.get(i)
        item = {
            "draft": d.model_dump(), "when": slot_display(d.day, d.hour),
            "features": draft_features(d), "p_yes": s["p"], "label": s["label"],
            "reason": s["reason"], "what_if": s["what_if"],
        }
        if r:
            item["rewrite"] = {
                **{k: v for k, v in r.items() if k != "draft"},
                "draft": r["draft"].model_dump() if r.get("draft") else None,
                "features": draft_features(r["draft"]) if r.get("draft") else None,
            }
        items.append(item)
    timings["total"] = round(time.perf_counter() - t_start, 1)
    save_json(run_dir / "drafts.json", {
        "brief": brief, "project_facts": project_facts, "label_meaning": evidence,
        "cutoff": cutoff, "strong_threshold": strong, "timings_s": timings,
        "supported_features": sorted(supported),
        "failed_slots": [r["error"] for r in results if not r["draft"]], "drafts": items})
    (run_dir / "plan.csv").write_text(plan_csv(items), encoding="utf-8")
    say(f"Total {timings['total']} s. Saved to {run_dir}")
    (run_dir / "log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir
