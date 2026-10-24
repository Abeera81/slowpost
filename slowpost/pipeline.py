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
from .plan import plan_csv, run_verdict
from .predict import fit_model, run_backtest
from .profiles import DEV
from .runs import save_json
from .score import (
    STRONG,
    choose_final,
    evidence_text,
    revise_draft,
    revision_evidence,
    score_drafts,
)
from .signals import signal_summary, supported_features, to_jsonable
from .timefmt import slot_display


def _free_gb() -> float:
    return round(psutil.virtual_memory().available / 1e9, 2)


def revise_non_strong(drafts: list, scored: list[dict], evidence: list[str],
                      allowed_tags: list[str] | None, chat: Chat, project_facts: str,
                      say: Callable[[str], None] = print) -> dict[int, tuple]:
    """One Gemma revision for every draft that is not Strong: {index: (draft | None, status)}."""
    out: dict[int, tuple] = {}
    todo = [i for i, s in enumerate(scored) if s["label"] != STRONG]
    for k, i in enumerate(todo, 1):
        out[i] = revise_draft(drafts[i], scored[i]["p"], scored[i]["label"], evidence,
                              allowed_tags, chat, project_facts)
        say(f"Revised {k}/{len(todo)}")
    return out


def build_items(drafts: list, scored: list[dict], revised: dict[int, tuple],
                rescored: dict[int, tuple[dict, dict]]) -> tuple[list[dict], dict]:
    """Final items and the run's revision summary. The final score and label always come from
    TabPFN: a revision is kept only if it scores higher than the original (same fitted model).
    `rescored[i]` = (original entry, revised entry) from score_drafts on the refitted model."""
    items: list[dict] = []
    moves, attempted, kept_n = [], 0, 0
    for i, (d, s0) in enumerate(zip(drafts, scored, strict=True)):
        entry, draft, revision, original = s0, d, {"attempted": False,
                                                    "status": "not needed: already Strong"}, None
        if i in revised:
            new, status = revised[i]
            if new is None:
                revision = {"attempted": True, "kept": False, "status": status}
            else:
                attempted += 1
                so, sr = rescored[i]
                final = choose_final(so, sr)
                kept = final is sr
                entry, draft = final, (new if kept else d)
                if kept:
                    kept_n += 1
                    original = {"draft": d.model_dump(), "p_yes": so["p"], "label": so["label"]}
                    if so["label"] != sr["label"]:
                        moves.append({"title": new.title, "from": so["label"], "to": sr["label"]})
                revision = {
                    "attempted": True, "kept": kept, "p_original": so["p"], "p_revised": sr["p"],
                    "label_original": so["label"], "label_revised": sr["label"],
                    "revised_draft": new.model_dump(),
                    "status": (f"revised version kept: {so['p']:.0%} to {sr['p']:.0%}" if kept else
                               f"original kept: the revision scored {sr['p']:.0%}, not above "
                               f"{so['p']:.0%}")}
        item = {"draft": draft.model_dump(), "when": slot_display(draft.day, draft.hour),
                "features": draft_features(draft), "p_yes": entry["p"], "label": entry["label"],
                "reason": entry["reason"], "what_if": entry["what_if"], "revision": revision}
        if original:
            item["original"] = original
        items.append(item)
    summary = {"attempted": attempted, "kept": kept_n, "changed_band": len(moves), "moves": moves,
               "strong_before": sum(1 for s in scored if s["label"] == STRONG),
               "strong_after": sum(1 for it in items if it["label"] == STRONG)}
    return items, summary


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
    scored = score_drafts(model, drafts, facts, cutoff, strong, supported, summary)
    timed("2b scoring drafts + what-ifs (TabPFN)", t0)
    model.release()  # free TabPFN before Gemma loads again

    # 3. Gemma revises every draft that is not Strong, once. It sees the draft, its score and
    # band, and the backed-by-the-data evidence with observed rates.
    t0 = time.perf_counter()
    revised = revise_non_strong(drafts, scored, revision_evidence(summary),
                                facts.get("allowed_tags"), chat, project_facts, say)
    timed("3 revising drafts (Gemma)", t0)

    # 4. TabPFN rescoring on a freshly fitted model (same sample and seed): original and revised
    # versions together, so the comparison is like for like. The ranking comes from TabPFN only.
    rescored: dict[int, tuple[dict, dict]] = {}
    pairs = [i for i, (new, _status) in revised.items() if new is not None]
    if pairs:
        t0 = time.perf_counter()
        model = fit_model(history, DEV, seed=seed)
        both = [x for i in pairs for x in (drafts[i], revised[i][0])]
        out = score_drafts(model, both, facts, cutoff, strong, supported, summary)
        rescored = {i: (out[2 * k], out[2 * k + 1]) for k, i in enumerate(pairs)}
        model.release()
        timed("4 rescoring revisions (TabPFN)", t0)
    items, revision_summary = build_items(drafts, scored, revised, rescored)
    say(f"Revisions: {revision_summary['kept']} of {revision_summary['attempted']} kept; "
        f"{revision_summary['changed_band']} changed band; strong drafts "
        f"{revision_summary['strong_before']} to {revision_summary['strong_after']}.")
    evidence = evidence_text(res)
    timings["total"] = round(time.perf_counter() - t_start, 1)
    verdict = run_verdict(items)
    say(f"Verdict: {verdict['text']}")
    save_json(run_dir / "drafts.json", {
        "verdict": verdict,
        "brief": brief, "project_facts": project_facts, "label_meaning": evidence,
        "revision_summary": revision_summary,
        "cutoff": cutoff, "strong_threshold": strong, "timings_s": timings,
        "supported_features": sorted(supported),
        "failed_slots": [r["error"] for r in results if not r["draft"]], "drafts": items})
    (run_dir / "plan.csv").write_text(plan_csv(items, verdict["text"]), encoding="utf-8")
    say(f"Total {timings['total']} s. Saved to {run_dir}")
    (run_dir / "log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir
