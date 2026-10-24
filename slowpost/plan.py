"""Plan helpers for the UI: apply the user's edits and build the plan.csv download."""
from __future__ import annotations

import csv
import io

from .drafts import DevDraft, DraftText, draft_features

PLAN_COLUMNS = ["run_verdict", "when_utc", "when_local", "title", "tags", "tag_notes", "cover_idea", "outline",
                "reading_time_minutes", "p_yes", "label", "edited", "best_change",
                "revision", "original_title", "original_p_yes", "original_label"]

LEGACY_LABELS = {"Okay": "Too close to call"}  # runs saved before the rename


def normalize_labels(data: dict) -> dict:
    """Saved runs from before the rename say "Okay"; show them as "Too close to call"."""
    fix = lambda label: LEGACY_LABELS.get(label, label)
    data["label_meaning"] = {fix(k): v for k, v in data["label_meaning"].items()}
    for it in data["drafts"]:
        it["label"] = fix(it["label"])
        if it.get("original"):
            it["original"]["label"] = fix(it["original"]["label"])
        for k in ("label_original", "label_revised"):
            if (it.get("revision") or {}).get(k):
                it["revision"][k] = fix(it["revision"][k])
    if data.get("verdict", {}).get("best"):
        data["verdict"]["best"]["label"] = fix(data["verdict"]["best"]["label"])
    return data


def apply_edits(item: dict, edit: dict | None) -> tuple[dict, str | None]:
    """Apply title/tags/cover/outline edits to a saved draft item. The text is validated like
    Gemma output; invalid edits are refused with a plain message. Features are recomputed by
    code. Returns (item, error). The saved score belongs to the ORIGINAL draft."""
    if not edit:
        return item, None
    d = dict(item["draft"])
    try:
        text = DraftText(
            title=edit.get("title", d["title"]),
            tags=[t for t in edit.get("tags", ",".join(d["tags"])).replace(" ", ",").split(",") if t],
            cover_idea=edit.get("cover_idea", d["cover_idea"]),
            outline=[ln for ln in edit.get("outline", "\n".join(d["outline"])).splitlines()
                     if ln.strip()],
            reading_time_minutes=d["reading_time_minutes"])
    except ValueError as e:
        return item, "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}"
                               for x in getattr(e, "errors", list)()[:3]) or str(e)
    new = DevDraft(**{**text.model_dump(), "day": d["day"], "hour": d["hour"],
                      "notes": d["notes"]})
    changed = new.model_dump() != d
    out = {**item, "draft": new.model_dump(), "features": draft_features(new), "edited": changed}
    return out, None


def run_verdict(items: list[dict]) -> dict:
    """The top-line verdict for a run, from the final TabPFN scores. If no draft is Strong the
    plan says so and names the best candidate anyway."""
    n = sum(1 for it in items if it["label"] == "Strong")
    best = max(items, key=lambda it: it["p_yes"]) if items else None
    text = ("Nothing stands out this week. Go back to sleep." if n == 0 else
            f"{n} strong idea{'s' if n != 1 else ''} this week.")
    return {"text": text, "n_strong": n,
            "best": None if best is None else {
                "title": best["draft"]["title"], "p_yes": best["p_yes"],
                "label": best["label"], "when_utc": best["when"]["utc"]}}


def plan_csv(items: list[dict], verdict: str | None = None) -> str:
    """plan.csv text for the (possibly edited) items. Every row repeats the run verdict."""
    verdict = verdict or (run_verdict(items)["text"] if items else "")
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=PLAN_COLUMNS, lineterminator="\n")
    w.writeheader()
    for it in items:
        d, rv, orig = it["draft"], it.get("revision") or {}, it.get("original") or {}
        w.writerow({
            "run_verdict": verdict, "when_utc": it["when"]["utc"], "when_local": it["when"]["local"],
            "title": d["title"], "tags": ",".join(d["tags"]), "tag_notes": "; ".join(d["notes"]),
            "cover_idea": d["cover_idea"], "outline": " | ".join(d["outline"]),
            "reading_time_minutes": d["reading_time_minutes"], "p_yes": f"{it['p_yes']:.3f}",
            "label": it["label"] + (" (score is for the original draft)" if it.get("edited") else ""),
            "edited": "yes" if it.get("edited") else "no", "best_change": it["reason"],
            "revision": rv.get("status", ""),
            "original_title": (orig.get("draft") or {}).get("title", ""),
            "original_p_yes": f"{orig['p_yes']:.3f}" if "p_yes" in orig else "",
            "original_label": orig.get("label", ""),
        })
    return buf.getvalue()
