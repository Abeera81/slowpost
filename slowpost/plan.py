"""Plan helpers for the UI: apply the user's edits and build the plan.csv download."""
from __future__ import annotations

import csv
import io

from .drafts import DevDraft, DraftText, draft_features

PLAN_COLUMNS = ["when_utc", "when_local", "title", "tags", "tag_notes", "cover_idea", "outline",
                "reading_time_minutes", "p_yes", "label", "edited", "best_change",
                "rewrite_title", "rewrite_p_yes", "rewrite_label", "rewrite_note"]


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


def plan_csv(items: list[dict]) -> str:
    """plan.csv text for the (possibly edited) items."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=PLAN_COLUMNS, lineterminator="\n")
    w.writeheader()
    for it in items:
        d, rw = it["draft"], it.get("rewrite") or {}
        rd = rw.get("draft") or {}
        w.writerow({
            "when_utc": it["when"]["utc"], "when_local": it["when"]["local"],
            "title": d["title"], "tags": ",".join(d["tags"]), "tag_notes": "; ".join(d["notes"]),
            "cover_idea": d["cover_idea"], "outline": " | ".join(d["outline"]),
            "reading_time_minutes": d["reading_time_minutes"], "p_yes": f"{it['p_yes']:.3f}",
            "label": it["label"] + (" (score is for the original draft)" if it.get("edited") else ""),
            "edited": "yes" if it.get("edited") else "no", "best_change": it["reason"],
            "rewrite_title": rd.get("title", ""),
            "rewrite_p_yes": f"{rw['p_after']:.3f}" if "p_after" in rw else "",
            "rewrite_label": rw.get("label_after", ""), "rewrite_note": rw.get("status", ""),
        })
    return buf.getvalue()
