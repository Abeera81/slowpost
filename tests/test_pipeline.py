"""Revision bookkeeping: the final score and band always come from TabPFN's rescoring."""
from __future__ import annotations

from slowpost.drafts import DevDraft
from slowpost.pipeline import build_items


def draft(title: str) -> DevDraft:
    return DevDraft(title=title, tags=["python"], cover_idea="A sloth", outline=["a", "b", "c"],
                    reading_time_minutes=6, day="Monday", hour=9)


def entry(p: float, label: str) -> dict:
    return {"p": p, "label": label, "reason": "r", "what_if": []}


def test_kept_revision_changes_band_and_keeps_the_original_for_display():
    drafts = [draft("Original one here"), draft("Strong already!"), draft("Original three")]
    scored = [entry(0.12, "Weak"), entry(0.30, "Strong"), entry(0.17, "Too close to call")]
    revised = {0: (draft("Better: 5 lessons"), "ok"), 2: (draft("Worse title here"), "ok")}
    rescored = {0: (entry(0.12, "Weak"), entry(0.18, "Too close to call")),
                2: (entry(0.17, "Too close to call"), entry(0.15, "Too close to call"))}
    items, summary = build_items(drafts, scored, revised, rescored)
    assert items[0]["draft"]["title"] == "Better: 5 lessons" and items[0]["label"] == "Too close to call"
    assert items[0]["original"]["draft"]["title"] == "Original one here"
    assert items[0]["revision"]["kept"] and items[0]["p_yes"] == 0.18
    assert items[1]["revision"]["attempted"] is False and items[1]["label"] == "Strong"
    assert items[2]["draft"]["title"] == "Original three" and not items[2]["revision"]["kept"]
    assert "original kept" in items[2]["revision"]["status"]
    assert items[2]["revision"]["revised_draft"]["title"] == "Worse title here"
    assert summary["attempted"] == 2 and summary["kept"] == 1 and summary["changed_band"] == 1
    assert summary["moves"] == [{"title": "Better: 5 lessons", "from": "Weak",
                                 "to": "Too close to call"}]
    assert summary["strong_before"] == summary["strong_after"] == 1


def test_failed_revision_keeps_the_original_and_says_why():
    items, summary = build_items([draft("Original one here")], [entry(0.1, "Weak")],
                                 {0: (None, "could not revise: x")}, {})
    assert items[0]["label"] == "Weak" and items[0]["revision"]["status"] == "could not revise: x"
    assert summary["attempted"] == 0 and summary["changed_band"] == 0
