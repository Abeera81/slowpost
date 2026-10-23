"""Plan helpers and an app smoke test (no Ollama, no TabPFN)."""
from __future__ import annotations

import csv
import io
from pathlib import Path

from slowpost.drafts import DevDraft, draft_features
from slowpost.plan import apply_edits, plan_csv

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def item() -> dict:
    d = DevDraft(title="Plain title here", tags=["python", "ai"], cover_idea="A sloth",
                 outline=["a", "b", "c"], reading_time_minutes=6, day="Friday", hour=9,
                 notes=["dropped tag 'x'"])
    return {"draft": d.model_dump(), "features": draft_features(d), "p_yes": 0.18,
            "label": "Okay", "reason": "r", "what_if": [],
            "when": {"utc": "Friday 09:00 UTC", "local": "Friday 14:00 PKT"}}


def test_no_edit_leaves_item_alone():
    out, err = apply_edits(item(), None)
    assert err is None and "edited" not in out


def test_valid_edit_recomputes_features_by_code():
    out, err = apply_edits(item(), {"title": "Why 5 tools?", "tags": "python, ai, tools"})
    assert err is None and out["edited"] is True
    assert out["features"]["title_has_question"] == 1 and out["features"]["title_has_number"] == 1
    assert out["features"]["n_tags"] == 3
    assert out["draft"]["day"] == "Friday" and out["draft"]["hour"] == 9  # slot untouched


def test_same_text_is_not_marked_edited():
    out, err = apply_edits(item(), {"title": "Plain title here"})
    assert err is None and out["edited"] is False


def test_invalid_edit_is_refused_and_original_kept():
    original = item()
    out, err = apply_edits(original, {"outline": "only one point"})
    assert err and out is original
    out, err = apply_edits(original, {"tags": "a,b,c,d,e"})
    assert err and out is original


def test_plan_csv_has_all_rows_and_flags_edits():
    edited, _ = apply_edits(item(), {"title": "Edited title here"})
    rows = list(csv.DictReader(io.StringIO(plan_csv([item(), edited]))))
    assert len(rows) == 2 and rows[0]["edited"] == "no" and rows[1]["edited"] == "yes"
    assert rows[1]["title"] == "Edited title here" and "original draft" in rows[1]["label"]
    assert rows[0]["outline"] == "a | b | c" and rows[0]["when_local"] == "Friday 14:00 PKT"


def test_app_pages_render_without_errors():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    for page in ("1 Data", "3 Brief", "4 Plan", "Instagram sample (no signal)"):
        at.sidebar.radio[0].set_value(page)
        at.run()
        assert not at.exception, page


def test_empty_brief_is_refused():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.radio[0].set_value("3 Brief")
    at.run()
    at.button[0].click()
    at.run()
    assert any("write a brief" in e.value for e in at.error)
