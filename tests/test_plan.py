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
            "label": "Too close to call", "reason": "r", "what_if": [],
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
    for page in ("0 Home", "1 Data", "3 Brief", "4 Plan", "Instagram sample (no signal)"):
        at.session_state["nav"] = page
        at.run()
        assert not at.exception, page


def test_empty_brief_is_refused():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=30).run()
    at.session_state["nav"] = "3 Brief"
    at.run()
    at.main.button[0].click()
    at.run()
    assert any("write a brief" in e.value for e in at.error)


def test_palette_text_contrast_is_aa_and_config_matches():
    import tomllib

    from slowpost.ui import PALETTE, contrast

    p = PALETTE
    for kind in ("strong", "close", "weak"):
        assert contrast(p[f"{kind}_fg"], p[f"{kind}_bg"]) >= 4.5, kind
    assert contrast(p["muted"], p["bg"]) >= 4.5 and contrast(p["muted"], p["panel"]) >= 4.5
    assert contrast(p["text"], p["bg"]) >= 4.5 and contrast("#ffffff", p["primary"]) >= 4.5
    theme = tomllib.loads((Path(APP).parent / ".streamlit" / "config.toml").read_text())["theme"]
    assert (theme["primaryColor"], theme["backgroundColor"], theme["textColor"],
            theme["secondaryBackgroundColor"]) == (p["primary"], p["bg"], p["text"], p["panel"])


def test_css_builds_with_palette_variables():
    from slowpost.ui import PALETTE, css

    out = css()
    assert "--primary: " + PALETTE["primary"] in out and "/*VARS*/" not in out


def test_badge_has_word_and_glyph():
    from slowpost.ui import badge

    for label in ("Strong", "Too close to call", "Weak"):
        assert label in badge(label)
        assert {"Strong": "sp-strong", "Too close to call": "sp-close", "Weak": "sp-weak"}[label] in badge(label)


def test_progress_steps_follow_the_log():
    from slowpost.ui import parse_progress

    def states(log):
        return [s["state"] for s in parse_progress(log)]

    assert states("[0s] History: 10 DEV posts.") == ["active", "wait", "wait", "wait"]
    drafted = "[233s] 1 drafting (Gemma): 232.1 s (free RAM 2.3 GB)"
    steps = parse_progress(drafted)
    assert [s["state"] for s in steps] == ["done", "active", "wait", "wait"]
    assert steps[0]["seconds"] == 232.1
    scored = (drafted + "\n[358s] 2a backtest (TabPFN fit): 125.7 s (free)"
              "\n[398s] 2b scoring drafts: 39.6 s (f")
    assert states(scored) == ["done", "done", "done", "active"]
    assert states(scored + "\nTotal 399 s. Saved to x") == ["done"] * 4


def test_progress_shows_drafting_count_while_drafting():
    from slowpost.ui import parse_progress

    steps = parse_progress("[40s] Drafted 1/5\n[90s] Drafted 2/5")
    assert steps[0]["state"] == "active" and steps[0]["name"] == "Drafting with Gemma 2/5"


def _scored(label: str, p: float) -> dict:
    return {**item(), "label": label, "p_yes": p}


def test_verdict_with_no_strong_names_the_best_candidate():
    from slowpost.plan import run_verdict

    v = run_verdict([_scored("Too close to call", 0.15), _scored("Weak", 0.10)])
    assert v["text"] == "Nothing stands out this week. Go back to sleep." and v["n_strong"] == 0
    assert v["best"]["p_yes"] == 0.15


def test_verdict_counts_strong_ideas():
    from slowpost.plan import run_verdict

    assert run_verdict([_scored("Strong", 0.3)])["text"] == "1 strong idea this week."
    assert run_verdict([_scored("Strong", 0.3), _scored("Strong", 0.25)])["text"] == \
        "2 strong ideas this week."


def test_csv_carries_the_verdict():
    from slowpost.plan import run_verdict

    items = [_scored("Weak", 0.1)]
    rows = list(csv.DictReader(io.StringIO(plan_csv(items, run_verdict(items)["text"]))))
    assert rows[0]["run_verdict"] == "Nothing stands out this week. Go back to sleep."


def test_old_runs_with_okay_are_shown_with_the_new_name():
    from slowpost.plan import normalize_labels

    data = {"label_meaning": {"Okay": "x", "Weak": "y"}, "drafts": [{**item(), "label": "Okay"}],
            "verdict": {"best": {"label": "Okay"}}}
    out = normalize_labels(data)
    assert set(out["label_meaning"]) == {"Too close to call", "Weak"}
    assert out["drafts"][0]["label"] == "Too close to call"
    assert out["verdict"]["best"]["label"] == "Too close to call"


def test_csv_has_revision_columns():
    it = {**item(), "revision": {"status": "revised version kept: 14% to 22%"},
          "original": {"draft": {"title": "Old title"}, "p_yes": 0.14, "label": "Weak"}}
    row = next(csv.DictReader(io.StringIO(plan_csv([it]))))
    assert row["revision"].startswith("revised version kept") and row["original_title"] == "Old title"
    assert row["original_p_yes"] == "0.140" and row["original_label"] == "Weak"
