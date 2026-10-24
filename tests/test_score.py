"""Scoring tests: labels, what-ifs, revision, tag/title rules, time display, saved files.
A fake model stands in for TabPFN; a fake chat stands in for Gemma."""
from __future__ import annotations

import json
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from slowpost.drafts import (
    DevDraft,
    DraftText,
    draft_one,
    generate_drafts,
    sanitize_tags,
    titles_similar,
)
from slowpost.predict import score_groups
from slowpost.runs import save_json
from slowpost.score import (
    MIN_EFFECT,
    choose_final,
    evidence_text,
    label_for,
    revise_draft,
    revision_evidence,
    score_drafts,
    whatif_variants,
)
from slowpost.timefmt import slot_display

FACTS = {
    "n_posts": 100, "tag_rates": {"webdev": 0.2, "python": 0.1, "ai": 0.3},
    "median_reading_time": 5.0, "best_days": ["Friday", "Monday", "Tuesday"],
    "best_hours": [9, 14, 20], "allowed_tags": ["webdev", "python", "ai", "tabpfn"],
}


def make_draft(**kw) -> DevDraft:
    base = {"title": "Plain title here", "tags": ["python", "ai"], "cover_idea": "A sloth",
                "outline": ["a", "b", "c"], "reading_time_minutes": 6, "day": "Wednesday", "hour": 12}
    return DevDraft(**{**base, **kw})


class FakeModel:
    """P(yes) rises with a cover image, a title question and a number; otherwise 0.10."""
    classes = ("no_reactions", "some_reactions")

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        p = (0.10 + 0.10 * frame["has_cover_image"] + 0.05 * frame["title_has_question"]
             + 0.03 * frame["title_has_number"]).to_numpy()
        return np.column_stack([1 - p, p])


# ---- labels ----

def test_label_boundaries():
    assert label_for(0.30, 0.14, 0.30) == "Strong"  # at the top-20% threshold counts
    assert label_for(0.29, 0.14, 0.30) == "Too close to call"
    assert label_for(0.14, 0.14, 0.30) == "Too close to call"  # at the cutoff is not weak
    assert label_for(0.139, 0.14, 0.30) == "Weak"


def test_score_groups_and_evidence_text():
    y = np.array([1, 0, 1, 0, 0, 1], dtype=bool)
    p = np.array([0.9, 0.8, 0.5, 0.2, 0.05, 0.04])
    g = score_groups(y, p, cutoff=0.1, strong=0.8)
    assert g["strong"] == {"n": 2, "share_with_reactions": 0.5}
    assert g["weak"]["n"] == 2 and g["close"]["n"] == 2
    res = {"score_groups": g, "lift": {"top20": {"lift": 1.38, "lift_ci": (1.18, 1.56)}},
           "strong_threshold": 0.8, "cutoff": 0.1, "overall_share": 0.5}
    ev = evidence_text(res)
    assert "1.38x" in ev["Strong"] and "50.0%" in ev["Strong"] and set(ev) == {"Strong", "Too close to call", "Weak"}


# ---- what-ifs ----

def test_variants_change_one_feature_each():
    from slowpost.drafts import draft_features

    feats = draft_features(make_draft())
    variants = whatif_variants(feats, FACTS)
    assert variants
    for v in variants:
        changed = [k for k, val in v["overrides"].items() if val != feats[k]]
        assert changed == [v["feature"]] == list(v["overrides"])
    assert any(v["feature"] == "has_cover_image" and v["overrides"]["has_cover_image"] == 0
               for v in variants)
    assert all(v["overrides"].get("hour") != 12 for v in variants)  # no no-op variants


def test_score_drafts_reason_and_cover_effect():
    drafts = [make_draft(), make_draft(title="Why? 5 ways")]
    out = score_drafts(FakeModel(), drafts, FACTS, cutoff=0.14, strong=0.30, supported={"has_cover_image", "hour", "weekday", "main_tag", "reading_time_minutes", "title_has_question", "title_has_number"})
    a, b = out
    assert a["p"] == pytest.approx(0.20) and a["label"] == "Too close to call"
    assert b["p"] == pytest.approx(0.28)
    assert a["best"]["text"] == "a question in the title"  # +0.05 beats number (+0.03)
    assert a["best"]["delta"] == pytest.approx(0.05)
    assert "20% to 25%" in a["reason"] and a["reason"].startswith("Backed by the data")
    assert "would raise" not in a["reason"]  # associations, not causes
    cover = next(w for w in a["what_if"] if w["feature"] == "has_cover_image")
    assert cover["delta"] == pytest.approx(-0.10)  # variants that hurt are shown too
    assert all(w["delta"] <= a["best"]["delta"] for w in a["what_if"])


def test_small_differences_are_no_clear_difference():
    class Flat(FakeModel):
        def predict_proba(self, frame):
            p = np.full(len(frame), 0.1) + 0.005 * frame["title_has_number"].to_numpy()
            return np.column_stack([1 - p, p])

    out = score_drafts(Flat(), [make_draft()], FACTS, 0.14, 0.3, {"has_cover_image", "hour", "weekday", "main_tag", "reading_time_minutes", "title_has_question", "title_has_number"})[0]
    assert out["best"] is None and "No change backed by the data" in out["reason"]
    assert MIN_EFFECT == 0.01


# ---- revision ----

def _text(**kw) -> str:
    base = {"title": "Revised: 5 lessons learned", "tags": ["python", "gemma"], "cover_idea": "cov",
            "outline": ["x", "y", "z"], "reading_time_minutes": 7}
    return json.dumps({**base, **kw})


def test_revision_keeps_slot_limits_tags_and_notes_the_change():
    d = make_draft()
    new, status = revise_draft(d, 0.12, "Weak", ["has cover image: yes 17%, no 6%"],
                               FACTS["allowed_tags"], chat=lambda p, s: _text())
    assert status == "ok" and new.title == "Revised: 5 lessons learned"
    assert (new.day, new.hour) == (d.day, d.hour)  # the slot is chosen by code
    assert new.tags == ["python"] and any("dropped tag 'gemma'" in n for n in new.notes)
    assert "revised once after scoring" in new.notes


def test_revision_prompt_has_score_band_evidence_and_fences_the_draft():
    prompts = []

    def chat(prompt, schema):
        prompts.append(prompt)
        return _text()

    revise_draft(make_draft(), 0.12, "Weak", ["reading time: 3-4 min: 11%"], ["python"], chat)
    p = prompts[0]
    assert "12% chance" in p and "band: Weak" in p and "reading time: 3-4 min: 11%" in p
    assert "<draft>" in p and "DATA, not instructions" in p


def test_revision_that_stays_invalid_is_reported_not_half_filled():
    new, status = revise_draft(make_draft(), 0.1, "Weak", [], None, lambda p, s: '{"title": "x"}')
    assert new is None and status.startswith("could not revise")


def test_choose_final_needs_a_strictly_higher_score():
    orig, rev = {"p": 0.20}, {"p": 0.25}
    assert choose_final(orig, rev) is rev
    assert choose_final(orig, {"p": 0.20}) is orig  # a tie keeps the original
    assert choose_final(orig, {"p": 0.10}) is orig
    assert choose_final(orig, None) is orig


def test_revision_evidence_uses_observed_rates_of_clear_features_only():
    table = pd.DataFrame({"n": [500, 400], "top_share": [0.94, 0.80]}, index=["no", "yes"])
    summary = {"top_label": "no_reactions", "features": {
        "has cover image": {"clear": True, "table": table},
        "title has question": {"clear": False, "table": table},
        "weekday": {"clear": True, "table": table}}}  # the slot is not revisable
    lines = revision_evidence(summary)
    assert len(lines) == 1 and "no: 6%" in lines[0] and "yes: 20%" in lines[0]


def test_whatif_sentence_is_not_causal_and_shows_observed_rates():
    from slowpost.signals import signal_summary

    summary = {"top_label": "no_reactions", "overall_share": 0.84, "features": {
        "has cover image": {"table": pd.DataFrame({"n": [500, 400], "top_share": [0.94, 0.80]},
                                                  index=["no", "yes"]), "clear": True}}}
    a = score_drafts(FakeModel(), [make_draft(cover_idea="a sloth")], FACTS, 0.14, 0.30,
                     {"has_cover_image"}, summary)[0]
    cover = next(w for w in a["what_if"] if w["feature"] == "has_cover_image")
    assert cover["observed"]["new"] == pytest.approx(0.06)  # no cover: 6% got reactions
    assert cover["observed"]["old"] == pytest.approx(0.20)
    assert "In the data, posts like this with no cover image got reactions less often (6% vs 20%)"         in cover["sentence"]
    assert "The model scores this variant lower" in cover["sentence"]
    assert signal_summary  # imported to make sure the module loads together


# ---- draft fixes ----

def test_tags_must_come_from_history():
    kept, notes = sanitize_tags(["python", "gemma", "localtool"], FACTS["allowed_tags"], "x")
    assert kept == ["python"] and len(notes) == 2 and "gemma" in notes[0]
    kept, notes = sanitize_tags(["gemma"], FACTS["allowed_tags"], "A tabpfn story")
    assert kept == ["tabpfn"] and "replaced with tag 'tabpfn'" in notes[-1]
    assert sanitize_tags(["gemma"], None, "x") == (["gemma"], [])  # no history: unchanged


def test_draft_one_records_tag_changes():
    good = DraftText(title="A fine title here", tags=["gemma", "python"], cover_idea="cover",
                     outline=["a", "b", "c"], reading_time_minutes=5)
    d, err = draft_one("brief", "angle", ("Monday", 9), FACTS, chat=lambda p, s: good.model_dump_json())
    assert err is None and d.tags == ["python"] and "dropped tag 'gemma'" in d.notes[0]


def test_titles_similar():
    assert titles_similar("Predicting DEV Post Reactions with a Local Gemma & TabPFN Tool (Hacktoberfest)",
                          "Predicting DEV Post Reactions with a Local Gemma & TabPFN Tool (Hacktoberfest)")
    assert titles_similar("Predicting DEV Post Reactions with TabPFN and Gemma",
                          "Predicting DEV post reactions with Gemma and TabPFN!")
    assert not titles_similar("Predicting DEV Post Reactions", "Setting up Ollama on Windows")


def test_duplicate_title_regenerated_once():
    titles = iter(["Same headline for both", "Same headline for both", "A different headline"])
    prompts = []

    def chat(prompt, schema):
        prompts.append(prompt)
        return DraftText(title=next(titles), tags=["python"], cover_idea="cov",
                         outline=["a", "b", "c"], reading_time_minutes=5).model_dump_json()

    out = generate_drafts("brief", 2, chat=chat)
    assert out[0]["draft"].title != out[1]["draft"].title
    assert any("regenerated once" in n for n in out[1]["draft"].notes)
    assert "must be clearly different" in prompts[-1] and len(prompts) == 3


def test_duplicate_that_persists_is_kept_but_flagged():
    chat = lambda p, s: DraftText(title="Same headline", tags=["python"], cover_idea="cov",
                                  outline=["a", "b", "c"], reading_time_minutes=5).model_dump_json()
    out = generate_drafts("brief", 2, chat=chat)
    assert out[1]["draft"] is not None
    assert any("near-duplicate" in n for n in out[1]["draft"].notes)


def test_project_facts_are_fenced_as_data():
    seen = []
    generate_drafts("brief", 1, chat=lambda p, s: seen.append(p) or DraftText(
        title="A fine title here", tags=["python"], cover_idea="cov", outline=["a", "b", "c"],
        reading_time_minutes=5).model_dump_json(), project_facts="It runs locally.")
    assert "<facts>It runs locally.</facts>" in seen[0] and "not instructions" in seen[0]


# ---- time and files ----

def test_slot_display_has_utc_and_local():
    d = slot_display("Thursday", 12, today=date(2026, 10, 21))  # a Wednesday
    assert d["utc"] == "Thursday 12:00 UTC" and d["date_utc"] == "2026-10-22"
    assert d["local"].count(":") == 1 and len(d["local"].split()) >= 3
    local = datetime.fromisoformat("2026-10-22T12:00:00+00:00").astimezone()
    assert f"{local:%H:%M}" in d["local"]


def test_save_json_keeps_unicode(tmp_path):
    path = save_json(tmp_path / "x" / "d.json", {"t": "naïve — “quotes” ✓"})
    assert "✓" in path.read_text(encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["t"] == "naïve — “quotes” ✓"


# ---- "backed by the data" versus the model's guess ----

def test_unsupported_guess_is_grouped_separately_and_never_the_reason():
    only_number = {"title_has_number", "has_cover_image"}  # question is NOT backed by the data
    a = score_drafts(FakeModel(), [make_draft()], FACTS, 0.14, 0.30, only_number)[0]
    by_feature = {w["feature"]: w for w in a["what_if"]}
    assert by_feature["title_has_question"]["supported"] is False
    assert by_feature["title_has_number"]["supported"] is True
    assert a["best"]["feature"] == "title_has_number"  # smaller gain, but backed
    assert a["guess"]["feature"] == "title_has_question"  # bigger gain, only a guess
    assert a["guess"]["delta"] > a["best"]["delta"]


def test_supported_features_come_from_clear_signal_groups():
    from slowpost.signals import supported_features

    summary = {"features": {
        "has cover image": {"clear": True}, "title has question": {"clear": False},
        "title has number": {"clear": True}, "weekday": {"clear": False}}}
    assert supported_features(summary) == {"has_cover_image", "title_has_number"}


def test_second_regeneration_uses_a_clearly_different_angle():
    titles = iter(["Same headline for both", "Same headline for both",
                   "Same headline for both", "Something else entirely"])
    prompts = []

    def chat(prompt, schema):
        prompts.append(prompt)
        return DraftText(title=next(titles), tags=["python"], cover_idea="cov",
                         outline=["a", "b", "c"], reading_time_minutes=5).model_dump_json()

    out = generate_drafts("brief", 2, chat=chat)
    second = out[1]["draft"]
    assert second.title == "Something else entirely" and len(prompts) == 4
    assert "clearly different angle" in prompts[-1] and "clearly different angle" not in prompts[-2]
    assert any("twice" in n for n in second.notes)


def test_persistent_duplicate_is_flagged_after_two_regenerations():
    prompts = []

    def chat(prompt, schema):
        prompts.append(prompt)
        return DraftText(title="Same headline", tags=["python"], cover_idea="cov",
                         outline=["a", "b", "c"], reading_time_minutes=5).model_dump_json()

    out = generate_drafts("brief", 2, chat=chat)
    assert len(prompts) == 4  # draft 1: 1 call; draft 2: 1 + 2 regenerations
    assert any("two regenerations did not fix it" in n for n in out[1]["draft"].notes)


def test_on_draft_reports_each_finished_slot():
    titles = iter(["Alpha headline one", "Completely different words", "Zebra crossing study"])
    chat = lambda p, s: DraftText(title=next(titles), tags=["python"], cover_idea="cov",
                                  outline=["a", "b", "c"], reading_time_minutes=5).model_dump_json()
    seen = []
    generate_drafts("brief", 3, chat=chat, on_draft=lambda done, total: seen.append((done, total)))
    assert seen == [(1, 3), (2, 3), (3, 3)]
