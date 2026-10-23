"""Slow Post: local Streamlit app. UI only; the logic lives in slowpost/ and scripts/.

Run: streamlit run app.py
Heavy work (fetch, backtest, overnight run) runs in separate processes so TabPFN and Gemma never
share memory with the app, and a browser refresh does not stop a run.
Environment (optional): SLOWPOST_DEV_CSV = path of the DEV posts CSV.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import psutil
import streamlit as st

from slowpost.plan import apply_edits, plan_csv

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEV_CSV = Path(os.environ.get("SLOWPOST_DEV_CSV", DATA / "dev_posts.csv"))
RUNS = DATA / "runs"
BACKTESTS = DATA / "backtests"
CURRENT = RUNS / "_current.json"
CURRENT_LOG = RUNS / "_current.log"
MODEL = "gemma3:4b"
PAGES = ["1 Data", "2 Backtest", "3 Brief", "4 Plan", "Instagram sample (no signal)"]
BADGE = {"Strong": "green", "Okay": "orange", "Weak": "red"}

st.set_page_config(page_title="Slow Post", page_icon="🦥", layout="centered")


# ---------- helpers ----------

def run_script(args: list[str]) -> tuple[bool, str]:
    """Run a script in a separate process; return (ok, last lines of output)."""
    proc = subprocess.run([sys.executable, "-u", *args], cwd=ROOT, capture_output=True, check=False,
                          text=True, encoding="utf-8", errors="replace")
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-8:])
    return proc.returncode == 0, tail


def ollama_problem() -> str | None:
    """None if Ollama and the model are ready, else a plain message."""
    try:
        import ollama

        names = [m.model for m in ollama.list().models]
    except Exception:  # noqa: BLE001 - any failure means "not reachable"
        return ("Ollama is not running. Start it with `ollama serve` (or open the Ollama app), "
                "then try again.")
    if not any(n == MODEL or n.startswith(MODEL + ":") for n in names):
        return f"The model {MODEL} is not installed. Run `ollama pull {MODEL}`."
    return None


@st.cache_data(show_spinner=False)
def dev_summary(path: str, mtime: float) -> dict:
    df = pd.read_csv(path, usecols=["published_at", "public_reactions_count"])
    return {"n": len(df), "first": str(df["published_at"].min())[:10],
            "last": str(df["published_at"].max())[:10],
            "share": float((df["public_reactions_count"] >= 1).mean())}


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def run_state() -> dict:
    """{active, started, error?}: is an overnight run going, from the pid file."""
    info = load_json(CURRENT)
    if not info:
        return {"active": False}
    alive = psutil.pid_exists(info["pid"])
    try:
        alive = alive and "python" in psutil.Process(info["pid"]).name().lower()
    except psutil.Error:
        alive = False
    return {"active": alive, **info}


def latest_run() -> Path | None:
    done = sorted(p for p in RUNS.glob("2*") if (p / "drafts.json").exists()) if RUNS.exists() else []
    return done[-1] if done else None


def pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1%}"


# ---------- backtest view (used for DEV and Instagram) ----------

def render_backtest(data: dict) -> None:
    r, sig = data["result"], data["signals"]
    st.caption(f"{data['title']}: {data['n_posts']:,} posts, {data['first_post'][:10]} to "
               f"{data['last_post'][:10]}. Test window: {r['test_window']}.")
    if "no better" in r["verdict"].lower() or r["verdict"].startswith("Worse"):
        st.warning(r["verdict"])
    else:
        st.success(r["verdict"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Balanced accuracy", f"{r['balanced_accuracy']:.3f}",
              f"{r['ba_gap']:+.3f} vs baseline {r['baseline_balanced_accuracy']:.3f}",
              delta_color="normal" if r["ba_gap_ci"][0] > 0 else "off",
              delta_arrow="up" if r["ba_gap_ci"][0] > 0 else "off")
    c2.metric("ROC AUC", f"{r['auc']:.3f}", "0.5 = chance", delta_color="off",
              delta_arrow="off")
    c3.metric("Test posts", f"{r['n_test']:,}", f"{r['n_train']:,} trained on", delta_color="off",
              delta_arrow="off")
    st.caption(f"95% intervals: balanced-accuracy gap {r['ba_gap_ci'][0]:+.3f} to "
               f"{r['ba_gap_ci'][1]:+.3f}; ROC AUC {r['auc_ci'][0]:.3f} to {r['auc_ci'][1]:.3f}. "
               f"Baseline: always say '{r['baseline_label']}'. Raw accuracy (reference only): "
               f"{r['accuracy']:.3f} vs {r['baseline_accuracy']:.3f}.")
    if r.get("cutoff") is not None:
        st.caption(f"The model says '{r['positive']}' when its probability is above "
                   f"{r['cutoff']:.3f}, the share of such posts in the training sample.")
    if r.get("lift"):
        rows = []
        for name, v in r["lift"].items():
            rows.append({"Model's highest-ranked": f"top {name[3:]}% ({v['n_top']} posts)",
                         "Got reactions": f"{v['share']:.1%} ({v['share_ci'][0]:.1%}-{v['share_ci'][1]:.1%})",
                         "Overall": f"{r['overall_share']:.1%}",
                         "Lift": f"{v['lift']:.2f}x ({v['lift_ci'][0]:.2f}-{v['lift_ci'][1]:.2f})"})
        st.markdown("**Lift: do the posts the model ranks highest do better?**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("**Signal summary: which features change the odds?**")
    st.write(sig["text"])
    st.caption(f"Share of '{sig['top_label']}' posts by group (overall {sig['overall_share']:.1%}). "
               "p-values are corrected for testing several features at once.")
    for name, v in sig["features"].items():
        word = "clear difference" if v["clear"] else "no clear difference"
        with st.expander(f"{name}: {word} (corrected p = {v['p_adj']:.3g})"):
            g = pd.DataFrame(v["groups"]).set_index("group")
            st.bar_chart(g["top_share"])
            st.dataframe(g.rename(columns={"n": "posts", "top_share": "share", "gap": "gap vs overall"}),
                         use_container_width=True)


# ---------- pages ----------

def page_data() -> None:
    st.header("1 · Your post history")
    st.write("Slow Post learns from real DEV posts (fetched from DEV's public API). "
             "Only post id, time, title, tags, reading time, cover flag and reaction counts are "
             "kept, never author names. Everything stays on this computer.")
    if not DEV_CSV.exists():
        st.error(f"No DEV posts found at `{DEV_CSV}`. Click the button to fetch them.")
    else:
        s = dev_summary(str(DEV_CSV), DEV_CSV.stat().st_mtime)
        c1, c2 = st.columns(2)
        c1.metric("Posts", f"{s['n']:,}")
        c2.metric("Got any reactions", f"{s['share']:.1%}")
        st.write(f"Published from **{s['first']}** to **{s['last']}**.")
        st.caption("Posts are at least 14 days old, so reactions have mostly settled.")
    label = "Fetch DEV posts again" if DEV_CSV.exists() else "Fetch DEV posts"
    if st.button(label, help="Tags webdev, python, javascript, beginners; about 3 to 5 minutes"):
        with st.spinner("Fetching DEV posts (about 3 to 5 minutes)..."):
            ok, tail = run_script(["scripts/fetch_dev_posts.py", "--out", str(DEV_CSV)])
        dev_summary.clear()
        (st.success if ok else st.error)("Fetched." if ok else "The fetch failed.")
        st.code(tail)


def page_backtest() -> None:
    st.header("2 · Does the history predict anything?")
    st.write("Honest check: train on older posts, test on the newest week, and compare with "
             "always guessing the most common outcome.")
    if not DEV_CSV.exists():
        st.error("Fetch the DEV posts first (step 1).")
        return
    if st.button("Run backtest (about 3 minutes)"):
        with st.spinner("Running the backtest with TabPFN on this computer..."):
            ok, tail = run_script(["scripts/run_backtest.py", "--source", "dev", "--save"])
        if not ok:
            st.error("The backtest failed.")
            st.code(tail)
    data = load_json(BACKTESTS / "dev.json")
    if data:
        render_backtest(data)
    else:
        st.info("No backtest yet. Click the button above.")


def start_run() -> None:
    """Button callback: validate, then start the overnight run in its own process."""
    brief = st.session_state.get("brief", "").strip()
    st.session_state.start_msg = None
    if not brief:
        st.session_state.start_msg = "Please write a brief first (what the posts should be about)."
        return
    if not DEV_CSV.exists():
        st.session_state.start_msg = "Fetch the DEV posts first (step 1)."
        return
    if run_state()["active"]:
        st.session_state.start_msg = "A run is already going. See the Plan page."
        return
    problem = ollama_problem()
    if problem:
        st.session_state.start_msg = problem
        return
    RUNS.mkdir(parents=True, exist_ok=True)
    log = CURRENT_LOG.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-u", "scripts/run_overnight.py", "--brief", brief,
         "--facts", st.session_state.get("facts", ""), "--n", str(st.session_state.get("n", 5)),
         "--csv", str(DEV_CSV)],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    CURRENT.write_text(json.dumps({"pid": proc.pid, "started": time.time(),
                                   "started_text": datetime.now().strftime("%H:%M:%S")}),
                       encoding="utf-8")
    st.session_state.nav = "4 Plan"


def page_brief() -> None:
    st.header("3 · What should next week's posts be about?")
    st.text_area("Brief", key="brief", height=90,
                 placeholder="A DEV post about building a local tool that ...")
    st.text_area("Facts (optional)", key="facts", height=110,
                 help="Reference facts about your project. Gemma treats them as data, so it "
                      "doesn't invent how things work.")
    st.slider("Number of drafts", 1, 7, 5, key="n")
    st.caption("A run takes about 10 minutes: Gemma drafts, then TabPFN scores, one after the "
               "other. You can leave this tab open or come back later.")
    problem = ollama_problem()
    if problem:
        st.error(problem)
    st.button("Start the run", type="primary", on_click=start_run)
    if st.session_state.get("start_msg"):
        st.error(st.session_state.start_msg)


def show_card(run: Path, i: int, item: dict) -> dict:
    """One draft card with editable fields. Returns the (possibly edited) item."""
    key = f"{run.name}-{i}"
    d = item["draft"]
    with st.container(border=True):
        top, right = st.columns([3, 2])
        top.markdown(f"**{item['when']['utc']}**  \n"
                     f"{item['when']['local']} (your local time)")
        right.badge(item["label"], color=BADGE[item["label"]])
        right.caption(f"Chance of any reactions: {item['p_yes']:.0%}")
        title = st.text_input("Title", d["title"], key=f"t-{key}")
        tags = st.text_input("Tags (comma separated, up to 4)", ",".join(d["tags"]), key=f"g-{key}")
        cover = st.text_input("Cover image idea", d["cover_idea"], key=f"c-{key}")
        outline = st.text_area("Outline (one point per line)", "\n".join(d["outline"]),
                               key=f"o-{key}", height=130)
        st.caption(f"Target reading time: {d['reading_time_minutes']} min")
        edited, err = apply_edits(item, {"title": title, "tags": tags, "cover_idea": cover,
                                         "outline": outline})
        if err:
            st.error(f"These edits can't be used yet ({err}). The original draft is kept.")
        elif edited.get("edited"):
            st.info("Edited. The score above belongs to the original draft; edits are not rescored.")
        if d["notes"]:
            st.caption("Notes: " + "; ".join(d["notes"]))
        with st.expander("Features computed by code"):
            st.json(edited["features"])
        with st.expander("What-if: what would change the score?"):
            st.write(item["reason"])
            sup = [w for w in item["what_if"] if w.get("supported")]
            rest = [w for w in item["what_if"] if not w.get("supported")]
            st.markdown("**Backed by the data** (the signal summary shows a clear difference)")
            if sup:
                for w in sup:
                    st.write(f"{w['delta']:+.1%} points: {w['text']} ({w['p']:.0%})")
            else:
                st.caption("No what-if feature is backed by the data.")
            st.markdown("**The model's guess**")
            st.caption("The data shows no clear difference for these, so treat them as guesses.")
            for w in rest:
                st.markdown(f":gray[{w['delta']:+.1%} points: {w['text']} ({w['p']:.0%})]")
        rw = item.get("rewrite")
        if rw:
            with st.expander("Rewrite (once, for weak drafts)", expanded=True):
                if rw.get("draft"):
                    a, b = st.columns(2)
                    a.markdown(f"**Before** ({rw['p_before']:.0%})  \n{d['title']}")
                    b.markdown(f"**After** ({rw['p_after']:.0%}, {rw['label_after']})  \n"
                               f"{rw['draft']['title']}")
                    st.caption("Aimed at: " + rw["change"] + ". "
                               + ("The score went up." if rw["improved"] else
                                  "The score did not clearly improve."))
                else:
                    st.write(rw["status"])
    return edited


@st.fragment(run_every=4)
def live_status() -> None:
    state = run_state()
    if not state["active"]:
        st.rerun()
    st.info(f"A run is going (started {state['started_text']}, "
            f"{int(time.time() - state['started'])} s ago). This page updates by itself.")
    text = CURRENT_LOG.read_text(encoding="utf-8", errors="replace") if CURRENT_LOG.exists() else ""
    st.code("\n".join(text.splitlines()[-8:]) or "Starting...")


def page_plan() -> None:
    st.header("4 · The week's plan")
    state = run_state()
    if state["active"]:
        live_status()  # refreshes itself; the full page reruns when the run ends
        return
    elif CURRENT_LOG.exists() and "Saved to" not in CURRENT_LOG.read_text(encoding="utf-8", errors="replace"):
        newest = latest_run()
        newer_than_plan = not newest or CURRENT_LOG.stat().st_mtime > (newest / "drafts.json").stat().st_mtime
        if newer_than_plan:
            st.error("The last run stopped before finishing:")
            tail = CURRENT_LOG.read_text(encoding="utf-8", errors="replace").strip().splitlines()
            st.code("\n".join(tail[-8:]))
    run = latest_run()
    if not run:
        st.info("No plan yet. Write a brief in step 3 and start a run.")
        return
    data = load_json(run / "drafts.json")
    st.caption(f"Run {run.name}. Brief: {data['brief']}")
    st.markdown("**What the labels mean** (measured on the backtest test window)")
    for name, text in data["label_meaning"].items():
        st.markdown(f":{BADGE[name]}-badge[{name}] {text}")
    st.caption("Lift is modest: use the labels to rank ideas, not to promise results.")
    items = sorted(data["drafts"], key=lambda it: (it["features"]["weekday"], it["features"]["hour"]))
    edited = [show_card(run, i, it) for i, it in enumerate(items)]
    if data["failed_slots"]:
        st.warning("Some slots could not be drafted: " + "; ".join(data["failed_slots"]))
    st.download_button("Download plan.csv", plan_csv(edited), file_name="plan.csv",
                       mime="text/csv", type="primary")
    t = data["timings_s"]
    st.caption("Run time: " + ", ".join(f"{k.split(' ', 1)[-1]} {v:.0f}s" for k, v in t.items()))


def page_instagram() -> None:
    st.header("Instagram sample: a test case with no signal")
    st.write("One account from a public Kaggle dataset (CC BY 4.0, Kundan Sagar Bedmutha). "
             "It is here to show that Slow Post reports honestly when a history has nothing "
             "to learn. The verdict below is computed, not scripted. Backtest only; no drafting.")
    sample = ROOT / "sample" / "sample_posts.csv"
    if not sample.exists():
        st.error("sample/sample_posts.csv is missing.")
        return
    if st.button("Run backtest on the sample (about 1 minute)"):
        with st.spinner("Running the backtest with TabPFN on this computer..."):
            ok, tail = run_script(["scripts/run_backtest.py", "--source", "instagram", "--save"])
        if not ok:
            st.error("The backtest failed.")
            st.code(tail)
    data = load_json(BACKTESTS / "instagram.json")
    if data:
        render_backtest(data)
    else:
        st.info("No backtest yet. Click the button above.")


# ---------- main ----------

st.sidebar.title("🦥 Slow Post")
st.sidebar.caption("Drafts next week's posts overnight and flags the weak ones. Local only.")
page = st.sidebar.radio("Steps", PAGES, key="nav")
{"1 Data": page_data, "2 Backtest": page_backtest, "3 Brief": page_brief, "4 Plan": page_plan,
 "Instagram sample (no signal)": page_instagram}[page]()
