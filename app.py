"""Slow Post: local Streamlit app. UI only; the logic lives in slowpost/ and scripts/.

Run: streamlit run app.py
Heavy work (fetch, backtest, overnight run) runs in separate processes so TabPFN and Gemma never
share memory with the app, and a browser refresh does not stop a run.
Environment (optional): SLOWPOST_DEV_CSV = path of the DEV posts CSV; SLOWPOST_RUNS = runs folder.
"""
from __future__ import annotations

import html
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import psutil
import streamlit as st
import streamlit.components.v1 as components

from slowpost import schedule, ui
from slowpost.compare import NAMES
from slowpost.plan import apply_edits, normalize_labels, plan_csv, run_verdict

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEV_CSV = Path(os.environ.get("SLOWPOST_DEV_CSV", DATA / "dev_posts.csv"))
RUNS = Path(os.environ.get("SLOWPOST_RUNS", DATA / "runs"))
BACKTESTS = DATA / "backtests"
CURRENT = RUNS / "_current.json"
CURRENT_LOG = RUNS / "_current.log"
MODEL = "gemma3:4b"
NAV = [("0 Home", "Home", ":material/home:"), ("1 Data", "Data", ":material/database:"), ("2 Backtest", "Backtest", ":material/query_stats:"),
       ("3 Brief", "Brief", ":material/edit_note:"), ("4 Plan", "Plan", ":material/calendar_month:")]
SECONDARY = ("Instagram sample (no signal)", "Instagram sample (no signal)", ":material/science:")

st.set_page_config(page_title="Slow Post", page_icon="🦥", layout="wide")


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

def render_headline(r: dict) -> None:
    """One big result before the details. Every number is read from the saved result."""
    top = (r.get("lift") or {}).get("top10")
    if top:
        lo, hi = top["lift_ci"]
        caution = ("" if lo > 1 else " The interval includes 1.0x, so this may be chance.")
        st.markdown(
            f'<div class="sp-hero"><div class="sp-hero-num">{top["lift"]:.2f}×</div>'
            f'<div class="sp-hero-head">Top picks got reactions {top["lift"]:.2f}× more often</div>'
            f'<div>Of the {top["n_top"]} posts the model ranked highest in the newest week, '
            f'{top["share"]:.1%} got reactions, against {r["overall_share"]:.1%} overall.{caution}</div>'
            f'<div class="sp-small">95% confidence interval: {lo:.2f}× to {hi:.2f}×. '
            f'Lift is modest: use it to rank ideas, not to promise results.</div></div>',
            unsafe_allow_html=True)
    else:
        head = r["verdict"].split(":")[0].strip().rstrip(".")
        st.markdown(
            f'<div class="sp-hero"><div class="sp-hero-num sp-none">{html.escape(head)}</div>'
            f'<div class="sp-hero-head">The history has no usable signal</div>'
            f'<div>Balanced accuracy {r["balanced_accuracy"]:.3f} against '
            f'{r["baseline_balanced_accuracy"]:.3f} for always guessing the most common outcome.</div>'
            f'<div class="sp-small">95% interval for the gap: {r["ba_gap_ci"][0]:+.3f} to '
            f'{r["ba_gap_ci"][1]:+.3f}.</div></div>', unsafe_allow_html=True)


def _ci(v: float, ci, fmt: str = ".3f", suffix: str = "") -> str:
    if ci is None:
        return f"{v:{fmt}}{suffix}"
    return f"{v:{fmt}}{suffix} ({ci[0]:{fmt}}{suffix} to {ci[1]:{fmt}}{suffix})"


def render_compare(cmp: dict) -> None:
    """TabPFN against simpler models on the identical split, and the random-split comparison."""
    st.markdown(ui.section("Is TabPFN better than simpler models?"), unsafe_allow_html=True)
    rows = []
    for key, m in cmp["models"].items():
        row = {"Model": NAMES[key],
               "Balanced accuracy": _ci(m["balanced_accuracy"], m["ba_ci"]),
               "ROC AUC": _ci(m["auc"], m["auc_ci"])}
        for name in ("top10", "top20"):
            if name in m:
                v = m[name]
                row[f"Top {name[3:]}% lift"] = _ci(v["lift"], v["lift_ci"], ".2f", "x")
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    (st.warning if "does not clearly win" in cmp["winner"] or "ranks posts worse" in cmp["winner"]
     else st.info)(cmp["winner"])
    diffs = "; ".join(f"vs {NAMES[k].lower()}: AUC {d['auc_diff']:+.3f} (paired 95% interval "
                      f"{d['auc_diff_ci'][0]:+.3f} to {d['auc_diff_ci'][1]:+.3f})"
                      for k, d in cmp["tabpfn_minus"].items())
    st.caption(f"TabPFN minus each model, {diffs}. Same 1,000 training posts and same test posts for "
               "every model; the simple models use fixed ordinary settings, with no tuning.")
    rs = cmp["random_split"]
    top = rs.get("top10")
    st.markdown(f"**TabPFN on a random 80/20 split (for comparison only):** balanced accuracy "
                f"{_ci(rs['balanced_accuracy'], rs['ba_ci'])}, ROC AUC {_ci(rs['auc'], rs['auc_ci'])}"
                + (f", top-10% lift {_ci(top['lift'], top['lift_ci'], '.2f', 'x')}" if top else "") + ".")
    st.caption("The time-ordered split is the honest one: a plan is written before the week starts, "
               "so the model must predict newer posts from older ones. A random split lets it learn "
               "from posts published in the same weeks it is tested on.")


def render_calibration(bins: list[dict]) -> None:
    """Predicted vs actual share with reactions, equal-count bins of the test posts."""
    st.markdown(ui.section("Calibration: when the model says 20%, what happens?"),
                unsafe_allow_html=True)
    df = pd.DataFrame([{"Predicted": b["mean_predicted"], "Actual": b["actual"],
                        "Low": b["actual_ci"][0], "High": b["actual_ci"][1], "Posts": b["n"]}
                       for b in bins])
    top = float(max(df["High"].max(), df["Predicted"].max())) * 1.05
    scale = alt.Scale(domain=[0, top])
    base = alt.Chart(df)
    line = alt.Chart(pd.DataFrame({"x": [0, top], "y": [0, top]})).mark_line(
        strokeDash=[4, 4], color="#6b6252").encode(x=alt.X("x", scale=scale), y=alt.Y("y", scale=scale))
    bars = base.mark_rule(color="#4a7a3c").encode(
        x=alt.X("Predicted", scale=scale, axis=alt.Axis(format="%", title="Predicted chance of reactions")),
        y=alt.Y("Low", scale=scale, axis=alt.Axis(format="%", title="Share that got reactions")),
        y2="High")
    pts = base.mark_circle(size=90, color="#4a7a3c").encode(
        x="Predicted", y="Actual", tooltip=["Predicted", "Actual", "Low", "High", "Posts"])
    st.altair_chart((line + bars + pts).properties(height=300), width="stretch")
    total = sum(b["n"] for b in bins)
    predicted = sum(b["mean_predicted"] * b["n"] for b in bins) / total
    actual = sum(b["actual"] * b["n"] for b in bins) / total
    st.caption(f"{len(bins)} equal-sized groups of test posts, from lowest to highest predicted chance; "
               "bars are 95% intervals. Dots on the dashed line would mean the percentages can be "
               f"taken at face value. On average the model predicted {predicted:.0%} and {actual:.0%} "
               "really got reactions, so read the percentages as a ranking, not as exact odds.")


def render_controls(c: dict) -> None:
    """Does the cover-image gap hold within reading-time bands, and the reverse?"""
    st.markdown(ui.section("Controlled check: are cover and reading time separate effects?"),
                unsafe_allow_html=True)
    st.caption(c["note"])
    for key, title, exposed in (
            ("cover_gap_within_reading_time", "Cover-image gap within each reading-time band", "cover"),
            ("reading_time_gap_by_cover",
             f"Reading-time gap ({c['median_reading_time']:.0f}+ minutes vs shorter) with and without a cover",
             f"{c['median_reading_time']:.0f}+ min")):
        block = c[key]
        st.markdown(f"**{title}**")
        rows = [{"Group": s["stratum"], f"Got reactions ({exposed})": f"{s['rate1']:.1%} (n={s['n1']})",
                 "Got reactions (other)": f"{s['rate0']:.1%} (n={s['n0']})",
                 "Gap": f"{s['diff']:+.1%} ({s['diff_ci'][0]:+.1%} to {s['diff_ci'][1]:+.1%})"
                        + ("" if s["enough"] else " (small group)")}
                for s in block["strata"]]
        cr = block["crude"]
        rows.append({"Group": "All posts (unadjusted)",
                     f"Got reactions ({exposed})": f"{cr['rate1']:.1%} (n={cr['n1']})",
                     "Got reactions (other)": f"{cr['rate0']:.1%} (n={cr['n0']})",
                     "Gap": f"{cr['diff']:+.1%} ({cr['diff_ci'][0]:+.1%} to {cr['diff_ci'][1]:+.1%})"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.write(f"{block['statement']} Odds ratio: {block['crude_or']:.2f} unadjusted, "
                 f"{block['mh_or']:.2f} adjusted ({block['mh_or_ci'][0]:.2f} to {block['mh_or_ci'][1]:.2f}).")


def render_backtest(data: dict) -> None:
    r, sig = data["result"], data["signals"]
    render_headline(r)
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
        st.markdown(ui.section("Lift: do the posts the model ranks highest do better?"), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    if r.get("compare"):
        render_compare(r["compare"])
    if r.get("calibration"):
        render_calibration(r["calibration"])
    st.markdown(ui.section("Signal summary: which features change the odds?"), unsafe_allow_html=True)
    st.write(sig["text"])
    st.caption(f"Share of '{sig['top_label']}' posts by group (overall {sig['overall_share']:.1%}). "
               "p-values are corrected for testing several features at once.")
    for name, v in sig["features"].items():
        word = "clear difference" if v["clear"] else "no clear difference"
        with st.expander(f"{name}: {word} (corrected p = {v['p_adj']:.3g})"):
            g = pd.DataFrame(v["groups"]).set_index("group")
            st.bar_chart(g["top_share"])
            st.dataframe(g.rename(columns={"n": "posts", "top_share": "share", "gap": "gap vs overall"}),
                         width="stretch")
    if data.get("controls"):
        render_controls(data["controls"])


# ---------- pages ----------

def page_title(title: str, lede: str) -> None:
    st.markdown(ui.page_title(title, lede), unsafe_allow_html=True)


STEPS = [("Data", "Fetch real DEV posts"), ("Backtest", "Check if history predicts anything"),
         ("Brief", "Say what to write about"), ("Plan", "Drafts, scored and labelled")]


def page_home() -> None:
    st.markdown('<div class="sp-page-title">🦥 Slow Post</div>'
                '<div class="sp-lede">Drafts your next posts while you rest, and tells you which '
                'ones are worth writing.</div>', unsafe_allow_html=True)
    st.markdown(ui.section("How it works"), unsafe_allow_html=True)
    cells = "".join(f'<div class="sp-flow-step"><span class="sp-num">{i}</span>'
                    f'<div class="sp-flow-name">{name}</div><div class="sp-small">{text}</div></div>'
                    for i, (name, text) in enumerate(STEPS, 1))
    st.markdown(f'<div class="sp-flow">{cells}</div>', unsafe_allow_html=True)
    st.markdown(ui.section("The result so far"), unsafe_allow_html=True)
    data = load_json(BACKTESTS / "dev.json")
    if data:
        render_headline(data["result"])
    else:
        st.info("No backtest yet. Start with the Data step, then run the backtest.")
    st.caption("Runs on this computer. Nothing is uploaded.")
    st.button("Start", type="primary", on_click=go, args=("1 Data",))


def page_data() -> None:
    page_title("Your post history",
               "The real DEV posts Slow Post learns from. Check the numbers, then run the backtest.")
    have = DEV_CSV.exists()
    if not have:
        st.error(f"No DEV posts found at `{DEV_CSV}`. Click the button to fetch them.")
    else:
        s = dev_summary(str(DEV_CSV), DEV_CSV.stat().st_mtime)
        c1, c2 = st.columns(2)
        c1.metric("Posts", f"{s['n']:,}")
        c2.metric("Got any reactions", f"{s['share']:.1%}")
        st.write(f"Published from **{s['first']}** to **{s['last']}**.")
        st.caption("Posts are at least 14 days old, so reactions have mostly settled.")
    st.markdown(ui.section("Next"), unsafe_allow_html=True)
    first, second, _ = st.columns([2, 2, 3])
    if have:
        first.button("Next: run the backtest →", type="primary", on_click=go, args=("2 Backtest",))
    fetch = (second if have else first).button(
        "Fetch DEV posts again" if have else "Fetch DEV posts",
        type="secondary" if have else "primary",
        help="Tags webdev, python, javascript, beginners; about 3 to 5 minutes")
    st.caption("Fetched from DEV's public API. Only post id, time, title, tags, reading time, "
               "cover flag and reaction counts are kept, never author names. Everything stays "
               "on this computer.")
    if fetch:
        with st.spinner("Fetching DEV posts (about 3 to 5 minutes)..."):
            ok, tail = run_script(["scripts/fetch_dev_posts.py", "--out", str(DEV_CSV)])
        dev_summary.clear()
        (st.success if ok else st.error)("Fetched." if ok else "The fetch failed.")
        st.code(tail)


def page_backtest() -> None:
    page_title("Does the history predict anything?",
               "Train on older posts, test on the newest week, and compare with always guessing "
               "the most common outcome. Then write a brief.")
    if not DEV_CSV.exists():
        st.error("Fetch the DEV posts first (step 1).")
        return
    data = load_json(BACKTESTS / "dev.json")
    first, second, _ = st.columns([2, 2, 3])
    if data:
        first.button("Next: write a brief →", type="primary", on_click=go, args=("3 Brief",))
    run = (second if data else first).button(
        "Run the backtest again (3 min)" if data else "Run backtest (about 3 minutes)",
        type="secondary" if data else "primary")
    if run:
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
    page_title("What should next week's posts be about?",
               "Say what next week's posts should be about, then start the run. Gemma drafts, then TabPFN "
               "scores, one after the other.")
    with st.container(border=True):
        st.text_area("Brief", key="brief", height=90,
                     placeholder="A DEV post about building a local tool that ...")
        st.text_area("Facts (optional)", key="facts", height=110,
                     help="Reference facts about your project. Gemma treats them as data, so it "
                          "doesn't invent how things work.")
        st.slider("Number of drafts", 1, 7, 5, key="n")
    st.caption("A run takes about 10 minutes. You can leave this tab open or come back later.")
    problem = ollama_problem()
    if problem:
        st.error(problem)
    st.button("Start the run", type="primary", on_click=start_run)
    if st.session_state.get("start_msg"):
        st.error(st.session_state.start_msg)
    schedule_section()


def next_two_am() -> datetime:
    now = datetime.now()
    when = now.replace(hour=2, minute=0, second=0, microsecond=0)
    return when if when > now else when + timedelta(days=1)


def schedule_run(when: datetime) -> None:
    """Button callback: save the job file and create the one-time Windows task."""
    brief = st.session_state.get("brief", "").strip()
    st.session_state.schedule_msg = None
    if not brief:
        st.session_state.schedule_msg = ("error", "Please write a brief first.")
    elif not DEV_CSV.exists():
        st.session_state.schedule_msg = ("error", "Fetch the DEV posts first (step 1).")
    elif when <= datetime.now():
        st.session_state.schedule_msg = ("error", "Pick a time in the future.")
    else:
        job = schedule.write_job(when, brief, st.session_state.get("facts", ""),
                                 st.session_state.get("n", 5), DEV_CSV)
        ok, message = schedule.create(when, job)
        st.session_state.schedule_msg = (
            "success" if ok else "error",
            f"Scheduled {schedule.task_name(when)} for {when:%A %d %b, %H:%M}. Delete it any time "
            f"with: {schedule.delete_text(when)}" if ok else f"Windows refused: {message}")


def schedule_section() -> None:
    """Overnight mode: a one-time Task Scheduler task. Nothing is created until the box is
    ticked and the button clicked."""
    st.markdown(ui.section("Or schedule it for tonight"), unsafe_allow_html=True)
    with st.container(border=True):
        st.caption("Creates a one-time Windows Task Scheduler task that runs this brief while you "
                   "are away. The computer must be on and not asleep, you must be signed in, and "
                   "Ollama must be running at that time. Runs here, nothing is uploaded.")
        default = next_two_am()
        c1, c2 = st.columns(2)
        day = c1.date_input("Date", value=default.date(), min_value=datetime.now().date(),
                            key="sched_date")
        at = c2.time_input("Time", value=default.time(), step=900, key="sched_time")
        when = datetime.combine(day, at)
        job = schedule.SCHEDULED / f"{schedule.task_name(when)}.json"
        st.markdown("The exact command that will run:")
        st.code(schedule.command_text(schedule.create_args(when, job)), language="text")
        agreed = st.checkbox("Yes, create this scheduled task on this computer", key="sched_ok")
        st.button("Create scheduled task", disabled=not agreed, on_click=schedule_run,
                  args=(when,))
        msg = st.session_state.get("schedule_msg")
        if msg:
            (st.success if msg[0] == "success" else st.error)(msg[1])
    pending_notice()


def pending_notice() -> None:
    for job in schedule.pending():
        st.info(f"Scheduled for {datetime.fromisoformat(job['when']):%A %d %b, %H:%M}: "
                f"{job['brief'][:80]}. Open the Plan page in the morning.")


def revision_block(item: dict) -> None:
    """One line on the card about the revision pass (Strong drafts are not revised)."""
    rv = item.get("revision")
    if not rv or not rv.get("attempted"):
        if rv and rv.get("status", "").startswith(("could not", "revision")):
            st.markdown(f'<div class="sp-rw"><b>Not revised.</b> {html.escape(rv["status"])}</div>',
                        unsafe_allow_html=True)
        return
    if "p_revised" not in rv:
        st.markdown(f'<div class="sp-rw"><b>Revision failed.</b> {html.escape(rv["status"])}</div>',
                    unsafe_allow_html=True)
    elif rv["kept"]:
        move = ("" if rv["label_original"] == rv["label_revised"] else
                f" ({html.escape(rv['label_original'])} → {html.escape(rv['label_revised'])})")
        st.markdown(f'<div class="sp-rw"><b>Revised once:</b> {rv["p_original"]:.0%} → '
                    f'{rv["p_revised"]:.0%}{move}. This is the revised draft. Compared below the '
                    f'week.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="sp-rw"><b>Revision tried, not kept:</b> it scored '
                    f'{rv["p_revised"]:.0%}, not above {rv["p_original"]:.0%}. The original stays.'
                    f'</div>', unsafe_allow_html=True)


def revision_compare(items: list[dict], summary: dict | None) -> None:
    """Full-width before/after for every revised draft, side by side with the score change."""
    done = [it for it in items if (it.get("revision") or {}).get("revised_draft")]
    if not done:
        return
    st.markdown(ui.section("Revisions: before → after"), unsafe_allow_html=True)
    if summary:
        st.write(f"{summary['kept']} of {summary['attempted']} revisions were kept; "
                 f"{summary['changed_band']} changed band; strong drafts "
                 f"{summary['strong_before']} → {summary['strong_after']}.")
    st.caption("Gemma writes the revision; the score comes from TabPFN. A revision is kept only if "
               "TabPFN scores it higher than the original. Picking the higher of two scores favours "
               "whatever the model likes, so a gain here is the model's own opinion, not a measured "
               "result on real posts.")
    for it in done:
        rv = it["revision"]
        before = it["original"]["draft"] if it.get("original") else it["draft"]
        with st.container(border=True):
            st.markdown(f'**{it["when"]["utc"]}** · {rv["p_original"]:.0%} → {rv["p_revised"]:.0%} '
                        f'chance of reactions' + ("" if rv["kept"] else " (revision not kept)"))
            a, b = st.columns(2)
            for col, head, p, dr, label in (
                    (a, "Before", rv["p_original"], before, rv["label_original"]),
                    (b, "After", rv["p_revised"], rv["revised_draft"], rv["label_revised"])):
                col.markdown(f'{head} · {p:.0%} {ui.badge(label)}', unsafe_allow_html=True)
                col.markdown(f'<div class="sp-title">{html.escape(dr["title"])}</div>',
                             unsafe_allow_html=True)
                col.caption(f"Tags: {', '.join(dr['tags'])}  \nCover: {dr['cover_idea']}  \n"
                            f"Reading time: {dr['reading_time_minutes']} min")


def show_card(run: Path, i: int, item: dict) -> dict:
    """One compact week card with details in expanders. Returns the (possibly edited) item."""
    key = f"{run.name}-{i}"
    d = item["draft"]
    ss = st.session_state
    # Widget values are in session state before the widgets are drawn, so the card top can show
    # the edited title even though the edit fields sit lower down.
    edits = {"title": ss.get(f"t-{key}", d["title"]),
             "tags": ss.get(f"g-{key}", ",".join(d["tags"])),
             "cover_idea": ss.get(f"c-{key}", d["cover_idea"]),
             "outline": ss.get(f"o-{key}", "\n".join(d["outline"]))}
    edited, err = apply_edits(item, edits)
    with st.container(border=True):
        top, right = st.columns([5, 3])
        when = item["when"]
        top.markdown(f'<div class="sp-when">{html.escape(when["utc"])}</div>'
                     f'<div class="sp-local">{html.escape(when["local"])} (your time)</div>',
                     unsafe_allow_html=True)
        right.markdown(f'<div style="text-align:right">{ui.badge(item["label"])}</div>',
                       unsafe_allow_html=True)
        st.markdown(f'<div class="sp-title">{html.escape(edited["draft"]["title"])}</div>'
                    f'<div class="sp-chance"><b>{item["p_yes"]:.0%}</b> chance of any reactions</div>',
                    unsafe_allow_html=True)
        status = st.container()  # always present, so the expanders below keep their open state
        with status:
            if err:
                st.error(f"These edits can't be used yet ({err}). The original draft is kept.")
            elif edited.get("edited"):
                st.info("Edited. The score is for the original draft; edits are not rescored.")
        revision_block(item)
        with st.expander("Tags, cover, outline (edit)"):
            st.text_input("Title", d["title"], key=f"t-{key}")
            st.text_input("Tags (comma separated, up to 4)", ",".join(d["tags"]), key=f"g-{key}")
            st.text_input("Cover image idea", d["cover_idea"], key=f"c-{key}")
            st.text_area("Outline (one point per line)", "\n".join(d["outline"]),
                         key=f"o-{key}", height=130)
            st.caption(f"Target reading time: {d['reading_time_minutes']} min")
            if d["notes"]:
                st.caption("Notes: " + "; ".join(d["notes"]))
        with st.expander("Features (computed by code)"):
            st.json(edited["features"])
        with st.expander("What-if: what would change it?"):
            sup = [w for w in item["what_if"] if w.get("supported")]
            rest = [w for w in item["what_if"] if not w.get("supported")]
            if not sup:
                st.write(item["reason"])
            st.markdown("**Backed by the data** (the signal summary shows a clear difference)")
            if sup:
                for w in sup:
                    st.write(w.get("sentence") or f"{w['delta']:+.1%} points: {w['text']} ({w['p']:.0%})")
            else:
                st.caption("No what-if feature is backed by the data.")
            st.markdown("**The model's guess**")
            st.caption("The data shows no clear difference for these, or points the other way, so "
                       "treat them as guesses.")
            for w in rest:
                st.markdown(":gray[" + (w.get("sentence")
                                         or f"{w['delta']:+.1%} points: {w['text']} ({w['p']:.0%})") + "]")
    return edited


def sloth_line(started: float) -> None:
    mins, secs = divmod(int(time.time() - started), 60)
    st.markdown(f'<div class="sp-elapsed">{mins}:{secs:02d} <span class="sp-small">elapsed, '
                f'about 10 minutes in total</span></div>'
                f'<div class="sp-lede"><span class="sp-sloth">🦥</span> Slow and steady. '
                f'Sloths don\'t rush, and neither does this run.</div>', unsafe_allow_html=True)


@st.fragment(run_every=4)
def live_status() -> None:
    state = run_state()
    if not state["active"]:
        st.rerun()
    text = CURRENT_LOG.read_text(encoding="utf-8", errors="replace") if CURRENT_LOG.exists() else ""
    sloth_line(state["started"])
    st.markdown(ui.steps_html(ui.parse_progress(text)), unsafe_allow_html=True)
    st.caption(f"Started {state['started_text']}. This page updates by itself.")
    with st.expander("Details"):
        st.code("\n".join(text.splitlines()[-12:]) or "Starting...")


def good_morning(run: Path) -> None:
    """The latest finished run, opened first: when it finished and a pointer to the verdict."""
    done = datetime.fromtimestamp((run / "drafts.json").stat().st_mtime)
    hours = (datetime.now() - done).total_seconds() / 3600
    minutes = round(hours * 60)
    ago = ("just now" if minutes < 1 else f"{minutes} minute{'s' if minutes != 1 else ''} ago") if hours < 1 else (
        f"{hours:.0f} hours ago" if hours < 48 else f"{hours / 24:.0f} days ago")
    greeting = "Good morning. Your run is ready." if hours < 18 else "Your latest run."
    st.markdown(ui.section(greeting), unsafe_allow_html=True)
    st.caption(f"It finished at {done:%H:%M} on {done:%A %d %b} ({ago}).")
    pending_notice()


def verdict_banner(verdict: dict) -> None:
    """The run's top line. With no Strong draft it says so and still names the best candidate."""
    best = verdict.get("best")
    if verdict["n_strong"] == 0:
        extra = (f'Best candidate: {html.escape(best["title"])} '
                 f'({best["p_yes"]:.0%} chance of reactions, {html.escape(best["label"])}).'
                 if best else "")
        st.markdown(f'<div class="sp-verdict sp-verdict-none"><div class="sp-verdict-head">'
                    f'{html.escape(verdict["text"])}</div><div>{extra}</div></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="sp-verdict"><div class="sp-verdict-head">'
                    f'{html.escape(verdict["text"])}</div></div>', unsafe_allow_html=True)


def label_legend(data: dict) -> None:
    """Compact legend: badge plus the measured meaning, then the honesty note."""
    rows = "".join(f"<div>{ui.badge(name)}<span>{html.escape(text)}</span></div>"
                   for name, text in data["label_meaning"].items())
    st.markdown(f'<div class="sp-legend">{rows}</div>', unsafe_allow_html=True)
    st.caption("Measured on the backtest test window. Lift is modest: use the labels to rank "
               "ideas, not to promise results.")


def page_plan() -> None:
    page_title("The week's plan", "Next week's drafts by day and time, each with a label and a chance of reactions. "
               "Edit what you like, then download the plan.")
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
    data = normalize_labels(load_json(run / "drafts.json"))
    verdict = data.get("verdict") or run_verdict(data["drafts"])
    good_morning(run)
    verdict_banner(verdict)
    st.caption(f"Run {run.name}. Brief: {data['brief']}")
    label_legend(data)
    items = sorted(data["drafts"], key=lambda it: (it["features"]["weekday"], it["features"]["hour"]))
    edited: list[dict] = []
    for start in range(0, len(items), 2):
        cols = st.columns(2)
        for col, i in zip(cols, range(start, min(start + 2, len(items))), strict=False):
            with col:
                edited.append(show_card(run, i, items[i]))
    revision_compare(items, data.get("revision_summary"))
    if data["failed_slots"]:
        st.warning("Some slots could not be drafted: " + "; ".join(data["failed_slots"]))
    st.download_button("Download plan.csv", plan_csv(edited, verdict["text"]), file_name="plan.csv",
                       mime="text/csv", type="primary")
    t = data["timings_s"]
    st.caption("Run time: " + ", ".join(f"{k.split(' ', 1)[-1]} {v:.0f}s" for k, v in t.items()))


def page_instagram() -> None:
    page_title("Instagram sample: a test case with no signal",
               "One account from a public Kaggle dataset (CC BY 4.0, Kundan Sagar Bedmutha), here to "
               "show that Slow Post says so when a history has nothing to learn. The verdict is "
               "computed, not scripted. Backtest only.")
    sample = ROOT / "sample" / "sample_posts.csv"
    if not sample.exists():
        st.error("sample/sample_posts.csv is missing.")
        return
    data = load_json(BACKTESTS / "instagram.json")
    if st.button("Run backtest again (about 1 minute)" if data else "Run backtest on the sample (about 1 minute)",
                 type="secondary" if data else "primary"):
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

def go(page: str) -> None:
    st.session_state.nav = page
    st.session_state.scroll_top = True  # a new page should open at its title


def scroll_to_top() -> None:
    """Open a new page at its title. The script differs on every call (a counter), because an
    identical script would not be run again by the browser, and it retries for a second while
    the new page finishes drawing."""
    if st.session_state.pop("scroll_top", False):
        st.session_state.scroll_count = st.session_state.get("scroll_count", 0) + 1
        components.html(
            "<script>/* " + str(st.session_state.scroll_count) + " */"
            "const go = () => { const m = window.parent.document.querySelector("
            "'[data-testid=\"stMain\"]'); if (m) m.scrollTo(0, 0); };"
            "go(); [100, 300, 700, 1200].forEach((t) => setTimeout(go, t));</script>", height=0)


def sidebar_nav() -> str:
    """Brand, the four steps in order, and the Instagram sample set apart as a secondary item."""
    st.sidebar.markdown('<div class="sp-brand">🦥 Slow Post</div>'
                        '<div class="sp-tagline">Drafts next week\'s posts overnight and flags the '
                        'weak ones. Local only.</div>', unsafe_allow_html=True)
    st.session_state.setdefault("nav", "0 Home")
    current = st.session_state.nav
    for page_id, label, icon in NAV:
        st.sidebar.button(label, icon=icon, key=f"nav-{page_id}", width="stretch",
                          type="primary" if current == page_id else "secondary",
                          on_click=go, args=(page_id,))
    st.sidebar.markdown('<div class="sp-nav-label">Test case</div>', unsafe_allow_html=True)
    st.sidebar.button(SECONDARY[1], icon=SECONDARY[2], key="nav-secondary", width="stretch",
                      type="primary" if current == SECONDARY[0] else "secondary",
                      on_click=go, args=(SECONDARY[0],))
    return current


st.markdown(ui.css(), unsafe_allow_html=True)
page = sidebar_nav()
scroll_to_top()
{"0 Home": page_home, "1 Data": page_data, "2 Backtest": page_backtest, "3 Brief": page_brief, "4 Plan": page_plan,
 SECONDARY[0]: page_instagram}[page]()
