"""Presentation helpers for the Streamlit app: palette, CSS, badges, run-progress parsing.

No pipeline, model or scoring logic lives here. The palette is defined once (PALETTE) and
becomes the CSS variables; .streamlit/config.toml repeats the theme colours (a test keeps the
two in sync).
"""
from __future__ import annotations

import html
import re

PALETTE = {
    "bg": "#faf7f0",         # page
    "panel": "#efe9dc",      # secondary background
    "sidebar": "#f3eee2",
    "text": "#3b3326",
    "muted": "#6b6252",      # captions; AA on bg and panel
    "primary": "#4a7a3c",    # buttons, active nav (white text on it passes AA)
    "border": "#dcd3bf",
    # Label colours: (text, background). Always shown together with the word.
    "strong_fg": "#2d5a22", "strong_bg": "#e1eed9",
    "close_fg": "#7a4a08", "close_bg": "#f8ebcf",
    "weak_fg": "#8c2b23", "weak_bg": "#f7e0dc",
}

LABEL_GLYPH = {"Strong": "▲", "Too close to call": "●", "Weak": "▼"}  # a second cue besides colour
LABEL_CLASS = {"Strong": "strong", "Too close to call": "close", "Weak": "weak"}
LEGACY_LABELS = {"Okay": "Too close to call"}  # runs saved before the rename

CSS = """
<style>
:root {
/*VARS*/
  --s1: .25rem; --s2: .5rem; --s3: .75rem; --s4: 1rem; --s5: 1.5rem; --s6: 2.5rem;
  /* The four text levels (used everywhere): title, section, body, help */
  --t-title: 2.5rem; --t-section: 1.3rem; --t-body: 1rem; --t-help: .85rem;
  --fs-hero: 3.6rem;
  --radius: 12px;
}
.block-container { max-width: 1180px; padding-top: 4rem; }
h1, h2, h3 { letter-spacing: -.01em; }
.sp-page-title { font-size: var(--t-title); font-weight: 800; line-height: 1.1; margin: 0 0 var(--s2); }
.sp-lede { color: var(--muted); font-size: var(--t-body); margin: 0 0 var(--s5); }
.sp-small { color: var(--muted); font-size: var(--t-help); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted); font-size: var(--t-help); }
.sp-section { font-size: var(--t-section); font-weight: 600; margin: var(--s6) 0 var(--s2); }

/* metric boxes: small muted label above a large number */
[data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--border);
  border-radius: var(--radius); padding: var(--s4); }
[data-testid="stMetricLabel"] p { color: var(--muted); font-size: var(--t-help); }
[data-testid="stMetricValue"] { font-size: 2.2rem; font-weight: 700; }

/* buttons: one filled primary per page; secondary actions are quieter */
[data-testid="stMain"] [data-testid="stBaseButton-primary"] { background: var(--primary);
  color: #fff; font-weight: 700; padding: var(--s2) var(--s4); border: 0; }
[data-testid="stMain"] [data-testid="stBaseButton-secondary"] { background: transparent;
  color: var(--muted); font-weight: 500; border: 1px solid var(--border); }
[data-testid="stMain"] [data-testid="stBaseButton-secondary"]:hover { color: var(--text);
  border-color: var(--muted); }

/* sidebar */
[data-testid="stSidebar"] { border-right: 1px solid var(--border); }
.sp-brand { font-size: var(--t-section); font-weight: 700; line-height: 1.1; margin: 0; }
.sp-tagline { color: var(--muted); font-size: var(--t-help); margin: var(--s1) 0 var(--s4); }
.sp-nav-label { color: var(--muted); font-size: var(--t-help); text-transform: uppercase;
  letter-spacing: .08em; margin: var(--s5) 0 var(--s1); }
[data-testid="stSidebar"] .stButton button { justify-content: flex-start; border: 0;
  box-shadow: none; padding: var(--s2) var(--s3); border-radius: 10px; }
[data-testid="stSidebar"] button > div { justify-content: flex-start !important; width: 100%; }
[data-testid="stSidebar"] button p { text-align: left; }
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] { background: transparent;
  color: var(--text); }
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover { background: var(--panel); }
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] { background: var(--primary);
  color: #fff; font-weight: 600; }

/* badges */
.sp-badge { display: inline-block; padding: 2px 10px; border-radius: 999px;
  font-size: var(--t-help); font-weight: 600; white-space: nowrap; }
.sp-strong { color: var(--strong_fg); background: var(--strong_bg); }
.sp-close { color: var(--close_fg); background: var(--close_bg); }
.sp-weak { color: var(--weak_fg); background: var(--weak_bg); }

/* week cards */
.sp-when { font-weight: 600; font-size: var(--t-body); margin: 0; }
.sp-local { color: var(--muted); font-size: var(--t-help); margin: 0 0 var(--s2); }
.sp-title { font-size: var(--t-section); font-weight: 700; line-height: 1.25; margin: var(--s2) 0; }
.sp-chance { margin-bottom: var(--s3); color: var(--muted); font-size: var(--t-help); }
.sp-chance b { color: var(--text); font-size: var(--t-section); }
.sp-legend { display: flex; flex-direction: column; gap: var(--s2); margin: var(--s2) 0 var(--s3);
  font-size: var(--t-help); }
.sp-legend div { display: flex; gap: var(--s3); align-items: baseline; }
.sp-legend .sp-badge { min-width: 5.5rem; text-align: center; }
.sp-rw { font-size: var(--t-help); margin: var(--s2) 0 0; }

/* home: how it works */
.sp-flow { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: var(--s3); margin: var(--s2) 0 var(--s4); }
.sp-flow-step { background: var(--panel); border: 1px solid var(--border);
  border-radius: var(--radius); padding: var(--s4); }
.sp-num { display: inline-block; width: 1.8rem; height: 1.8rem; line-height: 1.8rem;
  text-align: center; border-radius: 50%; background: var(--primary); color: #fff;
  font-weight: 700; margin-bottom: var(--s2); }
.sp-flow-name { font-weight: 700; font-size: var(--t-section); }

/* run verdict */
.sp-verdict { background: var(--strong_bg); color: var(--strong_fg); border-radius: var(--radius);
  padding: var(--s4) var(--s5); margin: 0 0 var(--s4); }
.sp-verdict-none { background: var(--close_bg); color: var(--close_fg); }
.sp-verdict-head { font-size: var(--t-section); font-weight: 700; }

/* backtest headline */
.sp-hero { background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius);
  padding: var(--s5); margin: 0 0 var(--s4); }
.sp-hero-num { font-size: var(--fs-hero); font-weight: 800; line-height: 1; color: var(--primary); }
.sp-hero-num.sp-none { color: var(--weak_fg); font-size: var(--t-section); }
.sp-hero-head { font-size: var(--t-section); font-weight: 700; margin: var(--s2) 0 var(--s1); }

/* progress */
.sp-steps { list-style: none; padding: 0; margin: var(--s3) 0; }
.sp-steps li { display: flex; gap: var(--s3); align-items: baseline; padding: var(--s2) 0;
  border-bottom: 1px solid var(--border); font-size: var(--t-body); }
.sp-steps .sp-mark { width: 1.2rem; text-align: center; font-weight: 700; }
.sp-steps .sp-time { margin-left: auto; color: var(--muted); font-size: var(--t-help); }
.sp-steps .sp-done .sp-mark { color: var(--strong_fg); }
.sp-steps .sp-active { font-weight: 700; }
.sp-steps .sp-active .sp-mark { color: var(--close_fg); }
.sp-steps .sp-wait { color: var(--muted); }
.sp-elapsed { font-size: var(--t-section); font-weight: 700; margin: 0; }
.sp-sloth { display: inline-block; animation: sp-sway 6s ease-in-out infinite;
  transform-origin: top center; }
@keyframes sp-sway { 0%, 100% { transform: rotate(-6deg); } 50% { transform: rotate(6deg); } }
@media (prefers-reduced-motion: reduce) { .sp-sloth { animation: none; } }
</style>
"""


def css() -> str:
    """The page CSS with the palette as CSS variables."""
    lines = "".join(f"  --{k}: {v};\n" for k, v in PALETTE.items())
    return CSS.replace("/*VARS*/", lines)


def badge(label: str) -> str:
    """HTML for a label badge: glyph + word + colour (never colour alone)."""
    label = LEGACY_LABELS.get(label, label)
    return (f'<span class="sp-badge sp-{LABEL_CLASS[label]}">{LABEL_GLYPH[label]} '
            f'{html.escape(label)}</span>')


def _luminance(hex_colour: str) -> float:
    rgb = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    """WCAG contrast ratio between two #rrggbb colours (4.5 is AA for normal text)."""
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


# ---------- run progress, read from the run log ----------

STEPS = [
    ("Drafting with Gemma", "1 drafting"),
    ("Backtest with TabPFN", "2a backtest"),
    ("Scoring drafts with TabPFN", "2b scoring"),
    ("Revising drafts", None),  # done when the run is saved
]
_DRAFTED = re.compile(r"Drafted (\d+)/(\d+)")
_TIME = re.compile(r"\]\s*(.+?): ([\d.]+) s \(")


def parse_progress(log: str) -> list[dict]:
    """Steps with state 'done' | 'active' | 'wait' and seconds (when known), from the run log.

    The log only reports a step when it finishes, so the first step without a finish line is
    'active'. Revising (and the rescore after it) counts as done when the run is saved.
    """
    seconds: dict[str, float] = {}
    for m in _TIME.finditer(log):
        seconds[m.group(1)] = float(m.group(2))
    saved = "Saved to" in log
    out, found_active = [], False
    for name, prefix in STEPS:
        if prefix is None:
            took = [v for k, v in seconds.items() if k.startswith(("3 revising", "4 rescoring"))]
            done, secs = saved, (sum(took) if took else None)
        else:
            hit = next((k for k in seconds if k.startswith(prefix)), None)
            done, secs = hit is not None, seconds.get(hit) if hit else None
        label = name
        if done:
            state = "done"
        elif not found_active:
            state, found_active = "active", True
            counts = _DRAFTED.findall(log) if prefix == "1 drafting" else []
            if counts:
                label = f"{name} {counts[-1][0]}/{counts[-1][1]}"
        else:
            state = "wait"
        out.append({"name": label, "state": state, "seconds": secs})
    return out


def steps_html(steps: list[dict]) -> str:
    marks = {"done": ("✓", "done"), "active": ("●", "running"), "wait": ("○", "waiting")}
    rows = []
    for s in steps:
        mark, word = marks[s["state"]]
        secs = f"{s['seconds']:.0f} s" if s["seconds"] is not None and s["state"] == "done" else word
        rows.append(f'<li class="sp-{s["state"]}"><span class="sp-mark">{mark}</span>'
                    f'<span>{html.escape(s["name"])}</span><span class="sp-time">{secs}</span></li>')
    return '<ul class="sp-steps">' + "".join(rows) + "</ul>"


# ---------- text levels as HTML (page title, section heading, helper text) ----------

def page_title(title: str, lede: str) -> str:
    """Level 1 (page title) plus one muted sentence saying what this step does."""
    return (f'<div class="sp-page-title">{html.escape(title)}</div>'
            f'<div class="sp-lede">{html.escape(lede)}</div>')


def section(text: str) -> str:
    """Level 2: section heading."""
    return f'<div class="sp-section">{html.escape(text)}</div>'
