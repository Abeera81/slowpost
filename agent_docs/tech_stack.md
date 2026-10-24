# Tech Stack

Last verified: 2026-10-23. The M0 spike, the DEV fetch and both backtests have run on this laptop (Python 3.14.4, tabpfn 9.2.0, torch 2.14.1). Streamlit, drafting and `run_overnight.py` are not built yet.

## Stack

| Area | Choice | Notes |
|------|--------|-------|
| Frontend | Streamlit (Python) | File upload and download built in; one language |
| Backend | Python 3.14 in `.venv` | Fall back to Python 3.12 if a package fails to install |
| Prediction | TabPFN 9.2.0, local `tabpfn` package, CPU | Weights download on first run after accepting the licence at platform.priorlabs.ai; `TABPFN_TOKEN` in `.env` if needed |
| Drafting | Ollama + `gemma3:4b` | Chosen by the spike (valid draft in 64 s) |
| Data | pandas, scikit-learn, scipy, pydantic | CSV/JSON files under `data/`; no database. DEV posts come from DEV's public API |
| Auth | Not applicable | No accounts |
| Styling | Streamlit theme | Sloth theme; green/amber/red plus words |
| Deployment | Not applicable | Local only on a Windows laptop (16 GB RAM, about 7.3 GB free, integrated GPU, so treat as CPU-only) |

## Commands (Git Bash on Windows)

- Install Ollama (one time; ask the user first): `winget install Ollama.Ollama`, or the installer from ollama.com
- Gemma: `gemma3:4b` is already pulled. Ask before downloading any other model.
- Create the venv: `python -m venv .venv`
- Activate: `source .venv/Scripts/activate`
- Install packages: `pip install -r requirements.txt`
- Machine check: `python scripts/spike.py`
- Fetch DEV posts: `python scripts/fetch_dev_posts.py [--days 60]`
- Backtest: `python scripts/run_backtest.py --source dev` or `--source instagram`
- Dev: `streamlit run app.py`
- Draft demo: `python scripts/draft_demo.py --brief "..." [--n 5]`
- Headless run: `python scripts/run_overnight.py --brief "..." [--facts "..."] [--n 5]` (about 10 min; writes `data/runs/<timestamp>/`)
- Test: `pytest` (skip slow tests: `pytest -m "not slow"`)
- Typecheck: Not applicable (type hints are used, but no type checker was chosen)
- Lint/format: `ruff check .`
- Build: Not applicable
- Browser check: `streamlit run app.py`, then the manual flow in `agent_docs/testing.md`

The PowerShell `Set-ExecutionPolicy` step in the Tech Design is not needed in Git Bash.

## AI Runtime

- Provider/runtime: local only. Ollama (Gemma) and the `tabpfn` package. Never `tabpfn-client`.
- Model can see:
  - Public: DEV posts fetched from DEV's public API (no author names stored), and the sample slice of the Kaggle Instagram Analytics dataset (CC BY 4.0)
  - User-owned: an uploaded past-posts CSV and the brief, on the local machine only
  - Never send: anything off the laptop. No cloud calls, no telemetry.
- Tools/actions: read files; write only under `data/` (`data/uploads/`, `data/runs/<timestamp>/`)
- Approval gates: none in the app, because it never posts or sends
- Retention/training setting to verify: confirm TabPFN runs local inference (no network calls after weights are cached); confirm the app works with Wi-Fi off
- Fallback: plain messages for Ollama not running (include the start command), TabPFN weights missing (point to the licence setup), and a draft invalid twice ("could not draft", the other slots continue)

## Important Patterns

- Modules in `slowpost/`: `profiles.py` (one profile per source), `data.py`, `labels.py`, `features.py`, `predict.py`, `signals.py`, `compare.py` (logistic regression and gradient boosting on the same split, random-split TabPFN), `controls.py` (cover vs reading-time checks), `schedule.py` (Task Scheduler), `ui.py` (palette, CSS, progress parser), `devfetch.py`, `drafts.py` (DEV drafting), `score.py` (labels, what-ifs, rewrite), `pipeline.py`, `timefmt.py`, `runs.py`. `app.py` is UI only.
- A profile holds the column map, outcome, label rule, features and draft fields. A Meta export becomes a new profile.
- DEV label: reactions >= 1. Instagram label: thirds of computed engagement, thresholds from training rows only.
- Backtest: test = seeded random sample (up to 2,000) of the newest 7 days (newest 20% if the week has under 100 posts); train = seeded random sample (up to 1,000) of all older posts. Predict "yes" when P(yes) is above the training share of "yes". Report balanced accuracy, ROC AUC and a plain verdict.
- Gemma calls: Ollama `format` = JSON schema, temperature about 0.7, `keep_alive=0` after drafting. Validate with pydantic; retry once.
- What-if variants are scored in one TabPFN batch call. The biggest gain in P(above) becomes the reason.
- Each run writes `data/runs/<timestamp>/` with `backtest.json`, `drafts.json`, `plan.csv`, `log.txt`.
- Errors: explicit, with a plain message to the user. Type hints throughout.
