# Tech Stack

Last verified: 2026-10. Only the Python version (3.14.4 on PATH) and the dataset columns have been checked. Package installs and all commands below are **unverified until the M0 spike**.

## Stack

| Area | Choice | Notes |
|------|--------|-------|
| Frontend | Streamlit (Python) | File upload and download built in; one language |
| Backend | Python 3.14 in `.venv` | Fall back to Python 3.12 if a package fails to install |
| Prediction | TabPFN 3.5, local `tabpfn` package, CPU | Weights download on first run after accepting the licence at platform.priorlabs.ai; `TABPFN_TOKEN` in `.env` if needed |
| Drafting | Ollama + Gemma | Try `gemma4:e2b`, then `gemma3:4b`, then `gemma3:1b`; the spike decides |
| Data | pandas, pydantic | CSV/JSON files under `data/`; no database |
| Auth | Not applicable | No accounts |
| Styling | Streamlit theme | Sloth theme; green/amber/red plus words |
| Deployment | Not applicable | Local only on a Windows laptop (16 GB RAM, about 7.3 GB free, integrated GPU, so treat as CPU-only) |

## Commands (Git Bash on Windows)

- Install Ollama (one time; ask the user first): `winget install Ollama.Ollama`, or the installer from ollama.com
- Pull Gemma (one time; ask first, it's a large download): `ollama pull gemma4:e2b`
- Create the venv: `python -m venv .venv`
- Activate: `source .venv/Scripts/activate`
- Install packages: `pip install -r requirements.txt` (tabpfn, streamlit, ollama, pandas, pydantic, pytest, ruff)
- Machine check: `python scripts/spike.py`
- Dev: `streamlit run app.py`
- Headless run: `python scripts/run_overnight.py --csv sample/sample_posts.csv --brief "..."`
- Test: `pytest` (skip slow tests: `pytest -m "not slow"`)
- Typecheck: Not applicable (type hints are used, but no type checker was chosen)
- Lint/format: `ruff check .`
- Build: Not applicable
- Browser check: `streamlit run app.py`, then the manual flow in `agent_docs/testing.md`

The PowerShell `Set-ExecutionPolicy` step in the Tech Design is not needed in Git Bash.

## AI Runtime

- Provider/runtime: local only. Ollama (Gemma) and the `tabpfn` package. Never `tabpfn-client`.
- Model can see:
  - Public: the sample slice of the Kaggle Instagram Analytics dataset (CC BY 4.0)
  - User-owned: an uploaded past-posts CSV and the brief, on the local machine only
  - Never send: anything off the laptop. No cloud calls, no telemetry.
- Tools/actions: read files; write only under `data/` (`data/uploads/`, `data/runs/<timestamp>/`)
- Approval gates: none in the app, because it never posts or sends
- Retention/training setting to verify: confirm TabPFN runs local inference (no network calls after weights are cached); confirm the app works with Wi-Fi off
- Fallback: plain messages for Ollama not running (include the start command), TabPFN weights missing (point to the licence setup), and a draft invalid twice ("could not draft", the other slots continue)

## Important Patterns

- Pipeline modules in `slowpost/`: `data.py`, `labels.py`, `features.py`, `predict.py`, `drafts.py`, `pipeline.py`. `app.py` is UI only.
- Column map in `data.py` so a Meta Business Suite export can map to the same names later.
- Engagement = `(likes + comments + shares + saves) / reach`. Thirds use thresholds from training rows only.
- Backtest: sort by `post_datetime`, oldest 80% train and newest 20% test; cap training at the most recent 1,000 rows.
- Gemma calls: Ollama `format` = JSON schema, temperature about 0.7, `keep_alive=0` after drafting. Validate with pydantic; retry once.
- What-if variants are scored in one TabPFN batch call. The biggest gain in P(above) becomes the reason.
- Each run writes `data/runs/<timestamp>/` with `backtest.json`, `drafts.json`, `plan.csv`, `log.txt`.
- Errors: explicit, with a plain message to the user. Type hints throughout.
