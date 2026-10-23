# Memory

Update this after major decisions, completed phases, or bugs that future agents need to know about. Keep it short. This shared project handoff is separate from a coding tool's private automatic memory.

## Current State

- Current task: M0 spike passed.
- Current phase: M0 done; M1 (data + backtest) is next.
- Next step: Propose M1: `data.py`, `labels.py`, `features.py`, backtest in `predict.py`, `sample/sample_posts.csv` (account 15), tests. Wait for approval.
- Blocked by: none.

## Decisions

- 2026-10-23 Shell is Git Bash on Windows (user instruction). This replaces the Tech Design's PowerShell commands: use `source .venv/Scripts/activate`. It still runs natively on Windows, so the reason WSL was rejected (memory cap) doesn't apply.
- 2026-10-23 Full dataset moved to `data/Instagram_Analytics.csv`. `data/`, `.venv/`, `.env` are git-ignored.
- 2026-10-23 Stack, metric, labels, features, and 80/20 time split as in `docs/TechDesign-SlowPost-MVP.md`.

## AI / Tooling Decisions

- 2026-10-23 TabPFN local `tabpfn` only, never `tabpfn-client`. Gemma via Ollama: try `gemma4:e2b`, then `gemma3:4b`, then `gemma3:1b`. The spike decides.
- 2026-10-23 Gemma and TabPFN never loaded together (`keep_alive=0` after drafting).

## Known Issues / Facts checked

Checked 2026-10-23 by reading `data/Instagram_Analytics.csv` (setup only, no app code):

- 29,999 rows, 20 accounts, posts from 2024-11-19 to 2025-11-19. Header matches the Tech Design data contract. No rows with empty or zero `reach`.
- Largest accounts: `account_id` 15 and 13 tie at 1,599 posts; 17 has 1,572. Every account has well over 150 posts.
- 2026-10-23 Decision (user): the sample uses `account_id` 15.
- Ollama 0.40.8 is installed; `gemma3:4b` is already pulled. `gemma4:e2b` is not pulled.
- `media_type`: carousel, image, reel (the PRD says "photo"; use "image").
- `content_category`: Beauty, Comedy, Fashion, Fitness, Food, Lifestyle, Music, Photography, Technology, Travel.
- Python 3.14.4 is on PATH; all packages in requirements.txt installed fine on it.
- M0 spike results (2026-10-23, `python scripts/spike.py`, account 15, 1,599 usable posts, CPU, Python 3.14.4, tabpfn 9.2.0, torch 2.14.1; packages installed fine on 3.14):
  - Gemma `gemma3:4b`: one schema-valid draft on attempt 1, 63.8 s, 3.05 GB RAM free before the call. Chosen model; `gemma4:e2b` not tried.
  - TabPFN 300 train / 100 test: fit 173.9 s (first run, likely includes loading weights), predict 28.2 s. Accuracy 0.26 vs baseline 0.30 ('above').
  - TabPFN 1,000 train / 300 test (the real backtest size): fit 0.5 s, predict 60.3 s, about 5 GB RAM free throughout. Accuracy 0.383 vs baseline 0.38 ('below').
  - Accuracies come from the oldest 1,300 posts of one account, not the real backtest. A 0.3-point gap is within noise. The features may carry little signal; M1 must report this honestly.
  - Predict at 1,000 rows takes about 1 min on CPU, so what-if scoring in M3 must be batched.
- Signal check (2026-10-23, `scripts/signal_check.py`): pooled over all 20 accounts, no feature moves engagement (every group's mean is within 0.5-3% of the overall mean; every Kruskal-Wallis p is above 0.12; label mix is about 33/33/33 in every group). Account 15 alone has p < 0.05 for media_type (0.033), has_call_to_action (0.009) and hashtags_count (0.004), but the effects are small, hashtags is not monotonic, and it is one of 20 accounts x 6 features, so chance is a likely explanation. Pooled time-ordered backtest (1,000 train, 1,500 test): TabPFN 0.341 vs baseline 0.317 (+0.023, about 1.9 standard errors); TabPFN predicted 'around' only 0.3% of the time, and the baseline's label is an arbitrary tie-break. The data looks close to random noise (possibly simulated). Plan M1 to report this honestly; do not overstate.
- Open: does the Kaggle data have real patterns (it may be simulated)? The signal check says little or none.
- Open: will the real export from Shandrika arrive, and in which column format?

## Completed

- [x] Agent setup files (AGENTS.md, CLAUDE.md, MEMORY.md, REVIEW-CHECKLIST.md, agent_docs/)
- [x] M0 Spike: TabPFN + Gemma run on this laptop; Gemma size chosen (gemma3:4b)
- [ ] M1 Data + backtest: data/labels/features/predict, sample slice committed, tests pass (Fri 23 Oct)
- [ ] M2 Drafting: 7 schema-valid drafts from a brief (Sat 24 Oct morning)
- [ ] M3 Scoring + rewrite: what-if reasons, single rewrite, `pipeline.py` run folder (Sat 24 Oct)
- [ ] M4 UI: Streamlit upload, backtest, calendar, edit, CSV download (Sat 24 Oct afternoon)
- [ ] M5 Evidence: eval cases, timed real run, screenshots, demo, README (Sat 24 Oct night)
- [ ] M6 Article: DEV post published (Sun 25 Oct, deadline 11:59 PM PDT)
