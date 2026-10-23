# Memory

Update this after major decisions, completed phases, or bugs that future agents need to know about. Keep it short. This shared project handoff is separate from a coding tool's private automatic memory.

## Current State

- Current task: Agent setup done (Part 4). No app code yet.
- Current phase: Before M0 (spike).
- Next step: Propose the M0 spike plan (`requirements.txt`, venv, `scripts/spike.py`: TabPFN fit/predict on 300 rows on CPU; Gemma returns one schema-valid draft; record times and free RAM). Wait for approval before building.
- Blocked by: none. M0 needs Ollama installed, a Gemma model pulled, and the TabPFN licence accepted at platform.priorlabs.ai. The user does or approves these.

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
- Python 3.14.4 is on PATH. Package installs not yet tried.
- Open: does the Kaggle data have real patterns (it may be simulated)? The backtest will show.
- Open: will the real export from Shandrika arrive, and in which column format?

## Completed

- [x] Agent setup files (AGENTS.md, CLAUDE.md, MEMORY.md, REVIEW-CHECKLIST.md, agent_docs/)
- [ ] M0 Spike: TabPFN + Gemma run on this laptop; Gemma size chosen (target Fri 23 Oct)
- [ ] M1 Data + backtest: data/labels/features/predict, sample slice committed, tests pass (Fri 23 Oct)
- [ ] M2 Drafting: 7 schema-valid drafts from a brief (Sat 24 Oct morning)
- [ ] M3 Scoring + rewrite: what-if reasons, single rewrite, `pipeline.py` run folder (Sat 24 Oct)
- [ ] M4 UI: Streamlit upload, backtest, calendar, edit, CSV download (Sat 24 Oct afternoon)
- [ ] M5 Evidence: eval cases, timed real run, screenshots, demo, README (Sat 24 Oct night)
- [ ] M6 Article: DEV post published (Sun 25 Oct, deadline 11:59 PM PDT)
