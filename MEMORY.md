# Memory

Update this after major decisions, completed phases, or bugs that future agents need to know about. Keep it short. This shared project handoff is separate from a coding tool's private automatic memory.

## Current State

- Current task: M4 built (Streamlit `app.py`) and browser-checked in Chrome. M5 (evidence, README, demo) and M6 (article) are next.
- Current phase: M1 done. M2 (drafting) is next.
- Next step: M2 drafting with Gemma per source profile. README credits done (Kundan Sagar Bedmutha, Kaggle link; DEV public API).
- Blocked by: none.

## Decisions

- 2026-10-23 Shell is Git Bash on Windows (user instruction). This replaces the Tech Design's PowerShell commands: use `source .venv/Scripts/activate`. It still runs natively on Windows, so the reason WSL was rejected (memory cap) doesn't apply.
- 2026-10-23 Full dataset moved to `data/Instagram_Analytics.csv`. `data/`, `.venv/`, `.env` are git-ignored.
- 2026-10-23 Stack, metric, labels, features, and 80/20 time split as in `docs/TechDesign-SlowPost-MVP.md`.
- 2026-10-23 Decision change (user): Shandrika's real export is not coming. Slow Post works on any post history via one source profile per data source. Demo on real DEV posts (DEV public API, tags webdev, python, javascript, beginners, posts at least 14 days old, no author names or usernames). Kaggle Instagram stays as the no-signal test case and the app must report that honestly. Target user: anyone who posts regularly.
- 2026-10-23 Signal check (23 Oct 2026, `scripts/signal_check.py`): the Kaggle Instagram file is close to random noise. Pooled over all 20 accounts no feature moves engagement (every Kruskal-Wallis p above 0.12; label mix about 33/33/33 in every group). Account 15 alone shows p below 0.05 for media_type, has_call_to_action and hashtags_count, but the effects are small and likely chance across 20 accounts x 6 features. Pooled time-ordered backtest (1,000 train, 1,500 test): TabPFN 0.341 vs baseline 0.317, about 1.9 standard errors, with TabPFN predicting 'around' 0.3% of the time.

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
- DEV API probe (2026-10-23, read-only, nothing saved): `per_page=1000` works; one page covers only about 2-3 days for webdev (about 500 posts/day) and about 9 days for beginners, so reaching 14+ days old needs many pages (12 pages: webdev reaches 24 Sep, beginners 16 Jul). Rows overlap across tags (about 1% for beginners). Among posts at least 14 days old: about 87% (webdev) and 85% (beginners) have 0 reactions, 12% / 11% have 1-4, 1% / 4% have 5+; median 0. About 28-31% have a cover image, mean 3.8 tags per post, 7-9% have comments. `tag_list` is a list; the `user` object (author) must not be stored.
- M1 results (2026-10-23, `scripts/run_backtest.py`):
  - DEV fetch (`--days 60`): 44,428 unique posts, 2026-08-10 to 2026-10-09, 73 requests (webdev 37 pages, python 16, javascript 11, beginners 9). 15.6% have at least 1 reaction (label "some_reactions"). Label = reactions >= 1.
  - DEV backtest (revised 2026-10-23: cutoff + new test window): train 1,000 sampled (seed 0) from older posts; test 2,000 sampled from the newest 7 days (5,863 posts, 2 Oct to 9 Oct). Cutoff = training share of 'yes' = 0.141 (predict yes when P > cutoff). Train 14.1% yes, test 21.6% yes. TabPFN balanced accuracy 0.544 vs baseline 0.500 (gap +0.044, CI +0.019 to +0.070); ROC AUC 0.588 (CI 0.556-0.620). Verdict: better than baseline (weak signal). Raw accuracy 0.492 vs 0.784 (reference only; the model flags 57% of posts as yes). Cover image is the strongest single feature (no cover: 87.9% no-reaction vs cover: 76.6%).
  - DEV lift (same run): model's top 10% (200 test posts) 36.0% got reactions (95% CI 29.5-42.5%) vs 21.6% overall (CI 19.9-23.4%), lift 1.67x (1.37-1.97). Top 20% (400): 29.8% (25.0-34.5%), lift 1.38x (1.18-1.56).
  - M2 demo (5 drafts, gemma3:4b, 163 s total, 28-40 s each, all schema-valid first run): cover idea always non-empty so has_cover_image is always 1 (decision); Gemma tags were mostly invented words and drafts 4 and 5 got the same title; some content errors (TabPFN described as retrieval). Day and hour come from code (best days/hours in the history), not Gemma.
  - M3 real run (2026-10-23, `run_overnight.py`, n=5, with facts): 621 s total = drafting 403 s (longer prompt with 200 allowed tags, 2 title regenerations), backtest fit+predict 112 s, scoring + what-ifs 31 s, rewrite 39 s, rescoring 36 s. Cutoff 0.141, Strong threshold (top 20% of test) 0.200. Test-window evidence: Strong group 29.8% got reactions (1.38x), Okay 21.1%, Weak 18.1%, overall 21.6%. Labels of the 5 drafts: Strong 2, Okay 2, Weak 1. Rewrite of the weak draft (adding a '?') rescored 14.0% -> 22.1% (Strong). Caveat: the question-mark effect is the model's belief; the signal summary found title_has_question not clear after correction (p 0.036 raw), so the reason text overstates evidence. Free RAM dropped to 1.79 GB during Gemma rewrite.
  - M4 (2026-10-23): `app.py` has 5 pages (Data, Backtest, Brief, Plan, Instagram sample). Heavy work runs in subprocesses (fetch, backtest `--save` -> `data/backtests/*.json`, overnight run with `data/runs/_current.json` pid file), so refresh does not stop a run. What-if panel has two groups ("Backed by the data" = clear after correction; "The model's guess" greyed). Rewrite hint comes only from backed-by-data title changes (unit-tested); in the real UI run the one Weak draft was skipped with the reason shown because the title-question change was not backed. Edits are validated and features recomputed by code but NOT rescored (UI says so); plan.csv flags edited rows. Chrome check: missing CSV, Ollama down (test instance on port 8502 with SLOWPOST_DEV_CSV pointing nowhere and OLLAMA_HOST=http://127.0.0.1:1), empty brief, full run (559 s), edit, CSV download all OK. Screenshots in `docs/screenshots/` (no DEV titles or authors). Real run: labels Strong 3, Okay 1, Weak 1; supported features: has_cover_image, hour, main_tag, reading_time_minutes, title_has_number, weekday.
  - Instagram sample (account 15): 1,000 train; newest week too small so the newest 20% (320 posts) is the test window. Balanced accuracy 0.379 vs 0.333 (gap +0.045, CI -0.001 to +0.095); AUC 0.526 (0.479-0.573). Verdict: no better than baseline. No feature has a corrected p below 0.05 (smallest 0.12, hashtags).
  - Training is a random seeded sample from all older posts (user decision), not the 1,000 just before the test window.
- Open: does the Kaggle data have real patterns (it may be simulated)? The signal check says little or none.
- Resolved: Shandrika's real export is not coming.

## Completed

- [x] Agent setup files (AGENTS.md, CLAUDE.md, MEMORY.md, REVIEW-CHECKLIST.md, agent_docs/)
- [x] M0 Spike: TabPFN + Gemma run on this laptop; Gemma size chosen (gemma3:4b)
- [x] M1 Profiles + data + backtest: `fetch_dev_posts.py`, source profiles, labels, features, backtest (balanced accuracy, AUC if binary), signal summary, tests; sample slice (account 15) committed (Fri 23 Oct)
- [x] M2 Drafting: 7 schema-valid drafts from a brief (Sat 24 Oct morning)
- [x] M3 Scoring + rewrite: what-if reasons, single rewrite, `pipeline.py` run folder (Sat 24 Oct)
- [x] M4 UI: Streamlit upload, backtest, calendar, edit, CSV download (Sat 24 Oct afternoon)
- [ ] M5 Evidence: eval cases, timed real run, screenshots, demo, README (Sat 24 Oct night)
- [ ] M6 Article: DEV post published (Sun 25 Oct, deadline 11:59 PM PDT)
