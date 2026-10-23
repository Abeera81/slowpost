# Project Brief

## Product

- One-line vision: Learns from a post history, drafts next week's posts overnight on your laptop, and flags the ones likely to underperform, with an honest backtest that says when there is no signal.
- Target users: Anyone who posts regularly and plans posts ahead.
- Demo source: real DEV posts from DEV's public API (tags webdev, python, javascript, beginners). Test case: the Kaggle Instagram sample, where the app must report that it finds no signal.
- Primary user outcome: In the evening, load a post history (or fetched DEV posts) and type a short brief, then start the run. In the morning, open a plan of drafts, each with a predicted label, a reason, and the backtest result. Edit them and download the CSV.

## Scope

- Must ship (acceptance criteria are in `docs/PRD-SlowPost-MVP.md`):
  - Source profiles (DEV, Instagram), with a column summary and clear errors for missing columns
  - Backtest: time-ordered; TabPFN vs most-common-label baseline; balanced accuracy at a training-share cutoff, ROC AUC, plain verdict; signal summary by feature
  - Draft next posts (DEV first): schema-validated Gemma drafts; works offline
  - Score and rewrite: label, confidence, plain-language reason per draft; weak drafts rewritten at most once
  - Plan and export: editable cards, CSV download
- Nice to have (only after the core works): overnight scheduling, best-time-slot suggestion, run log with timings, a Meta export profile
- Not in v1:
  - Auto-posting, or logging in to any platform
  - Image or video generation
  - Hosted version (would send data off the laptop)
  - Fine-tuning any model

## Principles

- Honest numbers: show the baseline next to the model and never hide a weak result.
- One screen at a time, with sensible defaults and no settings maze.
- Colour is never the only signal; labels are also written in words.
- Calm, clean, a little playful (sloth theme).
- Features that half-work are cut, not shipped.
- Solve the user story before adding polish.

## AI Position

- AI is used for: drafting and rewriting post text (Gemma), and predicting performance labels (TabPFN). Both run locally.
- AI is not used for: computing features or metrics, deciding what is posted, or anything that leaves the laptop.
- Human approval required for: nothing is automated outward. The user copies drafts out manually.
