# Project Brief

## Product

- One-line vision: Drafts next week's Instagram posts overnight on your laptop and predicts which ones will underperform, from the account's own history.
- Target users: One person who plans an Instagram account's posts weekly (social media manager, small business owner, creator).
- Primary user outcome: In the evening, load the past-posts CSV and type a two-line brief, then start the run. In the morning, open a week calendar of 7 drafts, each labelled below, around or above with a reason and the backtest result. Edit them, download the CSV, and schedule them by hand.

## Scope

- Must ship (acceptance criteria are in `docs/PRD-SlowPost-MVP.md`):
  - Load past posts: upload or "Use sample data"; show row count, date range, columns found or missing, skipped rows
  - Backtest: time-ordered split; TabPFN vs majority-label baseline; plain verdict; saved result
  - Draft next week's posts: 7 text-only, schema-validated Gemma drafts; works offline
  - Score and rewrite: label, confidence, and plain-language reason per draft; "below" drafts rewritten at most once, keeping before and after
  - Calendar and export: 7 colour-and-word-labelled cards, editable, CSV download
- Nice to have (only after the core works): overnight scheduling, best-time-slot suggestion, run log with timings
- Not in v1:
  - Auto-posting, or logging in to Instagram/Meta
  - Image or video generation
  - Multiple accounts, other platforms
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

- AI is used for: drafting and rewriting caption text (Gemma), and predicting performance labels (TabPFN). Both run locally.
- AI is not used for: computing features or metrics, deciding what is posted, or anything that leaves the laptop.
- Human approval required for: nothing is automated outward. The user copies drafts out manually.
