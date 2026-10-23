# Product Requirements Document: Slow Post MVP

## Overview

**Product Name:** Slow Post
**Problem Statement:** Planning a week of posts takes an evening of writing, and even then the person guesses which posts will do well. The account's own past results show what works, if anything does, but nobody has time to read them before every post.
**MVP Goal:** A tool that runs on one laptop, learns from any account's post history, drafts next week's 7 posts overnight, and marks which drafts are likely to underperform. A backtest shows honestly whether its predictions beat a simple baseline, including when they do not. Demoed on real DEV posts from DEV's public API; the Kaggle Instagram dataset stays as the "no signal" test case.
**Target Launch:** Submitted to the DEV Hacktoberfest Week 3 "Slow and Steady" challenge by 25 Oct 2026, 11:59 PM PDT (26 Oct, 11:59 AM PKT).

## Decision Update (23 Oct 2026)

- **Real export dropped.** Shandrika's Instagram export is not coming. Slow Post is built to work on any post history, with one "source profile" per data source (column mapping, features, label rule, draft format).
- **Demo source: DEV posts.** Real posts fetched from DEV's public API for the tags webdev, python, javascript and beginners (posts at least 14 days old, no author names or usernames stored).
- **Instagram (Kaggle) is the no-signal test case.** Signal check (23 Oct 2026, `scripts/signal_check.py`): the Kaggle Instagram file is close to random noise. Pooled over all 20 accounts no feature moves engagement (every Kruskal-Wallis p above 0.12; label mix about 33/33/33 in every group). Account 15 alone shows p below 0.05 for media_type, has_call_to_action and hashtags_count, but the effects are small and likely chance across 20 accounts x 6 features. Pooled time-ordered backtest (1,000 train, 1,500 test): TabPFN 0.341 vs baseline 0.317, about 1.9 standard errors, with TabPFN predicting 'around' 0.3% of the time. The app must report a no-signal source honestly instead of hiding it.
- **Target user widened:** anyone who posts regularly, not only Instagram account managers.

## Target Users

### Primary User Profile
**Who:** Anyone who posts regularly and plans posts weekly: a developer writing on DEV, a social media manager, a small business owner, or a creator.
**Problem:** Writing a week of posts takes hours, and there is no quick way to know which drafts will flop before publishing.
**Current Solution:** Writing posts by hand or with a chatbot, then guessing; checking stats after the fact.
**Why They'll Switch:** Slow Post does the drafting overnight and predicts each draft's performance from that account's own history, on their own laptop, for free, without sending the account's data anywhere.

### User Persona: Javeria (builder and first user)
- **Demographics:** 20s, Pakistan, social media manager and software engineering student
- **Tech Level:** Intermediate
- **Goals:** Turn a past-posts export into a ready weekly plan with minimal effort
- **Frustrations:** Sunday-evening planning; guessing which post types and times work

## User Journey

### The Story
1. In the evening, Javeria picks a source (the DEV demo posts, a CSV export, or the sample dataset) and loads the past posts.
2. She types a two-line brief for next week, e.g. "Open house Saturday in Fairfield; staging tips; one client testimonial".
3. She clicks **Start overnight run** and goes to sleep.
4. Slow Post checks the history, runs the backtest, has Gemma write 7 drafts, scores each with TabPFN, rewrites low scorers once and scores them again.
5. In the morning she opens a calendar: 7 posts, each marked green, amber or red, with the reason and the backtest result at the top.
6. She edits what she likes, downloads the CSV and schedules the posts herself.

### Key Touchpoints
1. **Discovery:** The DEV article and GitHub README.
2. **First Contact:** README quick-start; one command starts the local app.
3. **Onboarding:** Upload screen with a sample CSV button so it works before the user has their own export.
4. **Core Loop:** Weekly: new export + brief → overnight run → morning calendar.
5. **Retention:** Each week's export adds more history, so predictions are based on more data.

## MVP Features

### Core Features (Must Have)

#### 1. Load past posts
- **Description:** Loads past posts from a source with a **source profile** (DEV posts fetched from DEV's public API, the Kaggle Instagram Analytics dataset, and room for a Meta Business Suite export later). The profile maps columns to a standard set, and defines the features, the label rule and the draft format. The app reports what it found.
- **User Value:** Works with the file the user already has; no account login.
- **Success Criteria:**
  - User can pick a source, upload a CSV, or click "Use sample data"
  - Adding a new source means adding a profile; the rest of the code does not change
  - System shows row count, date range, and which needed columns were found or missing
  - Rows missing an engagement value are skipped and counted, not silently dropped
  - Uploaded data stays on the local machine and is never committed to git
- **Priority:** Critical

#### 2. Backtest
- **Description:** Labels each past post using the source's label rule (Instagram: below, around or above by thirds; DEV: a rule chosen from the real reaction distribution). Hides the most recent posts in time order, predicts them with TabPFN using only older posts, and compares with a simple baseline (always guessing the most common label).
- **User Value:** Shows honestly whether the predictions can be trusted before using them.
- **Success Criteria:**
  - Split is by date: the model never sees posts newer than the ones it predicts
  - Screen shows TabPFN balanced accuracy (and ROC AUC when the label is binary) next to the most-common-label baseline, the number of test posts, and a plain verdict sentence
  - If TabPFN does not beat the baseline, the screen says so plainly
  - Result is saved so the article can quote it
- **Priority:** Critical

#### 3. Draft next week's posts
- **Description:** A local Gemma model (via Ollama) writes 7 text-only drafts from the brief and the account's best past captions.
- **User Value:** Removes the blank-page work.
- **Success Criteria:**
  - Each draft has the fields its source profile defines (Instagram: post type, suggested day and time, caption with call-to-action, hashtags, visual idea; DEV: title, tags, suggested day and time, outline; exact fields set at M2)
  - Output is checked against a fixed structure; a malformed draft is retried once, then reported, never shown half-filled
  - Works with the Wi-Fi off after models are downloaded
- **Priority:** Critical

#### 4. Score and rewrite
- **Description:** Code turns each draft into the same features used for past posts (defined by the source profile, e.g. weekday, hour, length, tag count); TabPFN predicts its label and confidence. Drafts predicted in the lowest band are rewritten once by Gemma with a hint and scored again.
- **User Value:** Flags likely flops before they are posted.
- **Success Criteria:**
  - Every draft shows label, confidence and the main reason in plain words
  - Feature values are computed by code, never by the language model
  - Rewrite happens at most once per draft; before and after scores are both kept
- **Priority:** Critical

#### 5. Calendar and export
- **Description:** A week view of the 7 drafts colour-coded by predicted performance, with editing and a CSV download.
- **User Value:** A ready plan to copy into any scheduler.
- **Success Criteria:**
  - User can read, edit and copy each draft
  - Download produces a CSV with one row per draft and all fields
  - Nothing is posted or sent anywhere
- **Priority:** Critical

### Nice to Have (if time allows)
- Overnight scheduling (cron or Task Scheduler) instead of a manual start button
- Best-time-slot suggestion per draft
- Run log with timings for the "lazy payoff" section of the article

## Out of Scope (Not in MVP)
| Feature | Why Wait | Planned For |
|---------|----------|-------------|
| Auto-posting to Instagram | Needs Meta API approval; risky | Version 2 |
| Logging in to Instagram / Meta | Not needed; CSV export is enough | Version 2 |
| Image or video generation | Large effort; not needed for scoring | Version 2 |
| Multiple accounts in one workspace | One account proves the idea | Version 2 |
| Sources with no profile yet (Meta export, Facebook, LinkedIn) | Each needs its own profile | Version 2 |
| Hosted web version | Would send data off the laptop | Not planned |
| Fine-tuning any model | Out of time | Not planned |

## Success Metrics

### Primary Metrics
1. **Honest backtest for both sources:** DEV posts and the Instagram no-signal case each get a backtest with balanced accuracy (and ROC AUC when binary) next to the baseline, by 26 Oct 2026. A win is not required; a plain verdict is.
   - How to measure: Backtest screen (Feature 2)
   - Why it matters: Shows where the predictions add value and where they do not
2. **Real weekly plan completed:** One full week of 7 drafts produced and timed against planning by hand, by 25 Oct 2026
   - How to measure: Timed run log plus a hand-planning estimate
   - Why it matters: The "lazy payoff" story judges reward

### Secondary Metrics
- End-to-end run finishes without errors on the sample dataset: yes
- Overnight run time on the builder's laptop: recorded, target under 30 minutes

## UI/UX Direction

**Design Feel:** Calm, clean, a little playful (sloth theme)
**Inspiration:** TBD

### Key Screens
1. **Upload**
   - Purpose: Load history and brief
   - Key Elements: CSV upload, "Use sample data" button, column check summary, brief text box, Start button
   - User Actions: Upload, review the column check, type the brief, start
2. **Backtest**
   - Purpose: Show whether predictions can be trusted
   - Key Elements: TabPFN vs baseline accuracy, test-post count, plain-language verdict
   - User Actions: Read, continue
3. **Week calendar**
   - Purpose: The morning result
   - Key Elements: 7 cards by day, colour by predicted label, confidence, reason, rewrite marker
   - User Actions: Expand, edit, copy, download CSV

### Design Principles
- **Honest numbers:** Show baseline next to model; never hide a weak result
- **One screen at a time:** No settings maze; sensible defaults
- **Colour is not the only signal:** Labels are written in words as well as colours

## Technical Considerations

**Platform:** Local web app opened in the browser on the user's laptop
**Responsive:** Desktop-first
**Performance Goals:**
- UI responds within 2 seconds on load
- Full overnight run under 30 minutes on the builder's laptop (to be measured)

**Security/Privacy:** No accounts. Uploaded CSVs, fetched DEV posts (`data/dev_posts.csv`) and results stay on the laptop in a git-ignored folder. DEV posts are fetched from a public API; author names and usernames are not stored. TabPFN runs in local mode only (not the Prior Labs cloud API). No telemetry. The article shows only overall results and anonymised examples.
**Scalability:** Not applicable; single user, single machine.

## AI / Automation Scope

**AI Surface:** Overnight automation inside a local app
**Models:** Gemma (open-weight) via Ollama for drafting and rewriting; TabPFN (open-weight, run locally under its non-commercial licence) for prediction
**Allowed Data:** Past-post fields and the user's brief, on the local machine only
**Provider / Retention:** No cloud provider; nothing leaves the laptop. Verify TabPFN is configured for local inference.
**Output Contract:** Gemma returns JSON matching a fixed draft schema (post type, day, time, caption, hashtags, visual idea); validated by code
**Confirmation Rules:** The tool never posts or sends anything; the user copies drafts out manually
**Verification Prompts:** Normal brief; empty brief; brief containing instructions to ignore rules; CSV with missing columns; CSV with too few rows; Ollama not running; TabPFN weights missing

## Constraints & Requirements

### Budget
- Development tools: $0/month
- Hosting/Infrastructure: $0/month (runs locally)
- Third-party services: $0/month
- **Total:** $0/month

### Timeline
- MVP Development: 23–24 Oct 2026
- Testing and real use: 24 Oct 2026
- Article and submission: 25 Oct 2026; deadline 25 Oct, 11:59 PM PDT

### Technical Constraints
- Must work offline after models are downloaded
- Must run on the builder's laptop (specs TBD)
- The repository must be new and created within the challenge window; post-deadline commits must be noted in the README
- Build follows the vibe-coding-prompt-template workflow with Claude Code

## Open Questions & Assumptions
- Resolved: Column names of `Instagram_Analytics.csv` confirmed from the header row (see Tech Design).
- Resolved: The Kaggle file has 20 accounts and behaves like noise; the sample is account 15 (decided by the user).
- Resolved: Shandrika's real export is not coming; DEV posts are the real-data demo.
- Open: DEV label rule (most posts get 0 reactions); decided in the M1 plan after checking the real distribution.
- Resolved: Laptop has 16 GB RAM and an integrated GPU (CPU-only); Gemma `gemma3:4b` passed the M0 spike.
- Assumption (Instagram profile): Engagement = (likes + comments + shares + saves) / reach, labelled by thirds of the training history. Other profiles define their own outcome and label rule.
- Assumption: About 150+ posts give TabPFN enough history; to be confirmed by the backtest.

## Quality Standards

**Code Quality:**
- Type hints in Python; validate all CSV input and model output
- Handle errors explicitly with a plain message to the user
- Tests for column mapping, feature building, time-ordered split and draft validation

**Design Quality:**
- Consistent colours and spacing
- Readable contrast; labels in words, not colour alone

**What This Project Will NOT Accept:**
- Invented results or metrics in the UI or article
- Features that half-work: complete or cut
- Real account data committed to the repository

## Risk Mitigation

| Risk | Impact | Mitigation Strategy |
|------|--------|-------------------|
| TabPFN does not beat the baseline | High | Report honestly; try fewer, clearer features; still ship |
| TabPFN install or licence step blocks | High | Spike it first thing; fall back to a classical model only as a labelled baseline |
| Gemma too slow or weak on the laptop | Medium | Smaller Gemma size; overnight run absorbs slowness |
| DEV posts are barely predictable (most get 0 reactions) | Medium | Pick a balanced-enough label; report balanced accuracy and ROC AUC honestly; the no-signal verdict is a feature, not a failure |
| Running out of time | High | Must-haves only; nice-to-haves after the core works |

## MVP Completion Checklist

### Development Complete
- [ ] All five core features working
- [ ] Error handling for bad CSV, missing models, malformed drafts
- [ ] Tested in Chrome on the builder's laptop

### Launch Ready
- [ ] README with setup, usage, data privacy note and dataset credit
- [ ] Demo video or GIF recorded
- [ ] DEV article with every template section filled

### Quality Checks
- [ ] Core journey works end to end on sample data
- [ ] Backtest result recorded honestly
- [ ] No real account data in the repository
- [ ] No critical bugs

## Next Steps

1. **Immediate:** Review and approve this PRD
2. **Next:** Create Technical Design Document (Part 3)
3. **Then:** Set up agent files and repository (Part 4)
4. **Build:** Implement with Claude Code in small verified steps
5. **Test:** Real weekly run and timing
6. **Launch:** Publish the DEV submission

---
*Created: 23 Oct 2026; updated 23 Oct 2026 (decision change: DEV posts, source profiles)*
*Status: Ready for Technical Design*
*Questions? Javeria Lodhi*

---
## Handoff Context
<!-- Machine-readable summary for the next workflow step. Do not delete; the next prompt in the workflow reads this block. -->
- Stage: prd
- App name: Slow Post
- User level: C  (A = vibe coder, B = developer, C = in-between)
- Target platform: Local web app on the user's laptop (desktop browser)
- Budget: $0
- Timeline: Submit by 25 Oct 2026, 11:59 PM PDT
- AI in product scope: yes
- Research status: Completed (challenge, entries, winners, workflow); see research doc "HF26 Week 3 — Research & Winning Strategy"
- Constraints: Offline after model download; data never leaves the laptop; TabPFN local mode only; new repo in challenge window; text-only drafts; no auto-posting
- Decisions: Target Best Use of TabPFN (primary), Gemma (secondary); Gemma via Ollama for drafting; TabPFN for prediction; time-ordered backtest vs most-common-label baseline; one source profile per data source; demo on real DEV posts from DEV's public API; Kaggle "Instagram Analytics Dataset" (CC BY 4.0) kept as the no-signal test case; Shandrika's real export is not coming
- Open questions: DEV label rule (decided at M1); which Gemma output format DEV drafts use (decided at M2)
- Source files: research doc (HF26 Week 3 — Research & Winning Strategy) → docs/PRD-SlowPost-MVP.md
---

```json
{
  "schemaVersion": 1,
  "documentType": "prd",
  "appName": "Slow Post",
  "oneLiner": "Drafts next week's posts overnight on your laptop and predicts which ones will underperform, from the account's own history.",
  "targetUsers": "Anyone who posts regularly and plans posts weekly: developers on DEV, social media managers, small business owners, creators",
  "phase": "Foundation",
  "mustHave": ["Load past posts", "Backtest", "Draft next week's posts", "Score and rewrite", "Calendar and export"],
  "niceToHave": ["Overnight scheduling", "Best-time-slot suggestion", "Run log with timings"],
  "notInMvp": ["Auto-posting", "Logging in to Instagram / Meta", "Image or video generation", "Multiple accounts", "Sources with no profile yet", "Hosted web version", "Fine-tuning"],
  "successMetrics": ["Backtest accuracy above baseline", "Real weekly plan completed and timed"]
}
```
