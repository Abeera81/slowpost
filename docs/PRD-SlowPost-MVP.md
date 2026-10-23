# Product Requirements Document: Slow Post MVP

## Overview

**Product Name:** Slow Post
**Problem Statement:** Planning a week of Instagram posts takes an evening of writing, and even then the person guesses which posts will do well. The account's own past results already show what works, but nobody has time to read them before every post.
**MVP Goal:** A tool that runs on one laptop, learns from an account's past posts, drafts next week's 7 posts overnight, and marks which drafts are likely to underperform, with a backtest showing its predictions beat a simple baseline.
**Target Launch:** Submitted to the DEV Hacktoberfest Week 3 "Slow and Steady" challenge by 25 Oct 2026, 11:59 PM PDT (26 Oct, 11:59 AM PKT).

## Target Users

### Primary User Profile
**Who:** One person who runs an Instagram account and plans posts weekly: a social media manager, a small business owner, or a creator.
**Problem:** Writing a week of posts takes hours, and there is no quick way to know which drafts will flop before publishing.
**Current Solution:** Writing captions by hand or with a chatbot, then guessing; checking Insights after the fact.
**Why They'll Switch:** Slow Post does the drafting overnight and predicts each draft's performance from that account's own history, on their own laptop, for free, without sending the account's data anywhere.

### User Persona: Javeria (builder and first user)
- **Demographics:** 20s, Pakistan, social media manager and software engineering student
- **Tech Level:** Intermediate
- **Goals:** Turn a past-posts export into a ready weekly plan with minimal effort
- **Frustrations:** Sunday-evening planning; guessing which post types and times work

## User Journey

### The Story
1. In the evening, Javeria exports the account's past posts as a CSV (or uses the sample dataset) and drops it into Slow Post.
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
- **Description:** Accepts a CSV of past posts (the public Kaggle Instagram Analytics dataset now, a Meta Business Suite export later), maps its columns to a standard set and reports what it found.
- **User Value:** Works with the file the user already has; no account login.
- **Success Criteria:**
  - User can upload a CSV or click "Use sample data"
  - System shows row count, date range, and which needed columns were found or missing
  - Rows missing an engagement value are skipped and counted, not silently dropped
  - Uploaded data stays on the local machine and is never committed to git
- **Priority:** Critical

#### 2. Backtest
- **Description:** Labels each past post as below, around or above the account's typical engagement; hides the most recent posts in time order, predicts them with TabPFN using only older posts, and compares with a simple baseline (always guessing the most common label).
- **User Value:** Shows honestly whether the predictions can be trusted before using them.
- **Success Criteria:**
  - Split is by date: the model never sees posts newer than the ones it predicts
  - Screen shows TabPFN accuracy, baseline accuracy and number of test posts
  - If TabPFN does not beat the baseline, the screen says so plainly
  - Result is saved so the article can quote it
- **Priority:** Critical

#### 3. Draft next week's posts
- **Description:** A local Gemma model (via Ollama) writes 7 text-only drafts from the brief and the account's best past captions.
- **User Value:** Removes the blank-page work.
- **Success Criteria:**
  - Each draft has: post type (photo, carousel or reel), suggested day and time, caption with call-to-action, hashtags, and a visual idea described in words
  - Output is checked against a fixed structure; a malformed draft is retried once, then reported, never shown half-filled
  - Works with the Wi-Fi off after models are downloaded
- **Priority:** Critical

#### 4. Score and rewrite
- **Description:** Code turns each draft into the same features used for past posts (type, weekday, hour, caption length, hashtag count, call-to-action yes/no, etc.); TabPFN predicts its label and confidence. Drafts predicted "below" are rewritten once by Gemma with a hint and scored again.
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
| Other platforms (Facebook, LinkedIn) | Different columns | Version 2 |
| Hosted web version | Would send data off the laptop | Not planned |
| Fine-tuning any model | Out of time | Not planned |

## Success Metrics

### Primary Metrics
1. **Backtest accuracy above baseline:** TabPFN beats the always-most-common-label baseline on the held-out posts, by 26 Oct 2026
   - How to measure: Backtest screen (Feature 2)
   - Why it matters: Proves the predictions add value; the core judging evidence
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

**Security/Privacy:** No accounts. Uploaded CSVs and results stay on the laptop in a git-ignored folder. TabPFN runs in local mode only (not the Prior Labs cloud API). No telemetry. The article shows only overall results and anonymised examples.
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
- Open: Exact column names in `Instagram_Analytics.csv`; confirm from the header row before building Feature 1.
- Open: The Kaggle dataset covers many accounts and may be simulated; decide whether to filter to one account or sample a slice.
- Open: Will Shandrika's real export arrive in time, and is her account Professional?
- Open: Laptop RAM/GPU, which decides the Gemma model size.
- Assumption: Engagement = (likes + comments + shares + saves) ÷ reach, labelled by thirds of the account's history.
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
| Real export does not arrive | Medium | Build and submit on the public dataset; say so in the article |
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
*Created: 23 Oct 2026*
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
- Decisions: Target Best Use of TabPFN (primary), Gemma (secondary); Gemma via Ollama for drafting; TabPFN for prediction; time-ordered backtest vs most-common-label baseline; build first on Kaggle "Instagram Analytics Dataset" (CC BY 4.0), swap in a real export if it arrives
- Open questions: Kaggle CSV column names; one-account filter vs sample; real export timing; laptop specs
- Source files: research doc (HF26 Week 3 — Research & Winning Strategy) → docs/PRD-SlowPost-MVP.md
---

```json
{
  "schemaVersion": 1,
  "documentType": "prd",
  "appName": "Slow Post",
  "oneLiner": "Drafts next week's Instagram posts overnight on your laptop and predicts which ones will underperform, from the account's own history.",
  "targetUsers": "One person who plans an Instagram account's posts weekly: social media managers, small business owners, creators",
  "phase": "Foundation",
  "mustHave": ["Load past posts", "Backtest", "Draft next week's posts", "Score and rewrite", "Calendar and export"],
  "niceToHave": ["Overnight scheduling", "Best-time-slot suggestion", "Run log with timings"],
  "notInMvp": ["Auto-posting to Instagram", "Logging in to Instagram / Meta", "Image or video generation", "Multiple accounts", "Other platforms", "Hosted web version", "Fine-tuning"],
  "successMetrics": ["Backtest accuracy above baseline", "Real weekly plan completed and timed"]
}
```
