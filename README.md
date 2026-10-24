# Slow Post

Slow Post learns from a post history, drafts next week's posts overnight on your laptop with Gemma, and flags the drafts that are likely to underperform with TabPFN. It reports honestly how well that works: every prediction is backed by a time-ordered backtest against a "most common outcome" baseline, and if a history holds no signal, the app says so.

Everything runs locally: no cloud AI, no telemetry. Built for the DEV Hacktoberfest challenge. The demo source is real DEV posts; a public Instagram sample is the "no signal" test case.

![The week's plan: Strong, Too close to call and Weak drafts with the measured lift](docs/screenshots/09-plan-labels.jpg)

Walkthrough (Data, Backtest, Brief, Plan, Download): `docs/slowpost-walkthrough.gif`. More screenshots are in `docs/screenshots/`.

## Headline results

All numbers come from a time-ordered backtest: train on a random sample of up to 1,000 older posts, test on up to 2,000 posts sampled from the newest 7 days. Intervals are 95% bootstrap intervals.

**DEV posts** (44,428 public posts from 10 Aug to 9 Oct 2026, label: "gets any reactions"; 15.6% do):

| | Result |
|---|---|
| Balanced accuracy | 0.544 vs 0.500 for the baseline (gap +0.044, CI +0.019 to +0.070) |
| ROC AUC | 0.588 (CI 0.556 to 0.620; 0.5 is chance) |
| Lift, model's top 10% (200 test posts) | 36.0% got reactions vs 21.6% overall = **1.67x** (CI 1.37 to 1.97) |
| Lift, model's top 20% (400 test posts) | 29.8% got reactions vs 21.6% overall = 1.38x (CI 1.18 to 1.56) |
| Verdict | Better than the baseline. A weak but real signal: use it to rank ideas, not to promise results. |

Raw accuracy is not a good measure here: "always no reactions" would score 78%, so the app reports balanced accuracy, ROC AUC and lift instead.

### Is TabPFN better than a simple model?

Same training sample, same 2,000 test posts, same 95% bootstrap intervals (the app shows this table on the Backtest page):

| Model | Balanced accuracy | ROC AUC | Lift, top 10% | Lift, top 20% |
|---|---|---|---|---|
| Most common outcome | 0.500 | 0.500 | 1.00x | 1.00x |
| Logistic regression | 0.555 (0.530 to 0.579) | 0.573 (0.546 to 0.602) | 1.53x (1.26x to 1.82x) | 1.24x (1.09x to 1.46x) |
| Gradient boosting | 0.523 (0.498 to 0.549) | 0.529 (0.497 to 0.560) | 1.35x (1.10x to 1.70x) | 1.22x (1.03x to 1.38x) |
| **TabPFN** | 0.544 (0.519 to 0.570) | 0.588 (0.556 to 0.620) | 1.67x (1.37x to 1.97x) | 1.38x (1.18x to 1.56x) |

TabPFN ranks posts clearly better than gradient boosting (AUC +0.059, paired 95% interval +0.034 to +0.084); it is not clearly different from logistic regression (AUC +0.014, paired 95% interval -0.007 to +0.035). So TabPFN does not clearly win here: a simple model does about as well. Logistic regression has the higher balanced accuracy (0.555 vs 0.544).

TabPFN on a *random* 80/20 split (comparison only): balanced accuracy 0.610 (0.582 to 0.638), ROC AUC 0.647 (0.616 to 0.676), top-10% lift 1.96x (1.61x to 2.30x). The random split looks better because the model gets to learn from posts published in the same weeks it is tested on. The time-ordered split is the honest one: a plan is written before the week starts, so the model has to predict newer posts from older ones.

The calibration chart on the Backtest page shows that the model's percentages run below what really happened (about 16% predicted on average, 22% of test posts got reactions), so read them as a ranking, not as odds. The decisions behind all of this are in [`docs/analysis-log.md`](docs/analysis-log.md).

**Instagram sample** (one account, 1,599 posts, label: thirds of engagement):

| | Result |
|---|---|
| Balanced accuracy | 0.379 vs 0.333 for the baseline (gap +0.045, CI -0.001 to +0.095) |
| ROC AUC | 0.526 (CI 0.479 to 0.573) |
| Signal summary | No feature shows a clear difference after correcting for testing several at once |
| Verdict | **No better than the baseline.** The history has no signal and the app says so. |

## How a run works

1. **Data:** fetch public DEV posts (tags webdev, python, javascript, beginners; posts at least 14 days old).
2. **Backtest:** TabPFN is fitted on older posts and tested on the newest week. The result includes a signal summary: which features (cover image, tags, hour, title, and so on) show a clear difference.
3. **Brief:** write what the posts are about, plus optional facts about your project. Gemma drafts posts (title, up to 4 tags from your history's most common tags, cover idea, outline, reading time, day and hour in UTC and local time). Features are computed by code, never by Gemma.
4. **Plan:** each draft gets Strong, Too close to call or Weak, with the measured test-window numbers next to the labels. The plan leads with a verdict: "N strong ideas this week." or, if nothing is Strong, "Nothing stands out this week. Go back to sleep." (with the best candidate named anyway). A what-if panel separates "Backed by the data" from "The model's guess"; its wording is about what the history shows ("posts like this with X got reactions more often, 20% vs 6%"), not what a change would cause.
5. **Revision:** every draft that is not Strong gets one revision from Gemma, which sees the draft, its score and band, and the backed-by-the-data evidence with observed rates. Code recomputes the features and TabPFN rescores both versions; the revision is kept only if it scores higher. The final labels and ranking always come from TabPFN, never from Gemma. The Plan page shows before and after and how many drafts changed band.
6. **Download:** edit the cards and download `plan.csv` (it repeats the run verdict on every row). Edits are not rescored.

Nothing is ever posted for you.

## Setup (Git Bash on Windows)

Tested on Python 3.14.4 on Windows (Git Bash), with [Ollama](https://ollama.com) and roughly 6 GB of free RAM. It runs on CPU. A run takes about 10 to 15 minutes for 5 drafts. Gemma and TabPFN never run at the same time.

```bash
git clone git@github.com:Abeera81/slowpost.git
cd slowpost
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt

ollama pull gemma3:4b          # one time; keep Ollama running (ollama serve, or the app)
```

TabPFN-3.5 weights are free for non-commercial use. On first run, the `tabpfn` package opens a Prior Labs login page where you accept the licence; your data is never sent to Prior Labs. For non-interactive use, set `TABPFN_TOKEN` to your Prior Labs API key (we load it from a git-ignored `.env`). See <https://docs.priorlabs.ai/models/accessing-model-weights>.

### Run

```bash
source .venv/Scripts/activate
streamlit run app.py                         # the app: Data, Backtest, Brief, Plan

# or step by step
python scripts/fetch_dev_posts.py            # fetch DEV posts into data/dev_posts.csv
python scripts/run_backtest.py --source dev  # backtest + signal summary (about 3 min)
python scripts/run_backtest.py --source instagram   # the "no signal" sample
python scripts/run_overnight.py --brief "..." --facts "..."   # headless run (about 10 min)
python scripts/draft_demo.py --brief "..."   # drafts only
pytest -m "not slow"                         # tests (slow ones need Ollama)
ruff check .
```

## Overnight mode (Windows Task Scheduler)

The Brief page can schedule one run for later ("Schedule for tonight"). Nothing is created until you tick the box and click "Create scheduled task"; the page shows the exact command first. The brief is saved in `data/scheduled/<task>.json` (git-ignored) so no free text has to be quoted on the command line. For example, for 02:00 on 26 October 2026 (the paths are yours; the app prints the exact command for the time you pick):

```bash
schtasks /Create /SC ONCE /TN SlowPost-20261026-0200 /TR "\"D:\Hackathon\slowpost\.venv\Scripts\python.exe\" \"D:\Hackathon\slowpost\scripts\run_overnight.py\" --job \"D:\Hackathon\slowpost\data\scheduled\SlowPost-20261026-0200.json\"" /ST 02:00 /SD 26/10/2026 /F
```

The date must be written the way Windows shows short dates (for example `26/10/2026` or `10/26/2026`); the app reads the pattern from your regional settings. To remove a task: `schtasks /Delete /TN SlowPost-YYYYMMDD-HHMM /F`. The computer must be on and not asleep, you must be signed in, and Ollama must be running at that time. In the morning the Plan page opens the latest finished run with its verdict ("Good morning"); a scheduled run writes the same files as a run started from the app.

## Data source

Posts come from DEV's public Forem API v1, `GET https://dev.to/api/articles`, at about 1 request per second. Only 8 fields are stored: post id, time, title, tags, reading time, whether there is a cover image, reaction count and comment count. Author names and usernames are never stored. The file lives in `data/` on your computer and is never redistributed. Reading time and the cover flag reflect their current values (authors can edit them after publishing).

## Your data stays on your laptop

- Gemma runs through Ollama on your machine; TabPFN runs with the local `tabpfn` package, never the cloud client. There is no telemetry and no cloud AI.
- The only network calls are fetching public DEV posts and downloading models. After that, the app works offline.
- `data/` (fetched posts, backtests, run outputs) is git-ignored. Only `sample/sample_posts.csv` is committed.
- DEV posts are fetched from the public API, and only these fields are stored: post id, time, title, tags, reading time, cover-image flag, reaction count and comment count. Author names and usernames are never stored. Gemma sees computed summaries, never DEV titles. See "Data source" above.

## Licences

- **Our code is MIT** (see `LICENSE`).
- **TabPFN model weights have their own licence, and they are non-commercial.** Read Prior Labs' current terms before using Slow Post with them for anything commercial. Slow Post does not include or redistribute the weights.
- Gemma is used through Ollama under Google's Gemma terms.
- The Instagram sample is CC BY 4.0 (see credits).

## Credits

- **Instagram Analytics Dataset** by Kundan Sagar Bedmutha, <https://www.kaggle.com/datasets/kundanbedmutha/instagram-analytics-dataset>, licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). `sample/sample_posts.csv` is one account (`account_id` 15, 1,599 posts) with rows unchanged.
- **DEV posts** come from [DEV's public API](https://developers.forem.com/api) (`https://dev.to/api/articles`), fetched at run time and not redistributed.
- **TabPFN** by Prior Labs, and **Gemma** by Google, run through **Ollama**.
- Built with Claude Code, following the [vibe-coding-prompt-template](https://github.com/KhazP/vibe-coding-prompt-template) workflow (product requirements, tech design, agent instructions and milestones).

## Changes after the deadline

_None yet._
