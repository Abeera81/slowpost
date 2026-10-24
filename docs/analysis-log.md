# Analysis log: every choice, and when it was made

This is the decision log for Slow Post's analysis. It is written from `MEMORY.md`, the git history
and the project documents, and it separates three kinds of choice:

- **Fixed in advance**: decided before any result for that choice could be seen.
- **Chosen from the label distribution**: decided after looking at how common each outcome is
  (not at how well any model did).
- **Changed after seeing results**: decided or revised after a model result was on the screen.
  These are the ones to read most sceptically. Where I could only partly tell why, the entry says
  so instead of guessing.

Commit ids are the places to look in `git log`. "User" means Javeria's instruction in the
conversation; the project notes record the decision but not always the reason.

## 1. Outcome and labels

| Choice | Value | Status | Why / where |
|---|---|---|---|
| DEV outcome | `public_reactions_count` (reactions) | Fixed in advance | Tech Design (`e284f49`). |
| DEV label | "gets any reactions": reactions >= 1, yes or no | **Chosen from the label distribution** | A read-only API probe (23 Oct) showed 85-87% of posts at least 14 days old have 0 reactions, so a count-based or thirds label would be mostly ties at 0. The user approved "any reactions" at the M1 plan. Not tuned on model results. |
| DEV post age | only posts at least 14 days old | Fixed in advance | So reactions have mostly settled (user, M1 plan). |
| DEV tags and window | webdev, python, javascript, beginners; `--days 60` | Fixed in advance | User, M1 plan. Result: 44,428 posts, 10 Aug to 9 Oct 2026, 15.6% with reactions. |
| Instagram outcome | engagement computed as (likes + comments + shares + saves) / reach, never the file's `engagement_rate` | Fixed in advance | Tech Design: the file's own rate is a derived column that may leak. |
| Instagram label | thirds (below / around / above), thresholds learned from training rows only | Fixed in advance | Tech Design (`e284f49`). |

## 2. Features

| Choice | Value | Status | Why / where |
|---|---|---|---|
| DEV features | weekday, hour (UTC), reading time, number of tags, main tag, has cover image, title length, title has a question, title has a number | Fixed in advance | Everything known before posting that the public API provides (M1 plan, `b36b825`). No feature was added or dropped after seeing results. |
| Main-tag categories | the 10 most common tags in training rows, others "other" | **Chosen from the feature distribution** | Keeps the category count small for one-hot encoding. |
| Instagram features | media_type, content_category, day_of_week, post_hour, caption_length, hashtags_count, has_call_to_action | Fixed in advance | Tech Design. `impressions`, `engagement_rate`, `followers_gained`, `traffic_source`, `follower_count`, `account_type`, `performance_bucket_label` are excluded as leakage or constants. |
| Draft features | computed by code from the draft; `has_cover_image` = 1 when the cover idea is non-empty (so always 1 for Gemma drafts) | Fixed in advance (user, M2 plan) | Gemma never supplies a feature value. |
| What-if features | the one-feature changes in `score.py` | Fixed in advance | Cover, hour, weekday, main tag, reading time (+/-3 min), title question, title number. |

## 3. Test window, training sample and seeds

| Choice | Value | Status | Why / where |
|---|---|---|---|
| First split | the newest 20% of posts as the test window, sampled to at most 1,500 rows, scored with the default most-likely class | Fixed in advance | Tech Design (`e284f49`), first code in `b36b825`. |
| Training sample | a random sample (seed 0) of up to 1,000 posts from **all** posts older than the test window | Decided at M1 approval, before any result (user) | Replaces "the 1,000 just before the test window", so every weekday and hour is covered. The 1,000-row cap is the CPU limit found in the M0 spike. |
| Test window | a random sample (seed 0) of up to 2,000 posts from the **newest 7 days**; if the newest week has under 100 posts, the newest 20% | **Changed after seeing results** (`ea7473c`, user) | Replaced the 80/20 split after the first M1 backtest. The reason is not recorded in the notes. Plausible, unconfirmed reasons: the newest week matches the planning horizon, and predicting thousands of posts is slow on CPU (about 160 s for 1,500 rows). |
| No leakage | nothing in training is newer than anything in test; the cut is on the timestamp | Fixed in advance | Unit-tested. |
| Seed | 0 everywhere | Fixed in advance | Never varied. |
| Known weakness | one test window only. The test week has more posts with reactions (21.6%) than the training sample (14.1%) | Not a choice, a result | Shown in the app next to the numbers. |

## 4. Prediction rule and verdict

| Choice | Value | Status | Why / where |
|---|---|---|---|
| Binary prediction cutoff | "yes" when P(yes) is above the training share of "yes" (0.141 for DEV) | **Changed after seeing results** (`ea7473c`, user) | The first M1 backtest labelled posts with the model's default most-likely class, which is poor on an 84%/16% outcome. The cutoff moves the decision point to the base rate. It uses training rows only. |
| Multi-class prediction | the class with the highest probability relative to its training share | Same change | Used for the Instagram thirds. |
| Reported measures | balanced accuracy at that cutoff, ROC AUC, both with 95% bootstrap intervals (1,000 resamples), next to the most-common-label baseline | Fixed in advance (Tech Design); raw accuracy kept as reference only | "Always no reactions" scores 78% raw accuracy on DEV. |
| Verdict rule | "Better" needs both the balanced-accuracy-gap interval and the AUC interval fully above chance; "Some signal" if only one; "Worse" if the gap interval is below 0; otherwise "No better" | Written in `ea7473c`, after the first backtest | The rule is symmetric and was not tuned to produce a particular verdict, but it was written after a first result had been seen. |
| Lift in the top 10% and 20% | share of the model's highest-ranked test posts that got reactions, divided by the overall share | **Added after seeing results** (user, M2 plan) | An extra measure that is easier to read than AUC. It did not replace the others. The headline "1.67x" is the top-10% number; the top-20% number is 1.38x. Choosing the more flattering of two cut-offs would be a forking path, so both are always shown with intervals. |

## 5. Labels in the plan and the evidence next to them

| Choice | Value | Status | Why / where |
|---|---|---|---|
| Strong | P(yes) at or above the model's top-20% threshold in the backtest test window | Defined by the user at the M3 plan, after seeing the lift | The threshold comes from the same test posts that the band's measured rate is reported on, so the Strong group's 29.8% is somewhat optimistic. |
| Weak | P(yes) below the training cutoff | Same | |
| Middle band | "Too close to call" (was "Okay") | **Renamed after seeing results** (B5) | "Okay" sounded like a verdict. The middle band's measured rate (21.1%) is about the overall rate (21.6%), so the model cannot tell these posts from an average one. |
| What-if "backed by the data" | a feature whose signal-summary group has a clear difference after correction | **Changed after seeing results** (M3 fix, `5e603a7`) | The first real run's rewrite leaned on "add a question mark", which the model liked but the data did not support (corrected p = 0.325). Only data-backed changes may be reasons or hints. |
| Minimum effect for a what-if | 0.01 (one percentage point) | Fixed in advance, a rule of thumb | Smaller changes are called "no clear difference". Not estimated from data. |
| What-if wording | "In the data, posts like this with X got reactions more often (a% vs b%). The model scores this variant higher." | **Changed after seeing results** (B5, user) | The earlier wording ("would raise") implied causation. |

## 6. Signal summary and controlled checks

| Choice | Value | Status |
|---|---|---|
| Signal test | chi-square per feature group, Bonferroni-corrected across the features, alpha 0.05, groups under 20 posts left out | Fixed in advance (M1) |
| Group bins | weekday; hour in 4-hour blocks; reading time 0-2, 3-4, 5-7, 8+ min; tags 0-3, 4+; title-length quartiles | **Chosen from the feature distribution** |
| Cover vs reading time | the cover gap inside each reading-time band; the reading-time gap (at or above the median reading time, 4 min, vs shorter) with and without a cover; Mantel-Haenszel pooled odds ratio next to the crude one | Fixed in advance (B5), written before the run. The median split point is chosen from the distribution. Result shown whichever way it falls. |
| Calibration | 8 equal-count bins of the test posts, Wilson 95% intervals | Fixed in advance (B5) |

## 7. Model comparison (B2)

| Choice | Value | Status |
|---|---|---|
| Comparison models | logistic regression (standard scaling for numbers, one-hot for the categorical main tag, `max_iter=1000`) and `HistGradientBoostingClassifier` at scikit-learn defaults, seed 0 | Fixed in advance. No tuning, no cross-validation, so the simple models are not given an advantage by selection. |
| Same data | identical training sample and identical test posts | Fixed in advance |
| Is TabPFN better? | paired bootstrap of the ROC AUC difference on the same test posts (1,000 resamples). "Clearly better" needs the interval above 0 | Fixed in advance. The sentence template was **rewritten after seeing results**: the first draft lumped both baselines together, but TabPFN clearly beats gradient boosting and not logistic regression. No number changed. |
| Random 80/20 split | TabPFN on a random 80/20 split of all posts, same size caps | Fixed in advance. Shown for comparison only, because it lets the model learn from the same weeks it is tested on. |
| Result | TabPFN top-10% lift 1.67x, AUC 0.588; logistic regression AUC 0.573 and the higher balanced accuracy (0.555 vs 0.544); gradient boosting AUC 0.529. TabPFN minus logistic regression: AUC +0.014 (paired 95% interval -0.007 to +0.035). | Result, reported as found: **TabPFN does not clearly beat a simple logistic regression on this data.** |

## 8. Instagram sample: account and pooling

| Choice | Value | Status | Why / where |
|---|---|---|---|
| Exploration | `scripts/signal_check.py` pooled all 20 accounts, plus each account alone | Exploratory, before the app existed | Pooled: no feature moves engagement (every p above 0.12). Pooled time-ordered backtest 0.341 vs 0.317 (+0.023, about 1.9 standard errors). Account 15 alone had p below 0.05 for media type (0.033), call to action (0.009) and hashtags (0.004), judged likely chance (one of 20 accounts x 6 features). |
| Account for the sample | `account_id` 15 (tie with 13 at 1,599 posts) | Fixed in advance (user), before any backtest on it | The sample is committed under CC BY 4.0. |
| What the app reports | account 15 alone, never the pooled number | Fixed in advance | The pooled result and the per-account p-values above were never used to select the sample. |
| Instagram test window | the newest week has too few posts, so the newest 20% (320 posts) | Rule from section 3 | Result: balanced accuracy 0.379 vs 0.333 (gap interval -0.001 to +0.095), AUC 0.526 (0.479-0.573): no better than the baseline. |

## 9. Drafting, revision and the verdict (Gemma side)

| Choice | Value | Status |
|---|---|---|
| Model | `gemma3:4b` via Ollama | Chosen in the M0 spike (it ran; `gemma4:e2b` was not tried) |
| Day and hour | chosen by code from the history's best days and hours, never by Gemma | Fixed in advance |
| Tags | limited to the history's 200 most common tags, every change noted | **Changed after seeing results** (M3): the first demo's tags were mostly invented words |
| Near-duplicate titles | up to two regenerations, the second with a different angle, then a flag | **Changed after seeing results** (M3, M5): drafts 4 and 5 of the first demo had the same title |
| Revision pass | every draft that is not Strong gets one revision; the revision is kept only if TabPFN scores it strictly higher | Fixed in advance (B3). The ranking and the labels always come from TabPFN. |
| Run verdict | no Strong draft: "Nothing stands out this week. Go back to sleep."; otherwise "N strong ideas this week." | Fixed in advance (B1) |

## 10. Things that could still mislead

- **One test week.** Everything depends on one 7-day window with a higher share of reactions
  than the training sample. A different week could give different numbers.
- **The Strong threshold is set on the test posts it is then measured on**, so the Strong group's
  rate is somewhat optimistic.
- **Calibration**: the model's percentages run below what really happened (it predicted 16% on
  average, 22% had reactions). Read them as a ranking, not as odds.
- **Reading time and the cover flag are current values**: authors can edit them after publishing.
- **Associations, not causes.** Cover image and reading time both stay associated with reactions
  after holding the other fixed, but nothing here shows that adding a cover causes reactions.
- **Fewer than 2,000 test posts, one seed.** Intervals reflect resampling of the test posts, not
  a different training sample.
- **Revision gains are the model's own opinion.** A revision is kept when TabPFN scores it higher
  than the original, so the choice is selected by the same model that then reports the gain
  (a winner's curse). The gain has not been tested on real posts.
- **Strong is a threshold, not a cliff.** A draft at 20.3% is Strong and one at 19.9% is not; the
  threshold is 0.1997. Treat drafts near it as the same.
- **Slot matters as much as wording.** Day and hour are chosen by code from the history's best
  days and hours, so the first draft gets the best slot. In the real runs the Strong drafts tended
  to be the ones in the best slots, even for a brief meant to be dull.
