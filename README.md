# slowpost

A local app that learns from your past posts, drafts next week's posts overnight with Gemma, and flags likely underperformers with TabPFN. Everything runs on your laptop. Work in progress for the DEV Hacktoberfest challenge.

## Data credits

- **Sample file `sample/sample_posts.csv`** is one account (`account_id` 15, 1,599 posts) from the Kaggle "Instagram Analytics Dataset", licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Rows are unchanged; only the one account was selected.
  - Author: **TODO (Javeria): add the dataset author's name from the Kaggle page**
  - Link: **TODO (Javeria): add the Kaggle dataset URL**
- **DEV posts** are fetched at run time from DEV's public API into `data/dev_posts.csv`, which is git-ignored and never committed. Author names and usernames are not stored.
