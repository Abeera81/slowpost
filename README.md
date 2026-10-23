# slowpost

A local app that learns from your past posts, drafts next week's posts overnight with Gemma, and flags likely underperformers with TabPFN. Everything runs on your laptop. Work in progress for the DEV Hacktoberfest challenge.

## Data credits

- **Sample file `sample/sample_posts.csv`** is one account (`account_id` 15, 1,599 posts) from the "Instagram Analytics Dataset" by Kundan Sagar Bedmutha, <https://www.kaggle.com/datasets/kundanbedmutha/instagram-analytics-dataset>, licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Rows are unchanged; only the one account was selected.
- **DEV posts**: the demo source. Public post data comes from [DEV's public API](https://developers.forem.com/api) (`https://dev.to/api/articles`). It is fetched at run time into `data/dev_posts.csv`, which is git-ignored and never committed. Only post id, time, title, tags, reading time, cover-image flag, and reaction and comment counts are kept. Author names and usernames are not stored.
