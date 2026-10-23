"""Fetch public DEV posts. Pure helpers here; the network loop is `fetch_tag`.

Only these fields are kept. The author (`user`) object is never read, so no names or
usernames can reach the file.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta

API = "https://dev.to/api/articles"
HEADERS = {
    "Accept": "application/vnd.forem.api-v1+json",
    "User-Agent": "slowpost-local-research/0.1 (personal project)",
}
FIELDS = ["id", "published_at", "title", "tag_list", "reading_time_minutes",
          "has_cover_image", "public_reactions_count", "comments_count"]
MIN_AGE_DAYS = 14  # reactions have mostly settled by then


def article_to_row(article: dict) -> dict:
    """One API record -> one CSV row with only the allowed fields."""
    tags = article.get("tag_list")
    if isinstance(tags, str):
        tags = [t for t in tags.split(",")]
    tags = [str(t).strip() for t in (tags or []) if str(t).strip()]
    return {
        "id": article["id"],
        "published_at": article["published_at"],
        "title": article.get("title", ""),
        "tag_list": ",".join(tags),
        "reading_time_minutes": article.get("reading_time_minutes"),
        "has_cover_image": bool(article.get("cover_image")),
        "public_reactions_count": article.get("public_reactions_count", 0),
        "comments_count": article.get("comments_count", 0),
    }


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def in_window(row: dict, now: datetime, days: int) -> bool:
    """At least MIN_AGE_DAYS old, and no older than MIN_AGE_DAYS + days."""
    age = now - parse_time(row["published_at"])
    return timedelta(days=MIN_AGE_DAYS) <= age <= timedelta(days=MIN_AGE_DAYS + days)


def get_page(tag: str, page: int, per_page: int = 1000, retries: int = 4) -> list[dict]:
    url = f"{API}?tag={tag}&per_page={per_page}&page={page}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
        except OSError:  # connection reset, timeout, DNS: all worth a retry
            if attempt == retries - 1:
                raise
        time.sleep(5 * (attempt + 1))
    return []


def fetch_tag(tag: str, now: datetime, days: int, max_pages: int, wait: float = 1.0,
              log=print) -> list[dict]:
    """Page backwards in time until the window is passed, the pages run out, or the cap hits."""
    rows: list[dict] = []
    window_start = now - timedelta(days=MIN_AGE_DAYS + days)
    for page in range(1, max_pages + 1):
        articles = get_page(tag, page)
        time.sleep(wait)
        if not articles:
            break
        batch = [article_to_row(a) for a in articles]
        kept = [r for r in batch if in_window(r, now, days)]
        rows.extend(kept)
        oldest = min(parse_time(r["published_at"]) for r in batch)
        log(f"  {tag} page {page}: {len(batch)} posts, {len(kept)} in window, "
            f"oldest {oldest:%Y-%m-%d}")
        if oldest < window_start:
            break
    return rows


def utc_now() -> datetime:
    return datetime.now(UTC)
