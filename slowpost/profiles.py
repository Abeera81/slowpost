"""Source profiles: everything that differs between data sources lives here.

A profile says how to read a source (column mapping), what to predict (outcome and label rule),
which features to use, and what a draft looks like. The rest of the code takes a profile and
has no source-specific branches.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Profile:
    name: str
    title: str
    column_map: dict[str, str]  # source column -> internal column
    time_col: str  # internal name of the post time column
    label_rule: str  # "thirds" (training percentiles) or "min_value" (fixed cutoff)
    label_names: tuple[str, ...]  # ordered low -> high
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    draft_fields: tuple[str, ...]  # what a drafted post contains (used from M2)
    min_value: float | None = None  # for "min_value": outcome >= this is the high label
    top_categories: dict[str, int] = field(default_factory=dict)  # keep N most common in training

    @property
    def required_columns(self) -> list[str]:
        return list(self.column_map)

    @property
    def features(self) -> list[str]:
        return [*self.numeric_features, *self.categorical_features]


# Kaggle "Instagram Analytics Dataset": the "no signal" test case.
# Engagement = (likes + comments + shares + saves) / reach, computed in data.py.
INSTAGRAM = Profile(
    name="instagram_kaggle",
    title="Instagram (Kaggle sample)",
    column_map={
        "post_datetime": "post_datetime", "media_type": "media_type",
        "content_category": "content_category", "day_of_week": "day_of_week",
        "post_hour": "post_hour", "caption_length": "caption_length",
        "hashtags_count": "hashtags_count", "has_call_to_action": "has_call_to_action",
        "likes": "likes", "comments": "comments", "shares": "shares", "saves": "saves",
        "reach": "reach",
    },
    time_col="post_datetime",
    label_rule="thirds",
    label_names=("below", "around", "above"),
    numeric_features=("post_hour", "caption_length", "hashtags_count", "has_call_to_action"),
    categorical_features=("media_type", "content_category", "day_of_week"),
    draft_fields=("day", "hour", "media_type", "content_category", "caption", "hashtags",
                  "visual_idea"),
)

# Real DEV posts from DEV's public API. Label: did the post get any reactions?
DEV = Profile(
    name="dev_posts",
    title="DEV posts",
    column_map={
        "id": "id", "published_at": "post_datetime", "title": "title", "tag_list": "tag_list",
        "reading_time_minutes": "reading_time_minutes", "has_cover_image": "has_cover_image",
        "public_reactions_count": "reactions", "comments_count": "comments",
    },
    time_col="post_datetime",
    label_rule="min_value",
    min_value=1,
    label_names=("no_reactions", "some_reactions"),
    numeric_features=("weekday", "hour", "reading_time_minutes", "n_tags", "has_cover_image",
                      "title_length", "title_has_question", "title_has_number"),
    categorical_features=("main_tag",),
    top_categories={"main_tag": 10},
    draft_fields=("title", "tags", "cover_image", "day", "hour"),
)

# Room for a Meta (Facebook/Instagram) export: add a Profile here and register it below.
PROFILES = {p.name: p for p in (INSTAGRAM, DEV)}
