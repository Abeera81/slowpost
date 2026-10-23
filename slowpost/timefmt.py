"""Show a UTC slot in UTC and in this computer's local time zone. Data stays in UTC."""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from .drafts import DAYS


def slot_display(day: str, hour: int, today: date | None = None) -> dict:
    """The next occurrence of `day` at `hour` UTC, as text in UTC and in local time
    (the computer's time zone, including daylight saving on that date)."""
    today = today or date.today()
    ahead = (DAYS.index(day) - today.weekday()) % 7
    utc = datetime(today.year, today.month, today.day, hour, tzinfo=UTC) + timedelta(days=ahead)
    local = utc.astimezone()
    return {
        "utc": f"{day} {hour:02d}:00 UTC",
        "local": f"{DAYS[local.weekday()]} {local:%H:%M} {local.tzname()}",
        "date_utc": utc.date().isoformat(),
    }
