"""Scheduling helpers: no task is ever created by these tests."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from slowpost.schedule import (
    command_text,
    create_args,
    format_date,
    pending,
    task_command,
    task_name,
    write_job,
)

WHEN = datetime(2026, 10, 26, 2, 30)


def test_dates_follow_the_windows_pattern():
    assert format_date(WHEN, "dd/MM/yyyy") == "26/10/2026"
    assert format_date(WHEN, "MM/dd/yyyy") == "10/26/2026"
    assert format_date(datetime(2026, 3, 5), "d/M/yyyy") == "5/3/2026"
    assert format_date(WHEN, "yyyy-MM-dd") == "2026-10-26"


def test_job_file_keeps_the_brief_out_of_the_command_line(tmp_path):
    brief = 'A "quoted" brief with & ampersands, ünïcode and 100% symbols'
    job = write_job(WHEN, brief, "facts", 3, Path("D:/x/dev_posts.csv"), folder=tmp_path)
    assert json.loads(job.read_text(encoding="utf-8"))["brief"] == brief
    args = create_args(WHEN, job, python="C:/py/python.exe", pattern="dd/MM/yyyy")
    assert args[:6] == ["schtasks", "/Create", "/SC", "ONCE", "/TN", task_name(WHEN)]
    assert "/ST" in args and args[args.index("/ST") + 1] == "02:30"
    assert args[args.index("/SD") + 1] == "26/10/2026" and args[-1] == "/F"
    tr = args[args.index("/TR") + 1]
    assert tr == task_command(job, "C:/py/python.exe") and "ampersands" not in tr
    assert "--job" in tr and "run_overnight.py" in tr and len(tr) < 261  # schtasks limit


def test_command_text_is_one_pasteable_line():
    args = create_args(WHEN, Path("D:/x/job.json"), python="C:/py/python.exe", pattern="dd/MM/yyyy")
    text = command_text(args)
    assert text.startswith("schtasks /Create /SC ONCE /TN SlowPost-20261026-0230 /TR ")
    assert "\n" not in text


def test_pending_lists_only_future_jobs(tmp_path):
    write_job(WHEN, "future", "", 5, Path("c.csv"), folder=tmp_path)
    write_job(WHEN - timedelta(days=3), "past", "", 5, Path("c.csv"), folder=tmp_path)
    (tmp_path / "SlowPost-bad.json").write_text("not json", encoding="utf-8")
    jobs = pending(now=WHEN - timedelta(hours=1), folder=tmp_path)
    assert [j["brief"] for j in jobs] == ["future"]
