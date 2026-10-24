"""Overnight mode: schedule one run with Windows Task Scheduler (schtasks).

Nothing is created unless the user clicks the button in the app (or runs the printed command).
The brief goes in a small JSON job file under data/scheduled/, so no free text has to be
quoted on the command line. The task runs scripts/run_overnight.py --job <file> once.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEDULED = ROOT / "data" / "scheduled"
SCRIPT = ROOT / "scripts" / "run_overnight.py"
FALLBACK_DATE_PATTERN = "MM/dd/yyyy"


def short_date_pattern() -> str:
    """The Windows short-date pattern (schtasks /SD must use it), e.g. 'dd/MM/yyyy'."""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Control Panel\International") as key:
            return str(winreg.QueryValueEx(key, "sShortDate")[0])
    except (ImportError, OSError):
        return FALLBACK_DATE_PATTERN


def format_date(when: datetime, pattern: str) -> str:
    """Format a date with a Windows pattern made of d, M and y runs (other characters kept)."""
    def part(m: re.Match) -> str:
        run = m.group(0)
        value = {"d": when.day, "M": when.month, "y": when.year}[run[0]]
        if run[0] == "y":
            return f"{value:04d}" if len(run) >= 3 else f"{value % 100:02d}"
        return f"{value:02d}" if len(run) >= 2 else str(value)

    return re.sub(r"d+|M+|y+", part, pattern)


def task_name(when: datetime) -> str:
    return "SlowPost-" + when.strftime("%Y%m%d-%H%M")


def write_job(when: datetime, brief: str, facts: str, n: int, csv: Path,
              folder: Path = SCHEDULED) -> Path:
    """Save what the task will run. The job file never leaves this computer."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{task_name(when)}.json"
    path.write_text(json.dumps({"brief": brief, "facts": facts, "n": n, "csv": str(csv),
                                "when": when.isoformat(timespec="minutes"),
                                "task": task_name(when)}, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path


def task_command(job: Path, python: str | None = None, script: Path = SCRIPT) -> str:
    """The /TR value: what Task Scheduler runs."""
    return f'"{python or sys.executable}" "{script}" --job "{job}"'


def create_args(when: datetime, job: Path, python: str | None = None,
                pattern: str | None = None) -> list[str]:
    """The exact schtasks arguments that create the one-time task."""
    return ["schtasks", "/Create", "/SC", "ONCE", "/TN", task_name(when),
            "/TR", task_command(job, python), "/ST", when.strftime("%H:%M"),
            "/SD", format_date(when, pattern or short_date_pattern()), "/F"]


def command_text(args: list[str]) -> str:
    """The same command as one line to paste into a terminal (Windows quoting, as subprocess
    sends it)."""
    return subprocess.list2cmdline(args)


def delete_text(when: datetime) -> str:
    return f"schtasks /Delete /TN {task_name(when)} /F"


def create(when: datetime, job: Path) -> tuple[bool, str]:
    """Create the task. Only called after the user confirmed in the app."""
    proc = subprocess.run(create_args(when, job), capture_output=True, text=True, check=False)
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def pending(now: datetime | None = None, folder: Path = SCHEDULED) -> list[dict]:
    """Jobs saved by this app whose time is still ahead, soonest first."""
    now = now or datetime.now()
    jobs = []
    for path in sorted(folder.glob("SlowPost-*.json")) if folder.exists() else []:
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
            if datetime.fromisoformat(job["when"]) > now:
                jobs.append(job)
        except (ValueError, KeyError, OSError):
            continue
    return jobs
