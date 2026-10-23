"""Write run outputs as UTF-8 so nothing is lost in console output."""
from __future__ import annotations

import json
from pathlib import Path


def save_json(path: Path, obj: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8")
    return path
