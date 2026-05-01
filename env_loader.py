from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent
_loaded = False


def load_project_env() -> None:
    global _loaded
    if _loaded:
        return
    load_dotenv(_PROJECT_ROOT / ".env", override=False)
    _loaded = True
