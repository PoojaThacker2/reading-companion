"""schedule_store.py - the entire "scheduling system": a JSON file on disk.

No calendar API, no notifications. A session is just a dict appended to a list.
"""
import json
from datetime import datetime
from pathlib import Path

SCHEDULE_FILE = Path("schedule.json")


def load_sessions() -> list[dict]:
    """Return all saved sessions (empty list if the file doesn't exist yet)."""
    if not SCHEDULE_FILE.exists():
        return []
    return json.loads(SCHEDULE_FILE.read_text(encoding="utf-8"))


def add_session(book: str, date: str, time: str) -> dict:
    """Append one reading session and write the file back. Returns the saved entry."""
    session = {
        "book": book,
        "date": date,  # YYYY-MM-DD
        "time": time,  # HH:MM (24h)
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    sessions = load_sessions()
    sessions.append(session)
    SCHEDULE_FILE.write_text(json.dumps(sessions, indent=2), encoding="utf-8")
    return session
