"""Persisted user settings — recent projects, last-opened path."""
import json
import os


SETTINGS_DIR = os.path.expanduser("~/.config/nudent")
SETTINGS_FILE = os.path.join(SETTINGS_DIR, "settings.json")
MAX_RECENT = 8


def _load():
    if not os.path.exists(SETTINGS_FILE):
        return {"last_project": None, "recent_projects": []}
    try:
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"last_project": None, "recent_projects": []}


def _save(data):
    os.makedirs(SETTINGS_DIR, exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def record_recent(path):
    """Add `path` to recent list and set as last_project."""
    data = _load()
    recent = [p for p in data.get("recent_projects", []) if p != path]
    recent.insert(0, path)
    data["recent_projects"] = recent[:MAX_RECENT]
    data["last_project"] = path
    _save(data)


def forget(path):
    """Remove `path` from recent + last_project (e.g. file no longer exists)."""
    data = _load()
    data["recent_projects"] = [p for p in data.get("recent_projects", []) if p != path]
    if data.get("last_project") == path:
        data["last_project"] = None
    _save(data)


def get_recent():
    return _load().get("recent_projects", [])


def get_last_project():
    return _load().get("last_project")
