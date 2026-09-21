"""Standing preferences that outlive a run. A JSON file on disk.

The whole of Part 5 rests on one property: `state/prefs.json` is on disk, so a
preference stated in one run is still there after the process has exited and a
new one has started. Nothing here is held in memory between calls.

`remember()` writes one preference, `recall()` finds the ones that match a
query, `summary()` renders them for a prompt. One key holds one value;
remembering an existing key overwrites it and reports what it replaced.

This module stores and retrieves. It does not decide what may be stored -- that
is `prefs.py`, which is the only caller of `remember()`, and which refuses
anything that is not a constraint. Keeping the two apart is deliberate: a store
that can hold any key at all is easy to reason about, and the judgement about
what deserves to be in it lives in one place rather than being spread across a
file format.

Usage:
    python memory.py    # write, restate, recall, print the file
"""

import json

import config


def _file():
    return config.STATE_PATH / "prefs.json"


def _normalise_key(key):
    """'CC on Legal mail ' -> 'cc_on_legal_mail'."""
    return "_".join(str(key or "").strip().lower().split())


def _load():
    path = _file()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _save(data):
    path = _file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def remember(key, value, source="user", **extra):
    """Store one preference. `extra` carries structured fields (e.g. applies_to, rule)."""
    name = _normalise_key(key)
    if not name:
        return {"error": "No key supplied. Name the preference, e.g. 'cc_legal'."}
    if value is None or str(value).strip() == "":
        return {"error": f"No value supplied for '{name}'."}

    data = _load()
    previous = data.get(name)
    data[name] = {"value": str(value).strip(), "source": str(source or "user"), **extra}
    _save(data)

    result = {"stored": name, **data[name]}
    if previous is not None and previous.get("value") != data[name]["value"]:
        result["replaced"] = previous.get("value")
    return result


def recall(query=""):
    """Preferences whose key, value or extra fields contain the query. Empty query returns all."""
    needle = str(query or "").strip().lower()
    data = _load()
    matches = {
        key: entry
        for key, entry in data.items()
        if not needle or needle in key or needle in json.dumps(entry).lower()
    }
    return {"query": query, "matches": matches, "count": len(matches), "stored_keys": sorted(data)}


def all_prefs():
    return _load()


def forget(key):
    data = _load()
    name = _normalise_key(key)
    if name not in data:
        return {"error": f"No preference named '{name}'."}
    removed = data.pop(name)
    _save(data)
    return {"forgot": name, **removed}


def clear():
    path = _file()
    if path.exists():
        path.unlink()


def summary():
    """The store as lines for a prompt; empty string when nothing is stored."""
    data = _load()
    if not data:
        return ""
    lines = [f"- {key}: {entry['value']} (from {entry.get('source', '?')})" for key, entry in data.items()]
    return "Standing preferences recorded in earlier runs:\n" + "\n".join(lines)


def show():
    path = _file()
    print(path.read_text(encoding="utf-8") if path.exists() else "(no prefs file yet)")


if __name__ == "__main__":
    print("=== the preference store ===")
    print(f"  file: {_file()}\n")
    print("remember:", remember("cc legal", "cc priya@paperjet.io on mail from hartwellcho.com", "m015", applies_to="hartwellcho.com"))
    print("remember:", remember("meeting floor", "no meetings before 11:00", "m041"))
    print("restate: ", remember("meeting_floor", "no meetings before 11:30", "m041"))
    print("recall:  ", recall("hartwell"))
    print("recall:  ", recall("cricket"))
    print("\n--- summary, as a prompt will see it ---")
    print(summary())
    print("\n--- prefs.json on disk ---")
    show()
