"""trace.jsonl: one JSON object per line, for everything the run did.

`event(kind, **fields)` appends a line with a timestamp, the run id and the
capability tag the manifest refers to. Moya's own
pipeline and step events are forwarded into the same file by `attach()`, so
one file tells the whole story: what the framework ran, what the model was
asked, what the validator decided, what the gate did.

Usage from code:
    trace.start_run(cap="R1", fresh=True)   # new run id; truncate the file
    trace.event("decision", msg_id="m001", disposition="reply")
    trace.attach(event_bus)                  # forward Moya step events
    trace.read()                             # list of dicts, for replay and tests
"""

import json
import time
from datetime import datetime, timezone

import config

_run_id = None
_cap = None


def start_run(cap=None, fresh=False):
    """Begin a run. `fresh=True` truncates the trace so it holds exactly one run."""
    global _run_id, _cap
    _run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _cap = cap
    if fresh:
        config.TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        config.TRACE_PATH.write_text("", encoding="utf-8")
    event("run_start", cap=cap, provider=config.PROVIDER, model=config.MODEL)
    return _run_id


def set_cap(cap):
    global _cap
    _cap = cap


def event(kind, **fields):
    """Append one line. Never raises: a broken trace must not stop a run."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "run_id": _run_id,
        "cap": fields.pop("cap", _cap),
        "event": kind,
    }
    record.update(fields)
    try:
        config.TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with config.TRACE_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
    except OSError as error:
        print(f"  [trace] could not write {config.TRACE_PATH}: {error}")
    return record


def read(path=None, cap=None, msg_id=None, kind=None):
    """All events, optionally filtered. Malformed lines are skipped, not fatal."""
    path = path or config.TRACE_PATH
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if cap and record.get("cap") != cap:
            continue
        if msg_id and record.get("msg_id") != msg_id:
            continue
        if kind and record.get("event") != kind:
            continue
        records.append(record)
    return records


# --- Moya bridge -----------------------------------------------------------

_MOYA_EVENT_TYPES = (
    "pipeline.started",
    "pipeline.completed",
    "pipeline.error",
    "step.started",
    "step.completed",
    "step.error",
)


def _forward(moya_event):
    fields = {
        "source": getattr(moya_event, "source", None),
        "thread_id": getattr(moya_event, "thread_id", None),
    }
    for name in ("step_name", "step_type", "duration_ms", "error", "pipeline_id"):
        value = getattr(moya_event, name, None)
        if value is not None:
            fields[name] = value
    event(f"moya.{moya_event.event_type}", **fields)


def attach(event_bus):
    """Subscribe to Moya's pipeline and step events and write them to the trace."""
    for event_type in _MOYA_EVENT_TYPES:
        event_bus.subscribe(event_type, _forward)
    return event_bus


if __name__ == "__main__":
    start_run(cap="demo", fresh=False)
    event("decision", msg_id="m000", disposition="archive", reason="trace self-test")
    print(f"wrote 2 events to {config.TRACE_PATH}; total lines now {len(read())}")
    time.sleep(0)
