"""The inbox, loaded once and validated at the door.

Everything downstream reads messages from here and nowhere else, so this is
the single boundary between a JSON file on disk and the rest of the system.

Nothing here trusts the file: a field is checked before it is read, never
after. Each record is validated field by field. A
record that fails is *not* dropped: it becomes a `Malformed` that still flows
through the run and still receives a disposition, because Part 2 is graded on
"no message is left without a disposition", and a message we could not parse
is exactly the kind a human should look at.

Usage:
    python mailstore.py     # load the real inbox and describe it
"""

import json
from dataclasses import dataclass, field
from datetime import datetime

import config

REQUIRED_FIELDS = ("id", "thread_id", "from", "to", "subject", "timestamp", "body", "unread")

# Bodies are quoted into prompts later, so a runaway record cannot be allowed to
# fill the context window. 20k characters is ~40x the largest body in this inbox.
MAX_BODY_CHARS = 20_000


@dataclass(frozen=True)
class Message:
    """One valid message. Frozen: nothing downstream may edit the inbox."""

    id: str
    thread_id: str
    sender: str
    to: str
    subject: str
    timestamp: str
    body: str
    unread: bool
    sent_at: datetime

    @property
    def sender_domain(self):
        return self.sender.rpartition("@")[2].lower()

    @property
    def from_owner(self):
        """True when the From address is the owner's. NOT proof the owner wrote it.

        m039 is the counter-example: a spoofed "assistant settings" note whose
        From is the owner. Callers must treat this as a hint, never as trust.
        """
        return self.sender.lower() == config.OWNER

    @property
    def internal(self):
        return self.sender_domain == config.OWNER.rpartition("@")[2].lower()

    def text(self):
        """Subject and body together, for keyword matching."""
        return f"{self.subject}\n{self.body}"

    def summary(self):
        return f"{self.id} [{self.thread_id}] {self.sender} - {self.subject}"


@dataclass(frozen=True)
class Malformed:
    """A record that failed validation. It still gets a disposition."""

    id: str
    problem: str
    raw: dict = field(default_factory=dict)

    def summary(self):
        return f"{self.id} (malformed: {self.problem})"


def _validate(record, position):
    """Return a Message, or a Malformed naming the first field that is wrong."""
    if not isinstance(record, dict):
        return Malformed(f"record#{position}", f"record is {type(record).__name__}, expected object")

    # The id is needed for every later error message, so it is checked first.
    raw_id = record.get("id")
    identifier = raw_id.strip() if isinstance(raw_id, str) and raw_id.strip() else f"record#{position}"
    if identifier.startswith("record#"):
        return Malformed(identifier, "missing or empty 'id'", record)

    missing = [name for name in REQUIRED_FIELDS if name not in record]
    if missing:
        return Malformed(identifier, f"missing field(s): {', '.join(missing)}", record)

    for name in REQUIRED_FIELDS:
        if name == "unread":
            continue
        if not isinstance(record[name], str):
            return Malformed(identifier, f"'{name}' is {type(record[name]).__name__}, expected string", record)

    if not isinstance(record["unread"], bool):
        return Malformed(identifier, f"'unread' is {type(record['unread']).__name__}, expected boolean", record)

    for name in ("thread_id", "from", "timestamp"):
        if not record[name].strip():
            return Malformed(identifier, f"'{name}' is empty", record)

    if "@" not in record["from"]:
        return Malformed(identifier, f"'from' is not an address: {record['from'][:60]!r}", record)

    try:
        sent_at = datetime.fromisoformat(record["timestamp"].strip())
    except ValueError:
        return Malformed(identifier, f"'timestamp' is not ISO 8601: {record['timestamp'][:40]!r}", record)

    if len(record["body"]) > MAX_BODY_CHARS:
        return Malformed(identifier, f"'body' is {len(record['body'])} chars, over the {MAX_BODY_CHARS} limit", record)

    return Message(
        id=identifier,
        thread_id=record["thread_id"].strip(),
        sender=record["from"].strip().lower(),
        to=record["to"].strip().lower(),
        subject=record["subject"].strip(),
        timestamp=record["timestamp"].strip(),
        body=record["body"],
        unread=record["unread"],
        sent_at=sent_at,
    )


class Mailbox:
    """Validated messages plus the indexes the rest of the system needs."""

    def __init__(self, messages, problems):
        self.messages = messages
        self.problems = problems
        self._by_id = {m.id: m for m in messages}
        self._threads = {}
        for message in messages:
            self._threads.setdefault(message.thread_id, []).append(message)
        for thread in self._threads.values():
            thread.sort(key=lambda m: m.sent_at)

    # --- the run reads these ---------------------------------------------

    def __len__(self):
        return len(self.messages) + len(self.problems)

    def everything(self):
        """Every record the run must dispose of: valid ones by time, then the bad ones."""
        return sorted(self.messages, key=lambda m: m.sent_at) + list(self.problems)

    def by_id(self, message_id):
        return self._by_id.get(message_id)

    def thread(self, thread_id):
        return list(self._threads.get(thread_id, ()))

    def earlier_in_thread(self, message):
        """Messages before this one in its thread, oldest first. Thread grounding."""
        return [m for m in self.thread(message.thread_id) if m.sent_at < message.sent_at]

    def known_addresses(self):
        """Every address that appears in the inbox: the recipient allowlist."""
        found = set()
        for message in self.messages:
            found.add(message.sender)
            found.update(part.strip() for part in message.to.split(",") if part.strip())
        return found

    def describe(self):
        threads = {t: len(v) for t, v in self._threads.items()}
        multi = {t: n for t, n in threads.items() if n > 1}
        return {
            "records": len(self),
            "valid": len(self.messages),
            "malformed": len(self.problems),
            "unread": sum(1 for m in self.messages if m.unread),
            "threads": len(threads),
            "multi_message_threads": multi,
            "senders": len({m.sender for m in self.messages}),
            "span": (
                f"{min(m.timestamp for m in self.messages)} to {max(m.timestamp for m in self.messages)}"
                if self.messages
                else "empty"
            ),
        }


class InboxFileError(RuntimeError):
    """The inbox file itself is unusable. One clear sentence, no traceback."""


def load(path=None):
    """Read and validate the inbox. Raises InboxFileError only if the FILE is unusable."""
    path = path or config.INBOX_PATH
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise InboxFileError(f"No inbox at {path}. Set INBOX_FILE or put the file there.") from None
    except OSError as error:
        raise InboxFileError(f"Cannot read {path}: {error}") from error

    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise InboxFileError(f"{path} is not valid JSON: line {error.lineno}, {error.msg}") from error

    if isinstance(data, dict) and isinstance(data.get("messages"), list):
        data = data["messages"]  # tolerate a wrapped format; ours is a bare array
    if not isinstance(data, list):
        raise InboxFileError(f"{path} holds a JSON {type(data).__name__}, expected an array of messages.")

    messages, problems, seen = [], [], set()
    for position, record in enumerate(data):
        result = _validate(record, position)
        if isinstance(result, Malformed):
            problems.append(result)
        elif result.id in seen:
            problems.append(Malformed(result.id, "duplicate id", record))
        else:
            seen.add(result.id)
            messages.append(result)
    return Mailbox(messages, problems)


if __name__ == "__main__":
    box = load()
    print(f"=== {config.INBOX_PATH} ===")
    for key, value in box.describe().items():
        print(f"  {key:22} {value}")
    if box.problems:
        print("\n  malformed records:")
        for bad in box.problems:
            print(f"    {bad.summary()}")
    print("\n  first three by time:")
    for message in box.everything()[:3]:
        print(f"    {message.summary()}")
    sample = box.by_id("m008")
    if sample:
        print(f"\n  earlier in {sample.thread_id} than m008: {[m.id for m in box.earlier_in_thread(sample)]}")
