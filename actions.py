"""What the system does to the mailbox, and which of it can be taken back.

Part 4 asks for every action to be classified as reversible or irreversible.
A label is cheap, so the register below is the thing the code actually reads:
`gate.py` decides what needs a human by asking `REGISTER` whether the action can
be undone, and `undo()` can only reverse an action the register says is
reversible. If a row here is wrong, the system behaves wrongly, which is the
only way to keep a classification honest.

Two files hold the result, and they answer different questions:

  state/mailbox.json   where every message sits now      (one row per message)
  state/actions.json   how it got there                  (append-only, ordered)

Everything here stays inside the mailbox, and that is why almost all of it is
reversible: archiving moves a message from one folder to another, and a move
within a mailbox this system owns is undone by moving it back.

The one action that escapes that is `send`, because a delivered reply is outside
the mailbox and outside anyone's reach. It is not in this module at all; it lives
in `gate.py`, behind the only approval this system asks for.

Usage:
    python actions.py     # apply the recorded decisions and print the folders
"""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import config

# The status a message carries once a disposition has been applied to it.
# One status at a time, which is the same guarantee Part 2 makes about
# dispositions: a message is in exactly one place.
INBOX = "inbox"
STATUS_FOR = {
    "reply": "awaiting_reply",
    "archive": "archived",
    "defer": "deferred",
    "delegate": "delegated",
    "escalate": "escalated",
    "flag": "flagged",
    "send": "answered",
    "delete": "bin",
}


@dataclass(frozen=True)
class ActionKind:
    """One row of the reversible/irreversible classification Part 4 asks for."""

    name: str
    effect: str  # what it does to the mailbox
    reversible: bool
    undo: str  # how it is taken back, or why it cannot be
    proposer: str  # "model" or "human": who is allowed to ask for it
    gated: bool  # does it have to pass through gate.py

    def row(self):
        return {
            "action": self.name,
            "effect": self.effect,
            "reversible": self.reversible,
            "undo": self.undo,
            "proposed_by": self.proposer,
            "gated": self.gated,
        }


REGISTER = {
    kind.name: kind
    for kind in (
        ActionKind(
            "reply",
            "records that a reply is owed; the message waits in the inbox",
            True,
            "a decision, not an act: re-running triage replaces it",
            "model",
            False,
        ),
        ActionKind(
            "draft",
            "writes reply text against the message",
            True,
            "rewrite it or discard it; nothing has left",
            "model",
            False,
        ),
        ActionKind(
            "archive",
            "the message leaves the inbox for the archive",
            True,
            "`--undo` puts it back in the inbox",
            "model",
            False,
        ),
        ActionKind(
            "defer",
            "snoozed; it returns to the inbox later",
            True,
            "`--undo` puts it back in the inbox",
            "model",
            False,
        ),
        ActionKind(
            "delegate",
            "marked as someone else's to answer",
            True,
            "`--undo` clears the mark; the hand-off note is a draft",
            "model",
            False,
        ),
        ActionKind(
            "escalate",
            "marked for the owner's attention",
            True,
            "`--undo` clears the mark",
            "model",
            False,
        ),
        ActionKind(
            "flag",
            "marked as refused, and left exactly where it is",
            True,
            "`--undo` clears the mark; the message was never moved",
            "model",
            False,
        ),
        ActionKind(
            "delete",
            "moved to the bin, recoverable until the retention window closes",
            True,
            "`--undo` restores it, until the retention window closes on its own",
            "human",
            True,
        ),
        ActionKind(
            "send",
            "the reply leaves as a file in outbox/, which is treated as delivered",
            False,
            "nothing: a sent message cannot be unsent",
            "model",
            True,
        ),
    )
}

REVERSIBLE = tuple(k.name for k in REGISTER.values() if k.reversible)
IRREVERSIBLE = tuple(k.name for k in REGISTER.values() if not k.reversible)


def now():
    return datetime.now(timezone.utc)


def stamp(moment=None):
    return (moment or now()).isoformat(timespec="seconds")


def purge_date(deleted_at=None):
    """When a message deleted now would stop being recoverable.

    Quoted in the approval prompt. Nothing in this build acts on it: the bin is
    a status and no message is ever removed from the store, so `delete` really
    is reversible here. The date is what makes the warning true rather than
    decorative -- a reversibility that expires on a timer is not the same as one
    that waits for someone to change their mind.
    """
    return stamp((deleted_at or now()) + timedelta(days=config.BIN_RETENTION_DAYS))


class Mailbox:
    """Where every message sits, and the ordered record of how it got there.

    Loaded from disk, changed in memory, written back by `save()`. A caller that
    forgets to save loses the change rather than half-applying it.
    """

    def __init__(self, folders=None, log=None):
        self.folders = dict(folders or {})
        self.log = list(log or [])

    # --- reading ---------------------------------------------------------

    def status(self, message_id):
        return self.folders.get(message_id, {}).get("status", INBOX)

    def detail(self, message_id):
        return dict(self.folders.get(message_id, {}).get("detail") or {})

    def in_status(self, status):
        return sorted(mid for mid in self.folders if self.status(mid) == status)

    def tally(self):
        counts = {}
        for message_id in self.folders:
            key = self.status(message_id)
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    # --- writing ---------------------------------------------------------

    def apply(self, action, message_id, reason="", detail=None, by="model"):
        """Put one message in the status this action implies, and log the move.

        Returns the log row, or None when the message is already there. Applying
        the same decision twice is not an error -- re-running a capability must
        not fill the audit trail with moves that did not happen.
        """
        kind = REGISTER.get(action)
        if kind is None:
            raise ValueError(f"{action!r} is not an action this system performs")
        target = STATUS_FOR.get(action)
        if target is None:
            raise ValueError(f"{action!r} does not change where a message sits")

        was = self.status(message_id)
        detail = dict(detail or {})
        if was == target and self.detail(message_id) == detail:
            return None

        row = {
            "seq": len(self.log) + 1,
            "at": stamp(),
            "action": action,
            "message_id": message_id,
            "from_status": was,
            "to_status": target,
            "reversible": kind.reversible,
            "by": by,
            "reason": reason,
            "detail": detail,
            "undone": False,
        }
        self.folders[message_id] = {
            "status": target,
            "since": row["at"],
            "reason": reason,
            "detail": detail,
        }
        self.log.append(row)
        return row

    def undo(self, seq):
        """Put a message back where it was before action `seq`.

        Refuses on two grounds, and both are the point of the register: an
        irreversible action cannot be undone whatever the caller wants, and an
        action already undone cannot be undone twice.

        The undo is itself logged, and logged as irreversible, so the trail
        always reads forwards. Taking back an undo means applying the original
        action again, which is a new row rather than a rewritten one.
        """
        row = next((r for r in self.log if r["seq"] == seq), None)
        if row is None:
            raise ValueError(f"there is no action {seq} in the log")
        if not row["reversible"]:
            raise ValueError(
                f"action {seq} is {row['action']}, which is irreversible: {REGISTER[row['action']].undo}"
            )
        if row["undone"]:
            raise ValueError(f"action {seq} was already undone")

        message_id = row["message_id"]
        if row["from_status"] == INBOX:
            self.folders.pop(message_id, None)
        else:
            self.folders[message_id] = {
                "status": row["from_status"],
                "since": stamp(),
                "reason": f"restored by undo of action {seq}",
                "detail": {},
            }
        row["undone"] = True
        self.log.append(
            {
                "seq": len(self.log) + 1,
                "at": stamp(),
                "action": "undo",
                "message_id": message_id,
                "from_status": row["to_status"],
                "to_status": row["from_status"],
                "reversible": False,
                "by": "human",
                "reason": f"undo of action {seq} ({row['action']})",
                "detail": {"undoes": seq},
                "undone": False,
            }
        )
        return row

    # --- persistence -----------------------------------------------------

    def save(self):
        config.STATE_PATH.mkdir(parents=True, exist_ok=True)
        _write(config.STATE_PATH / "mailbox.json", self.folders)
        _write(config.STATE_PATH / "actions.json", self.log)
        return config.STATE_PATH


def _write(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read(path, empty):
    if not path.exists():
        return empty
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        # A state file that cannot be read is treated as absent rather than
        # fatal. The alternative is a run that refuses to start because of a
        # file it is about to rewrite anyway.
        return empty


def load():
    """The mailbox as it stands. An empty one means nothing has been applied."""
    return Mailbox(
        folders=_read(config.STATE_PATH / "mailbox.json", {}),
        log=_read(config.STATE_PATH / "actions.json", []),
    )


def apply_decisions(decisions, mailbox=None):
    """Apply every recorded disposition to the mailbox. Idempotent.

    `decisions` is either the `Decision` objects a run produced or the rows read
    back from `decisions.json`; both answer to `message_id` and `disposition`.
    """
    mailbox = mailbox if mailbox is not None else load()
    applied = 0
    for decision in decisions:
        if isinstance(decision, dict):
            message_id = decision.get("message_id")
            disposition = decision.get("disposition")
            reason = decision.get("reason", "")
        else:
            message_id = decision.message_id
            disposition = decision.disposition
            reason = decision.reason
        if not message_id or disposition not in STATUS_FOR:
            continue
        if mailbox.apply(disposition, message_id, reason=reason) is not None:
            applied += 1
    return mailbox, applied


def load_applied(decisions_path=None):
    """The mailbox, built from the recorded decisions if it does not exist yet.

    A capability that acts on the mailbox should not require a fresh triage run
    first. If `mailbox.json` is missing but `decisions.json` is there, the
    dispositions that were already decided are applied and saved, so the folders
    agree with the decisions rather than starting empty beside them.
    """
    mailbox = load()
    if mailbox.folders:
        return mailbox
    path = decisions_path or (config.STATE_PATH / "decisions.json")
    rows = _read(path, [])
    if not rows:
        return mailbox
    mailbox, applied = apply_decisions(rows, mailbox)
    if applied:
        mailbox.save()
    return mailbox


def register_rows():
    """The classification, as plain data. The manifest reads this, not prose."""
    return [kind.row() for kind in REGISTER.values()]


if __name__ == "__main__":
    print("=== what this system does to the mailbox ===")
    for kind in REGISTER.values():
        mark = "reversible  " if kind.reversible else "IRREVERSIBLE"
        gate = " [gated]" if kind.gated else ""
        print(f"  {kind.name:9} {mark} {kind.effect}{gate}")
        print(f"            undo: {kind.undo}")

    box = load_applied()
    print(f"\n=== the mailbox now ({len(box.folders)} messages moved) ===")
    for status, count in box.tally().items():
        print(f"  {status:15} {count}")
    print(f"  actions logged  {len(box.log)}")
