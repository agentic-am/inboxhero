"""X1. Who owes the next move, on every thread at once.

At a hundred messages a day this is a convenience. At eleven thousand it is the
only question that still scales, because the owner cannot read the inbox and the
useful reduction is not "what arrived" but "what is stuck, and on whom".

Three states, and the rule for each is one sentence:

    waiting on you    the last word is theirs, and the system has not closed it
    waiting on them   the last word is yours, and nobody has answered
    closed            archived, refused, or a note to yourself; nothing is owed

`waiting on them` is the one a mail client will not tell you. Gmail and Outlook
both surface unread mail well and neither notices that you asked a colleague for
something eight days ago and they never replied -- the message is in your *sent*
folder, out of sight, and no unread badge will ever appear for it.

No model runs. The state of a thread is a fact about who sent the last message
and what the system decided, both already recorded, so asking a model would add
latency and variance to a lookup.

Usage:
    python waiting.py     # every thread, grouped by who owes the next move
"""

import json
from dataclasses import dataclass

import config
import mailstore
import rules

# Dispositions that close a thread. `reply` and `defer` deliberately do not:
# deciding to answer something is not answering it, and a deferred message comes
# back. `escalate` does not close it either -- it is the owner's move.
#
# `delegate` is not here, and that is the correction this view exists to make:
# handing something to a colleague does not finish it, it moves whose move it is.
# Counting it as closed hid the one message in this inbox where the owner asked
# somebody for something and never heard back.
CLOSED_BY = ("archive", "flag")


@dataclass
class Thread:
    """One conversation, and whose move it is."""

    thread_id: str
    subject: str
    messages: tuple
    state: str  # "you" | "them" | "closed"
    why: str
    last_at: object = None
    last_from: str = ""
    waiting_days: float = 0.0
    disposition: str = ""

    @property
    def ids(self):
        return tuple(m.id for m in self.messages)

    def line(self):
        age = f"{self.waiting_days:.0f}d" if self.waiting_days >= 1 else "today"
        return f"{age:>6}  {self.ids[-1]:6} {self.last_from:32} {self.subject[:44]}"


def classify(thread_id, messages, rows, now, owner):
    """Whose move it is on one thread.

    Order matters. A refused message is closed however it looks, because acting on
    it is the thing Part 6 exists to prevent; and a thread whose last word is the
    owner's is waiting on the other side even if an earlier message in it was
    archived.
    """
    ordered = sorted(messages, key=lambda m: m.sent_at)
    last = ordered[-1]
    row = rows.get(last.id, {})
    disposition = row.get("disposition", "")
    subject = last.subject or "(no subject)"
    waiting = max((now - last.sent_at).total_seconds() / 86400.0, 0.0)

    if rules.classify(last).hostile:
        state, why = "closed", "refused and left in place; nothing is owed"
    elif last.sender.lower() == owner and last.to.lower() == owner:
        # A note the owner wrote to themselves. Nobody owes a reply to it, and
        # reading it as "waiting on them" put a standing instruction on the chase
        # list, which is nonsense: there is no them.
        state, why = "closed", "a note to yourself; nobody owes a reply"
    elif disposition in CLOSED_BY:
        state, why = "closed", f"the system settled it as {disposition}"
    elif last.sender.lower() == owner:
        # The owner spoke last. Nobody has come back, and no mail client will
        # ever raise this, because the message is sitting in `sent`.
        owed = "it was delegated, so it is theirs" if disposition == "delegate" else "nobody has answered"
        state, why = "them", f"you wrote last {waiting:.0f} day(s) ago and {owed}"
    else:
        reason = {
            "reply": "a reply is owed and has not been sent",
            "defer": "deferred; it comes back",
            "escalate": "escalated for you to look at",
        }.get(disposition, "no disposition is recorded for it")
        state, why = "you", reason

    return Thread(
        thread_id=thread_id,
        subject=subject,
        messages=tuple(ordered),
        state=state,
        why=why,
        last_at=last.sent_at,
        last_from=last.sender,
        waiting_days=waiting,
        disposition=disposition,
    )


def survey(mailbox=None, rows=None, now=None):
    """Every thread in the inbox, oldest wait first within each state."""
    mailbox = mailbox if mailbox is not None else mailstore.load()
    if rows is None:
        path = config.STATE_PATH / "decisions.json"
        loaded = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        rows = {r.get("message_id"): r for r in loaded}
    # "Now" is the last thing that arrived, not the wall clock. The inbox is a
    # fixed corpus, and measuring its ages against today would report every
    # thread as months stale purely because the file is not from today.
    now = now or max(m.sent_at for m in mailbox.messages)
    owner = config.OWNER

    found = [
        classify(thread_id, mailbox.thread(thread_id), rows, now, owner)
        for thread_id in {m.thread_id for m in mailbox.messages}
    ]
    found.sort(key=lambda t: -t.waiting_days)
    return found


def grouped(threads):
    return {
        "you": [t for t in threads if t.state == "you"],
        "them": [t for t in threads if t.state == "them"],
        "closed": [t for t in threads if t.state == "closed"],
    }


def oldest_unanswered(threads, days=2.0):
    """Threads waiting on somebody else for longer than `days`. The nudge list."""
    return [t for t in threads if t.state == "them" and t.waiting_days >= days]


def render(threads):
    out = []
    by_state = grouped(threads)
    total = len(threads)
    out.append(f"=== {total} threads ===")
    out.append(
        f"  waiting on you {len(by_state['you']):4}    "
        f"waiting on them {len(by_state['them']):4}    "
        f"closed {len(by_state['closed']):4}"
    )

    out.append(f"\n--- waiting on you ({len(by_state['you'])}), longest first ---")
    for thread in by_state["you"]:
        out.append(thread.line())
        out.append(f"          {thread.why}")
    if not by_state["you"]:
        out.append("  (nothing)")

    out.append(f"\n--- waiting on them ({len(by_state['them'])}) ---")
    out.append("  mail you sent that nobody answered. No inbox raises these: they are in `sent`.")
    for thread in by_state["them"]:
        out.append(thread.line())
        out.append(f"          {thread.why}")
    if not by_state["them"]:
        out.append("  (nothing)")

    out.append(f"\n--- closed ({len(by_state['closed'])}) ---")
    counts = {}
    for thread in by_state["closed"]:
        counts[thread.why] = counts.get(thread.why, 0) + 1
    for why, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        out.append(f"  {count:4}  {why}")
    return "\n".join(out)


if __name__ == "__main__":
    threads = survey()
    print(render(threads))
    nudge = oldest_unanswered(threads)
    if nudge:
        print(f"\n  {len(nudge)} worth chasing: {', '.join(t.ids[-1] for t in nudge)}")
