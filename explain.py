"""X4. Why did the system do that to this message?

Every other part of this system decides something. This one only reads, and it
reads the one file that was written as those decisions were made. Nothing here
recomputes a verdict, asks a model, or reasons about what *would* have happened:
if the answer is not in `trace.jsonl`, this says so rather than reconstructing a
plausible story, because a plausible story is exactly what an audit trail is for
avoiding.

That makes it the cheapest capability in the manifest and the one the others lean
on. A gate that refuses, a preference that fires, a draft that is thrown away --
each is defensible only if somebody can ask why afterwards and get the actual
reason rather than a summary written later.

The shape of an answer follows the pipeline:

    read -> rule -> [retrieve -> prompt -> triage -> validate -> draft] -> decision
                                                                    -> gate

Usage:
    python explain.py m030      # why that message ended up where it did
"""

from dataclasses import dataclass, field

import config
import mailstore
import trace

# The events that tell the story, in the order they happen, and how to render one.
# A kind not listed here is machinery (Moya's own step events) and is left out of
# the narrative unless nothing else is there.
STORY = (
    "read",
    "rule",
    "refusal",
    "retrieve",
    "prompt",
    "batched_hit",
    "llm_call",
    "proposal_rejected",
    "validate",
    "draft_rejected",
    "draft",
    "decision",
    "pref_extracted",
    "gate",
)

MACHINERY = ("moya.", "run_start")


@dataclass
class Step:
    """One thing that happened, with the line that says it."""

    at: str
    kind: str
    cap: str
    said: str
    raw: dict = field(default_factory=dict)

    def line(self):
        # Date and time, not time alone. A message's story can span several runs
        # days apart, and the first version showed only the clock -- so a step
        # from yesterday's run appeared to happen after one from today's.
        when = f"{self.at[5:10]} {self.at[11:19]}" if len(self.at) > 19 else self.at
        return f"  {when}  [{self.cap or '-':3}] {self.kind:18} {self.said}"


def _rule(event):
    bits = [f"the rule tier said {event.get('rule')!r}"]
    if event.get("disposition"):
        bits.append(f"and settled it as {event['disposition']}")
    else:
        bits.append(f"and left it to the model, allowing {', '.join(event.get('allowed') or [])}")
    if event.get("flags"):
        bits.append(f"(flags: {', '.join(event['flags'])})")
    return " ".join(bits)


def _retrieve(event):
    found = event.get("evidence") or []
    if not found:
        return "retrieval found nothing to ground it"
    parts = ", ".join(f"{e['id']} via {e['source']}" for e in found)
    return f"retrieval offered {parts}"


def _validate(event):
    if not event.get("accepted"):
        return f"the answer was thrown away: {event.get('problem')}"
    said = f"accepted {event.get('disposition')!r} on attempt {event.get('attempts')}"
    if event.get("cites"):
        said += f", citing {', '.join(event['cites'])}"
    return said


def _gate(event):
    return (
        f"proposed {event.get('proposed')}; "
        f"{'asked because ' + '; '.join(event.get('needs_human') or []) if event.get('needs_human') else 'not over the line'}; "
        f"human said {event.get('human_said')!r}; {event.get('happened')}"
    )


SAID = {
    "read": lambda e: f"the run picked it up: {e.get('subject', '')!r}",
    "rule": _rule,
    "refusal": lambda e: f"REFUSED. it attempted: {e.get('attempted')}. instead: {e.get('instead')}",
    "retrieve": _retrieve,
    "prompt": lambda e: f"a prompt of {e.get('chars')} characters was built, carrying {', '.join(e.get('evidence') or []) or 'no evidence'}",
    "batched_hit": lambda e: f"answered by a batched call as {e.get('disposition')}",
    "llm_call": lambda e: f"the model was called ({e.get('prompt_tokens', '?')} in, {e.get('completion_tokens', '?')} out)",
    "proposal_rejected": lambda e: f"attempt {e.get('attempt')} was rejected: {e.get('problem')}",
    "validate": _validate,
    "draft_rejected": lambda e: f"a draft was rejected: {e.get('problem')}",
    "draft": lambda e: (
        f"drafted, citing {', '.join(e.get('cites') or []) or 'nothing'}"
        if e.get("drafted", True) and e.get("outcome", "answered") == "answered"
        else f"no draft [{e.get('outcome')}]: {e.get('reason')}"
    ),
    "decision": lambda e: f"FINAL: {e.get('disposition')} via the {e.get('path')} path -- {e.get('reason')}",
    "pref_extracted": lambda e: f"a standing instruction was taken from it: {e.get('key')} = {e.get('value')}",
    "gate": _gate,
}


def story(message_id, events=None):
    """Every recorded step for one message, oldest first."""
    events = events if events is not None else trace.read()
    found = []
    for event in events:
        if event.get("msg_id") != message_id:
            continue
        kind = event.get("event", "")
        if kind.startswith(MACHINERY):
            continue
        render = SAID.get(kind)
        found.append(
            Step(
                at=event.get("ts", ""),
                kind=kind,
                cap=event.get("cap") or "",
                said=render(event) if render else str({k: v for k, v in event.items() if k not in ("ts", "run_id", "cap", "event", "msg_id")})[:150],
                raw=event,
            )
        )
    return found


def runs_covering(steps):
    """Which capabilities the story came from, in order of first appearance."""
    seen = []
    for step in steps:
        if step.cap and step.cap not in seen:
            seen.append(step.cap)
    return seen


def explain(message_id, mailbox=None, events=None):
    """The whole answer for one message: what it was, what happened, and the outcome."""
    mailbox = mailbox if mailbox is not None else mailstore.load()
    message = mailbox.by_id(message_id)
    steps = story(message_id, events)
    final = next((s for s in reversed(steps) if s.kind == "decision"), None)
    return {
        "message_id": message_id,
        "message": message,
        "steps": steps,
        "decision": final,
        "caps": runs_covering(steps),
    }


def render(answer):
    out = []
    message = answer["message"]
    if message is None:
        out.append(f"{answer['message_id']} is not in {config.INBOX_PATH.name}.")
    else:
        out.append(f"=== {message.id}  {message.sender} -> {message.to} ===")
        out.append(f"  subject: {message.subject}")
        out.append(f"  sent:    {message.timestamp}")
        out.append(f"  body:    {message.body[:200].replace(chr(10), ' ')}")

    steps = answer["steps"]
    if not steps:
        # The honest answer. A story invented here would be indistinguishable
        # from a recorded one, which is the whole reason the trace exists.
        out.append(f"\n  Nothing about {answer['message_id']} is in {config.TRACE_PATH.name}.")
        out.append("  Either it has not been through a run, or the trace was truncated by a later one.")
        return "\n".join(out)

    out.append(f"\n  {len(steps)} recorded step(s), from {', '.join(answer['caps']) or 'an untagged run'}:\n")
    for step in steps:
        out.append(step.line())

    if answer["decision"]:
        out.append(f"\n  {answer['decision'].said}")
    else:
        out.append("\n  No final decision was recorded for it.")
    return "\n".join(out)


if __name__ == "__main__":
    import sys

    box = mailstore.load()
    wanted = sys.argv[1:]
    if not wanted:
        # No id written down: take the messages whose story has the most in it,
        # which are the ones worth reading an explanation of.
        events = trace.read()
        counts = {}
        for event in events:
            if event.get("msg_id") and not event.get("event", "").startswith(MACHINERY):
                counts[event["msg_id"]] = counts.get(event["msg_id"], 0) + 1
        wanted = [mid for mid, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:2]]
        print(f"(no message named; showing the two with the most recorded: {', '.join(wanted)})\n")
    for message_id in wanted:
        print(render(explain(message_id, box)))
        print()
