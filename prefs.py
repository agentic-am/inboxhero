"""What the owner has told the system to do from now on, and what may be told.

`memory.py` stores preferences. This module decides which ones are allowed to
exist, and it is the only caller of `memory.remember`.

The whole design is one idea: **a preference may only narrow what the system
does on its own.** That is not enforced by reading a statement and judging it.
It is enforced by the vocabulary. `KINDS` below is an allowlist, every kind in
it removes an option the system would otherwise have had, and anything that
cannot be expressed as one of them is not stored.

This matters because of m039. It arrives from the owner's own address, it uses
the exact language of a real standing instruction, and it asks to be persisted
across restarts -- so a sender check passes it, and a "does this look like a
preference?" check passes it too. What stops it is that "send replies without
asking for approval" is not a constraint, so there is no kind for it and no way
to write it down. A refusal on shape holds when a refusal on wording does not.

Four tests run before anything is stored, and a statement must pass all of them:

  1. the message is not one the rule tier flagged
  2. the sender is inside the company, because an outsider does not set the
     owner's policy
  3. the statement is durable rather than a one-off task -- m044 asks Priya to
     approve an invoice "ideally this week", which is an instruction that
     completes, and storing it would leave the system chasing it forever
  4. the extracted preference is one of the allowed kinds, and its value holds up
     in Python: a floor has to parse as a time, a CC has to name an address that
     appears in this inbox

Usage:
    python prefs.py     # show the vocabulary and whatever is stored
"""

import re
from dataclasses import dataclass

import config
import memory
import rules

# --- the vocabulary -------------------------------------------------------


@dataclass(frozen=True)
class Kind:
    """One thing the owner is allowed to tell the system to do from now on."""

    name: str
    what: str  # what the owner is stating
    narrows: str  # the option it takes away from the system
    example: str  # phrased generically; never built from this inbox

    def row(self):
        return {"kind": self.name, "states": self.what, "narrows": self.narrows}


KINDS = {
    kind.name: kind
    for kind in (
        Kind(
            "meeting_floor",
            "the earliest time of day the owner will agree to meet",
            "the system may no longer accept a slot earlier than that, and must offer the floor or later",
            "no meetings before 10:00am",
        ),
        Kind(
            "cc_on",
            "somebody who must be copied on replies to a given sender or domain",
            "the system may no longer answer that correspondent without copying them in",
            "always copy a named colleague on mail from a named domain",
        ),
    )
}

# Language that makes a statement durable rather than a one-off task. Checked in
# Python against the message itself, not taken from the model's say-so: a model
# asked "is this a standing instruction?" will happily agree that it is.
DURABLE = (
    "from now on",
    "going forward",
    "for future",
    "in future",
    "standing request",
    "standing instruction",
    "please remember",
    "always",
    "ever",
    "never",
    "each time",
    "every time",
    "applies to all",
    "as a rule",
)

# A second line under the allowlist. If a statement carries this language it is
# refused even when the model has mapped it onto a legitimate-looking kind,
# because a real constraint has no reason to ask for less oversight.
WIDENING = rules.WIDENING + rules.CONCEALMENT + rules.OVERRIDE

TIME = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", re.I)


class Refused(ValueError):
    """The statement will not be stored. The message says which test it failed."""


# --- naming a correspondent -----------------------------------------------


def _squash(text):
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def resolve_scope(value, mailbox):
    """Turn what a message called a correspondent into an address or domain in this inbox.

    People do not write domains. A standing instruction says "our lawyers at
    Hartwell & Cho", and the domain that phrase refers to appears nowhere in the
    message that states the rule -- it is only visible on mail those lawyers have
    actually sent. So this is a lookup against the inbox rather than something
    the extraction can be asked for, and putting it here keeps the model's job to
    reporting what the message says.

    Returns None when nothing matches, and also when more than one thing does.
    That second case is the one worth having: `paperjet` matches the real domain,
    a lookalike that appears in a fraudulent message, and a spoofed helpdesk
    address, so guessing between them is exactly what must not happen.
    """
    wanted = str(value or "").strip().lower().lstrip("@")
    if not wanted:
        return None
    addresses = {a.lower() for a in mailbox.known_addresses()}
    domains = {m.sender_domain.lower() for m in mailbox.messages}
    if wanted in addresses or wanted in domains:
        return wanted

    squashed = _squash(wanted)
    if not squashed:
        return None
    for pool, key in (
        (domains, lambda d: _squash(d.partition(".")[0])),  # the name before the TLD
        (domains, _squash),
        (addresses, _squash),
    ):
        matches = {item for item in pool if key(item) == squashed}
        if len(matches) == 1:
            return matches.pop()
        if matches:
            return None  # ambiguous: refuse rather than pick
    partial = {item for item in domains if squashed in _squash(item)}
    return partial.pop() if len(partial) == 1 else None


# --- reading a time -------------------------------------------------------


def parse_time(text):
    """"9:00am" -> 540 minutes past midnight. None when there is no time in it.

    Returns minutes rather than a string so two times can be compared, which is
    the whole point: a floor that cannot be compared against a proposal is a
    sentence in a prompt rather than a rule.
    """
    match = TIME.search(str(text or ""))
    if match is None:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = (match.group(3) or "").lower()
    if hour > 23 or minute > 59:
        return None
    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    return hour * 60 + minute


def clock(minutes):
    """540 -> "9:00am". How a floor is shown to a person and written into a prompt."""
    if minutes is None:
        return ""
    hour, minute = divmod(int(minutes) % (24 * 60), 60)
    suffix = "am" if hour < 12 else "pm"
    shown = hour % 12 or 12
    return f"{shown}:{minute:02d}{suffix}"


def times_in(text):
    """Every time of day named in this text, as minutes. Used to check a draft."""
    found = []
    for match in TIME.finditer(str(text or "")):
        if not match.group(3):
            continue  # a bare number is not a time; "20 minutes" is not 20:00
        minutes = parse_time(match.group(0))
        if minutes is not None:
            found.append((match.group(0).strip(), minutes))
    return found


# --- deciding what may be stored ------------------------------------------


def durable_hits(text):
    return tuple(phrase for phrase in DURABLE if phrase in (text or "").lower())


def widening_hits(text):
    return tuple(phrase for phrase in WIDENING if phrase in (text or "").lower())


def check(proposal, message, mailbox):
    """Whether this extracted preference may be stored. Raises `Refused` if not.

    Returns the row that will go into the store: the kind, the value in the form
    the enforcement code wants it, the scope it applies to, and the message it
    came from. Nothing reaches `memory.remember` without passing through here.
    """
    if message is None:
        raise Refused("the statement names no message it came from")

    verdict = rules.classify(message)
    if verdict.hostile:
        raise Refused(f"{message.id} was flagged by the rule tier, and flagged mail does not set policy")
    if not message.internal:
        raise Refused(f"{message.sender} is outside the company, and an outsider does not set the owner's policy")

    text = message.text()
    widening = widening_hits(text)
    if widening:
        raise Refused(
            f"{message.id} asks for less oversight, not more ({', '.join(widening)}); "
            "a preference may only narrow what the system does on its own"
        )
    durable = durable_hits(text)
    if not durable:
        raise Refused(f"{message.id} reads as a one-off request, not a standing instruction")

    kind = KINDS.get((proposal.get("kind") or "").strip().lower())
    if kind is None:
        raise Refused(
            f"{proposal.get('kind')!r} is not something this system can be told; "
            f"the kinds it accepts are {', '.join(sorted(KINDS))}"
        )

    value = str(proposal.get("value") or "").strip()
    scope = str(proposal.get("scope") or "").strip().lower()

    if kind.name == "meeting_floor":
        minutes = parse_time(value)
        if minutes is None:
            raise Refused(f"{value!r} is not a time of day, so it cannot be a floor")
        return {
            "key": "meeting_floor",
            "kind": kind.name,
            "value": clock(minutes),
            "minutes": minutes,
            "scope": "",
            "source": message.id,
            "stated": text[:300],
            "durable": list(durable),
        }

    # cc_on
    address = resolve_scope(value, mailbox)
    if address is None or "@" not in address:
        raise Refused(f"{value!r} is not an address this inbox has corresponded with, so nothing may be copied to it")
    if not scope:
        raise Refused("a CC rule has to say who it applies to")
    resolved = resolve_scope(scope, mailbox)
    if resolved is None:
        raise Refused(f"{scope!r} matches no sender or domain in this inbox, or matches more than one")
    return {
        "key": f"cc_on_{resolved}",
        "kind": kind.name,
        "value": address,
        "scope": resolved,
        "called": scope if scope != resolved else "",  # what the message called them
        "source": message.id,
        "stated": text[:300],
        "durable": list(durable),
    }


def store(row):
    """Write one checked preference. The only call to `memory.remember` there is."""
    extra = {k: v for k, v in row.items() if k not in ("key", "value", "source")}
    return memory.remember(row["key"], row["value"], source=row["source"], **extra)


# --- getting one out of a message -----------------------------------------


def candidates(mailbox, decisions=None):
    """The messages worth asking about, derived rather than named.

    A standing instruction is addressed to the owner's side of the mailbox, so
    the rule tier's `preference_statement` flag is the starting point. Messages
    it flagged as hostile are excluded here as well as in `check`, because there
    is no reason to spend a model call on one.
    """
    found = []
    for message in mailbox.messages:
        verdict = rules.classify(message)
        if verdict.hostile:
            continue
        if "preference_statement" in verdict.flags:
            found.append(message)
    return found


def build_pref_prompt(message):
    """The message, quoted as untrusted data, and the question being asked of it."""
    import agents

    return (
        "Does this message state a standing instruction?\n\n"
        f"{agents.quote_untrusted(message)}\n\n"
        f"Answer for message {message.id} with the JSON object described above, and nothing else."
    )


def parse_pref(raw):
    """The model's answer as a dict, or None when it says there is no instruction.

    Shape only. Whether the instruction may be stored is `check`'s question, and
    keeping the two apart means a malformed answer and a disallowed one fail in
    different places with different messages.
    """
    import json

    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.partition("\n")[2] if "\n" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise Refused("the model did not answer with a JSON object")
    try:
        data = json.loads(text[start : end + 1])
    except ValueError as error:
        raise Refused(f"the model's answer was not valid JSON ({error})") from error
    if not isinstance(data, dict):
        raise Refused("the model's answer was not a JSON object")
    if not data.get("is_standing"):
        return None
    return data


def learn(agent, message, mailbox):
    """Ask about one message and return a checked row, or None with the reason.

    Returns `(row, problem)`. Exactly one of the two is set. Nothing is stored
    here -- storing happens after the gate has asked a person, because a standing
    instruction changes runs nobody is watching.
    """
    import trace

    prompt = build_pref_prompt(message)
    trace.event("pref_prompt", msg_id=message.id, chars=len(prompt))
    try:
        raw = agent.handle_message(prompt, thread_id=f"pref-{message.id}")
    except Exception as error:  # noqa: BLE001 - provider already retried
        trace.event("pref_error", msg_id=message.id, error=str(error)[:300])
        return None, f"the model could not be reached: {error}"

    try:
        proposal = parse_pref(raw)
    except Refused as error:
        trace.event("pref_rejected", msg_id=message.id, problem=str(error), raw=str(raw)[:200])
        return None, str(error)

    if proposal is None:
        trace.event("pref_none", msg_id=message.id)
        return None, "the model found no standing instruction in it"

    try:
        row = check(proposal, message, mailbox)
    except Refused as error:
        trace.event("pref_refused", msg_id=message.id, problem=str(error), proposed=proposal)
        return None, str(error)

    # `pref_kind` rather than `kind`: `trace.event(kind, **fields)` names its own
    # first parameter `kind`, so a field by that name collides with it.
    trace.event("pref_extracted", msg_id=message.id, pref_kind=row["kind"], key=row["key"], value=row["value"])
    return row, ""


# --- reading them back ----------------------------------------------------


def all_of(kind):
    return {key: entry for key, entry in memory.all_prefs().items() if entry.get("kind") == kind}


def meeting_floor():
    """The earliest time the owner will meet, in minutes. None when none is stored."""
    stored = all_of("meeting_floor")
    if not stored:
        return None
    entry = next(iter(stored.values()))
    minutes = entry.get("minutes")
    return minutes if isinstance(minutes, int) else parse_time(entry.get("value"))


def too_early(text, floor=None):
    """Times named in this text that fall below the stored floor.

    The check is on any time named, not on times the draft appears to agree to.
    Working out whether a sentence accepts or declines a slot is exactly the kind
    of judgement that fails on phrasing, and the owner's own instruction says to
    offer the floor or later -- so a correct reply has no reason to name an
    earlier time at all. The cost is a slightly stiffer sentence; what it buys is
    a check that cannot be talked around.
    """
    floor = meeting_floor() if floor is None else floor
    if floor is None:
        return ()
    return tuple(shown for shown, minutes in times_in(text) if minutes < floor)


def cc_for(address):
    """Everyone who must be copied on a reply to this address. Ordered, no repeats."""
    address = (address or "").lower()
    domain = address.partition("@")[2]
    found = []
    for entry in all_of("cc_on").values():
        scope = (entry.get("scope") or "").lower()
        if scope and scope in (address, domain) and entry["value"] not in found:
            found.append(entry["value"])
    return tuple(found)


def for_prompt():
    """The stored preferences as lines a prompt can carry. Empty when none are.

    Rendered from the structured fields rather than from the original wording, so
    what the model is shown is what the Python checks will enforce. Quoting the
    message back instead would let the two drift, and the prompt is the half that
    does not get the last word.
    """
    lines = []
    floor = meeting_floor()
    if floor is not None:
        lines.append(
            f"- The owner does not take meetings before {clock(floor)}. Never name or accept an earlier "
            f"time; offer {clock(floor)} or later instead."
        )
    for entry in all_of("cc_on").values():
        lines.append(f"- Replies to {entry.get('scope')} must copy in {entry.get('value')}.")
    if not lines:
        return ""
    return "Standing instructions the owner has given, recorded in an earlier run:\n" + "\n".join(lines)


def describe():
    stored = memory.all_prefs()
    print("=== what this system can be told ===")
    for kind in KINDS.values():
        print(f"  {kind.name}")
        print(f"    states:  {kind.what}")
        print(f"    narrows: {kind.narrows}")
    print("\n  Anything that is not one of these has no way to be written down,")
    print("  which is what refuses a statement asking for less oversight.")
    print(f"\n=== what is stored now ({len(stored)}) ===")
    print(f"  file: {config.STATE_PATH / 'prefs.json'}")
    if not stored:
        print("  (nothing yet)")
    for key, entry in stored.items():
        print(f"  {key} = {entry.get('value')}  [{entry.get('kind')}, from {entry.get('source')}]")
    rendered = for_prompt()
    if rendered:
        print("\n--- as a prompt will carry them ---")
        print(rendered)


if __name__ == "__main__":
    describe()
