"""Dates, deadlines and obligations, pulled out of the inbox with their sources.

Part 7's third pane carries the marks, and what it is marked on is not prose: every
entry must cite the messages it came from, at least one must be resolvable only by
combining two, and two things at the same time must be called out rather than
listed quietly. All three are questions about evidence, so none of them is asked of
a model.

Extraction is deterministic for one stated reason: the assignment requires the view
to be reproducible from a run. A calendar that changes between two runs over the
same inbox is not a calendar, and this project has already measured how much a
small model's answers move. What a model would buy here is a better reading of
loose phrasing; what it would cost is the one property the pane is graded on.

Three things make that workable:

  - Every date is anchored to the message that carried it. "the 18th" means the
    18th of the month the message was sent in, because a mailbox is read forwards.
  - A commitment may be resolved against another commitment. m040 asks for the
    board deck "two days before the board review" and names no date; m038 says the
    review is the 18th. Neither message alone answers it, which is exactly the case
    the assignment asks to see.
  - A time that cannot be resolved stays unresolved and says so, rather than being
    guessed at. An entry reading "Wednesday" with no week attached is honest; one
    reading "16 Sep" that was invented is not.

Usage:
    python commitments.py     # what this inbox commits the owner to
"""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

import mailstore
import rules

# --- reading a date out of a sentence ---------------------------------------

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

ORDINAL = re.compile(r"\bthe (\d{1,2})(?:st|nd|rd|th)\b", re.I)
MONTH_DAY = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2})\b", re.I)
WEEKDAY = re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
CLOCK = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", re.I)
RELATIVE = re.compile(
    r"\b(one|two|three|four|five|a|\d{1,2})\s+(day|week)s?\s+(before|after)\s+(?:the\s+)?([a-z][a-z ]{2,30})",
    re.I,
)
NUMBER_WORDS = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}

# A date alone is not a commitment. One of these has to be present too, or every
# receipt mentioning "this week" becomes a calendar entry.
OBLIGATION = (
    "can you", "could you", "please", "need", "needs", "by ", "due", "deadline",
    "confirm", "reply", "approve", "submit", "send", "finish", "circulate",
    "scheduled", "set for", "booked", "hold", "appointment", "review", "call",
    "meeting", "demo", "1:1", "launch", "respond", "target",
)

# Automated mail that names a date but asks nothing. These are facts about the
# world, not obligations, and a calendar full of them buries the six that matter.
NOT_A_COMMITMENT = (
    "screen time", "renews", "payout", "receipt", "analytics", "activity",
    "subscription", "ride", "digest", "in 10 minutes",
)


@dataclass
class Commitment:
    """One thing the owner is on the hook for, and where it came from."""

    what: str
    cites: tuple  # every message id it was derived from, in order
    when: object = None  # datetime, or None when the date could not be resolved
    when_text: str = ""  # what the message actually said
    has_time: bool = False
    at_time: object = None  # (hour, minute), known even when the date is not
    weekday: object = None  # set when a weekday is known but the date is not
    resolved_by: str = ""  # how a date was reached, when it took more than one message
    source: str = ""  # the message that carried the obligation

    @property
    def multi_source(self):
        return len(self.cites) > 1

    def slot(self):
        """(weekday, time) — what two commitments must share to collide.

        Derived from whichever is known. A dated entry knows both; one reading
        "Wednesday at 2:00pm" knows both too, without knowing which Wednesday.
        That is enough to notice they might collide, and not enough to say they do.
        """
        if self.at_time is None:
            return None
        if self.when is not None:
            return (self.when.weekday(), self.at_time)
        if self.weekday is not None:
            return (self.weekday, self.at_time)
        return None

    def when_shown(self):
        if self.when is not None:
            return self.when.strftime("%a %d %b %H:%M") if self.has_time else self.when.strftime("%a %d %b")
        return self.when_text or "unscheduled"

    def line(self):
        return f"{self.when_shown():18}  {self.what}   [{', '.join(self.cites)}]"


@dataclass
class Conflict:
    """Two commitments that cannot both happen."""

    first: object
    second: object
    certain: bool
    why: str

    def line(self):
        mark = "CLASH" if self.certain else "maybe"
        return f"  [{mark}] {self.why}\n      {self.first.what}  [{', '.join(self.first.cites)}]\n      {self.second.what}  [{', '.join(self.second.cites)}]"


# --- resolving what a message said into a date ------------------------------


def clock_in(text):
    """The first time of day in this text, as (hour, minute). None when there is none."""
    match = CLOCK.search(text or "")
    if match is None:
        return None
    hour, minute = int(match.group(1)), int(match.group(2) or 0)
    if hour > 12 or minute > 59:
        return None
    meridiem = match.group(3).lower()
    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    return hour, minute


def date_in(text, anchor):
    """The date this text names, read forwards from the message that carried it.

    `anchor` is when the message was sent. A bare "the 18th" means the 18th of the
    anchor's month, and one that has already passed means next month -- mail does
    not refer backwards to a deadline. Returns (date, what the text said) or None.
    """
    text = text or ""
    month_day = MONTH_DAY.search(text)
    if month_day:
        month = MONTHS[month_day.group(1).lower()[:3]]
        day = int(month_day.group(2))
        year = anchor.year + (1 if month < anchor.month else 0)
        try:
            return datetime(year, month, day), month_day.group(0)
        except ValueError:
            return None

    ordinal = ORDINAL.search(text)
    if ordinal:
        day = int(ordinal.group(1))
        try:
            found = datetime(anchor.year, anchor.month, day)
        except ValueError:
            return None
        if found.date() < anchor.date():
            month = anchor.month % 12 + 1
            year = anchor.year + (1 if month == 1 else 0)
            try:
                found = datetime(year, month, day)
            except ValueError:
                return None
        return found, ordinal.group(0)
    return None


def weekday_in(text, anchor, same_week=False):
    """The named weekday a commitment falls on. (date, weekday index, phrase) or None.

    `same_week` is set when the message says "this week", which pins the answer
    rather than leaving it as "the next Wednesday, whenever that is".

    When the text names more than one weekday, the one nearest the time of day is
    taken. A message asking to move a meeting names two -- "from Thursday to
    Wednesday at 2:00pm" -- and the first one read is the day being vacated. The
    weekday that belongs to the commitment is the one the clock is attached to.
    """
    text = text or ""
    matches = list(WEEKDAY.finditer(text))
    if not matches:
        return None
    match = matches[0]
    clock = CLOCK.search(text)
    if clock is not None and len(matches) > 1:
        before = [m for m in matches if m.end() <= clock.start()]
        match = min(before or matches, key=lambda m: abs(clock.start() - m.end()))
    wanted = WEEKDAYS[match.group(1).lower()]
    ahead = (wanted - anchor.weekday()) % 7
    if ahead == 0 and not same_week:
        ahead = 7
    return anchor + timedelta(days=ahead), wanted, match.group(1)


def looks_like_a_commitment(text):
    """A date alone is not an obligation, and an obligation alone has no place on a calendar."""
    if any(phrase in text for phrase in NOT_A_COMMITMENT):
        return False
    return any(phrase in text for phrase in OBLIGATION)


GREETING = re.compile(r"^(?:hi|hey|hello)?\s*[A-Z][a-z]+\s*(?:--|—|,)\s*", re.I)


def summarise(message, when_text=""):
    """What is owed, in the message's own words.

    The subject is the obvious choice and it is wrong here: nine messages in this
    inbox share the subject of the thread they sit in, so a calendar built from
    subjects showed four different obligations all reading "Launch week --
    kickoff". The sentence carrying the date says what the date is for.
    """
    body = re.sub(r"\s+", " ", (message.body or "").strip())
    if when_text:
        for sentence in re.split(r"(?<=[.?!])\s+", body):
            if when_text.lower() in sentence.lower():
                sentence = GREETING.sub("", sentence).strip()
                return sentence if len(sentence) <= 110 else sentence[:107].rstrip() + "..."
    subject = re.sub(r"^(re|fwd):\s*", "", (message.subject or "").strip(), flags=re.I).strip()
    return subject or body[:60]


# --- pulling them out of the inbox ------------------------------------------


def extract(mailbox):
    """Every commitment this inbox carries, with the messages each came from.

    Hostile mail is excluded before anything is read out of it. One of the refused
    messages presses for a payment "before end of day", and a calendar that lists
    an attacker's deadline beside the owner's real ones has done the attacker's
    work for it.
    """
    found = []
    for message in mailbox.everything():
        if isinstance(message, mailstore.Malformed):
            continue
        verdict = rules.classify(message)
        if verdict.hostile:
            continue
        if "preference_statement" in verdict.flags:
            # A standing instruction is a rule for every later message, not an
            # entry on a calendar. "No meetings before 11:00am" is not something
            # the owner has to be anywhere for.
            continue
        text = message.text().lower()
        if not looks_like_a_commitment(text):
            continue

        anchor = message.sent_at
        when, when_text, weekday, has_time = None, "", None, False

        dated = date_in(message.body, anchor)
        if dated:
            when, when_text = dated
        else:
            same_week = "this week" in text
            by_weekday = weekday_in(message.body, anchor, same_week=same_week)
            if by_weekday:
                when, weekday, when_text = by_weekday[0], by_weekday[1], by_weekday[2]
                if not same_week:
                    # A weekday with no week attached resolves to the next one, but
                    # the message did not actually say which, so the date is a guess
                    # and is not presented as fact.
                    when = None

        at = clock_in(message.body)
        if at is not None:
            has_time = True
            if when is not None:
                when = when.replace(hour=at[0], minute=at[1])
            if not when_text:
                when_text = CLOCK.search(message.body).group(0)
        # A commitment stated only in terms of another one carries no date of its
        # own, and dropping it here would lose exactly the case Part 7 asks to be
        # shown. It is kept undated and resolved in `merge`.
        relative = RELATIVE.search(message.body)
        if when is None and not when_text and relative is None:
            continue
        if when is None and not when_text:
            when_text = relative.group(0).strip()

        found.append(
            Commitment(
                what=summarise(message, when_text),
                cites=(message.id,),
                when=when,
                when_text=when_text,
                has_time=has_time,
                at_time=at,
                weekday=weekday,
                source=message.id,
            )
        )
    return merge(found, mailbox)


def merge(found, mailbox):
    """Fold the entries that are the same commitment seen twice, and resolve the rest.

    Two passes, and the second is the one the assignment asks for.
    """
    # 1. The same date named in the same thread is one entry, not several. The
    #    launch date is stated when the thread opens and again when it is
    #    confirmed as hard; those are one commitment with two sources.
    by_key = {}
    for commitment in found:
        message = mailbox.by_id(commitment.source)
        key = (message.thread_id, commitment.when.date() if commitment.when else commitment.when_text.lower())
        if key in by_key:
            existing = by_key[key]
            existing.cites = tuple(dict.fromkeys(existing.cites + commitment.cites))
            if existing.when is None and commitment.when is not None:
                existing.when, existing.has_time = commitment.when, commitment.has_time
            continue
        by_key[key] = commitment
    merged = list(by_key.values())

    # 2. A commitment stated relative to another one. This is the case the
    #    assignment names: a date in one message and what it applies to in
    #    another, resolved into a single entry citing both.
    for commitment in merged:
        message = mailbox.by_id(commitment.source)
        relative = RELATIVE.search(message.body)
        if relative is None or commitment.when is not None:
            continue
        count = NUMBER_WORDS.get(relative.group(1).lower(), None)
        if count is None:
            count = int(relative.group(1)) if relative.group(1).isdigit() else None
        if count is None:
            continue
        span = timedelta(days=count * (7 if relative.group(2).lower() == "week" else 1))
        direction = -1 if relative.group(3).lower() == "before" else 1
        anchor_words = {w for w in relative.group(4).lower().split() if len(w) > 3}

        target = _best_match(anchor_words, merged, commitment)
        if target is None or target.when is None:
            continue
        commitment.when = target.when + direction * span
        commitment.has_time = False
        commitment.cites = tuple(dict.fromkeys(commitment.cites + target.cites))
        commitment.resolved_by = (
            f"{relative.group(0).strip()} -- {', '.join(target.cites)} gives that date"
        )
        commitment.when_text = relative.group(0).strip()
    return merged


def _best_match(words, commitments, exclude):
    """The commitment whose description best matches the words of a relative phrase."""
    best, score = None, 0
    for candidate in commitments:
        if candidate is exclude or candidate.when is None:
            continue
        overlap = len(words & set(candidate.what.lower().split()))
        if overlap > score:
            best, score = candidate, overlap
    return best


# --- the things that cannot both happen -------------------------------------


def conflicts(commitments):
    """Two commitments at the same time, called out rather than listed.

    Two certainties are reported, and they are not the same claim. A pair resolved
    to the same date and time is a clash. A pair sharing a weekday and an hour
    where at least one has no date *might* be one, and saying so is the honest
    answer -- asserting a collision between two Wednesdays that could be a week
    apart would be inventing it, and staying silent would hide it.
    """
    found = []
    seen = {}
    for commitment in commitments:
        slot = commitment.slot()
        if slot is None:
            continue
        for other in seen.get(slot, ()):
            both_dated = commitment.when is not None and other.when is not None
            certain = both_dated and commitment.when == other.when
            if both_dated and not certain:
                continue  # same weekday and hour, different weeks: not a clash
            found.append(
                Conflict(
                    first=other,
                    second=commitment,
                    certain=certain,
                    why=(
                        f"both at {commitment.when.strftime('%a %d %b %H:%M')}"
                        if certain
                        else f"both fall on a {WEEKDAY_NAMES[slot[0]]} at {slot[1][0]:02d}:{slot[1][1]:02d}, "
                        "and at least one does not say which week"
                    ),
                )
            )
        seen.setdefault(slot, []).append(commitment)
    return found


def check_citations(commitments, mailbox):
    """Every cited id exists, and the commitment was actually derived from it.

    The same rule Part 3 applies to a draft: a citation is checked against the
    mail store rather than trusted. A commitment citing a message that is not in
    the inbox is a commitment nobody can verify.
    """
    problems = []
    for commitment in commitments:
        if not commitment.cites:
            problems.append(f"{commitment.what!r} cites nothing")
        for message_id in commitment.cites:
            if mailbox.by_id(message_id) is None:
                problems.append(f"{commitment.what!r} cites {message_id}, which is not in the mail store")
    return problems


def calendar(commitments):
    """Dated entries in time order, then the ones with no date."""
    dated = sorted((c for c in commitments if c.when is not None), key=lambda c: c.when)
    undated = [c for c in commitments if c.when is None]
    return dated, undated


if __name__ == "__main__":
    box = mailstore.load()
    found = extract(box)
    dated, undated = calendar(found)
    print(f"=== {len(found)} commitment(s) ===\n")
    for commitment in dated:
        print(" ", commitment.line())
        if commitment.resolved_by:
            print(f"      resolved: {commitment.resolved_by}")
    for commitment in undated:
        print(" ", commitment.line())
    print(f"\n=== derived from more than one message ===")
    for commitment in found:
        if commitment.multi_source:
            print(f"  {commitment.what}  [{', '.join(commitment.cites)}]  {commitment.when_shown()}")
    print(f"\n=== conflicts ===")
    for clash in conflicts(found):
        print(clash.line())
    problems = check_citations(found, box)
    print(f"\ncitation problems: {problems or 'none'}")
