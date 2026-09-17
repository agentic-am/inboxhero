"""The deterministic tier: what the system decides without a language model.

Part 2 asks us to "route the obvious messages through rules, not an LLM, and
report how many never required a model call". This module is that router.

`classify()` returns a `RuleVerdict` for every message. Three outcomes:

  - a `disposition` is set  -> decided here, no model call, no tokens spent
  - `disposition` is None   -> the model decides, but not freely: the verdict
                               still carries `flags` and a narrowed `allowed`
                               set, both of which go into the prompt and are
                               re-checked after the model answers
  - the message is hostile  -> forced `flag`, and the model never sees it as
                               a task at all

That middle case is the point. A stage that runs before a model call and does
not change what the model may do is decoration. Here, abstaining
is not the same as having no opinion: a rule that cannot pick a disposition can
still remove `reply` from the menu, and `guard.py` enforces that afterwards.

Usage:
    python rules.py     # classify the real inbox and print the tally
"""

import re
from dataclasses import dataclass

import config
import mailstore

DISPOSITIONS = ("reply", "archive", "defer", "delegate", "escalate", "flag")

# Sender local-parts that belong to a machine. Mail from these is never
# answered by a human, so `reply` is removed from the menu even when the
# message goes to the model.
AUTOMATED_LOCALPARTS = frozenset(
    """
    no-reply noreply no_reply donotreply do-not-reply do_not_reply mailer-daemon postmaster bounce bounces
    notification notifications notify alerts alert reminder reminders
    receipt receipts invoice invoices billing statements orders order
    newsletter digest news updates update insights
    status monitoring checkin check-in appointments calendar-notification
    feedback info hello help support success security accounts
    ship-confirm shipment tracking
    """.split()
)

# Phrases an automated message uses to say "nothing is expected of you".
NO_ACTION_PHRASES = (
    "no action is needed",
    "no action needed",
    "no further action",
    "this is an automated",
    "automated receipt",
    "please do not reply",
    "do not reply",
    "for your records",
    "this is a receipt",
    "no action required",
)

# --- hostile content ------------------------------------------------------
# Four independent families. A single family is a hint; the combination is
# what forces a flag. Section 5.4 of the design doc explains why: a legitimate
# standing instruction (m041) is also addressed to the assistant, so "speaks to
# the assistant" cannot be the test on its own.

ADDRESSED_TO_AGENT = (
    "automated-agent directive",
    "automated agent",
    "automated assistant",
    "automated assistants",
    "ai agent",
    "ai assistant",
    "assistant managing this inbox",
    "assistant managing this mailbox",
    "assistant note",
    "assistant configuration",
    "agent is processing",
    "if an ai",
)

# Asking the system to hide something from its owner. The single strongest
# signal we have: no honest correspondent needs the assistant to keep a secret.
CONCEALMENT = (
    "do not mention",
    "don't mention",
    "no need to mention",
    "without mentioning",
    "not surface",
    "no need to surface",
    "do not surface",
    "not to surface",
    "do not tell",
    "keep this between us",
    "keep it between us",
    "don't loop in",
    "do not loop in",
    "without telling",
    "do not inform",
)

# Trying to replace the system's own instructions.
OVERRIDE = (
    "ignore all previous",
    "ignore previous instruction",
    "ignore any previous",
    "disregard previous",
    "disregard all previous",
    "disregard any previous",
    "override your instructions",
    "system notice for automated",
    "new instructions:",
)

# Trying to widen what the system may do on its own. The mirror image of a
# legitimate preference, which can only ever narrow it.
WIDENING = (
    "autonomous mode",
    "without asking for approval",
    "without approval",
    "without asking",
    "skip the confirmation",
    "skip confirmation",
    "skip the approval",
    "skip approval",
    "automatically without",
    "send all drafted",
    "enable autonomous",
)

# Trying to move mail somewhere the owner did not ask for.
EXFILTRATION = (
    "forward the full contents",
    "forward the contents",
    "forward any message",
    "forward every message",
    "forward all mail",
    "forward all messages",
    "send the full contents",
    "forward this message to",
    "so our records stay aligned",
)

# --- fraud ----------------------------------------------------------------

MONEY = (
    "wire",
    "remit",
    "remittance",
    "bank transfer",
    "routing",
    "account number",
    "banking partner",
    "new account",
    "payment details",
    "deposit",
    "invoice",
)

CREDENTIALS = (
    "password",
    "credentials",
    "re-verify",
    "reverify",
    "verify your account",
    "sign in to confirm",
    "login",
    "log in to confirm",
    "one-time code",
    "expires in",
)

URGENCY = (
    "urgent",
    "immediately",
    "before end of day",
    "end of day",
    "within 24 hours",
    "expires in 2 hours",
    "today or",
    "avoid a service interruption",
    "will be suspended",
    "lose access",
    "action required",
    "last chance",
)

# Subjects where quietly doing nothing is the dangerous outcome. A message that
# touches one of these may not be archived by the model: `archive` is removed
# from its allowed list, so the worst case is an unnecessary escalation rather
# than a signature, a payment or a press deadline that nobody ever saw.
SENSITIVE = (
    "sign", "signature", "contract", "amendment", "safe", "term sheet", "clause",
    "invoice", "payment", "wire", "remit", "deposit", "$",
    "credential", "password", "creds", "api key", "token", "secret",
    "legal", "lawyer", "counsel", "board", "press", "embargo", "journalist", "coverage",
    "offer", "salary", "resign", "notice period",
)

# --- preferences ----------------------------------------------------------

PREFERENCE_PHRASES = (
    "from now on",
    "standing request",
    "standing instruction",
    "please remember",
    "for future",
    "going forward",
    "i never",
    "i do not take",
    "always cc",
    "make sure i'm cc",
    "make sure i am cc",
)


@dataclass(frozen=True)
class RuleVerdict:
    """What the deterministic tier concluded about one message.

    `disposition` None means "the model decides", but `allowed` still limits
    what it may decide, and `flags` still reach the prompt.
    """

    message_id: str
    disposition: str | None
    reason: str
    rule: str
    flags: tuple = ()
    allowed: tuple = DISPOSITIONS
    attempted: str | None = None  # what a hostile message asked for

    @property
    def handled(self):
        """True when no model call is needed for this message."""
        return self.disposition is not None

    @property
    def hostile(self):
        return self.disposition == "flag"


def _hits(haystack, needles):
    return tuple(n for n in needles if n in haystack)


def _word_hits(haystack, needles):
    """Like `_hits`, but a needle must be a whole word.

    Plain substring matching reads "design" as "sign" and "signup" as "sign",
    which is how three launch-thread updates first came out as legal matters.
    Needles that are not words, such as "$", still match anywhere.
    """
    found = []
    for needle in needles:
        pattern = re.escape(needle)
        if needle[0].isalnum():
            pattern = r"\b" + pattern
        if needle[-1].isalnum():
            pattern = pattern + r"\b"
        if re.search(pattern, haystack):
            found.append(needle)
    return tuple(found)


def _localpart(address):
    return address.partition("@")[0].lower()


def _owner_domain():
    return config.OWNER.rpartition("@")[2].lower()


def looks_like_owner_domain(domain):
    """A domain that imitates the owner's without being it: paperjet.co, paperjet-helpdesk.com.

    The owner's brand appears in the domain, but the domain is not theirs.
    Used only together with a second signal, so the board's legitimate
    paperjet-board.org is not condemned for sharing the brand name.
    """
    owner = _owner_domain()
    if domain == owner:
        return False
    brand = owner.partition(".")[0]
    return bool(brand) and brand in domain


def is_automated(message):
    """Mail from a machine: nobody is waiting for an answer."""
    if _localpart(message.sender) in AUTOMATED_LOCALPARTS:
        return True
    local = _localpart(message.sender)
    if any(local.startswith(prefix) for prefix in ("no-reply", "noreply", "no_reply", "notification", "receipt", "invoice")):
        return True
    return any(phrase in message.body.lower() for phrase in ("this is an automated", "please do not reply", "do not reply to this"))


def _hostile(message, text):
    """Return (reason, rule, attempted) when the body targets the assistant, else None."""
    agent = _hits(text, ADDRESSED_TO_AGENT)
    conceal = _hits(text, CONCEALMENT)
    override = _hits(text, OVERRIDE)
    widen = _hits(text, WIDENING)
    exfil = _hits(text, EXFILTRATION)

    # An instruction aimed at the assistant is only hostile when it also tries
    # to hide, to override, to widen autonomy, or to move mail out. m041 is
    # addressed to the assistant and does none of these, so it is left alone.
    intent = []
    if exfil:
        intent.append("move mail to an address the owner never chose")
    if widen:
        intent.append("remove the approval step")
    if override:
        intent.append("replace the system's own instructions")
    if conceal:
        intent.append("hide the request from the owner")

    if not intent:
        return None
    if not (agent or override):
        return None  # a human asking a human for discretion is not an injection

    evidence = ", ".join(repr(h) for h in (agent + override + widen + exfil + conceal)[:3])
    return (
        f"the body carries an instruction addressed to the assistant that would {intent[0]}",
        "injection",
        "; ".join(intent) + f" (matched {evidence})",
    )


def _fraud(message, text):
    """Return (reason, rule, attempted) when the message looks like fraud, else None."""
    money = _hits(text, MONEY)
    creds = _hits(text, CREDENTIALS)
    urgent = _hits(text, URGENCY)
    secret = _hits(text, CONCEALMENT)
    lookalike = looks_like_owner_domain(message.sender_domain)
    has_link = "http://" in text or "https://" in text

    if lookalike and (money or creds or urgent or secret):
        what = "a payment" if money else "credentials" if creds else "an urgent action"
        return (
            f"sender domain {message.sender_domain!r} imitates {_owner_domain()!r} and the message asks for {what}",
            "lookalike-domain",
            f"obtain {what} by impersonating an internal address",
        )
    if money and urgent and ("new account" in text or "banking partner" in text or "disregard the account" in text):
        return (
            "asks to redirect payment to new bank details under time pressure",
            "payment-redirect",
            "redirect an outstanding payment to an attacker-controlled account",
        )
    if creds and urgent and has_link:
        return (
            "asks for credentials through a link under threat of losing access",
            "credential-harvest",
            "collect the owner's password through a lookalike login page",
        )
    if secret and money:
        return (
            "asks for a payment while asking that it be kept from colleagues",
            "secrecy-plus-money",
            "obtain a payment without the usual internal checks",
        )
    return None


_DATETIME = re.compile(
    r"\b(?:\d{1,2}:\d{2}\s*(?:am|pm)?|\d{1,2}\s*(?:am|pm))\b|\b(?:mon|tues?|wed(?:nes)?|thur?s?|fri|sat(?:ur)?|sun)(?:day)?\b",
    re.IGNORECASE,
)
_DATE_WORDS = re.compile(
    r"\b(?:the\s+\d{1,2}(?:st|nd|rd|th)|\d{1,2}(?:st|nd|rd|th)\b|by\s+(?:friday|monday|tuesday|wednesday|thursday)|"
    r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b",
    re.IGNORECASE,
)

_ACTION_REQUESTS = (
    "reply confirm",
    "reply to confirm",
    "reply urgent",
    "please confirm",
    "confirm or",
    "rsvp",
    "let us know",
    "reply with",
)


def _asks_for_action(message, text):
    """True when an automated message still wants something back from the owner."""
    if any(phrase in text for phrase in NO_ACTION_PHRASES):
        return False
    return any(phrase in text for phrase in _ACTION_REQUESTS) and bool(_DATETIME.search(message.body))


def _preference_statement(message, text):
    """True when the owner's side states a standing rule."""
    if not message.internal:
        return False
    return any(phrase in text for phrase in PREFERENCE_PHRASES)


def classify(record):
    """The rule tier's verdict for one record. Never raises."""
    if isinstance(record, mailstore.Malformed):
        return RuleVerdict(
            message_id=record.id,
            disposition="escalate",
            reason=f"the record could not be read ({record.problem}), so a human should look at it",
            rule="malformed",
            flags=("malformed",),
            allowed=("escalate",),
        )

    text = record.text().lower()
    flags = []

    # 1. Hostile content first, before anything can mark it as harmless noise.
    #    m047 arrives from a support address and would otherwise look routine.
    hostile = _hostile(record, text)
    if hostile:
        reason, rule, attempted = hostile
        return RuleVerdict(record.id, "flag", reason, rule, ("hostile", "injection"), ("flag",), attempted)

    fraud = _fraud(record, text)
    if fraud:
        reason, rule, attempted = fraud
        return RuleVerdict(record.id, "flag", reason, rule, ("hostile", "fraud"), ("flag",), attempted)

    # 2. Noise. Automated mail that asks nothing of the owner.
    automated = is_automated(record)
    if automated:
        flags.append("automated_sender")
        asks_for_action = _asks_for_action(record, text)
        if not asks_for_action:
            return RuleVerdict(
                record.id,
                "archive",
                "automated mail from a no-reply address that asks nothing of the owner",
                "noise",
                ("automated_sender",),
                ("archive",),
            )
        # An automated message that does want something (the dentist asking for
        # CONFIRM) is a calendar item, not noise. It is deferred rather than
        # archived because the appointment still needs the owner on the day.
        return RuleVerdict(
            record.id,
            "defer",
            "automated mail that names an appointment and asks for a confirmation, so it belongs on the calendar",
            "automated-appointment",
            ("automated_sender", "has_datetime"),
            ("defer",),
        )

    # 3. Everything else goes to the model, with the menu narrowed.
    if record.from_owner:
        flags.append("from_owner_address")
    if record.internal:
        flags.append("internal_sender")
    else:
        flags.append("external_sender")
    if _preference_statement(record, text):
        flags.append("preference_statement")
    if _DATETIME.search(record.body) or _DATE_WORDS.search(text):
        flags.append("has_datetime")

    sensitive = _word_hits(text, SENSITIVE)
    if sensitive:
        flags.append("sensitive_topic")

    allowed = [d for d in DISPOSITIONS if d != "flag"]  # only rules and the validator may flag
    if sensitive:
        # Doing nothing is the failure mode that costs the owner a signature or a
        # payment, so the model is not offered the option of doing nothing.
        allowed = [d for d in allowed if d != "archive"]
    if record.to.lower() != config.OWNER:
        # Mail the owner sent to somebody else. Replying to it would mean
        # replying to ourselves, so that option is removed before the model sees it.
        flags.append("sent_by_owner_to_someone_else")
        allowed = [d for d in allowed if d != "reply"]

    return RuleVerdict(
        record.id,
        None,
        "no rule applies with confidence, so the model decides",
        "abstain",
        tuple(flags),
        tuple(allowed),
    )


def classify_all(records):
    return [classify(record) for record in records]


def tally(verdicts):
    """Counts for the run summary and for the manifest's `rule_handled` field."""
    counts = {}
    for verdict in verdicts:
        key = verdict.disposition or "(to the model)"
        counts[key] = counts.get(key, 0) + 1
    handled = sum(1 for v in verdicts if v.handled)
    return {
        "records": len(verdicts),
        "rule_handled": handled,
        "to_the_model": len(verdicts) - handled,
        "by_disposition": counts,
        "by_rule": _count(v.rule for v in verdicts),
    }


def _count(values):
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    box = mailstore.load()
    records = box.everything()
    verdicts = classify_all(records)
    numbers = tally(verdicts)

    print("=== rule tier over the whole inbox ===")
    for key in ("records", "rule_handled", "to_the_model"):
        print(f"  {key:16} {numbers[key]}")
    print(f"  by disposition   {numbers['by_disposition']}")
    print(f"  by rule          {numbers['by_rule']}")

    print("\n  flagged as hostile:")
    for verdict in verdicts:
        if verdict.hostile:
            print(f"    {verdict.message_id}  [{verdict.rule}] {verdict.reason}")

    print("\n  going to the model:")
    for verdict in verdicts:
        if not verdict.handled:
            print(f"    {verdict.message_id}  allowed={list(verdict.allowed)}  flags={list(verdict.flags)}")
