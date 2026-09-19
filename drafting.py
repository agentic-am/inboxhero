"""Part 3: drafting a reply that is grounded in a specific earlier message.

Retrieval (`retrieval.py`) finds what the inbox already says. This module turns
that into an answer, under three rules the spec states plainly and one the inbox
forces on us:

    1. a draft cites the message ids it drew on, checked against the mail store
    2. a detail that appears nowhere in the inbox fails the draft
    3. no evidence means no draft, and the system says so
    4. a draft never carries a credential, even a correctly cited one

The fourth is ours. m008 asks for the staging credentials that m003 contains,
and the honest grounded answer to it is "that is in m003, and I am not putting
it in an email" -- citing the message without quoting the secret out of it.
Mail is the one channel the owner already knows is readable by strangers, and a
correctly cited credential is still a leaked credential.

Every one of these is enforced here, in Python, after the model has answered.
The prompt asks; this decides.

Usage:
    python drafting.py       # draft for the messages triage chose; nothing is sent
"""

import dataclasses
import re
from dataclasses import dataclass

import config
import provider
import trace

# A draft is a reply, not an essay. Long drafts are usually the model padding
# around a fact it does not actually have.
MAX_DRAFT_CHARS = 1200

# Specifics: the parts of a draft that claim to be facts rather than courtesy.
# Each one has to be traceable to a cited message or to the message being
# answered, because these are exactly what a model invents when it is guessing.
URL = re.compile(r"\b[a-z][a-z0-9+.-]*://\S+", re.I)
REFERENCE = re.compile(r"\b(?:[A-Z]{2,}-\d+|#\d{3,})\b")
TIME = re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b", re.I)
DATE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}(?:st|nd|rd|th)|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2})\b", re.I)
MONEY = re.compile(r"[$€£]\s?[\d,]+(?:\.\d{2})?")
ADDRESS = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b")
# Durations and proportions are claims too, and the first pass did not look for
# them: a draft offering "a 30-minute call" or reporting "80% done" is stating a
# number about the world exactly as much as one naming a date.
DURATION = re.compile(r"\b\d{1,3}\s*-?\s*(?:minute|min|hour|hr|day|week|month)s?\b", re.I)
PERCENT = re.compile(r"\b\d{1,3}(?:\.\d+)?\s*%")

SPECIFIC_PATTERNS = (URL, REFERENCE, TIME, DATE, MONEY, ADDRESS, DURATION, PERCENT)

# Our own identifiers. They belong in `cites`, where the system reads them, and
# never in the sentence, where the recipient does: a reply telling Devika the
# URL is "in m003" names something that exists only inside this program. She has
# no m003. The model reaches for it because the prompt is full of these ids, so
# the rule has to be enforced rather than only asked for.
MESSAGE_ID = re.compile(r"\bm\d{3}\b", re.I)

# Actions a message can ask for, and the word a draft uses to claim it is done.
# This is the failure no other check here can see, because nothing is invented:
# m048 asked Sam to *review* board minutes and *flag* corrections, and the draft
# told outside counsel that both had been done. Every word came from the
# message. Only the tense was false.
ACTION_VERBS = {
    "review": ("reviewed",),
    "sign": ("signed",),
    "flag": ("flagged",),
    "send": ("sent",),
    "resend": ("resent",),
    "submit": ("submitted",),
    "approve": ("approved",),
    "confirm": ("confirmed",),
    "update": ("updated",),
    "pay": ("paid",),
    "book": ("booked",),
    "schedule": ("scheduled",),
    "cancel": ("cancelled", "canceled"),
    "forward": ("forwarded",),
    "circulate": ("circulated",),
    "complete": ("completed",),
    "fix": ("fixed",),
}

# "I have reviewed", "we've already signed". Deliberately a perfect-tense
# construction rather than the two words merely being near each other: English
# uses these past forms as adjectives constantly, and "I have reviewed the
# *updated* IP assignment" would otherwise read as a claim to have updated
# something. At most one word may sit between, which admits "I have already
# signed" and excludes the adjective case.
#
# The subject has to be the owner. Reporting that somebody *else* finished
# something is a different claim, and usually a true one drawn from evidence.
CLAIMED_DONE = r"\b(?:i|we)\s*(?:'ve|have|already|just)\s+(?:\w+\s+){{0,1}}?{past}\b"

# Secrets. A URL carrying userinfo (`scheme://user:password@host`) is the shape
# m003 uses, and it is the shape that matters: the credential is inside a string
# that otherwise looks like an ordinary address.
USERINFO_URL = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s/@]+:[^\s/@]+@\S+", re.I)
SECRET_ASSIGNMENT = re.compile(
    r"\b(?:password|passwd|pwd|api[ _-]?key|secret|token|credential)s?\b\s*(?:is|=|:)\s*\S{6,}", re.I
)
SECRET_PATTERNS = (USERINFO_URL, SECRET_ASSIGNMENT)

# Echo detection. A draft may quote a fact -- "the 20th is a hard date" is six
# shared words and entirely correct -- but not reproduce a message. Ten
# consecutive words is past quoting, and so is a shorter run that accounts for
# three quarters of everything the draft says.
ECHO_RUN_WORDS = 10
ECHO_MIN_RUN = 6
ECHO_RUN_RATIO = 0.75

# A run of consecutive words is trivially broken. Asked to reply to m027
# ("Design assets are 80% there. Hero image still in review."), the model
# returned that sentence with the word "is" inserted -- which cut the longest
# run from eleven words to seven and slid it under both thresholds above, while
# leaving the draft a verbatim copy in every way a reader would care about. So
# order-preserving overlap is measured too, which an insertion does not disturb.
#
# Overlap alone is not enough to judge on, though, and the test suite said so:
# "The 20th is a hard date, and the press is briefed" draws nine of its eleven
# words from m036 and is a perfectly good reply that happens to quote a fact.
# What separates it from m027 is not how much of the *draft* came from the
# source but how much of the *source* the draft gave back -- the quote covers
# half of m036, m027 covers all of the message it answers. Reproducing a whole
# message is the failure; quoting a line out of one is the job.
ECHO_SUBSEQUENCE_RATIO = 0.8
ECHO_SOURCE_COVERAGE = 0.6
ECHO_SUBSEQUENCE_MIN_WORDS = 9

# The prompt is the third text in the model's context, after the message and the
# evidence, and the only one that is not mail. A phrase the draft shares with it
# and with no message came from the instructions rather than from the inbox.
# Six words is long enough to be a borrowed phrase rather than ordinary English
# two texts about the same subject would both contain.
LEAK_RUN_WORDS = 6


# Whether a message wants anything back. Deliberately generous: the cost of
# wrongly deciding a message asks nothing is a message quietly left unanswered,
# which is the failure this whole system exists to prevent. A question mark is
# the strongest signal and most requests carry one.
REQUEST_PHRASES = (
    "please",
    "can you",
    "could you",
    "would you",
    "are you able",
    "any chance",
    "let me know",
    "let us know",
    "get back to me",
    "send me",
    "send us",
    "need your",
    "needs your",
    "waiting on",
    "action required",
    "rsvp",
    "confirm",
    "by friday",
    "by monday",
    "asap",
)


def asks_anything(message):
    """Whether this message wants something back from the owner. No model call."""
    text = message.text().lower()
    if "?" in text:
        return True
    return any(phrase in text for phrase in REQUEST_PHRASES)


class DraftRejected(ValueError):
    """The draft cannot be used. The message says why, in one line."""


@dataclass(frozen=True)
class Draft:
    """A drafted reply, or a recorded refusal to draft one.

    `text` of None is a real outcome rather than a failure: it is what Part 3.4
    asks for when the inbox does not hold the answer, and `reason` then says so
    in words a person can read.
    """

    message_id: str
    text: str | None = None
    cites: tuple = ()
    reason: str = ""
    withheld: bool = False  # a fact was found, and deliberately not quoted
    # Why nothing was drafted, when nothing was. The first version of this
    # module had one answer for that -- "the inbox does not hold it" -- and so
    # forced a reply onto messages that simply do not want one, which is how a
    # status update in a thread came back with its own body as the draft.
    #   answered  : a draft was written
    #   no_reply  : the message asks for nothing; a reply would be noise
    #   not_known : the message asks something the inbox cannot answer
    #   rejected  : every attempt failed a check
    outcome: str = "answered"

    @property
    def drafted(self):
        return bool(self.text)

    @property
    def needs_no_reply(self):
        return self.outcome == "no_reply"

    def line(self):
        if not self.drafted:
            label = "no reply needed" if self.needs_no_reply else "no draft"
            return f"{self.message_id}: {label} -- {self.reason}"
        cited = f"  cited: [{', '.join(self.cites)}]" if self.cites else "  cited: []"
        return f"{self.message_id}:\n{self.text}\n{cited}"


def contains_secret(text):
    """Whether this text carries a credential, whoever it belongs to."""
    return any(pattern.search(text or "") for pattern in SECRET_PATTERNS)


def redact_secrets(text, holder=""):
    """Replace any credential in this text with a note saying where it lives.

    The post-hoc check below will reject a draft that carries a credential, and
    it does work -- a model copied m003's AMQP URL into the draft on both
    attempts and was refused both times. But refusing twice and giving up leaves
    the message with no draft at all, which is a worse answer than the correct
    one. So the credential does not reach the prompt: a model cannot copy out a
    string it was never shown, and the check stays as the second line rather
    than the only one. This is the difference between telling the model a rule
    and arranging for the rule to be unbreakable.
    """
    where = f" in {holder}" if holder else ""
    marker = f"[credential withheld{where}; cite the message, do not quote it]"
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(marker, text or "")
    return text


def longest_shared_run(draft, source):
    """The longest run of consecutive words the two texts have in common.

    Quoting a fact is fine and often right; reproducing a paragraph is not. A
    run length measures the difference, where a word-overlap ratio does not: a
    correct short reply and a copied one share most of their vocabulary either
    way.
    """
    left = re.findall(r"[a-z0-9']+", (draft or "").lower())
    right = re.findall(r"[a-z0-9']+", (source or "").lower())
    if not left or not right:
        return 0
    best = 0
    previous = [0] * (len(right) + 1)
    for i in range(1, len(left) + 1):
        current = [0] * (len(right) + 1)
        for j in range(1, len(right) + 1):
            if left[i - 1] == right[j - 1]:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


def longest_shared_subsequence(draft, source):
    """How many of the draft's words appear in the source, in order, with gaps.

    The complement of `longest_shared_run`: that one asks whether a passage was
    copied intact, this one whether it was copied and fiddled with. Reordering
    and rewording defeat both; inserting a word defeats only the first, and
    inserting a word is what actually happened.
    """
    left = re.findall(r"[a-z0-9']+", (draft or "").lower())
    right = re.findall(r"[a-z0-9']+", (source or "").lower())
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for i in range(1, len(left) + 1):
        current = [0] * (len(right) + 1)
        for j in range(1, len(right) + 1):
            if left[i - 1] == right[j - 1]:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current
    return previous[-1]


def echoes(draft, sources):
    """Whether this draft is a copy of one of its sources rather than a reply.

    A model returned m010's own body as the reply to m043 -- Aria's words
    played back to Aria as though Sam had written them -- and every other check
    passed it, because each fact in it genuinely was in a cited message. Being
    grounded is not the same as being written.
    """
    words = len(re.findall(r"[a-z0-9']+", (draft or "").lower()))
    if not words:
        return None
    for source in sources:
        run = longest_shared_run(draft, source)
        if run >= ECHO_RUN_WORDS or (run >= ECHO_MIN_RUN and run / words >= ECHO_RUN_RATIO):
            return run
        if words >= ECHO_SUBSEQUENCE_MIN_WORDS:
            shared = longest_shared_subsequence(draft, source)
            source_words = len(re.findall(r"[a-z0-9']+", (source or "").lower())) or 1
            if shared / words >= ECHO_SUBSEQUENCE_RATIO and shared / source_words >= ECHO_SOURCE_COVERAGE:
                return shared
    return None


def completion_claims(draft):
    """Every place the draft says the owner has already done something."""
    said = (draft or "").lower()
    found = []
    for verb, past_forms in ACTION_VERBS.items():
        for past in past_forms:
            if re.search(CLAIMED_DONE.format(past=past), said):
                found.append((verb, past))
                break
    return found


def unsupported_claims(draft, message, cited_text):
    """Completion claims with nothing in the mail behind them.

    "The work is done" is a statement of fact about the world, so it is checked
    the way a date is: it has to appear somewhere in the mail. "I have rotated
    the creds" is fine when a cited message says the creds were rotated. It is
    not fine when the message being answered is the *request* to rotate them,
    and it is the hardest failure to see, because every word of it is on topic
    and nothing has been invented -- only the tense is false.

    "I will sign it by Friday" is untouched. A commitment is not a lie, and
    whether the owner should be making one is a different question from whether
    what the draft says is true. This check answers only the second.
    """
    grounds = f"{cited_text}\n{message.text()}".lower()
    asked = message.text().lower()
    found = []
    for verb, past in completion_claims(draft):
        if past in grounds:
            continue
        found.append((verb, past, bool(re.search(rf"\b{verb}", asked))))
    return found


def specifics(text):
    """The factual claims in a draft, as raw strings."""
    found = []
    for pattern in SPECIFIC_PATTERNS:
        for value in pattern.findall(text or ""):
            value = value.strip().rstrip(".,;:)")
            if value and value not in found:
                found.append(value)
    return found


def build_draft_prompt(message, evidence, withhold_secret=False, owner=None):
    """Ask for a reply grounded in the evidence, and say what will be checked.

    The order of this prompt is load-bearing, which cost a run to learn. With
    the evidence block first, every model tried answered the *evidence* rather
    than the message, and one returned the same sentence for m019 and m046 --
    one a venue confirming a booking, the other a journalist asking whether a
    date is public. The message being replied to now comes
    first, the evidence follows as clearly-labelled background, and the message
    is named again at the end where the answer format is.

    The prompt also states every rule `parse_draft` enforces. Saying them makes
    a usable answer likelier; it is not what makes the answer safe, because the
    same rules are applied again to whatever comes back.
    """
    import agents  # local import: agents imports nothing from here, and this keeps it that way

    owner = owner or message.to
    lines = [
        f"You are writing as {owner}, replying to {message.sender}.",
        "",
        "REPLY TO THIS MESSAGE:",
        "",
        agents.quote_untrusted(message),
        "",
    ]

    if evidence:
        # Redacted before it is quoted: see `redact_secrets`.
        safe = [
            dataclasses.replace(item, snippet=redact_secrets(item.snippet, item.message_id))
            for item in evidence
        ]
        lines.append(
            "BACKGROUND -- earlier mail, for facts only. Do NOT reply to these, "
            "do not copy their wording, and do not answer the questions they ask."
        )
        lines.append("")
        lines.append(agents.quote_evidence(safe))
    else:
        lines.append(
            "No earlier mail was found for this message. That does not by itself "
            "prevent a reply, so decide which of these it is:"
        )
        lines.append("")
        lines.append(
            "  (a) It asks for nothing factual -- a standing request, an instruction, "
            "a note to acknowledge. Acknowledge it in your own words."
        )
        lines.append(
            "  (b) Everything it asks about is named in the message itself. Answer it "
            "from the message."
        )
        lines.append(
            "  (c) It refers to something you cannot identify -- \"that thing we "
            "discussed\", \"the item from our call\", a conversation you were not "
            "shown. You do not know what it is. Promising to handle it is a promise "
            "about nothing, and agreeing that it is in hand is worse. Answer with "
            "outcome 'not_known' and say what is missing."
        )
        lines.append("")

    lines.extend(
        [
            "Rules, all of which are checked after you answer:",
            "  - The background is there to be used. If it answers what the message is",
            "    asking, give that answer plainly and put the id it came from in \"cites\".",
            "    Saying you will check and come back, when the answer is already in front",
            "    of you, is the one outcome that helps nobody.",
            "  - Use only what the message and the background actually say.",
            '  - Put every message id you used in "cites". Citing anything else is rejected.',
            "  - Do not state a date, time, number, duration, address or URL that is not in them.",
            "  - Write your own sentences. Copying the message you are replying to, or the",
            "    background, is rejected -- and repeating the sender's own words back to them",
            "    is the most common way this fails.",
            "  - Do not say a task is done. This message is what asks for it, so 'I have",
            "    reviewed it' or 'I have signed it' is false however natural it reads.",
            "  - If the message asks for nothing -- a status update, an FYI, a note of thanks",
            '    -- answer with {"draft": null, "outcome": "no_reply", "reason": "..."}.',
            "  - If it asks for a fact that neither it nor the background contains, answer",
            '    with {"draft": null, "outcome": "not_known", "reason": "..."}.',
        ]
    )
    if withhold_secret:
        lines.append(
            "  - The background contains a credential and it has been withheld. Do NOT "
            "invent it. Say which message holds it and that it will be sent another way."
        )
    lines.append("")
    lines.append(f"Write the reply to {message.id} from {message.sender}, and nothing else.")
    lines.append(
        'Answer with only: {"draft": "...", "cites": ["m000"]}'
        '  or {"draft": null, "outcome": "no_reply", "reason": "..."}'
        '  or {"draft": null, "outcome": "not_known", "reason": "..."}'
    )
    return "\n".join(lines)


def instruction_leak(text, prompt):
    """Wording the draft took from its own instructions rather than from the mail.

    Asked what makes the product different -- a fact the inbox does not contain
    anywhere -- the model answered a journalist with "a hero assistant that
    clears one person's inbox", which is this system's own description of
    itself, lifted out of the sentence that tells it what it is. It is not an
    invented date, not a copy of any message, and not a false claim about work;
    every other check passes it clean, and it was on its way to a press contact
    as PaperJet's positioning.

    The prompt is the one text in the model's context that is neither the
    message nor the evidence, so anything the draft shares with it and with
    nothing else came from the wrong place. Short overlaps are ignored -- the
    prompt is full of ordinary English, and a reply is allowed to contain the
    word "launch" -- so this looks for a run long enough to be a borrowed
    phrase.
    """
    if not prompt:
        return 0
    run = longest_shared_run(text, prompt)
    return run if run >= LEAK_RUN_WORDS else 0


def parse_draft(raw, message, evidence, mailbox, prompt=""):
    """Check a drafted reply against the inbox. Raises `DraftRejected`.

    This is the Part 3 safety net, and it is deliberately the same shape as the
    disposition validator: whatever the model returns, the claims it makes are
    re-checked here against what it was actually shown.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise DraftRejected("the model returned nothing")
    if raw.lstrip().startswith("["):
        raise DraftRejected(f"the model output is an error string: {raw.strip()[:120]}")

    try:
        data = provider.parse_json(raw)
    except ValueError as error:
        raise DraftRejected(str(error)) from error

    evidence_ids = [item.message_id for item in evidence]
    text = data.get("draft")

    # "drafts nothing" is an answer, and it needs a reason a person can read.
    # It also needs to say *which* kind of nothing, because the two are not the
    # same decision and only one of them is about the inbox.
    if text is None or (isinstance(text, str) and not text.strip()):
        reason = data.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise DraftRejected("the model declined to draft but gave no reason")
        outcome = data.get("outcome", "not_known")
        if outcome not in ("no_reply", "not_known"):
            raise DraftRejected(f"'outcome' is {outcome!r}, expected 'no_reply' or 'not_known'")
        return Draft(message.id, None, (), " ".join(reason.split())[:300], outcome=outcome)

    if not isinstance(text, str):
        raise DraftRejected(f"'draft' is {type(text).__name__}, expected text or null")
    text = text.strip()
    if len(text) > MAX_DRAFT_CHARS:
        raise DraftRejected(f"the draft is {len(text)} characters, over the {MAX_DRAFT_CHARS} limit")

    cites = data.get("cites") or []
    if isinstance(cites, str):
        cites = [cites]
    if not isinstance(cites, list):
        raise DraftRejected(f"'cites' is {type(cites).__name__}, expected a list of message ids")

    checked = []
    for entry in cites:
        if not isinstance(entry, str):
            raise DraftRejected("'cites' must contain message ids as strings")
        entry = entry.strip()
        if not entry:
            continue
        if entry == message.id:
            # Both models reach for this, and they are not wrong to: the message
            # being answered is something the draft drew on, and it is certainly
            # in the mail store. It is dropped rather than rejected, because a
            # message is not evidence for itself and recording it as grounding
            # would make "cited: [m003]" mean less than it does.
            continue
        if entry not in evidence_ids:
            raise DraftRejected(
                f"cited {entry!r}, which was not in the evidence"
                + (f" ({', '.join(evidence_ids)})" if evidence_ids else " (nothing was retrieved)")
            )
        if mailbox.by_id(entry) is None:
            # The spec asks for citations checked against the mail store, not
            # merely against what we believe we showed the model.
            raise DraftRejected(f"cited {entry!r}, which is not in the mail store")
        if entry not in checked:
            checked.append(entry)

    cited_text = "\n".join((mailbox.by_id(mid).text() for mid in checked))
    own_text = message.text()
    grounds = f"{cited_text}\n{own_text}".lower()

    # A credential may be cited but never carried. This is the rule the prompt
    # asks for and the reason the check exists twice: asking is not enforcing.
    if contains_secret(text):
        raise DraftRejected("the draft carries a credential; cite the message that holds it instead")

    # The message being answered is a source a draft can be copied from, and the
    # first version of this check could not see it: with nothing cited it
    # compared the draft against an empty string and passed everything. Measured
    # on one full run, thirteen of twenty-four drafts lifted six or more
    # consecutive words from the message they were replying to, and eight were
    # that message word for word -- a status update handed back to the person
    # who wrote it. It is always a source now, cited or not.
    run = echoes(text, [mailbox.by_id(mid).text() for mid in checked] + [own_text])
    if run:
        raise DraftRejected(
            f"the draft copies {run} consecutive words out of the mail it is answering "
            "or citing; write a reply instead"
        )

    # Wording lifted out of the instructions instead of the mail.
    leaked = instruction_leak(text, prompt)
    if leaked:
        raise DraftRejected(
            f"the draft takes {leaked} consecutive words from its own instructions, "
            "which are not something the inbox says; answer from the mail or say you cannot"
        )

    # An internal id printed where the recipient will read it.
    named = [value for value in MESSAGE_ID.findall(text) if mailbox.by_id(value.lower()) is not None]
    if named:
        raise DraftRejected(
            f"the draft prints the internal id {named[0]!r}, which means nothing to the recipient; "
            "put it in 'cites' and describe the message in words"
        )

    # Saying the work is already done, when the message is what asks for it.
    # Nothing is invented here and every earlier check passes it, which is what
    # makes it worth a check of its own.
    unsupported = unsupported_claims(text, message, cited_text)
    if unsupported:
        verb, past, was_asked = unsupported[0]
        why = (
            f"but this message is the request to {verb}"
            if was_asked
            else f"but nothing in the mail says the {verb} happened"
        )
        raise DraftRejected(f"the draft says {past!r}, {why}; do not report work as finished")

    # A fact that came out of the evidence has to name where it came from. A
    # fact already present in the message being answered does not -- quoting the
    # sender's own time back at them is how a reply confirms it.
    for value in specifics(text):
        lowered = value.lower()
        if lowered in own_text.lower() or lowered in cited_text.lower():
            continue
        borrowed = [
            item.message_id
            for item in evidence
            if item.message_id not in checked and lowered in (mailbox.by_id(item.message_id).text() or "").lower()
        ]
        if borrowed:
            raise DraftRejected(
                f"used {value!r} from {borrowed[0]}, which it did not cite; cite what you used"
            )

    invented = [value for value in specifics(text) if value.lower() not in grounds]
    if invented:
        raise DraftRejected(
            "the draft states something the inbox does not: " + ", ".join(repr(v) for v in invented[:3])
        )

    withheld = contains_secret(cited_text)
    return Draft(message.id, text, tuple(checked), "", withheld)


def draft(agent, message, evidence, mailbox, thread_id="default", retries=1):
    """Draft a reply, or record why none was drafted. Never raises.

    No evidence used to end the work here, on the reasoning that an ungrounded
    answer is a guess. That was too broad. A standing request ("from now on, CC
    me on anything from our lawyers") asks for no fact at all, and the honest
    reply to it -- "noted" -- needs nothing but the message itself. Refusing it
    conflated *I have no facts to answer with* and *no reply is possible*. The
    model is asked either way now, and told which situation it is in.
    """
    if message.sender.lower() == config.OWNER:
        # The owner's own sent mail is in this store, and a reply to yourself is
        # not a thing the system should ever produce.
        result = Draft(
            message.id,
            None,
            (),
            "this is the owner's own message; there is nobody to reply to",
            outcome="no_reply",
        )
        trace.event("draft", msg_id=message.id, drafted=False, cites=[], outcome="no_reply", reason=result.reason)
        return result

    withhold = any(contains_secret(mailbox.by_id(item.message_id).text()) for item in evidence)
    prompt = build_draft_prompt(message, evidence, withhold_secret=withhold, owner=config.OWNER)

    problem = ""
    echoed_every_attempt = True
    for attempt in range(1, retries + 2):
        try:
            raw = agent.handle_message(prompt, thread_id=thread_id)
        except Exception as error:  # noqa: BLE001 - provider already retried
            trace.event("draft_error", msg_id=message.id, attempt=attempt, error=str(error)[:300])
            return Draft(message.id, None, (), f"the model could not be reached: {error}", outcome="rejected")
        try:
            # The system prompt only. The per-message prompt quotes the mail, so
            # checking against it would flag a draft for correctly quoting the
            # message it is answering.
            result = parse_draft(raw, message, evidence, mailbox, getattr(agent, "system_prompt", ""))
        except DraftRejected as error:
            problem = str(error)
            echoed_every_attempt = echoed_every_attempt and "copies" in problem
            trace.event("draft_rejected", msg_id=message.id, attempt=attempt, problem=problem, raw=raw[:200])
            hint = ""
            if "copies" in problem:
                # Answering a message by repeating it is what a model does when
                # the message wants no answer. Offering the honest exit at the
                # moment the mistake is made works where stating it once in the
                # system prompt did not: a first run rejected six status updates
                # twice each and never once reached for "no_reply".
                hint = (
                    "\nIf this message is a status update, an FYI or a note of thanks and "
                    'wants no answer at all, say so: {"draft": null, "outcome": "no_reply", '
                    '"reason": "..."}.'
                )
            prompt = f"{prompt}\n\nYour previous answer was rejected: {problem}{hint}\nAnswer again, correctly."
            continue
        trace.event(
            "draft",
            msg_id=message.id,
            drafted=result.drafted,
            cites=list(result.cites),
            chars=len(result.text or ""),
            withheld_credential=result.withheld,
            outcome=result.outcome,
            attempts=attempt,
        )
        return result

    # Every attempt was unusable. Before recording that as a failure, one
    # deterministic reading of it: if the model could only answer by handing the
    # message back, every single time, and the message asks for nothing, then
    # there was no reply to write and the repeated echo was the model saying so
    # in the only way it had. Both signals are required, both are ours rather
    # than the model's -- the rejection reason is this module's own, and
    # `asks_anything` is a text rule. A small model needs this; a larger one
    # reaches for "no_reply" by itself and never gets here.
    if echoed_every_attempt and not asks_anything(message):
        reason = "the message asks for nothing, and every attempt to answer it only repeated it"
        trace.event(
            "draft",
            msg_id=message.id,
            drafted=False,
            cites=[],
            outcome="no_reply",
            reason=reason,
            inferred=True,
            attempts=retries + 1,
        )
        return Draft(message.id, None, (), reason, outcome="no_reply")

    trace.event(
        "draft", msg_id=message.id, drafted=False, cites=[], outcome="rejected", reason=problem, attempts=retries + 1
    )
    return Draft(message.id, None, (), f"no usable draft: {problem}", outcome="rejected")


if __name__ == "__main__":
    import agents
    import config
    import mailstore
    import retrieval
    import rules

    config.check()
    box = mailstore.load()
    index = retrieval.Index(box)
    bot = agents.drafter_agent()
    trace.start_run(cap="R2", fresh=False)

    import sys

    # Ids may be named on the command line; with none, the same rule demo.py's
    # R2 uses derives them. No id is written down here: this entry point makes
    # no claim about which messages happen to draft well.
    # Ids may be named on the command line. With none, the same source demo.py's
    # R2 uses: what the triage tier already decided to reply to. No id is
    # written down here, and this entry point does not second-guess that call.
    wanted = sys.argv[1:]
    if not wanted:
        import json

        recorded = config.STATE_PATH / "decisions.json"
        if not recorded.exists():
            raise SystemExit(f"no decisions at {recorded}: run `python demo.py --cap R1` first, or name ids here")
        rows = json.loads(recorded.read_text(encoding="utf-8"))
        wanted = [r["message_id"] for r in rows if r.get("disposition") == "reply"]
        print(f"  {len(wanted)} dispositioned 'reply' in {recorded.name}; pass ids to override")
    for message_id in wanted:
        message = box.by_id(message_id)
        found = retrieval.retrieve(box, message, k=config.RETRIEVAL_K, index=index)
        print(f"\n=== {message.summary()} ===")
        print(f"  evidence: {found.ids or 'none'}")
        result = draft(bot, message, found.evidence, box, thread_id=f"draft-{message_id}")
        print(result.line())
        if result.withheld:
            print("  (a credential in the cited message was deliberately not quoted)")
