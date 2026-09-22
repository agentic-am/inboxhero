"""X3. Write back in the register the correspondent writes in.

One voice for the whole inbox is wrong in two directions, and only one of them
costs anything. Answering a friend of ten years in the language of a contract is
merely stiff. Answering outside counsel, a journalist or the board the way you
answer a friend is the kind of mistake that gets forwarded.

So the register is learned from the correspondent's own mail, stored, applied when
a reply is drafted, and checked afterwards in Python -- and the check is
deliberately one-directional. It refuses a draft that is too familiar for the
correspondent and says nothing about one that is too formal, because that is where
the cost actually is.

**A profile is an observation, not an instruction**, and that is why it is not
gated the way a standing preference is. Part 5 gates preference writes because a
preference is something somebody *asserts* -- and one message in this inbox forges
exactly that, from the owner's own address. A tone profile cannot be forged in the
same way: it is measured from how a person actually writes, it is recomputed from
their mail whenever asked, and nothing it can say widens what the system may do.
The human stays in the loop where it matters, on the send, through the Part 4 gate.

Measured, not asked of a model: the signals are countable, and a model asked "is
this formal?" would answer differently on different runs for the same mail.

Usage:
    python tone.py            # the register learned for every correspondent
    python tone.py m048       # what a reply to that message has to sound like
"""

import json
import re
from dataclasses import dataclass

import config
import mailstore
import rules

FORMAL, NEUTRAL, CASUAL = "formal", "neutral", "casual"

CONTRACTION = re.compile(r"\b\w+'(?:s|t|re|ve|ll|d|m)\b", re.I)
EXCLAMATION = re.compile(r"!")
SENTENCE = re.compile(r"[^.!?]+[.!?]*")
# A sign-off naming a firm, a title or a full name is the strongest single signal
# that somebody is writing in a professional capacity rather than a personal one.
FORMAL_SIGNOFF = re.compile(
    r"--\s*[A-Z][a-z]+\s+[A-Z][a-z]+|\b(?:LLP|LLC|Ltd|Inc|Esq|Partner|Counsel|Regards|Sincerely)\b"
)
CASUAL_OPENER = re.compile(r"^(?:hey|heya|yo|hiya)\b", re.I)


@dataclass(frozen=True)
class Profile:
    """How one correspondent writes, and what that means for writing back."""

    address: str
    register: str
    messages: int
    contractions: float  # per 100 words
    exclamations: int
    lowercase_starts: float  # share of sentences
    words_per_sentence: float
    formal_signoff: bool
    why: tuple = ()

    def row(self):
        return {
            "address": self.address,
            "register": self.register,
            "messages": self.messages,
            "why": list(self.why),
        }

    def line(self):
        return f"  {self.address:34} {self.register:8} from {self.messages} message(s) -- {'; '.join(self.why)}"


def measure(messages):
    """The countable things. No model, so the same mail always scores the same."""
    body = "\n".join(m.body for m in messages)
    words = body.split()
    sentences = [s.strip() for s in SENTENCE.findall(body) if s.strip()]
    starts = [s for s in sentences if s[:1].isalpha()]
    lower = [s for s in starts if s[:1].islower()]
    return {
        "words": len(words),
        "contractions": (len(CONTRACTION.findall(body)) * 100.0 / max(len(words), 1)),
        "exclamations": len(EXCLAMATION.findall(body)),
        "lowercase_starts": (len(lower) / max(len(starts), 1)),
        "words_per_sentence": (len(words) / max(len(sentences), 1)),
        "formal_signoff": bool(FORMAL_SIGNOFF.search(body)),
        "casual_opener": bool(CASUAL_OPENER.search(body.strip())),
    }


def register_for(counts):
    """Which of three registers the numbers say. Returns (register, reasons)."""
    casual, formal, why = 0, 0, []

    if counts["lowercase_starts"] >= 0.4:
        casual += 2
        why.append(f"{counts['lowercase_starts']:.0%} of sentences start lowercase")
    if counts["exclamations"] >= 1:
        casual += 1
        why.append(f"{counts['exclamations']} exclamation(s)")
    if counts["casual_opener"]:
        casual += 1
        why.append("opens with an informal greeting")
    if counts["contractions"] >= 3.0:
        casual += 1
        why.append(f"{counts['contractions']:.1f} contractions per 100 words")

    if counts["formal_signoff"]:
        formal += 2
        why.append("signs off with a name, title or firm")
    if counts["contractions"] == 0 and counts["words"] >= 25:
        formal += 1
        why.append("no contractions")
    if counts["words_per_sentence"] >= 18:
        formal += 1
        why.append(f"{counts['words_per_sentence']:.0f} words per sentence")

    if casual >= 2 and casual > formal:
        return CASUAL, tuple(why)
    if formal >= 2 and formal > casual:
        return FORMAL, tuple(why)
    return NEUTRAL, tuple(why) or ("nothing distinctive either way",)


def learn(mailbox=None):
    """A profile for every correspondent who has written, from their own mail.

    Refused mail is excluded. Learning a register from a phishing message would
    let an attacker choose the voice the system answers in, which is a small thing
    on its own and a free one to refuse.

    Automated senders are excluded too, and for a duller reason: nobody replies to
    them, so a register for them is an answer to a question no one asks. It is also
    wrong more often than not -- two machine notices here open every sentence in
    lowercase and scored as casual, which they are not; they are templates.
    """
    mailbox = mailbox if mailbox is not None else mailstore.load()
    owner = config.OWNER
    by_sender = {}
    for message in mailbox.messages:
        if message.sender.lower() == owner or rules.classify(message).hostile:
            continue
        if rules.is_automated(message):
            continue
        by_sender.setdefault(message.sender.lower(), []).append(message)

    profiles = {}
    for address, messages in by_sender.items():
        counts = measure(messages)
        register, why = register_for(counts)
        profiles[address] = Profile(
            address=address,
            register=register,
            messages=len(messages),
            contractions=round(counts["contractions"], 2),
            exclamations=counts["exclamations"],
            lowercase_starts=round(counts["lowercase_starts"], 2),
            words_per_sentence=round(counts["words_per_sentence"], 1),
            formal_signoff=counts["formal_signoff"],
            why=why,
        )
    return profiles


# --- storing and reading back ----------------------------------------------


def _file():
    return config.STATE_PATH / "tone.json"


def save(profiles):
    config.STATE_PATH.mkdir(parents=True, exist_ok=True)
    _file().write_text(
        json.dumps({a: p.row() for a, p in sorted(profiles.items())}, indent=2) + "\n", encoding="utf-8"
    )
    return _file()


def stored():
    path = _file()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def register_of(address, profiles=None):
    """The stored register for an address, or neutral when nothing is known."""
    profiles = profiles if profiles is not None else stored()
    entry = profiles.get((address or "").lower())
    return (entry or {}).get("register", NEUTRAL)


# --- applying it ------------------------------------------------------------

GUIDANCE = {
    FORMAL: (
        "Write formally. No contractions, no exclamation marks, no abbreviations. "
        "Complete sentences, and address them by name."
    ),
    NEUTRAL: "Write in plain professional English. Courteous, direct, no slang.",
    CASUAL: "Write warmly and briefly, as you would to someone you know well. Contractions are fine.",
}


def for_prompt(address, profiles=None):
    """The line the drafter is given. Rules, never examples.

    Showing the model two of the correspondent's own sentences would teach the
    register faster and would also get them copied back verbatim -- the recitation
    failure this project has already hit twice. The register is described instead.
    """
    register = register_of(address, profiles)
    return f"Tone: {GUIDANCE[register]} (this correspondent writes {register}ly)"


TOO_FAMILIAR = (
    re.compile(r"!"),
    re.compile(r"\b(?:hey|yo|cheers|thanks a ton|no worries|cool|awesome|sure thing)\b", re.I),
)


def too_familiar(text, register):
    """What in this draft is too informal for that correspondent.

    Only ever fires on `formal`. A reply that is stiffer than the correspondent is
    not a problem worth rejecting a draft over, and a check that fired both ways
    would reject far more than it saved.
    """
    if register != FORMAL:
        return ()
    found = []
    for pattern in TOO_FAMILIAR:
        match = pattern.search(text or "")
        if match:
            found.append(match.group(0).strip())
    if CONTRACTION.search(text or ""):
        found.append(CONTRACTION.search(text).group(0))
    return tuple(found)


def describe(profiles):
    out = [f"=== how {len(profiles)} correspondent(s) write ==="]
    for register in (FORMAL, NEUTRAL, CASUAL):
        group = [p for p in profiles.values() if p.register == register]
        out.append(f"\n--- {register} ({len(group)}) ---")
        for profile in sorted(group, key=lambda p: -p.messages):
            out.append(profile.line())
    return "\n".join(out)


if __name__ == "__main__":
    import sys

    box = mailstore.load()
    profiles = learn(box)
    wanted = sys.argv[1:]
    if wanted:
        for message_id in wanted:
            message = box.by_id(message_id)
            if message is None:
                print(f"{message_id}: not in the inbox")
                continue
            profile = profiles.get(message.sender.lower())
            print(f"{message_id}  {message.sender}")
            print(f"  register: {profile.register if profile else NEUTRAL}")
            if profile:
                print(f"  because:  {'; '.join(profile.why)}")
            print(f"  drafter is told: {for_prompt(message.sender, {a: p.row() for a, p in profiles.items()})}")
    else:
        print(describe(profiles))
        print(f"\n  written to {save(profiles)}")
