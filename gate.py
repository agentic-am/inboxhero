"""The only way an irreversible action can happen. Part 4.

Everything else the system does stays inside the mailbox, where a move can be
moved back. Sending is the one action that crosses the boundary: once a reply is
delivered it is outside the mailbox and outside anyone's reach. So it is the one
action that does not go through `actions.py` at all. It goes through here, and
`write_outbox` below is the only function in the project that writes mail
anywhere.

Delivery is a file in `outbox/` rather than an SMTP conversation. That is a
deliberate simplification of the plumbing and not of the commitment: a transport
and a server to reach would change nothing about which sends need a person, and
a file written here is treated as delivered rather than as a draft waiting to go.

The pipeline is four steps and no model:

    screen  ->  ask  ->  execute  ->  record

`screen` is the part that matters. It produces two separate lists, and they are
not the same kind of thing:

  refusals  the action will not happen, whatever anyone says. A human `yes` is
            permission, not authority: it can allow what the design allows, and
            it cannot unlock what the design refuses.
  asks      the reasons this one crosses the escalation line and a person has to
            read it before it goes.

An action with neither goes through, logged and unasked. That is deliberate and
it is the trade Part 4 asks us to name: a human asked about seventeen things
approves seventeen things, so this asks about the ten that can hurt.

Usage:
    python gate.py     # screen the recorded drafts, ask nothing, write nothing
"""

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone

from moya.flows.pipeline import Pipeline
from moya.flows.steps import FunctionStep

import actions
import config
import drafting
import rules
import trace

# Where the escalation line falls. Each entry is a reason a person has to read
# the thing before it goes, and each is a sentence rather than a code because it
# is printed to the person being asked: "sensitive" tells them nothing, "the
# draft commits you to a specific time" tells them what to check.
ASK_SENSITIVE = "the message or the draft touches {terms}"
ASK_CREDENTIAL = "the thread being answered carries a credential"
ASK_CROSS_THREAD = "the draft leans on {ids}, from a different thread"
ASK_COMMITMENT = "the draft commits the owner to a specific time or date"
ASK_IRREVERSIBLE = "{action} cannot be undone once the retention window closes"

# What no answer can authorise.
NO_SUCH_MESSAGE = "there is no message {id} in the inbox"
REFUSE_FLAGGED = "{id} was flagged by the rule tier, and flagged mail is neither answered nor moved"
REFUSE_EMPTY = "there is no draft to send"
REFUSE_UNKNOWN_ADDRESS = "{address} has never appeared in this inbox"
REFUSE_CREDENTIAL = "the draft carries a credential, which may not leave in a reply"
REFUSE_ALREADY_SENT = "{id} was already sent; {path} exists and is treated as delivered"


# What the human said, as three answers rather than free text. "not asked" is a
# separate answer from "no" and must never be read as one: `"not asked"` does
# begin with `"no"`, and a prefix test here would have turned every send that
# fell below the escalation line into a refusal.
SAID_YES = "yes"
SAID_NO = "no"
SAID_UNASKED = "not asked"


def said_yes(answer):
    return (answer or "").strip().lower() == SAID_YES


def said_no(answer):
    answer = (answer or "").strip().lower()
    return answer == SAID_NO or answer.startswith(SAID_NO + ":")


@dataclass(frozen=True)
class Proposal:
    """What the system wants to do. Carries no permission of its own."""

    message_id: str
    action: str  # "send" or "delete"
    recipient: str = ""
    subject: str = ""
    body: str = ""
    cites: tuple = ()
    thread_id: str = ""
    by: str = "model"  # who asked for it

    def line(self):
        if self.action == "send":
            return f"send to {self.recipient}: {self.subject!r}"
        return f"{self.action} {self.message_id}"


@dataclass
class Verdict:
    """One gated decision, in the three fields Part 4 asks to be logged."""

    proposal: object
    refusals: tuple = ()
    asks: tuple = ()
    human_said: str = "not asked"
    happened: str = ""
    path: object = None  # the outbox file, when one was written

    @property
    def needs_human(self):
        return bool(self.asks)

    @property
    def blocked(self):
        return bool(self.refusals)

    @property
    def approved(self):
        return said_yes(self.human_said)

    @property
    def declined(self):
        return said_no(self.human_said)

    @property
    def did_something(self):
        return self.happened.startswith(("sent", "moved"))

    def line(self):
        return f"{self.proposal.message_id:6} {self.proposal.line()}"


# --- the escalation line --------------------------------------------------


def commits_to_a_time(text):
    """Whether this draft agrees to a specific time or date.

    Not a suspicious thing to write, which is exactly why it needs its own
    criterion: accepting a meeting uses none of the vocabulary that makes a
    message sensitive, and it is still a commitment the owner cannot walk back
    once it has gone. Four of this inbox's drafts accept a slot, and one of them
    accepts a 9:00am that the owner has said in writing they never take.
    """
    return bool(drafting.TIME.search(text or "") or drafting.DATE.search(text or ""))


def screen(proposal, message, mailbox, folders):
    """Every reason to refuse, and every reason to ask. Never acts.

    `mailbox` is the inbox being worked from; `folders` is the mailbox state, so
    the gate can tell an unsent draft from one that already went.
    """
    refusals, asks = [], []

    if message is None:
        return (NO_SUCH_MESSAGE.format(id=proposal.message_id),), ()

    if folders.status(proposal.message_id) == actions.STATUS_FOR["flag"]:
        refusals.append(REFUSE_FLAGGED.format(id=proposal.message_id))
    elif rules.classify(message).hostile:
        # The recorded status is the usual route, but a proposal made before any
        # triage was applied must not slip past on a missing file.
        refusals.append(REFUSE_FLAGGED.format(id=proposal.message_id))

    if proposal.action == "send":
        if not (proposal.body or "").strip():
            refusals.append(REFUSE_EMPTY)
        if proposal.recipient not in mailbox.known_addresses():
            refusals.append(REFUSE_UNKNOWN_ADDRESS.format(address=proposal.recipient or "(nobody)"))
        if drafting.contains_secret(proposal.body):
            refusals.append(REFUSE_CREDENTIAL)
        path = outbox_path(proposal.message_id)
        if path.exists():
            refusals.append(REFUSE_ALREADY_SENT.format(id=proposal.message_id, path=path.name))

        terms = rules.sensitive_hits(message.text() + " " + proposal.body)
        if terms:
            asks.append(ASK_SENSITIVE.format(terms=", ".join(sorted(set(terms)))))
        if drafting.contains_secret(message.text()):
            asks.append(ASK_CREDENTIAL)
        elsewhere = [
            cited
            for cited in proposal.cites
            if (found := mailbox.by_id(cited)) is not None and found.thread_id != proposal.thread_id
        ]
        if elsewhere:
            asks.append(ASK_CROSS_THREAD.format(ids=", ".join(elsewhere)))
        if commits_to_a_time(proposal.body):
            asks.append(ASK_COMMITMENT)

    if proposal.action == "delete":
        # Always asked. The bin is reversible, but on a timer rather than on
        # someone changing their mind, so it is gated as if it were not.
        asks.append(ASK_IRREVERSIBLE.format(action="delete"))

    return tuple(refusals), tuple(asks)


# --- the one place mail is written ----------------------------------------


def outbox_path(message_id):
    return config.OUTBOX_PATH / f"{message_id}.txt"


def _reply_subject(subject):
    subject = (subject or "").strip()
    if not subject:
        return "Re:"
    return subject if subject.lower().startswith("re:") else f"Re: {subject}"


def write_outbox(proposal, verdict, run_id=""):
    """One file per message, and nowhere else.

    The headers are the audit: who it went to, what it answers, what it leaned
    on, and why the gate let it through. A reader holding only this file can
    tell whether a person approved it or whether it fell below the line.
    """
    path = outbox_path(proposal.message_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    approval = "human approved" if said_yes(verdict.human_said) else "below the escalation line, not asked"
    why = "; ".join(verdict.asks) if verdict.asks else "no reason to ask"
    header = [
        f"To: {proposal.recipient}",
        f"From: {config.OWNER}",
        f"Subject: {_reply_subject(proposal.subject)}",
        f"Date: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"In-Reply-To: {proposal.message_id}",
        f"Thread: {proposal.thread_id}",
        f"Cites: {', '.join(proposal.cites) if proposal.cites else '(none)'}",
        f"Approved-By: {approval} ({why})",
        f"Run: {run_id}",
    ]
    path.write_text("\n".join(header) + "\n\n" + proposal.body.strip() + "\n", encoding="utf-8")
    return path


# --- asking a person ------------------------------------------------------


def terminal_asker(question, detail):
    """Ask on the terminal. Anything but an explicit yes is a no.

    A gate that cannot reach a person must not decide that silence means yes, so
    a closed or redirected stdin is a refusal rather than an approval. That is
    the difference between a prompt and a formality.
    """
    if not sys.stdin or not sys.stdin.isatty():
        return f"{SAID_NO}: nobody could be asked, stdin is not a terminal"
    print(detail)
    try:
        answer = input(f"{question} [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return f"{SAID_NO}: the prompt was interrupted"
    return SAID_YES if answer in ("y", "yes") else SAID_NO


def describe(proposal, verdict):
    """What a person is shown before answering. The whole draft, never a summary."""
    lines = [f"\n  --- {proposal.line()} ---"]
    for reason in verdict.asks:
        lines.append(f"  asked because: {reason}")
    if proposal.action == "send":
        lines.append(f"  in reply to:   {proposal.message_id} in thread {proposal.thread_id}")
        if proposal.cites:
            lines.append(f"  grounded in:   {', '.join(proposal.cites)}")
        lines.append("  draft:")
        lines.extend(f"      {line}" for line in proposal.body.splitlines())
    if proposal.action == "delete":
        lines.append(f"  recoverable until {actions.purge_date()}, then not")
    return "\n".join(lines)


# --- the steps ------------------------------------------------------------


def screen_step(ctx):
    run = ctx.metadata["run"]
    proposal = ctx.metadata["proposal"]
    message = run.mailbox.by_id(proposal.message_id)
    refusals, asks = screen(proposal, message, run.mailbox, run.folders)
    ctx.metadata["verdict"] = Verdict(proposal=proposal, refusals=refusals, asks=asks)
    return ctx


def ask_step(ctx):
    """Dry-run says what it would do; approval asks; neither writes anything."""
    run = ctx.metadata["run"]
    verdict = ctx.metadata["verdict"]

    if verdict.blocked:
        verdict.human_said = SAID_UNASKED
        return ctx
    if run.mode == "dry-run":
        verdict.human_said = f"{SAID_UNASKED} (dry-run)"
        return ctx
    if not verdict.asks:
        verdict.human_said = SAID_UNASKED
        return ctx

    verdict.human_said = run.asker(f"  {verdict.proposal.line()} -- send it?", describe(verdict.proposal, verdict))
    return ctx


def execute_step(ctx):
    """The only step that changes anything, and it acts on the verdict alone.

    It never reads the proposal's own opinion of itself: a refusal is a refusal
    and an unanswered prompt is a no, so there is one place to look to know
    whether something happened.
    """
    run = ctx.metadata["run"]
    verdict = ctx.metadata["verdict"]
    proposal = verdict.proposal

    if verdict.blocked:
        verdict.happened = "refused: " + "; ".join(verdict.refusals)
        return ctx
    if run.mode == "dry-run":
        verdict.happened = "nothing (dry-run): it would have " + (
            f"written {outbox_path(proposal.message_id).name}"
            if proposal.action == "send"
            else f"moved {proposal.message_id} to the bin"
        )
        return ctx
    if verdict.asks and not said_yes(verdict.human_said):
        # Anything short of an explicit yes on an action that crossed the line is
        # a no. The permissive form of this test -- act unless someone said no --
        # turns a closed stdin, an interrupted prompt and a typo into approvals.
        verdict.happened = f"nothing: not approved ({verdict.human_said})"
        return ctx

    if proposal.action == "send":
        verdict.path = write_outbox(proposal, verdict, run_id=run.run_id)
        run.folders.apply(
            "send",
            proposal.message_id,
            reason=f"answered and sent to {proposal.recipient}",
            detail={"outbox": verdict.path.name, "cites": list(proposal.cites)},
            by=proposal.by,
        )
        verdict.happened = f"sent: wrote {verdict.path.name}, {proposal.message_id} marked answered"
    else:
        moved = actions.now()
        run.folders.apply(
            "delete",
            proposal.message_id,
            reason="deleted on request",
            detail={"deleted_at": actions.stamp(moved), "purge_after": actions.purge_date(moved)},
            by=proposal.by,
        )
        verdict.happened = f"moved {proposal.message_id} to the bin, recoverable until {actions.purge_date(moved)}"
    return ctx


def record_step(ctx):
    """One `gate` line per decision, with the three fields Part 4 names."""
    run = ctx.metadata["run"]
    verdict = ctx.metadata["verdict"]
    trace.event(
        "gate",
        msg_id=verdict.proposal.message_id,
        action=verdict.proposal.action,
        mode=run.mode,
        proposed=verdict.proposal.line(),
        needs_human=list(verdict.asks),
        refusals=list(verdict.refusals),
        human_said=verdict.human_said,
        happened=verdict.happened,
    )
    run.verdicts.append(verdict)
    ctx.output = verdict.happened
    return ctx


def build_pipeline(event_bus=None):
    return Pipeline(
        steps=[
            FunctionStep(screen_step, name="screen"),
            FunctionStep(ask_step, name="ask"),
            FunctionStep(execute_step, name="execute"),
            FunctionStep(record_step, name="record"),
        ],
        name="gate",
        event_bus=event_bus,
    )


@dataclass
class Run:
    """One pass of the gate over a list of proposals."""

    mailbox: object  # the inbox being worked from
    folders: object = None  # actions.Mailbox: where each message sits
    mode: str = ""
    asker: object = terminal_asker
    run_id: str = ""
    verdicts: list = field(default_factory=list)

    def __post_init__(self):
        self.mode = self.mode or config.GATE_MODE
        if self.folders is None:
            self.folders = actions.load_applied()


def run(proposals, mailbox, mode=None, asker=None, folders=None, event_bus=None, run_id=""):
    """Put every proposal through the gate once, in the given mode.

    `both` is two passes over the same list: the dry-run reports everything with
    nothing at stake, then the approval pass asks about the few that crossed the
    line. The dry-run's verdicts are returned too, because "here is what it said
    it would do, and here is what it did" is only checkable if both are kept.
    """
    mode = mode or config.GATE_MODE
    passes = ["dry-run", "approval"] if mode == "both" else [mode]
    pipeline = build_pipeline(event_bus=event_bus)
    done = []
    for this_pass in passes:
        current = Run(
            mailbox=mailbox,
            folders=folders,
            mode=this_pass,
            asker=asker or terminal_asker,
            run_id=run_id,
        )
        folders = current.folders  # the approval pass sees what the dry-run saw
        for proposal in proposals:
            pipeline.run(
                thread_id=proposal.thread_id or proposal.message_id,
                message=proposal.line(),
                proposal=proposal,
                run=current,
            )
        done.append((this_pass, current))
    return done


def proposals_from_decisions(rows, mailbox):
    """A send proposal for every recorded draft. No model call, no re-deciding.

    The drafts were written and checked earlier; the gate's job is to decide
    whether they may leave, not whether they are any good. A row with no draft
    text is not a proposal at all -- a message the drafter declined to answer has
    nothing to send.
    """
    found = []
    for row in rows:
        if not (row.get("draft") or "").strip():
            continue
        message = mailbox.by_id(row.get("message_id"))
        if message is None:
            continue
        found.append(
            Proposal(
                message_id=message.id,
                action="send",
                recipient=message.sender,
                subject=message.subject,
                body=row["draft"],
                cites=tuple(row.get("draft_cites") or ()),
                thread_id=message.thread_id,
            )
        )
    return found


if __name__ == "__main__":
    import json

    import mailstore

    box = mailstore.load()
    path = config.STATE_PATH / "decisions.json"
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    proposals = proposals_from_decisions(rows, box)

    print(f"=== screening {len(proposals)} recorded draft(s); nothing is asked and nothing is written ===")
    folders = actions.load_applied()
    asked = 0
    for proposal in proposals:
        refusals, asks = screen(proposal, box.by_id(proposal.message_id), box, folders)
        if refusals:
            print(f"  {proposal.message_id}  REFUSED   {'; '.join(refusals)}")
        elif asks:
            asked += 1
            print(f"  {proposal.message_id}  ask       {'; '.join(asks)}")
        else:
            print(f"  {proposal.message_id}  unasked")
    print(f"\n  {asked} of {len(proposals)} cross the escalation line")
