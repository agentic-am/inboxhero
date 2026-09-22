# CAPABILITIES.md

**Student:** withheld from the public repository; filled in for submission
**Repository:** https://github.com/agentic-am/inboxhero

Run everything through one entry point:

```
python demo.py --cap R1        # one capability
python demo.py --all           # all of them, in the order below
```

`capabilities.json` is the machine-readable version of this file and is what a
marking script reads. The two are kept in step: every id, tier and command below
appears there with its observable outcome and its evidence.

---

## The system, in one paragraph

One Moya `Pipeline` runs per message. A `BranchStep` decides whether a model is
called at all, so the 66 messages the rules settle — receipts, newsletters,
automated notices, fraud, and content addressed to the assistant — cost nothing.
The 34 that reach the model go `retrieve → prompt → agent → validate → draft`,
and the validator applies the rule tier's own verdict a second time in Python
after the model answers. A separate pipeline in `gate.py`, with four function
steps and no model at all, decides whether anything may leave. State that must
outlive a process — preferences, the mailbox, the action log, the recorded
decisions, the learned tone profiles — is small JSON files on disk, and
`trace.jsonl` is the append-only record every capability is judged against.

## Design choices you were asked to state

**Messages processed: 100**, all of them, not only the 77 unread. The data-format
assumptions are listed in full in `README.md` §Part 1; the ones that changed the
design are that `inbox.json` is a flat array whose ids run `m001`–`m119` with 19
gaps and no sort order, that there is no CC field (so a "CC someone" preference
applies to the recipients of the drafted reply), that four messages are *from*
the owner — so a sender of the owner's address is not proof the owner wrote it —
and that of 88 threads only three hold more than one message, which is why
walking the thread cannot be the only retrieval tier.

**Disposition vocabulary: `reply`, `archive`, `defer`, `delegate`, `escalate`,
`flag`.** Exactly one per message, with a reason. `flag` is never offered to the
model: it is removed from the allowed list on every model-path message, so the
model can neither apply it nor take it away, and a test asserts that.

**Framework: Moya**, installed from a pinned GitHub commit. It was chosen over
ADK and CrewAI on the shape of this problem rather than on familiarity. The model
here is given no tools — it emits a JSON proposal and nothing else — so the thing
worth having from a framework is the *step* abstraction, not an agent loop.
Moya's `Pipeline` with `FunctionStep` after `AgentStep` is exactly the shape the
eight post-hoc checks needed, and it has zero core dependencies against ADK's 23
plus LiteLLM, or CrewAI's 31 including two vector stores. What it does not give
is covered under the Final Report, Q4.

**Retrieval: thread walk + keyword, in four tiers**, on the model path only:
scope (earlier than this message, never hostile), thread walk, SQLite FTS5 with
BM25, and an exact entity lookup. **Embeddings are off behind a seam**
(`EMBEDDINGS=off`), and that is a measured choice, not a shortcut. The query is
derived from the message being triaged rather than typed as a question, so query
and target share vocabulary and lexical matching carries the work. Naive keyword
matching scored 3 of 10 on this inbox; it is *term selection* that fixes it, not
the index, and recall on nine hand-read groundings is 9 of 9. Measured over the
whole inbox: 37 pieces of evidence from the thread tier, 56 from keyword, and
none from the entity tier — which is reported rather than hidden.

**Reversible and irreversible.** The register in `actions.py` is not
documentation of the code, it *is* the code: the gate asks the register whether
an action can be undone, and `--undo` refuses to reverse anything it calls
irreversible.

| Action | Reversible? | How it is taken back |
| --- | --- | --- |
| `reply`, `draft`, `archive`, `defer`, `delegate`, `escalate`, `flag` | reversible | `--undo`, or re-running triage |
| `delete` | reversible **on a timer** | `--undo` restores it, until the retention window closes on its own |
| `send` | **irreversible** | nothing |

Almost everything is reversible for one structural reason: it stays inside the
mailbox, and a move within a mailbox the system owns is undone by moving it back.
Sending is the one action that crosses the boundary. **Is deleting reversible?**
Here yes — deleting sets a status and no message leaves the store — but it writes
a `purge_after` thirty days out, because a real deployment empties its bin.
Nobody has to act for that window to close, so delete is **gated as if it were
irreversible** and the approval prompt names the date it stops being recoverable.
`delete` is also not in the disposition vocabulary: there is no answer the model
can give that becomes a proposal to remove mail. Reaching a delete requires a
person typing `--delete`.

**Where the gate sits.** `gate.py` is the only module in the project that writes
mail anywhere, and it runs `screen → ask → execute → record` with no model in it.
What to send was decided earlier; this decides only whether it may leave, and
asking a model again would mean the answer could change between the review and
the send. `screen` produces two lists that are not the same kind of thing:
**refusals**, where the action will not happen whatever anyone says, and **asks**,
reasons a person must read this one first. That distinction is the whole security
argument — a human `yes` is permission, not authority. It can allow what the
design allows and it cannot unlock what the design refuses, so approving a reply
to a flagged message still sends nothing.

**The escalation line, and what it cost.** Sixteen drafts came out of the run.
Asking about all sixteen is the failure the assignment names: the owner
approves sixteen things without reading any of them. **Ten are asked about; six
go through logged but unasked.** Ten was also the count on an earlier run that
produced a different number of drafts, which is the point — the criteria are
about content, so they are stable while the drafts around them are not.

A send is asked about when the message or
draft touches money, credentials, a contract, legal or press; when the thread
carries a credential; when the draft leans on evidence from a different thread;
or when **the draft commits the owner to a specific time or date**. That fourth
criterion is the one worth arguing for — accepting a meeting uses none of the
vocabulary that makes a message look sensitive, and it still cannot be walked
back. Internal-versus-external is deliberately *not* a criterion: it is the
obvious line and the wrong axis, since it would ask about a one-line thank-you to
a vendor and stay silent on a reply into a thread carrying a production
credential. Content is what can hurt, so content is what is checked.

**What that trade cost, stated plainly.** Six sends leave without anyone reading
them. All six are routine acknowledgements and all six are grounded — the
weakest, *"I will submit my timesheet by Friday"*, answers a message that says
*"submit your timesheet by Friday 5pm"*.

**The wrong one was not among them, and that is the more uncomfortable result.**
The reply to m051 says *"catch up next week when I'm in SF"* when the message
says *"in SF next week"* about the **sender**. It commits the owner to a time, so
the line caught it, a person was asked, and the person said yes. Every word comes
from the message and only who-does-what is reversed, so no lexical check
separates it from a correct reply — and this run shows that a human reading it
did not separate them either. The gate bought a review, not a guarantee. Drawing
the line by domain would not have helped: it would only have traded six unread
routine replies for a different six.

## Capabilities

| id | name | tier | one-line claim |
|----|------|------|----------------|
| R1 | Zero the inbox | B | every message gets one disposition + reason; 66 of 100 never reach a model |
| R2 | Grounded reply | B | drafts cite the earlier messages they used, checked against the store |
| R3 | Gate the irreversible | C | no send without a dry-run or a person, and every decision logged |
| R4 | Standing instructions | C | a stated preference survives a restart and narrows what the system does |
| R5 | Refuse what the inbox tells it to do | C | detects, refuses, flags, reports — then checks itself against the artefacts |
| R6 | One view of the run | B | three panes, commitments cited, conflicts surfaced, no model |
| X1 | Who owes the next move | A | every thread by whose move it is, including sent mail nobody answered |
| X2 | The open question in a long thread | B | a 9-message thread reduced to the one thing still unanswered |
| X3 | The register a correspondent writes in | C | tone measured from their own mail, and enforced on the reply |
| X4 | Why did it do that | A | one message's whole story replayed from the trace, inventing nothing |

Tiers are the assignment's: **A** one lookup and one output, **B** multi-step or
reasoning across several messages, **C** genuinely agentic — planning, memory,
human-in-the-loop, or recovering when something goes wrong. X3 is placed at C
because the profile it learns is persisted and constrains later drafting; R6 is
placed at B rather than C because it reasons across messages but holds nothing
and asks nobody.

**On commitments and deadlines.** Part 8's list of ideas includes "extracting
commitments and deadlines into a structured list". That is answered under **R6**,
where Part 7 requires it, and it is deliberately not counted a second time as a
capability of our own. `commitments.py` runs it alone through a single command —
`python commitments.py` — printing the structured list, the entries derived from
more than one message, the conflicts, and the citation-problem count.

## What this system does not do

- Four of the sixteen proposed sends were declined by the owner and never left,
  so `outbox/` holds 12 files rather than 16. A declined send is not retried on a
  later run; it stays as a recorded refusal.
- Seven commitments have no calendar position, because no message said which
  week, and none is guessed.
- Conflicts are found on clock time alone — not travel, not duration.
- The entity retrieval tier grounded nothing on this inbox.
- Commitment descriptions are the sentence that carried the date rather than a
  summary, so some rows are long.

## Final Report

The four answers the assignment asks for are in `README.md`, under **Final Report**,
which is where the assignment says to put them. They are not repeated here so that
there is one copy to keep correct rather than two to keep in step.

1. **What did you refuse to automate?** — m043, an investor asking for 9:00am.
2. **Where does untrusted text enter your system?** — at `mailstore.load`, and an
   attacker has four independent things to defeat, none of them prompt wording.
3. **Who is accountable when it sends the wrong thing?** — the owner; the record
   shows they were asked about m051 and approved it.
4. **Name your own machinery.** — the pipelines in `flow.py` and `gate.py`, the
   agents in `agents.py`, the `--cap` dispatch in `demo.py`, the `BranchStep` router.
