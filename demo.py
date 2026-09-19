"""inboxHero's single entry point. Every capability in the manifest runs from here.

    python demo.py --cap R1              # zero the inbox: one disposition per message
    python demo.py --cap R1 --limit 12   # the same, over the first 12 messages by time
    python demo.py --msg m024            # one message, with its trace

Arguments are validated before anything else happens, and a bad one exits with
status 2 and a sentence saying what was wrong. The command line is a boundary,
and a boundary answers rather than raises.
"""

import argparse
import json
import os
import sys

from moya.observability.event_bus import EventBus

import agents
import config
import drafting
import flow
import mailstore
import retrieval
import rules
import trace

# The manifest's --cap ids, one entry per capability.
CAPABILITIES = {
    "R1": "Zero the inbox: every message gets exactly one disposition and a reason.",
    "R2": "Answer properly: draft a reply grounded in a specific earlier message, citing its id.",
}

# R2's candidate rule, stated once and applied to whatever inbox is loaded. No
# message id appears anywhere in this file: naming one would make the capability
# a claim about three messages that were known in advance to behave well, which
# is not the claim being made.
CANDIDATE_RULE = "R1 dispositioned it 'reply'"


def drafting_candidates(box):
    """The messages R1 decided to reply to, read back from its recorded decisions.

    R2 does not choose for itself and does not ask the model again. Which
    messages deserve a reply is the triage tier's judgement, it was made in R1,
    and it was written to `decisions.json`; re-deciding it here would mean the
    same inbox got two different answers from one system, and the second one
    would be the one nobody had reviewed.

    Reading the record rather than recomputing it also keeps R1 and R2 honest
    when they are run days apart, and it is what makes the disposition durable
    rather than an artefact of a process that happened to still be running.

    An earlier version derived the set from the rules instead -- every message
    where `reply` was *allowed*. That was 32 messages rather than 2, and it
    handed the drafter a launch thread full of status updates that triage would
    have archived. The drafter then did what a small model does when asked to
    reply to something that asks nothing: it sent the message back.

    No model call, and the same answer every run for a given decisions file.
    """
    path = config.STATE_PATH / "decisions.json"
    if not path.exists():
        raise Usage(
            f"no recorded decisions at {path}. R2 drafts for what R1 decided to reply to, "
            "so run `python demo.py --cap R1` first."
        )
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        raise Usage(f"could not read {path}: {error}") from error

    found = []
    for row in rows:
        if row.get("disposition") != "reply":
            continue
        message = box.by_id(row.get("message_id"))
        if message is None or message.sender.lower() == config.OWNER:
            continue
        found.append(message)
    return found


def record_drafts(results):
    """Write R2's drafts back into the decisions R1 recorded. Never touches a disposition.

    R2 owns the draft columns and nothing else. The disposition, the reason and
    the rule that produced them stay exactly as R1 wrote them -- re-running the
    drafting must not quietly re-decide anything, and a row R2 did not draft for
    is left untouched rather than blanked.

    Without this the two artefacts drift: `decisions.json` would keep whatever
    draft the last full run happened to produce while the trace showed a newer,
    different one, and a reader comparing them would find two answers for the
    same message and no way to tell which the system stands behind.
    """
    path = config.STATE_PATH / "decisions.json"
    if not path.exists():
        return 0
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_id = {result.message_id: result for result in results}
    updated = 0
    for row in rows:
        result = by_id.get(row.get("message_id"))
        if result is None:
            continue
        row["draft"] = result.text or ""
        row["draft_cites"] = list(result.cites)
        row["draft_reason"] = result.reason
        row["draft_outcome"] = result.outcome
        updated += 1
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return updated


class Usage(SystemExit):
    """A bad command line. Exit status 2, one sentence, no traceback."""

    def __init__(self, message):
        print(f"error: {message}", file=sys.stderr)
        print(f"hint:  python demo.py --cap R1   (capabilities: {', '.join(sorted(CAPABILITIES))})", file=sys.stderr)
        super().__init__(2)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="demo.py",
        description="inboxHero: take an inbox from unread to empty.",
        add_help=True,
    )
    parser.add_argument("--cap", help="capability id from the manifest, e.g. R1")
    parser.add_argument("--all", action="store_true", help="run every capability in order")
    parser.add_argument("--msg", help="process a single message id, e.g. m024")
    parser.add_argument("--limit", type=int, help="process only the first N messages, by timestamp")
    parser.add_argument("--batch", type=int, help="messages per model call; overrides BATCH_SIZE for this run")
    parser.add_argument("--quiet", action="store_true", help="print the summary only, not every row")
    args = parser.parse_args(argv)

    if not (args.cap or args.all or args.msg):
        raise Usage("nothing to do: pass --cap, --msg or --all")
    if args.cap and args.cap.upper() not in CAPABILITIES:
        raise Usage(f"unknown capability {args.cap!r}")
    if args.cap:
        args.cap = args.cap.upper()
    if args.limit is not None and args.limit < 1:
        raise Usage(f"--limit must be 1 or more, got {args.limit}")
    if args.batch is not None and args.batch < 1:
        raise Usage(f"--batch must be 1 or more, got {args.batch}")
    return args


def _write_decisions(decisions):
    path = config.STATE_PATH / "decisions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([d.as_row() for d in decisions], indent=2) + "\n", encoding="utf-8")
    return path


def zero_the_inbox(box, records, quiet=False):
    """R1. Every record through the pipeline, one disposition each, nothing left over."""
    bus = trace.attach(EventBus())
    state = flow.RunState(mailbox=box)
    agent = agents.triage_agent()
    pipeline = flow.build_pipeline(agent=agent, event_bus=bus)

    verdicts = rules.classify_all(records)
    numbers = rules.tally(verdicts)
    print(f"  {numbers['rule_handled']} of {numbers['records']} decided by rules, {numbers['to_the_model']} need the model.")

    if config.BATCH_SIZE > 1:
        by_id = {v.message_id: v for v in verdicts}
        eligible = flow.batchable(records, by_id)
        alone = numbers["to_the_model"] - len(eligible)
        print(f"  batching {len(eligible)} of them {config.BATCH_SIZE} at a time; {alone} sensitive one(s) stay on their own.")
        summary = flow.prefill_batches(agent, records, state, config.BATCH_SIZE, by_id)
        print(
            f"  {summary['batches']} batched call(s) answered {summary['batched']} message(s); "
            f"{summary['fell_back']} fell back to a call of their own."
        )
    print()

    for index, record in enumerate(records, start=1):
        decision = flow.run_one(pipeline, record, state)
        if not quiet:
            print(f"  {index:3}. {decision.line()}")
        elif index % 20 == 0:
            print(f"  ...{index} of {len(records)}")

    return state.decisions


def grounded_reply(box, messages):
    """R2. Retrieve, draft, and show what each draft was allowed to lean on.

    Nothing is sent and no file is written: a draft is reversible, and the
    system sends nothing and writes no file here.
    """
    index = retrieval.Index(box)
    drafter = agents.drafter_agent()
    results = []

    for message in messages:
        message_id = message.id
        found = retrieval.retrieve(box, message, k=config.RETRIEVAL_K, index=index)
        print(f"\n  {message.summary()}")
        print(f"    retrieved: {', '.join(e.line() for e in found.evidence) or 'nothing'}")

        result = drafting.draft(drafter, message, found.evidence, box, thread_id=f"draft-{message_id}")
        results.append(result)

        if not result.drafted:
            label = "no reply needed" if result.needs_no_reply else "no draft"
            print(f"    {label} -- {result.reason}")
            continue
        print("    draft:")
        for line in result.text.splitlines():
            print(f"      {line}")
        print(f"    cited: [{', '.join(result.cites) or ''}]")
        if result.withheld:
            print("    (a credential in a cited message was withheld, not quoted)")
    return results


def summarise_drafts(results):
    drafted = [r for r in results if r.drafted]
    cited = [r for r in drafted if r.cites]
    withheld = [r for r in drafted if r.withheld]
    # The three ways of not drafting are different answers and are counted
    # apart. Rolling them together was how "this message wants no reply" hid
    # inside "the inbox cannot answer this".
    no_reply = [r for r in results if r.outcome == "no_reply"]
    not_known = [r for r in results if r.outcome == "not_known"]
    rejected = [r for r in results if r.outcome == "rejected"]
    print("\n=== run summary ===")
    print(f"  messages considered  {len(results)}")
    print(f"  drafts written       {len(drafted)}")
    print(f"  drafts citing mail   {len(cited)}")
    print(f"  credentials withheld {len(withheld)}  (cited, deliberately not quoted)")
    print(f"  no reply needed      {len(no_reply)}  (the message asks for nothing)")
    print(f"  could not answer     {len(not_known)}  (nothing in the inbox answers it)")
    print(f"  no usable draft      {len(rejected)}  (every attempt failed a check)")
    for result in results:
        if not result.drafted:
            print(f"    {result.message_id} [{result.outcome}]: {result.reason}")
    # Part 3 asks for at least one grounded draft. Say so plainly rather than
    # leaving a reader to count.
    return 0 if cited else 1


def summarise(records, decisions):
    """The run summary. `undecided` is the first thing Part 2 is checked on."""
    by_disposition, by_path = {}, {}
    for decision in decisions:
        by_disposition[decision.disposition] = by_disposition.get(decision.disposition, 0) + 1
        by_path[decision.path] = by_path.get(decision.path, 0) + 1

    decided = {d.message_id for d in decisions}
    missing = [getattr(r, "id", "?") for r in records if getattr(r, "id", "?") not in decided]

    print("\n=== run summary ===")
    print(f"  messages processed   {len(records)}")
    print(f"  undecided            {len(missing)}" + (f"  {missing}" if missing else ""))
    print(f"  rule handled         {by_path.get('rules', 0)}   (no model call)")
    print(f"  model handled        {by_path.get('model', 0)}")

    asked = [d for d in decisions if d.path == "model"]
    if asked:
        grounded = [d for d in asked if d.evidence]
        cited = [d for d in asked if d.cites]
        print(
            f"  grounded             {len(grounded)} of {len(asked)} model decisions"
            f"  ({len(asked) - len(grounded)} had nothing in the inbox to ground them)"
        )
        print(f"  cited evidence       {len(cited)}")

    drafted = [d for d in decisions if d.draft]
    declined = [d for d in decisions if d.disposition == "reply" and not d.draft]
    if drafted or declined:
        print(f"  drafted replies      {len(drafted)}" + (f", {len(declined)} declined" if declined else ""))

    print("  dispositions         " + ", ".join(f"{k}={v}" for k, v in sorted(by_disposition.items())))

    if drafted:
        print("\n  drafted replies (nothing is sent):")
        for decision in drafted:
            cited = ", ".join(decision.draft_cites) or "nothing"
            print(f"    {decision.message_id}  cited: [{cited}]")
            for line in decision.draft.splitlines():
                print(f"      {line}")
    for decision in declined:
        print(f"\n  {decision.message_id}: no draft -- {decision.draft_reason}")

    flagged = [d for d in decisions if d.disposition == "flag"]
    if flagged:
        print(f"\n  flagged and left in place ({len(flagged)}):")
        for decision in flagged:
            print(f"    {decision.message_id}  {decision.reason}")

    retried = [d for d in decisions if d.attempts > 1]
    if retried:
        print(f"\n  needed a second attempt: {', '.join(d.message_id for d in retried)}")
    rejected = [d for d in decisions if d.problem]
    if rejected:
        print(f"  model answer unusable, escalated: {', '.join(d.message_id for d in rejected)}")
    return len(missing)


def main(argv=None):
    args = parse_args(argv)

    if args.batch:
        os.environ["BATCH_SIZE"] = str(args.batch)
        config.reload()

    try:
        for line in config.check():
            print(f"  warning: {line}")
    except config.ConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    try:
        box = mailstore.load()
    except mailstore.InboxFileError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    records = box.everything()
    if args.msg:
        record = box.by_id(args.msg)
        if record is None:
            raise Usage(f"no message {args.msg!r} in {config.INBOX_PATH.name}")
        records = [record]
    elif args.limit:
        records = records[: args.limit]

    cap = args.cap or ("R1" if args.all or args.msg else None)
    # R2 continues the run R1 recorded rather than starting over, so it appends
    # instead of truncating. The trace then holds the triage of all 100 messages
    # and the drafting that followed from it, which is what a reader needs to
    # check a citation: a `draft` event is only worth anything next to the
    # `read` events for the ids it cites. Every other capability starts clean.
    trace.start_run(cap=cap, fresh=(cap != "R2"))

    print(f"=== {cap}: {CAPABILITIES[cap]} ===")
    print(f"  inbox {config.INBOX_PATH.name}: {len(box)} records, {len(box.problems)} malformed")
    print(f"  model {config.MODEL} via {config.PROVIDER}\n")

    if cap == "R2":
        if args.msg:
            wanted = [record]
            print(f"  drafting for {record.id}, named on the command line\n")
        else:
            wanted = drafting_candidates(box)
            print(f"  {len(wanted)} to draft for, read from {config.STATE_PATH.name}/decisions.json: {CANDIDATE_RULE}")
            if args.limit:
                # A truncation of the derived set, in inbox order, not a
                # selection out of it. Said plainly so the count is readable.
                wanted = wanted[: args.limit]
                print(f"  --limit {args.limit}: drafting for the first {len(wanted)} of them")
            print()
        results = grounded_reply(box, wanted)
        status = summarise_drafts(results)
        if not args.msg:
            # A single --msg run is a spot check, not the record; it leaves the
            # recorded decisions alone.
            updated = record_drafts(results)
            print(f"\n  drafts recorded      {updated} rows in {config.STATE_PATH.name}/decisions.json"
                  "  (dispositions untouched)")
        print(f"  trace appended to    {config.TRACE_PATH}  ({len(trace.read())} events)")
        return status

    decisions = zero_the_inbox(box, records, quiet=args.quiet)
    missing = summarise(records, decisions)
    path = _write_decisions(decisions)
    print(f"\n  decisions written to {path}")
    print(f"  trace written to     {config.TRACE_PATH}  ({len(trace.read())} events)")
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
