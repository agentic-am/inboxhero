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
import flow
import mailstore
import rules
import trace

# The manifest's --cap ids, one entry per capability.
CAPABILITIES = {
    "R1": "Zero the inbox: every message gets exactly one disposition and a reason.",
}


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
    print("  dispositions         " + ", ".join(f"{k}={v}" for k, v in sorted(by_disposition.items())))

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
    trace.start_run(cap=cap, fresh=True)

    print(f"=== {cap}: {CAPABILITIES[cap]} ===")
    print(f"  inbox {config.INBOX_PATH.name}: {len(box)} records, {len(box.problems)} malformed")
    print(f"  model {config.MODEL} via {config.PROVIDER}\n")

    decisions = zero_the_inbox(box, records, quiet=args.quiet)
    missing = summarise(records, decisions)
    path = _write_decisions(decisions)
    print(f"\n  decisions written to {path}")
    print(f"  trace written to     {config.TRACE_PATH}  ({len(trace.read())} events)")
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
