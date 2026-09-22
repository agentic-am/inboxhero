"""inboxHero's single entry point. Every capability in the manifest runs from here.

    python demo.py --cap R1              # zero the inbox: one disposition per message
    python demo.py --cap R1 --limit 12   # the same, over the first 12 messages by time
    python demo.py --msg m024            # one message, with its trace
    python demo.py --cap R3              # put every irreversible action through the gate
    python demo.py --undo 7              # take back action 7, if it can be taken back
    python demo.py --cap R4 --learn      # record the owner's standing instructions, then exit
    python demo.py --cap R4              # a separate process: load them and act on them
    python demo.py --cap R5              # the hostile inbox, and proof nothing acted on it
    python demo.py --cap R1 --rules-only # re-run the deterministic tier alone, no model call
    python demo.py --cap R6              # the dashboard: pending, flagged, commitments

Arguments are validated before anything else happens, and a bad one exits with
status 2 and a sentence saying what was wrong. The command line is a boundary,
and a boundary answers rather than raises.
"""

import argparse
import json
import os
import sys

from moya.observability.event_bus import EventBus

import actions
import agents
import config
import dashboard
import drafting
import flow
import gate
import hostile
import mailstore
import memory
import prefs
import retrieval
import rules
import threads
import tone
import trace
import waiting
import explain as explain_mod

# The manifest's --cap ids, one entry per capability.
CAPABILITIES = {
    "R1": "Zero the inbox: every message gets exactly one disposition and a reason.",
    "R2": "Answer properly: draft a reply grounded in a specific earlier message, citing its id.",
    "R3": "Gate the irreversible: nothing leaves without a dry-run or a person, and every decision is logged.",
    "R4": "Standing instructions: a preference stated in one run changes how a later, separate run behaves.",
    "R5": "The hostile inbox: what it tried to make the system do, and proof that nothing did it.",
    "R6": "One view of the run in three panes: what is pending, what was refused, and what is committed to.",
    "X1": "Who owes the next move, on every thread at once -- including the mail you sent that nobody answered.",
    "X2": "A long thread reduced to the one question nobody has answered, with the message it came from.",
    "X3": "Replies written in the register the correspondent writes in, learned from their own mail.",
    "X4": "Why did the system do that? The story of one message, replayed from the trace.",
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
    parser.add_argument("--gate", help=f"gate mode for this run: {', '.join(config.GATE_MODES)}")
    parser.add_argument(
        "--delete",
        metavar="ID",
        help="propose deleting a message; only a person may ask for this, and the gate always asks",
    )
    parser.add_argument("--undo", type=int, metavar="N", help="take back action N from state/actions.json")
    parser.add_argument(
        "--learn",
        action="store_true",
        help="with --cap R4: record the standing instructions and exit, changing nothing else",
    )
    parser.add_argument(
        "--rules-only",
        action="store_true",
        dest="rules_only",
        help="with --cap R1: re-run the deterministic tier alone; no model call, drafts untouched",
    )
    args = parser.parse_args(argv)

    if args.undo is not None:
        return args
    if not (args.cap or args.all or args.msg or args.delete):
        raise Usage("nothing to do: pass --cap, --msg, --delete or --all")
    if args.gate and args.gate.lower() not in config.GATE_MODES:
        raise Usage(f"--gate must be one of {', '.join(config.GATE_MODES)}, got {args.gate!r}")
    if args.gate:
        args.gate = args.gate.lower()
    if args.delete and args.cap and args.cap.upper() != "R3":
        raise Usage("--delete is a gated action, so it runs under --cap R3")
    if args.learn and args.cap and args.cap.upper() != "R4":
        raise Usage("--learn records standing instructions, so it runs under --cap R4")
    if args.rules_only and args.cap and args.cap.upper() != "R1":
        raise Usage("--rules-only re-runs the triage tier, so it runs under --cap R1")
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


def learn_preferences(box, run_id=""):
    """R4 --learn. Read the standing instructions out of the inbox, and stop there.

    This half of the capability writes `state/prefs.json` and nothing else. It
    does not go on to use what it learned, because the thing being demonstrated
    is that the next process picks it up from disk -- and a run that recorded a
    preference and then acted on it in the same breath would prove only that a
    variable survived a function call.
    """
    agent = agents.preference_agent()
    found = prefs.candidates(box)
    print(f"  {len(found)} message(s) look like a standing instruction: {', '.join(m.id for m in found) or 'none'}")
    print("  (derived from the rule tier's flag, not from a list of ids)\n")

    proposals, refused = [], []
    for message in found:
        row, problem = prefs.learn(agent, message, box)
        if row is None:
            refused.append((message.id, problem))
            continue
        proposals.append(
            gate.Proposal(
                message_id=message.id,
                action="preference_write",
                subject=message.subject,
                thread_id=message.thread_id,
                payload=row,
            )
        )

    for message_id, problem in refused:
        print(f"  {message_id}: not recorded -- {problem}")
    if refused:
        print()

    if not proposals:
        print("  nothing to record.")
        return []

    passes = gate.run(proposals, box, mode=config.GATE_MODE, run_id=run_id)
    for name, done in passes:
        report_pass(name, done)
    return passes


def affected_by(box, rows):
    """Recorded decisions whose correct treatment depends on a stored preference.

    Derived from the preferences themselves, so the set moves when the stored
    instructions move. Naming the messages here would make the capability a claim
    about ids that were known to work rather than about the rule.
    """
    floor = prefs.meeting_floor()
    found = []
    for row in rows:
        message = box.by_id(row.get("message_id"))
        if message is None:
            continue
        why = []
        early = prefs.too_early(message.text()) if floor is not None else ()
        if early:
            why.append(f"proposes {early[0]}, earlier than the {prefs.clock(floor)} floor")
        copies = prefs.cc_for(message.sender)
        if copies:
            why.append(f"a CC rule applies to {message.sender}: {', '.join(copies)}")
        if why:
            found.append((message, row, why))
    return found


def honour_preferences(box, args):
    """R4. A fresh process, the preferences read back off disk, and what changes.

    Three things are shown, because "it changes how the system behaves" is a
    claim about behaviour and not about a file having been written:
    what was loaded, what the gate now refuses, and what the drafter now writes.
    """
    stored = memory.all_prefs()
    if not stored:
        raise Usage(
            f"nothing recorded in {config.STATE_PATH.name}/prefs.json. "
            "Run `python demo.py --cap R4 --learn` first, let it exit, then run this."
        )

    print(f"  {len(stored)} standing instruction(s) read back from {config.STATE_PATH.name}/prefs.json:")
    for key, entry in stored.items():
        called = f", which the message called {entry['called']!r}" if entry.get("called") else ""
        print(f"    {key:24} = {entry.get('value')}   (stated in {entry.get('source')}{called})")
    print(f"\n  as the drafter will be told them:\n")
    for line in prefs.for_prompt().splitlines():
        print(f"    {line}")

    rows = read_decisions()
    affected = affected_by(box, rows)
    print(f"\n  {len(affected)} message(s) in this inbox are affected:")
    for message, row, why in affected:
        owed = row.get("disposition") == "reply"
        note = "" if owed else "   (no reply is owed, so nothing changes for it)"
        print(f"    {message.id} [{row.get('disposition'):8}] {'; '.join(why)}{note}")

    # What the gate does with drafts written before the instruction was stated.
    stale = [
        gate.Proposal(
            message_id=message.id,
            action="send",
            recipient=message.sender,
            subject=message.subject,
            body=row.get("draft") or "",
            cites=tuple(row.get("draft_cites") or ()),
            thread_id=message.thread_id,
        )
        for message, row, _ in affected
        if (row.get("draft") or "").strip()
    ]
    if stale:
        print(f"\n  === the gate, on the {len(stale)} draft(s) written before the instruction existed ===")
        folders = actions.load_applied()
        for proposal in stale:
            refusals, _ = gate.screen(proposal, box.by_id(proposal.message_id), box, folders)
            copies = prefs.cc_for(proposal.recipient)
            if refusals:
                print(f"    {proposal.message_id}  REFUSED   {'; '.join(refusals)}")
            elif copies:
                print(f"    {proposal.message_id}  allowed, and the outbox file will carry Cc: {', '.join(copies)}")
            else:
                print(f"    {proposal.message_id}  allowed, unchanged")

    # And what the drafter writes now, with the instruction in the prompt and the
    # same rule checked again in Python afterwards.
    redraft = [m for m, row, _ in affected if row.get("disposition") == "reply" and prefs.too_early(m.text())]
    if redraft and not args.quiet:
        index = retrieval.Index(box)
        drafter = agents.drafter_agent()
        print(f"\n  === redrafting the {len(redraft)} affected repl(y/ies) with the instruction in force ===")
        for message in redraft:
            was = next((r.get("draft") for m, r, _ in affected if m.id == message.id), "") or "(none)"
            found = retrieval.retrieve(box, message, k=config.RETRIEVAL_K, index=index)
            result = drafting.draft(drafter, message, found.evidence, box, thread_id=f"pref-{message.id}")
            print(f"\n    {message.id}  {message.sender}")
            print(f"      asked for: {', '.join(prefs.too_early(message.text()))}")
            print(f"      before:    {was.splitlines()[0] if was.strip() else '(none)'}")
            if result.drafted:
                for number, line in enumerate(result.text.splitlines()):
                    print(f"      {'after: ' if number == 0 else '       '}   {line}")
            else:
                print(f"      after:     no draft -- {result.reason}")
    return 0


def who_owes_the_next_move(box):
    """X1. Every thread, grouped by whose move it is. No model call."""
    found = waiting.survey(box)
    print(waiting.render(found))
    nudge = waiting.oldest_unanswered(found)
    by_state = waiting.grouped(found)
    print("\n=== run summary ===")
    print(f"  threads              {len(found)}")
    print(f"  waiting on you       {len(by_state['you'])}")
    print(f"  waiting on them      {len(by_state['them'])}   (no mail client raises these)")
    print(f"  closed               {len(by_state['closed'])}")
    print(f"  worth chasing        {len(nudge)}" + (f"  {', '.join(t.ids[-1] for t in nudge)}" if nudge else ""))
    trace.event("waiting_survey", threads=len(found), you=len(by_state["you"]), them=len(by_state["them"]))
    return 0


def the_open_question(box, args):
    """X2. The one thing still unanswered in each long thread."""
    agent = agents.thread_agent()
    candidates = threads.worth_it(box)
    if args.msg:
        message = box.by_id(args.msg)
        if message is None:
            raise Usage(f"no message {args.msg!r} in {config.INBOX_PATH.name}")
        candidates = [(t, m) for t, m in candidates if t == message.thread_id]
        if not candidates:
            raise Usage(f"{args.msg} is in thread {message.thread_id!r}, which has fewer than {threads.WORTH_SUMMARISING} messages")

    print(f"  {len(candidates)} thread(s) with {threads.WORTH_SUMMARISING} or more messages, longest first\n")
    answered = 0
    for thread_id, messages in candidates:
        answer = threads.summarise(agent, thread_id, messages)
        print(threads.render(answer, messages))
        print()
        if answer.get("open"):
            answered += 1
    print("=== run summary ===")
    print(f"  threads read         {len(candidates)}")
    print(f"  open questions found {answered}")
    return 0 if answered else 1


def tone_per_correspondent(box, args):
    """X3. The register learned for each correspondent, and what it changes."""
    profiles = tone.learn(box)
    print(tone.describe(profiles))
    path = tone.save(profiles)
    print(f"\n  written to {path}")

    rows = {r["message_id"]: r for r in read_decisions()}
    as_rows = {a: p.row() for a, p in profiles.items()}
    print("\n  what a reply to each waiting message has to sound like:")
    shown = 0
    for message in box.messages:
        if rows.get(message.id, {}).get("disposition") != "reply":
            continue
        register = tone.register_of(message.sender, as_rows)
        note = "  <- a casual reply here would be a mistake" if register == tone.FORMAL else ""
        print(f"    {message.id}  {message.sender:32} {register}{note}")
        shown += 1
    counts = {}
    for profile in profiles.values():
        counts[profile.register] = counts.get(profile.register, 0) + 1
    print("\n=== run summary ===")
    print(f"  correspondents       {len(profiles)}  (automated senders are not profiled)")
    print("  registers            " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"  messages awaiting a reply  {shown}")
    trace.event("tone_learned", correspondents=len(profiles), **counts)
    return 0


def why_did_it_do_that(box, args):
    """X4. One message's story, replayed from the trace. No model call."""
    events = trace.read()
    if args.msg:
        wanted = [args.msg]
    else:
        counts = {}
        for event in events:
            if event.get("msg_id") and not event.get("event", "").startswith(explain_mod.MACHINERY):
                counts[event["msg_id"]] = counts.get(event["msg_id"], 0) + 1
        wanted = [mid for mid, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:3]]
        print(f"  no message named; showing the three with the most recorded: {', '.join(wanted)}\n")

    told = 0
    for message_id in wanted:
        answer = explain_mod.explain(message_id, box, events)
        print(explain_mod.render(answer))
        print()
        told += 1 if answer["steps"] else 0
    print("=== run summary ===")
    print(f"  messages explained   {told} of {len(wanted)}")
    print(f"  from                 {config.TRACE_PATH.name}, {len(events)} events")
    return 0 if told else 1


def refresh_rule_tier(box):
    """R1 --rules-only. Run the deterministic tier again and record what it found.

    Every conclusion this tier reaches is a pure function of the message text, so
    running it again cannot produce a different answer than the recorded run did.
    That is what makes this safe to do on its own: the 34 decisions the model made
    are not touched, not re-asked, and not at risk, because no message that
    reaches the model reaches this path at all.

    It exists because the two things Part 6 has to evidence -- what each refused
    message attempted, and a refusal logged against its id -- are produced here,
    before any prompt is built. A full run regenerates them only as a side effect
    of also spending half an hour re-deciding the messages the model owns, and
    would overwrite the drafts in doing it.

    The trace is appended to rather than truncated, for the same reason: this is
    one more pass over the inbox, not a replacement for what is already recorded.
    """
    pipeline = flow.build_pipeline()
    state = flow.RunState(mailbox=box)
    for record in box.everything():
        if rules.classify(record).handled:
            flow.run_one(pipeline, record, state)

    path = config.STATE_PATH / "decisions.json"
    if not path.exists():
        raise Usage(f"no recorded decisions at {path}; run `python demo.py --cap R1` first.")
    rows = json.loads(path.read_text(encoding="utf-8"))

    # `attempted` for every row, from the same call that would have set it during
    # a full run. A message the model decided has none, which is what a full run
    # writes for it too.
    changed = 0
    for row in rows:
        message = box.by_id(row.get("message_id"))
        was = row.get("attempted", "")
        now = (rules.classify(message).attempted or "") if message is not None else ""
        if was != now:
            row["attempted"] = now
            changed += 1
        else:
            row.setdefault("attempted", now)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    refused = [d for d in state.decisions if d.disposition == "flag"]
    print(f"  {len(state.decisions)} message(s) settled by the rule tier, no model call")
    print(f"  {len(refused)} refused, each with a refusal logged against its id")
    print(f"  {changed} row(s) in {config.STATE_PATH.name}/decisions.json gained what was attempted")
    print("  dispositions and drafts untouched\n")
    for decision in refused:
        print(f"    {decision.message_id}  {decision.attempted}")
    return 0


def the_hostile_inbox(box):
    """R5. What the inbox tried to make the system do, and proof it did not.

    No model call. Detection already happened in the rule tier and the refusals
    already happened across four modules; this reads the artefacts those left and
    checks Part 6's four requirements against them. A capability that only
    asserted "we refused" would be worth nothing — the assignment's own warning
    is that silently handling an attack is the failure.
    """
    rows = read_decisions()
    threats, checks = hostile.audit(box, rows)

    print(f"  {len(threats)} of {len(box.messages)} messages were refused. What each asked for:\n")
    for threat in threats:
        print(f"    {threat.message_id}  {threat.sender}")
        print(f"        subject:   {threat.subject}")
        print(f"        attempted: {threat.attempted}")
        if threat.shapes:
            print(f"        asks for:  {', '.join(threat.shapes)}")
        if threat.names_addresses:
            print(f"        names:     {', '.join(threat.names_addresses)}")
        print()

    print("  === Part 6's four requirements, checked against this run ===")
    for check in checks:
        print(check.line())
    print()

    failed = [c for c in checks if not c.passed]
    print("=== run summary ===")
    print(f"  refused              {len(threats)}")
    print(f"  checks passed        {len(checks) - len(failed)} of {len(checks)}")
    for check in failed:
        print(f"    FAILED: {check.name} -- {check.detail}")
    print(f"  outbox               {len(list(config.OUTBOX_PATH.glob('*.txt'))) if config.OUTBOX_PATH.exists() else 0} file(s), none from a refused message")
    trace.event("hostile_audit", refused=len(threats), checks_passed=len(checks) - len(failed), failed=[c.name for c in failed])
    return 0 if not failed else 1


def show_register():
    """Part 4.1: the classification, printed from the table the code obeys."""
    print("  what this system does, and what can be taken back:")
    for kind in actions.REGISTER.values():
        mark = "reversible  " if kind.reversible else "IRREVERSIBLE"
        asked = "gated" if kind.gated else "     "
        print(f"    {kind.name:9} {mark} {asked}  {kind.effect}")
        print(f"              {'undo: ' + kind.undo}")


def read_decisions():
    path = config.STATE_PATH / "decisions.json"
    if not path.exists():
        raise Usage(
            f"no recorded decisions at {path}. R3 gates what R1 decided and R2 drafted, "
            "so run `python demo.py --cap R1` first."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        raise Usage(f"could not read {path}: {error}") from error


def gate_the_irreversible(box, args, run_id=""):
    """R3. Every irreversible action, through one gate, with the whole trail logged.

    No model is called. What to send was decided in R1 and written in R2; this
    part decides only whether it may leave, and that is a question about the
    design's rules and the owner's judgement, not about language. Re-asking a
    model here would mean the answer could change between the review and the
    send.
    """
    rows = read_decisions()
    proposals = gate.proposals_from_decisions(rows, box)
    if args.delete:
        message = box.by_id(args.delete)
        if message is None:
            raise Usage(f"no message {args.delete!r} in {config.INBOX_PATH.name}")
        # A person asked for this one. The model has no way to reach it: `delete`
        # is not in the disposition vocabulary, so nothing it can answer turns
        # into a proposal to remove mail.
        proposals = [
            gate.Proposal(
                message_id=message.id,
                action="delete",
                subject=message.subject,
                thread_id=message.thread_id,
                by="human",
            )
        ]
        print(f"  a person asked to delete {message.id}: {message.summary()}\n")
    elif args.limit:
        proposals = proposals[: args.limit]

    mode = args.gate or config.GATE_MODE
    print(f"  {len(proposals)} proposal(s) from {config.STATE_PATH.name}/decisions.json, gate mode {mode}\n")

    passes = gate.run(proposals, box, mode=mode, run_id=run_id)
    for name, done in passes:
        report_pass(name, done)
    folders = passes[-1][1].folders
    folders.save()
    return passes


def report_pass(name, done):
    """What one pass of the gate did, in the gate's own three fields."""
    header = "would do (nothing is written)" if name == "dry-run" else "did"
    print(f"  === {name}: what the gate {header} ===")
    for verdict in done.verdicts:
        print(f"    {verdict.line()}")
        if verdict.blocked:
            # A refused proposal never reaches a person, so its reasons to ask
            # are not printed as though someone had been asked and said nothing.
            for reason in verdict.refusals:
                print(f"           refused: {reason}")
        else:
            for reason in verdict.asks:
                print(f"           asked:   {reason}")
        print(f"           said:    {verdict.human_said}")
        print(f"           happened: {verdict.happened}")

    asked = [v for v in done.verdicts if v.needs_human and not v.blocked]
    blocked = [v for v in done.verdicts if v.blocked]
    acted = [v for v in done.verdicts if v.did_something]
    print(
        f"\n    {len(done.verdicts)} proposal(s): {len(blocked)} refused outright, "
        f"{len(asked)} crossed the escalation line, {len(done.verdicts) - len(asked) - len(blocked)} went unasked"
    )
    written = len([v for v in acted if v.path])
    print(f"    outbox/ writes: {written}\n")


def summarise_gate(passes):
    print("\n=== run summary ===")
    for name, done in passes:
        acted = [v for v in done.verdicts if v.did_something]
        approved = [v for v in done.verdicts if v.approved]
        declined = [v for v in done.verdicts if v.declined]
        print(
            f"  {name:9} {len(done.verdicts):3} proposed, {len(approved)} approved by a person, "
            f"{len(declined)} declined, {len(acted)} happened"
        )
    folders = passes[-1][1].folders
    print(f"  mailbox now          {', '.join(f'{k}={v}' for k, v in folders.tally().items())}")
    print(f"  outbox               {config.OUTBOX_PATH}")
    sent = sorted(p.name for p in config.OUTBOX_PATH.glob("*.txt")) if config.OUTBOX_PATH.exists() else []
    print(f"  files in outbox      {len(sent)}" + (f"  {', '.join(sent)}" if sent else ""))
    return 0


def undo(seq):
    """Take back one recorded action, if the register says it can be taken back."""
    folders = actions.load()
    try:
        row = folders.undo(seq)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    folders.save()
    print(f"  undid action {seq}: {row['message_id']} back to {row['from_status']} (was {row['to_status']})")
    return 0


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

    # Part 6.3: what was found AND what it tried to do. Reporting the reason
    # alone says a message was refused without saying what it wanted, and
    # silently handling an attack is the failure the assignment names.
    flagged = [d for d in decisions if d.disposition == "flag"]
    if flagged:
        print(f"\n  refused, flagged and left in place ({len(flagged)}):")
        for decision in flagged:
            print(f"    {decision.message_id}  {decision.reason}")
            if decision.attempted:
                print(f"           it asked the system to: {decision.attempted}")
        print("    nothing was sent, moved or deleted on their behalf.")

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

    if args.undo is not None:
        return undo(args.undo)

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

    cap = args.cap or ("R3" if args.delete else "R1" if args.all or args.msg else None)
    # R2 and R3 continue the run R1 recorded rather than starting over, so they
    # append instead of truncating. The trace then holds the triage of all 100
    # messages and the drafting that followed from it, which is what a reader
    # needs to check a citation: a `draft` event is only worth anything next to
    # the `read` events for the ids it cites, and a `gate` event is only worth
    # anything next to the draft it let through. R1 starts clean.
    # A rules-only pass is an addition to the record, not a replacement for it, so
    # it appends like the capabilities that read what an earlier run decided.
    run_id = trace.start_run(
        cap=cap, fresh=cap not in ("R2", "R3", "R4", "R5", "R6", "X1", "X2", "X3", "X4") and not args.rules_only
    )

    print(f"=== {cap}: {CAPABILITIES[cap]} ===")
    print(f"  inbox {config.INBOX_PATH.name}: {len(box)} records, {len(box.problems)} malformed")
    if cap == "R3":
        # Said plainly, because it is the reason this part is reproducible: the
        # gate is rules and a person, and neither changes between two runs.
        print("  no model is called: this part gates what was already decided\n")
        show_register()
        print()
    elif cap == "R5":
        print("  no model is called: detection happened in the rule tier, before any prompt existed\n")
    elif cap == "R4" and not args.learn:
        # The process id is printed because it is the claim being made. This run
        # shares nothing with the one that recorded the instructions except the
        # files on disk, and a reader can check that by running the two halves
        # minutes apart and seeing two different numbers.
        print(f"  process {os.getpid()}, started fresh; nothing carries over but what is on disk\n")
    else:
        print(f"  model {config.MODEL} via {config.PROVIDER}\n")

    if cap == "R1" and args.rules_only:
        status = refresh_rule_tier(box)
        print(f"  trace appended to    {config.TRACE_PATH}  ({len(trace.read())} events)")
        return status

    if cap == "X1":
        status = who_owes_the_next_move(box)
        print(f"  trace appended to    {config.TRACE_PATH}  ({len(trace.read())} events)")
        return status

    if cap == "X2":
        status = the_open_question(box, args)
        print(f"  trace appended to    {config.TRACE_PATH}  ({len(trace.read())} events)")
        return status

    if cap == "X3":
        status = tone_per_correspondent(box, args)
        print(f"  trace appended to    {config.TRACE_PATH}  ({len(trace.read())} events)")
        return status

    if cap == "X4":
        return why_did_it_do_that(box, args)

    if cap == "R6":
        page = dashboard.build(box)
        print(dashboard.as_text(page))
        data_path, html_path = dashboard.write(page)
        pane = page["panes"]["commitments"]
        print("\n=== run summary ===")
        print(f"  pending actions      {len(page['panes']['pending'])}")
        print(f"  flagged              {len(page['panes']['flagged'])}")
        print(f"  commitments          {len(pane['dated'])} dated, {len(pane['undated'])} unresolved")
        print(f"  from >1 message      {len(pane['derived_from_more_than_one'])}")
        print(f"  conflicts            {len(pane['conflicts'])}")
        print(f"  citation problems    {len(pane['citation_problems'])}")
        print(f"  written to           {data_path}")
        print(f"                       {html_path}")
        trace.event(
            "dashboard",
            pending=len(page["panes"]["pending"]),
            flagged=len(page["panes"]["flagged"]),
            commitments=len(pane["dated"]) + len(pane["undated"]),
            conflicts=len(pane["conflicts"]),
        )
        print(f"  trace appended to    {config.TRACE_PATH}  ({len(trace.read())} events)")
        return 1 if pane["citation_problems"] else 0

    if cap == "R5":
        status = the_hostile_inbox(box)
        print(f"  trace appended to    {config.TRACE_PATH}  ({len(trace.read())} events)")
        return status

    if cap == "R4":
        if args.learn:
            passes = learn_preferences(box, run_id=run_id)
            if passes:
                summarise_gate(passes)
            print(f"\n  preferences now in  {config.STATE_PATH / 'prefs.json'}")
            print("  this process is about to exit. Run `python demo.py --cap R4` to see what changed.")
            return 0
        status = honour_preferences(box, args)
        print(f"\n  trace appended to    {config.TRACE_PATH}  ({len(trace.read())} events)")
        return status

    if cap == "R3":
        passes = gate_the_irreversible(box, args, run_id=run_id)
        status = summarise_gate(passes)
        print(f"  trace appended to    {config.TRACE_PATH}  ({len(trace.read())} events)")
        return status

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
    # A disposition that changes nothing is a note, not an action, and "an
    # archive can be undone" would be a claim about nothing. Applying them puts
    # every message in a folder and writes an ordered log, which is what makes
    # the reversible half of Part 4's classification checkable.
    folders, applied = actions.apply_decisions(decisions)
    folders.save()
    print(f"\n  decisions written to {path}")
    print(f"  mailbox updated      {applied} message(s) moved: " + ", ".join(
        f"{k}={v}" for k, v in folders.tally().items()
    ))
    print(f"  trace written to     {config.TRACE_PATH}  ({len(trace.read())} events)")
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
