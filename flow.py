"""The Moya pipeline: one run per message, the same steps every time.

    rule  ->  route  -+->  rule decision                      -+->  record
                      |                                        |
                      +->  prompt  ->  agent  ->  validate  ---+

`route` is a Moya `BranchStep`. When the rule tier reached a conclusion the
model branch is never entered, which is what "route the obvious messages
through rules, not an LLM" means in Part 2: not a cheaper prompt, no prompt.

Every step is either a `FunctionStep` (plain Python) or the single `AgentStep`
(the model), and the validator sits after the agent. That is the shape the
course tutorial recommends: the function step is the safety net, because an
agent can be asked to behave but never made to.

Usage:
    python flow.py     # run the pipeline over five messages
"""

import json
from dataclasses import dataclass, field

from moya.flows.pipeline import Pipeline
from moya.flows.steps import AgentStep, BranchStep, FunctionStep

import agents
import config
import drafting
import mailstore
import retrieval
import rules
import trace


class Steps:
    """A list of steps that behaves like one step.

    Moya's `BranchStep` takes one step per branch, and its `Pipeline` cannot be
    nested because `Pipeline.run` takes a message rather than a context. This is
    the smallest thing that lets the model branch stay three visible steps
    instead of one opaque function.
    """

    def __init__(self, steps, name="steps"):
        self.steps = list(steps)
        self.name = name

    def run(self, ctx):
        for step in self.steps:
            ctx = step.run(ctx)
        return ctx


@dataclass
class Decision:
    """What the system concluded about one message. One row of `decisions.json`."""

    message_id: str
    disposition: str
    reason: str
    path: str  # "rules" or "model"
    rule: str
    flags: list = field(default_factory=list)
    attempts: int = 0
    model_raw: str = ""
    problem: str = ""  # set when the model's answer had to be thrown away
    # What Part 3 retrieved, and what the model actually leaned on. Both are
    # kept: evidence offered but not cited is as much a part of the record as
    # evidence cited, because it is what the decision could have used.
    evidence: list = field(default_factory=list)
    cites: list = field(default_factory=list)
    # Part 3's other half. `draft_reason` is filled instead of `draft` when the
    # inbox did not hold the answer, because "nothing was drafted" is a result
    # that has to be readable, not an empty field.
    draft: str = ""
    draft_cites: list = field(default_factory=list)
    draft_reason: str = ""
    # answered | no_reply | not_known | rejected, or "" when the disposition was
    # never `reply` and drafting was not attempted at all.
    draft_outcome: str = ""

    def line(self):
        note = f"   <- {self.problem}" if self.problem else ""
        return f"{self.message_id:6} {self.disposition:9} [{self.path:5}] {self.reason}{note}"

    def as_row(self):
        return {
            "message_id": self.message_id,
            "disposition": self.disposition,
            "reason": self.reason,
            "path": self.path,
            "rule": self.rule,
            "flags": list(self.flags),
            "attempts": self.attempts,
            "problem": self.problem,
            "evidence": list(self.evidence),
            "cites": list(self.cites),
            "draft": self.draft,
            "draft_cites": list(self.draft_cites),
            "draft_reason": self.draft_reason,
            "draft_outcome": self.draft_outcome,
        }


@dataclass
class RunState:
    """Carried through the pipeline by reference.

    `Pipeline.run` copies the keyword arguments into a fresh metadata dict, so a
    step cannot hand anything back by writing into that dict. It can, however,
    write into an object the dict points at. This is that object.
    """

    mailbox: object = None
    decisions: list = field(default_factory=list)
    # message id -> proposal already obtained from a batched call. `TriageStep`
    # uses one of these instead of calling the model again. Empty when
    # BATCH_SIZE is 1, which is the default.
    batched: dict = field(default_factory=dict)
    # The FTS5 and entity indexes, built once per run rather than once per
    # message: indexing is cheap but not free, and rebuilding it per message
    # would be the one part of retrieval whose cost grows with the inbox.
    _index: object = None

    _drafter: object = None

    @property
    def drafter(self):
        """Built on first use: a run whose messages all archive never needs it.

        A separate agent from the triage one, on the same model. What separates
        them is the system prompt: a model asked to draft under the triage
        prompt answers like a classifier.
        """
        if self._drafter is None:
            self._drafter = agents.drafter_agent()
        return self._drafter

    @property
    def index(self):
        if self._index is None and self.mailbox is not None:
            self._index = retrieval.Index(self.mailbox)
        return self._index

    @property
    def last(self):
        return self.decisions[-1] if self.decisions else None


# --- the steps ------------------------------------------------------------


def rule_step(ctx):
    """The deterministic tier.

    Wrapped so a bug in a rule escalates one message instead of stopping the run:
    Part 2 is graded on every message getting a disposition, and a crash halfway
    through would leave the rest with none.
    """
    record = ctx.metadata["record"]
    try:
        verdict = rules.classify(record)
    except Exception as error:  # noqa: BLE001 - deliberate, see docstring
        verdict = rules.RuleVerdict(
            message_id=getattr(record, "id", "?"),
            disposition="escalate",
            reason=f"the rule tier failed on this message ({error.__class__.__name__}), so a human should look at it",
            rule="rule-error",
            allowed=("escalate",),
        )
    ctx.metadata["verdict"] = verdict
    trace.event(
        "rule",
        msg_id=verdict.message_id,
        disposition=verdict.disposition,
        rule=verdict.rule,
        reason=verdict.reason,
        flags=list(verdict.flags),
        allowed=list(verdict.allowed),
        attempted=verdict.attempted,
    )
    return ctx


def route(ctx):
    """The router: the only thing that decides whether a model is called at all."""
    return "rules" if ctx.metadata["verdict"].handled else "model"


def rule_decision(ctx):
    """The rule branch. The verdict is already the answer, so no prompt is built."""
    verdict = ctx.metadata["verdict"]
    ctx.metadata["decision"] = Decision(
        message_id=verdict.message_id,
        disposition=verdict.disposition,
        reason=verdict.reason,
        path="rules",
        rule=verdict.rule,
        flags=list(verdict.flags),
    )
    return ctx


def retrieve_step(ctx):
    """Part 3. Find what in the inbox grounds this message, before asking anything.

    Only the model branch reaches here, so the two thirds of the inbox the rules
    already settled cost nothing. Retrieval never raises: evidence is grounding,
    not a prerequisite, and a message with none still gets a disposition -- it
    just gets one that has to admit it could not see what was being referred to.
    """
    verdict = ctx.metadata["verdict"]
    state = ctx.metadata["state"]
    record = ctx.metadata["record"]
    try:
        found = retrieval.retrieve(
            state.mailbox,
            record,
            k=config.RETRIEVAL_K,
            index=state.index,
            window_days=config.RETRIEVAL_WINDOW_DAYS,
        )
    except Exception as error:  # noqa: BLE001 - grounding is best-effort, a disposition is not
        trace.event("retrieve", msg_id=verdict.message_id, error=str(error)[:300], evidence=[])
        found = retrieval.Retrieved()

    ctx.metadata["evidence"] = found
    trace.event(
        "retrieve",
        msg_id=verdict.message_id,
        terms=list(found.terms),
        evidence=[
            {"id": item.message_id, "source": item.source, "terms": list(item.terms), "score": item.score}
            for item in found.evidence
        ],
        grounded=bool(found),
        window_days=found.window_days,
        widened=found.widened,
    )
    return ctx


def prompt_step(ctx):
    """Build the model's input from the rule verdict, the evidence and the quoted message."""
    verdict = ctx.metadata["verdict"]
    state = ctx.metadata["state"]
    found = ctx.metadata.get("evidence") or retrieval.Retrieved()
    ctx.output = agents.build_prompt(ctx.metadata["record"], verdict, state.mailbox, found.evidence)
    trace.event(
        "prompt",
        msg_id=verdict.message_id,
        chars=len(ctx.output),
        allowed=list(verdict.allowed),
        evidence=found.ids,
    )
    return ctx


class TriageStep(AgentStep):
    """Moya's `AgentStep`, with the retry the framework does not provide.

    The stock step calls the agent exactly once, hands it only `ctx.output`, and
    has no idea what a usable answer looks like. The validator needs the rule
    verdict as well as the answer, so this override reads the whole context and
    leaves the parse outcome in it for the next step.
    """

    def __init__(self, agent, retries=1, name="triage"):
        super().__init__(agent, name=name)
        self.retries = retries

    def run(self, ctx):
        verdict = ctx.metadata["verdict"]
        ready = ctx.metadata["state"].batched.pop(verdict.message_id, None)
        if ready is not None:
            # A batched call already answered for this message and the answer
            # passed the same validator a single answer would have. No second
            # call, but the validate step still runs.
            trace.event("batched_hit", msg_id=verdict.message_id, disposition=ready["disposition"])
            ctx.metadata["attempt_result"] = agents.Attempted(ready, json.dumps(ready), 1)
            ctx.output = ctx.metadata["attempt_result"].raw
            return ctx

        found = ctx.metadata.get("evidence")
        result = agents.ask(
            self.agent,
            ctx.output,
            verdict,
            thread_id=ctx.thread_id,
            retries=self.retries,
            evidence_ids=tuple(found.ids) if found else (),
        )
        ctx.metadata["attempt_result"] = result
        ctx.output = result.raw
        return ctx


def validate_step(ctx):
    """The safety net after the agent.

    The allowed list that went into the prompt is applied again to the answer, in
    Python. A model that ignores it is overruled and the message is escalated,
    never silently accepted and never silently dropped.
    """
    verdict = ctx.metadata["verdict"]
    result = ctx.metadata.get("attempt_result")
    grounded = ctx.metadata.get("evidence") or retrieval.Retrieved()

    if result is None or not result.ok:
        problem = result.problem if result else "the agent step produced nothing"
        trace.event("validate", msg_id=verdict.message_id, accepted=False, problem=problem)
        ctx.metadata["decision"] = Decision(
            message_id=verdict.message_id,
            disposition="escalate",
            reason="the model did not return a usable decision, so a human should look at this",
            path="model",
            rule=verdict.rule,
            flags=list(verdict.flags),
            attempts=result.attempts if result else 0,
            model_raw=(result.raw[:300] if result else ""),
            problem=problem,
            evidence=grounded.ids,
        )
        return ctx

    trace.event(
        "validate",
        msg_id=verdict.message_id,
        accepted=True,
        disposition=result.proposal["disposition"],
        checked_against=list(verdict.allowed),
        attempts=result.attempts,
        evidence=grounded.ids,
        cites=result.proposal.get("cites", []),
    )
    ctx.metadata["decision"] = Decision(
        message_id=verdict.message_id,
        disposition=result.proposal["disposition"],
        reason=result.proposal["reason"],
        path="model",
        rule=verdict.rule,
        flags=list(verdict.flags),
        attempts=result.attempts,
        model_raw=result.raw[:300],
        evidence=grounded.ids,
        cites=result.proposal.get("cites", []),
    )
    return ctx


def draft_step(ctx):
    """Part 3's answer, for the messages the system decided to reply to.

    Only `reply` reaches here. A message being archived or escalated needs no
    reply written for it, and drafting one anyway would be work nobody asked for
    on the majority of the inbox.

    Nothing is sent. A draft is reversible and this step writes no file; the
    system sends nothing and writes no file here.
    """
    decision = ctx.metadata.get("decision")
    if decision is None or decision.disposition != "reply":
        return ctx

    state = ctx.metadata["state"]
    record = ctx.metadata["record"]
    grounded = ctx.metadata.get("evidence") or retrieval.Retrieved()

    result = drafting.draft(
        state.drafter,
        record,
        grounded.evidence,
        state.mailbox,
        thread_id=f"draft-{record.id}",
    )
    decision.draft = result.text or ""
    decision.draft_cites = list(result.cites)
    decision.draft_reason = result.reason
    decision.draft_outcome = result.outcome
    return ctx


def record_step(ctx):
    """Nothing leaves the pipeline without a disposition. Guaranteed here."""
    decision = ctx.metadata.get("decision")
    if decision is None:  # no branch produced one; silence is not an option
        record = ctx.metadata["record"]
        decision = Decision(
            message_id=getattr(record, "id", "?"),
            disposition="escalate",
            reason="the pipeline produced no decision for this message",
            path="none",
            rule="missing",
        )
    trace.event(
        "decision",
        msg_id=decision.message_id,
        disposition=decision.disposition,
        reason=decision.reason,
        path=decision.path,
        rule=decision.rule,
    )
    ctx.metadata["state"].decisions.append(decision)
    ctx.output = decision.disposition
    return ctx


# --- assembling it --------------------------------------------------------


def build_pipeline(agent=None, event_bus=None, retries=1):
    """The one pipeline this system runs, in the order the design doc lists."""
    agent = agent or agents.triage_agent()
    return Pipeline(
        steps=[
            FunctionStep(rule_step, name="rule"),
            BranchStep(
                condition=route,
                branches={
                    "rules": FunctionStep(rule_decision, name="rule_decision"),
                    "model": Steps(
                        [
                            FunctionStep(retrieve_step, name="retrieve"),
                            FunctionStep(prompt_step, name="prompt"),
                            TriageStep(agent, retries=retries),
                            FunctionStep(validate_step, name="validate"),
                            FunctionStep(draft_step, name="draft"),
                        ],
                        name="model_path",
                    ),
                },
                name="route",
            ),
            FunctionStep(record_step, name="record"),
        ],
        name="inboxhero",
        event_bus=event_bus,
    )


def batchable(records, verdicts):
    """The messages that may share a call: model-path, valid, and not sensitive.

    A sensitive message is one the rules marked as touching money, credentials,
    a contract, legal or press. Those are exactly the ones where a decision
    swayed by a neighbouring message would cost the most, so they always get
    their own context however BATCH_SIZE is set.
    """
    out = []
    for record in records:
        verdict = verdicts.get(getattr(record, "id", None))
        if verdict is None or verdict.handled:
            continue
        if isinstance(record, mailstore.Malformed):
            continue
        if "sensitive_topic" in verdict.flags:
            continue
        out.append(record)
    return out


def prefill_batches(agent, records, state, batch_size, verdicts=None):
    """Answer as many model-path messages as possible in grouped calls.

    Fills `state.batched`. Anything a batch could not answer is simply left out,
    and the pipeline then asks about that message on its own. Returns a small
    summary for the run output.
    """
    if batch_size <= 1:
        return {"batches": 0, "batched": 0, "fell_back": 0}

    verdicts = verdicts or {v.message_id: v for v in rules.classify_all(records)}
    eligible = batchable(records, verdicts)
    groups = [eligible[i : i + batch_size] for i in range(0, len(eligible), batch_size)]

    batched = fell_back = 0
    for group in groups:
        accepted, failures = agents.ask_batch(agent, group, verdicts, state.mailbox)
        state.batched.update(accepted)
        batched += len(accepted)
        fell_back += len(failures)
    return {"batches": len(groups), "batched": batched, "fell_back": fell_back}


def run_one(pipeline, record, state):
    """Push one record through the pipeline and return its Decision."""
    trace.event("read", msg_id=getattr(record, "id", "?"), subject=getattr(record, "subject", ""))
    pipeline.run(
        thread_id=getattr(record, "thread_id", "malformed"),
        message=getattr(record, "subject", ""),
        record=record,
        state=state,
    )
    return state.last


if __name__ == "__main__":
    import config

    for line in config.check():
        print(f"  warning: {line}")
    box = mailstore.load()
    trace.start_run(cap="flow-demo", fresh=True)
    state = RunState(mailbox=box)
    pipeline = build_pipeline()

    print("=== five messages through the pipeline ===")
    for message_id in ("m096", "m024", "m001", "m044", "m012"):
        decision = run_one(pipeline, box.by_id(message_id), state)
        print(" ", decision.line() if decision else f"{message_id}: NO DECISION")
    print(f"\ntrace events written: {len(trace.read())}")
