"""Part 2 tests: the mail store, the rule tier, the validator and the CLI.

No network and no model: the agent is replaced by a stub that returns whatever
a test tells it to, which is how the bad-answer paths are exercised.

Run:  python -m unittest discover -s tests -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agents  # noqa: E402
import config  # noqa: E402
import demo  # noqa: E402
import flow  # noqa: E402
import mailstore  # noqa: E402
import rules  # noqa: E402
import trace  # noqa: E402

GOOD_RECORD = {
    "id": "m001",
    "thread_id": "t-api",
    "from": "raghav@paperjet.io",
    "to": "sam@paperjet.io",
    "subject": "Staging is down again",
    "timestamp": "2026-09-02T09:12:00",
    "unread": True,
    "body": "Sam, staging has been throwing 500s since last night.",
}


def record(**overrides):
    merged = dict(GOOD_RECORD)
    merged.update(overrides)
    return merged


def write_inbox(directory, records):
    path = Path(directory) / "inbox.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


class Sandbox(unittest.TestCase):
    """A temp state dir and a temp inbox, so no test touches the real files."""

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in config._DEFAULTS}
        for key in config._DEFAULTS:
            os.environ.pop(key, None)
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["STATE_DIR"] = self.tmp.name
        os.environ["TRACE_FILE"] = str(Path(self.tmp.name) / "trace.jsonl")
        self._env_file = config.ENV_FILE
        config.ENV_FILE = Path(self.tmp.name) / "absent.env"
        config.reload()

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        config.ENV_FILE = self._env_file
        config.reload()
        self.tmp.cleanup()

    def use_inbox(self, records):
        os.environ["INBOX_FILE"] = str(write_inbox(self.tmp.name, records))
        config.reload()
        return mailstore.load()

    def set_env(self, **values):
        for key, value in values.items():
            os.environ[key] = value
        config.reload()


# --- the mail store boundary ---------------------------------------------


class MailstoreTests(Sandbox):
    def test_valid_record_becomes_a_message(self):
        box = self.use_inbox([record()])
        self.assertEqual(len(box.messages), 1)
        message = box.by_id("m001")
        self.assertEqual(message.sender, "raghav@paperjet.io")
        self.assertEqual(message.sender_domain, "paperjet.io")
        self.assertTrue(message.internal)
        self.assertFalse(message.from_owner)

    def test_malformed_records_are_kept_and_named_not_dropped(self):
        """Every bad record still reaches the run, and the problem names the field."""
        bad = [
            record(id="m002", body=None),
            record(id="m003", unread="yes"),
            record(id="m004", timestamp="last tuesday"),
            record(id="m005", **{"from": "not-an-address"}),
            {"id": "m006", "subject": "only a subject"},
            {"thread_id": "t", "from": "a@b.c", "to": "d@e.f", "subject": "", "timestamp": "2026-09-02T09:12:00", "body": "", "unread": True},
            "not an object at all",
        ]
        box = self.use_inbox(bad)
        self.assertEqual(len(box.messages), 0)
        self.assertEqual(len(box.problems), 7)
        self.assertEqual(len(box), 7, "malformed records must still be counted as records")
        problems = {p.id: p.problem for p in box.problems}
        self.assertIn("'body' is NoneType", problems["m002"])
        self.assertIn("'unread' is str", problems["m003"])
        self.assertIn("not ISO 8601", problems["m004"])
        self.assertIn("not an address", problems["m005"])
        self.assertIn("missing field(s)", problems["m006"])

    def test_duplicate_ids_are_a_problem_not_a_silent_overwrite(self):
        box = self.use_inbox([record(), record()])
        self.assertEqual(len(box.messages), 1)
        self.assertEqual(box.problems[0].problem, "duplicate id")

    def test_oversized_body_is_rejected_before_it_reaches_a_prompt(self):
        box = self.use_inbox([record(body="x" * (mailstore.MAX_BODY_CHARS + 1))])
        self.assertIn("over the", box.problems[0].problem)

    def test_unusable_file_gives_a_sentence_not_a_traceback(self):
        os.environ["INBOX_FILE"] = str(Path(self.tmp.name) / "missing.json")
        config.reload()
        with self.assertRaises(mailstore.InboxFileError) as caught:
            mailstore.load()
        self.assertIn("No inbox at", str(caught.exception))

        broken = Path(self.tmp.name) / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        os.environ["INBOX_FILE"] = str(broken)
        config.reload()
        with self.assertRaises(mailstore.InboxFileError) as caught:
            mailstore.load()
        self.assertIn("not valid JSON", str(caught.exception))

    def test_thread_walk_returns_only_earlier_messages_oldest_first(self):
        box = self.use_inbox(
            [
                record(id="m005", timestamp="2026-09-02T10:05:00"),
                record(id="m001", timestamp="2026-09-02T09:12:00"),
                record(id="m003", timestamp="2026-09-02T09:41:00"),
                record(id="m009", thread_id="t-other", timestamp="2026-09-02T09:00:00"),
            ]
        )
        earlier = box.earlier_in_thread(box.by_id("m005"))
        self.assertEqual([m.id for m in earlier], ["m001", "m003"])


# --- the rule tier --------------------------------------------------------


class RuleTests(Sandbox):
    """Run against the real inbox: these are claims about the graded data."""

    def setUp(self):
        super().setUp()
        os.environ.pop("INBOX_FILE", None)
        config.reload()
        self.box = mailstore.load()
        self.verdicts = {v.message_id: v for v in rules.classify_all(self.box.everything())}

    def test_every_record_gets_a_verdict(self):
        self.assertEqual(len(self.verdicts), len(self.box))

    def test_the_seven_hostile_messages_are_flagged_by_rules_alone(self):
        """No model is consulted about any of these; the rule tier forces flag."""
        for message_id in ("m017", "m024", "m047", "m039", "m021", "m023", "m045"):
            verdict = self.verdicts[message_id]
            self.assertEqual(verdict.disposition, "flag", f"{message_id} should be flagged")
            self.assertTrue(verdict.handled, f"{message_id} must never reach the model")
            self.assertEqual(verdict.allowed, ("flag",))
            self.assertTrue(verdict.attempted, f"{message_id} must record what was attempted")

    def test_legitimate_standing_instructions_are_not_mistaken_for_attacks(self):
        """m041 is addressed to the assistant and is not an attack. m015 likewise."""
        for message_id in ("m041", "m015"):
            verdict = self.verdicts[message_id]
            self.assertNotEqual(verdict.disposition, "flag", f"{message_id} is a real preference")
            self.assertIn("preference_statement", verdict.flags)

    def test_the_board_domain_is_not_condemned_for_sharing_the_brand_name(self):
        """paperjet-board.org looks like the owner's domain but asks for nothing."""
        self.assertNotEqual(self.verdicts["m038"].disposition, "flag")

    def test_noise_is_archived_without_a_model_call(self):
        for message_id in ("m062", "m087", "m104", "m049"):
            verdict = self.verdicts[message_id]
            self.assertEqual(verdict.disposition, "archive")
            self.assertEqual(verdict.rule, "noise")

    def test_an_automated_message_that_wants_an_answer_is_not_noise(self):
        """m061 is a dentist reminder with a slot and a CONFIRM request."""
        self.assertEqual(self.verdicts["m061"].disposition, "defer")

    def test_real_work_is_left_to_the_model(self):
        for message_id in ("m001", "m008", "m010", "m018", "m030", "m043"):
            self.assertIsNone(self.verdicts[message_id].disposition, f"{message_id} needs judgement")

    def test_mail_the_owner_sent_cannot_be_replied_to(self):
        """The narrowing that proves the rule tier is not decoration."""
        for message_id in ("m003", "m044"):
            verdict = self.verdicts[message_id]
            self.assertNotIn("reply", verdict.allowed)
            self.assertIn("sent_by_owner_to_someone_else", verdict.flags)

    def test_the_model_can_never_be_offered_flag(self):
        """Only rules and the validator may flag, so the model cannot un-flag either."""
        for verdict in self.verdicts.values():
            if not verdict.handled:
                self.assertNotIn("flag", verdict.allowed)

    def test_a_malformed_record_escalates(self):
        verdict = rules.classify(mailstore.Malformed("mX", "missing field(s): body"))
        self.assertEqual(verdict.disposition, "escalate")
        self.assertIn("body", verdict.reason)

    def test_tally_reports_what_the_manifest_needs(self):
        numbers = rules.tally(list(self.verdicts.values()))
        self.assertEqual(numbers["records"], 100)
        self.assertEqual(numbers["rule_handled"] + numbers["to_the_model"], 100)
        self.assertGreater(numbers["rule_handled"], 50)


# --- the validator --------------------------------------------------------


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.verdict = rules.RuleVerdict("m001", None, "r", "abstain", allowed=("reply", "archive", "defer"))

    def parse(self, raw):
        return agents.parse_proposal(raw, self.verdict)

    def test_a_good_answer_is_accepted(self):
        got = self.parse('{"disposition": "Reply", "reason": "  Raghav   is blocked. "}')
        self.assertEqual(got, {"disposition": "reply", "reason": "Raghav is blocked."})

    def test_a_fenced_answer_is_accepted(self):
        self.assertEqual(self.parse('```json\n{"disposition":"archive","reason":"noise"}\n```')["disposition"], "archive")

    def test_a_disposition_outside_the_allowed_list_is_refused(self):
        """The narrowing that went into the prompt is applied again to the answer."""
        with self.assertRaises(agents.Rejected) as caught:
            self.parse('{"disposition": "delegate", "reason": "pass it on"}')
        self.assertIn("not in the allowed list", str(caught.exception))

    def test_an_invented_disposition_is_refused(self):
        with self.assertRaises(agents.Rejected):
            self.parse('{"disposition": "send_all", "reason": "go"}')

    def test_the_model_cannot_flag_on_this_path(self):
        with self.assertRaises(agents.Rejected):
            self.parse('{"disposition": "flag", "reason": "looks odd"}')

    def test_missing_pieces_are_refused(self):
        for raw in ('{"reason": "no disposition"}', '{"disposition": "reply"}', '{"disposition": "reply", "reason": "   "}'):
            with self.assertRaises(agents.Rejected):
                self.parse(raw)

    def test_non_json_is_refused(self):
        for raw in ("", "   ", "I think you should reply to this.", "null", "[1,2,3]"):
            with self.assertRaises(agents.Rejected):
                self.parse(raw)

    def test_a_moya_error_string_is_never_read_as_an_answer(self):
        """Moya reports transport failures as text that would otherwise parse as prose."""
        with self.assertRaises(agents.Rejected) as caught:
            self.parse("[OllamaAgent error: connection refused]")
        self.assertIn("error string", str(caught.exception))

    def test_a_long_reason_is_truncated_not_rejected(self):
        got = self.parse(json.dumps({"disposition": "defer", "reason": "x" * 999}))
        self.assertEqual(len(got["reason"]), agents.MAX_REASON_CHARS)


# --- the pipeline ---------------------------------------------------------


class StubAgent:
    """Stands in for the model. Returns the queued answers in order."""

    agent_name = "stub"

    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts = []

    def handle_message(self, message, **kwargs):
        self.prompts.append(message)
        if not self.answers:
            return '{"disposition": "archive", "reason": "stub default"}'
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class PipelineTests(Sandbox):
    def setUp(self):
        super().setUp()
        os.environ.pop("INBOX_FILE", None)
        config.reload()
        trace.start_run(cap="test", fresh=True)
        self.box = mailstore.load()

    def decide(self, message_id, agent):
        pipeline = flow.build_pipeline(agent=agent)
        state = flow.RunState(mailbox=self.box)
        return flow.run_one(pipeline, self.box.by_id(message_id), state)

    def test_a_rule_handled_message_never_calls_the_model(self):
        agent = StubAgent()
        decision = self.decide("m062", agent)
        self.assertEqual(decision.disposition, "archive")
        self.assertEqual(decision.path, "rules")
        self.assertEqual(agent.prompts, [], "the model must not be called for obvious noise")

    def test_a_hostile_message_never_calls_the_model(self):
        agent = StubAgent()
        decision = self.decide("m024", agent)
        self.assertEqual(decision.disposition, "flag")
        self.assertEqual(agent.prompts, [])

    def test_the_prompt_carries_the_rule_verdict_and_marks_the_body_untrusted(self):
        agent = StubAgent('{"disposition": "defer", "reason": "ok"}')
        self.decide("m003", agent)
        prompt = agent.prompts[0]
        self.assertIn("ALLOWED dispositions", prompt)
        self.assertNotIn("reply", prompt.split("ALLOWED dispositions")[1].split("\n")[0])
        self.assertIn("<<<UNTRUSTED MESSAGE id=m003", prompt)
        self.assertIn("<<<END UNTRUSTED MESSAGE id=m003", prompt)

    def test_a_bad_answer_is_retried_once_then_escalated(self):
        agent = StubAgent("not json at all", "still not json")
        decision = self.decide("m001", agent)
        self.assertEqual(len(agent.prompts), 2, "exactly one corrective retry")
        self.assertIn("rejected", agent.prompts[1])
        self.assertEqual(decision.disposition, "escalate")
        self.assertTrue(decision.problem)

    def test_a_retry_that_succeeds_is_used(self):
        agent = StubAgent("sorry, no", '{"disposition": "reply", "reason": "Raghav is blocked"}')
        decision = self.decide("m001", agent)
        self.assertEqual(decision.disposition, "reply")
        self.assertEqual(decision.attempts, 2)

    def test_a_model_that_ignores_the_allowed_list_is_overruled(self):
        """m003 was sent by the owner, so 'reply' was removed. The model says reply anyway."""
        agent = StubAgent('{"disposition": "reply", "reason": "answer it"}', '{"disposition": "reply", "reason": "answer it"}')
        decision = self.decide("m003", agent)
        self.assertEqual(decision.disposition, "escalate")
        self.assertIn("allowed list", decision.problem)

    def test_an_unreachable_model_escalates_rather_than_crashing(self):
        agent = StubAgent(RuntimeError("ollama is down"))
        decision = self.decide("m001", agent)
        self.assertEqual(decision.disposition, "escalate")
        self.assertIn("could not be reached", decision.problem)

    def test_every_record_including_malformed_ends_with_a_disposition(self):
        pipeline = flow.build_pipeline(agent=StubAgent())
        state = flow.RunState(mailbox=self.box)
        records = [mailstore.Malformed("mBAD", "missing field(s): body")] + self.box.everything()[:6]
        for item in records:
            flow.run_one(pipeline, item, state)
        self.assertEqual(len(state.decisions), len(records))
        self.assertTrue(all(d.disposition in rules.DISPOSITIONS for d in state.decisions))
        self.assertTrue(all(d.reason for d in state.decisions))

    def test_the_trace_shows_the_rule_verdict_and_the_decision(self):
        self.decide("m062", StubAgent())
        kinds = [e["event"] for e in trace.read()]
        for expected in ("read", "rule", "decision"):
            self.assertIn(expected, kinds)


class AgentWiringTests(unittest.TestCase):
    def test_the_triage_agent_has_no_tools(self):
        """The architecture claim, asserted rather than described."""
        agent = agents.triage_agent()
        self.assertEqual(agent.discover_tools(), [])
        self.assertIsNone(agent.tool_registry)
        self.assertFalse(agent.is_tool_caller)

    def test_the_registry_names_the_agent(self):
        registry = agents.build_registry()
        self.assertIsNotNone(registry.get_agent("triage"))


# --- the command line -----------------------------------------------------


class CliTests(unittest.TestCase):
    def test_good_arguments_parse(self):
        args = demo.parse_args(["--cap", "r1", "--limit", "5"])
        self.assertEqual(args.cap, "R1")
        self.assertEqual(args.limit, 5)

    def test_bad_arguments_exit_with_status_two(self):
        for argv in ([], ["--cap", "Z9"], ["--cap", "R1", "--limit", "0"]):
            with self.assertRaises(SystemExit) as caught:
                demo.parse_args(argv)
            self.assertEqual(caught.exception.code, 2, f"{argv} should exit 2")


if __name__ == "__main__":
    unittest.main()


# --- batching -------------------------------------------------------------


class BatchTests(Sandbox):
    """BATCH_SIZE > 1 must not weaken any check that applies at size 1."""

    def setUp(self):
        super().setUp()
        os.environ.pop("INBOX_FILE", None)
        config.reload()
        trace.start_run(cap="test", fresh=True)
        self.box = mailstore.load()
        self.records = self.box.everything()
        self.verdicts = {v.message_id: v for v in rules.classify_all(self.records)}

    def group(self, *ids):
        return [self.box.by_id(i) for i in ids]

    def answer(self, *pairs):
        return json.dumps({"decisions": [{"id": i, "disposition": d, "reason": "because"} for i, d in pairs]})

    def test_sensitive_messages_are_never_batched(self):
        """m018 and m055 touch signatures; m008 asks for credentials."""
        eligible = {r.id for r in flow.batchable(self.records, self.verdicts)}
        for message_id in ("m008", "m018", "m055", "m019", "m046"):
            self.assertNotIn(message_id, eligible, f"{message_id} is sensitive and must get its own call")

    def test_rule_handled_and_malformed_records_are_never_batched(self):
        records = self.records + [mailstore.Malformed("mBAD", "missing field(s): body")]
        verdicts = dict(self.verdicts)
        verdicts["mBAD"] = rules.classify(records[-1])
        eligible = {r.id for r in flow.batchable(records, verdicts)}
        self.assertNotIn("mBAD", eligible)
        for message_id in ("m062", "m024", "m061"):
            self.assertNotIn(message_id, eligible)

    def test_each_message_is_checked_against_its_own_allowed_list(self):
        """m003 cannot be replied to. The same batch says reply for both."""
        group = self.group("m003", "m005")
        accepted, failures = agents.parse_batch(
            self.answer(("m003", "reply"), ("m005", "reply")), self.verdicts, [m.id for m in group]
        )
        self.assertEqual(sorted(accepted), ["m005"])
        self.assertIn("allowed list", failures["m003"])

    def test_a_missing_entry_fails_only_that_message(self):
        group = self.group("m005", "m027", "m033")
        accepted, failures = agents.parse_batch(
            self.answer(("m005", "archive"), ("m033", "archive")), self.verdicts, [m.id for m in group]
        )
        self.assertEqual(sorted(accepted), ["m005", "m033"])
        self.assertIn("left this message out", failures["m027"])

    def test_an_id_we_did_not_ask_about_is_ignored(self):
        """A batched answer cannot smuggle in a decision about another message."""
        group = self.group("m005")
        accepted, failures = agents.parse_batch(
            self.answer(("m005", "archive"), ("m024", "archive")), self.verdicts, [m.id for m in group]
        )
        self.assertEqual(sorted(accepted), ["m005"])
        self.assertEqual(failures, {})

    def test_a_duplicate_entry_fails_that_message(self):
        group = self.group("m005")
        accepted, failures = agents.parse_batch(
            self.answer(("m005", "archive"), ("m005", "defer")), self.verdicts, [m.id for m in group]
        )
        self.assertEqual(accepted, {})
        self.assertIn("twice", failures["m005"])

    def test_a_structurally_broken_batch_fails_every_message(self):
        group = self.group("m005", "m027")
        for raw in ("not json", '{"decisions": "nope"}', "[OllamaAgent error: down]"):
            with self.assertRaises(agents.Rejected):
                agents.parse_batch(raw, self.verdicts, [m.id for m in group])

    def test_ask_batch_never_raises_and_reports_every_id(self):
        agent = StubAgent("garbage that is not json")
        accepted, failures = agents.ask_batch(agent, self.group("m005", "m027"), self.verdicts, self.box)
        self.assertEqual(accepted, {})
        self.assertEqual(sorted(failures), ["m005", "m027"])

    def test_the_batch_prompt_marks_every_body_untrusted_and_lists_each_allowed_set(self):
        prompt = agents.build_batch_prompt(self.group("m005", "m027"), self.verdicts, self.box)
        for message_id in ("m005", "m027"):
            self.assertIn(f"<<<UNTRUSTED MESSAGE id={message_id}", prompt)
            self.assertIn(f"ALLOWED for {message_id}", prompt)

    def test_a_batched_answer_is_used_without_a_second_model_call(self):
        agent = StubAgent()
        state = flow.RunState(mailbox=self.box)
        state.batched["m005"] = {"disposition": "archive", "reason": "from the batch"}
        pipeline = flow.build_pipeline(agent=agent)
        decision = flow.run_one(pipeline, self.box.by_id("m005"), state)
        self.assertEqual(decision.disposition, "archive")
        self.assertEqual(decision.reason, "from the batch")
        self.assertEqual(agent.prompts, [], "a batched answer must not trigger another call")

    def test_a_message_the_batch_could_not_answer_falls_back_to_its_own_call(self):
        agent = StubAgent('{"disposition": "defer", "reason": "asked on its own"}')
        state = flow.RunState(mailbox=self.box)  # nothing batched for m005
        pipeline = flow.build_pipeline(agent=agent)
        decision = flow.run_one(pipeline, self.box.by_id("m005"), state)
        self.assertEqual(decision.reason, "asked on its own")
        self.assertEqual(len(agent.prompts), 1)

    def test_batch_size_one_does_no_batching_at_all(self):
        state = flow.RunState(mailbox=self.box)
        summary = flow.prefill_batches(StubAgent(), self.records, state, batch_size=1)
        self.assertEqual(summary["batches"], 0)
        self.assertEqual(state.batched, {})

    def test_config_rejects_a_batch_size_below_one(self):
        self.set_env(BATCH_SIZE="0")
        self.assertTrue(any("BATCH_SIZE" in p for p in config.problems()))
