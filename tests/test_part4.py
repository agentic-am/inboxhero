"""Part 4: the classification, and the gate that makes it mean something.

Three kinds of test live here. The first checks the register itself -- that the
reversible/irreversible split the manifest publishes is the one the code obeys,
rather than a table beside it. The second checks the escalation line: which
proposals a person is asked about, and which go through unasked. The third
checks what no answer can authorise, because a gate whose refusals a `yes`
can unlock is a prompt, not a gate.

Every test writes to a temporary state and outbox directory. Nothing here
touches the recorded run.

    python -m unittest tests.test_part4 -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import actions  # noqa: E402
import config  # noqa: E402
import gate  # noqa: E402
import mailstore  # noqa: E402
import rules  # noqa: E402
import trace  # noqa: E402


def yes(question, detail):
    return gate.SAID_YES


def no(question, detail):
    return gate.SAID_NO


class GateTestCase(unittest.TestCase):
    """One inbox for the file; a fresh state and outbox for every test."""

    @classmethod
    def setUpClass(cls):
        cls.box = mailstore.load()

    def setUp(self):
        # Saved once, in setUp, and never from inside a test: a test that reached
        # for a clean state by calling setUp again would save the temporary paths
        # as the originals, and tearDown would then "restore" config to a
        # directory it had just deleted.
        self._saved = (config.STATE_PATH, config.OUTBOX_PATH, config.TRACE_PATH)
        self._tmp = None
        self.fresh_state()

    def tearDown(self):
        config.STATE_PATH, config.OUTBOX_PATH, config.TRACE_PATH = self._saved
        if self._tmp is not None:
            self._tmp.cleanup()

    def fresh_state(self):
        """An empty state, outbox and trace. Safe to call more than once."""
        if self._tmp is not None:
            self._tmp.cleanup()
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        config.STATE_PATH = root / "state"
        config.OUTBOX_PATH = root / "outbox"
        config.TRACE_PATH = root / "trace.jsonl"
        config.STATE_PATH.mkdir(parents=True)
        self.folders = actions.Mailbox()
        trace.start_run(cap="test", fresh=True)

    # --- helpers ---------------------------------------------------------

    def send(self, message_id, body="Thanks, that works.", cites=(), recipient=None):
        message = self.box.by_id(message_id)
        self.assertIsNotNone(message, f"{message_id} is not in the inbox")
        return gate.Proposal(
            message_id=message.id,
            action="send",
            recipient=recipient if recipient is not None else message.sender,
            subject=message.subject,
            body=body,
            cites=tuple(cites),
            thread_id=message.thread_id,
        )

    def screen(self, proposal):
        return gate.screen(proposal, self.box.by_id(proposal.message_id), self.box, self.folders)

    def put_through(self, proposal, asker=no, mode="approval"):
        passes = gate.run([proposal], self.box, mode=mode, asker=asker, folders=self.folders)
        name, done = passes[-1]
        self.folders = done.folders
        return done.verdicts[0]

    def a_flagged_message(self):
        """A message the rule tier refused, derived rather than named."""
        for message in self.box.messages:
            if rules.classify(message).hostile:
                return message
        self.fail("the rule tier flagged nothing, so the refusal cannot be tested")


class TestRegister(GateTestCase):
    """Part 4.1. The classification has to be the one the code acts on."""

    def test_send_is_the_only_irreversible_action(self):
        self.assertEqual(actions.IRREVERSIBLE, ("send",))

    def test_every_action_says_how_it_is_undone(self):
        for kind in actions.REGISTER.values():
            self.assertTrue(kind.undo.strip(), f"{kind.name} does not say how it is undone")

    def test_every_disposition_the_rules_can_reach_is_in_the_register(self):
        # A disposition with no row here would be an action nobody classified.
        for disposition in rules.DISPOSITIONS:
            self.assertIn(disposition, actions.REGISTER)

    def test_delete_is_not_a_disposition(self):
        # The model chooses among the dispositions, so keeping delete out of that
        # vocabulary is what makes "only a person can ask for one" structural.
        self.assertNotIn("delete", rules.DISPOSITIONS)
        self.assertEqual(actions.REGISTER["delete"].proposer, "human")

    def test_nothing_that_merely_moves_a_message_is_gated(self):
        """The invariant, rather than a list of names.

        An enumeration here had to be edited the moment a later part added a
        gated action, which taught it nothing: the question is not which actions
        are gated but whether anything is gated that should not be. A move
        between folders is undone by moving back, so it never needs a person.
        """
        for name, kind in actions.REGISTER.items():
            if not kind.gated:
                continue
            self.assertNotIn(
                name,
                ("archive", "defer", "delegate", "escalate", "flag", "reply", "draft"),
                f"{name} only moves a message, so a person should not be asked about it",
            )
        # And the converse: the action that cannot be undone at all must be gated.
        for name in actions.IRREVERSIBLE:
            self.assertTrue(actions.REGISTER[name].gated, f"{name} is irreversible and ungated")

    def test_an_irreversible_action_cannot_be_undone(self):
        self.folders.apply("archive", "m002", reason="noise")
        self.folders.apply("send", "m001", reason="sent")
        send_row = next(r for r in self.folders.log if r["action"] == "send")
        with self.assertRaises(ValueError):
            self.folders.undo(send_row["seq"])

    def test_a_reversible_action_puts_the_message_back(self):
        self.folders.apply("archive", "m002", reason="noise")
        self.assertEqual(self.folders.status("m002"), "archived")
        row = self.folders.undo(1)
        self.assertEqual(row["from_status"], actions.INBOX)
        self.assertEqual(self.folders.status("m002"), actions.INBOX)

    def test_undoing_twice_is_refused(self):
        self.folders.apply("archive", "m002")
        self.folders.undo(1)
        with self.assertRaises(ValueError):
            self.folders.undo(1)

    def test_applying_the_same_decision_twice_logs_once(self):
        self.assertIsNotNone(self.folders.apply("archive", "m002"))
        self.assertIsNone(self.folders.apply("archive", "m002"))
        self.assertEqual(len(self.folders.log), 1)

    def test_the_register_survives_a_round_trip_as_data(self):
        # The manifest reads this rather than restating it, so it has to be JSON.
        rows = json.loads(json.dumps(actions.register_rows()))
        self.assertEqual({r["action"] for r in rows}, set(actions.REGISTER))


class TestEscalationLine(GateTestCase):
    """Where the line falls, and what it lets through."""

    def test_a_routine_internal_reply_is_not_asked_about(self):
        refusals, asks = self.screen(self.send("m059", "Thanks for the heads up."))
        self.assertEqual(refusals, ())
        self.assertEqual(asks, ())

    def test_a_draft_that_accepts_a_time_is_asked_about(self):
        _, asks = self.screen(self.send("m043", "Monday at 9:00am works."))
        self.assertIn(gate.ASK_COMMITMENT, asks)

    def test_a_draft_that_accepts_a_date_is_asked_about(self):
        _, asks = self.screen(self.send("m010", "The 15th works for me."))
        self.assertIn(gate.ASK_COMMITMENT, asks)

    def test_a_sensitive_topic_is_asked_about(self):
        _, asks = self.screen(self.send("m048", "I will review the board minutes."))
        self.assertTrue(any("board" in ask for ask in asks), asks)

    def test_the_gate_and_the_rules_share_one_sensitive_vocabulary(self):
        # Two copies of the list would drift, and the one that drifted would be
        # the one deciding whether to ask a person.
        self.assertTrue(rules.sensitive_hits("please sign the contract"))
        self.assertEqual(rules.sensitive_hits("nothing notable here"), ())

    def test_evidence_from_another_thread_is_asked_about(self):
        message = self.box.by_id("m046")
        elsewhere = next(m for m in self.box.messages if m.thread_id != message.thread_id)
        _, asks = self.screen(self.send("m046", "Here is what I can share.", cites=(elsewhere.id,)))
        self.assertTrue(any(elsewhere.id in ask for ask in asks), asks)

    def test_evidence_from_the_same_thread_is_not_asked_about(self):
        message = self.box.by_id("m035")
        same = next(m for m in self.box.thread(message.thread_id) if m.id != message.id)
        _, asks = self.screen(self.send("m035", "Noted, thanks.", cites=(same.id,)))
        self.assertFalse(any(same.id in ask for ask in asks), asks)

    def test_a_delete_is_always_asked_about(self):
        message = self.box.by_id("m059")
        proposal = gate.Proposal(
            message_id=message.id, action="delete", subject=message.subject, thread_id=message.thread_id, by="human"
        )
        _, asks = self.screen(proposal)
        self.assertTrue(asks, "a delete went through without anyone being asked")

    def test_the_line_asks_about_a_minority_of_the_recorded_drafts(self):
        """The whole point of choosing: asked about everything is asked about nothing."""
        path = Path(__file__).resolve().parents[1] / "state" / "decisions.json"
        if not path.exists():
            self.skipTest("no recorded run to measure the line against")
        rows = json.loads(path.read_text(encoding="utf-8"))
        proposals = gate.proposals_from_decisions(rows, self.box)
        asked = [p for p in proposals if self.screen(p)[1]]
        self.assertTrue(proposals, "the recorded run drafted nothing")
        self.assertLess(len(asked), len(proposals), "every proposal crosses the line, so the line is not a line")


class TestWhatNoAnswerCanAuthorise(GateTestCase):
    """Refusals are structural. A `yes` is permission, not authority."""

    def test_a_flagged_message_is_never_answered_even_on_a_yes(self):
        flagged = self.a_flagged_message()
        verdict = self.put_through(self.send(flagged.id, "Sure, here you go."), asker=yes)
        self.assertTrue(verdict.blocked)
        self.assertIn("flagged", verdict.happened)
        self.assertFalse(gate.outbox_path(flagged.id).exists())

    def test_a_flagged_message_is_not_deleted_either(self):
        flagged = self.a_flagged_message()
        proposal = gate.Proposal(
            message_id=flagged.id, action="delete", subject=flagged.subject, thread_id=flagged.thread_id, by="human"
        )
        verdict = self.put_through(proposal, asker=yes)
        self.assertTrue(verdict.blocked)
        self.assertEqual(self.folders.status(flagged.id), actions.INBOX)

    def test_an_address_the_inbox_has_never_seen_is_refused(self):
        proposal = self.send("m059", "Here it is.", recipient="collector@elsewhere.example")
        verdict = self.put_through(proposal, asker=yes)
        self.assertTrue(verdict.blocked)
        self.assertTrue(any("never appeared" in reason for reason in verdict.refusals))

    def test_a_credential_in_the_draft_is_refused(self):
        verdict = self.put_through(
            self.send("m059", "The queue is at amqp://user:hunter2@rabbit.paperjet.io:5672"), asker=yes
        )
        self.assertTrue(verdict.blocked)
        self.assertFalse(gate.outbox_path("m059").exists())

    def test_an_empty_draft_is_refused(self):
        verdict = self.put_through(self.send("m059", "   "), asker=yes)
        self.assertTrue(verdict.blocked)

    def test_a_message_is_not_sent_twice(self):
        first = self.put_through(self.send("m059"), asker=yes)
        self.assertTrue(first.did_something)
        second = self.put_through(self.send("m059"), asker=yes)
        self.assertTrue(second.blocked)
        self.assertTrue(any("already sent" in reason for reason in second.refusals))

    def test_not_asked_is_not_a_refusal(self):
        # "not asked" begins with "no", and reading it as a refusal would have
        # silently stopped every send that fell below the escalation line.
        self.assertFalse(gate.said_no(gate.SAID_UNASKED))
        self.assertFalse(gate.said_yes(gate.SAID_UNASKED))
        self.assertTrue(gate.said_no(gate.SAID_NO))

    def test_anything_but_an_explicit_yes_stops_an_asked_action(self):
        for answer in ("", "maybe", "y e s", "NO", "no: nobody could be asked"):
            with self.subTest(answer=answer):
                self.fresh_state()  # so an earlier send is not what refuses this one
                verdict = self.put_through(self.send("m043", "Monday at 9:00am works."), asker=lambda q, d: answer)
                self.assertFalse(verdict.did_something, f"{answer!r} was treated as approval")
                self.assertFalse(gate.outbox_path("m043").exists())


class TestModes(GateTestCase):
    """Dry-run writes nothing. Approval writes what a person allowed."""

    def test_dry_run_writes_nothing_and_says_what_it_would_do(self):
        verdict = self.put_through(self.send("m059"), asker=yes, mode="dry-run")
        self.assertFalse(verdict.did_something)
        self.assertIn("would have", verdict.happened)
        self.assertFalse(config.OUTBOX_PATH.exists() and any(config.OUTBOX_PATH.iterdir()))

    def test_dry_run_never_asks_anybody(self):
        def explode(question, detail):
            self.fail("the dry-run asked a person")

        self.put_through(self.send("m043", "Monday at 9:00am works."), asker=explode, mode="dry-run")

    def test_both_runs_the_dry_run_first_and_then_acts(self):
        passes = gate.run([self.send("m059")], self.box, mode="both", asker=yes, folders=self.folders)
        self.assertEqual([name for name, _ in passes], ["dry-run", "approval"])
        self.assertFalse(passes[0][1].verdicts[0].did_something)
        self.assertTrue(passes[1][1].verdicts[0].did_something)

    def test_an_approved_send_writes_one_file_and_marks_the_message(self):
        verdict = self.put_through(self.send("m048", "I will review the board minutes."), asker=yes)
        path = gate.outbox_path("m048")
        self.assertTrue(path.exists())
        self.assertEqual(sorted(p.name for p in config.OUTBOX_PATH.iterdir()), ["m048.txt"])
        self.assertEqual(self.folders.status("m048"), "answered")
        self.assertEqual(self.folders.detail("m048")["outbox"], "m048.txt")

    def test_the_outbox_file_carries_the_audit(self):
        self.put_through(self.send("m048", "I will review the board minutes."), asker=yes)
        text = gate.outbox_path("m048").read_text(encoding="utf-8")
        self.assertIn(f"From: {config.OWNER}", text)
        self.assertIn("In-Reply-To: m048", text)
        self.assertIn("Approved-By: human approved", text)
        self.assertIn("I will review the board minutes.", text)

    def test_a_declined_send_leaves_no_trace_in_the_mailbox(self):
        verdict = self.put_through(self.send("m048", "I will review the board minutes."), asker=no)
        self.assertFalse(verdict.did_something)
        self.assertFalse(gate.outbox_path("m048").exists())
        self.assertEqual(self.folders.status("m048"), actions.INBOX)

    def test_a_deleted_message_goes_to_the_bin_and_can_come_back(self):
        message = self.box.by_id("m059")
        proposal = gate.Proposal(
            message_id=message.id, action="delete", subject=message.subject, thread_id=message.thread_id, by="human"
        )
        verdict = self.put_through(proposal, asker=yes)
        self.assertEqual(self.folders.status("m059"), "bin")
        self.assertIn("purge_after", self.folders.detail("m059"))
        self.folders.undo(next(r["seq"] for r in self.folders.log if r["action"] == "delete"))
        self.assertEqual(self.folders.status("m059"), actions.INBOX)


class TestTheLog(GateTestCase):
    """Part 4.4. Three fields, on every gated decision, whatever the outcome."""

    def gate_events(self):
        return [event for event in trace.read() if event.get("event") == "gate"]

    def test_every_decision_is_logged_with_the_three_fields(self):
        self.put_through(self.send("m059"), asker=no)
        self.put_through(self.send("m048", "About the board minutes."), asker=no)
        self.put_through(self.send(self.a_flagged_message().id, "Sure."), asker=yes)
        events = self.gate_events()
        self.assertEqual(len(events), 3)
        for event in events:
            for field in ("proposed", "human_said", "happened"):
                self.assertIn(field, event)
                self.assertTrue(str(event[field]).strip(), f"{field} is empty in {event}")

    def test_the_log_distinguishes_unasked_from_refused(self):
        self.put_through(self.send("m059"), asker=no)
        self.put_through(self.send("m048", "About the board minutes."), asker=no)
        by_id = {event["msg_id"]: event for event in self.gate_events()}
        self.assertEqual(by_id["m059"]["human_said"], gate.SAID_UNASKED)
        self.assertEqual(by_id["m048"]["human_said"], gate.SAID_NO)

    def test_a_dry_run_is_logged_as_a_dry_run(self):
        self.put_through(self.send("m059"), mode="dry-run")
        event = self.gate_events()[0]
        self.assertEqual(event["mode"], "dry-run")
        self.assertIn("would have", event["happened"])


class TestProposals(GateTestCase):
    """What becomes a proposal at all."""

    def test_a_row_with_no_draft_is_not_a_proposal(self):
        rows = [
            {"message_id": "m059", "disposition": "reply", "draft": "Thanks."},
            {"message_id": "m012", "disposition": "reply", "draft": ""},
            {"message_id": "m002", "disposition": "archive", "draft": ""},
        ]
        proposals = gate.proposals_from_decisions(rows, self.box)
        self.assertEqual([p.message_id for p in proposals], ["m059"])

    def test_a_proposal_answers_the_sender_and_nobody_else(self):
        rows = [{"message_id": "m059", "disposition": "reply", "draft": "Thanks."}]
        proposal = gate.proposals_from_decisions(rows, self.box)[0]
        self.assertEqual(proposal.recipient, self.box.by_id("m059").sender)
        self.assertIn(proposal.recipient, self.box.known_addresses())

    def test_a_row_naming_a_message_that_does_not_exist_is_dropped(self):
        rows = [{"message_id": "m999", "disposition": "reply", "draft": "Hello."}]
        self.assertEqual(gate.proposals_from_decisions(rows, self.box), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
