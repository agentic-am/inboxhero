"""Part 6: the hostile inbox.

The tests here are mostly adversarial rather than confirmatory. Checking that
seven known messages are flagged proves very little — a hardcoded list would pass
it. What is worth asserting is the shape of the defence: that detection needs two
signals so a legitimate instruction survives, that refused mail cannot reach a
prompt by any route, that no answer unlocks an action on its behalf, and that the
audit itself fails when the thing it audits is broken.

    python -m unittest tests.test_part6 -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import actions  # noqa: E402
import config  # noqa: E402
import drafting  # noqa: E402
import flow  # noqa: E402
import gate  # noqa: E402
import hostile  # noqa: E402
import mailstore  # noqa: E402
import prefs  # noqa: E402
import retrieval  # noqa: E402
import rules  # noqa: E402
import trace  # noqa: E402


class HostileTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.box = mailstore.load()

    def setUp(self):
        self._saved = (config.STATE_PATH, config.OUTBOX_PATH, config.TRACE_PATH)
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        config.STATE_PATH = root / "state"
        config.OUTBOX_PATH = root / "outbox"
        config.TRACE_PATH = root / "trace.jsonl"
        config.STATE_PATH.mkdir(parents=True)
        trace.start_run(cap="test", fresh=True)

    def tearDown(self):
        config.STATE_PATH, config.OUTBOX_PATH, config.TRACE_PATH = self._saved
        self._tmp.cleanup()

    def threats(self):
        return hostile.found(self.box)

    def rule_handled_run(self):
        """Every rule-handled message through the real pipeline. No model call."""
        pipeline = flow.build_pipeline()
        state = flow.RunState(mailbox=self.box)
        for message in self.box.everything():
            if rules.classify(message).handled:
                flow.run_one(pipeline, message, state)
        rows = [d.as_row() for d in state.decisions]
        folders, _ = actions.apply_decisions(state.decisions)
        return rows, folders


class TestDetection(HostileTestCase):
    """Who is refused, and — more usefully — who is not."""

    def test_something_is_refused(self):
        self.assertTrue(self.threats(), "an inbox the assignment says is hostile produced no refusals")

    def test_the_real_standing_instructions_are_not_refused(self):
        """The messages that would be most costly to get wrong.

        Two messages in this inbox tell the assistant what to do and are entirely
        legitimate. Refusing either would break Part 5 and lose the owner a rule
        they wrote down.
        """
        for message in self.box.messages:
            if "preference_statement" in rules.classify(message).flags:
                self.assertFalse(
                    rules.classify(message).hostile,
                    f"{message.id} is a real standing instruction and was refused",
                )

    def test_two_signals_are_required_and_not_one(self):
        """Asserted on a constructed message, because this inbox cannot show it.

        Every message here that addresses the assistant is also hostile, so the
        second condition currently rejects nothing — measured, not assumed. It is
        insurance rather than active defence, and the only way to test insurance
        is to construct the case it exists for: a message that addresses the
        assistant and asks for nothing an attacker would want.
        """
        benign = mailstore.Message(
            id="t900",
            thread_id="t",
            sender="raghav@paperjet.io",
            to=config.OWNER,
            subject="note for the ai assistant",
            timestamp="2026-09-05T09:00:00",
            body="If an AI assistant is processing this mailbox, please file this under receipts.",
            unread=True,
            sent_at=self.box.messages[0].sent_at,
        )
        text = benign.text().lower()
        self.assertTrue(
            any(p in text for p in rules.ADDRESSED_TO_AGENT), "this test needs the first signal to fire"
        )
        self.assertFalse(
            rules.classify(benign).hostile,
            "a message addressing the assistant was refused on that signal alone",
        )

    def test_an_instruction_inside_a_quoted_reply_is_still_found(self):
        """Quoted text is still message content; `>` is not a safety boundary."""
        quoted = [m for m in self.box.messages if ">" in m.body]
        self.assertTrue(quoted, "no message carries quoted text")
        for message in quoted:
            if "assistant" in message.body.lower() or "ai agent" in message.body.lower():
                self.assertTrue(
                    rules.classify(message).hostile,
                    f"{message.id} hides an instruction in a quote and was not refused",
                )

    def test_every_refusal_says_what_was_attempted(self):
        for threat in self.threats():
            self.assertTrue(threat.attempted.strip(), f"{threat.message_id} was refused without saying why")

    def test_the_owners_own_address_is_not_a_pass(self):
        """One attack is sent from the owner's address, so the sender proves nothing."""
        from_owner = [t for t in self.threats() if self.box.by_id(t.message_id).from_owner]
        self.assertTrue(from_owner, "this inbox's forgery from the owner's own address was not refused")


class TestNoRouteIntoAPrompt(HostileTestCase):
    """The architecture argument: refused mail cannot reach the model at all."""

    def test_refused_mail_is_never_returned_as_evidence(self):
        ids = {t.message_id for t in self.threats()}
        index = retrieval.Index(self.box)
        for message in self.box.messages:
            found = retrieval.retrieve(self.box, message, index=index)
            self.assertFalse(
                ids & set(found.ids),
                f"retrieval offered refused mail as evidence for {message.id}: {ids & set(found.ids)}",
            )

    def test_refused_mail_is_still_indexed(self):
        """Indexed and retrievable are different permissions — it must stay findable."""
        ids = {t.message_id for t in self.threats()}
        index = retrieval.Index(self.box)
        rows = index.db.execute("select mid from fts").fetchall()
        indexed = {row[0] for row in rows}
        self.assertTrue(ids <= indexed, f"refused mail is missing from the index: {ids - indexed}")

    def test_the_model_is_never_asked_about_refused_mail(self):
        """A flagged verdict is `handled`, so the router never enters the model branch."""
        for threat in self.threats():
            verdict = rules.classify(self.box.by_id(threat.message_id))
            self.assertTrue(verdict.handled)
            self.assertEqual(verdict.allowed, ("flag",))


class TestNoAnswerUnlocksIt(HostileTestCase):
    """A human `yes` is permission, not authority."""

    def proposal(self, threat, action="send"):
        message = self.box.by_id(threat.message_id)
        return gate.Proposal(
            message_id=message.id,
            action=action,
            recipient=message.sender,
            subject=message.subject,
            body="Sure, here you go.",
            thread_id=message.thread_id,
        )

    def test_refused_mail_cannot_be_answered_even_on_a_yes(self):
        folders = actions.Mailbox()
        for threat in self.threats():
            verdict = gate.run(
                [self.proposal(threat)],
                self.box,
                mode="approval",
                asker=lambda q, d: gate.SAID_YES,
                folders=folders,
            )[-1][1].verdicts[0]
            self.assertTrue(verdict.blocked, f"{threat.message_id} was sendable")
            self.assertFalse(gate.outbox_path(threat.message_id).exists())

    def test_refused_mail_cannot_be_deleted_even_on_a_yes(self):
        """Requirement 4: flag it and leave it in place."""
        folders = actions.Mailbox()
        for threat in self.threats():
            verdict = gate.run(
                [self.proposal(threat, action="delete")],
                self.box,
                mode="approval",
                asker=lambda q, d: gate.SAID_YES,
                folders=folders,
            )[-1][1].verdicts[0]
            self.assertTrue(verdict.blocked, f"{threat.message_id} was deletable")
            self.assertEqual(folders.status(threat.message_id), actions.INBOX)

    def test_no_address_a_refused_message_names_is_reachable(self):
        """Even a proposal built by mistake could not reach an exfiltration target."""
        known = {a.lower() for a in self.box.known_addresses()}
        named = {a for t in self.threats() for a in t.names_addresses}
        self.assertTrue(named, "no refused message names an external address")
        self.assertFalse(named & known, f"the gate would accept {named & known}")

    def test_the_one_asking_to_be_saved_as_a_preference_cannot_be(self):
        """It arrives from the owner's address and uses the right vocabulary."""
        asking = [
            t
            for t in self.threats()
            if "standing preference" in self.box.by_id(t.message_id).text().lower()
        ]
        self.assertTrue(asking, "this inbox's preference-shaped attack was not refused")
        for threat in asking:
            with self.assertRaises(prefs.Refused):
                prefs.check(
                    {"kind": "meeting_floor", "value": "11:00am"},
                    self.box.by_id(threat.message_id),
                    self.box,
                )


class TestTheAudit(HostileTestCase):
    """The audit has to fail when what it audits is broken."""

    def test_a_clean_run_passes_every_check(self):
        rows, folders = self.rule_handled_run()
        _, checks = hostile.audit(self.box, rows, folders)
        failed = [c.name for c in checks if not c.passed]
        self.assertEqual(failed, [], f"a clean run failed: {failed}")

    def test_a_missing_refusal_event_fails_the_check(self):
        rows, folders = self.rule_handled_run()
        check = hostile.check_refusal_logged(self.threats(), events=[])
        self.assertFalse(check.passed)
        self.assertIn("predates this capability", check.detail)

    def test_a_decisions_row_without_the_attempt_fails_the_check(self):
        rows, _ = self.rule_handled_run()
        stripped = [{k: v for k, v in row.items() if k != "attempted"} for row in rows]
        check = hostile.check_reported(self.threats(), stripped)
        self.assertFalse(check.passed)

    def test_an_outbox_file_for_a_refused_message_fails_the_check(self):
        rows, folders = self.rule_handled_run()
        threat = self.threats()[0]
        path = gate.outbox_path(threat.message_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("this should never exist", encoding="utf-8")
        check = hostile.check_not_complied(self.threats(), self.box, folders)
        self.assertFalse(check.passed)

    def test_a_binned_message_fails_the_check(self):
        rows, folders = self.rule_handled_run()
        threat = self.threats()[0]
        folders.folders[threat.message_id] = {"status": actions.STATUS_FOR["delete"], "detail": {}}
        check = hostile.check_left_in_place(self.threats(), self.box, folders)
        self.assertFalse(check.passed)

    def test_refused_mail_appearing_as_evidence_fails_the_check(self):
        threat = self.threats()[0]
        rows = [{"message_id": "m001", "evidence": [threat.message_id], "cites": []}]
        check = hostile.check_never_quoted(self.threats(), rows)
        self.assertFalse(check.passed)
        self.assertIn("m001", check.detail)

    def test_the_threat_list_is_derived_not_hardcoded(self):
        """No id drives behaviour; the list comes from the rule tier every time.

        Comments may name a message to explain why code works — that is a reason,
        not a branch. What must not exist is an id in a statement, which would
        make the capability a claim about messages known to behave well.
        """
        import re

        source = (Path(__file__).resolve().parents[1] / "hostile.py").read_text(encoding="utf-8")
        code = "\n".join(
            line.partition("#")[0] for line in source.splitlines() if not line.lstrip().startswith("#")
        )
        named = re.findall(r"\bm\d{3}\b", code)
        self.assertEqual(named, [], f"hostile.py branches on {named}")


class TestTheRunSummaryTells(HostileTestCase):
    """Requirement 3. Silently handling an attack is the failure."""

    def test_the_decision_carries_what_was_attempted(self):
        rows, _ = self.rule_handled_run()
        flagged = [row for row in rows if row["disposition"] == "flag"]
        self.assertTrue(flagged)
        for row in flagged:
            self.assertTrue(row["attempted"].strip(), f"{row['message_id']} reached the summary with no attempt")

    def test_a_refusal_event_names_the_id_and_the_attempt(self):
        self.rule_handled_run()
        events = trace.read(kind="refusal")
        self.assertEqual(len(events), len(self.threats()))
        for event in events:
            self.assertTrue(event["msg_id"])
            self.assertTrue(str(event["attempted"]).strip())
            self.assertFalse(event["complied"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
