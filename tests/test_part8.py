"""Part 8: the four capabilities of our own.

    X1  who owes the next move        deterministic
    X2  the open question in a thread  model, checked
    X3  tone per correspondent         measured, stored, enforced
    X4  why did it do that             deterministic, from the trace

The tests worth writing here are the ones that would catch a capability quietly
becoming a demo: a threshold that only works on this inbox, a register learned
from a phishing message, a citation that points outside the thread it claims to
summarise, and an explanation invented when the trace holds nothing.

    python -m unittest tests.test_part8 -v
"""

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import drafting  # noqa: E402
import explain  # noqa: E402
import mailstore  # noqa: E402
import rules  # noqa: E402
import threads  # noqa: E402
import tone  # noqa: E402
import trace  # noqa: E402
import waiting  # noqa: E402


class CapabilityTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.box = mailstore.load()

    def setUp(self):
        self._saved = (config.STATE_PATH, config.TRACE_PATH)
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        config.STATE_PATH = root / "state"
        config.TRACE_PATH = root / "trace.jsonl"
        config.STATE_PATH.mkdir(parents=True)

    def tearDown(self):
        config.STATE_PATH, config.TRACE_PATH = self._saved
        self._tmp.cleanup()

    def recorded_rows(self):
        path = Path(__file__).resolve().parents[1] / "state" / "decisions.json"
        rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        return {r.get("message_id"): r for r in rows}


# --- X1 ---------------------------------------------------------------------


class TestWhoOwesTheNextMove(CapabilityTestCase):
    def survey(self):
        return waiting.survey(self.box, self.recorded_rows())

    def test_every_thread_lands_in_exactly_one_state(self):
        found = self.survey()
        self.assertEqual(len(found), len({m.thread_id for m in self.box.messages}))
        for thread in found:
            self.assertIn(thread.state, ("you", "them", "closed"))

    def test_mail_the_owner_sent_and_nobody_answered_is_found(self):
        """The case no mail client raises, because it lives in `sent`."""
        theirs = waiting.grouped(self.survey())["them"]
        self.assertTrue(theirs, "nothing is waiting on anybody else, which this inbox should not produce")
        for thread in theirs:
            self.assertEqual(thread.messages[-1].sender.lower(), config.OWNER)

    def test_delegating_does_not_close_a_thread(self):
        """Handing something over moves whose move it is; it does not finish it."""
        self.assertNotIn("delegate", waiting.CLOSED_BY)
        delegated = [t for t in self.survey() if t.disposition == "delegate"]
        for thread in delegated:
            self.assertNotEqual(thread.state, "closed")

    def test_a_note_to_yourself_waits_on_nobody(self):
        """A standing instruction the owner mailed themselves is not a chase item."""
        selves = [
            t
            for t in self.survey()
            if t.messages[-1].sender.lower() == config.OWNER and t.messages[-1].to.lower() == config.OWNER
        ]
        self.assertTrue(selves, "this inbox has notes the owner wrote to themselves")
        for thread in selves:
            self.assertEqual(thread.state, "closed")

    def test_refused_mail_is_closed_whatever_its_disposition(self):
        for thread in self.survey():
            if rules.classify(thread.messages[-1]).hostile:
                self.assertEqual(thread.state, "closed")

    def test_ages_are_measured_from_the_inbox_not_the_wall_clock(self):
        """The corpus is fixed; measuring against today would call everything stale."""
        newest = max(m.sent_at for m in self.box.messages)
        for thread in self.survey():
            self.assertLessEqual(thread.waiting_days, (newest - min(m.sent_at for m in self.box.messages)).days + 1)

    def test_the_chase_list_respects_its_threshold(self):
        found = self.survey()
        self.assertEqual(waiting.oldest_unanswered(found, days=999), [])
        for thread in waiting.oldest_unanswered(found, days=0):
            self.assertEqual(thread.state, "them")


# --- X2 ---------------------------------------------------------------------


class TestTheOpenQuestion(CapabilityTestCase):
    def a_long_thread(self):
        found = threads.worth_it(self.box)
        self.assertTrue(found, "no thread is long enough to summarise")
        return found[0]

    def test_only_threads_worth_reading_are_offered(self):
        for _, messages in threads.worth_it(self.box):
            self.assertGreaterEqual(len(messages), threads.WORTH_SUMMARISING)

    def test_refused_mail_is_not_part_of_a_thread_summary(self):
        for _, messages in threads.worth_it(self.box):
            for message in messages:
                self.assertFalse(rules.classify(message).hostile)

    def test_a_citation_outside_the_thread_is_rejected(self):
        """Checked against the thread, not the mail store.

        A real id from another conversation is still a citation that does not
        support what it is attached to, and the mail store would accept it.
        """
        thread_id, messages = self.a_long_thread()
        outsider = next(m for m in self.box.messages if m.thread_id != thread_id)
        raw = json.dumps({"open_question": "something", "cites": [outsider.id]})
        with self.assertRaises(threads.NotUsable) as caught:
            threads.parse(raw, messages)
        self.assertIn(outsider.id, str(caught.exception))

    def test_an_open_question_must_cite_something(self):
        _, messages = self.a_long_thread()
        raw = json.dumps({"open_question": "something is open", "cites": []})
        with self.assertRaises(threads.NotUsable):
            threads.parse(raw, messages)

    def test_nothing_open_is_a_valid_answer_and_needs_no_citation(self):
        _, messages = self.a_long_thread()
        raw = json.dumps({"open_question": None, "cites": [], "reason": "all answered"})
        answer = threads.parse(raw, messages)
        self.assertFalse(answer["open"])

    def test_an_internal_id_in_the_question_is_rejected(self):
        """A reader has no m030; the id belongs in `cites`."""
        thread_id, messages = self.a_long_thread()
        raw = json.dumps({"open_question": f"see {messages[0].id} for the ask", "cites": [messages[0].id]})
        with self.assertRaises(threads.NotUsable):
            threads.parse(raw, messages)

    def test_malformed_answers_are_rejected_not_guessed_at(self):
        _, messages = self.a_long_thread()
        for raw in ("not json at all", "{broken", ""):
            with self.assertRaises(threads.NotUsable):
                threads.parse(raw, messages)

    def test_the_prompt_quotes_the_thread_as_untrusted(self):
        thread_id, messages = self.a_long_thread()
        prompt = threads.build_prompt(thread_id, messages)
        self.assertIn("UNTRUSTED", prompt.upper())
        for message in messages:
            self.assertIn(message.id, prompt)


# --- X3 ---------------------------------------------------------------------


class TestTone(CapabilityTestCase):
    def profiles(self):
        return tone.learn(self.box)

    def test_a_correspondent_who_signs_off_with_a_firm_is_formal(self):
        found = self.profiles()
        formal = [p for p in found.values() if p.register == tone.FORMAL]
        self.assertTrue(formal, "no correspondent in this inbox reads as formal")
        for profile in formal:
            self.assertTrue(profile.why)

    def test_someone_writing_in_lowercase_with_exclamations_is_casual(self):
        casual = [p for p in self.profiles().values() if p.register == tone.CASUAL]
        self.assertTrue(casual, "no correspondent in this inbox reads as casual")

    def test_refused_mail_never_sets_a_register(self):
        """An attacker must not get to choose the voice the system answers in."""
        refused = {m.sender.lower() for m in self.box.messages if rules.classify(m).hostile}
        learned = set(self.profiles())
        # A refused sender may still appear if they also sent legitimate mail; what
        # must not happen is a profile built only from refused messages.
        for address in refused & learned:
            sources = [
                m
                for m in self.box.messages
                if m.sender.lower() == address and not rules.classify(m).hostile
            ]
            self.assertTrue(sources, f"{address} was profiled entirely from refused mail")

    def test_automated_senders_are_not_profiled(self):
        for address in self.profiles():
            sample = next(m for m in self.box.messages if m.sender.lower() == address)
            self.assertFalse(rules.is_automated(sample), f"{address} is a machine and was given a register")

    def test_measuring_is_repeatable(self):
        """Countable signals, so the same mail scores the same every time."""
        first = {a: p.register for a, p in self.profiles().items()}
        second = {a: p.register for a, p in self.profiles().items()}
        self.assertEqual(first, second)

    def test_an_unknown_address_is_neutral(self):
        self.assertEqual(tone.register_of("nobody@nowhere.example", {}), tone.NEUTRAL)

    def test_the_check_only_fires_on_formal(self):
        """Stiffer than the correspondent costs nothing; too familiar costs a lot."""
        casual_draft = "Thanks! I'll take a look."
        self.assertTrue(tone.too_familiar(casual_draft, tone.FORMAL))
        self.assertEqual(tone.too_familiar(casual_draft, tone.NEUTRAL), ())
        self.assertEqual(tone.too_familiar(casual_draft, tone.CASUAL), ())

    def test_a_formal_draft_passes_the_formal_check(self):
        self.assertEqual(
            tone.too_familiar("Thank you. I will review the minutes and revert by Monday.", tone.FORMAL), ()
        )

    def test_the_guidance_gives_rules_not_examples(self):
        """An example gets copied back; this project has learned that twice."""
        profiles = {a: p.row() for a, p in self.profiles().items()}
        for address in profiles:
            line = tone.for_prompt(address, profiles)
            for message in self.box.messages:
                if message.sender.lower() == address:
                    for sentence in message.body.split("."):
                        if len(sentence.split()) >= 5:
                            self.assertNotIn(sentence.strip(), line)

    def test_the_drafting_check_is_wired_in(self):
        """The register has to reach `parse_draft`, not just exist."""
        tone.save(self.profiles())
        formal = next(m for m in self.box.messages if tone.register_of(m.sender) == tone.FORMAL)
        raw = json.dumps({"draft": "Thanks! I'll get to it.", "cites": []})
        with self.assertRaises(drafting.DraftRejected) as caught:
            drafting.parse_draft(raw, formal, (), self.box)
        self.assertIn("familiar", str(caught.exception))


# --- X4 ---------------------------------------------------------------------


class TestExplain(CapabilityTestCase):
    def events(self):
        return [
            {"ts": "2026-09-09T10:00:00.000+00:00", "cap": "R1", "event": "read", "msg_id": "m001", "subject": "s"},
            {"ts": "2026-09-09T10:00:01.000+00:00", "cap": "R1", "event": "rule", "msg_id": "m001",
             "rule": "abstain", "disposition": None, "allowed": ["reply"], "flags": []},
            {"ts": "2026-09-09T10:00:02.000+00:00", "cap": "R1", "event": "moya.step.started", "msg_id": "m001"},
            {"ts": "2026-09-09T10:00:03.000+00:00", "cap": "R1", "event": "decision", "msg_id": "m001",
             "disposition": "reply", "path": "model", "reason": "because"},
            {"ts": "2026-09-09T10:00:04.000+00:00", "cap": "R1", "event": "read", "msg_id": "m002", "subject": "t"},
        ]

    def test_only_that_message_is_reported(self):
        steps = explain.story("m001", self.events())
        self.assertTrue(steps)
        self.assertTrue(all(s.raw["msg_id"] == "m001" for s in steps))

    def test_framework_machinery_is_left_out(self):
        kinds = {s.kind for s in explain.story("m001", self.events())}
        self.assertNotIn("moya.step.started", kinds)

    def test_nothing_recorded_means_nothing_is_invented(self):
        """The one failure that would make an audit trail worthless."""
        answer = explain.explain("m999", self.box, self.events())
        self.assertEqual(answer["steps"], [])
        rendered = explain.render(answer)
        self.assertIn("Nothing about m999", rendered)

    def test_the_final_decision_is_picked_out(self):
        answer = explain.explain("m001", self.box, self.events())
        self.assertIsNotNone(answer["decision"])
        self.assertIn("reply", answer["decision"].said)

    def test_the_story_names_which_capabilities_it_came_from(self):
        answer = explain.explain("m001", self.box, self.events())
        self.assertEqual(answer["caps"], ["R1"])

    def test_a_step_from_an_unknown_event_kind_still_renders(self):
        events = self.events() + [
            {"ts": "2026-09-09T10:00:05.000+00:00", "cap": "X9", "event": "something_new", "msg_id": "m001", "detail": 1}
        ]
        steps = explain.story("m001", events)
        self.assertTrue(steps[-1].said)

    def test_the_timeline_carries_a_date(self):
        """A message's story spans runs days apart; the clock alone misleads."""
        step = explain.story("m001", self.events())[0]
        self.assertIn("09-09", step.line())


if __name__ == "__main__":
    unittest.main(verbosity=2)
