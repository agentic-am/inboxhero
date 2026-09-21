"""Part 7: the dashboard, and the calendar that carries its marks.

Three panes, and the third is graded on evidence rather than presentation: every
commitment cites its sources, at least one is resolvable only by combining two
messages, and two things at the same time are called out. Those are the claims
tested here, along with the ones that are easy to get wrong in the other
direction -- inventing a date that no message gave, putting an attacker's deadline
on the owner's calendar, or asserting a clash between two dates that might be a
week apart.

    python -m unittest tests.test_part7 -v
"""

import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import actions  # noqa: E402
import commitments  # noqa: E402
import config  # noqa: E402
import dashboard  # noqa: E402
import mailstore  # noqa: E402
import rules  # noqa: E402


class DashboardTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.box = mailstore.load()
        cls.found = commitments.extract(cls.box)

    def setUp(self):
        self._saved = (config.STATE_PATH, config.OUTBOX_PATH, config.TRACE_PATH, config.DASHBOARD_PATH)
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        config.STATE_PATH = root / "state"
        config.OUTBOX_PATH = root / "outbox"
        config.TRACE_PATH = root / "trace.jsonl"
        config.DASHBOARD_PATH = root / "dashboard.html"
        config.STATE_PATH.mkdir(parents=True)

    def tearDown(self):
        config.STATE_PATH, config.OUTBOX_PATH, config.TRACE_PATH, config.DASHBOARD_PATH = self._saved
        self._tmp.cleanup()

    def recorded_rows(self):
        path = Path(__file__).resolve().parents[1] / "state" / "decisions.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


class TestReadingADate(DashboardTestCase):
    """The parsing, on its own, before anything is built out of it."""

    def test_an_ordinal_is_read_in_the_month_the_message_was_sent(self):
        anchor = datetime(2026, 9, 6, 12, 0)
        found = commitments.date_in("set for the 18th, 10:00am", anchor)
        self.assertEqual(found[0].date(), datetime(2026, 9, 18).date())

    def test_an_ordinal_that_has_passed_means_next_month(self):
        """Mail does not refer backwards to a deadline."""
        anchor = datetime(2026, 9, 20, 9, 0)
        found = commitments.date_in("due the 3rd", anchor)
        self.assertEqual(found[0].date(), datetime(2026, 10, 3).date())

    def test_a_named_month_and_day_is_taken_literally(self):
        found = commitments.date_in("on Sep 12, 1:00pm", datetime(2026, 9, 5, 9, 0))
        self.assertEqual(found[0].date(), datetime(2026, 9, 12).date())

    def test_times_convert_to_twenty_four_hour(self):
        self.assertEqual(commitments.clock_in("at 2:00pm"), (14, 0))
        self.assertEqual(commitments.clock_in("at 10:00am"), (10, 0))
        self.assertEqual(commitments.clock_in("at 3:00 PM"), (15, 0))
        self.assertIsNone(commitments.clock_in("no time here"))

    def test_the_weekday_taken_is_the_one_the_clock_belongs_to(self):
        """"from Thursday to Wednesday at 2:00pm" is a Wednesday commitment.

        The first weekday in that sentence is the day being vacated. Reading it
        put a meeting on the wrong day of the week until the clock was used to
        choose between them.
        """
        anchor = datetime(2026, 9, 8, 15, 26)  # a Tuesday
        found = commitments.weekday_in(
            "Can we move our weekly 1:1 from Thursday to Wednesday at 2:00pm this week?",
            anchor,
            same_week=True,
        )
        self.assertEqual(found[1], commitments.WEEKDAYS["wednesday"])


class TestWhatBecomesACommitment(DashboardTestCase):
    """What belongs on a calendar, and what emphatically does not."""

    def test_something_was_extracted(self):
        self.assertTrue(self.found)

    def test_no_refused_message_reaches_the_calendar(self):
        """An attacker's deadline must not sit beside the owner's real ones.

        One refused message presses for a payment "before end of day". A calendar
        listing it has done the attacker's work: it puts their urgency in front of
        the owner in the owner's own planner.
        """
        refused = {m.id for m in self.box.messages if rules.classify(m).hostile}
        self.assertTrue(refused)
        for commitment in self.found:
            self.assertFalse(
                refused & set(commitment.cites),
                f"{commitment.what!r} was built from refused mail {refused & set(commitment.cites)}",
            )

    def test_a_standing_instruction_is_not_a_commitment(self):
        """"No meetings before 11:00am" is a rule, not somewhere to be."""
        preferences = {
            m.id for m in self.box.messages if "preference_statement" in rules.classify(m).flags
        }
        self.assertTrue(preferences)
        for commitment in self.found:
            self.assertFalse(preferences & set(commitment.cites))

    def test_automated_mail_that_asks_nothing_is_not_a_commitment(self):
        """A payout arriving and a domain renewing are facts, not obligations."""
        for commitment in self.found:
            self.assertNotIn("renews", commitment.what.lower())
            self.assertNotIn("screen time", commitment.what.lower())

    def test_the_description_is_not_the_thread_subject(self):
        """Nine messages share a thread subject; a calendar of them says nothing.

        Four obligations once read "Launch week -- kickoff" because the subject
        was used. The sentence carrying the date says what the date is for.
        """
        descriptions = [c.what for c in self.found]
        self.assertEqual(len(descriptions), len(set(descriptions)), "two commitments describe themselves identically")


class TestCitations(DashboardTestCase):
    """Checked against the mail store, exactly as a draft's citations are."""

    def test_every_commitment_cites_at_least_one_message(self):
        for commitment in self.found:
            self.assertTrue(commitment.cites, f"{commitment.what!r} cites nothing")

    def test_every_cited_id_is_in_the_mail_store(self):
        self.assertEqual(commitments.check_citations(self.found, self.box), [])

    def test_a_citation_that_is_not_in_the_store_is_caught(self):
        bogus = commitments.Commitment(what="invented", cites=("m999",))
        problems = commitments.check_citations([bogus], self.box)
        self.assertTrue(problems)
        self.assertIn("m999", problems[0])


class TestMoreThanOneMessage(DashboardTestCase):
    """The requirement the assignment singles out."""

    def test_at_least_one_commitment_comes_from_more_than_one_message(self):
        multi = [c for c in self.found if c.multi_source]
        self.assertTrue(multi, "no commitment needed two messages, which the assignment requires")

    def test_one_is_a_date_in_one_message_and_its_subject_in_another(self):
        """The case the assignment describes, found rather than named.

        A message asking for something "two days before" an event names no date;
        the message that scheduled the event does. The entry has to cite both, and
        say how it got there.
        """
        resolved = [c for c in self.found if c.resolved_by]
        self.assertTrue(resolved, "nothing was resolved against another commitment")
        for commitment in resolved:
            self.assertGreater(len(commitment.cites), 1)
            self.assertIsNotNone(commitment.when)
            # the date came from the other message, not from this one
            source = self.box.by_id(commitment.source)
            self.assertNotIn(commitment.when.strftime("%-d") if sys.platform != "win32" else str(commitment.when.day),
                             source.body.split("two days")[0])

    def test_the_same_date_in_one_thread_is_one_entry(self):
        """Stated at kickoff and confirmed later is one commitment with two sources."""
        threaded = [c for c in self.found if c.multi_source and not c.resolved_by]
        self.assertTrue(threaded, "no repeated date was folded into a single entry")
        for commitment in threaded:
            threads = {self.box.by_id(i).thread_id for i in commitment.cites}
            self.assertEqual(len(threads), 1, "entries folded across threads, which would be a different claim")


class TestConflicts(DashboardTestCase):
    """Surfaced, not silently listed -- and not overstated either."""

    def clashes(self):
        return commitments.conflicts(self.found)

    def test_two_things_at_the_same_time_are_called_out(self):
        certain = [c for c in self.clashes() if c.certain]
        self.assertTrue(certain, "two commitments share a date and time and nothing was said")

    def test_a_certain_conflict_really_is_the_same_moment(self):
        for clash in self.clashes():
            if clash.certain:
                self.assertEqual(clash.first.when, clash.second.when)
                self.assertTrue(clash.first.has_time and clash.second.has_time)

    def test_an_unresolved_pair_is_reported_as_possible_not_certain(self):
        """Two Wednesdays that could be a week apart are not a clash.

        Asserting one would be inventing it; saying nothing would hide it. The
        pane distinguishes them.
        """
        for clash in self.clashes():
            if not clash.certain:
                self.assertTrue(clash.first.when is None or clash.second.when is None)

    def test_the_same_weekday_at_different_times_is_not_a_conflict(self):
        a = commitments.Commitment(what="a", cites=("m001",), weekday=2, at_time=(14, 0), has_time=True)
        b = commitments.Commitment(what="b", cites=("m002",), weekday=2, at_time=(16, 0), has_time=True)
        self.assertEqual(commitments.conflicts([a, b]), [])

    def test_two_dated_entries_a_week_apart_are_not_a_conflict(self):
        a = commitments.Commitment(
            what="a", cites=("m001",), when=datetime(2026, 9, 9, 14, 0), at_time=(14, 0), has_time=True
        )
        b = commitments.Commitment(
            what="b", cites=("m002",), when=datetime(2026, 9, 16, 14, 0), at_time=(14, 0), has_time=True
        )
        self.assertEqual(commitments.conflicts([a, b]), [])


class TestNoDateIsInvented(DashboardTestCase):
    """An entry reading "Wednesday" is honest; one reading a guessed date is not."""

    def test_a_weekday_with_no_week_stays_unresolved(self):
        undated = [c for c in self.found if c.when is None]
        self.assertTrue(undated, "every commitment got a date, which this inbox cannot support")
        for commitment in undated:
            self.assertTrue(commitment.when_text, "an undated entry says nothing about when")

    def test_this_week_pins_a_weekday_and_nothing_else_does(self):
        anchor = datetime(2026, 9, 8, 9, 0)  # Tuesday
        pinned = commitments.weekday_in("on Wednesday", anchor, same_week=True)
        self.assertEqual(pinned[0].date(), datetime(2026, 9, 9).date())


class TestThePage(DashboardTestCase):
    """Three panes, built from the record, written to a file."""

    def page(self):
        return dashboard.build(self.box, self.recorded_rows(), actions.Mailbox())

    def test_there_are_exactly_three_panes(self):
        self.assertEqual(list(self.page()["panes"]), ["pending", "flagged", "commitments"])

    def test_pending_rows_say_why_a_person_is_needed(self):
        for row in self.page()["panes"]["pending"]:
            self.assertTrue(row["needs_human"], f"{row['message_id']} is pending for no stated reason")
            self.assertTrue(row["action"])

    def test_nothing_refused_appears_as_pending(self):
        """A refused action is not waiting on anybody."""
        page = self.page()
        pending = {r["message_id"] for r in page["panes"]["pending"]}
        refused = {r["message_id"] for r in page["panes"]["flagged"] if r["kind"] == "hostile"}
        self.assertFalse(pending & refused)

    def test_flagged_rows_say_what_was_attempted_and_what_happened(self):
        for row in self.page()["panes"]["flagged"]:
            self.assertTrue(row["attempted"].strip())
            self.assertTrue(row["instead"].strip())

    def test_the_flagged_pane_holds_more_than_one_kind(self):
        """Refused mail and mail that could not be grounded share an outcome."""
        kinds = {r["kind"] for r in self.page()["panes"]["flagged"]}
        self.assertIn("hostile", kinds)
        self.assertGreater(len(kinds), 1)

    def test_a_send_the_gate_refuses_appears_in_the_flagged_pane(self):
        """The row type that was missing, and only showed up under a stored preference.

        A refused send is not pending -- nobody is being asked about it -- so
        without a row here it is the one outcome the page cannot show. Driven with
        a folder state that marks the message flagged, so the test does not depend
        on which preferences happen to be recorded.
        """
        rows = self.recorded_rows()
        drafted = next((r for r in rows if (r.get("draft") or "").strip()), None)
        self.assertIsNotNone(drafted, "the recorded run drafted nothing")
        folders = actions.Mailbox()
        folders.apply("flag", drafted["message_id"], reason="forced, for this test")
        page = dashboard.build(self.box, rows, folders)
        listed = {r["message_id"] for r in page["panes"]["flagged"]}
        self.assertIn(drafted["message_id"], listed)
        self.assertNotIn(drafted["message_id"], {r["message_id"] for r in page["panes"]["pending"]})

    def test_both_files_are_written_and_the_page_comes_from_the_json(self):
        page = self.page()
        data_path, html_path = dashboard.write(page)
        self.assertTrue(data_path.exists() and html_path.exists())
        stored = json.loads(data_path.read_text(encoding="utf-8"))
        self.assertEqual(stored["panes"].keys(), page["panes"].keys())
        markup = html_path.read_text(encoding="utf-8")
        for row in page["panes"]["commitments"]["dated"]:
            for cited in row["cites"]:
                self.assertIn(cited, markup)

    def test_the_page_is_the_same_twice(self):
        """Reproducible from a run, which is what the assignment asks for."""
        first = self.page()
        second = self.page()
        first.pop("generated"), second.pop("generated")
        self.assertEqual(first, second)

    def test_the_html_escapes_message_content(self):
        """Mail is untrusted here too; a body is not markup."""
        page = self.page()
        page["panes"]["flagged"].append(
            {
                "message_id": "m000",
                "from": "x@y.z",
                "subject": "s",
                "attempted": "<script>alert(1)</script>",
                "instead": "nothing",
                "kind": "hostile",
            }
        )
        markup = dashboard.as_html(page)
        self.assertNotIn("<script>alert(1)</script>", markup)
        self.assertIn("&lt;script&gt;", markup)


if __name__ == "__main__":
    unittest.main(verbosity=2)
