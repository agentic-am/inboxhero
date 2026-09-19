"""Part 3: the retrieval tier, and the boundary it has to hold.

Two kinds of test live here. The first kind checks grounding quality against
messages whose correct evidence is known by reading the inbox: m019 has to
reach m026 or m036 across threads, and m012 has to reach nothing at all. The
second kind checks the rules that make retrieval safe rather than merely
useful -- no message may be grounded in one that had not arrived yet, hostile
mail is never quoted back into a prompt, and a model may not cite a message it
was never shown.

    python -m unittest tests.test_part3 -v
"""

import sys
import unittest
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agents  # noqa: E402
import mailstore  # noqa: E402
import retrieval  # noqa: E402
import rules  # noqa: E402


class RetrievalTestCase(unittest.TestCase):
    """One inbox and one index for the whole file: both are read-only here."""

    @classmethod
    def setUpClass(cls):
        cls.box = mailstore.load()
        cls.index = retrieval.Index(cls.box)

    def find(self, message_id, **kwargs):
        message = self.box.by_id(message_id)
        self.assertIsNotNone(message, f"{message_id} is not in the inbox")
        return retrieval.retrieve(self.box, message, index=self.index, **kwargs)


class TestQuerySanitising(RetrievalTestCase):
    """FTS5's query language is a language, and message text is untrusted input."""

    def test_colon_term_does_not_crash(self):
        # A bare 1:1 reads as a column filter and raises "no such column: 1".
        self.assertEqual(retrieval.fts_query(["1:1"]), '"1:1"')
        rows = self.index.db.execute(
            "select count(*) from fts where fts match ?", (retrieval.fts_query(["1:1"]),)
        ).fetchone()
        self.assertGreaterEqual(rows[0], 1)

    def test_every_real_message_is_queryable(self):
        # m013, m016 and m119 all contain "1:1"; none of them may break the run.
        for message in self.box.messages:
            found = retrieval.retrieve(self.box, message, index=self.index)
            self.assertIsInstance(found, retrieval.Retrieved)

    def test_fts_operators_in_a_body_stay_literal(self):
        hostile = 'OR * NEAR("x") AND'
        self.assertEqual(retrieval.fts_query([hostile]), '"OR * NEAR(""x"") AND"')

    def test_empty_terms_produce_an_empty_query(self):
        self.assertEqual(retrieval.fts_query([]), "")
        self.assertEqual(retrieval.fts_query(["", "   "]), "")


class TestScope(RetrievalTestCase):
    """Tier 0. What may be considered at all."""

    def test_no_message_is_grounded_in_the_future(self):
        # Without this a replay of the run sees different evidence than the run
        # did, and the trace stops being an audit record.
        for message in self.box.messages:
            found = retrieval.retrieve(self.box, message, index=self.index)
            for item in found.evidence:
                cited = self.box.by_id(item.message_id)
                self.assertLess(
                    cited.sent_at,
                    message.sent_at,
                    f"{message.id} was grounded in {item.message_id}, which arrived later",
                )

    def test_a_message_never_grounds_itself(self):
        for message in self.box.messages:
            found = retrieval.retrieve(self.box, message, index=self.index)
            self.assertNotIn(message.id, found.ids)

    def test_hostile_messages_are_indexed_but_never_quoted(self):
        # Indexed and retrievable are two different permissions: m021 has to be
        # findable, but quoting it into a prompt would hand its payload to the
        # model inside somebody else's decision.
        self.assertTrue(self.index.hostile_ids, "the rule tier found no hostile mail")
        for message in self.box.messages:
            found = retrieval.retrieve(self.box, message, index=self.index)
            leaked = set(found.ids) & self.index.hostile_ids
            self.assertFalse(leaked, f"{message.id} was grounded in hostile mail {sorted(leaked)}")

    def test_hostile_mail_is_still_in_the_index(self):
        rows = self.index.db.execute("select count(*) from fts where mid = 'm021'").fetchone()
        self.assertEqual(rows[0], 1)


class TestWindow(RetrievalTestCase):
    """The window applies to the keyword tier only, and never loses recall silently."""

    def test_window_is_off_by_default(self):
        self.assertIsNone(self.find("m019").window_days)

    def test_a_narrow_window_widens_rather_than_returning_nothing(self):
        # m040 needs m038, which arrived just over a day earlier. A one-day
        # window cannot see it, so retrieval widens once and records that it did.
        found = self.find("m040", window_days=1)
        self.assertTrue(found.widened, "a window that finds nothing must widen, not give up")
        self.assertIn("m038", found.ids)

    def test_a_window_can_still_degrade_recall_without_widening(self):
        # Worth knowing rather than discovering later: widening triggers on an
        # empty result, not a worse one. A one-day window still finds *something*
        # for m019 (m046), so it never widens -- and quietly loses m036, which is
        # the message that actually holds the date. This is the whole reason
        # RETRIEVAL_WINDOW_DAYS is unset by default on an eight-day inbox.
        narrow = self.find("m019", window_days=1)
        self.assertFalse(narrow.widened)
        self.assertNotIn("m036", narrow.ids)
        self.assertIn("m036", self.find("m019").ids)

    def test_the_thread_tier_ignores_the_window(self):
        # m008's thread starts four days earlier. A window must not cut a thread.
        found = self.find("m008", window_days=1)
        self.assertIn("m003", found.ids)


class TestThreadTier(RetrievalTestCase):
    """Tier 1."""

    def test_thread_walk_grounds_the_staging_thread(self):
        found = self.find("m008")
        self.assertIn("m003", found.ids)
        source = {item.message_id: item.source for item in found.evidence}
        self.assertEqual(source["m003"], "thread")

    def test_a_long_thread_keeps_its_opener(self):
        # m026 sets "target is the 20th"; every later message leans on it, so
        # budgeting a long thread down must not drop the message that set it up.
        launch = sorted(self.box.thread("t-launch"), key=lambda m: m.sent_at)
        self.assertGreater(len(launch), 5, "t-launch should be the long thread")
        kept = retrieval.tier_thread(self.box, launch[-1], limit=3)
        self.assertEqual(kept[0].id, launch[0].id)
        self.assertEqual(len(kept), 3)


class TestKeywordTier(RetrievalTestCase):
    """Tier 2, including the term selection that decides what gets asked."""

    def test_selective_terms_beat_merely_rare_ones(self):
        # m046's rarest word is "coverage", which appears nowhere else and can
        # ground nothing. The word that grounds it is "launch".
        message = self.box.by_id("m046")
        terms = self.index.selective(retrieval.extract_terms(message))
        self.assertIn("launch", terms)
        self.assertNotIn("coverage", terms)

    def test_terms_unique_to_this_message_are_dropped(self):
        message = self.box.by_id("m046")
        weights = retrieval.extract_terms(message)
        for term in self.index.selective(weights):
            self.assertGreaterEqual(self.index.document_frequency(term), retrieval.DF_MIN)

    def test_one_weak_term_is_not_enough_to_search_on(self):
        self.assertFalse(retrieval.worth_querying(["call"], {"call": retrieval.WEIGHT_VOCABULARY}))
        self.assertTrue(retrieval.worth_querying(["board"], {"board": 6}))
        self.assertTrue(
            retrieval.worth_querying(["launch", "date"], {"launch": 3, "date": 3})
        )

    def test_cross_thread_grounding(self):
        # Thread walk structurally cannot reach these: different threads.
        self.assertTrue(set(self.find("m019").ids) & {"m026", "m036"})
        self.assertTrue(set(self.find("m046").ids) & {"m026", "m036"})
        self.assertIn("m038", self.find("m040").ids)

    def test_evidence_records_why_it_was_retrieved(self):
        found = self.find("m040")
        item = next(e for e in found.evidence if e.message_id == "m038")
        self.assertEqual(item.source, "keyword")
        self.assertIn("board", item.terms)
        self.assertIn("matched board", item.line())


class TestUngrounded(RetrievalTestCase):
    """The Part 3.4 case: ask, do not guess."""

    def test_a_message_with_nothing_to_match_retrieves_nothing(self):
        found = self.find("m012")
        self.assertEqual(found.ids, [])
        self.assertEqual(found.terms, ())
        self.assertFalse(found)

    def test_the_prompt_says_so_when_nothing_was_found(self):
        message = self.box.by_id("m012")
        prompt = agents.build_prompt(message, rules.classify(message), self.box, ())
        self.assertIn("Nothing in the inbox was found to ground this message", prompt)
        self.assertNotIn('"cites"', prompt)


class TestGoldGroundings(RetrievalTestCase):
    """The recall probe that chose this design, kept as a regression test.

    These nine were read out of the inbox by hand. They are here because the
    choice of a keyword tier over an embedding tier was made on measured recall
    over exactly this set, and a change that quietly loses one should fail.
    """

    GOLD = {
        "m008": {"m003"},  # thread: the rotated credential
        "m005": {"m001", "m003"},  # thread
        "m030": {"m026"},  # thread: target is the 20th
        "m019": {"m026", "m036"},  # cross-thread: the date locked in
        "m046": {"m026", "m036"},  # cross-thread: is the launch date public
        "m040": {"m038"},  # cross-thread: two days before the board review
        "m043": {"m010"},  # thread: the earlier intro call
        "m016": {"m010"},  # cross-thread: thanks for the intro
        "m012": set(),  # nothing to ground: ask, do not guess
    }

    def test_every_gold_grounding_is_found(self):
        for message_id, gold in self.GOLD.items():
            with self.subTest(message=message_id):
                ids = set(self.find(message_id).ids)
                if gold:
                    self.assertTrue(gold & ids, f"{message_id}: wanted one of {sorted(gold)}, got {sorted(ids)}")
                else:
                    self.assertFalse(ids, f"{message_id} should ground in nothing, got {sorted(ids)}")


class TestEvidenceInThePrompt(RetrievalTestCase):
    """Retrieved mail is quoted, and quoted as untrusted."""

    def setUp(self):
        self.message = self.box.by_id("m019")
        self.found = self.find("m019")
        self.prompt = agents.build_prompt(
            self.message, rules.classify(self.message), self.box, self.found.evidence
        )

    def test_evidence_is_marked_untrusted(self):
        for item in self.found.evidence:
            self.assertIn(f"<<<UNTRUSTED EVIDENCE id={item.message_id}", self.prompt)
            self.assertIn(f"<<<END UNTRUSTED EVIDENCE id={item.message_id}", self.prompt)

    def test_the_message_itself_is_still_marked(self):
        self.assertIn("<<<UNTRUSTED MESSAGE id=m019", self.prompt)

    def test_the_prompt_names_the_citable_ids(self):
        self.assertIn('"cites"', self.prompt)
        for item in self.found.evidence:
            self.assertIn(item.message_id, self.prompt)


class TestCitationChecking(RetrievalTestCase):
    """A citation is checked in Python, like the allowed list, after the fact."""

    def setUp(self):
        self.verdict = rules.classify(self.box.by_id("m019"))
        self.evidence_ids = ("m026", "m036")

    def parse(self, raw):
        return agents.parse_proposal(raw, self.verdict, self.evidence_ids)

    def test_a_citation_from_the_evidence_is_kept(self):
        result = self.parse('{"disposition": "reply", "reason": "the date is set", "cites": ["m036"]}')
        self.assertEqual(result["cites"], ["m036"])

    def test_an_uncited_proposal_keeps_its_part_2_shape(self):
        # Part 3 must not change what a decision looks like when there was
        # nothing to cite; the key appears only when it carries something.
        result = self.parse('{"disposition": "reply", "reason": "no grounding needed"}')
        self.assertEqual(set(result), {"disposition", "reason"})

    def test_citing_the_message_being_triaged_is_dropped_not_rejected(self):
        result = self.parse('{"disposition": "reply", "reason": "x", "cites": ["m019", "m036"]}')
        self.assertEqual(result["cites"], ["m036"])

    def test_a_citation_that_was_never_shown_is_rejected(self):
        # An invented citation is worse than none, because it reads as evidence.
        with self.assertRaises(agents.Rejected) as caught:
            self.parse('{"disposition": "reply", "reason": "see below", "cites": ["m999"]}')
        self.assertIn("m999", str(caught.exception))

    def test_a_real_message_that_was_not_retrieved_is_still_rejected(self):
        with self.assertRaises(agents.Rejected):
            self.parse('{"disposition": "reply", "reason": "see below", "cites": ["m001"]}')

    def test_citing_nothing_is_allowed(self):
        result = self.parse('{"disposition": "reply", "reason": "no grounding needed"}')
        self.assertEqual(result.get("cites", []), [])

    def test_a_citation_of_the_wrong_shape_is_rejected(self):
        with self.assertRaises(agents.Rejected):
            self.parse('{"disposition": "reply", "reason": "x", "cites": {"id": "m036"}}')

    def test_citations_are_rejected_when_no_evidence_was_retrieved(self):
        with self.assertRaises(agents.Rejected):
            agents.parse_proposal(
                '{"disposition": "reply", "reason": "x", "cites": ["m036"]}', self.verdict, ()
            )


class TestIndexSeam(RetrievalTestCase):
    """The index is the only thing that has to change when the inbox stops fitting in RAM."""

    def test_the_index_covers_every_valid_message(self):
        rows = self.index.db.execute("select count(*) from fts").fetchone()
        self.assertEqual(rows[0], len(self.box.messages))

    def test_document_frequency_is_cached_not_recomputed(self):
        self.index.document_frequency("launch")
        self.assertIn("launch", self.index._df_cache)

    def test_entities_are_indexed_exactly(self):
        # Domains and references are the terms BM25 is worst at ranking.
        self.assertTrue(self.index.entities, "no entities were extracted")
        for value, ids in self.index.entities.items():
            self.assertTrue(all(self.box.by_id(mid) for mid in ids), f"{value} points at a missing message")

    def test_scope_earliest_respects_the_window(self):
        message = self.box.by_id("m019")
        scope = retrieval.Scope(before=message.sent_at, window_days=2)
        self.assertEqual(scope.earliest(), message.sent_at - timedelta(days=2))
        self.assertIsNone(retrieval.Scope(before=message.sent_at).earliest())


if __name__ == "__main__":
    unittest.main(verbosity=2)
