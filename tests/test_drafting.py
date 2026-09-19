"""Part 3, second half: the draft, and the four ways it can be wrong.

`test_part3.py` covers finding the evidence. This file covers what is done with
it. Every check here runs without a model, because every one of them is a rule
applied *after* the model has answered -- which is the point: a drafted reply is
only as trustworthy as the checks that survive a bad answer.

The four failures each have a name and a real example behind them:

    invented    a date or URL that is in no cited message
    miscited    an id the model was never shown, or that is not in the inbox
    leaked      a credential copied out of a message it cited correctly
    echoed      the cited message played back as if it were the reply

    python -m unittest tests.test_drafting -v
"""

import json
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agents  # noqa: E402
import drafting  # noqa: E402
import mailstore  # noqa: E402
import retrieval  # noqa: E402


class StubAgent:
    """Stands in for the drafter: records that it was asked, returns a fixed answer."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def handle_message(self, prompt, **kwargs):
        self.calls += 1
        self.prompt = prompt
        return self.reply


def evidence_for(box, *message_ids):
    """Evidence objects for named messages, as retrieval would have produced them."""
    return tuple(
        retrieval.Evidence(
            message_id=mid,
            source="keyword",
            terms=("test",),
            snippet=retrieval.snippet_of(box.by_id(mid)),
            score=0.5,
            thread_id=box.by_id(mid).thread_id,
            sender=box.by_id(mid).sender,
            timestamp=box.by_id(mid).timestamp,
            subject=box.by_id(mid).subject,
        )
        for mid in message_ids
    )


class DraftingTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.box = mailstore.load()

    def parse(self, payload, message_id="m019", evidence_ids=("m036",)):
        return drafting.parse_draft(
            payload,
            self.box.by_id(message_id),
            evidence_for(self.box, *evidence_ids),
            self.box,
        )


class TestSecretDetection(DraftingTestCase):
    """m003 carries a live credential inside something shaped like a URL."""

    def test_a_userinfo_url_is_a_secret(self):
        self.assertTrue(drafting.contains_secret(self.box.by_id("m003").text()))

    def test_ordinary_mail_is_not(self):
        for message_id in ("m026", "m036", "m038", "m019"):
            self.assertFalse(drafting.contains_secret(self.box.by_id(message_id).text()), message_id)

    def test_a_stated_password_is_a_secret(self):
        self.assertTrue(drafting.contains_secret("the password is hunter2xyz"))
        self.assertTrue(drafting.contains_secret("API_KEY = sk-abc123def456"))

    def test_redaction_removes_the_secret_and_says_where_it_lives(self):
        redacted = drafting.redact_secrets(self.box.by_id("m003").text(), "m003")
        self.assertFalse(drafting.contains_secret(redacted))
        self.assertIn("credential withheld in m003", redacted)
        self.assertNotIn("Rk7-quiet-otter-51", redacted)

    def test_redaction_leaves_ordinary_text_alone(self):
        original = self.box.by_id("m036").text()
        self.assertEqual(drafting.redact_secrets(original, "m036"), original)


class TestEchoDetection(DraftingTestCase):
    """Grounded is not the same as written."""

    def test_the_real_echo_is_caught(self):
        # What a model actually returned as its reply to m043: m010's own
        # words, sent back to the person who wrote them.
        echo = (
            "I'd love 30 minutes to hear the vision before our partner meeting. "
            "Does Tuesday the 15th at 3:00pm work on your side?"
        )
        self.assertIsNotNone(drafting.echoes(echo, [self.box.by_id("m010").text()]))

    def test_quoting_a_fact_is_not_an_echo(self):
        # Both of these are correct replies that happen to reuse the wording of
        # the fact they cite. Neither may be rejected.
        m036 = self.box.by_id("m036").text()
        self.assertIsNone(drafting.echoes("Yes, the launch date is public. We are launching on the 20th.", [m036]))
        self.assertIsNone(drafting.echoes("The 20th is a hard date, and the press is briefed.", [m036]))

    def test_an_empty_draft_is_not_an_echo(self):
        self.assertIsNone(drafting.echoes("", [self.box.by_id("m010").text()]))

    def test_shared_run_counts_consecutive_words(self):
        self.assertEqual(drafting.longest_shared_run("a b c d", "x a b c d y"), 4)
        self.assertEqual(drafting.longest_shared_run("a b c", "c b a"), 1)
        self.assertEqual(drafting.longest_shared_run("", "anything"), 0)


class TestDraftValidation(DraftingTestCase):
    """The safety net, in the same shape as the Part 2 disposition validator."""

    def test_a_good_draft_is_accepted(self):
        result = self.parse('{"draft": "Yes, the date is public.", "cites": ["m036"]}')
        self.assertTrue(result.drafted)
        self.assertEqual(result.cites, ("m036",))
        self.assertEqual(result.message_id, "m019")

    def test_an_invented_detail_is_rejected(self):
        # Nothing in m036 or m019 mentions the 25th.
        with self.assertRaises(drafting.DraftRejected) as caught:
            self.parse('{"draft": "We are launching on the 25th.", "cites": ["m036"]}')
        self.assertIn("25th", str(caught.exception))

    def test_a_detail_from_the_message_itself_is_allowed(self):
        # The message being answered is grounding too, not only the evidence.
        result = self.parse('{"draft": "Confirming the booking for your space.", "cites": []}')
        self.assertTrue(result.drafted)

    def test_citing_the_message_being_answered_is_allowed_and_dropped(self):
        # A model did this for both m019 and m043. It is a reasonable
        # instinct and the id is really in the mail store, so it is not an
        # error -- but it is not grounding either, so it does not appear.
        result = self.parse('{"draft": "Confirming.", "cites": ["m019", "m036"]}')
        self.assertEqual(result.cites, ("m036",))

    def test_an_id_that_was_not_retrieved_is_rejected(self):
        with self.assertRaises(drafting.DraftRejected) as caught:
            self.parse('{"draft": "See earlier.", "cites": ["m001"]}')
        self.assertIn("m001", str(caught.exception))

    def test_an_id_that_is_not_a_message_is_rejected(self):
        with self.assertRaises(drafting.DraftRejected):
            self.parse('{"draft": "See earlier.", "cites": ["m999"]}')

    def test_an_id_missing_from_the_mail_store_is_rejected(self):
        # The spec asks for citations checked against the mail store, not merely
        # against what we believe we put in the prompt.
        ghost = retrieval.Evidence(message_id="m404", source="keyword", snippet="invented", subject="ghost")
        with self.assertRaises(drafting.DraftRejected) as caught:
            drafting.parse_draft(
                '{"draft": "As discussed.", "cites": ["m404"]}',
                self.box.by_id("m019"),
                (ghost,),
                self.box,
            )
        self.assertIn("mail store", str(caught.exception))

    def test_a_credential_is_rejected_even_when_correctly_cited(self):
        payload = (
            '{"draft": "The URL is amqp://pj_stage:Rk7-quiet-otter-51@broker-stg.paperjet.io:5672/pjs",'
            ' "cites": ["m003"]}'
        )
        with self.assertRaises(drafting.DraftRejected) as caught:
            self.parse(payload, message_id="m008", evidence_ids=("m003",))
        self.assertIn("credential", str(caught.exception))

    def test_an_echoed_draft_is_rejected(self):
        payload = (
            '{"draft": "I\'d love 30 minutes to hear the vision before our partner meeting. '
            'Does Tuesday the 15th at 3:00pm work on your side?", "cites": ["m010"]}'
        )
        with self.assertRaises(drafting.DraftRejected) as caught:
            self.parse(payload, message_id="m043", evidence_ids=("m010",))
        self.assertIn("consecutive words", str(caught.exception))

    def test_an_oversized_draft_is_rejected(self):
        payload = '{"draft": "%s", "cites": []}' % ("word " * 400)
        with self.assertRaises(drafting.DraftRejected):
            self.parse(payload)

    def test_junk_is_rejected(self):
        for payload in ("", "   ", "not json at all", "[OllamaAgent error: down]"):
            with self.assertRaises(drafting.DraftRejected):
                self.parse(payload)

    def test_cites_of_the_wrong_shape_are_rejected(self):
        with self.assertRaises(drafting.DraftRejected):
            self.parse('{"draft": "ok", "cites": {"id": "m036"}}')

    def test_a_single_cite_as_a_string_is_accepted(self):
        result = self.parse('{"draft": "Yes.", "cites": "m036"}')
        self.assertEqual(result.cites, ("m036",))


class TestDraftingNothing(DraftingTestCase):
    """Part 3.4: if the information is not in the inbox, say so and draft nothing."""

    def test_a_declared_refusal_is_a_valid_answer(self):
        result = self.parse('{"draft": null, "reason": "the inbox does not say when the review is"}')
        self.assertFalse(result.drafted)
        self.assertIn("inbox does not say", result.reason)

    def test_a_refusal_without_a_reason_is_rejected(self):
        with self.assertRaises(drafting.DraftRejected):
            self.parse('{"draft": null}')

    def test_an_empty_draft_string_counts_as_a_refusal(self):
        result = self.parse('{"draft": "   ", "reason": "nothing to say"}')
        self.assertFalse(result.drafted)

    def test_the_two_refusals_are_told_apart(self):
        # Rolling these together is what made a status update come back with its
        # own body as the draft: "this wants no reply" had nowhere to go.
        quiet = self.parse('{"draft": null, "outcome": "no_reply", "reason": "a status update"}')
        self.assertTrue(quiet.needs_no_reply)
        self.assertEqual(quiet.outcome, "no_reply")

        unknown = self.parse('{"draft": null, "outcome": "not_known", "reason": "not in the inbox"}')
        self.assertFalse(unknown.needs_no_reply)
        self.assertEqual(unknown.outcome, "not_known")

    def test_an_unlabelled_refusal_defaults_to_not_known(self):
        self.assertEqual(self.parse('{"draft": null, "reason": "no idea"}').outcome, "not_known")

    def test_an_outcome_outside_the_vocabulary_is_rejected(self):
        with self.assertRaises(drafting.DraftRejected):
            self.parse('{"draft": null, "outcome": "maybe", "reason": "no idea"}')

    def test_no_evidence_still_asks_the_model(self):
        # This used to return without calling anything, on the reasoning that an
        # ungrounded answer is a guess. Too broad: m015 is a standing request
        # ("CC me on anything from our lawyers") and the honest reply to it is
        # "noted", which needs no evidence at all.
        agent = StubAgent('{"draft": "Noted - I will CC you on anything from Hartwell & Cho.", "cites": []}')
        result = drafting.draft(agent, self.box.by_id("m015"), (), self.box)
        self.assertEqual(agent.calls, 1)
        self.assertTrue(result.drafted)
        self.assertIn("Noted", result.text)

    def test_the_owners_own_mail_is_never_replied_to(self):
        # m041 is Sam writing to himself. `agent=None` proves no model was
        # called: it would raise otherwise.
        result = drafting.draft(None, self.box.by_id("m041"), (), self.box)
        self.assertFalse(result.drafted)
        self.assertEqual(result.outcome, "no_reply")
        self.assertIn("owner's own message", result.reason)


class TestEchoOfTheIncomingMessage(DraftingTestCase):
    """The plagiarism source the first version of this check could not see.

    Measured on one full run: thirteen of twenty-four drafts lifted six or more
    consecutive words from the message they were replying to, and eight were
    that message word for word. All thirteen passed every check, because the
    echo test only ever compared a draft against the messages it *cited* -- and
    with nothing cited, against an empty string.
    """

    def test_the_incoming_message_played_back_is_rejected(self):
        # m034's own body, returned as the reply to m034.
        with self.assertRaises(drafting.DraftRejected) as caught:
            self.parse(
                '{"draft": "Hero image approved. Uploading final assets now.", "cites": []}',
                message_id="m034",
                evidence_ids=(),
            )
        self.assertIn("copies", str(caught.exception))

    def test_it_is_caught_with_no_citations_at_all(self):
        # The exact hole: an empty `cites` used to mean an empty comparison set.
        for message_id in ("m005", "m027", "m029", "m035", "m038"):
            body = self.box.by_id(message_id).body
            with self.assertRaises(drafting.DraftRejected, msg=message_id):
                self.parse(f'{{"draft": {body!r}, "cites": []}}'.replace("'", '"'), message_id, ())

    def test_a_written_reply_to_the_same_message_is_accepted(self):
        result = self.parse(
            '{"draft": "Thanks for the update - nothing needed from me on the hero image.", "cites": []}',
            message_id="m034",
            evidence_ids=(),
        )
        self.assertTrue(result.drafted)

    def test_one_inserted_word_does_not_defeat_the_check(self):
        # What the model actually returned for m027 once the run-length check
        # was in place: its own body with the word "is" added, which cut the
        # longest run from eleven words to seven and slipped underneath.
        with self.assertRaises(drafting.DraftRejected):
            self.parse(
                '{"draft": "Design assets are 80% there. Hero image is still in review.", "cites": []}',
                message_id="m027",
                evidence_ids=(),
            )

    def test_a_short_confirmation_is_not_an_echo(self):
        # Confirming a proposed time shares almost every word with the message
        # proposing it. The subsequence rule exempts short replies for exactly
        # this reason.
        body = self.box.by_id("m016").text()
        self.assertIsNone(drafting.echoes("Yes, that slot works on our side.", [body]))

    def test_a_real_reply_sharing_vocabulary_is_not_an_echo(self):
        body = self.box.by_id("m015").text()
        draft = "I will make sure to CC you on all messages from our lawyers at Hartwell & Cho from now on."
        self.assertIsNone(drafting.echoes(draft, [body]))

    def test_echoing_a_cited_message_is_still_caught(self):
        with self.assertRaises(drafting.DraftRejected):
            self.parse(
                '{"draft": "Looking good team. Reminder the 20th is a hard date, press is briefed.", '
                '"cites": ["m036"]}'
            )


class TestCompletionClaims(DraftingTestCase):
    """Saying the work is done when it is not.

    Nothing is invented, nothing is copied, every word is on topic, and the
    claim is false. m048 asked Sam to review board minutes and flag corrections;
    the draft told outside counsel that both had been done.
    """

    def test_claiming_work_the_message_asks_for_is_rejected(self):
        with self.assertRaises(drafting.DraftRejected) as caught:
            self.parse(
                '{"draft": "Yes, I have reviewed the minutes and will send corrections.", "cites": []}',
                message_id="m048",
                evidence_ids=(),
            )
        self.assertIn("request to review", str(caught.exception))

    def test_claiming_work_nobody_asked_for_is_also_rejected(self):
        # m055 asks for a signature, not a review, so the "was it requested?"
        # half does not fire -- but nothing in the mail says a review happened.
        with self.assertRaises(drafting.DraftRejected) as caught:
            self.parse(
                '{"draft": "I have reviewed the IP assignment and will sign by Friday.", "cites": []}',
                message_id="m055",
                evidence_ids=(),
            )
        self.assertIn("nothing in the mail says", str(caught.exception))

    def test_a_commitment_is_not_a_claim(self):
        # "I will sign it" is a promise, not a lie. Whether the owner should be
        # making one is a separate question from whether the draft is true.
        result = self.parse(
            '{"draft": "Thanks - I will go through the minutes and send any corrections before Monday.", '
            '"cites": []}',
            message_id="m048",
            evidence_ids=(),
        )
        self.assertTrue(result.drafted)

    def test_a_past_participle_used_as_an_adjective_is_not_a_claim(self):
        # "the updated IP assignment" is a noun phrase. An earlier version of
        # this check read it as a claim to have updated something.
        self.assertEqual(drafting.completion_claims("I have seen the updated IP assignment"), [])

    def test_a_claim_the_evidence_supports_is_allowed(self):
        # m003 is Sam's own message saying the creds were rotated. A draft that
        # says so, and cites it, is telling the truth.
        claims = drafting.unsupported_claims(
            "I have rotated the broker creds already.",
            self.box.by_id("m008"),
            self.box.by_id("m003").text(),
        )
        self.assertEqual(claims, [])

    def test_somebody_else_finishing_something_is_not_a_claim(self):
        self.assertEqual(drafting.completion_claims("Raghav has already signed it"), [])


class TestBorrowedSpecifics(DraftingTestCase):
    """A fact taken out of the evidence has to say where it came from."""

    def test_a_fact_lifted_from_uncited_evidence_is_rejected(self):
        # "the 20th" is m036's, and m036 was offered as evidence and not cited.
        with self.assertRaises(drafting.DraftRejected) as caught:
            self.parse('{"draft": "We go live on the 20th.", "cites": []}')
        self.assertIn("cite what you used", str(caught.exception))

    def test_the_same_fact_is_fine_once_cited(self):
        result = self.parse('{"draft": "We go live on the 20th.", "cites": ["m036"]}')
        self.assertEqual(list(result.cites), ["m036"])

    def test_a_fact_from_the_message_itself_needs_no_citation(self):
        # Confirming the sender's own proposed time back to them.
        result = self.parse(
            '{"draft": "Wednesday at 2:00pm works on our side.", "cites": []}',
            message_id="m016",
            evidence_ids=(),
        )
        self.assertTrue(result.drafted)


class TestWiderSpecifics(DraftingTestCase):
    """Durations and proportions are numbers about the world too."""

    def test_a_duration_is_a_specific(self):
        self.assertIn("30 minutes", drafting.specifics("I can do 30 minutes on Tuesday"))
        self.assertIn("20-minute", drafting.specifics("a 20-minute slot"))

    def test_a_percentage_is_a_specific(self):
        self.assertIn("80%", drafting.specifics("design is 80% done"))

    def test_an_invented_duration_is_rejected(self):
        with self.assertRaises(drafting.DraftRejected):
            self.parse('{"draft": "Happy to do a 45-minute call.", "cites": []}', "m010", ())


class TestPromptsCarryNoAnswers(DraftingTestCase):
    """A worked example built from this inbox is an answer key, not an example.

    The first version of the drafter's system prompt illustrated a good answer
    using m036 and the launch date -- which is m046's grounding and m046's
    answer. The model then returned that example almost verbatim as its reply
    to m046, and the citation it produced was the example's citation. The
    capability looked like it worked; what worked was recitation.
    """

    def test_no_real_message_id_appears_in_a_system_prompt(self):
        real = {message.id for message in self.box.messages}
        for name in ("DRAFT_SYSTEM_PROMPT", "TRIAGE_SYSTEM_PROMPT"):
            found = {mid for mid in re.findall(r"\bm\d{3}\b", getattr(agents, name)) if mid in real}
            self.assertEqual(found, set(), f"{name} names real inbox messages: {sorted(found)}")

    def test_no_sentence_from_the_inbox_appears_in_the_drafting_prompt(self):
        # A weaker id check would miss the wording; this catches the body text
        # being pasted in without its id.
        for message in self.box.messages:
            shared = drafting.longest_shared_run(agents.DRAFT_SYSTEM_PROMPT, message.body)
            self.assertLess(shared, 8, f"{message.id}'s wording is in the drafting prompt")


class TestInstructionLeak(DraftingTestCase):
    """The draft may be grounded in the mail. It may not be grounded in the prompt.

    m046 asks what makes the product different -- a fact the inbox contains
    nowhere. The model answered a journalist with "a hero assistant that clears
    one person's inbox", which is the drafter's own description of itself, taken
    from the first line of its system prompt. Nothing was invented in the usual
    sense, nothing was copied from any message, no claim about work was made:
    every other check passed it, and it was addressed to a press contact as
    PaperJet's positioning.
    """

    def setUp(self):
        self.system_prompt = agents.DRAFT_SYSTEM_PROMPT.format(owner="sam@paperjet.io")

    def test_the_real_leak_is_caught(self):
        leaked = (
            "The launch date is confirmed as the 20th, and I can share that the product's "
            "difference is that it is a hero assistant that clears one person's inbox."
        )
        self.assertTrue(drafting.instruction_leak(leaked, self.system_prompt))
        with self.assertRaises(drafting.DraftRejected) as caught:
            drafting.parse_draft(
                json.dumps({"draft": leaked, "cites": ["m036"]}),
                self.box.by_id("m046"),
                evidence_for(self.box, "m036"),
                self.box,
                self.system_prompt,
            )
        self.assertIn("own instructions", str(caught.exception))

    def test_ordinary_drafts_are_not_flagged(self):
        # Every draft a real run produced, minus the leak above. The prompt is
        # full of ordinary English and a reply is allowed to share words with it.
        for draft in (
            "I will take a look at the staging issue when I get in.",
            "Thanks for the update on the staging deploy.",
            "Wednesday at 2:00pm works for me. I'll use the same link.",
            "I will review the draft board minutes and flag any corrections by Monday.",
            "Yes, please send over the contract.",
            "The new staging AMQP URL was provided in my reply to the thread about staging being down again.",
        ):
            self.assertEqual(drafting.instruction_leak(draft, self.system_prompt), 0, draft)

    def test_a_short_overlap_is_not_a_leak(self):
        self.assertEqual(drafting.instruction_leak("the launch date is public", self.system_prompt), 0)

    def test_no_prompt_means_no_check(self):
        self.assertEqual(drafting.instruction_leak("anything at all", ""), 0)


class TestInternalIdsStayInternal(DraftingTestCase):
    """A message id is ours. The person reading the reply has never seen one."""

    def test_a_draft_naming_a_message_id_is_rejected(self):
        # What the credential reply actually said: "the URL was provided in my
        # reply to the initial issue in m003". Devika has no m003.
        with self.assertRaises(drafting.DraftRejected) as caught:
            self.parse(
                '{"draft": "The URL was provided in my reply in m036.", "cites": ["m036"]}',
            )
        self.assertIn("m036", str(caught.exception))
        self.assertIn("cites", str(caught.exception))

    def test_describing_the_message_in_words_is_accepted(self):
        result = self.parse(
            '{"draft": "The date was confirmed in the launch thread earlier this week.", "cites": ["m036"]}'
        )
        self.assertTrue(result.drafted)
        self.assertEqual(result.cites, ("m036",))

    def test_an_id_shaped_string_that_is_not_a_message_is_left_alone(self):
        # Only ids that really are in the mail store are rejected, so ordinary
        # text that happens to match the shape is not caught.
        result = self.parse('{"draft": "Our meeting room is m999 on the third floor.", "cites": []}')
        self.assertTrue(result.drafted)


class TestNoEvidencePrompt(DraftingTestCase):
    """Having no evidence is three different situations, not one.

    Collapsing them is what produced a reply to m012 -- "did you ever get a
    chance to sort out that thing we talked about" -- reading "I will sort out
    the thing we talked about": a commitment to something neither party has
    named, invented whole, from an empty evidence set.
    """

    def setUp(self):
        self.prompt = drafting.build_draft_prompt(self.box.by_id("m012"), (), owner="sam@paperjet.io")

    def test_the_prompt_offers_all_three_readings(self):
        self.assertIn("asks for nothing factual", self.prompt)
        self.assertIn("named in the message itself", self.prompt)
        self.assertIn("cannot identify", self.prompt)

    def test_the_unidentifiable_case_names_the_outcome(self):
        self.assertIn("not_known", self.prompt)

    def test_a_declared_not_known_survives_the_round_trip(self):
        agent = StubAgent(
            '{"draft": null, "outcome": "not_known", '
            '"reason": "the message refers to something discussed after the standup that is not in the inbox"}'
        )
        result = drafting.draft(agent, self.box.by_id("m012"), (), self.box)
        self.assertFalse(result.drafted)
        self.assertEqual(result.outcome, "not_known")
        self.assertFalse(result.needs_no_reply)


class TestAsksAnything(DraftingTestCase):
    """The deterministic half of deciding a message wants no reply."""

    def test_a_status_update_asks_nothing(self):
        for message_id in ("m005", "m026", "m027", "m033", "m034"):
            self.assertFalse(drafting.asks_anything(self.box.by_id(message_id)), message_id)

    def test_a_request_is_recognised(self):
        for message_id in ("m008", "m015", "m012", "m013", "m016", "m046", "m048"):
            self.assertTrue(drafting.asks_anything(self.box.by_id(message_id)), message_id)

    def test_a_question_mark_is_enough_on_its_own(self):
        self.assertTrue(drafting.asks_anything(self.box.by_id("m013")))


class TestInferredNoReply(DraftingTestCase):
    """A repeated echo, on a message that asks nothing, is an answer in itself.

    A small model never once emits outcome 'no_reply' -- not with the outcome in
    its system prompt, not with a worked example, not with a hint attached to
    the rejection. It answers a status update by handing it back, twice, and
    then the attempts run out. A larger one reaches for 'no_reply' unprompted
    and never reaches this path, so this is a floor under the weaker case rather
    than something the system depends on.
    """

    def test_a_message_that_asks_nothing_becomes_no_reply(self):
        message = self.box.by_id("m034")
        agent = StubAgent(f'{{"draft": {message.body!r}, "cites": []}}'.replace("'", '"'))
        result = drafting.draft(agent, message, (), self.box)
        self.assertFalse(result.drafted)
        self.assertEqual(result.outcome, "no_reply")
        self.assertIn("asks for nothing", result.reason)

    def test_a_message_that_does_ask_stays_a_rejection(self):
        # The guard that stops this becoming a way to silence real mail: m048
        # asks for a review, so a model that can only echo it is a failure and
        # is recorded as one.
        message = self.box.by_id("m048")
        agent = StubAgent(f'{{"draft": {message.body!r}, "cites": []}}'.replace("'", '"'))
        result = drafting.draft(agent, message, (), self.box)
        self.assertFalse(result.drafted)
        self.assertEqual(result.outcome, "rejected")

    def test_a_rejection_for_another_reason_is_not_inferred_away(self):
        # Only a repeated *echo* means this. An invented detail means the model
        # was guessing, which is a different failure and stays visible.
        message = self.box.by_id("m034")
        agent = StubAgent('{"draft": "Confirmed for 2029-01-01 at 4:00pm.", "cites": []}')
        result = drafting.draft(agent, message, (), self.box)
        self.assertEqual(result.outcome, "rejected")


class TestDraftPrompt(DraftingTestCase):
    """The prompt's order is load-bearing, so it is asserted rather than assumed."""

    def setUp(self):
        self.message = self.box.by_id("m019")
        self.prompt = drafting.build_draft_prompt(
            self.message, evidence_for(self.box, "m036", "m046"), owner="sam@paperjet.io"
        )

    def test_the_message_comes_before_the_background(self):
        # With the background first, every model tried answered the background,
        # returning the same sentence for m019 and m046.
        self.assertLess(self.prompt.index("REPLY TO THIS MESSAGE"), self.prompt.index("BACKGROUND"))
        self.assertLess(self.prompt.index("<<<UNTRUSTED MESSAGE id=m019"), self.prompt.index("<<<UNTRUSTED EVIDENCE"))

    def test_the_parties_are_named(self):
        self.assertIn("writing as sam@paperjet.io", self.prompt)
        self.assertIn("replying to events@thegrandvenue.com", self.prompt)

    def test_the_background_is_marked_as_not_to_be_answered(self):
        self.assertIn("Do NOT reply to these", self.prompt)

    def test_the_message_is_named_again_at_the_end(self):
        self.assertIn("Write the reply to m019", self.prompt)

    def test_a_credential_never_reaches_the_prompt(self):
        prompt = drafting.build_draft_prompt(
            self.box.by_id("m008"), evidence_for(self.box, "m003"), withhold_secret=True, owner="sam@paperjet.io"
        )
        self.assertNotIn("Rk7-quiet-otter-51", prompt)
        self.assertIn("credential withheld", prompt)

    def test_an_evidence_free_prompt_still_asks_for_a_reply(self):
        prompt = drafting.build_draft_prompt(self.message, (), owner="sam@paperjet.io")
        self.assertIn("REPLY TO THIS MESSAGE", prompt)
        self.assertNotIn("BACKGROUND", prompt)


class TestDrafterAgent(DraftingTestCase):
    """Drafting is a different job from triage, so it is a different agent."""

    def test_the_drafter_has_its_own_system_prompt(self):
        self.assertNotEqual(agents.DRAFT_SYSTEM_PROMPT, agents.TRIAGE_SYSTEM_PROMPT)
        self.assertIn("drafting stage", agents.DRAFT_SYSTEM_PROMPT)

    def test_the_drafter_has_no_tools(self):
        # No agent in this system may act, not only the triage one.
        self.assertEqual(agents.drafter_agent().discover_tools(), [])

    def test_the_drafter_runs_on_the_configured_model(self):
        # One model for the whole system. The drafter is a separate agent
        # because of its system prompt, not because of its model.
        import config

        self.assertEqual(agents.drafter_agent().model_name, config.MODEL)
        self.assertEqual(agents.triage_agent().model_name, config.MODEL)
