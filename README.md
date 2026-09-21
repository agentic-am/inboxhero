# inboxHero

Repository: https://github.com/agentic-am/inboxhero

An agentic system that takes `data/inbox.json` from unread to empty: every message
gets exactly one disposition and a reason, the reversible work is done, the
irreversible work goes through a gate, and hostile content is refused and reported.

## Part 1: what is in the inbox

Read in full before any code was written. `data/inbox.json` is the file supplied
with the assignment, copied unchanged.

**Messages processed: 100** (all of them, not only the 77 unread).

### Assumptions about the data format

- The file is a top-level JSON array of 100 objects, not wrapped in a dict. Every
  record has exactly the eight keys `id`, `thread_id`, `from`, `to`, `subject`,
  `timestamp`, `body`, `unread`. All are strings except `unread`, a boolean. No value
  is empty. The loader still validates each record and escalates a bad one instead
  of crashing.
- Ids run `m001` to `m119` with 19 gaps, and the file is not sorted by id.
  Messages are processed in timestamp order.
- Timestamps are ISO 8601 with no timezone (`2026-09-02T09:12:00`), spanning 2 to
  9 Sep 2026, and are treated as the owner's local time. Dates inside bodies
  ("the 20th", "Friday") carry no month, so they are resolved relative to the
  message timestamp.
- `to` is a single address string. There is no CC field, so a "CC someone"
  preference is applied to the recipients of the drafted reply.
- The file mixes received and sent mail. Four messages are from the owner
  `sam@paperjet.io` (m003 and m044 to colleagues, m039 and m041 to self), so a
  sender of the owner's address is not proof that the owner wrote it.
- There are 88 threads, but only three hold more than one message (`t-api` 4,
  `t-launch` 9, `t-invest` 2). Walking the thread cannot ground cross-thread asks
  such as m019, so keyword search is also needed.
- There is no attachments field. Bodies that say "attached" refer to a portal; the
  system never claims to have read an attachment.
- Four bodies span several lines (m017, m021, m024, m047). Quoted text prefixed
  with `>` is still message content and still untrusted.
- `paperjet.io` is the owner's trusted domain. Lookalike domains (`paperjet.co`,
  `paperjet-helpdesk.com`) are treated as fraud only together with urgency and a
  request for money, credentials or a link, so the board's `paperjet-board.org` is
  not misflagged.

### First design decision: what needs a model and what does not

| Tier                                                        | Messages                                                                                                                                                                                                                                                                                             | Count |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| Rules, no model: noise                                      | receipts, newsletters, notifications, alerts (m062 to m116); internal automated notices (m117 to m119); automated "no action needed" mail (m049, m053, m057); the dentist reminder m061, deferred with its slot recorded                                                                             | 62    |
| Rules, no model: hostile                                    | instructions addressed to the assistant (m017, m024, m047); phishing and social engineering (m021, m023, m045); a spoofed "assistant settings" note from the owner's own address (m039)                                                                                                              | 7     |
| Model, workflow-shaped: classify, retrieve, draft, validate | the staging thread (m001, m003, m005, m008); the launch thread with the real ask buried in m030 (m026 to m036); board review and deck (m038, m040); legal (m018, m048, m055); venue m019, press m046, candidate m042, the owner's own unanswered m044, PTO m059, coffee m051, and the ambiguous m012 | 25    |
| Model, agent-shaped: memory, planning, human in the loop    | scheduling requests that must be checked against the calendar rule and each other (m010, m043, m013, m016); standing preferences to record (m015, m041)                                                                                                                                              | 6     |

Roughly 69 messages never need a model call. The exact `rule_handled` figure is
reported by the run and goes into the manifest.

## Part 2: zeroing it

```
python demo.py --cap R1              # all 100 messages
python demo.py --cap R1 --limit 12   # the first 12 by timestamp, for a quick look
python demo.py --msg m024            # one message
```

### The disposition vocabulary

Exactly one of these is assigned to every record, with a reason in plain English.

| Disposition | Meaning                                                                       | Who may assign it              |
| ----------- | ----------------------------------------------------------------------------- | ------------------------------ |
| `reply`     | the owner should answer, and an answer can be built from what is in the inbox | model, after validation        |
| `archive`   | nothing is needed; noise, or a loop that is already closed                    | rules or model                 |
| `defer`     | the owner must deal with it later, or it carries a date worth keeping         | rules or model                 |
| `delegate`  | somebody else at the owner's company should own it                            | model                          |
| `escalate`  | the owner must look at it now, or the system could not decide safely          | rules, model, or the validator |
| `flag`      | hostile or fraudulent; refused, reported, and left in place                   | rules or the validator only    |

`flag` is never offered to the model. It is removed from the allowed list on
every model-path message, so the model can neither apply it nor remove it. A
test asserts that.

### How the routing works

One Moya `Pipeline` runs per message, and a `BranchStep` decides whether the
model is called at all.

```
rule  ->  route  -+->  rule decision                      -+->  record
                  |                                        |
                  +->  prompt  ->  agent  ->  validate  ---+
```

The rule tier in `rules.py` looks for four things, in this order: content
addressed to the assistant, fraud, automated mail, and then everything else. The
order matters. A hostile note arrives inside a routine support forward (m047)
and a phishing invoice comes from a billing address (m021), so if noise were
matched first both would be quietly archived instead of flagged.

Detection is by sender and body features, never by the message id or the thread
name. Flagging an instruction aimed at the assistant needs two signals, not one:
the text must address an assistant _and_ try to conceal, override, widen
autonomy, or move mail out. That second condition is what separates the four
attacks from m041, which is a real standing instruction from the owner that is
also addressed to the assistant and must not be flagged.

### The rule tier is not decoration

A stage whose output never reaches the model is decoration. Here every rule
conclusion does two jobs.

- It goes into the prompt, as the `ALLOWED` list and as plain-English notes about
  what the rules established.
- It is applied again in Python after the model answers. `agents.parse_proposal`
  rejects any disposition outside that same allowed list, and the message is
  escalated instead.

The clearest case is m003 and m044, which the owner sent to somebody else. The
rule tier removes `reply` before the prompt is built, and the validator rejects
`reply` if the model picks it anyway. A test drives exactly that.

### Boundaries that are checked

| Boundary     | Bad input                                                                        | What happens                                                             |
| ------------ | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| inbox file   | missing, not JSON, not an array                                                  | one sentence on stderr, exit 1, no traceback                             |
| inbox record | wrong type, non-ISO timestamp, bad address, duplicate id, oversized body         | the record is kept, named as malformed, and escalated; the run continues |
| model output | not JSON, unknown key, disposition outside the allowed list, a Moya error string | one corrective retry, then `escalate` with the reason recorded           |
| command line | unknown `--cap`, unknown `--msg`, `--limit 0`, no arguments                      | one sentence on stderr, exit 2, usage hint                               |
| rule tier    | a rule that raises                                                               | that message escalates; the other 99 are still processed                 |

### What a full run produces

`python demo.py --cap R1` over all 100 records, on the default `llama3.2:3b`:

|                                      |                                                               |
| ------------------------------------ | ------------------------------------------------------------- |
| messages processed                   | 100                                                           |
| undecided                            | 0                                                             |
| decided by rules, with no model call | 66                                                            |
| sent to the model                    | 34                                                            |
| dispositions                         | archive 70, defer 14, flag 7, delegate 4, escalate 3, reply 2 |

The seven flagged messages are m017, m021, m023, m024, m039, m045 and m047. None
is deleted; each is reported in the run summary with what it attempted.

The run writes `state/decisions.json` (one row per message) and `trace.jsonl`
(every step, including Moya's own pipeline and step events through the
`EventBus`). Both are reproducible from a run; nothing in them is hand-written.

A full run takes about twenty minutes, because the local model needs roughly
thirty seconds per message on CPU. `--limit N` runs the first N messages by
timestamp, which is enough to see the routing and the validator at work.

### The model matters more than the prompt

Nothing about the model is hard-coded. `MODEL` comes from the environment, so
the same code runs on any Ollama model:

```
MODEL=gemma4:e4b python demo.py --cap R1
```

That is worth trying, because the choice changes the output more than any prompt
edit did. Over the same eight messages, with an identical prompt:

| Model                   | Sensible dispositions | Seconds per message |
| ----------------------- | --------------------- | ------------------- |
| `llama3.2:3b` (default) | 3 of 8                | about 30            |
| `gemma4:e4b`            | 8 of 8                | about 51            |

The clearest single case is m008, where a colleague asks for the staging queue
credentials that appear in m003. `gemma4:e4b` escalates it. `llama3.2:3b`
delegates it, which is the wrong answer about a credential.

This is also the argument for the validator. The weaker model's answers were
well-formed JSON with plausible-sounding reasons, so nothing about the shape of
the output revealed the problem. Only the deterministic checks, and a human
reading the table, did. An earlier version of the prompt made it worse still:
`llama3.2:3b` used four of the six dispositions and never once chose `reply`,
while every answer still parsed cleanly.

### Falling back to Gemini

`provider.py` holds two backends behind one `chat()` call. Ollama is the
primary. Gemini is used when Ollama cannot be reached at all, and the switch
lasts for the rest of the run so the system does not flap between models. A bad
answer never triggers the fallback; the validator handles those.

To enable it, copy `.env.example` to `.env` and set `GEMINI_API_KEY`. `.env` is
gitignored and is not part of the submission, which is what the spec asks for.

```
cp .env.example .env      # then paste the key into GEMINI_API_KEY
python provider.py        # one JSON round trip, proves the wiring
PROVIDER=gemini python demo.py --cap R1
```

The free tier allows roughly fifteen requests a minute, so calls are paced. The
default spacing on Gemini is 4.5 seconds, about thirteen a minute, chosen rather
than a flat four so a clock disagreement does not put the run over the line.
HTTP 429 and 5xx are retried up to `MAX_RETRIES` times with exponential backoff,
honouring `Retry-After`. A message whose call never succeeds is escalated with
the reason recorded, never guessed at.

A full run needs 34 model calls, because the other 66 messages never reach a
model. At 4.5 seconds apart that is about three minutes, against roughly twenty
for the local model on CPU, and it costs 34 requests against the daily cap.

### Batching, and why it is off by default

`BATCH_SIZE` controls how many messages share one model call. It defaults to 1,
and `--batch N` overrides it for a single run. Messages the rules marked
sensitive are never batched whatever it is set to, and any message whose batched
answer fails validation is asked again on its own, so a bad group costs one
extra call rather than a wrong decision.

The default is 1 for two reasons, one of which was measured rather than assumed.

**Isolation.** One message per call means text written by one sender cannot
reach the decision about another. Batching gives that up: an injection we failed
to detect could influence the messages sitting beside it in the same context.

**On a local model it is slower, not faster.** Running the same inbox at
`--batch 8`:

|                                            | batch 1 | batch 8  |
| ------------------------------------------ | ------- | -------- |
| wall clock                                 | 6.7 min | 28.1 min |
| model requests                             | 34      | 24       |
| prompt tokens                              | 34,382  | 26,441   |
| timeouts and retries                       | 0       | 3        |
| model decisions matching the unbatched run | -       | 30 of 34 |

The reason is that generation, not prompt reading, is the cost on CPU. A batch
of eight produces eight answers, so the expensive part does not shrink, while
the context grows. One batch exceeded the 180-second timeout three times and
fell back to individual calls anyway. Quality dropped too: m030, where Priya
asks Sam directly to approve the pricing copy by the 12th, came back as
`archive` in the batched run and `defer` unbatched.

**Where batching does pay is a rate-limited API.** On Gemini's free tier the
binding constraint is fifteen requests a minute, not compute, so folding 34
requests into roughly 19 cuts both the wall clock and the daily-cap usage. That
is the case for raising `BATCH_SIZE`, and it is the only case:

```
PROVIDER=gemini python demo.py --cap R1 --batch 8
```

## Part 3: grounding a decision in the inbox

Part 2 gives every message a disposition. It does not answer any of them, and
for some messages a disposition cannot honestly be chosen without knowing what
an earlier message said. m040 asks for the board deck "two days before the board
review" — nothing in m040 says when that is. Part 3 is what closes that gap, in
two places:

- **Before the disposition is chosen.** The 34 messages that reach the model get
  the relevant earlier messages attached to the prompt. 25 of the 34 have
  something to attach; the other 9 are told explicitly that nothing was found,
  so the model states that rather than inventing it.
- **After it is chosen.** Any message whose disposition is `reply` gets a
  drafted reply, grounded in specific cited messages.

### The query comes from the message, not from a question

This is the decision the rest of the design follows from. The system is not
answering "who is on vacation"; it is asking what grounds m019. The query is
therefore derived from the message being triaged, so query and target usually
share vocabulary, and lexical matching carries the work that a general-purpose
search engine would need embeddings for.

Retrieval runs in four tiers, and only on the model path, so the 66 messages
the rules settle cost nothing:

| Tier | What it does | Window applies? |
| --- | --- | --- |
| 0 scope | earlier than this message, not itself, never hostile | n/a |
| 1 thread walk | earlier messages in the same thread, oldest first | no — threads outlive any window |
| 2 keyword | SQLite FTS5 over subject and body, BM25 | yes — the only tier whose cost grows with the corpus |
| 3 entity | exact lookup of references, addresses, domains | no — a reference is worth finding however old |

Measured over the whole inbox: 37 pieces of evidence came from the thread tier,
56 from the keyword tier, and **none from the entity tier**. That last number is
honest rather than embarrassing: of the twelve values it extracts, only two
appear in more than one message, and both pairs are automated mail the rules
settle without a model. The tier is kept because references recur constantly in
a real mailbox and because it is the one tier whose cost does not grow with the
inbox, but nothing here depends on it.

### Why keyword search rather than embeddings

Naive keyword matching on human-phrased questions is genuinely bad — measured
at 3 of 10 on this inbox. It is the *term selection* that makes it work, not the
index. Picking the rarest words picks the incidental ones: m046's rarest word is
"coverage", which appears nowhere else and can ground nothing, while the words
that actually ground it are "launch" and "date". So terms are scored by why they
look topical — a reference pattern, a subject word, known vocabulary, a proper
noun — and a term needs either one strong signal or two weak ones before it is
queried at all. A message with nothing topical to say produces no query, which
is the correct query for it.

Nine groundings were read out of the inbox by hand and are kept as a regression
test in `tests/test_part3.py`: m008 needs m003, m019 and m046 need m026 or m036
across thread boundaries, m040 needs m038, and m012 needs nothing. Recall is
**9 of 9**.

The synonym map in `retrieval.py` is the entire semantic layer, written down
where a test can read it. `EMBEDDINGS=off` is a seam rather than a missing
feature: because tier 0 has already narrowed the candidates to a few hundred, a
vector tier would be a flat scan over that set, needing no vector store and no
approximate index.

### Boundaries this part adds

**Nothing is grounded in the future.** Only messages that had already arrived
may be cited. Without it, replaying the run produces different evidence than the
run did, and the trace stops being an audit record.

**Indexed and retrievable are different permissions.** The seven hostile
messages stay in the index — m021 has to be findable — but are never returned as
evidence, because evidence is quoted into a prompt. Retrieved mail is wrapped in
its own `<<<UNTRUSTED EVIDENCE>>>` markers for the same reason.

**Message text never becomes query syntax.** FTS5's query language is a
language: a bare `1:1` parses as a column filter and raises `no such column: 1`,
and it is live in m013, m016 and m119. Every term is quoted before it reaches
`MATCH`.

**A citation is checked in Python, like the allowed list.** The model may cite
only ids it was actually shown. This is not theoretical — on the first live run
of m019 the model cited `m019` itself, the citation was rejected, and the
retry came back clean:

```
proposal_rejected  cited 'm019', which was not in the evidence (m035, m036, m038, m046, m096)
validate           accepted=true  attempts=2  evidence=[m036, m046, m096, m038, m035]
```

### What it costs at a real mailbox size

The tiers were written against 11,000 messages a day rather than 100, because
the shape of the answer changes with scale and the seams are cheaper to leave in
than to retrofit. Measured on synthetic corpora at 200k messages:

| | |
| --- | --- |
| thread walk, indexed on `(thread_id, sent_at)` | 0.28 ms |
| thread walk, no index | 381 ms |
| FTS5 query, rare term | 0.48 ms |
| FTS5 query, one unselective term added to eight good ones | 409 ms |

Two things follow. Thread-walk cost depends on thread length, not corpus size,
so tier 1 stays free at any scale provided the index exists. And FTS5 cost
tracks document frequency, not corpus size, which is why the document-frequency
filter in tier 2 is load-bearing rather than tidying.

What does *not* survive that scale is holding the whole mailbox in memory: 4M
messages is roughly 6 GB of text before Python object overhead. `retrieval.Index`
is the seam for that — swapping its in-memory SQLite for a file-backed table
leaves every tier above it unchanged.

### Answering properly: the drafted reply

Retrieval finds the evidence; `drafting.py` turns it into an answer.

```
python demo.py --cap R2                 # draft for everything R1 decided to reply to
python demo.py --cap R2 --msg m046      # one named message
python demo.py --cap R2 --limit 5       # the first five
```

**R2 does not choose its own messages.** It reads `state/decisions.json` and
drafts for the rows R1 dispositioned `reply` — 21 of the 100 on the recorded
run. Which mail deserves a reply is the triage tier's judgement; re-deciding it
here would mean one system giving the same inbox two answers, and the second
would be the one nobody reviewed. R2 writes its drafts back into those rows and
touches no disposition, and it appends to the trace rather than truncating it,
so one file holds the triage of all 100 messages and the drafting that followed.

An earlier version derived the set from the rules instead — every message where
`reply` was *allowed*, 32 of them. That handed the drafter status updates triage
would have archived, and it answered them by sending the message back: 13 of 24
drafts lifted six or more consecutive words from the message they answered, 8
were that message verbatim. That measurement shaped the echo check below.

#### The four things Part 3 asks for

**1. A reply grounded in a specific earlier message.** m046 is a journalist
asking whether the launch date is public. Nothing in m046 answers that; m036,
five days earlier in a thread m046 is not part of, says *"the 20th is a hard
date, press is briefed"*.

```
m046  retrieved: m036 (keyword, matched launch, date, press)
      draft: "The launch date is confirmed as the 20th, according to a previous note."
      cited: [m036]
```

**2. Every draft records the ids it drew on, checked against the mail store.**
A cited id must be in the evidence *and* in the store. Citing the message being
answered is dropped rather than rejected — a message is not evidence for itself.

**3. The retrieval method is named:** thread walk plus keyword search over
SQLite FTS5, with an exact-entity tier. Described above; named in the manifest.

**4. If the inbox does not hold the answer, the system says so and drafts
nothing.** m012 asks *"did you ever get a chance to sort out that thing we
talked about after the standup?"* — a conversation the inbox does not contain.

```
m012  retrieved: nothing
      no draft -- The message refers to "that thing we talked about after the
      standup," which is not identifiable from the provided context.
```

Having no evidence is three situations, not one. The first version refused
whenever retrieval came back empty, which also refused m015 — a standing request
(*"CC me on anything from our lawyers"*) that asks for no fact and whose honest
reply is "noted". The drafter is now told which case it is in: the message asks
nothing factual, everything it asks is named in the message itself, or it refers
to something unidentifiable. Only the third is `not_known`.

#### Eight checks, each from something that failed

| Check | What it caught |
| --- | --- |
| **invented** | a date, time, amount, duration or URL in no cited message and not in the message answered |
| **borrowed** | a fact lifted out of evidence that was offered and not cited |
| **miscited** | an id the model was never shown, or one absent from the mail store |
| **leaked credential** | a credential copied out of a message the draft cited correctly |
| **echoed** | the message played back as the reply — a status update returned to the person who wrote it |
| **claimed done** | *"I have reviewed the minutes"* in answer to *"please review the minutes"* |
| **internal id** | *"the URL is in m003"* — an identifier the recipient cannot resolve |
| **instruction leak** | the system prompt quoted into a reply as fact |

Three are worth the detail.

**Echo.** A run of consecutive words is trivially broken, and was: the model
returned one message's body with "is" inserted, cutting the longest run from
eleven words to seven. Order-preserving overlap is measured too. What separates
a copy from a quote is not how much of the *draft* came from the source but how
much of the *source* the draft gave back — *"the 20th is a hard date, and the
press is briefed"* takes nine of its eleven words from m036 and is a good reply,
because it covers half of m036. Reproducing a message is the failure; quoting a
line out of one is the job.

**Claimed done.** Asked to review board minutes and flag corrections, the draft
replied that both were done. Nothing invented, nothing copied, every word on
topic — only the tense false, and every other check passed it. A completion
claim is now grounded like a date: *"I have rotated the creds"* is fine when a
cited message says they were rotated. *"I will sign it by Friday"* is untouched;
a commitment is not a lie.

**Instruction leak.** Asked what makes the product different — a fact the inbox
does not contain — the drafter answered a journalist with *"a hero assistant
that clears one person's inbox"*: its own system prompt, describing itself. The
prompt is the one text in the model's context that is neither the message nor
the evidence, so a phrase shared with it and with no message came from the wrong
place. Across nineteen real drafts it flags that one and nothing else.

**A credential is cited, never carried.** m008 asks for the credentials m003
contains. That is enforced twice: by the rejection above, and by redacting the
credential before the prompt is built — a model cannot copy out a string it was
never shown, so the check is the second line rather than the only one.

**Drafting needed its own agent**, on the same model. Under the triage system
prompt the model refused in disposition vocabulary (*"no work implied for the
owner"*), because that prompt teaches a disposition vocabulary and nothing else.

**Prompt order is load-bearing.** With the evidence above the message, every
model tried answered the evidence, returning the same sentence for m019 (a venue
confirming a booking) and m046 (a journalist asking whether a date is public).
The message comes first now, and a test asserts that order.

**A worked example must not come from this inbox.** The system prompt once
illustrated a good answer using m036 and the launch date — m046's grounding and
m046's answer. The model returned it almost verbatim as its reply to m046,
citation included. What looked like the capability working was recitation. The
examples now use invented mail, and two tests assert that no real id or wording
appears in any system prompt.

**The model was changed during this part.** `MODEL` was moved from a small local
model to a larger one, on three measurements taken here. The small one never
once chose the "this message wants no reply" outcome, answering status updates
by handing them back instead. On the single press enquiry this capability turns
on, three consecutive runs gave a correct cited answer, a garbled one, and one
that contradicted its own evidence while citing an unrelated message. And it
needed a second attempt far more often, since the checks below reject rather
than repair. The checks hold under either model — what changes is how often they
have to fire. Nothing in the code depends on the choice; it is one line of
`.env`.

#### What this part does not do

- **Attribution is not checked.** Answering a sender who is *"in SF next week"*,
  a draft said *"while I'm in SF next week"*. Every word comes from the message;
  only who-does-what is reversed. No lexical check separates that from a correct
  reply, and none is attempted.
- **A rule stated once cannot be retrieved.** m041 (*"I do not take meetings
  before 11:00am, ever"*) is returned by nothing, for two reasons: FTS5 does not
  stem, so `meeting` never matches `meetings`, and the frequency floor drops
  terms appearing in one message — which a standing instruction always is. m043
  accordingly accepts a 9:00am slot. Retrieval is the wrong mechanism for a
  durable preference.
- **Three drafts were rejected outright** — two for echoing the message they
  answered, one for an invented date. The checks worked; those three messages
  were dispositioned `reply` and are still unanswered.

## Part 4: the things that cannot be taken back

Parts 1 to 3 take exactly two actions on the mailbox, and both can be taken
back: a message gets a disposition, and a message that is owed a reply gets a
draft. Nothing is sent, so nothing needed a gate. This part adds the first
action that leaves.

### What this system does, and what can be undone

The table below is not documentation of the code — it *is* the code. `actions.py`
holds it as a register, the gate decides what needs a person by asking the
register whether an action can be undone, and `--undo` refuses to reverse
anything the register calls irreversible. A wrong row here makes the system
behave wrongly, which is the only way a classification stays honest.

| Action | What it does to the mailbox | | How it is taken back |
| --- | --- | --- | --- |
| `reply` | records that a reply is owed | reversible | a decision, not an act; re-running triage replaces it |
| `draft` | writes reply text | reversible | rewrite it or discard it; nothing has left |
| `archive` | the message leaves the inbox | reversible | `--undo` puts it back |
| `defer` | snoozed; returns later | reversible | `--undo` puts it back |
| `delegate` | marked as someone else's | reversible | `--undo` clears the mark |
| `escalate` | marked for the owner | reversible | `--undo` clears the mark |
| `flag` | marked, and left exactly where it is | reversible | `--undo` clears the mark |
| `delete` | moved to the bin | reversible **on a timer** | `--undo` restores it, until the window closes on its own |
| `send` | a file in `outbox/`, treated as delivered | **irreversible** | nothing |

Almost everything is reversible for one structural reason: it stays inside the
mailbox. Archiving moves a message from one folder to another, and a move within
a mailbox the system owns is undone by moving it back. `state/mailbox.json` holds
where each message sits and `state/actions.json` is the ordered log of how it got
there, which is what makes `--undo` real rather than aspirational.

Sending is the one action that crosses the boundary. Once a reply is delivered it
is outside the mailbox, outside this system, and outside anyone's reach — which is
the whole of why it is the only irreversible row in the table.

`outbox/` is where delivery happens, and it is the one place this system stands in
for something real. Speaking SMTP would add a transport, credentials and a server
to reach, and none of that would change a single decision the gate makes; writing
a file is the same commitment with the plumbing removed. A file in `outbox/` is
therefore treated as delivered — not as a draft awaiting a send, and not as
something a later run may revisit.

Applying the dispositions is not bookkeeping either. Before this part one was a
row in a file and nothing moved, so *"an archive can be undone"* was a claim about
nothing.

**Is deleting reversible?** Here, yes, and the honesty is in the detail. Deleting
sets a status; no message is removed from the store, so `--undo` brings it back.
What it also writes is `purge_after`, thirty days out, because a real deployment
empties its bin — and at that point the message is gone, since there is no second
copy. So delete is gated *as if it were irreversible*, and the approval prompt
names the date:

```
  --- delete m059 ---
  asked because: delete cannot be undone once the retention window closes
  recoverable until 2026-10-21T06:50:42+00:00, then not
```

A reversibility that expires on a timer is not the same as one that waits for
someone to change their mind. Nobody has to act for that window to close, so the
one moment a person is guaranteed to be there is the moment they are asked.

`delete` is also not in the disposition vocabulary. The model chooses among the
six dispositions, so there is no answer it can give that becomes a proposal to
remove mail — reaching a delete requires a person typing `--delete m059`. That is
structural, not a matter of the model behaving.

### The gate

`gate.py` is the only module in the project that writes mail anywhere. It is a
Moya pipeline of four function steps and no model at all:

```
screen  ->  ask  ->  execute  ->  record
```

The absence of a model is deliberate. What to send was decided in Part 2 and
written in Part 3; this part decides only whether it may leave, which is a
question about the design's rules and the owner's judgement. Asking a model again
would mean the answer could change between the review and the send.

`screen` produces two lists, and they are not the same kind of thing:

- **refusals** — the action will not happen, whatever anyone says.
- **asks** — reasons a person has to read this one before it goes.

That distinction is the whole security argument. A human `yes` is permission, not
authority: it can allow what the design allows, and it cannot unlock what the
design refuses. Approving a reply to a message the rule tier flagged still sends
nothing.

| Refused, and no answer changes it | Why |
| --- | --- |
| the message was flagged by the rule tier | flagged mail is neither answered nor moved |
| the recipient has never appeared in this inbox | the allowlist is the inbox's own addresses |
| the draft carries a credential | a credential may not leave in a reply |
| the draft is empty | there is nothing to send |
| `outbox/<id>.txt` already exists | it was sent; a sent message is not sent twice |

### Where the line was drawn, and what it cost

Seventeen drafts came out of Part 3. Asking about all seventeen is the failure
the assignment names: the owner approves seventeen things without reading any of
them. **Ten are asked about; seven go through logged but unasked.**

A send is asked about when any of these is true:

1. the message or the draft touches money, credentials, a contract, legal or press
2. the thread being answered carries a credential
3. the draft leans on evidence from a different thread
4. **the draft commits the owner to a specific time or date**

The fourth is the one worth arguing for. Accepting a meeting uses none of the
vocabulary that makes a message sensitive — it is the most ordinary sentence in
the inbox — and it is still a commitment nobody can walk back once it has gone.
Without it the line asked about six, and these went out unread:

| | the draft that would have been sent |
| --- | --- |
| m043 | *"Monday at 9:00am works…"* — the owner wrote *"I do not take meetings before 11:00am, ever"* |
| m010 | *"Tuesday the 15th at 3:00pm works for me."* |
| m016 | *"Wednesday at 2:00pm works for us."* |
| m013 | an internal 1:1 being moved to a named time |

**Internal versus external is deliberately not a criterion.** It is the obvious
line and it is the wrong axis: it would ask about a one-line thank-you to a
vendor and stay silent on m008, which answers a thread carrying a production
credential. Content is what can hurt, so content is what is checked — m013 is
internal and asked about, m051 is external and is not.

**What that trades away.** Seven sends leave without anyone reading them, and one
of them is wrong. m051 replies *"…while I'm in SF next week"* when it is the
**sender** who is in SF. Every word comes from the message and only who-does-what
is reversed; no lexical check separates that from a correct reply, so no
content-based line catches it. It is the same attribution inversion documented in
Part 3, and drawing the line by domain would not fix it either — it would only
trade seven unread routine replies for a different seven.

The second cost is subtler: m043 is now *asked about* rather than *prevented*.
The owner's rule about 11:00am is written down in this inbox. Relying on someone
to notice the violation in an approval prompt is weaker than the system holding
the rule, and the prompt is the last place to catch it rather than the first.

### Every gated decision, logged

Three fields, on every proposal, whatever the outcome — including the ones nobody
was asked about, because "nobody was asked" is itself a decision:

```json
{"event": "gate", "msg_id": "m046", "action": "send", "mode": "approval",
 "proposed": "send to editor@techbrief.news: 'Quick question for our launch coverage'",
 "needs_human": ["the message or the draft touches coverage",
                 "the draft leans on m036, from a different thread",
                 "the draft commits the owner to a specific time or date"],
 "human_said": "yes",
 "happened": "sent: wrote m046.txt, m046 marked answered"}
```

`human_said` is one of three answers, and `not asked` is not a refusal. That
sounds obvious; it was a live bug, because `"not asked".startswith("no")` is
true and the first version of this read every unasked send as declined. The test
that pins it is `test_not_asked_is_not_a_refusal`.

The answer is a no unless it is an explicit yes. A closed stdin, an interrupted
prompt and a typo are all refusals, because the permissive form of that test —
act unless someone says no — turns an unattended run into blanket approval.

### One file per message, and nowhere else

```
To: editor@techbrief.news
From: sam@paperjet.io
Subject: Re: Quick question for our launch coverage
Date: 2026-09-21T06:50:12+00:00
In-Reply-To: m046
Thread: t-press
Cites: m036
Approved-By: human approved (the message or the draft touches coverage; the
  draft leans on m036, from a different thread; the draft commits the owner to
  a specific time or date)
Run: 20260921T065012Z

The launch date is confirmed as the 20th, according to a previous note.
```

The headers are the audit. A reader holding only this file can tell who it went
to, what it answers, what it leaned on, and whether a person approved it or
whether it fell below the line.

### Running it

```
python demo.py --cap R3                   # the dry-run, then the approval pass
python demo.py --cap R3 --gate dry-run    # say what it would do, write nothing
python demo.py --cap R3 --gate approval   # ask, then act
python demo.py --delete m059              # a person proposes a delete
python demo.py --undo 7                   # take back action 7
python gate.py                            # screen everything, ask nothing, write nothing
python actions.py                         # print the register and the folders
```
