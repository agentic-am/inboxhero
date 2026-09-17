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
