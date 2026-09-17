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
