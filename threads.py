"""X2. A long thread, reduced to the question nobody has answered.

Nine messages arrive on one thread over four days. Eight of them are people
reporting what they are doing; one, four messages in, asks the owner for a
decision that blocks the rest. Reading the thread top to bottom finds it in about
a minute. Reading the last message finds the wrong thing entirely, because the
last message is a status update.

That is the shape this capability is for: not "summarise the thread" but "what is
still open, and who is it on". A summary of a thread whose point is one buried
sentence is a worse artefact than the sentence.

The model writes the answer and Python checks it, the same division as everywhere
else here:

  - the thread is quoted as untrusted data, oldest first
  - the answer must cite the message the open question came from
  - a citation is checked against the thread, not merely against the mail store --
    citing a real message from a different conversation is still wrong here
  - the model is told the thread may have nothing open, because plenty do not

Usage:
    python threads.py t-launch     # the open question, and where it came from
    python threads.py              # every thread worth summarising
"""

import json
import re

import agents
import config
import mailstore
import rules
import trace

# Below this a thread is not worth a model call: the "summary" would be the
# message itself, and the reader is better served by reading it.
WORTH_SUMMARISING = 3


class NotUsable(ValueError):
    """The answer cannot be trusted. The message says why."""


def worth_it(mailbox, minimum=WORTH_SUMMARISING):
    """Threads long enough that reading them is a chore. Derived, never named."""
    found = []
    for thread_id in {m.thread_id for m in mailbox.messages}:
        messages = [m for m in mailbox.thread(thread_id) if not rules.classify(m).hostile]
        if len(messages) >= minimum:
            found.append((thread_id, sorted(messages, key=lambda m: m.sent_at)))
    return sorted(found, key=lambda pair: -len(pair[1]))


def build_prompt(thread_id, messages, owner=None):
    """The thread, oldest first, and the question being asked about it."""
    owner = owner or config.OWNER
    quoted = "\n\n".join(agents.quote_untrusted(m) for m in messages)
    return (
        f"You are reading one email thread belonging to {owner}, oldest message first.\n"
        f"There are {len(messages)} messages in it.\n\n"
        f"{quoted}\n\n"
        "Find the single most important question or request that is still OPEN -- something\n"
        "somebody asked for that has not been answered or done anywhere later in the thread.\n"
        "It is usually not in the last message. Status updates are not open questions.\n\n"
        "Answer with one JSON object and nothing else:\n"
        '{"open_question": "<the ask, in your own words, one sentence>",\n'
        ' "asked_by": "<the email address that asked>",\n'
        ' "owed_by": "<the email address who owes the answer>",\n'
        ' "cites": ["<the id of the message it came from>"],\n'
        ' "blocking": "<what it holds up, or an empty string>"}\n\n'
        'If nothing in the thread is still open, answer {"open_question": null, "cites": [], '
        '"reason": "<one sentence>"}.\n'
        "Cite only ids from the thread above. Do not invent a message id."
    )


def parse(raw, messages):
    """Shape and citations. Raises `NotUsable` rather than returning something wrong."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.partition("\n")[2] if "\n" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise NotUsable("the model did not answer with a JSON object")
    try:
        data = json.loads(text[start : end + 1])
    except ValueError as error:
        raise NotUsable(f"the answer was not valid JSON ({error})") from error

    known = {m.id for m in messages}
    cites = [str(c).strip().lower() for c in (data.get("cites") or [])]
    stray = [c for c in cites if c not in known]
    if stray:
        # Checked against the thread, not the mail store. A real id from another
        # conversation is a citation that does not support what it is attached to.
        raise NotUsable(f"cited {', '.join(stray)}, which is not in this thread")

    if not data.get("open_question"):
        return {"open": False, "reason": data.get("reason") or "nothing is open", "cites": tuple(cites)}

    if not cites:
        raise NotUsable("an open question was reported without citing the message it came from")

    # The wording must come from the thread, not from the instructions. Same
    # reasoning as the drafting leak check: the prompt is the one text in context
    # that is not mail.
    question = str(data["open_question"]).strip()
    if not question:
        raise NotUsable("the open question is empty")
    if re.search(r"\bm\d{3}\b", question, re.I):
        raise NotUsable("the open question prints an internal id, which means nothing to a reader")

    return {
        "open": True,
        "question": question,
        "asked_by": str(data.get("asked_by") or "").strip(),
        "owed_by": str(data.get("owed_by") or "").strip(),
        "blocking": str(data.get("blocking") or "").strip(),
        "cites": tuple(cites),
    }


def summarise(agent, thread_id, messages, retries=1):
    """Ask once, check, and ask again with the reason on failure. Never raises."""
    prompt = build_prompt(thread_id, messages)
    trace.event("thread_prompt", thread_id=thread_id, messages=len(messages), chars=len(prompt))
    problem = ""
    for attempt in range(1, retries + 2):
        try:
            raw = agent.handle_message(prompt, thread_id=f"thread-{thread_id}")
        except Exception as error:  # noqa: BLE001 - provider already retried
            trace.event("thread_error", thread_id=thread_id, error=str(error)[:300])
            return {"open": False, "reason": f"the model could not be reached: {error}", "failed": True}
        try:
            answer = parse(raw, messages)
        except NotUsable as error:
            problem = str(error)
            trace.event("thread_rejected", thread_id=thread_id, attempt=attempt, problem=problem)
            if attempt > retries:
                break
            prompt = f"{prompt}\n\nYour previous answer was rejected: {problem}\nAnswer again, correctly."
            continue
        answer["thread_id"] = thread_id
        trace.event(
            "thread_summary",
            thread_id=thread_id,
            open=answer["open"],
            cites=list(answer["cites"]),
            attempts=attempt,
        )
        return answer
    trace.event("thread_summary", thread_id=thread_id, open=False, failed=True, problem=problem)
    return {"open": False, "reason": f"no usable answer: {problem}", "failed": True, "thread_id": thread_id}


def render(answer, messages):
    out = [f"=== {answer.get('thread_id')} -- {len(messages)} messages, {messages[0].subject} ==="]
    out.append(f"  {messages[0].sent_at:%d %b} to {messages[-1].sent_at:%d %b}, "
               f"{len({m.sender for m in messages})} people")
    if not answer.get("open"):
        out.append(f"\n  nothing open -- {answer.get('reason')}")
        return "\n".join(out)
    out.append(f"\n  OPEN: {answer['question']}")
    if answer.get("asked_by"):
        out.append(f"  asked by: {answer['asked_by']}")
    if answer.get("owed_by"):
        out.append(f"  owed by:  {answer['owed_by']}")
    if answer.get("blocking"):
        out.append(f"  blocks:   {answer['blocking']}")
    out.append(f"  from:     {', '.join(answer['cites'])}")
    for cited in answer["cites"]:
        message = next((m for m in messages if m.id == cited), None)
        if message is not None:
            out.append(f"      {cited}: {message.body[:150].replace(chr(10), ' ')}")
    # Where in the thread it sat. The point of the capability is that this is
    # rarely the last message, so the position is worth printing.
    positions = [i + 1 for i, m in enumerate(messages) if m.id in answer["cites"]]
    if positions:
        out.append(f"  position: message {positions[0]} of {len(messages)}")
    return "\n".join(out)


if __name__ == "__main__":
    import sys

    box = mailstore.load()
    trace.start_run(cap="X2", fresh=False)
    agent = agents.thread_agent()
    wanted = [t.lower() for t in sys.argv[1:]]
    candidates = worth_it(box)
    if wanted:
        candidates = [(t, m) for t, m in candidates if t.lower() in wanted]
        if not candidates:
            raise SystemExit(f"no thread with {WORTH_SUMMARISING}+ messages matching {', '.join(wanted)}")
    for thread_id, messages in candidates:
        print(render(summarise(agent, thread_id, messages), messages))
        print()
