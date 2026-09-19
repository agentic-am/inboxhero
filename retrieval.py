"""Evidence for one message, gathered from the inbox and nothing else.

Part 3 asks the system to ground a decision in what the inbox actually says.
This module answers one question -- "which earlier messages ground *this*
message?" -- in four tiers, most precise first:

    tier 0  scope        the candidate set: earlier than this message, not
                         itself, and never a message the rules called hostile
    tier 1  thread walk  earlier messages in the same thread (exact, free)
    tier 2  keyword      SQLite FTS5 over subject and body, BM25 ranked
    tier 3  entity       exact lookup of references, addresses and domains

Two design choices are worth stating because they are easy to get wrong.

*The query comes from the message, not from a question.* We are not answering
"who is on vacation"; we are asking what grounds m019. Query and target
therefore usually share vocabulary, which is why keyword matching carries this
workload and embeddings stay off. The synonym map below closes the remaining
gap deterministically, so the query reaching the index is reproducible and
testable rather than whatever the model happened to invent.

*Indexed is not the same permission as retrievable.* Hostile messages stay in
the index -- m021's fraudulent "INV-4" has to be findable, or nothing can
connect it to the real invoice it imitates -- but they are never returned as
evidence, because evidence is quoted into a prompt.

Usage:
    python retrieval.py            # index the real inbox and show evidence
"""

import re
import sqlite3
from dataclasses import dataclass
from datetime import timedelta

import config
import rules

# How many messages may ground one decision, and how much of each is quoted.
DEFAULT_K = 5
SNIPPET_CHARS = 240

# Query construction. A term in more than this share of the corpus carries no
# signal ("paperjet" is in most of the inbox), and it is not merely useless:
# measured at 200k messages, adding one unselective term took an eight-term
# query from 55ms to 409ms, because FTS5 cost tracks document frequency rather
# than corpus size. A term in only one message is just as useless in the other
# direction -- it can never match anything -- so the filter is a band, not a
# ceiling.
MAX_QUERY_TERMS = 8
DF_MAX_RATIO = 0.2
DF_MIN = 2
MIN_TERM_CHARS = 3

# Not every word in a message is worth asking the index about. Picking the
# rarest words picks the incidental ones: m046's rarest word is "coverage",
# which appears nowhere else, while the word that actually grounds it is
# "launch". So terms are scored by why they look topical, and a term needs at
# least two independent signals (or to be a reference) before it is queried.
# This is the whole difference between retrieving m026/m036 and retrieving
# whatever happened to share an unusual word.
WEIGHT_REFERENCE = 5  # INV-4471, #55120: rare by construction
WEIGHT_SUBJECT = 3  # subjects are written to be topical
WEIGHT_VOCABULARY = 3  # a word the synonym map knows is domain vocabulary
WEIGHT_PROPER = 2  # capitalised mid-sentence: a name, a day, a product
MIN_TERM_WEIGHT = 3  # one signal is enough to ask about a word...
STRONG_TERM_WEIGHT = 4  # ...but see `worth_querying` for when one is enough to search on

# Source ranking. A message in the same thread is better evidence than a message
# that merely shares a word, whatever BM25 thinks of the word.
SOURCE_SCORE = {"thread": 1.0, "entity": 0.8, "keyword": 0.7}

# A keyword hit far below the best one in the same result set is not weak
# evidence; it is a different subject that happened to share a word. Asking for
# m046 (a press enquiry about the launch) returns m036 at 0.7 and m015 -- a
# standing request about legal correspondence -- at 0.0574, on the single word
# "deadline". Twelve times weaker and about nothing related.
#
# It was left in at first as harmless noise costing only tokens. It is not
# harmless. Handed that evidence, the model produced "We are not confirming a
# public launch date yet", cited m015, and contradicted m036 while doing it: an
# unrelated message in the evidence is something plausible to point at when the
# answer goes wrong. The floor is relative to the best hit rather than absolute
# because BM25 scores are only meaningful within one result set.
KEYWORD_SCORE_FLOOR = 0.15

STOPWORDS = frozenset(
    """
a an and are as at be been before but by can could did do does for from get got had has have
he her his how if in is it its just me my no not of on or our out re she should so than that
the their them then there these they this to too us was we were what when where which who why
will with would you your about all any more most new now one only other some such time
please thanks thank hi hello dear regards best sent message email mail week day today tomorrow
need needs want would like know let make sure back also still yet here going able
thing things stuff chance sort kind way bit lot everything else anything something nothing
got get give gave take took come came see saw look looks going done doing ever never really
quick short long good great nice sec bio ask asked tell told say said talk talked think
""".split()
)

# Synonym expansion, deliberately small and inbox-specific. Each entry is a set
# of words that mean the same thing *in this inbox*, so hitting any one of them
# pulls in the others. This is the entire "semantic" layer, written down where a
# test can read it instead of hidden inside a vector.
SYNONYMS = (
    {"payout", "payment", "paid", "invoice", "remittance", "billing", "charge"},
    {"pto", "vacation", "leave", "holiday", "away", "ooo"},
    {"venue", "space", "booking", "room", "location", "holding"},
    {"sign", "signature", "signed", "safe", "amendment", "countersign"},
    {"meeting", "call", "slot", "intro", "demo", "sync"},
    {"reschedule", "move", "shift", "postpone"},
    {"launch", "release", "ship", "announcement", "press"},
    {"deck", "slides", "presentation", "board"},
    {"candidate", "role", "hiring", "interview", "offer"},
    {"staging", "deploy", "outage", "incident", "broken"},
    {"credential", "password", "token", "secret", "creds"},
    {"legal", "contract", "minutes", "counsel", "agreement"},
    {"deadline", "due", "target", "date"},
    {"coffee", "drinks", "lunch"},
    {"dentist", "dental", "appointment", "cleaning"},
)

_SYNONYM_INDEX = {}
for _group in SYNONYMS:
    for _word in _group:
        _SYNONYM_INDEX.setdefault(_word, set()).update(_group - {_word})

# Reference-shaped tokens are the highest-value query terms in an inbox because
# they are rare by construction: INV-4, #55120, PJ-221, CD-9931.
REFERENCE = re.compile(r"\b(?:[A-Z]{2,}-\d+|#\d{3,})\b")
DOMAIN = re.compile(r"\b(?:[a-z0-9-]+\.)+(?:io|com|org|co|net|dev|ai)\b", re.I)
ADDRESS = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b")
WORD = re.compile(r"[A-Za-z][A-Za-z0-9']+")


@dataclass(frozen=True)
class Scope:
    """Tier 0: which messages may be considered at all.

    `before` is the hard one. Only messages that had already arrived may ground
    a decision, or a replay of the run sees different evidence than the live run
    did and the trace stops being an audit record.

    `window_days` is off by default and deliberately so: this inbox spans eight
    days, so any window wide enough to be safe is a no-op and any narrower one
    would silently drop m026 as grounding for m019. It exists for the 11k/day
    case, and it applies to the keyword tier only -- threads outlive any window,
    and an exact reference is worth finding however old it is.
    """

    before: object
    exclude_ids: frozenset = frozenset()
    window_days: int | None = None

    def earliest(self):
        """Oldest timestamp the keyword tier may reach, or None for unbounded."""
        if self.window_days is None:
            return None
        return self.before - timedelta(days=self.window_days)

    def allows(self, message):
        return message.sent_at < self.before and message.id not in self.exclude_ids


@dataclass(frozen=True)
class Evidence:
    """One message offered as grounding, with the reason it was offered.

    `terms` is the point of the whole module: it turns "m019 is relevant" into
    "m019 matched 'holding'", which is a sentence that can go in the trace, be
    read back by explain-why, and be asserted in a test. A similarity score can
    do none of those things.
    """

    message_id: str
    source: str  # thread | keyword | entity
    terms: tuple = ()
    snippet: str = ""
    score: float = 0.0
    thread_id: str = ""
    sender: str = ""
    timestamp: str = ""
    subject: str = ""

    def line(self):
        why = f"matched {', '.join(self.terms)}" if self.terms else f"same thread {self.thread_id}"
        return f"{self.message_id} ({self.source}, {why})"


def snippet_of(message):
    text = " ".join(f"{message.subject} {message.body}".split())
    return text[:SNIPPET_CHARS] + ("..." if len(text) > SNIPPET_CHARS else "")


# --- the FTS5 boundary ----------------------------------------------------


def fts_query(terms):
    """Turn terms into an FTS5 MATCH expression, safely.

    Every term is wrapped in double quotes, because FTS5's query language is a
    language: a bare `1:1` reads as a column filter and raises "no such column:
    1" (it is live in m013, m016 and m119), and message bodies are untrusted
    input that must never become query syntax. Quoting makes each term a literal
    phrase, which is the only thing this module ever wants.
    """
    quoted = []
    for term in terms:
        cleaned = str(term).strip().replace('"', '""')
        if cleaned:
            quoted.append('"' + cleaned + '"')
    return " OR ".join(quoted)


def extract_terms(message):
    """Candidate query terms for this message, as {term: weight}.

    The weight records *why* a word looks worth asking about, and the reasons
    stack: "launch" in m046 is both a subject word and known vocabulary, which
    is what lifts it past "coverage", a word that appears nowhere else in the
    inbox and can therefore ground nothing.
    """
    subject_words = {w.lower() for w in WORD.findall(message.subject)}
    text = f"{message.subject}\n{message.body}"

    # Capitalised away from the start of a line is a decent proxy for a proper
    # noun without a part-of-speech tagger: PaperJet, Priya, Thursday.
    proper = set()
    for line in text.splitlines():
        for word in WORD.findall(line)[1:]:
            if word[0].isupper():
                proper.add(word.lower())

    weights = {}

    def add(value, weight):
        value = value.strip().lower()
        if not value or value in STOPWORDS or len(value) < MIN_TERM_CHARS:
            return
        weights[value] = weights.get(value, 0) + weight

    for pattern in (REFERENCE, ADDRESS, DOMAIN):
        for found in pattern.findall(text):
            add(found, WEIGHT_REFERENCE)
    for word in WORD.findall(text):
        lowered = word.lower()
        if lowered in weights:
            continue
        weight = 0
        if lowered in subject_words:
            weight += WEIGHT_SUBJECT
        if lowered in _SYNONYM_INDEX:
            weight += WEIGHT_VOCABULARY
        if lowered in proper:
            weight += WEIGHT_PROPER
        if weight:
            add(word, weight)
    return weights


def worth_querying(terms, weights):
    """Whether this term set is enough to search on at all.

    A single weak term names a topic, not a fact: m012's only surviving term is
    "call", and searching the whole inbox for "call" returns messages that are
    about calls rather than messages about the thing m012 is asking for. So the
    keyword tier fires on one *strong* term (m040's "board", which is both the
    subject and known vocabulary) or on two weak ones (m019's "launch" and
    "date", which together pin the launch date down), and on nothing less.
    """
    if any(weights.get(term, 0) >= STRONG_TERM_WEIGHT for term in terms):
        return True
    return len(terms) >= 2


def expand(weights):
    """Synonyms for the terms that survived. The deterministic semantic layer.

    Expansions inherit a vocabulary-level weight rather than their parent's:
    they are real domain words, but the message never actually used them, so
    they should not outrank the words it did use.
    """
    extra = {}
    for term in weights:
        for synonym in sorted(_SYNONYM_INDEX.get(term, ())):
            if synonym not in weights:
                extra[synonym] = WEIGHT_VOCABULARY
    return extra


def entities_of(message):
    """Typed, exactly-matchable things in a message: references, addresses, domains.

    Kept apart from the keyword tier because BM25 ranks by how unusual a word is
    across the corpus, which can put a precise identifier below a topical
    near-miss. An exact index cannot make that mistake.

    Measured on this inbox, this tier contributes **no evidence at all**, and
    the honest reason is the data rather than the design: of twelve extracted
    values only two appear in more than one message, and both of those pairs are
    automated mail the rule tier settles without a model. The reference-shaped
    tokens that would justify the tier -- INV-4471, CD-9931, PJ-221 -- each
    occur exactly once, and most of them occur inside phishing that is excluded
    from evidence anyway. It is kept because references recur constantly in a
    real mailbox of reply chains and ticket numbers, and because it is the one
    tier whose cost does not grow with the corpus, but nothing here depends on
    it and a reader should not be told otherwise.
    """
    text = f"{message.subject}\n{message.body}"
    found = set()
    for pattern in (REFERENCE, ADDRESS, DOMAIN):
        for value in pattern.findall(text):
            found.add(value.strip().lower())
    return found


def _iso(value):
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _normalise(rank, best, matched, most_matched):
    """Score a keyword hit: BM25 strength, weighted by how much of the query it met.

    FTS5's `rank` is negative and *better when smaller*, so the strongest match
    has the largest magnitude, and dividing by the best magnitude in the result
    set keeps BM25's own ordering while leaving keyword evidence below thread
    and entity evidence, which are exact rather than ranked.

    The coverage factor is what separates evidence from coincidence. Asking for
    m019 pulls both m036, which matches "date", "launch" and "press", and m096
    ("free up space"), which matches "space" and nothing else. BM25 alone rates
    a short document with one lucky hit generously; a query that was answered
    three ways is better grounding than one that was answered once.
    """
    strength = abs(float(rank or 0.0)) / best
    coverage = (len(matched) / most_matched) if most_matched else 1.0
    return round(SOURCE_SCORE["keyword"] * strength * coverage, 4)


def hostile_ids(mailbox):
    """Messages the rule tier refuses. Deterministic, and no model call."""
    found = set()
    for message in mailbox.messages:
        try:
            if rules.classify(message).hostile:
                found.add(message.id)
        except Exception:  # noqa: BLE001 - a rule bug must not disable retrieval
            continue
    return found


class Index:
    """The seam between the tiers and wherever the messages actually live.

    Today the inbox is 100 messages and `Mailbox` holds them all in memory, so
    this builds a throwaway in-memory SQLite database per run. At 11k/day that
    stops being possible -- 4M messages is roughly 6GB of text before Python
    object overhead -- and this class is the only thing that then has to change:
    a file-backed database with `CREATE INDEX ON msg(thread_id, sent_at)` keeps
    every tier below exactly as written. Measured at 200k messages, a thread
    walk is 0.28ms with that index and 381ms without one.
    """

    def __init__(self, mailbox, hostile=None):
        self.mailbox = mailbox
        self.hostile_ids = frozenset(hostile if hostile is not None else hostile_ids(mailbox))
        self.db = sqlite3.connect(":memory:")
        self.db.execute("create virtual table fts using fts5(mid unindexed, sent_at unindexed, subject, body)")
        self.db.executemany(
            "insert into fts values (?,?,?,?)",
            [(m.id, m.timestamp, m.subject, m.body) for m in mailbox.messages],
        )
        self.total = len(mailbox.messages)
        self._df_cache = {}
        self.entities = {}
        for message in mailbox.messages:
            for value in entities_of(message):
                self.entities.setdefault(value, []).append(message.id)

    def document_frequency(self, term):
        """How many messages contain this term. Cached: asked once per term per run."""
        if term not in self._df_cache:
            try:
                row = self.db.execute("select count(*) from fts where fts match ?", (fts_query([term]),)).fetchone()
                self._df_cache[term] = row[0] if row else 0
            except sqlite3.OperationalError:
                self._df_cache[term] = 0  # a term the tokenizer rejects is simply dropped
        return self._df_cache[term]

    def selective(self, weights, limit=MAX_QUERY_TERMS, min_weight=MIN_TERM_WEIGHT):
        """Keep the terms worth querying: topical enough, and in the frequency band.

        Sorted by weight first and rarity second, so a term that looks topical
        beats a term that is merely unusual. A message with nothing topical to
        say -- m012, "that thing we talked about" -- yields nothing here, and
        nothing is the correct query for it.
        """
        ceiling = max(DF_MIN, int(self.total * DF_MAX_RATIO))
        scored = []
        for term, weight in weights.items():
            if weight < min_weight:
                continue
            frequency = self.document_frequency(term)
            if DF_MIN <= frequency <= ceiling:
                scored.append((-weight, frequency, term))
        scored.sort()
        return [term for _, _, term in scored[:limit]]

    def search(self, terms, scope, limit):
        """BM25 over the scoped set. Returns (message_id, score, matched_terms)."""
        if not terms:
            return []
        sql = ["select mid, rank from fts where fts match ? and sent_at < ?"]
        params = [fts_query(terms), _iso(scope.before)]
        earliest = scope.earliest()
        if earliest is not None:
            sql.append("and sent_at >= ?")
            params.append(_iso(earliest))
        if scope.exclude_ids:
            sql.append("and mid not in (" + ",".join("?" * len(scope.exclude_ids)) + ")")
            params.extend(sorted(scope.exclude_ids))
        sql.append("order by rank limit ?")
        params.append(limit)
        try:
            rows = self.db.execute(" ".join(sql), params).fetchall()
        except sqlite3.OperationalError:
            return []
        best = max((abs(float(rank or 0.0)) for _, rank in rows), default=0.0) or 1.0
        found = []
        for mid, rank in rows:
            message = self.mailbox.by_id(mid)
            if message is None:
                continue
            haystack = f"{message.subject}\n{message.body}".lower()
            found.append((mid, rank, tuple(term for term in terms if term in haystack)))
        most_matched = max((len(matched) for _, _, matched in found), default=0)
        return [(mid, _normalise(rank, best, matched, most_matched), matched) for mid, rank, matched in found]


# --- the tiers -------------------------------------------------------------


def tier_thread(mailbox, message, limit):
    """Tier 1. Earlier messages in the same thread, oldest first.

    When a thread is longer than the budget, keep the opener and the most recent
    rest rather than simply the last few: the launch thread's m026 is what sets
    "target is the 20th", and every later message leans on it.
    """
    earlier = mailbox.earlier_in_thread(message)
    if limit > 0 and len(earlier) > limit:
        earlier = [earlier[0]] + earlier[-(limit - 1) :] if limit > 1 else [earlier[0]]
    return earlier


def tier_entity(index, message, scope, limit):
    """Tier 3. Exact lookups. The cheapest tier at any corpus size."""
    hits = {}
    for value in entities_of(message):
        for mid in index.entities.get(value, ()):
            candidate = index.mailbox.by_id(mid)
            if candidate is None or not scope.allows(candidate):
                continue
            hits.setdefault(mid, set()).add(value)
    ordered = sorted(hits.items(), key=lambda item: (-len(item[1]), item[0]))
    return [(mid, tuple(sorted(terms))) for mid, terms in ordered[:limit]]


@dataclass(frozen=True)
class Retrieved:
    """What one retrieval produced, including how it got there.

    The query terms travel with the evidence because the trace records both:
    which terms were asked for is as much a part of the audit as which messages
    came back, and a retrieval that found nothing still has to explain itself.
    """

    evidence: tuple = ()
    terms: tuple = ()
    window_days: int | None = None
    widened: bool = False

    @property
    def ids(self):
        return [item.message_id for item in self.evidence]

    def __bool__(self):
        return bool(self.evidence)


def retrieve(mailbox, message, k=DEFAULT_K, index=None, window_days=None, hostile=None):
    """Evidence for one message: all four tiers, merged, deduped and budgeted.

    Returns a `Retrieved`, best evidence first, never longer than `k`. Empty is
    a real answer rather than a failure: m012 ("that thing we talked about") has
    nothing to extract and nothing to match, and the honest response is to ask
    rather than to guess.
    """
    if index is None:
        index = Index(mailbox, hostile=hostile)

    scope = Scope(
        before=message.sent_at,
        exclude_ids=frozenset({message.id}) | index.hostile_ids,
        window_days=window_days,
    )

    merged = {}

    def offer(mid, source, terms, score):
        candidate = mailbox.by_id(mid)
        if candidate is None or not scope.allows(candidate):
            return
        existing = merged.get(mid)
        if existing is not None:
            # A message found twice keeps its best source but collects every
            # reason it was found; the trace should show all of them.
            if existing.score >= score:
                merged[mid] = Evidence(
                    message_id=mid,
                    source=existing.source,
                    terms=tuple(dict.fromkeys(existing.terms + tuple(terms))),
                    snippet=existing.snippet,
                    score=existing.score,
                    thread_id=existing.thread_id,
                    sender=existing.sender,
                    timestamp=existing.timestamp,
                    subject=existing.subject,
                )
                return
        merged[mid] = Evidence(
            message_id=mid,
            source=source,
            terms=tuple(dict.fromkeys(terms)),
            snippet=snippet_of(candidate),
            score=score,
            thread_id=candidate.thread_id,
            sender=candidate.sender,
            timestamp=candidate.timestamp,
            subject=candidate.subject,
        )

    # Tier 1: the thread. No window applies -- threads outlive any window.
    for earlier in tier_thread(mailbox, message, k):
        offer(earlier.id, "thread", (), SOURCE_SCORE["thread"])

    # Tier 3: exact entities. No window here either -- INV-4 is INV-4 whenever
    # it arrived, and that is the entire point of an exact index.
    for mid, terms in tier_entity(index, message, scope, k):
        offer(mid, "entity", terms, SOURCE_SCORE["entity"])

    # Tier 2: keyword. The only tier whose cost grows with the corpus, so the
    # only tier the window applies to.
    weights = extract_terms(message)
    terms = index.selective(weights)
    if not worth_querying(terms, weights):
        terms = []
    else:
        expanded = index.selective(expand({term: weights[term] for term in terms}))
        terms = terms + [term for term in expanded if term not in terms]
    hits = index.search(terms, scope, k * 3)

    widened = False
    if not hits and window_days is not None:
        # A window that silently loses recall is worse than no window, so one
        # widening is allowed and it is recorded rather than hidden.
        widened = True
        hits = index.search(terms, Scope(scope.before, scope.exclude_ids, None), k * 3)

    # Only the keyword tier is filtered. A thread hit and an entity hit are
    # exact -- they are not scored against each other and there is no "far
    # below" for them to be.
    if hits:
        floor = max(score for _, score, _ in hits) * KEYWORD_SCORE_FLOOR
        hits = [hit for hit in hits if hit[1] >= floor]

    for mid, score, matched in hits:
        offer(mid, "keyword", matched, score)

    ranked = sorted(merged.values(), key=lambda item: (-item.score, item.timestamp, item.message_id))
    return Retrieved(
        evidence=tuple(ranked[:k]),
        terms=tuple(terms),
        window_days=window_days,
        widened=widened,
    )


def describe(mailbox, message_ids=None, k=DEFAULT_K):
    index = Index(mailbox)
    for mid in message_ids or [m.id for m in mailbox.messages]:
        message = mailbox.by_id(mid)
        if message is None:
            continue
        found = retrieve(mailbox, message, k=k, index=index)
        print(f"\n{message.summary()}")
        print(f"  query terms: {list(found.terms) or '(none extractable)'}")
        if not found:
            print("  evidence: none -- ask, do not guess")
        for item in found.evidence:
            print(f"  - {item.line()}")


if __name__ == "__main__":
    import mailstore

    box = mailstore.load()
    index = Index(box)
    print(f"=== retrieval over {config.INBOX_PATH} ===")
    print(f"  indexed {index.total} messages, {len(index.entities)} distinct entities")
    print(f"  never returned as evidence (hostile): {sorted(index.hostile_ids)}")
    # The messages retrieval actually runs on in a real run: the ones the rule
    # tier did not settle. Derived rather than listed, so this prints the honest
    # set -- including the ones that come back with nothing -- instead of a
    # hand-picked selection that only shows retrieval working.
    unsettled = [m.id for m in box.messages if not rules.classify(m).handled]
    print(f"  describing the {len(unsettled)} messages the rules did not settle")
    describe(box, unsettled)
