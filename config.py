"""Configuration, read from the environment. Nothing about the model is hardcoded.

Values come from the real environment first, then from a `.env` file beside
this module, then from the defaults below. A real environment variable always
wins, so `MODEL=<another-tag> python demo.py --cap R1` overrides the file
without editing it. `.env` is gitignored; `.env.example` documents the keys.

Every other module reads settings from here and never touches `os.environ`
itself, so there is exactly one place to look when a run points at the wrong
model or the wrong directory.

Usage:
    python config.py     # print the resolved settings and any problems
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"

PROVIDERS = ("ollama", "gemini")

_DEFAULTS = {
    # Which model answers. PROVIDER is the primary; FALLBACK_PROVIDER is tried
    # only when the primary cannot be reached at all (not on a bad answer).
    "PROVIDER": "ollama",
    "MODEL": "gemma4:e4b",
    "OLLAMA_HOST": "http://localhost:11434",
    "FALLBACK_PROVIDER": "gemini",
    "GEMINI_API_KEY": "",
    "GEMINI_MODEL": "gemini-3.5-flash-lite",
    # Rate limits. Empty spacing means "0 on Ollama, 4.5 seconds on Gemini",
    # which is about thirteen requests a minute against a free-tier limit of 15.
    "CALL_SPACING_SECONDS": "",
    "MAX_RETRIES": "3",
    "TEMPERATURE": "0.1",
    "REQUEST_TIMEOUT_S": "300",
    # How many messages may share one model call. 1 keeps every message in its
    # own context, which is the safe default: text from one sender can then
    # never influence the decision about another. Raising it trades that
    # isolation for speed and tokens. Messages the rules marked sensitive are
    # never batched, whatever this is set to.
    "BATCH_SIZE": "1",
    # Retrieval (Part 3). K is how many messages may ground one decision.
    # WINDOW_DAYS limits the keyword tier only, and is off here because this
    # inbox spans eight days: any window wide enough to be safe is a no-op, and
    # a narrower one would silently drop m026 as grounding for m019. It exists
    # for a real mailbox. EMBEDDINGS is the seam, not a feature: the tiers are
    # lexical, and turning this on would add a vector tier over the same scoped
    # candidates rather than replace anything.
    "RETRIEVAL_K": "5",
    "RETRIEVAL_WINDOW_DAYS": "",
    "EMBEDDINGS": "off",
    # The gate (Part 4). `dry-run` prints what it would do and writes nothing,
    # `approval` asks before each action that crosses the escalation line, and
    # `both` does the dry-run first and then the approval pass. `both` is the
    # default because the two answer different questions: the dry-run says what
    # the system wants to do while nothing is at stake, and the approval pass
    # asks about the few that matter while the answer still changes something.
    "GATE_MODE": "both",
    "OUTBOX_DIR": "outbox",
    # How long a deleted message stays recoverable. Nothing in this build purges
    # the bin; the value is what the approval prompt quotes when it warns the
    # owner when the message would stop being recoverable.
    "BIN_RETENTION_DAYS": "30",
    # The inbox and where the run leaves its footprints.
    "OWNER": "sam@paperjet.io",
    "INBOX_FILE": "data/inbox.json",
    "STATE_DIR": "state",
    "TRACE_FILE": "trace.jsonl",
    "DASHBOARD_FILE": "dashboard.html",
}

GATE_MODES = ("dry-run", "approval", "both")


class ConfigError(ValueError):
    """The configuration cannot work. The message says which key and why."""


def _load_env_file(path=None):
    """Read KEY=value lines into the environment without overwriting it.

    Hand-rolled rather than python-dotenv: the format is a dozen lines of
    KEY=value, and one less dependency is one less thing to install.

    `path` defaults to None rather than to ENV_FILE, and is resolved on every
    call. A default argument is bound once when the function is defined, so
    `path=ENV_FILE` would have pinned the original file for the life of the
    process and quietly ignored any later reassignment of `config.ENV_FILE`,
    which is exactly how the tests point this at a temporary file.
    """
    path = Path(path) if path is not None else ENV_FILE
    if not path.exists():
        return {}
    loaded = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.split(" #", 1)[0].strip().strip("\"'")
        loaded[key] = value
        os.environ.setdefault(key, value)  # the real environment wins
    return loaded


def _setting(key):
    return os.environ.get(key, _DEFAULTS[key])


def _path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def reload():
    """Resolve every setting into module globals. Called at import and by tests."""
    global LOADED_FROM_FILE, PROVIDER, MODEL, OLLAMA_HOST, FALLBACK_PROVIDER
    global GEMINI_API_KEY, GEMINI_MODEL, CALL_SPACING_SECONDS, MAX_RETRIES
    global TEMPERATURE, REQUEST_TIMEOUT_S, OWNER, BATCH_SIZE
    global RETRIEVAL_K, RETRIEVAL_WINDOW_DAYS, EMBEDDINGS
    global GATE_MODE, BIN_RETENTION_DAYS
    global INBOX_PATH, STATE_PATH, TRACE_PATH, OUTBOX_PATH, DASHBOARD_PATH

    LOADED_FROM_FILE = _load_env_file()

    PROVIDER = _setting("PROVIDER").strip().lower()
    MODEL = _setting("MODEL").strip()
    OLLAMA_HOST = _setting("OLLAMA_HOST").strip().rstrip("/")
    FALLBACK_PROVIDER = _setting("FALLBACK_PROVIDER").strip().lower()
    GEMINI_API_KEY = _setting("GEMINI_API_KEY").strip()
    GEMINI_MODEL = _setting("GEMINI_MODEL").strip()

    spacing = _setting("CALL_SPACING_SECONDS").strip()
    CALL_SPACING_SECONDS = float(spacing) if spacing else None  # None = per-provider default
    MAX_RETRIES = int(_setting("MAX_RETRIES"))
    TEMPERATURE = float(_setting("TEMPERATURE"))
    REQUEST_TIMEOUT_S = float(_setting("REQUEST_TIMEOUT_S"))
    BATCH_SIZE = int(_setting("BATCH_SIZE"))

    RETRIEVAL_K = int(_setting("RETRIEVAL_K"))
    window = _setting("RETRIEVAL_WINDOW_DAYS").strip()
    RETRIEVAL_WINDOW_DAYS = int(window) if window else None  # None = unbounded
    EMBEDDINGS = _setting("EMBEDDINGS").strip().lower()

    GATE_MODE = _setting("GATE_MODE").strip().lower()
    BIN_RETENTION_DAYS = int(_setting("BIN_RETENTION_DAYS"))

    OWNER = _setting("OWNER").strip().lower()

    INBOX_PATH = _path(_setting("INBOX_FILE"))
    STATE_PATH = _path(_setting("STATE_DIR"))
    TRACE_PATH = _path(_setting("TRACE_FILE"))
    OUTBOX_PATH = _path(_setting("OUTBOX_DIR"))
    DASHBOARD_PATH = _path(_setting("DASHBOARD_FILE"))


reload()


def problems():
    """Every reason the configuration cannot work, as plain sentences.

    Empty list means fine. Warnings (things that limit a run without
    breaking it) come back separately from `check()`.
    """
    found = []
    if PROVIDER not in PROVIDERS:
        found.append(f"PROVIDER={PROVIDER!r} is not one of {', '.join(PROVIDERS)}.")
    if FALLBACK_PROVIDER and FALLBACK_PROVIDER not in PROVIDERS:
        found.append(
            f"FALLBACK_PROVIDER={FALLBACK_PROVIDER!r} is not one of {', '.join(PROVIDERS)}; leave it empty to disable."
        )
    if FALLBACK_PROVIDER and FALLBACK_PROVIDER == PROVIDER:
        found.append("FALLBACK_PROVIDER is the same as PROVIDER; leave it empty to disable.")
    if not MODEL:
        found.append("MODEL is empty; set it to a model tag you have pulled in Ollama.")
    if PROVIDER == "gemini" and not GEMINI_API_KEY:
        found.append("PROVIDER=gemini needs GEMINI_API_KEY.")
    if MAX_RETRIES < 0:
        found.append("MAX_RETRIES must be 0 or more.")
    if BATCH_SIZE < 1:
        found.append(f"BATCH_SIZE must be 1 or more, got {BATCH_SIZE}.")
    if RETRIEVAL_K < 1:
        found.append(f"RETRIEVAL_K must be 1 or more, got {RETRIEVAL_K}.")
    if RETRIEVAL_WINDOW_DAYS is not None and RETRIEVAL_WINDOW_DAYS < 1:
        found.append("RETRIEVAL_WINDOW_DAYS must be 1 or more, or empty for unbounded.")
    if EMBEDDINGS not in ("off", "on"):
        found.append(f"EMBEDDINGS={EMBEDDINGS!r} must be off or on.")
    if GATE_MODE not in GATE_MODES:
        found.append(f"GATE_MODE={GATE_MODE!r} is not one of {', '.join(GATE_MODES)}.")
    if BIN_RETENTION_DAYS < 1:
        found.append(f"BIN_RETENTION_DAYS must be 1 or more, got {BIN_RETENTION_DAYS}.")
    if not INBOX_PATH.exists():
        found.append(f"INBOX_FILE {INBOX_PATH} does not exist.")
    return found


def warnings():
    found = []
    if FALLBACK_PROVIDER == "gemini" and not GEMINI_API_KEY:
        found.append("FALLBACK_PROVIDER=gemini but GEMINI_API_KEY is empty: the fallback is disabled for this run.")
    return found


def check():
    """Raise ConfigError before the first model call rather than halfway through a run."""
    found = problems()
    if found:
        raise ConfigError(" ".join(found))
    return warnings()


def fallback_available():
    return FALLBACK_PROVIDER == "gemini" and bool(GEMINI_API_KEY)


def describe():
    source = (
        f"loaded {len(LOADED_FROM_FILE)} key(s) from .env" if LOADED_FROM_FILE else "no .env file, using defaults"
    )
    key_state = "set" if GEMINI_API_KEY else "empty"
    print("=== Configuration ===")
    print(f"  provider        {PROVIDER}  (model {MODEL})")
    print(f"  ollama host     {OLLAMA_HOST}")
    print(f"  fallback        {FALLBACK_PROVIDER or 'none'}  (model {GEMINI_MODEL}, key {key_state})")
    print(f"  batch size      {BATCH_SIZE}" + ("  (one message per call)" if BATCH_SIZE == 1 else "  (messages share a call; sensitive ones never do)"))
    print(f"  retries         {MAX_RETRIES}, spacing {CALL_SPACING_SECONDS if CALL_SPACING_SECONDS is not None else 'provider default'}")
    print(f"  temperature     {TEMPERATURE}, timeout {REQUEST_TIMEOUT_S:.0f}s")
    window = f"{RETRIEVAL_WINDOW_DAYS}d" if RETRIEVAL_WINDOW_DAYS else "unbounded"
    print(f"  retrieval       k={RETRIEVAL_K}, keyword window {window}, embeddings {EMBEDDINGS}")
    print(f"  gate            {GATE_MODE}, deleted mail recoverable for {BIN_RETENTION_DAYS} days")
    print(f"  owner           {OWNER}")
    print(f"  inbox           {INBOX_PATH}")
    print(f"  state           {STATE_PATH}")
    print(f"  outbox          {OUTBOX_PATH}")
    print(f"  trace           {TRACE_PATH}")
    print(f"  dashboard       {DASHBOARD_PATH}")
    print(f"  ({source})")


if __name__ == "__main__":
    describe()
    for line in warnings():
        print(f"  warning: {line}")
    bad = problems()
    for line in bad:
        print(f"  PROBLEM: {line}")
    raise SystemExit(1 if bad else 0)
