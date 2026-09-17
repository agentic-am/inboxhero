"""Configuration, read from the environment. Nothing about the model is hardcoded.

Values come from the real environment first, then from a `.env` file beside
this module, then from the defaults below. A real environment variable always
wins, so `MODEL=llama3.1:8b python demo.py --cap R1` overrides the file
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
    "MODEL": "llama3.2:3b",
    "OLLAMA_HOST": "http://localhost:11434",
    "FALLBACK_PROVIDER": "gemini",
    "GEMINI_API_KEY": "",
    "GEMINI_MODEL": "gemini-3.5-flash-lite",
    # Rate limits. Empty spacing means "0 on Ollama, 4.5 seconds on Gemini",
    # which is about thirteen requests a minute against a free-tier limit of 15.
    "CALL_SPACING_SECONDS": "",
    "MAX_RETRIES": "3",
    "TEMPERATURE": "0.1",
    "REQUEST_TIMEOUT_S": "180",
    # How many messages may share one model call. 1 keeps every message in its
    # own context, which is the safe default: text from one sender can then
    # never influence the decision about another. Raising it trades that
    # isolation for speed and tokens. Messages the rules marked sensitive are
    # never batched, whatever this is set to.
    "BATCH_SIZE": "1",
    # The inbox and where the run leaves its footprints.
    "OWNER": "sam@paperjet.io",
    "INBOX_FILE": "data/inbox.json",
    "STATE_DIR": "state",
    "TRACE_FILE": "trace.jsonl",
}


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
    global INBOX_PATH, STATE_PATH, TRACE_PATH

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

    OWNER = _setting("OWNER").strip().lower()

    INBOX_PATH = _path(_setting("INBOX_FILE"))
    STATE_PATH = _path(_setting("STATE_DIR"))
    TRACE_PATH = _path(_setting("TRACE_FILE"))


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
        found.append("MODEL is empty; e.g. MODEL=llama3.2:3b")
    if PROVIDER == "gemini" and not GEMINI_API_KEY:
        found.append("PROVIDER=gemini needs GEMINI_API_KEY.")
    if MAX_RETRIES < 0:
        found.append("MAX_RETRIES must be 0 or more.")
    if BATCH_SIZE < 1:
        found.append(f"BATCH_SIZE must be 1 or more, got {BATCH_SIZE}.")
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
    print(f"  owner           {OWNER}")
    print(f"  inbox           {INBOX_PATH}")
    print(f"  state           {STATE_PATH}")
    print(f"  trace           {TRACE_PATH}")
    print(f"  ({source})")


if __name__ == "__main__":
    describe()
    for line in warnings():
        print(f"  warning: {line}")
    bad = problems()
    for line in bad:
        print(f"  PROBLEM: {line}")
    raise SystemExit(1 if bad else 0)
