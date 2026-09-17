"""The only file that knows how to talk to a model.

Two backends behind one function: `chat(messages, json_mode=True)`.

- Ollama: POST /api/chat with `format: json`, plain `requests`, no SDK.
- Gemini: POST generateContent with `responseMimeType: application/json`.

Both return a `ModelReply` whose `content` is the model's text. Nothing
here knows about tools, because in inboxHero the model has none: it reads a
prompt and answers with a JSON proposal that `guard.py` validates.

Rate limits (spec: "handle HTTP 429 without crashing"): every call is paced
by CALL_SPACING_SECONDS and retried MAX_RETRIES times on 429, 5xx and
timeouts with exponential backoff, honouring Retry-After when present.

Fallback: if the primary provider cannot be reached at all (connection
refused, not a bad answer) and `config.fallback_available()`, the process
switches to the fallback for the rest of the run and records that in the
trace. A bad answer is never a reason to switch; the validator handles it.

Usage:
    python provider.py     # one JSON round trip, to prove the wiring works
"""

import json
import re
import time
from dataclasses import dataclass

import requests

import config
import trace

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class ProviderError(RuntimeError):
    """The model could not be reached, or answered with something unusable."""


class Unreachable(ProviderError):
    """The provider is down. This, and only this, triggers the fallback."""


class Transient(ProviderError):
    """429 or 5xx or timeout: worth retrying after a pause."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class ModelReply:
    content: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_s: float = 0.0
    attempts: int = 1


def system_turn(text):
    return {"role": "system", "content": text}


def user_turn(text):
    return {"role": "user", "content": text}


def assistant_turn(text):
    return {"role": "assistant", "content": text}


# --- process state ------------------------------------------------------

_active_provider = None  # set on first call; changes once if the fallback kicks in
_last_call_at = 0.0
_calls = 0


def reset():
    """Forget the fallback switch and pacing clock. Tests call this."""
    global _active_provider, _last_call_at, _calls
    _active_provider = None
    _last_call_at = 0.0
    _calls = 0


def active_provider():
    return _active_provider or config.PROVIDER


def active_model():
    return config.GEMINI_MODEL if active_provider() == "gemini" else config.MODEL


# Gemini's free tier allows roughly fifteen requests a minute. Four seconds is
# exactly fifteen, which leaves no room for a clock that disagrees with theirs,
# so the default is 4.5 (about thirteen a minute). Ollama is local and unpaced.
GEMINI_SPACING_S = 4.5


def _spacing():
    if config.CALL_SPACING_SECONDS is not None:
        return config.CALL_SPACING_SECONDS
    return GEMINI_SPACING_S if active_provider() == "gemini" else 0.0


def _pace():
    global _last_call_at
    wait = _spacing() - (time.monotonic() - _last_call_at)
    if wait > 0:
        time.sleep(wait)
    _last_call_at = time.monotonic()


# --- backends -----------------------------------------------------------


def _post(url, body, headers=None, provider="ollama"):
    try:
        response = requests.post(url, json=body, headers=headers or {}, timeout=config.REQUEST_TIMEOUT_S)
    except requests.exceptions.ConnectionError as error:
        raise Unreachable(f"Cannot reach {provider} at {url.split('/v1beta')[0]}: {error.__class__.__name__}") from error
    except requests.exceptions.Timeout as error:
        raise Transient(f"{provider} did not answer within {config.REQUEST_TIMEOUT_S:.0f}s") from error

    if response.status_code == 429 or response.status_code >= 500:
        retry_after = response.headers.get("Retry-After")
        raise Transient(
            f"{provider} returned HTTP {response.status_code}: {response.text[:200]}",
            retry_after=float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else None,
        )
    if response.status_code == 404 and provider == "ollama":
        raise ProviderError(f"Ollama does not have model {config.MODEL!r}. Pull it first: ollama pull {config.MODEL}")
    if not response.ok:
        raise ProviderError(f"{provider} returned HTTP {response.status_code}: {response.text[:300]}")
    try:
        return response.json()
    except ValueError as error:
        raise ProviderError(f"{provider} sent a reply that is not JSON: {error}") from error


def _ollama(messages, json_mode, temperature):
    body = {
        "model": config.MODEL,
        "messages": messages,
        "stream": False,
        "think": False,  # a reasoning model would otherwise fold its thinking into the JSON
        "options": {"temperature": temperature},
    }
    if json_mode:
        body["format"] = "json"
    try:
        payload = _post(f"{config.OLLAMA_HOST}/api/chat", body, provider="ollama")
    except ProviderError as error:
        if "think" not in str(error).lower():
            raise
        body.pop("think")  # older models reject the flag; retry once without it
        payload = _post(f"{config.OLLAMA_HOST}/api/chat", body, provider="ollama")
    message = payload.get("message") if isinstance(payload, dict) else None
    if not isinstance(message, dict):
        raise ProviderError(f"No assistant message in Ollama reply: {str(payload)[:200]}")
    return (
        (message.get("content") or "").strip(),
        int(payload.get("prompt_eval_count") or 0),
        int(payload.get("eval_count") or 0),
    )


def _gemini(messages, json_mode, temperature):
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
        for m in messages
        if m["role"] != "system"
    ]
    body = {"contents": contents, "generationConfig": {"temperature": temperature}}
    if system:
        body["system_instruction"] = {"parts": [{"text": system}]}
    if json_mode:
        body["generationConfig"]["responseMimeType"] = "application/json"
    payload = _post(
        GEMINI_URL.format(model=config.GEMINI_MODEL),
        body,
        headers={"x-goog-api-key": config.GEMINI_API_KEY},
        provider="gemini",
    )
    try:
        parts = payload["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError, TypeError):
        reason = (payload.get("promptFeedback") or {}).get("blockReason") if isinstance(payload, dict) else None
        raise ProviderError(f"No candidate text in Gemini reply{f' (blocked: {reason})' if reason else ''}: {str(payload)[:200]}")
    usage = payload.get("usageMetadata") or {}
    return text, int(usage.get("promptTokenCount") or 0), int(usage.get("candidatesTokenCount") or 0)


_BACKENDS = {"ollama": _ollama, "gemini": _gemini}


# --- the public surface ---------------------------------------------------


def chat(messages, json_mode=True, temperature=None):
    """Send a conversation, get the model's text back. Retries, paces, falls back."""
    global _active_provider, _calls
    if _active_provider is None:
        _active_provider = config.PROVIDER
    temperature = config.TEMPERATURE if temperature is None else temperature

    started = time.perf_counter()
    attempts = 0
    while True:
        attempts += 1
        _pace()
        backend = _BACKENDS[_active_provider]
        try:
            content, prompt_tokens, completion_tokens = backend(messages, json_mode, temperature)
            break
        except Unreachable as error:
            if _active_provider == config.PROVIDER and config.fallback_available():
                trace.event("provider_fallback", from_provider=_active_provider, to_provider=config.FALLBACK_PROVIDER, reason=str(error))
                print(f"  [provider] {error}. Falling back to {config.FALLBACK_PROVIDER} for the rest of this run.")
                _active_provider = config.FALLBACK_PROVIDER
                continue
            raise
        except Transient as error:
            if attempts > config.MAX_RETRIES:
                raise ProviderError(f"Gave up after {attempts} attempt(s): {error}") from error
            pause = error.retry_after if error.retry_after else min(2.0 ** attempts, 30.0)
            trace.event("provider_retry", provider=_active_provider, attempt=attempts, pause_s=pause, reason=str(error)[:200])
            print(f"  [provider] {error}. Retrying in {pause:.0f}s ({attempts}/{config.MAX_RETRIES}).")
            time.sleep(pause)

    _calls += 1
    reply = ModelReply(
        content=content,
        provider=_active_provider,
        model=active_model(),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        elapsed_s=round(time.perf_counter() - started, 2),
        attempts=attempts,
    )
    trace.event(
        "llm_call",
        provider=reply.provider,
        model=reply.model,
        messages=len(messages),
        prompt_tokens=reply.prompt_tokens,
        completion_tokens=reply.completion_tokens,
        elapsed_s=reply.elapsed_s,
        attempts=attempts,
    )
    return reply


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def parse_json(text):
    """Turn model text into one JSON object, tolerating fences and stray prose.

    Raises ValueError with a short excerpt when it cannot. The caller decides
    what a bad answer means; this function never invents a default.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty model output")
    cleaned = _FENCE.sub("", text.strip())
    for candidate in (cleaned, cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
        raise ValueError(f"model output is JSON but not an object: {candidate[:80]!r}")
    raise ValueError(f"model output is not JSON: {text.strip()[:120]!r}")


def smoke_test():
    """Prove the configured model is reachable and can answer in JSON."""
    for line in config.check():
        print(f"  warning: {line}")
    config.describe()
    print(f"\nAsking {active_provider()} / {active_model()} for one JSON object...")
    reply = chat(
        [
            system_turn("You answer with a single JSON object and nothing else."),
            user_turn('Return {"ok": true, "model": "<the name you were trained as>", "greeting": "<one short line>"}'),
        ]
    )
    print(f"  raw: {reply.content[:200]}")
    print(f"  parsed: {parse_json(reply.content)}")
    print(f"  {reply.provider}/{reply.model}: {reply.prompt_tokens} in, {reply.completion_tokens} out, {reply.elapsed_s}s, {reply.attempts} attempt(s)")
    return reply


if __name__ == "__main__":
    smoke_test()
