"""Foundation tests: config, provider, trace. No network, no model.

Run:  python -m unittest discover -s tests -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import provider  # noqa: E402
import trace  # noqa: E402

CONFIG_KEYS = list(config._DEFAULTS)


class EnvSandbox(unittest.TestCase):
    """Each test gets a clean environment and a temp state dir, then config.reload()."""

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in CONFIG_KEYS}
        for k in CONFIG_KEYS:
            os.environ.pop(k, None)
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["STATE_DIR"] = self.tmp.name
        os.environ["TRACE_FILE"] = str(Path(self.tmp.name) / "trace.jsonl")
        self._env_file = config.ENV_FILE
        config.ENV_FILE = Path(self.tmp.name) / "no-such.env"  # tests never read the real .env
        config.reload()
        provider.reset()

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        config.ENV_FILE = self._env_file
        config.reload()
        provider.reset()
        self.tmp.cleanup()

    def set_env(self, **kv):
        for k, v in kv.items():
            os.environ[k] = v
        config.reload()


class ConfigTests(EnvSandbox):
    def test_env_file_is_parsed_and_real_environment_wins(self):
        env = Path(self.tmp.name) / ".env"
        env.write_text("# comment\nMODEL=from-file\nTEMPERATURE='0.7'  # trailing comment\nBROKEN LINE\n", encoding="utf-8")
        os.environ["MODEL"] = "from-env"
        loaded = config._load_env_file(env)
        self.assertEqual(loaded, {"MODEL": "from-file", "TEMPERATURE": "0.7"})
        self.assertEqual(os.environ["MODEL"], "from-env")
        self.assertEqual(os.environ["TEMPERATURE"], "0.7")

    def test_defaults_resolve(self):
        self.assertEqual(config.PROVIDER, "ollama")
        self.assertIsNone(config.CALL_SPACING_SECONDS)
        self.assertEqual(config.problems(), [])

    def test_a_bad_setting_is_a_problem_not_a_crash(self):
        self.set_env(PROVIDER="maybe")
        self.assertTrue(any("PROVIDER=" in p for p in config.problems()))
        with self.assertRaises(config.ConfigError):
            config.check()

    def test_gemini_primary_needs_key(self):
        self.set_env(PROVIDER="gemini", FALLBACK_PROVIDER="")
        self.assertTrue(any("GEMINI_API_KEY" in p for p in config.problems()))
        self.set_env(GEMINI_API_KEY="k")
        self.assertEqual(config.problems(), [])

    def test_fallback_without_key_is_a_warning_only(self):
        self.assertEqual(config.problems(), [])
        self.assertFalse(config.fallback_available())
        self.assertTrue(config.warnings())
        self.set_env(GEMINI_API_KEY="k")
        self.assertTrue(config.fallback_available())
        self.assertEqual(config.warnings(), [])

    def test_unknown_provider_and_same_fallback_are_problems(self):
        self.set_env(PROVIDER="openai")
        self.assertTrue(any("PROVIDER=" in p for p in config.problems()))
        self.set_env(PROVIDER="ollama", FALLBACK_PROVIDER="ollama")
        self.assertTrue(any("same as PROVIDER" in p for p in config.problems()))


class FakeResponse:
    def __init__(self, status, payload=None, headers=None, text=""):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._payload = payload
        self.headers = headers or {}
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def ollama_ok(text):
    return FakeResponse(200, {"message": {"role": "assistant", "content": text}, "prompt_eval_count": 10, "eval_count": 5})


def gemini_ok(text):
    return FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": text}]}}], "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3}})


class ProviderTests(EnvSandbox):
    def test_parse_json_tolerates_fences_and_prose(self):
        self.assertEqual(provider.parse_json('{"a": 1}'), {"a": 1})
        self.assertEqual(provider.parse_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(provider.parse_json('Sure! Here it is: {"a": {"b": 2}} hope that helps'), {"a": {"b": 2}})

    def test_parse_json_refuses_garbage_and_non_objects(self):
        for bad in ("", "   ", "not json at all", "[1, 2, 3]", '{"a": 1'):
            with self.assertRaises(ValueError, msg=repr(bad)):
                provider.parse_json(bad)

    @mock.patch("provider.time.sleep")
    @mock.patch("provider.requests.post")
    def test_429_is_retried_then_succeeds(self, post, sleep):
        post.side_effect = [FakeResponse(429, text="slow down", headers={"Retry-After": "2"}), ollama_ok('{"ok": true}')]
        reply = provider.chat([provider.user_turn("hi")])
        self.assertEqual(reply.content, '{"ok": true}')
        self.assertEqual(reply.attempts, 2)
        self.assertEqual(reply.provider, "ollama")
        sleep.assert_any_call(2.0)
        self.assertIn("format", post.call_args.kwargs["json"])

    @mock.patch("provider.time.sleep")
    @mock.patch("provider.requests.post")
    def test_gives_up_after_max_retries_with_provider_error(self, post, sleep):
        self.set_env(MAX_RETRIES="1")
        post.side_effect = [FakeResponse(503), FakeResponse(503), FakeResponse(503)]
        with self.assertRaises(provider.ProviderError) as ctx:
            provider.chat([provider.user_turn("hi")])
        self.assertIn("Gave up", str(ctx.exception))
        self.assertEqual(post.call_count, 2)

    @mock.patch("provider.time.sleep")
    @mock.patch("provider.requests.post")
    def test_falls_back_to_gemini_only_when_ollama_unreachable(self, post, sleep):
        import requests

        self.set_env(GEMINI_API_KEY="k")

        def route(url, **kwargs):
            if "localhost" in url:
                raise requests.exceptions.ConnectionError("refused")
            self.assertIn("generativelanguage", url)
            self.assertEqual(kwargs["headers"]["x-goog-api-key"], "k")
            self.assertEqual(kwargs["json"]["generationConfig"]["responseMimeType"], "application/json")
            return gemini_ok('{"ok": true}')

        post.side_effect = route
        reply = provider.chat([provider.system_turn("sys"), provider.user_turn("hi")])
        self.assertEqual(reply.provider, "gemini")
        self.assertEqual(reply.model, config.GEMINI_MODEL)
        self.assertEqual(provider.active_provider(), "gemini")
        kinds = [r["event"] for r in trace.read()]
        self.assertIn("provider_fallback", kinds)

    @mock.patch("provider.requests.post")
    def test_unreachable_without_fallback_raises_cleanly(self, post):
        import requests

        post.side_effect = requests.exceptions.ConnectionError("refused")
        with self.assertRaises(provider.Unreachable):
            provider.chat([provider.user_turn("hi")])

    @mock.patch("provider.requests.post")
    def test_missing_model_is_a_clear_error(self, post):
        post.return_value = FakeResponse(404, text="model not found")
        with self.assertRaises(provider.ProviderError) as ctx:
            provider.chat([provider.user_turn("hi")])
        self.assertIn("ollama pull", str(ctx.exception))

    @mock.patch("provider.requests.post")
    def test_gemini_blocked_reply_is_a_provider_error(self, post):
        self.set_env(PROVIDER="gemini", FALLBACK_PROVIDER="", GEMINI_API_KEY="k")
        post.return_value = FakeResponse(200, {"promptFeedback": {"blockReason": "SAFETY"}})
        with self.assertRaises(provider.ProviderError) as ctx:
            provider.chat([provider.user_turn("hi")])
        self.assertIn("SAFETY", str(ctx.exception))


class TraceTests(EnvSandbox):
    def test_events_carry_run_id_and_cap_and_are_readable(self):
        trace.start_run(cap="R1", fresh=True)
        trace.event("decision", msg_id="m001", disposition="reply")
        trace.event("decision", msg_id="m002", disposition="archive", cap="R2")
        records = trace.read()
        self.assertEqual([r["event"] for r in records], ["run_start", "decision", "decision"])
        self.assertEqual(records[1]["cap"], "R1")
        self.assertEqual(records[2]["cap"], "R2")
        self.assertEqual(len(trace.read(msg_id="m001")), 1)
        self.assertEqual(len(trace.read(cap="R1")), 2)

    def test_moya_events_are_forwarded(self):
        from moya.observability.event_bus import EventBus
        from moya.observability.events import StepCompletedEvent

        bus = trace.attach(EventBus())
        trace.start_run(cap="R1", fresh=True)
        bus.publish(StepCompletedEvent(event_type="step.completed", source="pipeline", pipeline_id="zero", thread_id="m001", step_name="validate", step_type="FunctionStep", duration_ms=3))
        forwarded = trace.read(kind="moya.step.completed")
        self.assertEqual(len(forwarded), 1)
        self.assertEqual(forwarded[0]["step_name"], "validate")


if __name__ == "__main__":
    unittest.main()
