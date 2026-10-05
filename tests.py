"""
PostCraft AI — test suite.

Two groups of tests live here:

1. Offline tests (default)
   -----------------------
   Fake out the Gemini SDK, so the suite is fast, deterministic and free.
   They cover user input validation, a missing API key, Gemini API errors,
   invalid/empty/blocked responses, the X 280-character enforcement
   (including the second shortening attempt and the controlled failure),
   character counting, platform-specific prompts, and the guarantee that the
   API key never reaches the frontend.

   Run with:  python -m unittest

2. Live tests (opt-in)
   --------------------
   These call the real Gemini API with the key from your .env file, so they
   are skipped unless you ask for them:

   Run with:  python -m unittest -v tests.LiveGeminiTests
              (or set RUN_LIVE_TESTS=1 and run the whole suite)

No extra packages are needed — this uses the Python standard library only.
"""

from __future__ import annotations

import os
import re
import unittest
import warnings
from unittest import mock

import app
import gemini_service
from gemini_service import GeminiError

# Every route, prompt and helper is imported after dotenv has run (app.py calls
# load_dotenv() at import time), so GEMINI_API_KEY is already in the environment.
API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()


# ---------------------------------------------------------------------------
# Fakes for the Gemini SDK
# ---------------------------------------------------------------------------


class FakeCandidate:
    """Stands in for one entry of response.candidates."""

    def __init__(self, finish_reason=None):
        self.finish_reason = finish_reason


class FakeResponse:
    """Stands in for a google-genai GenerateContentResponse."""

    def __init__(self, text="", finish_reason=None, block_reason=None, no_candidates=False):
        self.text = text
        if block_reason is not None:
            self.prompt_feedback = mock.Mock(block_reason=block_reason)
        if no_candidates:
            self.candidates = []
        else:
            self.candidates = [FakeCandidate(finish_reason=finish_reason)]


class FakeModels:
    """Records every prompt it is given and replays queued replies.

    Each queued item is either a FakeResponse, an Exception to raise, or a
    callable returning a FakeResponse.
    """

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.models_requested: list[str] = []
        self.configs: list[object] = []

    def generate_content(self, model=None, contents=None, config=None):
        self.prompts.append(contents)
        self.models_requested.append(model)
        self.configs.append(config)

        if not self.replies:
            raise AssertionError(
                f"Gemini was called {len(self.prompts)} time(s), but the test "
                "queued too few replies."
            )
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            return reply()
        return reply


class FakeClient:
    def __init__(self, replies):
        self.models = FakeModels(replies)


def use_fake_gemini(testcase, replies):
    """Point gemini_service at a fake client for the duration of a test.

    Returns the FakeModels object so the test can inspect the prompts that
    were actually sent.
    """
    client = FakeClient(replies)
    patcher = mock.patch.object(
        gemini_service, "get_client", return_value=client
    )
    patcher.start()
    testcase.addCleanup(patcher.stop)
    return client.models


def api_error(code=503, message="UNAVAILABLE. High demand."):
    """Build a real google-genai APIError so error handling is exercised."""
    from google.genai import errors as genai_errors

    return genai_errors.APIError(code, {"error": {"code": code, "message": message}})


_quota_state: dict = {}


def quota_is_exhausted() -> bool:
    """True when Gemini reports the free-tier quota is spent.

    Probed once per test run and cached, so a full live run costs a single
    extra request instead of one per test.
    """
    if "exhausted" not in _quota_state:
        try:
            gemini_service.get_client().models.generate_content(
                model=gemini_service.get_model(),
                contents="Reply with the single word: ok",
                config=gemini_service.types.GenerateContentConfig(max_output_tokens=8),
            )
            _quota_state["exhausted"] = False
        except Exception as exc:
            # Only a quota answer counts; anything else means the key is fine.
            text = str(exc).upper()
            _quota_state["exhausted"] = (
                "RESOURCE_EXHAUSTED" in text or "QUOTA" in text
            ) and "429" in text
    return _quota_state["exhausted"]


# ---------------------------------------------------------------------------
# Offline tests
# ---------------------------------------------------------------------------


class ApiTestCase(unittest.TestCase):
    """Base class giving each test a fresh Flask test client."""

    def setUp(self):
        app.app.config["TESTING"] = True
        self.client = app.app.test_client()

    def generate(self, **payload):
        return self.client.post("/api/generate", json=payload)

    def regenerate(self, **payload):
        return self.client.post("/api/regenerate", json=payload)


class TestHealth(ApiTestCase):
    def test_health_is_ok(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["success"])
        self.assertEqual(body["status"], "ok")

    def test_health_reports_configuration_without_revealing_it(self):
        body = self.client.get("/api/health").get_json()
        # It says whether a key exists, never the key itself.
        self.assertIsInstance(body["gemini_configured"], bool)
        self.assertNotIn("GEMINI_API_KEY", body)
        self.assertNotIn("api_key", body)
        if API_KEY:
            self.assertTrue(body["gemini_configured"])
            self.assertNotIn(API_KEY, str(body))


class TestApiKeyIsNeverExposed(ApiTestCase):
    """The key must not appear in anything the browser can see."""

    def test_key_is_not_in_the_page_or_static_assets(self):
        client = app.app.test_client()
        with warnings.catch_warnings():
            # Flask streams static files; the wrapper is closed by the GC.
            warnings.simplefilter("ignore", ResourceWarning)
            surfaces = {
                "templates/index.html": client.get("/").get_data(as_text=True),
                "static/css/style.css": client.get("/static/css/style.css").get_data(as_text=True),
                "static/js/script.js": client.get("/static/js/script.js").get_data(as_text=True),
            }
        self.assertTrue(API_KEY, "GEMINI_API_KEY should be loaded from .env")

        for name, text in surfaces.items():
            self.assertNotIn(API_KEY, text, f"{name} leaks the API key")
            # Also catch the key without its exact casing/whitespace.
            self.assertNotRegex(text, re.escape(API_KEY[:12]))

    def test_key_is_not_in_any_api_response(self):
        client = app.app.test_client()
        responses = [
            client.get("/api/health").get_data(as_text=True),
            client.post(
                "/api/generate", json={"content": "hello", "platforms": ["x"]}
            ).get_data(as_text=True),
            client.post("/api/generate", json={}).get_data(as_text=True),
        ]
        for text in responses:
            self.assertNotIn(API_KEY, text, "An API response leaks the API key")

    def test_model_name_is_not_exposed(self):
        """The model is a server-side detail and stays out of responses too."""
        body = self.client.get("/api/health").get_json()
        self.assertNotIn(gemini_service.get_model(), str(body))

    def test_frontend_never_references_the_api_key(self):
        with open(
            os.path.join(os.path.dirname(__file__), "static", "js", "script.js"),
            encoding="utf-8",
        ) as handle:
            script = handle.read()
        for forbidden in ("GEMINI_API_KEY", "api_key", "AIza", "googleapis.com"):
            self.assertNotIn(forbidden, script)


class TestInputValidation(ApiTestCase):
    def test_empty_content_is_rejected(self):
        response = self.generate(content="", platforms=["linkedin"])
        self.assertEqual(response.status_code, 400)
        body = response.get_json()
        self.assertFalse(body["success"])
        self.assertEqual(body["code"], "content_required")
        self.assertEqual(body["field"], "content")

    def test_whitespace_only_content_is_rejected(self):
        response = self.generate(content="   \n\t  ", platforms=["linkedin"])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["code"], "content_required")

    def test_missing_content_is_rejected(self):
        response = self.generate(platforms=["linkedin"])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["code"], "content_required")

    def test_non_string_content_is_rejected(self):
        response = self.generate(content=12345, platforms=["linkedin"])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["code"], "invalid_content")

    def test_over_long_content_is_rejected(self):
        response = self.generate(
            content="x" * (app.MAX_CONTENT_CHARS + 1), platforms=["linkedin"]
        )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.get_json()["code"], "content_too_long")

    def test_missing_platforms_is_rejected(self):
        response = self.generate(content="A real idea.")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["code"], "platform_required")

    def test_empty_platform_list_is_rejected(self):
        response = self.generate(content="A real idea.", platforms=[])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["code"], "platform_required")

    def test_unknown_platform_is_rejected_and_lists_supported_ones(self):
        response = self.generate(content="A real idea.", platforms=["tiktok"])
        self.assertEqual(response.status_code, 400)
        body = response.get_json()
        self.assertEqual(body["code"], "invalid_platform")
        self.assertIn("tiktok", body["error"])
        for name in ("LinkedIn", "X", "Dev.to / Medium"):
            self.assertIn(name, body["error"])

    def test_invalid_json_body_is_rejected(self):
        response = self.client.post(
            "/api/generate", data="not json", content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["code"], "invalid_json")

    def test_non_json_content_type_is_rejected(self):
        response = self.client.post(
            "/api/generate", data="content=x", content_type="text/plain"
        )
        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.get_json()["code"], "unsupported_media_type")

    def test_unknown_api_path_returns_json_404(self):
        response = self.client.get("/api/does-not-exist")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_json()["code"], "not_found")

    def test_wrong_method_returns_json_405(self):
        response = self.client.get("/api/generate")
        self.assertEqual(response.status_code, 405)
        self.assertEqual(response.get_json()["code"], "method_not_allowed")

    def test_validation_errors_never_call_gemini(self):
        models = use_fake_gemini(self, [])
        self.generate(content="", platforms=["linkedin"])
        self.assertEqual(models.prompts, [], "Gemini was called for invalid input")


class TestMissingApiKey(ApiTestCase):
    def test_missing_key_returns_controlled_error_not_a_crash(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            gemini_service.reset_client()
            response = self.generate(content="A real idea.", platforms=["linkedin"])

        gemini_service.reset_client()
        self.assertEqual(response.status_code, 500)
        body = response.get_json()
        self.assertFalse(body["success"])
        self.assertEqual(body["code"], "missing_api_key")
        self.assertNotIn("GEMINI_API_KEY", body["error"])
        self.assertNotIn("Traceback", body["error"])

    def test_whitespace_key_counts_as_missing(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "   "}):
            self.assertRaises(gemini_service.MissingApiKeyError, gemini_service.get_api_key)

    def test_key_is_never_raised_as_a_value(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            with self.assertRaises(gemini_service.MissingApiKeyError) as caught:
                gemini_service.get_api_key()
        self.assertNotIn(API_KEY, str(caught.exception))


class TestGeminiFailures(ApiTestCase):
    def test_api_error_returns_safe_json(self):
        use_fake_gemini(self, [api_error(503)])
        response = self.generate(content="A real idea.", platforms=["linkedin"])

        self.assertEqual(response.status_code, 502)
        body = response.get_json()
        self.assertFalse(body["success"])
        self.assertEqual(
            body["error"], "Unable to generate posts right now. Please try again."
        )
        self.assertNotIn("503", body["error"])
        self.assertNotIn("UNAVAILABLE", body["error"])
        self.assertNotIn("Traceback", body["error"])

    def test_quota_error_returns_safe_json(self):
        use_fake_gemini(self, [api_error(429, "RESOURCE_EXHAUSTED. Quota exceeded.")])
        response = self.generate(content="A real idea.", platforms=["linkedin"])
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("RESOURCE_EXHAUSTED", response.get_json()["error"])

    def test_network_failure_returns_safe_json(self):
        use_fake_gemini(self, [ConnectionError("connection reset by peer")])
        response = self.generate(content="A real idea.", platforms=["linkedin"])
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("connection reset", response.get_json()["error"])

    def test_unexpected_exception_does_not_leak_internals(self):
        def boom():
            raise ValueError("internal detail /secret/path")

        use_fake_gemini(self, [boom])
        response = self.generate(content="A real idea.", platforms=["linkedin"])
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("internal detail", response.get_json()["error"])

    def test_technical_detail_is_logged_server_side_only(self):
        models = use_fake_gemini(self, [api_error(503, "UNAVAILABLE. High demand.")])
        with self.assertLogs("gemini_service", level="WARNING") as logs:
            self.generate(content="A real idea.", platforms=["linkedin"])
        joined = "\n".join(logs.output)
        self.assertIn("UNAVAILABLE", joined)
        # Even the server log must not contain the key.
        self.assertNotIn(API_KEY, joined)


class TestInvalidResponses(ApiTestCase):
    def test_empty_text_is_reported_as_invalid(self):
        use_fake_gemini(self, [FakeResponse(text="   ")])
        response = self.generate(content="A real idea.", platforms=["linkedin"])
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.get_json()["code"], "invalid_response")

    def test_no_candidates_is_reported_as_invalid(self):
        use_fake_gemini(self, [FakeResponse(text="", no_candidates=True)])
        response = self.generate(content="A real idea.", platforms=["linkedin"])
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.get_json()["code"], "invalid_response")

    def test_blocked_prompt_is_reported_as_invalid(self):
        use_fake_gemini(self, [FakeResponse(text="", block_reason="SAFETY")])
        response = self.generate(content="A real idea.", platforms=["linkedin"])
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.get_json()["code"], "invalid_response")
        self.assertNotIn("SAFETY", response.get_json()["error"])

    def test_token_limited_response_is_rejected_not_returned(self):
        use_fake_gemini(
            self, [FakeResponse(text="Half a sen", finish_reason="MAX_TOKENS")]
        )
        response = self.generate(content="A real idea.", platforms=["linkedin"])
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.get_json()["code"], "invalid_response")

    def test_wrappers_around_the_post_are_stripped(self):
        wrapped = (
            "```\nHere is your LinkedIn post:\n\nThe actual opening line.\n"
            "The rest of it.\n```"
        )
        use_fake_gemini(self, [FakeResponse(text=wrapped)])
        response = self.generate(content="A real idea.", platforms=["linkedin"])
        self.assertEqual(response.status_code, 200)
        content = response.get_json()["results"][0]["content"]
        self.assertNotIn("```", content)
        self.assertNotIn("Here is your", content)
        self.assertTrue(content.startswith("The actual opening line."))


class TestSuccessfulGeneration(ApiTestCase):
    def test_single_platform_response_shape(self):
        use_fake_gemini(self, [FakeResponse(text="A neat post.")])
        response = self.generate(
            content="A real idea.", platforms=["linkedin"], tone="professional"
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()

        # Exactly the structure the frontend expects.
        self.assertTrue(body["success"])
        self.assertEqual(len(body["results"]), 1)
        result = body["results"][0]
        for field in ("platform", "content", "count"):
            self.assertIn(field, result)
        self.assertEqual(result["platform"], "linkedin")
        self.assertEqual(result["content"], "A neat post.")
        self.assertEqual(body["tone"], "professional")

    def test_all_platforms_generate_one_result_each(self):
        use_fake_gemini(
            self,
            [
                FakeResponse(text="LinkedIn copy."),
                FakeResponse(text="X copy."),
                FakeResponse(text="Medium intro."),
            ],
        )
        body = self.generate(
            content="A real idea.", platforms=["linkedin", "x", "medium"]
        ).get_json()

        self.assertTrue(body["success"])
        self.assertEqual(
            [r["platform"] for r in body["results"]], ["linkedin", "x", "medium"]
        )
        self.assertEqual(
            [r["content"] for r in body["results"]],
            ["LinkedIn copy.", "X copy.", "Medium intro."],
        )

    def test_devto_alias_maps_to_medium(self):
        """The spec's example payload uses "devto"."""
        use_fake_gemini(self, [FakeResponse(text="An article intro.")])
        body = self.generate(content="A real idea.", platforms=["devto"]).get_json()
        self.assertEqual(body["results"][0]["platform"], "medium")

    def test_count_comes_from_the_real_content_not_from_gemini(self):
        text = "one two three four five"
        use_fake_gemini(self, [FakeResponse(text=text)])
        body = self.generate(content="A real idea.", platforms=["linkedin"]).get_json()
        result = body["results"][0]
        self.assertEqual(result["count"], len(text))
        self.assertEqual(result["count"], 23)
        self.assertEqual(result["word_count"], 5)

    def test_prompt_never_asks_gemini_for_a_count(self):
        models = use_fake_gemini(self, [FakeResponse(text="Copy.")])
        self.generate(content="A real idea.", platforms=["linkedin"])
        prompt = models.prompts[0].lower()
        self.assertNotIn("how many characters", prompt)
        self.assertNotIn("word count", prompt.replace("no character or word counts", ""))

    def test_idea_is_passed_through_to_the_prompt(self):
        models = use_fake_gemini(self, [FakeResponse(text="Copy.")])
        idea = "I rebuilt onboarding from nine screens down to two."
        self.generate(content=idea, platforms=["linkedin"])
        self.assertIn(idea, models.prompts[0])

    def test_duplicate_platforms_are_generated_once(self):
        models = use_fake_gemini(self, [FakeResponse(text="Copy.")])
        body = self.generate(
            content="A real idea.", platforms=["x", "twitter", "X"]
        ).get_json()
        self.assertEqual(len(body["results"]), 1)
        self.assertEqual(len(models.prompts), 1)


class TestToneHandling(ApiTestCase):
    def test_tone_reaches_the_prompt(self):
        models = use_fake_gemini(self, [FakeResponse(text="Copy.")])
        self.generate(content="A real idea.", platforms=["linkedin"], tone="casual")
        self.assertIn(gemini_service.TONE_RULES["casual"], models.prompts[0])

    def test_tone_case_and_spacing_are_normalised(self):
        # Tone ids are single words, so case and padding are what we tolerate.
        for raw in ("Storytelling", "  STORYTELLING  ", "\tCasual\n"):
            with self.subTest(raw=raw):
                models = use_fake_gemini(self, [FakeResponse(text="Copy.")])
                body = self.generate(
                    content="A real idea.", platforms=["linkedin"], tone=raw
                ).get_json()
                expected = raw.strip().lower()
                self.assertEqual(body["tone"], expected)
                self.assertIn(gemini_service.TONE_RULES[expected], models.prompts[0])

    def test_every_advertised_tone_is_accepted(self):
        for tone_id in app.TONES:
            with self.subTest(tone=tone_id):
                use_fake_gemini(self, [FakeResponse(text="Copy.")])
                body = self.generate(
                    content="A real idea.", platforms=["linkedin"], tone=tone_id
                ).get_json()
                self.assertEqual(body["tone"], tone_id)
                self.assertNotIn("warning", body)

    def test_unknown_tone_falls_back_to_professional_with_a_warning(self):
        # A tone the app does not offer is a soft failure: generate anyway.
        use_fake_gemini(self, [FakeResponse(text="Copy.")])
        response = self.generate(
            content="A real idea.", platforms=["linkedin"], tone="story-telling"
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["tone"], "professional")
        self.assertIn("warning", body)
        # The fallback still produces a usable post.
        self.assertEqual(len(body["results"]), 1)

    def test_missing_tone_defaults_to_professional(self):
        models = use_fake_gemini(self, [FakeResponse(text="Copy.")])
        body = self.generate(content="A real idea.", platforms=["linkedin"]).get_json()
        self.assertEqual(body["tone"], "professional")
        self.assertIn(gemini_service.TONE_RULES["professional"], models.prompts[0])

    def test_unknown_tone_falls_back_and_warns(self):
        use_fake_gemini(self, [FakeResponse(text="Copy.")])
        body = self.generate(
            content="A real idea.", platforms=["linkedin"], tone="sarcastic"
        ).get_json()
        self.assertEqual(body["tone"], "professional")
        self.assertIn("warning", body)
        self.assertIn("sarcastic", body["warning"])


class TestPlatformSpecificPrompts(ApiTestCase):
    """Each platform needs its own instructions, not one generic prompt."""

    def test_the_three_prompts_are_different_from_each_other(self):
        prompts = {
            platform: gemini_service.build_generation_prompt("idea", platform, "casual")
            for platform in ("linkedin", "x", "medium")
        }
        self.assertEqual(len(set(prompts.values())), 3)

    def test_linkedin_prompt_covers_hooks_bullets_and_hashtags(self):
        prompt = gemini_service.build_generation_prompt("idea", "linkedin", "casual")
        lowered = prompt.lower()
        self.assertIn("hook", lowered)
        self.assertIn("bullet", lowered)
        self.assertIn("hashtag", lowered)

    def test_x_prompt_states_the_hard_280_limit(self):
        prompt = gemini_service.build_generation_prompt("idea", "x", "casual")
        self.assertIn("280", prompt)
        self.assertIn("characters or fewer", prompt)

    def test_medium_prompt_asks_for_a_blog_introduction(self):
        prompt = gemini_service.build_generation_prompt("idea", "medium", "casual")
        self.assertIn("introduction", prompt.lower())

    def test_shared_rules_forbid_fabrication(self):
        prompt = gemini_service.build_generation_prompt("idea", "linkedin", "casual")
        lowered = prompt.lower()
        self.assertIn("never invent", lowered)
        for topic in ("statistic", "personal experiences", "company"):
            self.assertIn(topic, lowered)

    def test_shared_rules_forbid_generic_ai_phrasing(self):
        prompt = gemini_service.build_generation_prompt("idea", "linkedin", "casual")
        self.assertIn("in today's fast-paced world", prompt.lower())

    def test_prompt_asks_for_the_post_only(self):
        prompt = gemini_service.build_generation_prompt("idea", "x", "casual")
        self.assertIn("output only the post copy", prompt.lower())

    def test_variant_changes_the_prompt_for_regenerate(self):
        first = gemini_service.build_generation_prompt("idea", "x", "casual", 0)
        second = gemini_service.build_generation_prompt("idea", "x", "casual", 1)
        self.assertNotEqual(first, second)
        self.assertIn("regeneration", second.lower())

    def test_unknown_platform_prompt_is_rejected(self):
        with self.assertRaises(ValueError):
            gemini_service.build_generation_prompt("idea", "tiktok", "casual")


class TestXCharacterLimit(ApiTestCase):
    def test_x_post_within_280_is_returned_untouched(self):
        text = "A" * 200
        models = use_fake_gemini(self, [FakeResponse(text=text)])
        body = self.generate(content="A real idea.", platforms=["x"]).get_json()

        self.assertEqual(body["results"][0]["content"], text)
        self.assertEqual(len(models.prompts), 1, "no rewrite should be needed")

    def test_over_280_triggers_a_second_shortening_call(self):
        long_text = "B" * 400
        short_text = "B" * 200
        models = use_fake_gemini(
            self, [FakeResponse(text=long_text), FakeResponse(text=short_text)]
        )
        body = self.generate(content="A real idea.", platforms=["x"]).get_json()

        result = body["results"][0]
        self.assertEqual(result["content"], short_text)
        self.assertEqual(len(models.prompts), 2)
        self.assertIn("280", models.prompts[1])
        self.assertIn(long_text, models.prompts[1])

    def test_over_280_is_never_silently_truncated(self):
        """The returned text must be Gemini's own words, not a cut-off copy."""
        long_text = "C" * 400
        models = use_fake_gemini(
            self, [FakeResponse(text=long_text), FakeResponse(text="D" * 120)]
        )
        body = self.generate(content="A real idea.", platforms=["x"]).get_json()
        content = body["results"][0]["content"]

        self.assertEqual(content, "D" * 120)
        self.assertNotEqual(content, long_text[:280])
        self.assertFalse(content.endswith("..."))

    def test_still_over_280_returns_a_controlled_error(self):
        models = use_fake_gemini(
            self, [FakeResponse(text="E" * 400), FakeResponse(text="F" * 350)]
        )
        response = self.generate(content="A real idea.", platforms=["x"])

        self.assertEqual(response.status_code, 502)
        body = response.get_json()
        self.assertFalse(body["success"])
        self.assertEqual(body["code"], "post_too_long")
        self.assertIn("280", body["error"])
        self.assertNotIn("results", body)
        # Exactly two attempts — no infinite retry loop.
        self.assertEqual(len(models.prompts), 2)

    def test_rewrite_asks_for_the_limit_and_keeps_the_tone(self):
        models = use_fake_gemini(
            self, [FakeResponse(text="G" * 300), FakeResponse(text="H" * 100)]
        )
        self.generate(content="A real idea.", platforms=["x"], tone="motivational")
        rewrite = models.prompts[1]
        self.assertIn("280 characters or fewer", rewrite)
        self.assertIn("same meaning and the same tone", rewrite)

    def test_exactly_280_is_accepted(self):
        text = "I" * 280
        models = use_fake_gemini(self, [FakeResponse(text=text)])
        body = self.generate(content="A real idea.", platforms=["x"]).get_json()
        self.assertEqual(body["results"][0]["content"], text)
        self.assertEqual(len(models.prompts), 1)

    def test_limit_flag_is_reported_on_the_result(self):
        use_fake_gemini(self, [FakeResponse(text="J" * 100)])
        body = self.generate(content="A real idea.", platforms=["x"]).get_json()
        result = body["results"][0]
        self.assertEqual(result["limit"], 280)
        self.assertTrue(result["within_limit"])

    def test_linkedin_is_not_held_to_the_x_limit(self):
        use_fake_gemini(self, [FakeResponse(text="K" * 900)])
        response = self.generate(content="A real idea.", platforms=["linkedin"])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["results"][0]["count"], 900)


class TestClientConfiguration(ApiTestCase):
    def test_client_uses_the_key_from_the_environment(self):
        gemini_service.reset_client()
        self.addCleanup(gemini_service.reset_client)
        with mock.patch.object(gemini_service.genai, "Client") as factory:
            gemini_service.get_client()
        _, kwargs = factory.call_args
        self.assertEqual(kwargs["api_key"], API_KEY)

    def test_client_is_cached_between_calls(self):
        gemini_service.reset_client()
        self.addCleanup(gemini_service.reset_client)
        with mock.patch.object(gemini_service.genai, "Client") as factory:
            gemini_service.get_client()
            gemini_service.get_client()
        self.assertEqual(factory.call_count, 1)

    def test_model_env_var_overrides_the_default(self):
        with mock.patch.dict(os.environ, {"GEMINI_MODEL": "gemini-3.7-flash"}):
            self.assertEqual(gemini_service.get_model(), "gemini-3.7-flash")

    def test_model_falls_back_to_the_default(self):
        with mock.patch.dict(os.environ, {"GEMINI_MODEL": ""}):
            self.assertEqual(gemini_service.get_model(), gemini_service.DEFAULT_MODEL)

    def test_request_uses_the_configured_model_and_disables_thinking(self):
        models = use_fake_gemini(self, [FakeResponse(text="Copy.")])
        with mock.patch.dict(os.environ, {"GEMINI_MODEL": "gemini-3.7-flash"}):
            self.generate(content="A real idea.", platforms=["linkedin"])
        self.assertEqual(models.models_requested[0], "gemini-3.7-flash")

        config = models.configs[0]
        self.assertEqual(config.max_output_tokens, gemini_service.MAX_OUTPUT_TOKENS)
        self.assertEqual(config.thinking_config.thinking_budget, 0)


# ---------------------------------------------------------------------------
# Partial results: one platform failing must not lose the others
# ---------------------------------------------------------------------------


class TestPartialPlatformFailure(ApiTestCase):
    """A single Gemini hiccup should never wipe out posts that worked."""

    def test_one_failure_keeps_the_other_two(self):
        use_fake_gemini(
            self,
            [
                FakeResponse(text="LinkedIn copy."),
                api_error(503),
                FakeResponse(text="Medium intro."),
            ],
        )
        response = self.generate(
            content="A real idea.", platforms=["linkedin", "x", "medium"]
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["success"])

        self.assertEqual(len(body["results"]), 3)
        by_platform = {r["platform"]: r for r in body["results"]}

        self.assertTrue(by_platform["linkedin"]["success"])
        self.assertEqual(by_platform["linkedin"]["content"], "LinkedIn copy.")
        self.assertTrue(by_platform["medium"]["success"])
        self.assertEqual(by_platform["medium"]["content"], "Medium intro.")

        failed = by_platform["x"]
        self.assertFalse(failed["success"])
        self.assertEqual(failed["platform"], "x")
        self.assertTrue(failed["error"])
        self.assertNotIn("content", failed)

    def test_results_keep_the_requested_platform_order(self):
        use_fake_gemini(
            self, [FakeResponse(text="a"), api_error(503), FakeResponse(text="b")]
        )
        body = self.generate(
            content="A real idea.", platforms=["linkedin", "x", "medium"]
        ).get_json()
        self.assertEqual(
            [r["platform"] for r in body["results"]], ["linkedin", "x", "medium"]
        )
        self.assertEqual(body["platforms"], ["linkedin", "x", "medium"])

    def test_failed_result_never_leaks_gemini_details(self):
        use_fake_gemini(
            self, [FakeResponse(text="a"), api_error(503, "UNAVAILABLE. High demand.")]
        )
        body = self.generate(
            content="A real idea.", platforms=["linkedin", "x"]
        ).get_json()

        failed = [r for r in body["results"] if not r["success"]][0]
        blob = str(failed)
        self.assertNotIn("UNAVAILABLE", blob)
        self.assertNotIn("503", blob)
        self.assertNotIn("Traceback", blob)
        self.assertNotIn(API_KEY, blob)
        # Only the shared safe message, plus a machine-readable code.
        self.assertEqual(
            failed["error"], "Unable to generate posts right now. Please try again."
        )
        self.assertEqual(failed["code"], "generation_failed")

    def test_failed_result_still_carries_platform_metadata(self):
        """The frontend needs the id, name and limit to build the card."""
        use_fake_gemini(self, [api_error(503)])
        body = self.generate(content="A real idea.", platforms=["x"]).get_json()
        failed = body["results"][0] if body.get("results") else None
        # Single failing platform means the whole request failed (see below),
        # so use a two-platform request to get a mixed body.
        use_fake_gemini(self, [FakeResponse(text="ok"), api_error(503)])
        body = self.generate(content="A real idea.", platforms=["linkedin", "x"]).get_json()
        failed = [r for r in body["results"] if not r["success"]][0]
        self.assertEqual(failed["platform"], "x")
        self.assertEqual(failed["name"], "X")
        self.assertEqual(failed["limit"], 280)

    def test_x_over_280_only_fails_the_x_card(self):
        """The 280-char failure is isolated to X; LinkedIn survives."""
        too_long = "Z" * 400
        use_fake_gemini(
            self,
            [
                FakeResponse(text="LinkedIn copy."),
                FakeResponse(text=too_long),
                FakeResponse(text="still too long " * 30),
            ],
        )
        body = self.generate(
            content="A real idea.", platforms=["linkedin", "x"]
        ).get_json()

        self.assertTrue(body["success"])
        by_platform = {r["platform"]: r for r in body["results"]}
        self.assertTrue(by_platform["linkedin"]["success"])
        self.assertFalse(by_platform["x"]["success"])
        self.assertEqual(by_platform["x"]["code"], "post_too_long")

    def test_missing_key_fails_every_platform_with_one_error(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            gemini_service.reset_client()
            response = self.generate(
                content="A real idea.", platforms=["linkedin", "x", "medium"]
            )
        gemini_service.reset_client()

        # Nothing was generated, so this is a whole-request failure.
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json()["code"], "missing_api_key")

    def test_a_failure_does_not_stop_later_platforms(self):
        """Generation continues after a failure instead of aborting early."""
        models = use_fake_gemini(
            self, [api_error(503), FakeResponse(text="Medium intro.")]
        )
        body = self.generate(
            content="A real idea.", platforms=["linkedin", "medium"]
        ).get_json()
        # Both platforms were attempted.
        self.assertEqual(len(models.prompts), 2)
        self.assertTrue(body["results"][1]["success"])


class TestAllPlatformsFailing(ApiTestCase):
    def test_all_failing_returns_an_error_response(self):
        use_fake_gemini(self, [api_error(503), api_error(503), api_error(503)])
        response = self.generate(
            content="A real idea.", platforms=["linkedin", "x", "medium"]
        )
        self.assertEqual(response.status_code, 502)
        body = response.get_json()
        self.assertFalse(body["success"])
        self.assertEqual(body["code"], "generation_failed")
        # No half-built results are handed to the frontend.
        self.assertNotIn("results", body)

    def test_all_failing_reports_the_first_failure_code(self):
        """A missing key must surface as missing_api_key, not a generic error."""
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            gemini_service.reset_client()
            response = self.generate(
                content="A real idea.", platforms=["linkedin", "medium"]
            )
        gemini_service.reset_client()
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json()["code"], "missing_api_key")

    def test_all_failing_never_leaks_internals(self):
        use_fake_gemini(self, [api_error(503, "UNAVAILABLE.")] * 3)
        body = self.generate(
            content="A real idea.", platforms=["linkedin", "x", "medium"]
        ).get_json()
        self.assertNotIn("UNAVAILABLE", str(body))
        self.assertNotIn(API_KEY, str(body))
        self.assertNotIn("Traceback", str(body))

    def test_unexpected_exception_on_every_platform_is_controlled(self):
        def boom():
            raise ValueError("internal detail")

        use_fake_gemini(self, [boom, boom])
        response = self.generate(content="A real idea.", platforms=["linkedin", "x"])
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("internal detail", response.get_json()["error"])


# ---------------------------------------------------------------------------
# POST /api/regenerate
# ---------------------------------------------------------------------------


class TestRegenerateEndpoint(ApiTestCase):
    def test_returns_only_the_target_platform(self):
        use_fake_gemini(self, [FakeResponse(text="A brand new LinkedIn post.")])
        response = self.regenerate(
            content="A real idea.", platform="linkedin", tone="professional"
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()

        self.assertTrue(body["success"])
        self.assertEqual(body["platform"], "linkedin")
        # A single `result` object, not a `results` array.
        self.assertNotIn("results", body)
        result = body["result"]
        self.assertTrue(result["success"])
        self.assertEqual(result["platform"], "linkedin")
        self.assertEqual(result["content"], "A brand new LinkedIn post.")
        self.assertEqual(result["count"], len(result["content"]))

    def test_only_the_named_platform_is_ever_requested(self):
        """One Gemini call, and its prompt targets just that platform."""
        models = use_fake_gemini(self, [FakeResponse(text="Copy.")])
        self.regenerate(content="A real idea.", platform="x", tone="casual")

        self.assertEqual(len(models.prompts), 1)
        prompt = models.prompts[0]
        self.assertIn("Platform: X (formerly Twitter)", prompt)
        self.assertIn("A real idea.", prompt)
        self.assertIn(gemini_service.TONE_RULES["casual"], prompt)

    def test_correct_platform_regenerated_for_each_target(self):
        """Each platform gets its own prompt, so only the right one changes."""
        expectations = {
            "linkedin": "Platform: LinkedIn",
            "x": "Platform: X (formerly Twitter)",
            "medium": "Platform: Dev.to / Medium",
        }
        for platform, marker in expectations.items():
            with self.subTest(platform=platform):
                models = use_fake_gemini(self, [FakeResponse(text="Copy.")])
                self.regenerate(content="A real idea.", platform=platform)
                self.assertEqual(len(models.prompts), 1)
                self.assertIn(marker, models.prompts[0])
                for other, other_marker in expectations.items():
                    if other != platform:
                        self.assertNotIn(other_marker, models.prompts[0])

    def test_regenerated_x_stays_within_280_characters(self):
        over = "Q" * 400
        under = "Q" * 240
        use_fake_gemini(self, [FakeResponse(text=over), FakeResponse(text=under)])
        body = self.regenerate(content="A real idea.", platform="x").get_json()

        result = body["result"]
        self.assertLessEqual(len(result["content"]), 280)
        self.assertEqual(len(result["content"]), 240)
        self.assertTrue(result["within_limit"])

    def test_x_still_too_long_returns_a_failed_result_card(self):
        use_fake_gemini(
            self, [FakeResponse(text="W" * 400), FakeResponse(text="V" * 300)]
        )
        response = self.regenerate(content="A real idea.", platform="x")
        self.assertEqual(response.status_code, 502)
        body = response.get_json()

        self.assertFalse(body["success"])
        self.assertEqual(body["code"], "post_too_long")
        # The failed card travels with the error so the UI can render it.
        self.assertFalse(body["result"]["success"])
        self.assertEqual(body["result"]["platform"], "x")
        self.assertNotIn("results", body)

    def test_api_failure_returns_a_safe_failed_result(self):
        use_fake_gemini(self, [api_error(503, "UNAVAILABLE. High demand.")])
        response = self.regenerate(content="A real idea.", platform="linkedin")

        self.assertEqual(response.status_code, 502)
        body = response.get_json()
        self.assertFalse(body["success"])
        self.assertEqual(
            body["error"], "Unable to generate posts right now. Please try again."
        )
        self.assertFalse(body["result"]["success"])
        self.assertNotIn("UNAVAILABLE", str(body))
        self.assertNotIn("503", str(body))
        self.assertNotIn(API_KEY, str(body))

    def test_missing_key_is_reported(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            gemini_service.reset_client()
            response = self.regenerate(content="A real idea.", platform="linkedin")
        gemini_service.reset_client()
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json()["code"], "missing_api_key")

    def test_invalid_x_response_is_reported(self):
        use_fake_gemini(self, [FakeResponse(text="")])
        response = self.regenerate(content="A real idea.", platform="linkedin")
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.get_json()["code"], "invalid_response")

    def test_variant_is_carried_through_so_takes_differ(self):
        models = use_fake_gemini(self, [FakeResponse(text="Copy."), FakeResponse(text="Copy.")])
        first = self.regenerate(content="A real idea.", platform="linkedin", variant=0)
        second = self.regenerate(content="A real idea.", platform="linkedin", variant=1)

        self.assertIn("regeneration", models.prompts[1].lower())
        self.assertNotEqual(first.get_json()["result"]["variant"], 1)
        self.assertEqual(second.get_json()["result"]["variant"], 1)

    def test_tone_is_honoured(self):
        models = use_fake_gemini(self, [FakeResponse(text="Copy.")])
        body = self.regenerate(
            content="A real idea.", platform="linkedin", tone="Educational"
        ).get_json()
        self.assertEqual(body["tone"], "educational")
        self.assertIn(gemini_service.TONE_RULES["educational"], models.prompts[0])

    def test_unknown_tone_falls_back_and_warns(self):
        use_fake_gemini(self, [FakeResponse(text="Copy.")])
        body = self.regenerate(
            content="A real idea.", platform="linkedin", tone="sarcastic"
        ).get_json()
        self.assertEqual(body["tone"], "professional")
        self.assertIn("warning", body)

    # --- validation -------------------------------------------------------

    def test_empty_content_is_rejected(self):
        use_fake_gemini(self, [])
        response = self.regenerate(content="  ", platform="linkedin")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["code"], "content_required")

    def test_missing_platform_is_rejected(self):
        use_fake_gemini(self, [])
        response = self.regenerate(content="A real idea.")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["code"], "platform_required")

    def test_unknown_platform_is_rejected(self):
        use_fake_gemini(self, [])
        response = self.regenerate(content="A real idea.", platform="tiktok")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["code"], "invalid_platform")

    def test_more_than_one_platform_is_rejected(self):
        """Regenerate is per-platform; asking for three is a client bug."""
        use_fake_gemini(self, [])
        response = self.regenerate(
            content="A real idea.", platforms=["linkedin", "x", "medium"]
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["code"], "single_platform_required")

    def test_single_item_platforms_list_is_accepted(self):
        use_fake_gemini(self, [FakeResponse(text="Copy.")])
        response = self.regenerate(content="A real idea.", platforms=["x"])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["platform"], "x")

    def test_devto_alias_is_accepted(self):
        use_fake_gemini(self, [FakeResponse(text="Copy.")])
        body = self.regenerate(content="A real idea.", platform="devto").get_json()
        self.assertEqual(body["platform"], "medium")

    def test_validation_never_calls_gemini(self):
        models = use_fake_gemini(self, [])
        self.regenerate(content="", platform="linkedin")
        self.regenerate(content="A real idea.", platform="tiktok")
        self.regenerate(content="A real idea.")
        self.assertEqual(models.prompts, [])

    def test_wrong_method_returns_json_405(self):
        self.assertEqual(self.client.get("/api/regenerate").status_code, 405)

    def test_retry_after_failure_succeeds(self):
        """The same request works on a second attempt, as the UI allows."""
        use_fake_gemini(self, [api_error(503)])
        first = self.regenerate(content="A real idea.", platform="linkedin")
        self.assertEqual(first.status_code, 502)

        use_fake_gemini(self, [FakeResponse(text="Recovered copy.")])
        second = self.regenerate(content="A real idea.", platform="linkedin", variant=1)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.get_json()["result"]["content"], "Recovered copy.")


# ---------------------------------------------------------------------------
# Frontend wiring
#
# The browser code is a plain <script>, and this project deliberately has no
# JavaScript runtime or test framework. These tests check that the handlers,
# selectors and states the features rely on are actually present and wired
# together, rather than executing them in a DOM.
# ---------------------------------------------------------------------------


def read_script() -> str:
    with open(
        os.path.join(os.path.dirname(__file__), "static", "js", "script.js"),
        encoding="utf-8",
    ) as handle:
        return handle.read()


def js_block(source: str, header: str) -> str:
    """Return the { ... } block that starts at `header`.

    Matching on braces beats splitting on a string, because the block may
    contain early `return;` statements.
    """
    start = source.find(header)
    if start == -1:
        raise AssertionError(f"no block found for header: {header!r}")
    opening = source.index("{", start + len(header))
    depth = 0
    for index in range(opening, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[opening : index + 1]
    raise AssertionError(f"unbalanced block for header: {header!r}")


class SourceAssertions(unittest.TestCase):
    """assertions that fail with a short message instead of dumping the file."""

    def assertSource(self, haystack: str, needle: str, label: str):
        if needle not in haystack:
            self.fail(f"{label}\n  expected to find: {needle!r}")

    def assertNotSource(self, haystack: str, needle: str, label: str):
        if needle in haystack:
            self.fail(f"{label}\n  did not expect to find: {needle!r}")


class TestFrontendWiring(SourceAssertions):
    def setUp(self):
        self.js = read_script()

    def test_regenerate_posts_to_the_regenerate_endpoint(self):
        self.assertSource(
            self.js,
            "var REGENERATE_ENDPOINT = '/api/regenerate';",
            "regenerate endpoint is configured",
        )
        self.assertSource(self.js, "requestRegeneration(payload)", "regeneration seam exists")

        handler = js_block(self.js, "if (action === 'regenerate')")
        self.assertSource(handler, "requestRegeneration({", "handler calls the new endpoint")
        self.assertNotSource(
            handler, "requestGeneration({", "handler must not use the bulk endpoint"
        )

    def test_regenerate_sends_only_content_platform_and_tone(self):
        handler = js_block(self.js, "if (action === 'regenerate')")
        sent = handler[handler.index("requestRegeneration({") :]
        self.assertSource(sent, "content: lastRequest.content", "original idea is reused")
        self.assertSource(sent, "platform: platformId", "target platform is sent")
        self.assertSource(sent, "tone: card.dataset.tone", "tone is sent")
        self.assertNotSource(sent, "platforms:", "must not send the full platform list")

    def test_original_request_is_kept_in_memory(self):
        """Content, platforms and tone are remembered for Regenerate."""
        self.assertSource(self.js, "var lastRequest = null;", "lastRequest exists")
        self.assertSource(self.js, "lastRequest = { content: content", "idea is stored")
        self.assertSource(self.js, "platforms: platforms", "platforms are stored")
        # Memory only: nothing is persisted anywhere.
        for store in ("localStorage", "sessionStorage", "indexedDB", "document.cookie"):
            self.assertNotSource(self.js, store, "no browser storage")

    def test_only_the_clicked_card_is_disabled_while_regenerating(self):
        handler = js_block(self.js, "if (action === 'regenerate')")
        self.assertSource(handler, "setRegenerating(card, true)", "busy state on")
        self.assertSource(handler, "setRegenerating(card, false)", "busy state off")
        self.assertNotSource(handler, "setFormLoading(", "must not touch the form button")

        # The helper is scoped to the card, so sibling cards stay usable.
        helper = js_block(self.js, "function setRegenerating")
        self.assertSource(
            helper, "$('[data-action=\"regenerate\"]', card)", "button looked up on the card"
        )
        self.assertSource(helper, "button.disabled = on", "only that button is disabled")
        self.assertNotSource(helper, "resultsList", "no other card is touched")

    def test_regenerating_shows_a_loading_label(self):
        helper = js_block(self.js, "function setRegenerating")
        self.assertSource(helper, "'Regenerating…'", "busy label is shown")
        self.assertSource(helper, "button.classList.toggle('is-busy', on)", "busy class")

    def test_regenerate_failure_keeps_the_existing_post(self):
        handler = js_block(self.js, "if (action === 'regenerate')")
        catch_block = handler[handler.index(".catch(function (error)") :]
        self.assertSource(catch_block, "showCardError(card,", "failure shows a card error")

        # Only the success path swaps the text, so a failure cannot clear it.
        success_block = handler[
            handler.index(".then(function (data)") : handler.index(".catch(function (error)")
        ]
        self.assertSource(success_block, "applyResult(card, result)", "success replaces text")
        self.assertNotSource(
            catch_block, "applyResult(", "failure must not touch the post text"
        )
        self.assertNotSource(catch_block, "resetResults(", "failure must not clear the list")

    def test_cards_carry_an_error_state(self):
        self.assertSource(self.js, "card.dataset.state = state", "cards record their state")
        self.assertSource(self.js, "data-error-state hidden", "error element exists")
        self.assertSource(self.js, "$('[data-error-state]', card)", "error element is queried")
        self.assertSource(
            self.js, "box.textContent = opts.errorMessage", "error text is set safely"
        )
        self.assertSource(self.js, "setRegenerateLabel(card, 'Retry')", "Retry label")

        # The three states the UI relies on.
        for state in ("'ready'", "'loading'", "'error'"):
            self.assertSource(self.js, state, f"{state} state exists")

    def test_failed_platforms_render_as_error_cards(self):
        renderer = js_block(self.js, "function renderResults")
        self.assertSource(renderer, "if (result.success === false)", "failure branch")
        self.assertSource(renderer, "errorMessage: result.error", "server message is used")

    def test_copy_uses_the_rendered_post_text(self):
        handler = js_block(self.js, "if (action === 'copy')")
        self.assertSource(
            handler, "copyText($('[data-text]', card).textContent)", "copies the post"
        )
        self.assertSource(handler, "'Copied!'", "copy feedback")

    def test_edit_writes_the_edited_text_back_to_the_card(self):
        self.assertSource(self.js, "function enterEditMode(card)", "enter edit mode")
        self.assertSource(self.js, "function exitEditMode(card, text)", "exit edit mode")

        save = js_block(self.js, "if (action === 'save')")
        self.assertSource(save, "$('[data-text]', card).textContent = next", "text is saved")
        self.assertSource(save, "updateCount(card, next)", "count is refreshed")

    def test_counts_update_while_editing(self):
        listener = js_block(self.js, "resultsList.addEventListener('input'")
        self.assertSource(listener, "updateCount(card, event.target.value)", "live count")

    def test_x_count_shows_limit_and_updates(self):
        counter = js_block(self.js, "function updateCount(")
        self.assertSource(counter, "chars + ' / ' + cfg.limit", "count shows n / limit")
        self.assertSource(
            counter, "countEl.classList.toggle('is-over', chars > cfg.limit)", "over-limit flag"
        )

    def test_edited_content_is_never_truncated(self):
        """No slicing of post text anywhere in the frontend."""
        for pattern in (".slice(0,", ".substring(0,", ".substr(0,"):
            self.assertNotSource(self.js, pattern, "user text must not be cut")
        # Over-limit is only ever styled, never enforced by removing text.
        self.assertSource(
            self.js,
            "countEl.classList.toggle('is-over', chars > cfg.limit)",
            "over-limit is a warning",
        )

    def test_copy_all_skips_failed_cards(self):
        block = js_block(self.js, "copyAllBtn.addEventListener")
        self.assertSource(block, "card.dataset.state === 'ready'", "only ready cards copied")

    def test_no_page_reload_on_regenerate(self):
        self.assertSource(self.js, "event.preventDefault()", "submit is intercepted")
        for reload in ("location.reload", "form.submit()", "location.href ="):
            self.assertNotSource(self.js, reload, "no page reload")

    def test_frontend_sends_no_key_material(self):
        for token in ("GEMINI_API_KEY", "api_key", "AIza", "googleapis.com"):
            self.assertNotSource(self.js, token, "no key material in the frontend")


class TestErrorCardStyles(SourceAssertions):
    def setUp(self):
        with open(
            os.path.join(os.path.dirname(__file__), "static", "css", "style.css"),
            encoding="utf-8",
        ) as handle:
            self.css = handle.read()

    def test_error_card_and_inline_error_are_styled(self):
        self.assertSource(self.css, ".result-error", "error card style")
        self.assertSource(self.css, ".card-error", "inline card error style")

    def test_failed_cards_hide_copy_and_edit(self):
        self.assertSource(
            self.css,
            '.result-card[data-state="error"] [data-action="copy"]',
            "copy hidden on failure",
        )
        self.assertSource(
            self.css,
            '.result-card[data-state="error"] [data-action="edit"]',
            "edit hidden on failure",
        )

    def test_regenerating_button_has_a_busy_state(self):
        self.assertSource(self.css, ".result-actions .btn.is-busy", "busy button style")

    def test_spin_keyframes_is_defined_only_once(self):
        self.assertEqual(self.css.count("@keyframes spin"), 1, "duplicate spin keyframes")


# ---------------------------------------------------------------------------
# Live tests — these really call Gemini.
# ---------------------------------------------------------------------------

RUN_LIVE = (os.getenv("RUN_LIVE_TESTS") or "").strip().lower() in ("1", "true", "yes")

LIVE_IDEA = (
    "I spent three weekends rebuilding our onboarding from scratch and cut the "
    "signup flow from nine screens down to two. The lesson was not that users "
    "needed more explanation, it was that they needed fewer decisions."
)


@unittest.skipUnless(RUN_LIVE, "Set RUN_LIVE_TESTS=1 to call the real Gemini API")
class LiveGeminiTests(ApiTestCase):
    """Real end-to-end generation. Needs a working GEMINI_API_KEY in .env.

    The free Gemini tier has a daily request quota. When it is spent every
    call comes back 429, which is an account limit rather than a bug in this
    project, so these tests skip instead of reporting a false failure.
    """

    def setUp(self):
        super().setUp()
        if not API_KEY:
            self.skipTest("GEMINI_API_KEY is not set in .env")
        if quota_is_exhausted():
            self.skipTest(
                "Gemini free-tier quota is exhausted for today; live tests skipped. "
                "They will run again once the quota resets."
            )

    def test_live_linkedin_only(self):
        response = self.generate(content=LIVE_IDEA, platforms=["linkedin"])
        self.assertEqual(response.status_code, 200)
        result = response.get_json()["results"][0]
        self.assertEqual(result["platform"], "linkedin")
        self.assertTrue(result["content"].strip())
        self.assertEqual(result["count"], len(result["content"]))
        self.assertTrue(result["within_limit"])

    def test_live_x_only_stays_within_280_characters(self):
        response = self.generate(content=LIVE_IDEA, platforms=["x"])
        self.assertEqual(response.status_code, 200)
        result = response.get_json()["results"][0]
        self.assertLessEqual(len(result["content"]), 280)
        self.assertEqual(result["count"], len(result["content"]))
        self.assertFalse(result["content"].rstrip().endswith("..."))

    def test_live_medium_only(self):
        response = self.generate(content=LIVE_IDEA, platforms=["devto"])
        self.assertEqual(response.status_code, 200)
        result = response.get_json()["results"][0]
        self.assertEqual(result["platform"], "medium")
        self.assertTrue(result["content"].strip())

    def test_live_all_platforms_together(self):
        body = self.generate(
            content=LIVE_IDEA, platforms=["linkedin", "x", "medium"]
        ).get_json()
        self.assertTrue(body["success"])
        self.assertEqual(len(body["results"]), 3)
        posts = {r["platform"]: r["content"] for r in body["results"]}
        # Same idea, genuinely different writing per platform.
        self.assertEqual(len(set(posts.values())), 3)
        self.assertLessEqual(len(posts["x"]), 280)

    def test_live_different_tones_produce_different_copy(self):
        casual = self.generate(
            content=LIVE_IDEA, platforms=["linkedin"], tone="casual"
        ).get_json()["results"][0]["content"]
        storytelling = self.generate(
            content=LIVE_IDEA, platforms=["linkedin"], tone="storytelling"
        ).get_json()["results"][0]["content"]
        self.assertNotEqual(casual, storytelling)

    def test_live_regenerate_variant_differs(self):
        first = self.generate(content=LIVE_IDEA, platforms=["x"]).get_json()
        again = self.generate(
            content=LIVE_IDEA, platforms=["x"], tone=first["tone"], variant=1
        ).get_json()
        self.assertNotEqual(
            first["results"][0]["content"], again["results"][0]["content"]
        )

    def test_live_response_never_contains_the_key(self):
        body = self.generate(content=LIVE_IDEA, platforms=["x"]).get_json()
        self.assertNotIn(API_KEY, str(body))


if __name__ == "__main__":
    unittest.main(verbosity=2)