"""
PostCraft AI — Gemini generation service.

A small, self-contained helper around the official Google GenAI SDK
(`google-genai`). The Flask route in app.py only calls
`generate_platform_post()`; every Gemini-specific detail (client, model,
prompts, retries, error handling) lives here.

Configuration (read from the environment, loaded from .env by app.py)
----------------------------------------------------------------------
GEMINI_API_KEY  required. The Gemini API key. Never sent to the browser.
GEMINI_MODEL    optional. Defaults to DEFAULT_MODEL below.

Nothing in this module ever returns the API key or a raw SDK traceback to
the caller: failures are converted into GeminiError with a short, safe
message, while the technical detail is logged server-side only.
"""

from __future__ import annotations

import logging
import os
import re

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

# The SDK logs a warning about automatic function calling on every
# models.generate_content() call. We do not use tools at all, so quiet it.
logging.getLogger("google_genai").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# Used when GEMINI_MODEL is not set in .env.
DEFAULT_MODEL = "gemini-3.8-flash"

# Hard platform limit that must never be violated: an X post over this is
# rejected rather than silently truncated.
X_CHAR_LIMIT = 280

# Longest post we ever ask Gemini for, so a runaway response is cheap.
MAX_OUTPUT_TOKENS = 2048

# Transient Gemini errors (429 / 5xx) are retried with backoff by the SDK.
RETRY_ATTEMPTS = 4


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class GeminiError(Exception):
    """Base class for every failure this service reports to the API layer."""

    code = "generation_failed"
    status = 502
    user_message = "Unable to generate posts right now. Please try again."

    def __init__(self, detail: str = "", user_message: str | None = None):
        super().__init__(detail or self.user_message)
        self.detail = detail
        if user_message:
            self.user_message = user_message


class MissingApiKeyError(GeminiError):
    """GEMINI_API_KEY is not set — a server configuration problem."""

    code = "missing_api_key"
    status = 500
    user_message = "Post generation is not configured on the server right now."


class InvalidResponseError(GeminiError):
    """Gemini replied with nothing usable (empty text, blocked, no candidate)."""

    code = "invalid_response"
    status = 502
    user_message = "The assistant returned an unusable response. Please try again."


class TooLongForPlatformError(GeminiError):
    """X output stayed above 280 characters after a shortening retry."""

    code = "post_too_long"
    status = 502
    user_message = "The X post came back longer than 280 characters. Please try again."


# ---------------------------------------------------------------------------
# Prompt building blocks
# ---------------------------------------------------------------------------

# Shared rules. Kept identical for every platform so the AI behaviour
# (preserve meaning, never invent facts, sound human) never drifts.
BASE_RULES = """You are an experienced social media writer turning one rough idea into finished post copy.

Rules you must always follow:
- Preserve the user's original meaning and every concrete detail they gave you.
- Never invent achievements, statistics, numbers, dates, quotes, company or product names, job titles, events, or personal experiences. If a detail is not in the idea, it does not go in the post.
- Improve clarity, structure and flow. You are polishing their idea, not replacing it with a different one.
- Sound like a real person writing, not a brand bot. No "In today's fast-paced world", no "Let's dive in", no "this isn't just X, it's Y", no "game-changer", no "unlock/leverage/supercharge", no emoji spam, no stacked em-dashes, no motivational clichés.
- Match the requested tone (see below). Tone changes word choice and rhythm, never the facts.
- Output ONLY the post copy itself as plain text. No preamble, no labels like "LinkedIn post:", no explanations, no character or word counts, no markdown code fences, no surrounding quotation marks."""

# One block per platform, so each network gets genuinely different writing
# instructions instead of the same generic prompt.
PLATFORM_RULES = {
    "linkedin": """Platform: LinkedIn. Professional but human.

- Open with a hook in the first one or two lines: a specific observation, a tension, or a question the reader genuinely has. No "I'm thrilled to announce" openings.
- Write in short natural paragraphs with a blank line between them. 120-250 words.
- Use bullet points prefixed with "- " when they genuinely help. Three to five bullets maximum, and only when the idea actually has points to list.
- Close with a real takeaway or a grounded question, not a call to action boilerplate.
- Add 2 to 5 relevant hashtags on their own final line. Relevant only — never invent trending tags, never stack more than five.""",
    "x": f"""Platform: X (formerly Twitter). One single post, not a thread.

- HARD LIMIT: the whole post, spaces and hashtags included, must be {X_CHAR_LIMIT} characters or fewer. Aim for 230-270 characters so you are comfortably inside it.
- Count the characters carefully. Choose the words so it fits — never cut a sentence in half, never end mid-word, and never use "..." to fake a fit.
- One strong idea, tight and natural wording. It should read as something a person chose to post, not a summary of a longer article.
- One punchy opening line, then the point. No line breaks needed.
- At most one hashtag, and only if it clearly earns its place.""",
    "medium": """Platform: Dev.to / Medium. Write the opening of a short blog post — the introduction a reader sees before the rest of the article exists.

- Three short paragraphs, roughly 90-180 words. More room for detail than a social post, but still an intro.
- Open with a line that gives the reader a reason to continue — a concrete situation, a surprising observation, or the question the article answers.
- Ground the second paragraph in the specifics the user gave, then set up what the rest of the piece would cover without listing it.
- No title, no headings, no bullet lists, no hashtags, no conclusion or summary paragraph.""",
}

# How to hit each tone without drifting into AI voice.
TONE_RULES = {
    "professional": "Professional and credible: clear, measured, confident, no hype or marketing gloss.",
    "casual": "Casual and conversational: like a knowledgeable colleague explaining it over coffee. Contractions are fine.",
    "friendly": "Friendly and warm: generous and encouraging, but still full of substance.",
    "storytelling": "Narrative: lead with the moment or turning point from the idea, then what it led to. Only shape what the user gave — never invent a person, place or event.",
    "educational": "Educational and practical: explain how it works, concrete and followable, so the reader can act on it.",
    "motivational": "Motivational and forward-moving: encouraging and energising, without clichés or hollow cheerleading.",
}

VARIANT_RULE = (
    "This is a regeneration. Produce a clearly different take from any previous "
    "attempt: change the opening angle and the structure, while keeping the exact "
    "same meaning, details and tone."
)

X_REWRITE_PROMPT = """Rewrite this X post so the entire post, spaces and hashtags included, is {limit} characters or fewer.

- Cut whole phrases and reword the rest. Keep the same meaning and the same tone.
- Never end mid-sentence, never cut mid-word, and never use "..." to fake a fit.
- Keep it natural and readable, not a list of keywords.
- Output only the rewritten post as plain text, with no preamble, labels or quotes.

Post to rewrite:
---
{post}
---"""


def build_generation_prompt(content: str, platform_id: str, tone_id: str, variant: int = 0) -> str:
    """Assemble the prompt for one platform from the shared + platform rules."""
    platform_rules = PLATFORM_RULES.get(platform_id)
    if not platform_rules:
        raise ValueError(f"No prompt configured for platform '{platform_id}'")

    sections = [
        BASE_RULES,
        f"Tone: {TONE_RULES.get(tone_id, TONE_RULES['professional'])}",
        platform_rules,
    ]
    if variant:
        sections.insert(2, VARIANT_RULE)
    sections.append(f"User's idea:\n---\n{content}\n---")
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Output cleanup
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*\n|\n?\s*```\s*$")
_LABEL_RE = re.compile(
    r"^\s*(?:here(?:'s| is)|sure[,!]?|certainly[,!]?)\b[^:\n]{0,60}:\s*\n+",
    re.IGNORECASE,
)
_PREFIX_LABEL_RE = re.compile(
    r"^\s*(?:final\s+)?(?:linkedin|x|twitter|medium|dev\.?to)?\s*"
    r"(?:post|tweet|thread|version|draft)\s*(?:\([^)]*\))?\s*[:\-–]\s*",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    """Strip the wrappers models sometimes add around the actual post."""
    out = (text or "").strip()
    out = _FENCE_RE.sub("", out).strip()
    out = _LABEL_RE.sub("", out)
    out = _PREFIX_LABEL_RE.sub("", out)

    # Unwrap a fully quoted reply, then collapse redundant blank lines.
    if len(out) > 1 and out[0] in "\"'“" and out[-1] in "\"'”":
        out = out[1:-1]
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _extract_text(response, model: str) -> str:
    """Pull the post text out of a Gemini response, or raise a safe error."""
    blocked = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
    if blocked:
        raise InvalidResponseError(f"prompt blocked by Gemini (block_reason={blocked})")

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        raise InvalidResponseError(f"Gemini returned no candidates (model={model})")

    finish_reason = getattr(candidates[0], "finish_reason", None)
    text = _clean(getattr(response, "text", None) or "")
    if not text:
        raise InvalidResponseError(
            f"Gemini returned empty text (model={model}, finish_reason={finish_reason})"
        )
    if str(finish_reason).endswith("MAX_TOKENS"):
        # A post cut off mid-sentence is worse than an error message.
        raise InvalidResponseError(
            f"Gemini response hit the token limit (model={model})"
        )
    return text


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

_client = None


def get_api_key() -> str:
    key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not key:
        raise MissingApiKeyError("GEMINI_API_KEY is not set in the environment")
    return key


def get_model() -> str:
    return (os.getenv("GEMINI_MODEL") or "").strip() or DEFAULT_MODEL


def get_client():
    """Return a cached GenAI client, or raise MissingApiKeyError.

    Retry options are set explicitly because the SDK does not retry by
    default. Gemini occasionally answers 429/503 under load, and retrying a
    couple of times turns most of those into a normal response.
    """
    global _client
    if _client is None:
        http_options = types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=RETRY_ATTEMPTS)
        )
        _client = genai.Client(api_key=get_api_key(), http_options=http_options)
    return _client


def reset_client() -> None:
    """Drop the cached client. Used when settings change during testing."""
    global _client
    _client = None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _config(temperature: float) -> types.GenerateContentConfig:
    """Generation settings shared by every call.

    Thinking is switched off: these are short copywriting tasks, so the
    model's reasoning tokens would only cost time and could eat into the
    output budget and leave us with a half-written post.
    """
    return types.GenerateContentConfig(
        temperature=temperature,
        top_p=0.95,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )


def _generate(prompt: str, temperature: float) -> str:
    """Send one prompt to Gemini and return cleaned text."""
    model = get_model()

    try:
        response = get_client().models.generate_content(
            model=model, contents=prompt, config=_config(temperature)
        )
    except genai_errors.APIError as exc:
        # APIError covers 4xx/5xx from Gemini. Never surface the SDK text.
        logger.warning("Gemini API error (model=%s): %s", model, exc)
        raise GeminiError(f"Gemini API error: {exc}") from exc
    except GeminiError:
        raise
    except Exception as exc:  # timeouts, connection resets, malformed SDK data
        logger.warning("Gemini call failed (model=%s): %s", model, exc)
        raise GeminiError(f"Gemini call failed: {exc}") from exc

    return _extract_text(response, model)


def generate_platform_post(content: str, platform_id: str, tone_id: str, variant: int = 0) -> str:
    """Generate one platform-optimised post for the user's idea.

    X output is verified against the 280-character limit; if Gemini overshoots
    we ask it once to rewrite shorter and, if that still fails, we raise
    TooLongForPlatformError instead of returning invalid content.
    """
    post = _generate(
        build_generation_prompt(content, platform_id, tone_id, variant),
        temperature=0.9 if not variant else 1.0,
    )

    if platform_id == "x" and len(post) > X_CHAR_LIMIT:
        logger.info(
            "X post came back at %d chars, asking Gemini to shorten it",
            len(post),
        )
        shorter = _generate(
            X_REWRITE_PROMPT.format(limit=X_CHAR_LIMIT, post=post),
            temperature=0.4,
        )
        if len(shorter) > X_CHAR_LIMIT:
            logger.warning(
                "X post still too long after rewrite (%d chars)", len(shorter)
            )
            raise TooLongForPlatformError(
                f"X post is {len(shorter)} characters, limit is {X_CHAR_LIMIT}"
            )
        post = shorter

    return post