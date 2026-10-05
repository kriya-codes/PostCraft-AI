"""
PostCraft AI — Flask application.

The frontend talks to this backend over JSON.

Routes
------
GET  /               -> renders templates/index.html
GET  /api/health     -> liveness probe
POST /api/generate   -> generates one post per selected platform

/api/generate calls the real Gemini API through `gemini_service`. The key is
read from GEMINI_API_KEY (loaded from .env with python-dotenv) and stays on
the server: it is never sent to the browser, and neither is the model name.
All Gemini logic (prompts, retries, SDK errors) lives in gemini_service.py.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template, request

import gemini_service
from gemini_service import GeminiError

try:  # python-dotenv is already a project dependency
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - never block startup on env loading
    pass

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


app = Flask(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_CONTENT_CHARS = int(os.getenv("MAX_CONTENT_CHARS", "5000"))
MIN_CONTENT_CHARS = 1

PLATFORMS = {
    "linkedin": {
        "id": "linkedin",
        "name": "LinkedIn",
        "limit": 3000,
        "aliases": ("linkedin", "linked-in", "li"),
    },
    "x": {
        "id": "x",
        "name": "X",
        "limit": 280,
        "aliases": ("x", "twitter", "tweet"),
    },
    "medium": {
        "id": "medium",
        "name": "Dev.to / Medium",
        "limit": None,
        "aliases": ("medium", "devto", "dev.to", "dev", "blog"),
    },
}

PLATFORM_ALIASES = {
    alias: cfg["id"] for cfg in PLATFORMS.values() for alias in cfg["aliases"]
}

# `id` is what the API accepts; `label` is what the frontend displays on the
# result cards. The actual tone wording lives in gemini_service.TONE_RULES.
TONES = {
    "professional": {"id": "professional", "label": "Professional"},
    "casual": {"id": "casual", "label": "Casual"},
    "friendly": {"id": "friendly", "label": "Friendly"},
    "storytelling": {"id": "storytelling", "label": "Storytelling"},
    "educational": {"id": "educational", "label": "Educational"},
    "motivational": {"id": "motivational", "label": "Motivational"},
}

DEFAULT_TONE = "professional"


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------


def json_error(
    message: str,
    status: int,
    code: str = None,
    field: str = None,
    extra: dict = None,
):
    payload = {"success": False, "error": message}
    if code:
        payload["code"] = code
    if field:
        payload["field"] = field
    if extra:
        payload.update(extra)
    response = jsonify(payload)
    response.status_code = status
    return response


def read_json_body():
    """Return (payload, error_response)."""
    mimetype = (request.mimetype or "").lower()
    if mimetype and "json" not in mimetype:
        return None, json_error(
            "Send this request as application/json.", 415, "unsupported_media_type"
        )

    payload = request.get_json(silent=True)
    if payload is None:
        return None, json_error(
            "Request body must be valid JSON.", 400, "invalid_json"
        )
    if not isinstance(payload, dict):
        return None, json_error(
            "Request body must be a JSON object.", 400, "invalid_json"
        )
    return payload, None


def normalise_platforms(raw) -> tuple[list[str], list[str]]:
    """Map incoming platform names to canonical ids.

    Returns (ids, unknown_names).
    """
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return [], []

    ids: list[str] = []
    unknown: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            unknown.append(str(item))
            continue
        key = re.sub(r"\s+", "", item.strip().lower())
        platform_id = PLATFORM_ALIASES.get(key)
        if platform_id and platform_id not in ids:
            ids.append(platform_id)
        elif not platform_id:
            unknown.append(item.strip())
    return ids, unknown


def normalise_tone(raw) -> str:
    if not isinstance(raw, str):
        return DEFAULT_TONE
    return raw.strip().lower().replace(" ", "_").replace("-", "_") or DEFAULT_TONE


def read_idea(payload):
    """Validate the user's idea. Returns (content, error_response)."""
    content = payload.get("content", payload.get("idea"))
    if content is None:
        content = ""

    if not isinstance(content, str):
        return None, json_error(
            "Content must be text.", 400, "invalid_content", "content"
        )

    content = content.strip()
    if len(content) < MIN_CONTENT_CHARS:
        return None, json_error(
            "Tell us what you want to post about.",
            400,
            "content_required",
            "content",
        )

    if len(content) > MAX_CONTENT_CHARS:
        return None, json_error(
            f"Content is too long. Keep it under {MAX_CONTENT_CHARS} characters.",
            413,
            "content_too_long",
            "content",
        )
    return content, None


def read_tone(payload):
    """Validate the tone.

    Returns (tone_id, tone, was_unknown, raw_tone). An unknown tone is not an
    error: we fall back to the default tone and tell the frontend about it.
    """
    raw = payload.get("tone")
    tone = TONES.get(normalise_tone(raw))
    if tone is None:
        return DEFAULT_TONE, TONES[DEFAULT_TONE], True, raw
    return tone["id"], tone, False, raw


def read_variant(payload) -> int:
    """How many times this post has been regenerated before."""
    try:
        variant = int(payload.get("variant", 0))
    except (TypeError, ValueError):
        variant = 0
    return max(0, variant)


def read_single_platform(payload):
    """Validate the one platform a regenerate request targets.

    Accepts "platform": "linkedin", or a one-item "platforms": [...]. Returns
    (platform_id, error_response).
    """
    raw = payload.get("platform")
    if raw is None:
        raw = payload.get("platforms")

    platform_ids, unknown = normalise_platforms(raw)
    if unknown:
        supported = ", ".join(cfg["name"] for cfg in PLATFORMS.values())
        return None, json_error(
            f"Unsupported platform(s): {', '.join(unknown)}. Supported: {supported}.",
            400,
            "invalid_platform",
            "platform",
        )
    if not platform_ids:
        return None, json_error(
            "Tell us which platform to regenerate.",
            400,
            "platform_required",
            "platform",
        )
    if len(platform_ids) > 1:
        return None, json_error(
            "Regenerate works on one platform at a time.",
            400,
            "single_platform_required",
            "platform",
        )
    return platform_ids[0], None


def build_result(platform_id: str, text: str, variant: int) -> dict:
    """A successful result. The count is measured here, never asked of Gemini."""
    cfg = PLATFORMS[platform_id]
    return {
        "platform": platform_id,
        "name": cfg["name"],
        "success": True,
        "content": text,
        "count": len(text),
        "word_count": len(text.split()),
        "limit": cfg["limit"],
        "within_limit": cfg["limit"] is None or len(text) <= cfg["limit"],
        "variant": variant,
    }


def build_failed_result(platform_id: str, error: GeminiError, variant: int) -> dict:
    """A failed result for one platform, so the other platforms still show up.

    Only the short, safe user_message travels to the browser.
    """
    cfg = PLATFORMS[platform_id]
    return {
        "platform": platform_id,
        "name": cfg["name"],
        "success": False,
        "error": error.user_message,
        "code": error.code,
        "limit": cfg["limit"],
        "variant": variant,
    }


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log_gemini_error(error: GeminiError) -> None:
    """Record the technical detail on the server, never in the response."""
    log = logger.error if error.status >= 500 else logger.warning
    log("Gemini generation failed [%s]: %s", error.code, error.detail)


def _gemini_error_response(error: GeminiError, result: dict = None):
    """Turn a GeminiError into a safe JSON error for the frontend.

    The technical detail is logged server-side; only the short, safe
    user_message ever reaches the browser. When a `result` is given it rides
    along in the body so the frontend can render the failed card itself.
    """
    _log_gemini_error(error)
    extra = {"result": result} if result is not None else None
    return json_error(error.user_message, error.status, error.code, extra=extra)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def api_health():
    # Reports whether the server can see a key, never the key or model itself.
    return jsonify(
        {
            "success": True,
            "status": "ok",
            "gemini_configured": bool((os.getenv("GEMINI_API_KEY") or "").strip()),
            "time": _iso_now(),
        }
    )


@app.route("/api/generate", methods=["POST"])
def api_generate():
    payload, error = read_json_body()
    if error:
        return error

    # --- content -----------------------------------------------------------
    content, error = read_idea(payload)
    if error:
        return error

    # --- platforms ---------------------------------------------------------
    if "platforms" not in payload or payload.get("platforms") is None:
        return json_error(
            "Pick at least one platform to post to.",
            400,
            "platform_required",
            "platforms",
        )

    platform_ids, unknown = normalise_platforms(payload.get("platforms"))
    if unknown:
        supported = ", ".join(cfg["name"] for cfg in PLATFORMS.values())
        return json_error(
            f"Unsupported platform(s): {', '.join(unknown)}. Supported: {supported}.",
            400,
            "invalid_platform",
            "platforms",
        )
    if not platform_ids:
        return json_error(
            "Pick at least one platform to post to.",
            400,
            "platform_required",
            "platforms",
        )

    # --- tone (optional, has a sensible default) ---------------------------
    tone_id, tone, tone_unknown, raw_tone = read_tone(payload)
    variant = read_variant(payload)

    # --- generate ----------------------------------------------------------
    # Each platform is generated on its own. If one of them fails the others
    # are still returned, so a single Gemini hiccup never wipes out posts that
    # were written successfully.
    results = []
    first_error = None

    for platform_id in platform_ids:
        try:
            text = gemini_service.generate_platform_post(
                content, platform_id, tone_id, variant
            )
            results.append(build_result(platform_id, text, variant))
        except GeminiError as exc:
            _log_gemini_error(exc)
            first_error = first_error or exc
            results.append(build_failed_result(platform_id, exc, variant))
        except Exception:
            logger.exception("Unexpected error while generating the %s post", platform_id)
            exc = GeminiError(f"Unexpected error on platform '{platform_id}'")
            first_error = first_error or exc
            results.append(build_failed_result(platform_id, exc, variant))

    # Nothing worked at all: this is a real failure of the whole request.
    if not any(result["success"] for result in results):
        return json_error(
            first_error.user_message, first_error.status, first_error.code
        )

    body = {
        "success": True,
        "tone": tone_id,
        "tone_label": tone["label"],
        "platforms": platform_ids,
        "results": results,
        "generated_at": _iso_now(),
    }
    if tone_unknown:
        body["warning"] = (
            f"Unknown tone '{raw_tone}'. Fell back to {tone['label']}."
        )
    return jsonify(body), 200


@app.route("/api/regenerate", methods=["POST"])
def api_regenerate():
    """Rewrite one platform's post, leaving every other card untouched."""
    payload, error = read_json_body()
    if error:
        return error

    content, error = read_idea(payload)
    if error:
        return error

    platform_id, error = read_single_platform(payload)
    if error:
        return error

    tone_id, tone, tone_unknown, raw_tone = read_tone(payload)
    variant = read_variant(payload)

    try:
        text = gemini_service.generate_platform_post(
            content, platform_id, tone_id, variant
        )
    except GeminiError as exc:
        return _gemini_error_response(
            exc, result=build_failed_result(platform_id, exc, variant)
        )
    except Exception:
        logger.exception("Unexpected error while regenerating the %s post", platform_id)
        exc = GeminiError(f"Unexpected error on platform '{platform_id}'")
        return _gemini_error_response(
            exc, result=build_failed_result(platform_id, exc, variant)
        )

    body = {
        "success": True,
        "tone": tone_id,
        "tone_label": tone["label"],
        "platform": platform_id,
        "result": build_result(platform_id, text, variant),
        "generated_at": _iso_now(),
    }
    if tone_unknown:
        body["warning"] = f"Unknown tone '{raw_tone}'. Fell back to {tone['label']}."
    return jsonify(body), 200


# ---------------------------------------------------------------------------
# JSON error handling for the API surface
# ---------------------------------------------------------------------------


@app.errorhandler(404)
def handle_404(error):
    if request.path.startswith("/api/"):
        return json_error("Endpoint not found.", 404, "not_found")
    return error


@app.errorhandler(405)
def handle_405(error):
    if request.path.startswith("/api/"):
        return json_error(
            "Method not allowed for this endpoint.", 405, "method_not_allowed"
        )
    return error


@app.errorhandler(500)
def handle_500(error):  # pragma: no cover - defensive
    if request.path.startswith("/api/"):
        return json_error(
            "Something went wrong on our side. Please try again.", 500, "server_error"
        )
    return error


if __name__ == "__main__":
    app.run(debug=True)