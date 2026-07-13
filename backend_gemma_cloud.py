"""Google AI Studio / Gemini API backend for hosted Gemma 4."""

from __future__ import annotations

import io
import logging
import os
import threading
import time

from backend_vlm import (
    PROMPT,
    THINKING_PROMPT,
    _extract_markdown,
)


log = logging.getLogger("kevintex")

# Google currently exposes these Gemma 4 models through the Gemini API.  The
# A4B model is the smaller hosted option and supports image input for OCR.
MODEL_ID = os.environ.get("LOCALTEX_GEMMA_CLOUD_MODEL", "gemma-4-26b-a4b-it")
MAX_TOKENS_FAST = 1024
MAX_TOKENS_THINKING = 3072
REQUEST_TIMEOUT_MS = 120_000

_STATUS = {"phase": "init", "progress": 0, "message": "Starting…", "error": None}
_status_lock = threading.Lock()


def set_status(
    phase: str,
    progress: int = 0,
    message: str = "",
    error: str | None = None,
) -> None:
    with _status_lock:
        _STATUS["phase"] = phase
        _STATUS["progress"] = int(progress)
        _STATUS["message"] = message or _STATUS.get("message", "")
        _STATUS["error"] = error


def get_status() -> dict:
    with _status_lock:
        return dict(_STATUS)


def _response_text(response) -> str:
    """Extract text while tolerating SDK response objects and test doubles."""
    try:
        text = response.text
    except (AttributeError, ValueError):
        text = None
    if isinstance(text, str) and text.strip():
        return text

    for candidate in getattr(response, "candidates", ()) or ():
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", ()) or ():
            value = getattr(part, "text", None)
            if isinstance(value, str) and value.strip():
                return value
    return ""


class GemmaCloudBackend:
    """Google AI Studio-backed Gemma 4 image OCR backend."""

    def __init__(self, api_key: str | None = None):
        api_key = (api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
        if not api_key:
            raise RuntimeError(
                "Google AI Studio API key is missing. Add one in Settings."
            )

        set_status("loading", 0, "Connecting to Google AI Studio…")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "The Google Gen AI SDK is not installed. Install the "
                "google-genai package and restart KevinTex."
            ) from exc

        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
        )
        self.types = types
        self.default_thinking = False
        self._inference_lock = threading.Lock()
        set_status("ready", 100, "Ready")

    def __call__(self, img, thinking: bool | None = None) -> str:
        return self.recognize(img, thinking=thinking)

    def close(self) -> None:
        """Close the SDK's pooled HTTP connections."""
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    def recognize(self, img, thinking: bool | None = None) -> str:
        """Recognize a formula image through the hosted Gemma 4 model."""
        think = self.default_thinking if thinking is None else bool(thinking)
        prompt = THINKING_PROMPT if think else PROMPT
        max_tokens = MAX_TOKENS_THINKING if think else MAX_TOKENS_FAST

        with io.BytesIO() as buffer:
            img.save(buffer, format="PNG")
            image_part = self.types.Part.from_bytes(
                data=buffer.getvalue(), mime_type="image/png"
            )

        config = self.types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=0.05,
            top_p=0.9,
            thinking_config=self.types.ThinkingConfig(
                thinking_level="high" if think else "minimal"
            ),
        )
        started = time.perf_counter()
        with self._inference_lock:
            response = self.client.models.generate_content(
                model=MODEL_ID,
                # Image before text follows Gemma's multimodal prompt guidance.
                contents=[image_part, prompt],
                config=config,
            )
        elapsed = time.perf_counter() - started
        content = _response_text(response)
        if not content.strip():
            raise RuntimeError(
                "Google AI Studio returned an empty response. Check the API key, "
                "model access, and safety settings, then try again."
            )
        log.info(
            "Cloud Gemma generated %d chars in %.2fs (thinking=%s)",
            len(content), elapsed, think,
        )
        return _extract_markdown(content)

    # The hosted 26B and 31B Gemma API models are image-capable, but not the
    # audio-capable E2B/E4B/12B variants.  Keep the existing voice button
    # hidden while this backend is active.
    def recognize_audio(self, audio_path: str, thinking: bool | None = None) -> str:
        raise RuntimeError(
            "Voice-to-LaTeX is available with local Gemma or MLX, not the "
            "hosted Gemma 4 model."
        )


def load(api_key: str | None = None) -> GemmaCloudBackend:
    try:
        return GemmaCloudBackend(api_key=api_key)
    except Exception as exc:
        set_status("error", 0, "Cloud model setup failed", error=str(exc))
        raise
