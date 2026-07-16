"""Tests for the hosted Gemma provider and provider selection state."""

import asyncio
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import app
import backend_gemma_cloud
import provider_config


class ProviderConfigTests(unittest.TestCase):
    def test_api_key_is_persisted_without_being_returned_by_provider_state(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "provider.json"
            with patch.dict(os.environ, {"LOCALTEX_PROVIDER_CONFIG": str(config)}, clear=False):
                provider_config.save_config("cloud", "AIza-test-key")
                saved = provider_config.read_config()
                self.assertEqual(saved["mode"], "cloud")
                self.assertEqual(saved["api_key"], "AIza-test-key")
                if os.name != "nt":
                    self.assertEqual(stat.S_IMODE(config.stat().st_mode), 0o600)

                with patch.object(app, "BACKEND", "gemma"):
                    state = app._provider_state()
                self.assertTrue(state["api_key_configured"])
                self.assertNotIn("api_key", state)

    def test_first_run_requires_a_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "provider.json"
            with patch.dict(
                os.environ,
                {"LOCALTEX_PROVIDER_CONFIG": str(config), "GEMINI_API_KEY": ""},
                clear=False,
            ), patch.object(app, "BACKEND", "gemma"):
                self.assertTrue(app._setup_required())
                self.assertEqual(app.status()["phase"], "setup")

    def test_provider_switch_is_rejected_during_inference(self):
        class Request:
            async def json(self):
                return {"mode": "local"}

        acquired = []
        try:
            for _ in range(app.INFERENCE_CONCURRENCY):
                self.assertTrue(app._inference_slots.acquire(blocking=False))
                acquired.append(True)
            with patch.object(app, "BACKEND", "gemma"):
                response = asyncio.run(app.configure_provider(Request()))
            self.assertEqual(response.status_code, 409)
            self.assertIn("recognition", response.body.decode())
        finally:
            for _ in acquired:
                app._inference_slots.release()

    def test_backend_cleanup_calls_close(self):
        class Backend:
            closed = False

            def close(self):
                self.closed = True

        backend = Backend()
        app._dispose_model(backend)
        self.assertTrue(backend.closed)


class CloudBackendTests(unittest.TestCase):
    def test_image_request_uses_gemma_api_and_keeps_image_before_prompt(self):
        calls = {}

        class FakePart:
            @classmethod
            def from_bytes(cls, **kwargs):
                calls["image"] = kwargs
                return "image-part"

        class FakeThinkingConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeGenerateContentConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        Types = type(
            "Types",
            (),
            {
                "Part": FakePart,
                "ThinkingConfig": FakeThinkingConfig,
                "GenerateContentConfig": FakeGenerateContentConfig,
            },
        )

        class Models:
            def generate_content(self, **kwargs):
                calls["request"] = kwargs
                return type("Response", (), {"text": "LATEX: $x^2$"})()

        backend = backend_gemma_cloud.GemmaCloudBackend.__new__(
            backend_gemma_cloud.GemmaCloudBackend
        )
        backend.types = Types
        backend.client = type("Client", (), {"models": Models()})()
        backend.default_thinking = False
        import threading

        backend._inference_lock = threading.Lock()

        image = Image.new("RGB", (3, 3), "white")
        try:
            self.assertEqual(backend.recognize(image), "$x^2$")
        finally:
            image.close()

        self.assertEqual(calls["request"]["model"], "gemma-4-26b-a4b-it")
        self.assertEqual(calls["request"]["contents"], ["image-part", backend_gemma_cloud.PROMPT])
        self.assertEqual(calls["image"]["mime_type"], "image/png")

    def test_audio_request_uses_gemini_inline_wav_input(self):
        calls = {}

        class FakePart:
            @classmethod
            def from_bytes(cls, **kwargs):
                calls["audio"] = kwargs
                return "audio-part"

        class FakeThinkingConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeGenerateContentConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        Types = type(
            "Types",
            (),
            {
                "Part": FakePart,
                "ThinkingConfig": FakeThinkingConfig,
                "GenerateContentConfig": FakeGenerateContentConfig,
            },
        )

        class Models:
            def generate_content(self, **kwargs):
                calls["request"] = kwargs
                return type("Response", (), {"text": "LATEX: $\\frac{1}{2}$"})()

        backend = backend_gemma_cloud.GemmaCloudBackend.__new__(
            backend_gemma_cloud.GemmaCloudBackend
        )
        backend.types = Types
        backend.client = type("Client", (), {"models": Models()})()
        backend.default_thinking = False
        import threading

        backend._inference_lock = threading.Lock()

        with tempfile.NamedTemporaryFile(suffix=".wav") as audio:
            audio.write(b"RIFF" + b"\0" * 36 + b"WAVEaudio")
            audio.flush()
            with patch.object(
                backend_gemma_cloud, "AUDIO_MODEL_ID", "gemini-audio-test"
            ):
                self.assertEqual(backend.recognize_audio(audio.name), "$\\frac{1}{2}$")

        self.assertEqual(calls["request"]["model"], "gemini-audio-test")
        self.assertEqual(
            calls["request"]["contents"],
            [backend_gemma_cloud.AUDIO_PROMPT, "audio-part"],
        )
        self.assertEqual(calls["audio"]["mime_type"], "audio/wav")
        self.assertTrue(calls["audio"]["data"].startswith(b"RIFF"))

    def test_oversized_audio_is_rejected_before_reading(self):
        backend = backend_gemma_cloud.GemmaCloudBackend.__new__(
            backend_gemma_cloud.GemmaCloudBackend
        )
        backend.default_thinking = False
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio:
            audio.write(b"12345")
            audio.flush()
            with patch.object(backend_gemma_cloud, "MAX_INLINE_AUDIO_BYTES", 4):
                with self.assertRaisesRegex(RuntimeError, "14 MiB maximum"):
                    backend.recognize_audio(audio.name)


if __name__ == "__main__":
    unittest.main()
