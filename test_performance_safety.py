"""Concurrency, backpressure, and resource-limit tests for KevinTex."""

import asyncio
import io
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from PIL import Image
from starlette.datastructures import UploadFile

import app
import backend_gemma
import backend_mlx
from pathlib import Path


class ModelInitializationTests(unittest.TestCase):
    def test_generic_backend_load_error_reaches_status(self):
        previous_backend = app.BACKEND
        previous_model = app._model
        previous_error = app._model_load_error
        app.BACKEND = "pix2tex"
        app._model = None
        app._model_load_error = "missing optional dependency"
        try:
            result = app.status()
        finally:
            app.BACKEND = previous_backend
            app._model = previous_model
            app._model_load_error = previous_error
        self.assertEqual(result["phase"], "error")
        self.assertIn("missing optional dependency", result["error"])

    def test_concurrent_get_model_initializes_once(self):
        sentinel = object()
        calls = 0
        calls_lock = threading.Lock()

        def fake_load():
            nonlocal calls
            with calls_lock:
                calls += 1
            time.sleep(0.03)
            return sentinel

        previous_model = app._model
        previous_backend = app.BACKEND
        app._model = None
        app.BACKEND = "gemma"
        try:
            with patch.object(backend_gemma, "load", side_effect=fake_load):
                with ThreadPoolExecutor(max_workers=8) as pool:
                    results = list(pool.map(lambda _: app.get_model(), range(8)))
            self.assertEqual(calls, 1)
            self.assertTrue(all(result is sentinel for result in results))
        finally:
            app._model = previous_model
            app.BACKEND = previous_backend

    def test_mlx_backend_is_selected_directly(self):
        sentinel = object()
        previous_model = app._model
        previous_backend = app.BACKEND
        app._model = None
        app.BACKEND = "mlx"
        try:
            with patch.object(backend_mlx, "load", return_value=sentinel) as load:
                self.assertIs(app.get_model(), sentinel)
            load.assert_called_once_with()
            self.assertEqual(app._device_label(), "metal")
            self.assertIs(app._backend_status_module(), backend_mlx)
        finally:
            app._model = previous_model
            app.BACKEND = previous_backend

    def test_optiq_model_skips_audio_repair(self):
        with patch.object(
            backend_mlx,
            "MODEL_ID",
            "mlx-community/gemma-4-e2b-it-OptiQ-4bit",
        ):
            self.assertFalse(backend_mlx._repair_audio_tower_weights(Path("/tmp/unused")))

    def test_bfloat16_loader_error_is_actionable(self):
        exc = backend_mlx._friendly_load_error(
            RuntimeError("data type 'bfloat16' not understood")
        )
        self.assertIn("OptiQ", str(exc))
        self.assertIn("bfloat16", str(exc).lower())


class InferenceConcurrencyTests(unittest.TestCase):
    def test_mlx_audio_uses_documented_multimodal_arguments(self):
        class Result:
            text = "LATEX: $x^2 + y^2$"

        calls = {}
        backend = backend_mlx.MLXGemmaVisionBackend.__new__(backend_mlx.MLXGemmaVisionBackend)
        backend.default_thinking = False
        backend.model = type("Model", (), {"config": object()})()
        backend.processor = object()

        def template(*args, **kwargs):
            calls["template"] = (args, kwargs)
            return "formatted"

        def generate(**kwargs):
            calls["generate"] = kwargs
            return Result()

        backend._apply_chat_template = template
        backend._generate = generate
        self.assertEqual(backend.recognize_audio("/tmp/formula.wav"), "$x^2 + y^2$")
        self.assertEqual(calls["template"][1]["num_audios"], 1)
        self.assertEqual(calls["generate"]["audio"], ["/tmp/formula.wav"])

    def test_gemma_context_is_never_entered_concurrently(self):
        active = 0
        max_active = 0
        state_lock = threading.Lock()

        class FakeLlama:
            def create_chat_completion(self, **_kwargs):
                nonlocal active, max_active
                with state_lock:
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(0.02)
                with state_lock:
                    active -= 1
                return {
                    "choices": [{"message": {"content": "$x$"}}],
                    "usage": {"completion_tokens": 1},
                }

        backend = backend_gemma.GemmaVisionBackend.__new__(
            backend_gemma.GemmaVisionBackend
        )
        backend.default_thinking = False
        backend._inference_lock = threading.Lock()
        backend.llm = FakeLlama()
        image = Image.new("RGB", (12, 12), "white")
        try:
            with ThreadPoolExecutor(max_workers=6) as pool:
                results = list(
                    pool.map(lambda _: backend.recognize(image, thinking=False), range(6))
                )
        finally:
            image.close()
        self.assertEqual(max_active, 1)
        self.assertEqual(results, ["$x$"] * 6)

    def test_application_backpressure_rejects_when_slot_is_occupied(self):
        previous_timeout = app.INFERENCE_QUEUE_TIMEOUT
        app.INFERENCE_QUEUE_TIMEOUT = 0.0
        acquired = app._inference_slots.acquire(timeout=0)
        self.assertTrue(acquired)
        try:
            with self.assertRaises(app.InferenceBusyError):
                app._run_model(Image.new("RGB", (1, 1)), False)
        finally:
            app._inference_slots.release()
            app.INFERENCE_QUEUE_TIMEOUT = previous_timeout


class UploadLimitTests(unittest.TestCase):
    def test_content_length_is_rejected_before_multipart_parsing(self):
        class Request:
            headers = {"content-length": str(app.MAX_MULTIPART_BYTES + 1)}

            class url:
                path = "/api/convert"

        async def should_not_run(_request):
            raise AssertionError("oversized body reached multipart parser")

        response = asyncio.run(
            app.limit_upload_content_length(Request(), should_not_run)
        )
        self.assertEqual(response.status_code, 413)

    def test_convert_rejects_oversized_upload_before_inference(self):
        previous_limit = app.MAX_UPLOAD_BYTES
        app.MAX_UPLOAD_BYTES = 8
        upload = UploadFile(io.BytesIO(b"x" * 9), filename="large.png")
        try:
            response = asyncio.run(
                app.convert(
                    upload,
                    thinking=False,
                    preprocess="auto",
                    rotation=0,
                    invert=False,
                )
            )
        finally:
            app.MAX_UPLOAD_BYTES = previous_limit
        self.assertEqual(response.status_code, 413)
        self.assertIn("too large", response.body.decode())

    def test_reader_accepts_exact_limit(self):
        previous_limit = app.MAX_UPLOAD_BYTES
        app.MAX_UPLOAD_BYTES = 8
        upload = UploadFile(io.BytesIO(b"x" * 8), filename="exact.bin")
        try:
            raw = asyncio.run(app._read_upload(upload))
        finally:
            app.MAX_UPLOAD_BYTES = previous_limit
        self.assertEqual(raw, b"x" * 8)

    def test_pixel_limit_is_checked_before_decode(self):
        class OversizedHeader:
            width = app.MAX_INPUT_PIXELS + 1
            height = 1

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def load(self):
                raise AssertionError("oversized image should not be decoded")

        with patch.object(app.Image, "open", return_value=OversizedHeader()):
            with self.assertRaisesRegex(ValueError, "50 megapixels"):
                app._open_image(b"header-only")

    def test_voice_is_rejected_on_backend_without_audio(self):
        previous_backend = app.BACKEND
        app.BACKEND = "gemma"
        upload = UploadFile(io.BytesIO(b"not audio"), filename="voice.wav")
        try:
            response = asyncio.run(app.voice(upload, thinking=False))
        finally:
            app.BACKEND = previous_backend
        self.assertEqual(response.status_code, 501)

    def test_voice_validates_wav_before_inference(self):
        previous_backend = app.BACKEND
        app.BACKEND = "mlx"
        upload = UploadFile(io.BytesIO(b"x" * 64), filename="voice.wav")
        try:
            response = asyncio.run(app.voice(upload, thinking=False))
        finally:
            app.BACKEND = previous_backend
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
