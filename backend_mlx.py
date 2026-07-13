"""Apple Silicon Gemma 4 E2B-it backend using MLX-VLM.

The model is downloaded on first use and runs locally through Apple's Metal
backend.  No network inference service or fallback backend is used.
"""

from __future__ import annotations

import logging
import os
import platform
import threading
import time
from pathlib import Path

from backend_vlm import PROMPT, THINKING_PROMPT, _extract_markdown, _to_markdown_math

log = logging.getLogger("kevintex")

MODEL_ID = os.environ.get(
    "LOCALTEX_MLX_MODEL", "mlx-community/gemma-4-e2b-it-4bit"
)
EXPECTED_BYTES = 3_550_000_000
MAX_TOKENS_FAST = 1024
MAX_TOKENS_THINKING = 8192


def _default_models_dir() -> Path:
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Application Support" / "KevinTex" / "models"
    return Path.home() / ".local" / "share" / "KevinTex" / "models"


MODELS_DIR = Path(os.environ.get("LOCALTEX_MODELS_DIR", _default_models_dir()))
MODEL_DIR = MODELS_DIR / "gemma-4-e2b-it-mlx-4bit"

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


def _require_apple_silicon() -> None:
    machine = platform.machine().lower()
    if platform.system() != "Darwin" or machine not in {"arm64", "aarch64"}:
        raise RuntimeError(
            "The MLX backend requires macOS on Apple Silicon (arm64); "
            f"detected {platform.system()} {platform.machine()}."
        )


def _download_size() -> int:
    total = 0
    if MODEL_DIR.exists():
        for path in MODEL_DIR.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                pass
    return total


def _ensure_model() -> Path:
    config = MODEL_DIR / "config.json"
    weights = list(MODEL_DIR.glob("*.safetensors")) if MODEL_DIR.exists() else []
    if config.is_file() and weights:
        return MODEL_DIR

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    set_status("downloading", 0, "Downloading MLX model (one time, ~3.6 GB)…")
    stop = threading.Event()

    def monitor() -> None:
        while not stop.wait(1.0):
            progress = min(99, int(_download_size() / EXPECTED_BYTES * 100))
            set_status(
                "downloading", progress, f"Downloading MLX model… {progress}%"
            )

    thread = threading.Thread(target=monitor, name="mlx-download-progress", daemon=True)
    thread.start()
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=MODEL_DIR,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.model",
                "*.txt",
                "*.jinja",
            ],
        )
    finally:
        stop.set()
        thread.join(timeout=2)

    if not config.is_file() or not list(MODEL_DIR.glob("*.safetensors")):
        raise RuntimeError(f"MLX model download is incomplete in {MODEL_DIR}")
    return MODEL_DIR


def _friendly_load_error(exc: Exception) -> RuntimeError:
    message = str(exc)
    lowered = message.lower()
    if any(
        marker in lowered
        for marker in (
            "unsupported",
            "model type",
            "architecture",
            "gemma4",
            "module not found",
        )
    ):
        return RuntimeError(
            "This mlx-vlm installation does not support the Gemma 4 "
            f"architecture used by {MODEL_ID}. Install a current mlx-vlm build. "
            f"Original error: {message}"
        )
    return RuntimeError(f"Could not load local MLX model {MODEL_ID}: {message}")


class MLXGemmaVisionBackend:
    """MLX-VLM backend with the same callable interface as the other backends."""

    def __init__(self, thinking: bool | None = None):
        _require_apple_silicon()
        self.default_thinking = bool(thinking) if thinking is not None else False
        model_path = _ensure_model()
        set_status("loading", 0, "Loading model with MLX/Metal…")
        started = time.monotonic()
        try:
            from mlx_vlm import generate as mlx_generate
            from mlx_vlm import load as mlx_load
            from mlx_vlm.prompt_utils import apply_chat_template

            self.model, self.processor = mlx_load(str(model_path))
            self._generate = mlx_generate
            self._apply_chat_template = apply_chat_template
        except Exception as exc:
            raise _friendly_load_error(exc) from exc
        log.info("MLX Gemma loaded in %.1fs", time.monotonic() - started)
        set_status("ready", 100, "Ready (Apple Silicon MLX)")

    def __call__(self, img, thinking: bool | None = None) -> str:
        return self.recognize(img, thinking=thinking)

    def recognize(self, img, thinking: bool | None = None) -> str:
        think = self.default_thinking if thinking is None else bool(thinking)
        prompt_text = THINKING_PROMPT if think else PROMPT
        formatted = self._apply_chat_template(
            self.processor,
            self.model.config,
            prompt_text,
            num_images=1,
            chat_template_kwargs={"enable_thinking": think},
        )
        image = img.convert("RGB") if getattr(img, "mode", None) != "RGB" else img
        owns_image = image is not img
        started = time.monotonic()
        try:
            result = self._generate(
                model=self.model,
                processor=self.processor,
                prompt=formatted,
                image=[image],
                max_tokens=MAX_TOKENS_THINKING if think else MAX_TOKENS_FAST,
                temperature=0.1,
                top_p=0.95,
                top_k=64,
                repetition_penalty=1.05,
                verbose=False,
            )
        finally:
            if owns_image:
                image.close()
        text = result.text if hasattr(result, "text") else str(result)
        log.info(
            "MLX Gemma generated in %.2fs (thinking=%s)",
            time.monotonic() - started,
            think,
        )
        return _extract_markdown(text) if think else _to_markdown_math(text)


def load():
    try:
        return MLXGemmaVisionBackend()
    except Exception as exc:
        set_status("error", 0, "MLX model load failed", error=str(exc))
        raise
