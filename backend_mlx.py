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

from backend_vlm import PROMPT, THINKING_PROMPT, _extract_markdown

log = logging.getLogger("kevintex")

MODEL_ID = os.environ.get(
    "LOCALTEX_MLX_MODEL", "mlx-community/gemma-4-e2b-it-OptiQ-4bit"
)
# Main weights (~4.3 GB) + optiq_vision sidecar (~0.95 GB) + tokenizer/config.
EXPECTED_BYTES = 5_400_000_000
MAX_TOKENS_FAST = 1024
MAX_TOKENS_THINKING = 8192
_LEGACY_MODEL_DIRS = (
    "gemma-4-e2b-it-mlx-4bit",
)


def _default_models_dir() -> Path:
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Application Support" / "KevinTex" / "models"
    return Path.home() / ".local" / "share" / "KevinTex" / "models"


MODELS_DIR = Path(os.environ.get("LOCALTEX_MODELS_DIR", _default_models_dir()))
MODEL_DIR = MODELS_DIR / "gemma-4-e2b-it-mlx-optiq-4bit"

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


def _purge_legacy_models() -> None:
    """Drop superseded MLX caches with broken audio-tower weight layouts."""
    for name in _LEGACY_MODEL_DIRS:
        legacy = MODELS_DIR / name
        if legacy == MODEL_DIR or not legacy.is_dir():
            continue
        try:
            import shutil

            shutil.rmtree(legacy)
            log.info("Removed legacy MLX model cache at %s", legacy)
        except OSError as exc:
            log.warning("Could not remove legacy MLX model cache %s: %s", legacy, exc)


def _repair_audio_tower_weights(model_dir: Path) -> bool:
    """Fix Gemma 4 audio conv weights saved in PyTorch channel-first layout."""
    # OptiQ MLX releases already ship channel-last audio conv weights. Rewriting
    # their bf16 safetensors through NumPy fails with "bfloat16 not understood".
    if "OptiQ-4bit" in MODEL_ID:
        return False

    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError:
        return False

    marker = "subsample_conv_projection"
    suffix = "conv.weight"
    repaired_any = False

    for weights_path in sorted(model_dir.rglob("*.safetensors")):
        updates: dict[str, torch.Tensor] = {}
        tensors: dict[str, torch.Tensor] = {}
        try:
            with safe_open(weights_path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    tensor = handle.get_tensor(key)
                    tensors[key] = tensor
                    if marker not in key or not key.endswith(suffix):
                        continue
                    if tensor.ndim != 4:
                        continue
                    tail = tuple(tensor.shape[1:])
                    if tail == (3, 3, 1):
                        continue
                    if tail == (1, 3, 3):
                        updates[key] = tensor.transpose(0, 2, 3, 1).contiguous()
                    elif tail == (3, 1, 3):
                        updates[key] = tensor.transpose(0, 3, 1, 2).contiguous()
        except Exception as exc:
            log.warning("Skipping audio-tower repair for %s: %s", weights_path.name, exc)
            continue
        if not updates:
            continue
        for key, value in updates.items():
            tensors[key] = value
        save_file(tensors, weights_path)
        repaired_any = True
        log.info(
            "Repaired %d audio-tower conv weight(s) in %s",
            len(updates),
            weights_path.name,
        )
    return repaired_any


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


def _model_files_ready() -> bool:
    if not (MODEL_DIR / "config.json").is_file():
        return False
    if not list(MODEL_DIR.glob("*.safetensors")):
        return False
    if "OptiQ-4bit" in MODEL_ID and not list(MODEL_DIR.glob("optiq/*.safetensors")):
        return False
    return True


def _ensure_model() -> Path:
    _purge_legacy_models()
    if _model_files_ready():
        _repair_audio_tower_weights(MODEL_DIR)
        return MODEL_DIR

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    set_status("downloading", 0, "Downloading MLX model (one time, ~5.4 GB)…")
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
                "optiq/*",
            ],
        )
    finally:
        stop.set()
        thread.join(timeout=2)

    if not _model_files_ready():
        raise RuntimeError(f"MLX model download is incomplete in {MODEL_DIR}")
    _repair_audio_tower_weights(MODEL_DIR)
    return MODEL_DIR


def _friendly_load_error(exc: Exception) -> RuntimeError:
    message = str(exc)
    lowered = message.lower()
    if "bfloat16" in lowered and "not understood" in lowered:
        return RuntimeError(
            "KevinTex hit an incompatible MLX weight loader while preparing the "
            "Gemma model cache. Update to the latest KevinTex build, then delete "
            f"{MODEL_DIR} and relaunch to re-download the OptiQ MLX weights. "
            f"Original error: {message}"
        )
    if "audio_tower" in lowered and "shape" in lowered:
        return RuntimeError(
            "The cached MLX Gemma weights have an incompatible audio-tower layout. "
            "KevinTex now uses mlx-community/gemma-4-e2b-it-OptiQ-4bit. "
            f"Delete {MODEL_DIR} and relaunch to re-download, or set LOCALTEX_MLX_MODEL "
            f"to another mlx-community Gemma 4 build. Original error: {message}"
        )
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
        from mlx_vlm import generate as mlx_generate
        from mlx_vlm import load as mlx_load
        from mlx_vlm.prompt_utils import apply_chat_template

        try:
            self.model, self.processor = mlx_load(str(model_path))
            self._generate = mlx_generate
            self._apply_chat_template = apply_chat_template
        except Exception as exc:
            if _repair_audio_tower_weights(model_path):
                try:
                    self.model, self.processor = mlx_load(str(model_path))
                    self._generate = mlx_generate
                    self._apply_chat_template = apply_chat_template
                except Exception as retry_exc:
                    raise _friendly_load_error(retry_exc) from retry_exc
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
                temperature=0.05 if think else 0.0,
                top_p=0.9 if think else 0.85,
                top_k=64,
                repetition_penalty=1.08,
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
        return _extract_markdown(text)


def load():
    try:
        return MLXGemmaVisionBackend()
    except Exception as exc:
        set_status("error", 0, "MLX model load failed", error=str(exc))
        raise
