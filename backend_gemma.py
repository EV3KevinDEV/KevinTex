"""Gemma 4 E2B-it backend: image -> Markdown+LaTeX via llama.cpp (GGUF).

Uses the `unsloth/gemma-4-E2B-it-GGUF` Q4_K_M quant plus the repo's
`mmproj-F16.gguf` vision projector, run through llama-cpp-python with CUDA
offload. Outputs the same Mathpix/SimpleTex-style Markdown+math as the LFM
backend (prose as plain text, inline math in `$...$`, display math in `$$...$$`)
and reuses the exact prompt text and `_to_markdown_math()` cleanup from
`backend_vlm`. Runs entirely on-device (GPU). No cloud.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import threading
import time

log = logging.getLogger("kevintex")

MODEL_REPO = "unsloth/gemma-4-E2B-it-GGUF"
MAIN_GGUF = "gemma-4-E2B-it-Q4_K_M.gguf"
MMPROJ_GGUF = "mmproj-F16.gguf"

# Approximate total download size (main GGUF ~3.107GB + mmproj ~0.986GB).
_EXPECTED_BYTES = 4_093_000_000

# --- Load-status reporting (read by app.py /api/status and the frontend splash) ---
_STATUS = {"phase": "init", "progress": 0, "message": "Starting…", "error": None}
_status_lock = threading.Lock()


def set_status(phase: str, progress: int = 0, message: str = "", error: str | None = None) -> None:
    with _status_lock:
        _STATUS["phase"] = phase
        _STATUS["progress"] = int(progress)
        _STATUS["message"] = message or _STATUS.get("message", "")
        if error is not None:
            _STATUS["error"] = error


def get_status() -> dict:
    with _status_lock:
        return dict(_STATUS)

# Stable local cache so weights aren't re-downloaded each run. Honors an override
# via the LOCALTEX_MODELS_DIR env var; defaults to <proj>/models/gemma-4-E2B-it.
_DEFAULT_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models"
)
MODELS_DIR = os.environ.get("LOCALTEX_MODELS_DIR", _DEFAULT_MODELS_DIR)
MODEL_DIR = os.path.join(MODELS_DIR, "gemma-4-E2B-it")

N_CTX = 16384
N_GPU_LAYERS = -1  # offload everything to the GPU
MAX_TOKENS_FAST = 1024
MAX_TOKENS_THINKING = 8192

# Reuse the exact prompt text + cleanup from the LFM backend so output style is
# identical across backends.
from backend_vlm import (  # noqa: E402
    PROMPT,
    THINKING_PROMPT,
    LATEX_SENTINEL,
    _to_markdown_math,
    _extract_markdown,
)


def _preload_cuda_libs() -> None:
    """Make torch's pip-bundled NVIDIA CUDA libs resolvable to libllama.so.

    The prebuilt cu121 `llama-cpp-python` wheel links against libcudart.so.12 /
    libcublas.so.12 etc. which aren't on the default ld path. Loading them with
    RTLD_GLOBAL before importing `llama_cpp` makes the symbols available to the
    dlopen of libllama.so. Best-effort: silently skip if torch/nvidia isn't
    present (CPU-only fallback).
    """
    import ctypes
    import glob
    import sys

    try:
        import torch  # noqa: F401  -- ensure torch's nvidia pip pkgs exist
    except Exception:
        return

    # torch >=2 ships CUDA runtime libs under site-packages/nvidia/<lib>/lib/.
    try:
        site_pkg = os.path.dirname(os.path.dirname(torch.__file__))
    except Exception:
        return
    nv_root = os.path.join(site_pkg, "nvidia")
    if not os.path.isdir(nv_root):
        return
    for lib in sorted(glob.glob(os.path.join(nv_root, "*", "lib", "*.so*"))):
        try:
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


def _ensure_model_files() -> tuple[str, str]:
    """Download the main GGUF + mmproj if missing; return their local paths."""
    main_path = os.path.join(MODEL_DIR, MAIN_GGUF)
    mmproj_path = os.path.join(MODEL_DIR, MMPROJ_GGUF)
    if os.path.exists(main_path) and os.path.exists(mmproj_path):
        return main_path, mmproj_path

    os.makedirs(MODEL_DIR, exist_ok=True)
    log.info("Downloading Gemma 4 E2B-it GGUFs into %s (one time, ~4.1GB)…", MODEL_DIR)
    set_status("downloading", 0, "Downloading model (one time, ~4.1GB)…")
    from huggingface_hub import snapshot_download

    # Background monitor: poll on-disk file sizes for a progress percentage.
    stop = threading.Event()

    def _monitor():
        while not stop.is_set():
            cur = 0
            for root, _dirs, files in os.walk(MODEL_DIR):
                for f in files:
                    try:
                        cur += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
            pct = min(100, int(cur / _EXPECTED_BYTES * 100)) if _EXPECTED_BYTES else 0
            set_status("downloading", pct, f"Downloading model… {pct}%")
            stop.wait(1.0)

    mon = threading.Thread(target=_monitor, daemon=True)
    mon.start()
    try:
        snapshot_download(
            MODEL_REPO,
            local_dir=MODEL_DIR,
            allow_patterns=[MAIN_GGUF, MMPROJ_GGUF],
        )
    finally:
        stop.set()
    if not (os.path.exists(main_path) and os.path.exists(mmproj_path)):
        raise RuntimeError(
            f"Model files missing after download: {main_path}, {mmproj_path}"
        )
    return main_path, mmproj_path


def _img_to_data_url(img) -> str:
    """PIL.Image -> base64 PNG data URL."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


class GemmaVisionBackend:
    """llama.cpp-backed Gemma 4 E2B-it multimodal backend.

    Same interface as backend_vlm.LFMVisionBackend: `recognize(img, thinking=None)`
    plus `__call__` delegating to it. `thinking=True` switches to a chain-of-
    thought prompt with a much larger token budget (Gemma 4 native thinking
    toggle isn't exposed in llama-cpp-python 0.3.34's chat API, so we use the
    CoT prompt + `LATEX:` sentinel extraction instead).
    """

    def __init__(self, thinking: bool | None = None):
        _preload_cuda_libs()
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Gemma4ChatHandler

        self.default_thinking = bool(thinking) if thinking is not None else False

        set_status("downloading", 0, "Checking model files…")
        main_path, mmproj_path = _ensure_model_files()
        self.main_path = main_path
        self.mmproj_path = mmproj_path

        t0 = time.time()
        set_status("loading", 0, "Loading model onto GPU…")
        log.info("Loading Gemma 4 E2B-it (Q4_K_M + mmproj) via llama.cpp…")
        handler = Gemma4ChatHandler(
            clip_model_path=mmproj_path,
            verbose=False,
            use_gpu=True,
        )
        self.llm = Llama(
            model_path=main_path,
            chat_handler=handler,
            n_gpu_layers=N_GPU_LAYERS,
            n_ctx=N_CTX,
            verbose=False,
        )
        log.info("Gemma loaded in %.1fs (n_ctx=%d, thinking=%s)",
                 time.time() - t0, N_CTX, self.default_thinking)
        set_status("ready", 100, "Ready")

    def __call__(self, img, thinking: bool | None = None) -> str:
        return self.recognize(img, thinking=thinking)

    def recognize(self, img, thinking: bool | None = None) -> str:
        """Recognize the formula in `img` and return cleaned Markdown+math."""
        think = self.default_thinking if thinking is None else bool(thinking)
        prompt = THINKING_PROMPT if think else PROMPT
        max_tokens = MAX_TOKENS_THINKING if think else MAX_TOKENS_FAST

        # Image BEFORE text, per Gemma multimodal docs.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _img_to_data_url(img)}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        t0 = time.time()
        resp = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1,
            top_p=0.95,
            repeat_penalty=1.05,
            stream=False,
        )
        elapsed = time.time() - t0
        content = resp["choices"][0]["message"].get("content") or ""
        usage = resp.get("usage", {}) or {}
        gen_tokens = usage.get("completion_tokens", 0)
        toks_per_s = (gen_tokens / elapsed) if elapsed > 0 and gen_tokens else 0.0
        log.info("Gemma generated %d tokens in %.2fs (%.1f tok/s, thinking=%s)",
                 gen_tokens, elapsed, toks_per_s, think)

        if think:
            return _extract_markdown(content)
        return _to_markdown_math(content)


def load():
    try:
        return GemmaVisionBackend()
    except Exception as e:
        set_status("error", 0, "Model load failed", error=str(e))
        raise
