"""KevinTex — formula-image → LaTeX/Markdown converter.

A local, free, unlimited SimpleTex-style app. The default backend is Google's
Gemma 4 E2B-it multimodal model (Q4_K_M GGUF + vision projector) run via
llama.cpp entirely on this machine. Users can also select the hosted Gemma 4
26B A4B model through Google AI Studio. Set LOCALTEX_BACKEND=mlx for Apple
Silicon MLX acceleration,
LOCALTEX_BACKEND=lfm-vl for the Liquid AI LFM2.5-VL backend, or
LOCALTEX_BACKEND=pix2tex for the smaller pix2tex model.
"""

import base64
import gc
import io
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from image_preprocessing import MAX_INPUT_PIXELS, preprocess_image, validate_options
from provider_config import get_api_key, read_config, save_config

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.environ.get("LOCALTEX_BACKEND", "gemma").lower()
_GEMMA_BACKENDS = {"gemma", "mlx", "gemma-cloud"}
_PROVIDER_MODES = {"local", "cloud"}
# Default thinking mode for the VLM backend (see backend_vlm). 0/1 via env.
THINKING_DEFAULT = os.environ.get("LOCALTEX_THINKING", "0") in ("1", "true", "True", "yes")


def _positive_env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _gemma_provider_supported() -> bool:
    return BACKEND in _GEMMA_BACKENDS


def _local_model_available() -> bool:
    """Detect an already-downloaded local model for first-run migration."""
    local_backend = "gemma" if BACKEND == "gemma-cloud" else BACKEND
    try:
        if local_backend == "gemma":
            import backend_gemma

            return backend_gemma.model_files_present()
        if local_backend == "mlx":
            import backend_mlx

            return backend_mlx.model_files_present()
    except (ImportError, OSError):
        return False
    return False


def _configured_provider() -> str | None:
    """Resolve an explicit provider, with compatibility for existing installs."""
    if BACKEND == "gemma-cloud":
        return "cloud"

    forced = os.environ.get("LOCALTEX_PROVIDER", "").strip().lower()
    if forced in _PROVIDER_MODES:
        return forced

    stored = read_config().get("mode")
    if stored in _PROVIDER_MODES:
        return stored
    if _gemma_provider_supported() and os.environ.get("GEMINI_API_KEY", "").strip():
        return "cloud"
    if _local_model_available():
        return "local"
    return None


def _active_backend() -> str:
    if BACKEND in _GEMMA_BACKENDS and _configured_provider() == "cloud":
        return "gemma-cloud"
    return "gemma" if BACKEND == "gemma-cloud" else BACKEND


def _setup_required() -> bool:
    mode = _configured_provider()
    return _gemma_provider_supported() and (
        mode is None or (mode == "cloud" and not get_api_key())
    )


def _provider_state() -> dict:
    mode = _configured_provider()
    api_key = get_api_key()
    supported = _gemma_provider_supported()
    setup_required = supported and (
        mode is None or (mode == "cloud" and not api_key)
    )
    return {
        "supported": supported,
        "mode": mode,
        "configured": mode is not None,
        "setup_required": setup_required,
        "api_key_configured": bool(api_key),
        "api_key_hint": ("••••" + api_key[-4:]) if api_key else "",
        "local_model_available": _local_model_available(),
        "cloud_model": os.environ.get(
            "LOCALTEX_GEMMA_CLOUD_MODEL", "gemma-4-26b-a4b-it"
        ),
    }


MAX_UPLOAD_BYTES = _positive_env_int("LOCALTEX_MAX_UPLOAD_BYTES", 20 * 1024 * 1024)
MAX_MULTIPART_BYTES = MAX_UPLOAD_BYTES + 1024 * 1024
INFERENCE_CONCURRENCY = _positive_env_int("LOCALTEX_INFERENCE_CONCURRENCY", 1)
try:
    INFERENCE_QUEUE_TIMEOUT = max(
        0.0, float(os.environ.get("LOCALTEX_INFERENCE_QUEUE_TIMEOUT", "0.25"))
    )
except ValueError:
    INFERENCE_QUEUE_TIMEOUT = 0.25

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("kevintex")

app = FastAPI(title="KevinTex")

_model = None
_model_load_error = None
_model_load_lock = threading.Lock()
_inference_slots = threading.BoundedSemaphore(INFERENCE_CONCURRENCY)


class InferenceBusyError(RuntimeError):
    """Raised when configured inference capacity is already occupied."""


class UploadTooLargeError(ValueError):
    """Raised before decoding when an upload exceeds the configured byte cap."""


class ImageTooLargeError(ValueError):
    """Raised from image headers before allocating the decoded pixel buffer."""


@app.middleware("http")
async def limit_upload_content_length(request: Request, call_next):
    """Reject normal oversized multipart requests before Starlette spools them."""
    if request.url.path in ("/api/convert", "/api/preprocess", "/api/voice"):
        value = request.headers.get("content-length")
        try:
            content_length = int(value) if value is not None else None
        except ValueError:
            content_length = None
        if content_length is not None and content_length > MAX_MULTIPART_BYTES:
            limit_mib = MAX_UPLOAD_BYTES / (1024 * 1024)
            return JSONResponse(
                {"error": f"Image upload is too large ({limit_mib:g} MiB maximum)."},
                status_code=413,
            )
    return await call_next(request)


def get_model():
    global _model
    if _model is not None:
        return _model
    with _model_load_lock:
        if _model is not None:
            return _model
        active_backend = _active_backend()
        if active_backend == "pix2tex":
            from pix2tex.cli import LatexOCR
            _model = LatexOCR()
        elif active_backend == "lfm-vl":
            import backend_vlm
            _model = backend_vlm.load()
        elif active_backend == "gemma":
            import backend_gemma
            _model = backend_gemma.load()
        elif active_backend == "mlx":
            import backend_mlx
            _model = backend_mlx.load()
        elif active_backend == "gemma-cloud":
            import backend_gemma_cloud
            _model = backend_gemma_cloud.load(api_key=get_api_key())
        else:
            raise RuntimeError(f"Unknown LOCALTEX_BACKEND={active_backend!r}")
        log.info("Backend '%s' loaded", active_backend)
    return _model


def _load_in_background() -> None:
    """Load the model off the request path so the server is reachable at once."""
    global _model_load_error
    if _setup_required():
        return
    try:
        get_model()
        _model_load_error = None
    except Exception as e:
        _model_load_error = str(e)
        log.exception("Model load failed")
        if _active_backend() in ("gemma", "mlx", "gemma-cloud"):
            try:
                backend_status = _backend_status_module()
                backend_status.set_status(
                    "error", 0, "Model load failed", error=str(e)
                )
            except Exception:
                pass


def model_ready() -> bool:
    return _model is not None


def _dispose_model(model) -> None:
    """Release backend resources when switching between local and cloud."""
    if model is None:
        return
    close = getattr(model, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            log.exception("Backend cleanup failed")
    del model
    gc.collect()


def _reserve_inference_capacity() -> int:
    """Reserve every inference slot, returning zero when any request is active."""
    acquired = 0
    for _ in range(INFERENCE_CONCURRENCY):
        if not _inference_slots.acquire(blocking=False):
            for _ in range(acquired):
                _inference_slots.release()
            return 0
        acquired += 1
    return acquired


def _release_inference_capacity(acquired: int) -> None:
    for _ in range(acquired):
        _inference_slots.release()


def _backend_status_module():
    active_backend = _active_backend()
    if active_backend == "gemma":
        import backend_gemma

        return backend_gemma
    if active_backend == "mlx":
        import backend_mlx

        return backend_mlx
    if active_backend == "gemma-cloud":
        import backend_gemma_cloud

        return backend_gemma_cloud
    return None


def _device_label() -> str:
    active_backend = _active_backend()
    if active_backend == "gemma-cloud":
        return "cloud"
    if active_backend == "mlx":
        return "metal"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _run_model(img, thinking: bool | None):
    """Call the loaded backend, forwarding `thinking` to VLM backends that
    support it. pix2tex ignores the kwarg."""
    if not _inference_slots.acquire(timeout=INFERENCE_QUEUE_TIMEOUT):
        raise InferenceBusyError("Inference capacity is busy; retry shortly.")
    try:
        backend = get_model()
        if hasattr(backend, "recognize"):
            return backend.recognize(img, thinking=thinking)
        return backend(img)
    finally:
        _inference_slots.release()


def _run_audio(audio_path: str, thinking: bool | None):
    """Run an audio-capable backend under the shared inference backpressure."""
    if not _inference_slots.acquire(timeout=INFERENCE_QUEUE_TIMEOUT):
        raise InferenceBusyError("Inference capacity is busy; retry shortly.")
    try:
        backend = get_model()
        if not hasattr(backend, "recognize_audio"):
            raise RuntimeError(
                "Voice-to-LaTeX requires the Apple Silicon MLX backend."
            )
        return backend.recognize_audio(audio_path, thinking=thinking)
    finally:
        _inference_slots.release()


def _open_image(raw: bytes) -> Image.Image:
    try:
        with io.BytesIO(raw) as source:
            with Image.open(source) as image:
                if image.width * image.height > MAX_INPUT_PIXELS:
                    raise ImageTooLargeError(
                        "Image is too large (50 megapixels maximum)."
                    )
                image.load()
                return image.copy()
    except ImageTooLargeError:
        raise
    except Exception as exc:
        raise ValueError("Could not read that file as an image.") from exc


async def _read_upload(image: UploadFile) -> bytes:
    """Read at most the configured request size without buffering an unbounded body."""
    try:
        raw = await image.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await image.close()
    if len(raw) > MAX_UPLOAD_BYTES:
        limit_mib = MAX_UPLOAD_BYTES / (1024 * 1024)
        raise UploadTooLargeError(
            f"Image upload is too large ({limit_mib:g} MiB maximum)."
        )
    return raw


def _prepare_image(
    image: Image.Image,
    preprocess: str,
    rotation: int,
    invert: bool,
) -> tuple[Image.Image, str]:
    mode, rotation = validate_options(preprocess, rotation)
    return preprocess_image(image, mode=mode, rotation=rotation, invert=invert), mode


def _preprocess_metadata(
    mode: str, rotation: int, invert: bool, image: Image.Image
) -> dict:
    return {
        "mode": mode,
        "rotation": rotation,
        "invert": invert,
        "width": image.width,
        "height": image.height,
    }


@app.on_event("startup")
def warm_up():
    # Load in a background thread so /api/status and /api/health respond right
    # away with progress instead of blocking until the model is ready.
    if _setup_required():
        return
    threading.Thread(target=_load_in_background, daemon=True).start()


@app.post("/api/preprocess")
async def preprocess_preview(
    image: UploadFile = File(...),
    preprocess: str = Form("auto"),
    rotation: int = Form(0),
    invert: bool = Form(False),
):
    """Apply image preparation without loading or invoking the OCR model."""
    try:
        raw = await _read_upload(image)
        source = _open_image(raw)
        try:
            prepared, mode = _prepare_image(source, preprocess, rotation, invert)
        finally:
            source.close()
    except UploadTooLargeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=413)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    try:
        with io.BytesIO() as out:
            prepared.save(out, format="PNG", optimize=True)
            body = out.getvalue()
        metadata = _preprocess_metadata(mode, rotation, invert, prepared)
        return Response(
            body,
            media_type="image/png",
            headers={
                "X-KevinTex-Preprocess-Mode": mode,
                "X-KevinTex-Preprocess-Size": f"{metadata['width']}x{metadata['height']}",
                "X-KevinTex-Rotation": str(rotation),
                "X-KevinTex-Invert": str(invert).lower(),
            },
        )
    finally:
        prepared.close()


@app.post("/api/convert")
async def convert(
    image: UploadFile = File(...),
    thinking: bool | None = Form(None),
    preprocess: str = Form("auto"),
    rotation: int = Form(0),
    invert: bool = Form(False),
):
    filename = image.filename
    try:
        validate_options(preprocess, rotation)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    try:
        raw = await _read_upload(image)
    except UploadTooLargeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=413)
    if not model_ready():
        return JSONResponse({"error": "model_warming_up"}, status_code=503)
    try:
        source = _open_image(raw)
        try:
            img, mode = _prepare_image(source, preprocess, rotation, invert)
        finally:
            source.close()
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    del raw

    try:
        use_thinking = THINKING_DEFAULT if thinking is None else thinking
        t0 = time.time()
        try:
            latex = await run_in_threadpool(_run_model, img, use_thinking)
        except InferenceBusyError as exc:
            return JSONResponse({"error": str(exc)}, status_code=429)
        except Exception as e:
            log.exception("Inference failed")
            return JSONResponse({"error": f"Recognition failed: {e}"}, status_code=500)
        elapsed = time.time() - t0
        log.info(
            "Converted %s (%dx%d) in %.2fs (thinking=%s, preprocess=%s, rotation=%d, invert=%s)",
            filename, img.width, img.height, elapsed, use_thinking, mode, rotation, invert,
        )
        return {
            "latex": latex,
            "elapsed": round(elapsed, 2),
            "thinking": use_thinking,
            "preprocessing": _preprocess_metadata(mode, rotation, invert, img),
        }
    finally:
        img.close()


@app.post("/api/voice")
async def voice(
    audio: UploadFile = File(...),
    thinking: bool | None = Form(None),
):
    """Convert a short WAV recording of spoken mathematics to LaTeX."""
    if _active_backend() not in ("gemma", "mlx"):
        return JSONResponse(
            {"error": "Voice-to-LaTeX requires local Gemma or MLX."},
            status_code=501,
        )
    try:
        raw = await _read_upload(audio)
    except UploadTooLargeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=413)
    if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        return JSONResponse({"error": "Voice input must be a WAV recording."}, status_code=400)
    if not model_ready():
        return JSONResponse({"error": "model_warming_up"}, status_code=503)

    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(raw)
            path = handle.name
        del raw
        use_thinking = THINKING_DEFAULT if thinking is None else thinking
        t0 = time.time()
        try:
            latex = await run_in_threadpool(_run_audio, path, use_thinking)
        except InferenceBusyError as exc:
            return JSONResponse({"error": str(exc)}, status_code=429)
        except Exception as exc:
            log.exception("Voice inference failed")
            return JSONResponse({"error": f"Voice recognition failed: {exc}"}, status_code=500)
        elapsed = time.time() - t0
        return {"latex": latex, "elapsed": round(elapsed, 2), "thinking": use_thinking}
    finally:
        if path:
            try:
                os.remove(path)
            except OSError:
                pass


@app.post("/api/snip")
def snip(
    thinking: bool | None = None,
    preprocess: str = "auto",
    rotation: int = 0,
    invert: bool = False,
):
    """Capture a screen region (via the desktop's screenshot tool) and convert it."""
    try:
        validate_options(preprocess, rotation)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    out = os.path.join(tempfile.gettempdir(), f"kevintex-snip-{os.getpid()}.png")
    try:
        # Run with the same interpreter as the server (the venv python): it has
        # Pillow + tkinter for a self-contained region selector.
        proc = subprocess.run(
            [sys.executable, os.path.join(APP_DIR, "snip.py"), out],
            capture_output=True, text=True, timeout=200,
        )
    except subprocess.TimeoutExpired:
        return JSONResponse({"error": "Screenshot timed out."}, status_code=500)
    if proc.returncode != 0:
        msg = proc.stderr.strip().replace("ERROR: ", "")
        log.error("snip failed: %s", msg)
        if "cancel" in msg.lower():
            return JSONResponse({"error": "Snip cancelled."}, status_code=400)
        return JSONResponse(
            {"error": f"Screen capture failed: {msg}"}, status_code=500
        )

    try:
        with open(out, "rb") as f:
            raw = f.read()
        source = _open_image(raw)
        try:
            img, mode = _prepare_image(source, preprocess, rotation, invert)
        finally:
            source.close()
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    finally:
        try:
            os.remove(out)
        except OSError:
            pass

    if not model_ready():
        img.close()
        del raw
        return JSONResponse({"error": "model_warming_up"}, status_code=503)
    try:
        use_thinking = THINKING_DEFAULT if thinking is None else thinking
        t0 = time.time()
        try:
            latex = _run_model(img, use_thinking)
        except InferenceBusyError as exc:
            return JSONResponse({"error": str(exc)}, status_code=429)
        except Exception as e:
            log.exception("Inference failed")
            return JSONResponse({"error": f"Recognition failed: {e}"}, status_code=500)
        elapsed = time.time() - t0
        log.info(
            "Snipped and converted (%dx%d) in %.2fs (thinking=%s, preprocess=%s, rotation=%d, invert=%s)",
            img.width, img.height, elapsed, use_thinking, mode, rotation, invert,
        )
        thumb = "data:image/png;base64," + base64.b64encode(raw).decode()
        return {
            "latex": latex,
            "elapsed": round(elapsed, 2),
            "thumb": thumb,
            "thinking": use_thinking,
            "preprocessing": _preprocess_metadata(mode, rotation, invert, img),
        }
    finally:
        img.close()
        del raw


@app.get("/api/health")
def health():
    active_backend = _active_backend()
    return {
        "status": "ok",
        "device": _device_label(),
        "backend": active_backend,
        "model_loaded": _model is not None,
        "thinking_default": THINKING_DEFAULT,
        "audio_supported": active_backend in ("gemma", "mlx"),
        "setup_required": _setup_required(),
    }


@app.get("/api/status")
def status():
    active_backend = _active_backend()
    if _setup_required():
        return {
            "phase": "setup",
            "progress": 0,
            "message": "Choose local model weights or Google AI Studio",
            "error": None,
            "backend": active_backend,
            "device": _device_label(),
            "model_loaded": False,
            "audio_supported": False,
            "setup_required": True,
            "provider": _provider_state(),
        }
    backend_status = _backend_status_module()
    if backend_status is not None:
        s = backend_status.get_status()
    else:
        s = {
            "phase": "ready" if model_ready() else ("error" if _model_load_error else "loading"),
            "progress": 0,
            "message": "Ready" if model_ready() else ("Model load failed" if _model_load_error else "Loading model…"),
            "error": _model_load_error,
        }
    s["backend"] = active_backend
    s["device"] = _device_label()
    s["model_loaded"] = model_ready()
    s["audio_supported"] = active_backend in ("gemma", "mlx")
    s["setup_required"] = False
    s["provider"] = _provider_state()
    return s


@app.get("/api/provider")
def provider():
    """Return provider metadata without ever returning the saved API key."""
    return _provider_state()


@app.post("/api/provider")
async def configure_provider(request: Request):
    """Select local weights or save a Google AI Studio key and reload."""
    if not _gemma_provider_supported():
        return JSONResponse(
            {"error": "Provider selection is only available for Gemma backends."},
            status_code=400,
        )
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Provider settings must be valid JSON."}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "Provider settings must be an object."}, status_code=400)

    mode = str(body.get("mode", "")).strip().lower()
    if mode not in ("local", "cloud"):
        return JSONResponse(
            {"error": "Provider mode must be 'local' or 'cloud'."}, status_code=400
        )
    api_key = body.get("api_key")
    if api_key is not None and not isinstance(api_key, str):
        return JSONResponse({"error": "API key must be text."}, status_code=400)
    if mode == "cloud" and not (api_key or get_api_key()):
        return JSONResponse(
            {"error": "Paste a Google AI Studio API key to use the cloud model."},
            status_code=400,
        )

    reserved_slots = _reserve_inference_capacity()
    if not reserved_slots:
        return JSONResponse(
            {"error": "Wait for the current recognition to finish before switching providers."},
            status_code=409,
        )
    load_lock_acquired = _model_load_lock.acquire(blocking=False)
    if not load_lock_acquired:
        _release_inference_capacity(reserved_slots)
        return JSONResponse(
            {"error": "Wait for the current model load to finish before switching providers."},
            status_code=409,
        )

    global _model, _model_load_error
    previous_model = None
    try:
        try:
            save_config(mode, api_key=api_key if api_key else None)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except OSError as exc:
            log.exception("Could not save provider settings")
            return JSONResponse(
                {"error": f"Could not save provider settings: {exc}"},
                status_code=500,
            )

        previous_model = _model
        _model = None
        _model_load_error = None
        active_backend = _active_backend()
        if active_backend in ("gemma", "mlx", "gemma-cloud"):
            try:
                _backend_status_module().set_status("loading", 0, "Starting model…")
            except Exception:
                pass
        _dispose_model(previous_model)
        previous_model = None
    finally:
        _model_load_lock.release()
        _release_inference_capacity(reserved_slots)

    threading.Thread(target=_load_in_background, daemon=True).start()
    return _provider_state()


@app.post("/api/reload")
def reload():
    """Re-trigger the background model load after a failure (splash Retry button)."""
    global _model, _model_load_error
    if _setup_required():
        return {"ok": True, "setup_required": True}
    if _model is None and not _loading_active():
        _model_load_error = None
        if _active_backend() in ("gemma", "mlx", "gemma-cloud"):
            try:
                backend_status = _backend_status_module()
                backend_status.set_status("loading", 0, "Retrying model load…")
            except Exception:
                pass
        threading.Thread(target=_load_in_background, daemon=True).start()
    return {"ok": True}


def _loading_active() -> bool:
    if _setup_required():
        return False
    if _model_load_lock.locked():
        return True
    if _active_backend() in ("gemma", "mlx", "gemma-cloud"):
        try:
            backend_status = _backend_status_module()
            return backend_status.get_status()["phase"] in ("downloading", "loading")
        except Exception:
            return False
    return _model is None and _model_load_error is None


@app.get("/")
def index():
    return FileResponse(os.path.join(APP_DIR, "static", "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(APP_DIR, "static")), name="static")
