"""KevinTex — offline formula-image → LaTeX/Markdown converter.

A local, free, unlimited SimpleTex-style app. Default backend is Google's
Gemma 4 E2B-it multimodal model (Q4_K_M GGUF + vision
projector) run via llama.cpp entirely on this machine — no cloud calls. Set
LOCALTEX_BACKEND=mlx for Apple Silicon MLX acceleration,
LOCALTEX_BACKEND=lfm-vl for the Liquid AI LFM2.5-VL backend, or
LOCALTEX_BACKEND=pix2tex for the smaller pix2tex model.
"""

import base64
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

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.environ.get("LOCALTEX_BACKEND", "gemma").lower()
# Default thinking mode for the VLM backend (see backend_vlm). 0/1 via env.
THINKING_DEFAULT = os.environ.get("LOCALTEX_THINKING", "0") in ("1", "true", "True", "yes")


def _positive_env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


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
    if request.url.path in ("/api/convert", "/api/preprocess"):
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
        if BACKEND == "pix2tex":
            from pix2tex.cli import LatexOCR
            _model = LatexOCR()
        elif BACKEND == "lfm-vl":
            import backend_vlm
            _model = backend_vlm.load()
        elif BACKEND == "gemma":
            import backend_gemma
            _model = backend_gemma.load()
        elif BACKEND == "mlx":
            import backend_mlx
            _model = backend_mlx.load()
        else:
            raise RuntimeError(f"Unknown LOCALTEX_BACKEND={BACKEND!r}")
        log.info("Backend '%s' loaded", BACKEND)
    return _model


def _load_in_background() -> None:
    """Load the model off the request path so the server is reachable at once."""
    try:
        get_model()
    except Exception as e:
        log.exception("Model load failed")
        if BACKEND in ("gemma", "mlx"):
            try:
                backend_status = _backend_status_module()
                backend_status.set_status(
                    "error", 0, "Model load failed", error=str(e)
                )
            except Exception:
                pass


def model_ready() -> bool:
    return _model is not None


def _backend_status_module():
    if BACKEND == "gemma":
        import backend_gemma

        return backend_gemma
    if BACKEND == "mlx":
        import backend_mlx

        return backend_mlx
    return None


def _device_label() -> str:
    if BACKEND == "mlx":
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
    return {
        "status": "ok",
        "device": _device_label(),
        "backend": BACKEND,
        "model_loaded": _model is not None,
        "thinking_default": THINKING_DEFAULT,
    }


@app.get("/api/status")
def status():
    backend_status = _backend_status_module()
    if backend_status is not None:
        s = backend_status.get_status()
    else:
        s = {
            "phase": "ready" if model_ready() else "loading",
            "progress": 0,
            "message": "Ready" if model_ready() else "Loading model…",
            "error": None,
        }
    s["backend"] = BACKEND
    s["device"] = _device_label()
    s["model_loaded"] = model_ready()
    return s


@app.post("/api/reload")
def reload():
    """Re-trigger the background model load after a failure (splash Retry button)."""
    global _model
    if _model is None and not _loading_active():
        if BACKEND in ("gemma", "mlx"):
            try:
                backend_status = _backend_status_module()
                backend_status.set_status("loading", 0, "Retrying model load…")
            except Exception:
                pass
        threading.Thread(target=_load_in_background, daemon=True).start()
    return {"ok": True}


def _loading_active() -> bool:
    if _model_load_lock.locked():
        return True
    if BACKEND in ("gemma", "mlx"):
        try:
            backend_status = _backend_status_module()
            return backend_status.get_status()["phase"] in ("downloading", "loading")
        except Exception:
            return False
    return _model is None


@app.get("/")
def index():
    return FileResponse(os.path.join(APP_DIR, "static", "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(APP_DIR, "static")), name="static")
