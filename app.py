"""KevinTex — offline formula-image → LaTeX/Markdown converter.

A local, free, unlimited SimpleTex-style app. Default backend is Google's
Gemma 4 E2B-it multimodal model (Q4_K_M GGUF + mmproj vision projector) run
via llama.cpp entirely on this machine — no cloud calls. Set
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

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.environ.get("LOCALTEX_BACKEND", "gemma").lower()
# Default thinking mode for the VLM backend (see backend_vlm). 0/1 via env.
THINKING_DEFAULT = os.environ.get("LOCALTEX_THINKING", "0") in ("1", "true", "True", "yes")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("kevintex")

app = FastAPI(title="KevinTex")

_model = None


def get_model():
    global _model
    if _model is None:
        if BACKEND == "pix2tex":
            from pix2tex.cli import LatexOCR
            _model = LatexOCR()
        elif BACKEND == "lfm-vl":
            import backend_vlm
            _model = backend_vlm.load()
        elif BACKEND == "gemma":
            import backend_gemma
            _model = backend_gemma.load()
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
        if BACKEND == "gemma":
            try:
                import backend_gemma
                backend_gemma.set_status("error", 0, "Model load failed", error=str(e))
            except Exception:
                pass


def model_ready() -> bool:
    return _model is not None


def _run_model(img, thinking: bool | None):
    """Call the loaded backend, forwarding `thinking` to VLM backends that
    support it. pix2tex ignores the kwarg."""
    backend = get_model()
    if hasattr(backend, "recognize"):
        return backend.recognize(img, thinking=thinking)
    return backend(img)


@app.on_event("startup")
def warm_up():
    # Load in a background thread so /api/status and /api/health respond right
    # away with progress instead of blocking until the model is ready.
    threading.Thread(target=_load_in_background, daemon=True).start()


@app.post("/api/convert")
async def convert(image: UploadFile = File(...), thinking: bool | None = Form(None)):
    if not model_ready():
        return JSONResponse({"error": "model_warming_up"}, status_code=503)
    raw = await image.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return JSONResponse({"error": "Could not read that file as an image."}, status_code=400)

    use_thinking = THINKING_DEFAULT if thinking is None else thinking
    t0 = time.time()
    try:
        latex = _run_model(img, use_thinking)
    except Exception as e:
        log.exception("Inference failed")
        return JSONResponse({"error": f"Recognition failed: {e}"}, status_code=500)
    elapsed = time.time() - t0
    log.info("Converted %s (%dx%d) in %.2fs (thinking=%s)",
             image.filename, img.width, img.height, elapsed, use_thinking)
    return {"latex": latex, "elapsed": round(elapsed, 2), "thinking": use_thinking}


@app.post("/api/snip")
def snip(thinking: bool | None = None):
    """Capture a screen region (via the desktop's screenshot tool) and convert it."""
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
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return JSONResponse({"error": "Could not read captured image."}, status_code=500)
    finally:
        try:
            os.remove(out)
        except OSError:
            pass

    if not model_ready():
        return JSONResponse({"error": "model_warming_up"}, status_code=503)
    use_thinking = THINKING_DEFAULT if thinking is None else thinking
    t0 = time.time()
    try:
        latex = _run_model(img, use_thinking)
    except Exception as e:
        log.exception("Inference failed")
        return JSONResponse({"error": f"Recognition failed: {e}"}, status_code=500)
    elapsed = time.time() - t0
    log.info("Snipped and converted (%dx%d) in %.2fs (thinking=%s)",
             img.width, img.height, elapsed, use_thinking)

    thumb = "data:image/png;base64," + base64.b64encode(raw).decode()
    return {"latex": latex, "elapsed": round(elapsed, 2), "thumb": thumb, "thinking": use_thinking}


@app.get("/api/health")
def health():
    import torch

    return {
        "status": "ok",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "backend": BACKEND,
        "model_loaded": _model is not None,
        "thinking_default": THINKING_DEFAULT,
    }


@app.get("/api/status")
def status():
    import torch

    if BACKEND == "gemma":
        import backend_gemma
        s = backend_gemma.get_status()
    else:
        s = {
            "phase": "ready" if model_ready() else "loading",
            "progress": 0,
            "message": "Ready" if model_ready() else "Loading model…",
            "error": None,
        }
    s["backend"] = BACKEND
    s["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    s["model_loaded"] = model_ready()
    return s


@app.post("/api/reload")
def reload():
    """Re-trigger the background model load after a failure (splash Retry button)."""
    global _model
    if _model is None and not _loading_active():
        if BACKEND == "gemma":
            try:
                import backend_gemma
                backend_gemma.set_status("loading", 0, "Retrying model load…")
            except Exception:
                pass
        threading.Thread(target=_load_in_background, daemon=True).start()
    return {"ok": True}


def _loading_active() -> bool:
    if BACKEND == "gemma":
        try:
            import backend_gemma
            return backend_gemma.get_status()["phase"] in ("downloading", "loading")
        except Exception:
            return False
    return _model is None


@app.get("/")
def index():
    return FileResponse(os.path.join(APP_DIR, "static", "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(APP_DIR, "static")), name="static")
