"""Native desktop launcher for KevinTex.

Starts the existing FastAPI application on loopback and hosts it in a pywebview
window.  Model files and Hugging Face state are kept outside the installation
in the current user's local application-data directory.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path


APP_NAME = "KevinTex"


def _user_data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        base = str(Path.home() / "AppData" / "Local")
    return Path(base) / APP_NAME


def _configure_user_paths() -> Path:
    data_dir = _user_data_dir()
    models_dir = data_dir / "models"
    cache_dir = data_dir / "cache"
    for directory in (data_dir, models_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LOCALTEX_DATA_DIR", str(data_dir))
    os.environ.setdefault("LOCALTEX_MODELS_DIR", str(models_dir))
    os.environ.setdefault("HF_HOME", str(cache_dir / "huggingface"))
    return data_dir


def _configure_llama_backend() -> None:
    """Make the shipped build CPU-safe while allowing custom CUDA builds."""
    bundled_cuda_marker = (
        Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / "cuda_enabled.txt"
    )
    cuda_requested = bundled_cuda_marker.is_file() or os.environ.get(
        "KEVINTEX_CUDA", ""
    ).lower() in {
        "1",
        "true",
        "yes",
    }
    if cuda_requested:
        return

    import backend_gemma
    import llama_cpp.llama_chat_format as chat_format

    backend_gemma.N_GPU_LAYERS = 0
    original = chat_format.Gemma4ChatHandler

    def cpu_chat_handler(*args, **kwargs):
        kwargs["use_gpu"] = False
        return original(*args, **kwargs)

    chat_format.Gemma4ChatHandler = cpu_chat_handler


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_until_ready(url: str, server, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not server_thread_alive(server):
            raise RuntimeError("KevinTex server stopped during startup")
        try:
            with urllib.request.urlopen(f"{url}/", timeout=0.5) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.1)
    raise TimeoutError("KevinTex server did not become ready")


def server_thread_alive(server) -> bool:
    return not server.should_exit


def _run_snip_subprocess() -> bool:
    """Handle app.py relaunching the frozen executable for snip.py."""
    if len(sys.argv) < 3 or Path(sys.argv[1]).name.lower() != "snip.py":
        return False
    sys.argv = sys.argv[1:]
    from snip import main

    main()
    return True


def _smoke_test() -> int:
    _configure_user_paths()
    _configure_llama_backend()
    import app

    index = Path(app.APP_DIR) / "static" / "index.html"
    if not index.is_file():
        raise FileNotFoundError(f"Bundled frontend is missing: {index}")
    return 0


def main() -> int:
    if _run_snip_subprocess():
        return 0
    if "--smoke-test" in sys.argv:
        return _smoke_test()

    _configure_user_paths()
    _configure_llama_backend()

    import uvicorn
    import webview
    from app import app

    port = _free_loopback_port()
    url = f"http://127.0.0.1:{port}"
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="info",
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="kevintex-server", daemon=True)
    thread.start()
    _wait_until_ready(url, server)

    webview.create_window(
        APP_NAME,
        url,
        width=1180,
        height=820,
        min_size=(760, 560),
        text_select=True,
    )
    try:
        webview.start()
    finally:
        server.should_exit = True
        thread.join(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
