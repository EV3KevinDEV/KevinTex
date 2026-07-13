"""Native macOS launcher for the MLX build of KevinTex."""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


APP_NAME = "KevinTex"


def _configure_runtime() -> Path:
    data_dir = Path.home() / "Library" / "Application Support" / APP_NAME
    models_dir = data_dir / "models"
    cache_dir = data_dir / "cache"
    for directory in (data_dir, models_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LOCALTEX_DATA_DIR", str(data_dir))
    os.environ.setdefault("LOCALTEX_MODELS_DIR", str(models_dir))
    os.environ.setdefault("HF_HOME", str(cache_dir / "huggingface"))
    os.environ["LOCALTEX_BACKEND"] = "mlx"
    return data_dir


def _run_snip_subprocess() -> bool:
    if len(sys.argv) < 3 or Path(sys.argv[1]).name.lower() != "snip.py":
        return False
    sys.argv = sys.argv[1:]
    from snip import main

    main()
    return True


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(url: str, server, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.should_exit:
            raise RuntimeError("KevinTex server stopped during startup")
        try:
            with urllib.request.urlopen(f"{url}/", timeout=0.5) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.1)
    raise TimeoutError("KevinTex server did not become ready")


def _smoke_test() -> int:
    _configure_runtime()
    import app

    index = Path(app.APP_DIR) / "static" / "index.html"
    if not index.is_file():
        raise FileNotFoundError(f"Bundled frontend is missing: {index}")
    if app.BACKEND != "mlx":
        raise RuntimeError("macOS MLX backend was not selected")
    return 0


def main() -> int:
    if _run_snip_subprocess():
        return 0
    if "--smoke-test" in sys.argv:
        return _smoke_test()

    _configure_runtime()
    import uvicorn
    import webview
    from app import app

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            access_log=False,
            log_level="info",
        )
    )
    thread = threading.Thread(target=server.run, name="kevintex-server", daemon=True)
    thread.start()
    _wait_for_server(url, server)

    webview.create_window(
        APP_NAME,
        url,
        width=1180,
        height=820,
        min_size=(760, 560),
        text_select=True,
    )
    try:
        webview.start(gui="cocoa")
    finally:
        server.should_exit = True
        thread.join(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
