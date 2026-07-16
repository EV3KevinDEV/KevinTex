"""Local configuration for selecting KevinTex's Gemma provider.

The Google AI Studio key is intentionally kept on the server side.  The web
UI sends it to the loopback FastAPI process, which stores it in this file with
user-only permissions where the platform supports POSIX permissions.
"""

from __future__ import annotations

import json
import os
import platform
import tempfile
import threading
from pathlib import Path


APP_NAME = "KevinTex"
CONFIG_NAME = "provider.json"
VALID_MODES = frozenset({"local", "cloud"})
_CONFIG_LOCK = threading.RLock()


def _default_data_dir() -> Path:
    configured = os.environ.get("LOCALTEX_DATA_DIR")
    if configured:
        return Path(configured).expanduser()

    system = platform.system()
    if system == "Windows":
        return Path(
            os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
        ) / APP_NAME
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / APP_NAME


def config_path() -> Path:
    """Return the provider config path, honoring the desktop launcher's data dir."""
    override = os.environ.get("LOCALTEX_PROVIDER_CONFIG")
    if override:
        return Path(override).expanduser()
    return _default_data_dir() / CONFIG_NAME


def read_config() -> dict:
    """Read a valid provider config without exposing errors to startup."""
    with _CONFIG_LOCK:
        try:
            value = json.loads(config_path().read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}
        if not isinstance(value, dict):
            return {}

        mode = value.get("mode")
        key = value.get("api_key")
        result = {}
        if mode in VALID_MODES:
            result["mode"] = mode
        if isinstance(key, str) and key.strip():
            result["api_key"] = key.strip()
        return result


def get_api_key() -> str | None:
    """Return the configured key, preferring an explicit environment override."""
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key
    return read_config().get("api_key")


def save_config(
    mode: str,
    api_key: str | None = None,
    *,
    clear_api_key: bool = False,
) -> dict:
    """Persist a provider selection and optionally replace its API key."""
    if mode not in VALID_MODES:
        raise ValueError("Provider mode must be 'local' or 'cloud'.")
    if api_key is not None:
        api_key = api_key.strip()
        if not api_key:
            raise ValueError("API key cannot be empty.")

    with _CONFIG_LOCK:
        current = read_config()
        payload = {"mode": mode}
        if clear_api_key:
            pass
        elif api_key is not None:
            payload["api_key"] = api_key
        elif current.get("api_key"):
            payload["api_key"] = current["api_key"]

        path = config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{CONFIG_NAME}.", dir=str(path.parent), text=True
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
            try:
                os.chmod(temporary, 0o600)
            except OSError:
                pass
            os.replace(temporary, path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        return read_config()
