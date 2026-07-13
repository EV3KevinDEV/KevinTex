#!/usr/bin/env bash
# Install KevinTex for the current user only (no root needed).
set -euo pipefail

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
APP="$HOME/.local/opt/kevintex"
BIN="$HOME/.local/bin"
APPS="$HOME/.local/share/applications"
ICONS="$HOME/.local/share/icons/hicolor/scalable/apps"

mkdir -p "$APP" "$BIN" "$APPS" "$ICONS"

cp "$PROJ/app.py" "$PROJ/backend_vlm.py" "$PROJ/backend_gemma.py" "$PROJ/snip.py" "$PROJ/requirements.txt" "$APP/"
rm -rf "$APP/static" && cp -r "$PROJ/static" "$APP/static"
install -m 755 "$PROJ/packaging/localtex-launcher" "$BIN/kevintex"
cp "$PROJ/packaging/localtex.svg" "$ICONS/kevintex.svg"

sed "s|^Exec=kevintex$|Exec=env KEVINTEX_APP_DIR=$APP $BIN/kevintex|" \
    "$PROJ/packaging/localtex.desktop" > "$APPS/kevintex.desktop"

# Remove obsolete launchers from releases branded LocalTeX. Keep their app data
# directory so existing model downloads and settings remain available.
rm -f "$BIN/localtex" "$APPS/localtex.desktop" "$ICONS/localtex.svg"

# Reuse the dev venv if present so first run doesn't re-download PyTorch.
DATA="$HOME/.local/share/localtex"
mkdir -p "$DATA"
if [ ! -e "$DATA/venv" ] && [ -x "$PROJ/.venv/bin/uvicorn" ]; then
    ln -s "$PROJ/.venv" "$DATA/venv"
fi

command -v update-desktop-database >/dev/null && update-desktop-database -q "$APPS" || true
command -v gtk-update-icon-cache >/dev/null && gtk-update-icon-cache -q "$HOME/.local/share/icons/hicolor" 2>/dev/null || true

echo "Installed. Find 'KevinTex' in your app menu, or run: $BIN/kevintex"
