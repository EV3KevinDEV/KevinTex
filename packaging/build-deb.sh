#!/usr/bin/env bash
# Build kevintex_<version>_all.deb from the project sources.
set -euo pipefail

VERSION="${KEVINTEX_VERSION:-1.2.8}"
if [[ ! "$VERSION" =~ ^[0-9]+([.][0-9]+){1,3}$ ]]; then
  echo "KEVINTEX_VERSION must look like 1.2.3" >&2
  exit 2
fi
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

PKG="$STAGE/kevintex_${VERSION}_all"

# --- payload ---
install -d "$PKG/opt/kevintex" "$PKG/usr/bin" \
           "$PKG/usr/share/applications" \
           "$PKG/usr/share/icons/hicolor/512x512/apps" \
           "$PKG/usr/share/doc/kevintex"

cp "$PROJ/app.py" "$PROJ/backend_vlm.py" "$PROJ/backend_gemma.py" "$PROJ/backend_gemma_cloud.py" "$PROJ/provider_config.py" "$PROJ/image_preprocessing.py" "$PROJ/snip.py" "$PROJ/requirements.txt" "$PKG/opt/kevintex/"
cp -r "$PROJ/static" "$PKG/opt/kevintex/static"
cp "$PROJ/README.md" "$PKG/usr/share/doc/kevintex/"
cp -r "$PROJ/assets" "$PKG/usr/share/doc/kevintex/assets"
install -m 755 "$PROJ/packaging/localtex-launcher" "$PKG/usr/bin/kevintex"
install -m 644 "$PROJ/packaging/localtex.desktop" "$PKG/usr/share/applications/kevintex.desktop"
install -m 644 "$PROJ/packaging/kevintex.png" "$PKG/usr/share/icons/hicolor/512x512/apps/kevintex.png"

# --- control files ---
install -d "$PKG/DEBIAN"
SIZE_KB=$(du -sk "$PKG" --exclude=DEBIAN | cut -f1)
cat > "$PKG/DEBIAN/control" <<EOF
Package: kevintex
Version: $VERSION
Section: science
Priority: optional
Architecture: all
Installed-Size: $SIZE_KB
Depends: python3 (>= 3.9), python3-venv, python3-pip, python3-tk, curl
Recommends: zenity, libnotify-bin
Maintainer: Kevin <kevin@localhost>
Description: Formula-image to LaTeX converter (Snip & Get)
 Local, free alternative to SimpleTex. Snip or paste a screenshot
 of a math formula and get editable LaTeX with a live preview.
 Choose local Gemma 4 E2B-it weights through llama.cpp or the
 hosted Gemma 4 Google AI Studio provider on first launch.
 .
 The snip tool uses Pillow and tkinter directly, with optional screenshot
 backends as fallbacks when the desktop session blocks Pillow access.
 .
 On first launch the app creates a Python environment. The local
 provider installs an auto-detected CPU, CUDA, ROCm, Vulkan, or SYCL
 llama.cpp runtime and downloads model weights (~4.1 GB);
 the cloud provider uses a Google AI Studio API key instead.
 LOCALTEX_BACKEND=lfm-vl or =pix2tex selects alternative backends.
EOF

cat > "$PKG/DEBIAN/postinst" <<'EOF'
#!/bin/sh
set -e
command -v update-desktop-database >/dev/null && update-desktop-database -q /usr/share/applications || true
command -v gtk-update-icon-cache >/dev/null && gtk-update-icon-cache -q /usr/share/icons/hicolor || true
exit 0
EOF
chmod 755 "$PKG/DEBIAN/postinst"

cat > "$PKG/DEBIAN/prerm" <<'EOF'
#!/bin/sh
# Stop a running KevinTex server so files aren't held open during removal.
set -e
pkill -f "uvicorn app:app.*8321" 2>/dev/null || true
exit 0
EOF
chmod 755 "$PKG/DEBIAN/prerm"

# --- build ---
OUT="$PROJ/dist"
mkdir -p "$OUT"
dpkg-deb --build --root-owner-group "$PKG" "$OUT/kevintex_${VERSION}_all.deb"
echo "Built: $OUT/kevintex_${VERSION}_all.deb"
