#!/usr/bin/env bash
# Build kevintex_<version>_all.deb from the project sources.
set -euo pipefail

VERSION="1.2.0"
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

PKG="$STAGE/kevintex_${VERSION}_all"

# --- payload ---
install -d "$PKG/opt/kevintex" "$PKG/usr/bin" \
           "$PKG/usr/share/applications" \
           "$PKG/usr/share/icons/hicolor/scalable/apps" \
           "$PKG/usr/share/doc/kevintex"

cp "$PROJ/app.py" "$PROJ/backend_vlm.py" "$PROJ/backend_gemma.py" "$PROJ/image_preprocessing.py" "$PROJ/snip.py" "$PROJ/requirements.txt" "$PKG/opt/kevintex/"
cp -r "$PROJ/static" "$PKG/opt/kevintex/static"
cp "$PROJ/README.md" "$PKG/usr/share/doc/kevintex/"
install -m 755 "$PROJ/packaging/localtex-launcher" "$PKG/usr/bin/kevintex"
install -m 644 "$PROJ/packaging/localtex.desktop" "$PKG/usr/share/applications/kevintex.desktop"
install -m 644 "$PROJ/packaging/localtex.svg" "$PKG/usr/share/icons/hicolor/scalable/apps/kevintex.svg"

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
Description: Offline formula-image to LaTeX converter (Snip & Get)
 Local, free, unlimited alternative to SimpleTex. Snip or paste a
 screenshot of a math formula and get editable LaTeX with a live
 preview. Recognition runs entirely on-device with Google's
 Gemma 4 E2B-it multimodal model (Q4_K_M GGUF + vision projector)
 via llama.cpp with CUDA offload; no cloud services are used.
 .
 The snip tool is self-contained (Pillow + tkinter region selector),
 no external screenshot utility required.
 .
 On first launch the app creates a Python environment and downloads
 PyTorch, llama.cpp and the Gemma 4 model weights (one time, ~8 GB).
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
