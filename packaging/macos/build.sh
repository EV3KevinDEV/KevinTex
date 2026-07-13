#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VERSION="${KEVINTEX_VERSION:-0.0.0}"
ARTIFACT_DIR="$ROOT/dist/macos"
STAGE_DIR="$ROOT/build/dmg"

cd "$ROOT"
rm -rf build/KevinTex dist/KevinTex dist/KevinTex.app "$STAGE_DIR"
mkdir -p "$ARTIFACT_DIR" "$STAGE_DIR"

# Build a native multi-resolution macOS icon from the supplied KevinTex art.
ICONSET="$ROOT/build/KevinTex.iconset"
rm -rf "$ICONSET"
mkdir -p "$ICONSET"
for SIZE in 16 32 128 256 512; do
  DOUBLE=$((SIZE * 2))
  sips -z "$SIZE" "$SIZE" assets/kevintex-icon.png \
    --out "$ICONSET/icon_${SIZE}x${SIZE}.png" >/dev/null
  sips -z "$DOUBLE" "$DOUBLE" assets/kevintex-icon.png \
    --out "$ICONSET/icon_${SIZE}x${SIZE}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o packaging/macos/kevintex.icns

KEVINTEX_VERSION="$VERSION" pyinstaller \
  --noconfirm --clean --distpath dist --workpath build/KevinTex \
  packaging/macos/KevinTex.spec

# Ad-hoc signing makes the nested Mach-O code internally consistent. Release
# builds remain unnotarized unless a downstream maintainer adds Developer ID
# signing credentials.
codesign --force --deep --sign - dist/KevinTex.app
dist/KevinTex.app/Contents/MacOS/KevinTex --smoke-test

ditto -c -k --sequesterRsrc --keepParent \
  dist/KevinTex.app "$ARTIFACT_DIR/KevinTex-${VERSION}-macOS-arm64.zip"

cp -R dist/KevinTex.app "$STAGE_DIR/"
ln -s /Applications "$STAGE_DIR/Applications"
hdiutil create -volname "KevinTex" -srcfolder "$STAGE_DIR" \
  -ov -format UDZO "$ARTIFACT_DIR/KevinTex-${VERSION}-macOS-arm64.dmg"

echo "Built macOS artifacts in $ARTIFACT_DIR"
