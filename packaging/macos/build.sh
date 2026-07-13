#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VERSION="${KEVINTEX_VERSION:-0.0.0}"
ARTIFACT_DIR="$ROOT/dist/macos"
STAGE_DIR="$ROOT/build/dmg"

cd "$ROOT"
rm -rf build/KevinTex dist/KevinTex dist/KevinTex.app "$STAGE_DIR"
mkdir -p "$ARTIFACT_DIR" "$STAGE_DIR"

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
