# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onedir build for the KevinTex Windows desktop application."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules


project = Path(SPECPATH).resolve().parents[1]

datas = [
    (str(project / "static"), "static"),
    (str(project / "snip.py"), "."),
]
acceleration_marker = project / "packaging" / "windows" / "acceleration.txt"
if acceleration_marker.exists():
    datas.append((str(acceleration_marker), "."))
binaries = []
hiddenimports = [
    "backend_gemma",
    "backend_gemma_cloud",
    "backend_vlm",
    "image_preprocessing",
    "provider_config",
    "snip",
    "google.genai",
    "google.genai.types",
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
]

for package in ("llama_cpp", "webview"):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

hiddenimports += collect_submodules("webview")
hiddenimports += collect_submodules("google.genai")

icon_path = project / "packaging" / "windows" / "kevintex.ico"
version_path = project / "packaging" / "windows" / "version_info.txt"

a = Analysis(
    [str(project / "desktop.py")],
    pathex=[str(project)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "accelerate",
        "pix2tex",
        "torchvision",
        "transformers",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="KevinTex",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(icon_path) if icon_path.exists() else None,
    version=str(version_path) if version_path.exists() else None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="KevinTex",
)
