# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller definition for the unsigned Apple Silicon KevinTex.app."""

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules


version = os.environ.get("KEVINTEX_VERSION", "0.0.0")
project = Path(SPECPATH).resolve().parents[1]
mlx_datas, mlx_binaries, mlx_hidden = collect_all("mlx")
vlm_datas, vlm_binaries, vlm_hidden = collect_all("mlx_vlm")

a = Analysis(
    [str(project / "packaging" / "macos" / "desktop_macos.py")],
    pathex=[str(project)],
    binaries=mlx_binaries + vlm_binaries,
    datas=[
        (str(project / "static"), "static"),
        (str(project / "snip.py"), "."),
    ]
    + mlx_datas
    + vlm_datas,
    hiddenimports=[
        "app",
        "backend_mlx",
        "backend_vlm",
        "image_preprocessing",
        "snip",
        "tkinter",
        "PIL.ImageTk",
        "webview.platforms.cocoa",
    ]
    + mlx_hidden
    + vlm_hidden
    + collect_submodules("mlx_vlm.models.gemma4"),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "backend_gemma",
        "llama_cpp",
        "pix2tex",
        "torch",
        "torchvision",
    ],
    noarchive=False,
    optimize=1,
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
    upx=False,
    console=False,
    argv_emulation=False,
    target_arch="arm64",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="KevinTex",
)

app = BUNDLE(
    coll,
    name="KevinTex.app",
    icon=str(project / "packaging" / "macos" / "kevintex.icns"),
    bundle_identifier="app.kevintex.desktop",
    info_plist={
        "CFBundleDisplayName": "KevinTex",
        "CFBundleName": "KevinTex",
        "CFBundleShortVersionString": version,
        "CFBundleVersion": version,
        "LSMinimumSystemVersion": "14.0",
        "LSApplicationCategoryType": "public.app-category.productivity",
        "NSHighResolutionCapable": True,
        "NSScreenCaptureUsageDescription": (
            "KevinTex needs screen recording access when you use Snip & Get."
        ),
    },
)
