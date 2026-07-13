# KevinTex for Windows

The release workflow builds a native-window, 64-bit Windows application using
pywebview, an embedded loopback Uvicorn server, and PyInstaller's `onedir`
layout. Extract the ZIP and run `KevinTex.exe`; no separate Python installation
or browser launch is required.

The default artifact uses llama.cpp's CPU wheel. A manually dispatched workflow
can select the best-effort CUDA 12.4 flavor, which is marked at build time and
requires a compatible NVIDIA driver. Source launches can opt into an installed
CUDA llama.cpp runtime with `KEVINTEX_CUDA=1`.

The CUDA runtime is larger than GitHub's single-file release limit, so that
flavor is published as numbered `.7z.001`, `.7z.002`, … volumes. Download every
part into one folder and open `.7z.001` with 7-Zip. The standard CPU build
remains a single ZIP.

On first launch, the Gemma GGUF files (about 4.1 GB) download to:

```text
%LOCALAPPDATA%\KevinTex\models
```

Hugging Face cache data is stored under `%LOCALAPPDATA%\KevinTex\cache`.
Weights are deliberately absent from the ZIP.

To build locally from a Windows PowerShell prompt:

```powershell
py -3.11 -m pip install -r requirements-windows.txt
py packaging/windows/generate_icon.py
py packaging/windows/generate_version_info.py v0.0.0
pyinstaller --clean --noconfirm packaging/windows/kevintex.spec
```
