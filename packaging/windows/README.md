# KevinTex for Windows

The release workflow builds a native-window, 64-bit Windows application using
pywebview, an embedded loopback Uvicorn server, and PyInstaller's `onedir`
layout. Extract the ZIP and run `KevinTex.exe`; no separate Python installation
or browser launch is required.

The default artifact uses llama.cpp's optimized x64 CPU wheel. The release
workflow can also produce CUDA 12.4 for NVIDIA GPUs and Vulkan for compatible
Intel and AMD GPUs. Each build contains an `acceleration.txt` marker so the app
uses the matching native runtime and reports it in the status badge.

The CUDA runtime is larger than GitHub's single-file release limit, so that
flavor is published as numbered `.7z.001`, `.7z.002`, … volumes. Download every
part into one folder and open `.7z.001` with 7-Zip. The standard CPU build
remains a single ZIP. The Vulkan flavor is also a single ZIP.

On first launch, the Gemma GGUF files (about 4.1 GB) download to:

```text
%LOCALAPPDATA%\KevinTex\models
```

Hugging Face cache data is stored under `%LOCALAPPDATA%\KevinTex\cache`.
Weights are deliberately absent from the ZIP.

To build locally from a Windows PowerShell prompt, install the normal
dependencies and then select a native backend before running PyInstaller:

```powershell
py -3.11 -m pip install -r requirements-windows.txt
& .\packaging\windows\build-llama-backend.ps1 -Acceleration cpu -Python py
py packaging/windows/generate_icon.py
py packaging/windows/generate_version_info.py v0.0.0
pyinstaller --clean --noconfirm packaging/windows/kevintex.spec
Copy-Item -Force packaging/windows/KevinTex.exe.config dist/KevinTex/KevinTex.exe.config
```

The sidecar `.config` file allows the bundled Python.NET assembly to load when
Windows preserves the downloaded archive's web-origin security marker during
extraction. Keep it next to `KevinTex.exe` when distributing or moving the app.

Valid backend names are `cpu`, `cuda`, `vulkan`, `rocm`, and `sycl`. Vulkan
requires the LunarG Vulkan SDK. ROCm requires AMD's Windows HIP SDK. SYCL must
be built from an Intel oneAPI command prompt. HIP and oneAPI are local build
modes because those vendor SDKs are not installed on standard GitHub-hosted
Windows runners. Vulkan is the simplest portable GPU option for most supported
Intel and AMD Windows systems.
