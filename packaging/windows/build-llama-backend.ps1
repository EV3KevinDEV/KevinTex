param(
    [ValidateSet("cpu", "cuda", "rocm", "vulkan", "sycl")]
    [string]$Acceleration = "cpu",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Acceleration = $Acceleration.ToLowerInvariant()
$marker = Join-Path $PSScriptRoot "acceleration.txt"

function Invoke-Pip {
    & $Python -m pip @args
    if ($LASTEXITCODE -ne 0) {
        throw "pip failed while installing the $Acceleration llama.cpp backend."
    }
}

Write-Host "Installing llama.cpp backend: $Acceleration"

switch ($Acceleration) {
    "cpu" {
        Invoke-Pip install --upgrade --force-reinstall --no-cache-dir `
            --only-binary llama-cpp-python `
            "llama-cpp-python>=0.3.34" `
            --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
    }
    "cuda" {
        Invoke-Pip install --upgrade --force-reinstall --no-cache-dir torch `
            --index-url https://download.pytorch.org/whl/cu124
        Invoke-Pip install --upgrade --force-reinstall --no-cache-dir `
            --only-binary llama-cpp-python `
            "llama-cpp-python>=0.3.34" `
            --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
    }
    "vulkan" {
        if (-not (Get-Command glslc.exe -ErrorAction SilentlyContinue) -and -not $env:VULKAN_SDK) {
            throw "Vulkan SDK not found. Install it from vulkan.lunarg.com and reopen PowerShell."
        }
        Invoke-Pip install --upgrade cmake ninja
        $env:CMAKE_ARGS = "-DGGML_VULKAN=ON"
        $env:FORCE_CMAKE = "1"
        Invoke-Pip install --upgrade --force-reinstall --no-cache-dir `
            --no-binary llama-cpp-python `
            "llama-cpp-python>=0.3.34"
    }
    "rocm" {
        if (-not (Get-Command hipconfig.exe -ErrorAction SilentlyContinue) -and -not $env:HIP_PATH) {
            throw "AMD HIP SDK not found. Install the Windows HIP SDK and reopen PowerShell."
        }
        Invoke-Pip install --upgrade cmake ninja
        $env:CMAKE_GENERATOR = "Ninja"
        $env:CMAKE_ARGS = "-DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release"
        $env:FORCE_CMAKE = "1"
        Invoke-Pip install --upgrade --force-reinstall --no-cache-dir `
            --no-binary llama-cpp-python `
            "llama-cpp-python>=0.3.34"
    }
    "sycl" {
        if (-not (Get-Command icx.exe -ErrorAction SilentlyContinue) -or -not (Get-Command icx-cl.exe -ErrorAction SilentlyContinue)) {
            throw "Intel oneAPI compiler environment not found. Run this from an Intel oneAPI command prompt."
        }
        Invoke-Pip install --upgrade cmake ninja
        $env:CMAKE_GENERATOR = "Ninja"
        $env:CMAKE_ARGS = "-DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx"
        $env:FORCE_CMAKE = "1"
        Invoke-Pip install --upgrade --force-reinstall --no-cache-dir `
            --no-binary llama-cpp-python `
            "llama-cpp-python>=0.3.34"
    }
}

Set-Content -Path $marker -Value $Acceleration -Encoding ascii
Write-Host "Wrote $marker. Build KevinTex with PyInstaller next."
