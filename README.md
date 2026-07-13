# KevinTex — Snip & Get, fully offline

A local, free, unlimited formula-image → LaTeX converter (a self-hosted
alternative to SimpleTex). Paste, drop, or **snip** a screenshot of a math
formula and get editable Markdown + LaTeX with a live rendered preview.
Recognition runs entirely on your machine using the quantized
[Gemma 4 E2B-it](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF)
vision-language model through llama.cpp — no cloud APIs, accounts, or quotas.

## Features

- **Snip button** (the computer-with-+ icon) — click it, drag a screen region,
  and the capture is converted instantly. Fully self-contained (Pillow +
  tkinter region selector); no `gnome-screenshot`/`flameshot`/portal needed.
- **Paste (Ctrl+V), drag-and-drop, or browse** for a formula image (PNG/JPG/BMP/WEBP)
- **Live KaTeX preview** (bundled locally — the app works with no internet at all)
- **Editable output** — fix the LaTeX by hand and the preview updates
- **Copy formats**: raw, `$…$`, `$$…$$`, `\[…\]`, `\(…\)`, `\begin{equation}…\end{equation}`, Markdown
- **Auto-copy after recognition** (configurable in Settings)
- **Thinking mode** for deeper multi-pass OCR verification on dense pages
- **History** with search, individual deletion, and safe clear-all controls
- **Recognize again** using the current source image and active Thinking setting
- **Download `.tex`** output and keyboard shortcuts (`Ctrl/Cmd+K`, `Ctrl/Cmd+S`)
- **GPU accelerated** when CUDA is available (falls back to CPU)
- **Switchable backend** — Gemma is the default; `LOCALTEX_BACKEND=lfm-vl` or
  `LOCALTEX_BACKEND=pix2tex` selects an alternative local backend.

## Install as an Ubuntu desktop app

Two options:

**System-wide (.deb):**

```bash
packaging/build-deb.sh
sudo apt install ./dist/kevintex_1.0.0_all.deb
```

**Current user only (no root):**

```bash
packaging/install-user.sh
```

Either way you get a **KevinTex** entry in the applications menu with its own
icon. Launching it starts the local server and opens the app in its own
native window (Chrome/Chromium app mode; falls back to your browser).
Closing the window stops the server. On a machine without the model
environment, the first launch shows a one-time setup dialog that downloads
PyTorch, llama.cpp, and the Gemma GGUF weights.

The app window's **Snip** button (computer-with-+ icon) opens a fullscreen
region selector: drag a box around a formula and it's converted immediately —
the LaTeX lands in your clipboard if auto-copy is on. No external screenshot
tool required.

## Run from source (dev)

```bash
./run.sh
```

Then open http://127.0.0.1:8321 (the script opens it for you). The first
launch downloads the Gemma weights (one time); after that it
is fully offline.

## Manual setup (if moving to another machine)

```bash
python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app:app --port 8321
```

## Tips for best recognition

- The Gemma vision model handles full formulas, sums, integrals, matrices, and
  mixed text+math better than the old formula-only model. Still, crop fairly
  tightly around the formula for best fidelity.
- Higher-resolution, high-contrast screenshots work best.
- Use the **Snip** button for the fastest workflow, or bind a region-screenshot
  hotkey (e.g. GNOME `Shift+PrtSc`) to copy a region to the clipboard, then
  Ctrl+V into the app.
