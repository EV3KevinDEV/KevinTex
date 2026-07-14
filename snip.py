#!/usr/bin/env python3
"""Self-contained screen-region snip for KevinTex.

Grabs the full screen with Pillow, shows a fullscreen tkinter overlay with the
screenshot as the background, lets the user drag a selection rectangle, then
crops and saves the selection as PNG. Prints the PNG path to stdout on success.

Works on X11 with no external screenshot tools (gnome-screenshot, flameshot,
maim, etc.) and no xdg-desktop-portal backend. Esc cancels.

Usage: snip.py <output_path>
"""

import math
import sys

from PIL import Image, ImageGrab, ImageTk
import tkinter as tk


def _display_size(
    capture_size: tuple[int, int],
    screen_size: tuple[int, int],
    platform: str | None = None,
) -> tuple[int, int]:
    """Choose the logical overlay size for a pixel-sized screen capture.

    macOS ImageGrab returns Retina pixels while Tk reports screen dimensions in
    logical points. Showing the capture at its raw pixel dimensions therefore
    zooms a 2x Retina image so that only its upper-left quadrant is visible.
    """
    platform = sys.platform if platform is None else platform
    screen_w, screen_h = screen_size
    if platform == "darwin" and screen_w > 0 and screen_h > 0:
        return screen_w, screen_h
    return capture_size


def _capture_crop_box(
    start: tuple[int, int],
    end: tuple[int, int],
    display_size: tuple[int, int],
    capture_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Map a selection in logical display coordinates to capture pixels."""
    display_w, display_h = display_size
    capture_w, capture_h = capture_size
    if min(display_w, display_h, capture_w, capture_h) <= 0:
        raise ValueError("screen dimensions must be positive")

    left, right = sorted((start[0], end[0]))
    top, bottom = sorted((start[1], end[1]))
    left = max(0, min(display_w, left))
    right = max(0, min(display_w, right))
    top = max(0, min(display_h, top))
    bottom = max(0, min(display_h, bottom))

    scale_x = capture_w / display_w
    scale_y = capture_h / display_h
    return (
        max(0, math.floor(left * scale_x)),
        max(0, math.floor(top * scale_y)),
        min(capture_w, math.ceil(right * scale_x)),
        min(capture_h, math.ceil(bottom * scale_y)),
    )


def main():
    if len(sys.argv) < 2:
        print("ERROR: missing output path", file=sys.stderr)
        sys.exit(2)
    out = sys.argv[1]

    try:
        full = ImageGrab.grab().convert("RGB")
    except Exception as e:
        print(f"ERROR: screen grab failed: {e}", file=sys.stderr)
        sys.exit(1)

    root = tk.Tk()
    screen_size = (root.winfo_screenwidth(), root.winfo_screenheight())
    display_size = _display_size(full.size, screen_size)
    display_w, display_h = display_size

    root.attributes("-fullscreen", True)
    root.configure(bg="black", cursor="crosshair")
    root.geometry(f"{display_w}x{display_h}+0+0")

    canvas = tk.Canvas(
        root, width=display_w, height=display_h, highlightthickness=0
    )
    canvas.pack(fill=tk.BOTH, expand=True)
    overlay = full
    if full.size != display_size:
        overlay = full.resize(display_size, Image.Resampling.LANCZOS)
    bg = ImageTk.PhotoImage(overlay, master=canvas)
    canvas.create_image(0, 0, anchor="nw", image=bg)

    state = {"start": None, "rect": None, "done": False}

    def on_press(e):
        state["start"] = (e.x, e.y)
        if state["rect"]:
            canvas.delete(state["rect"])
        state["rect"] = canvas.create_rectangle(
            e.x, e.y, e.x, e.y, outline="#3b82f6", width=2
        )

    def on_drag(e):
        if state["start"] and state["rect"]:
            canvas.coords(state["rect"], state["start"][0], state["start"][1], e.x, e.y)

    def on_release(e):
        if not state["start"] or state["done"]:
            return
        state["done"] = True
        x1, y1 = state["start"]
        x2, y2 = e.x, e.y
        root.destroy()

        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        if right - left < 4 or bottom - top < 4:
            print("ERROR: region too small", file=sys.stderr)
            sys.exit(1)

        crop_box = _capture_crop_box(
            (left, top), (right, bottom), display_size, full.size
        )
        crop = full.crop(crop_box)
        crop.save(out, "PNG")
        crop.close()
        print(out)

    def on_esc(_e):
        print("ERROR: cancelled", file=sys.stderr)
        root.destroy()
        sys.exit(1)

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("<Escape>", on_esc)
    root.focus_force()

    root.mainloop()


if __name__ == "__main__":
    main()
