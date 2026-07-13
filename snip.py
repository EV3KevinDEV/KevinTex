#!/usr/bin/env python3
"""Self-contained screen-region snip for KevinTex.

Grabs the full screen with Pillow, shows a fullscreen tkinter overlay with the
screenshot as the background, lets the user drag a selection rectangle, then
crops and saves the selection as PNG. Prints the PNG path to stdout on success.

Works on X11 with no external screenshot tools (gnome-screenshot, flameshot,
maim, etc.) and no xdg-desktop-portal backend. Esc cancels.

Usage: snip.py <output_path>
"""

import os
import sys

from PIL import Image, ImageGrab, ImageTk
import tkinter as tk


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

    w, h = full.size

    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.configure(bg="black", cursor="crosshair")
    root.geometry(f"{w}x{h}+0+0")

    canvas = tk.Canvas(root, width=w, height=h, highlightthickness=0)
    canvas.pack()
    bg = ImageTk.PhotoImage(full)
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

        crop = full.crop((left, top, right, bottom))
        crop.save(out, "PNG")
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
