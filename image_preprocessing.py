"""Conservative image preparation for KevinTex OCR.

The pipeline intentionally avoids thresholding and morphology: those operations
often erase fraction bars, radicals, and light subscripts.  Pillow is enough for
the useful operations here and keeps the desktop package lightweight.
"""

from __future__ import annotations

from typing import Literal

from PIL import Image, ImageFilter, ImageOps

PreprocessMode = Literal["off", "auto", "strong"]
VALID_MODES = frozenset(("off", "auto", "strong"))
VALID_ROTATIONS = frozenset((0, 90, 180, 270))
MAX_INPUT_PIXELS = 50_000_000
MAX_OUTPUT_EDGE = 3600


def validate_options(mode: str, rotation: int) -> tuple[PreprocessMode, int]:
    """Validate API-facing preprocessing options."""
    normalized = str(mode).strip().lower()
    if normalized not in VALID_MODES:
        raise ValueError("preprocess must be one of: off, auto, strong")
    if rotation not in VALID_ROTATIONS:
        raise ValueError("rotation must be one of: 0, 90, 180, 270")
    return normalized, rotation  # type: ignore[return-value]


def _flatten_to_rgb(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation and flatten transparency onto formula-white."""
    image = ImageOps.exif_transpose(image)
    if image.width * image.height > MAX_INPUT_PIXELS:
        raise ValueError("Image is too large (50 megapixels maximum).")
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, "white")
        return Image.alpha_composite(white, rgba).convert("RGB")
    return image.convert("RGB")


def _upscale_factor(width: int, height: int, mode: PreprocessMode) -> float:
    """Choose a useful OCR scale while bounding memory and model input size."""
    short = min(width, height)
    desired_short = 900 if mode == "strong" else 700
    limit = MAX_OUTPUT_EDGE / max(width, height)
    return max(1.0, min(3.0, desired_short / max(short, 1), limit))


def preprocess_image(
    image: Image.Image,
    mode: str = "auto",
    rotation: int = 0,
    invert: bool = False,
) -> Image.Image:
    """Return an OCR-ready image using conservative, formula-safe enhancement.

    ``off`` still applies EXIF orientation, requested rotation, transparency
    flattening, and optional inversion. ``auto`` normalizes contrast, gently
    upscales small captures, and sharpens edges. ``strong`` adds a small median
    denoise and stronger normalization/sharpening for noisy scans.
    """
    selected, rotation = validate_options(mode, rotation)
    result = _flatten_to_rgb(image)

    if rotation:
        # The UI describes positive values as clockwise.
        result = result.rotate(-rotation, expand=True, fillcolor="white")
    if invert:
        result = ImageOps.invert(result)
    if selected == "off":
        return result

    gray = ImageOps.grayscale(result)
    cutoff = 1.0 if selected == "strong" else 0.35
    gray = ImageOps.autocontrast(gray, cutoff=cutoff)

    if selected == "strong" and min(gray.size) >= 24:
        # Median filtering before scaling removes isolated scan noise without
        # repeatedly softening already-upscaled character edges.
        gray = gray.filter(ImageFilter.MedianFilter(size=3))

    factor = _upscale_factor(*gray.size, selected)
    if factor > 1.01:
        size = (
            max(1, round(gray.width * factor)),
            max(1, round(gray.height * factor)),
        )
        gray = gray.resize(size, Image.Resampling.LANCZOS)

    if selected == "strong":
        gray = gray.filter(ImageFilter.UnsharpMask(radius=1.4, percent=190, threshold=2))
    else:
        gray = gray.filter(ImageFilter.UnsharpMask(radius=1.0, percent=125, threshold=3))
    return gray.convert("RGB")
