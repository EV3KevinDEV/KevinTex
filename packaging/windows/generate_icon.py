"""Generate a multi-resolution Windows icon from the KevinTex artwork."""

from pathlib import Path

from PIL import Image


def main() -> None:
    project = Path(__file__).resolve().parents[2]
    image = Image.open(project / "assets" / "kevintex-icon.png").convert("RGBA")
    output = Path(__file__).with_name("kevintex.ico")
    image.save(output, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (256, 256)])


if __name__ == "__main__":
    main()
