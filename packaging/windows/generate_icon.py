"""Generate a multi-resolution Windows icon without platform-specific tools."""

from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    size = 256
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((8, 8, 248, 248), radius=52, fill="#2563eb")

    mark = 38
    end = 78
    width = 10
    for points in (
        ((mark, end), (mark, mark), (end, mark)),
        ((size - end, mark), (size - mark, mark), (size - mark, end)),
        ((mark, size - end), (mark, size - mark), (end, size - mark)),
        ((size - end, size - mark), (size - mark, size - mark), (size - mark, size - end)),
    ):
        draw.line(points, fill="white", width=width, joint="curve")

    # Font-independent sigma matching the application's SVG icon.
    draw.polygon(
        [
            (88, 76),
            (168, 76),
            (168, 98),
            (116, 98),
            (152, 142),
            (116, 186),
            (168, 186),
            (168, 208),
            (88, 208),
            (88, 188),
            (128, 142),
            (88, 96),
        ],
        fill="white",
    )

    output = Path(__file__).with_name("kevintex.ico")
    image.save(output, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (256, 256)])


if __name__ == "__main__":
    main()
