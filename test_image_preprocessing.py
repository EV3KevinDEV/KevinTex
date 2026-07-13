"""Focused generated-image tests for KevinTex preprocessing."""

import asyncio
import io
import unittest

from PIL import Image, ImageDraw, ImageStat
from starlette.datastructures import UploadFile

from image_preprocessing import preprocess_image, validate_options


class PreprocessingTests(unittest.TestCase):
    def formula_image(self, size=(320, 100)):
        image = Image.new("RGB", size, (224, 224, 224))
        draw = ImageDraw.Draw(image)
        draw.line((30, 52, 280, 52), fill=(105, 105, 105), width=2)
        draw.text((70, 20), "x + 1", fill=(95, 95, 95))
        draw.text((80, 60), "y - 2", fill=(95, 95, 95))
        return image

    def test_off_preserves_pixels_and_size_without_transforms(self):
        source = self.formula_image()
        result = preprocess_image(source, mode="off")
        self.assertEqual(result.size, source.size)
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.getpixel((0, 0)), source.getpixel((0, 0)))

    def test_auto_upscales_and_increases_contrast(self):
        source = self.formula_image()
        result = preprocess_image(source, mode="auto")
        self.assertGreater(result.width, source.width)
        self.assertGreater(
            ImageStat.Stat(result.convert("L")).rms[0],
            ImageStat.Stat(source.convert("L")).rms[0],
        )

    def test_rotation_and_inversion_are_available_when_off(self):
        source = Image.new("RGB", (30, 10), "black")
        result = preprocess_image(source, mode="off", rotation=90, invert=True)
        self.assertEqual(result.size, (10, 30))
        self.assertEqual(result.getpixel((0, 0)), (255, 255, 255))

    def test_transparency_is_flattened_on_white(self):
        source = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
        result = preprocess_image(source, mode="strong")
        self.assertEqual(result.getpixel((0, 0)), (255, 255, 255))

    def test_invalid_api_options_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "off, auto, strong"):
            validate_options("maximum", 0)
        with self.assertRaisesRegex(ValueError, "0, 90, 180, 270"):
            validate_options("auto", 45)

    def test_preprocess_api_runs_without_ocr_model(self):
        from app import preprocess_preview

        source = self.formula_image()
        encoded = io.BytesIO()
        source.save(encoded, "PNG")
        upload = UploadFile(io.BytesIO(encoded.getvalue()), filename="generated.png")
        response = asyncio.run(
            preprocess_preview(upload, preprocess="auto", rotation=90, invert=False)
        )
        prepared = Image.open(io.BytesIO(response.body))
        self.assertEqual(response.media_type, "image/png")
        self.assertEqual(response.headers["x-kevintex-preprocess-mode"], "auto")
        self.assertGreater(prepared.height, prepared.width)


if __name__ == "__main__":
    unittest.main()
