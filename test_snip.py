"""Screen-coordinate tests for the cross-platform snip overlay."""

import unittest
from unittest.mock import patch

from PIL import Image
from snip import _capture_crop_box, _capture_screen, _display_size


class SnipCoordinateTests(unittest.TestCase):
    def test_retina_capture_uses_logical_macos_screen_size(self):
        self.assertEqual(
            _display_size((3024, 1964), (1512, 982), platform="darwin"),
            (1512, 982),
        )

    def test_retina_selection_maps_back_to_capture_pixels(self):
        self.assertEqual(
            _capture_crop_box(
                (100, 50),
                (300, 150),
                display_size=(1512, 982),
                capture_size=(3024, 1964),
            ),
            (200, 100, 600, 300),
        )

    def test_crop_coordinates_are_ordered_and_clamped(self):
        self.assertEqual(
            _capture_crop_box(
                (900, 700),
                (-10, -20),
                display_size=(800, 600),
                capture_size=(1600, 1200),
            ),
            (0, 0, 1600, 1200),
        )

    def test_non_macos_keeps_capture_dimensions(self):
        self.assertEqual(
            _display_size((1920, 1080), (1920, 1080), platform="linux"),
            (1920, 1080),
        )

    def test_linux_capture_falls_back_when_pillow_cannot_access_desktop(self):
        fallback = Image.new("RGB", (64, 32), "black")
        with patch("snip.sys.platform", "linux"):
            with patch(
                "snip.ImageGrab.grab", side_effect=OSError("desktop unavailable")
            ):
                with patch("snip._capture_with_command", return_value=fallback):
                    captured = _capture_screen()
        try:
            self.assertEqual(captured.size, (64, 32))
        finally:
            captured.close()


if __name__ == "__main__":
    unittest.main()
