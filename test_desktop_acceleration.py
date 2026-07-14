"""Windows launcher acceleration-selection tests."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import desktop


class DesktopAccelerationTests(unittest.TestCase):
    def test_cpu_is_safe_default(self):
        with tempfile.TemporaryDirectory() as bundle, patch.object(
            sys, "_MEIPASS", bundle, create=True
        ), patch.dict(os.environ, {}, clear=True):
            self.assertEqual(desktop._bundled_acceleration(), "cpu")

    def test_packaged_acceleration_marker_is_used(self):
        with tempfile.TemporaryDirectory() as bundle:
            Path(bundle, "acceleration.txt").write_text("vulkan\n", encoding="utf-8")
            with patch.object(sys, "_MEIPASS", bundle, create=True), patch.dict(
                os.environ, {}, clear=True
            ):
                self.assertEqual(desktop._bundled_acceleration(), "vulkan")

    def test_packaged_marker_wins_over_environment_override(self):
        with tempfile.TemporaryDirectory() as bundle:
            Path(bundle, "acceleration.txt").write_text("cpu\n", encoding="utf-8")
            with patch.object(sys, "_MEIPASS", bundle, create=True), patch.dict(
                os.environ, {"KEVINTEX_ACCELERATION": "ROCM"}, clear=True
            ):
                self.assertEqual(desktop._bundled_acceleration(), "cpu")

    def test_environment_selects_a_source_build_without_marker(self):
        with tempfile.TemporaryDirectory() as bundle, patch.object(
            sys, "_MEIPASS", bundle, create=True
        ), patch.dict(
            os.environ, {"KEVINTEX_ACCELERATION": "ROCM"}, clear=True
        ):
            self.assertEqual(desktop._bundled_acceleration(), "rocm")

    def test_legacy_cuda_marker_remains_compatible(self):
        with tempfile.TemporaryDirectory() as bundle:
            Path(bundle, "cuda_enabled.txt").write_text("CUDA build", encoding="utf-8")
            with patch.object(sys, "_MEIPASS", bundle, create=True), patch.dict(
                os.environ, {}, clear=True
            ):
                self.assertEqual(desktop._bundled_acceleration(), "cuda")


if __name__ == "__main__":
    unittest.main()
