"""Windows launcher acceleration-selection tests."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

import desktop


class DesktopAccelerationTests(unittest.TestCase):
    def test_windows_bundle_allows_web_marked_managed_assemblies(self):
        config = Path(__file__).parent / "packaging/windows/KevinTex.exe.config"
        setting = ElementTree.parse(config).find("./runtime/loadFromRemoteSources")

        self.assertIsNotNone(setting)
        self.assertEqual(setting.attrib.get("enabled"), "true")

    def test_windows_interop_dependencies_are_reproducibly_pinned(self):
        requirements = (Path(__file__).parent / "requirements-windows.txt").read_text(
            encoding="utf-8"
        )

        for dependency in (
            "pyinstaller==6.21.0",
            "pywebview==6.2.1",
            "pythonnet==3.0.5",
            "clr-loader==0.2.7.post0",
        ):
            self.assertIn(dependency, requirements)

    def test_windows_smoke_check_imports_winforms_backend(self):
        with patch.object(desktop.sys, "platform", "win32"), patch(
            "desktop.importlib.import_module"
        ) as import_module:
            desktop._verify_native_window_backend()

        import_module.assert_called_once_with("webview.platforms.winforms")

    def test_non_windows_smoke_check_skips_winforms_backend(self):
        with patch.object(desktop.sys, "platform", "linux"), patch(
            "desktop.importlib.import_module"
        ) as import_module:
            desktop._verify_native_window_backend()

        import_module.assert_not_called()

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
