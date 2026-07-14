"""Regression tests for resilient macOS MLX model downloads."""

import os
import unittest
from unittest.mock import Mock, patch

import backend_mlx


class MLXDownloadTests(unittest.TestCase):
    def test_xet_is_disabled_before_huggingface_import(self):
        self.assertEqual(os.environ.get("HF_HUB_DISABLE_XET"), "1")

    def test_xet_401_refreshes_download_once(self):
        download = Mock(
            side_effect=[
                RuntimeError("401 Unauthorized from cas-bridge.xethub.hf.co"),
                "/tmp/model",
            ]
        )
        with patch.object(backend_mlx.time, "sleep") as sleep:
            backend_mlx._download_model_snapshot(download)

        self.assertEqual(download.call_count, 2)
        sleep.assert_called_once_with(1.0)
        for call in download.call_args_list:
            self.assertEqual(call.kwargs["max_workers"], 4)
            self.assertEqual(call.kwargs["etag_timeout"], 30)

    def test_persistent_xet_401_has_short_actionable_error(self):
        download = Mock(
            side_effect=RuntimeError(
                "401 Unauthorized https://cas-bridge.xethub.hf.co/very-long-url"
            )
        )
        with patch.object(backend_mlx.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "Date & Time") as raised:
                backend_mlx._download_model_snapshot(download)

        self.assertEqual(download.call_count, 2)
        self.assertNotIn("https://", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
