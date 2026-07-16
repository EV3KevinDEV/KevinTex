"""CUDA runtime discovery tests."""

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import backend_gemma


class CudaRuntimeTests(unittest.TestCase):
    def test_windows_torch_lib_is_kept_on_dll_search_path(self):
        with tempfile.TemporaryDirectory() as root:
            torch_dir = Path(root, "torch")
            torch_lib = torch_dir / "lib"
            torch_lib.mkdir(parents=True)
            fake_torch = types.SimpleNamespace(__file__=str(torch_dir / "__init__.py"))
            dll_handle = object()

            backend_gemma._CUDA_DLL_DIR_HANDLES.clear()
            with (
                patch.dict(sys.modules, {"torch": fake_torch}),
                patch.object(backend_gemma.os, "name", "nt"),
                patch.object(
                    backend_gemma.os,
                    "add_dll_directory",
                    return_value=dll_handle,
                    create=True,
                ) as add_dll_directory,
                patch.dict(os.environ, {"PATH": "existing"}, clear=True),
            ):
                backend_gemma._preload_cuda_libs()

                add_dll_directory.assert_called_once_with(str(torch_lib))
                self.assertEqual(backend_gemma._CUDA_DLL_DIR_HANDLES, [dll_handle])
                self.assertEqual(
                    os.environ["PATH"], f"{torch_lib}{os.pathsep}existing"
                )


if __name__ == "__main__":
    unittest.main()
