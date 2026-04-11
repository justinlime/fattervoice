"""Unit tests for the offline prefetch CLI entry points."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PrefetchCliTests(unittest.TestCase):
    """Verify that the prefetch module can be executed directly during Docker builds."""

    def test_python_module_invocation_runs_prefetch_main_and_writes_manifest(self) -> None:
        """Ensure `python -m fatterqwen.prefetch` performs the requested work.

        Usage:
            The Dockerfile invokes the prefetch helper as a Python module before
            the project itself is installed as a console script. This test guards
            that path by executing the module in a subprocess and asserting that
            it writes the requested manifest for a local model directory.

        Parameters:
            None.

        Returns:
            None. The test asserts on the subprocess exit code, stdout, and
            manifest contents.
        """
        repository_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            local_model_path = Path(temp_dir) / "local-model"
            local_model_path.mkdir()
            manifest_path = Path(temp_dir) / "prefetched-models.json"
            stub_module_root = Path(temp_dir) / "stubs"
            stub_module_root.mkdir()
            (stub_module_root / "huggingface_hub.py").write_text(
                "def snapshot_download(*args, **kwargs):\n"
                "    raise AssertionError('snapshot_download should not run for local model paths.')\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            existing_python_path = environment.get("PYTHONPATH", "")
            combined_python_path = os.pathsep.join(
                [
                    str(stub_module_root),
                    str(repository_root),
                    existing_python_path,
                ]
            )
            environment["PYTHONPATH"] = combined_python_path.rstrip(os.pathsep)

            completed_process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "fatterqwen.prefetch",
                    "--model",
                    str(local_model_path),
                    "--manifest",
                    str(manifest_path),
                ],
                capture_output=True,
                check=False,
                cwd=repository_root,
                env=environment,
                text=True,
            )

            self.assertEqual(completed_process.returncode, 0, completed_process.stderr)
            self.assertEqual(completed_process.stderr, "")
            self.assertIn(str(local_model_path.resolve()), completed_process.stdout)
            self.assertTrue(manifest_path.exists())
            self.assertEqual(
                json.loads(manifest_path.read_text(encoding="utf-8")),
                {
                    "model_paths": {
                        str(local_model_path): str(local_model_path.resolve()),
                    }
                },
            )


if __name__ == "__main__":
    unittest.main()
