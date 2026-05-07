"""Unit tests for the offline prefetch CLI entry points."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from fattervoice.prefetch import main as run_prefetch_cli


class PrefetchCliTests(unittest.TestCase):
    """Verify that the prefetch module can be executed directly during Docker builds."""

    def test_prefetch_cli_rejects_removed_cache_and_manifest_flags(self) -> None:
        """Ensure removed prefetch CLI flags now fail fast during parsing.

        Usage:
            The wrapper no longer exposes custom cache-dir or manifest flags on
            the prefetch helper, so this regression test verifies that old
            invocation patterns now raise `SystemExit` instead of being silently
            accepted.

        Parameters:
            None.

        Returns:
            None. The test asserts that argparse raises `SystemExit` for each
            removed flag.
        """
        with self.assertRaises(SystemExit):
            run_prefetch_cli(["--cache-dir", "/tmp/hf-cache"])

        with self.assertRaises(SystemExit):
            run_prefetch_cli(["--manifest", "/tmp/prefetched-models.json"])

    def test_python_module_invocation_runs_prefetch_main(self) -> None:
        """Ensure `python -m fattervoice.prefetch` performs the requested work.

        Usage:
            The Dockerfile invokes the prefetch helper as a Python module before
            the project itself is installed as a console script. This test guards
            that path by executing the module in a subprocess and asserting that
            it resolves a local model directory without trying to write any extra
            wrapper-managed manifest.

        Parameters:
            None.

        Returns:
            None. The test asserts on the subprocess exit code and stdout.
        """
        repository_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            local_model_path = Path(temp_dir) / "local-model"
            local_model_path.mkdir()
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
                    "fattervoice.prefetch",
                    "--model",
                    str(local_model_path),
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


if __name__ == "__main__":
    unittest.main()
