"""Unit tests for prefetched-model manifest helpers."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

from fatterqwen.prefetch_manifest import (
    read_prefetch_manifest,
    resolve_cached_model_snapshot_path,
    resolve_prefetched_model_path,
    write_prefetch_manifest,
)


class PrefetchManifestTests(unittest.TestCase):
    """Verify that prefetched model manifests can redirect runtime loading to local snapshots."""

    def test_write_and_read_prefetch_manifest(self) -> None:
        """Ensure the manifest records exact model IDs and local snapshot paths.

        Usage:
            This test protects the offline Docker flow by verifying that the JSON
            manifest written during prefetch can be read back into the same model
            ID to local path mapping at runtime.

        Parameters:
            None.

        Returns:
            None. The test asserts on the manifest contents after a write/read round-trip.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "prefetched-models.json"
            snapshot_path = Path(temp_dir) / "snapshot"
            snapshot_path.mkdir()

            write_prefetch_manifest(
                {"Qwen/Qwen3-TTS-12Hz-1.7B-Base": snapshot_path},
                manifest_path,
            )

            self.assertEqual(
                read_prefetch_manifest(manifest_path),
                {"Qwen/Qwen3-TTS-12Hz-1.7B-Base": str(snapshot_path.resolve())},
            )

    def test_resolve_prefetched_model_path_prefers_existing_local_snapshot(self) -> None:
        """Ensure runtime loading switches to the exact local snapshot when present.

        Usage:
            This test protects the offline startup path by verifying that a remote
            Hugging Face model ID is replaced with a concrete local snapshot path
            whenever the manifest contains a valid existing directory.

        Parameters:
            None.

        Returns:
            None. The test asserts that the resolved path points at the local snapshot.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "prefetched-models.json"
            snapshot_path = Path(temp_dir) / "snapshot"
            snapshot_path.mkdir()
            write_prefetch_manifest(
                {"Qwen/Qwen3-TTS-12Hz-1.7B-Base": snapshot_path},
                manifest_path,
            )

            self.assertEqual(
                resolve_prefetched_model_path(
                    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                    manifest_path,
                    None,
                ),
                str(snapshot_path.resolve()),
            )

    def test_resolve_cached_model_snapshot_path_uses_cached_config_file_without_network(self) -> None:
        """Ensure cache fallback can derive a snapshot directory from a cached config file.

        Usage:
            This test protects offline startup when the explicit prefetch manifest is
            missing or stale by verifying that the helper can still resolve the
            local snapshot directory from Hugging Face cache metadata alone.

        Parameters:
            None.

        Returns:
            None. The test asserts that the derived path points at the local snapshot.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir) / "snapshots" / "abc123"
            snapshot_path.mkdir(parents=True)
            cached_config_path = snapshot_path / "config.json"
            cached_config_path.write_text("{}", encoding="utf-8")

            fake_hf_module = types.ModuleType("huggingface_hub")
            fake_hf_module.try_to_load_from_cache = lambda **kwargs: str(cached_config_path)
            fake_hf_module.scan_cache_dir = lambda cache_dir=None: None
            previous_hf_module = sys.modules.get("huggingface_hub")
            sys.modules["huggingface_hub"] = fake_hf_module
            try:
                self.assertEqual(
                    resolve_cached_model_snapshot_path(
                        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                        None,
                    ),
                    str(snapshot_path.resolve()),
                )
            finally:
                if previous_hf_module is None:
                    del sys.modules["huggingface_hub"]
                else:
                    sys.modules["huggingface_hub"] = previous_hf_module


if __name__ == "__main__":
    unittest.main()
