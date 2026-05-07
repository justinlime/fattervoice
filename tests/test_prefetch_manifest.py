"""Unit tests for cached-model resolution helpers."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

from fattervoice.prefetch_manifest import (
    resolve_cached_model_snapshot_path,
    resolve_snapshot_directory_from_cached_file,
)


class PrefetchManifestTests(unittest.TestCase):
    """Verify that cached model resolution can redirect runtime loading locally."""

    def test_resolve_cached_model_snapshot_path_returns_existing_local_path_directly(self) -> None:
        """Ensure explicit local model paths bypass Hugging Face cache inspection.

        Usage:
            Runtime callers may already provide a filesystem path instead of a Hub
            model ID. This test verifies that the helper returns that local path
            directly without trying to interpret it as a cached repository ID.

        Parameters:
            None.

        Returns:
            None. The test asserts that the resolved path matches the local input.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            local_model_path = Path(temp_dir) / "local-model"
            local_model_path.mkdir()

            self.assertEqual(
                resolve_cached_model_snapshot_path(str(local_model_path), None),
                str(local_model_path.resolve()),
            )

    def test_resolve_snapshot_directory_from_cached_file_preserves_snapshot_parent_for_symlinks(self) -> None:
        """Ensure snapshot resolution does not collapse symlinked cache files into `blobs`.

        Usage:
            Real Hugging Face caches usually place snapshot files under
            `snapshots/<revision>/...` as symlinks to `blobs/<hash>`. This test
            protects the regression where resolving the file path before taking
            its parent turned a valid snapshot file into the shared `blobs`
            directory.

        Parameters:
            None.

        Returns:
            None. The test asserts that the derived directory is the snapshot
            directory rather than the blob store.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_cache_path = Path(temp_dir) / "models--k2-fsa--OmniVoice"
            blob_path = repo_cache_path / "blobs" / "deadbeef"
            snapshot_path = repo_cache_path / "snapshots" / "abc123"
            blob_path.parent.mkdir(parents=True)
            snapshot_path.mkdir(parents=True)
            blob_path.write_text("{}", encoding="utf-8")
            cached_config_path = snapshot_path / "config.json"
            try:
                cached_config_path.symlink_to(blob_path)
            except OSError as symlink_error:  # pragma: no cover - depends on host filesystem capabilities.
                self.skipTest(f"Symlink creation is unavailable in this environment: {symlink_error}")

            self.assertEqual(
                resolve_snapshot_directory_from_cached_file(str(cached_config_path)),
                snapshot_path.resolve(),
            )

    def test_resolve_cached_model_snapshot_path_uses_symlinked_snapshot_config_without_network(self) -> None:
        """Ensure cached config lookup returns the snapshot directory for symlinked files.

        Usage:
            Offline startup first tries `try_to_load_from_cache(...)` before it
            scans the full cache metadata. This regression test models the real
            Hugging Face snapshot symlink layout and verifies that the fast path
            still returns the snapshot directory that `from_pretrained(...)`
            expects.

        Parameters:
            None.

        Returns:
            None. The test asserts that the derived path points at the snapshot
            directory rather than the blob store.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_cache_path = Path(temp_dir) / "models--k2-fsa--OmniVoice"
            blob_path = repo_cache_path / "blobs" / "deadbeef"
            snapshot_path = repo_cache_path / "snapshots" / "abc123"
            blob_path.parent.mkdir(parents=True)
            snapshot_path.mkdir(parents=True)
            blob_path.write_text("{}", encoding="utf-8")
            cached_config_path = snapshot_path / "config.json"
            try:
                cached_config_path.symlink_to(blob_path)
            except OSError as symlink_error:  # pragma: no cover - depends on host filesystem capabilities.
                self.skipTest(f"Symlink creation is unavailable in this environment: {symlink_error}")

            fake_hf_module = types.ModuleType("huggingface_hub")
            fake_hf_module.try_to_load_from_cache = lambda **kwargs: str(cached_config_path)
            fake_hf_module.scan_cache_dir = lambda cache_dir=None: None
            previous_hf_module = sys.modules.get("huggingface_hub")
            sys.modules["huggingface_hub"] = fake_hf_module
            try:
                self.assertEqual(
                    resolve_cached_model_snapshot_path(
                        "k2-fsa/OmniVoice",
                        None,
                    ),
                    str(snapshot_path.resolve()),
                )
            finally:
                if previous_hf_module is None:
                    del sys.modules["huggingface_hub"]
                else:
                    sys.modules["huggingface_hub"] = previous_hf_module

    def test_resolve_cached_model_snapshot_path_falls_back_to_scan_cache_dir(self) -> None:
        """Ensure cache metadata scanning still resolves a snapshot when direct lookup misses.

        Usage:
            Some cache states may not return a fast-path `config.json` result even
            though the repository snapshot exists locally. This test verifies that
            the helper still resolves a usable snapshot directory from
            `scan_cache_dir(...)` without performing any network requests.

        Parameters:
            None.

        Returns:
            None. The test asserts that the snapshot path from cache metadata is
            returned.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir) / "snapshots" / "abc123"
            snapshot_path.mkdir(parents=True)

            fake_revision = types.SimpleNamespace(
                refs=frozenset({"main"}),
                snapshot_path=snapshot_path,
                last_modified=2,
            )
            fake_repo = types.SimpleNamespace(
                repo_id="k2-fsa/OmniVoice",
                repo_type="model",
                revisions=[fake_revision],
                last_modified=2,
            )
            fake_cache_info = types.SimpleNamespace(repos=[fake_repo])

            fake_hf_module = types.ModuleType("huggingface_hub")
            fake_hf_module.try_to_load_from_cache = lambda **kwargs: None
            fake_hf_module.scan_cache_dir = lambda cache_dir=None: fake_cache_info
            previous_hf_module = sys.modules.get("huggingface_hub")
            sys.modules["huggingface_hub"] = fake_hf_module
            try:
                self.assertEqual(
                    resolve_cached_model_snapshot_path(
                        "k2-fsa/OmniVoice",
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
