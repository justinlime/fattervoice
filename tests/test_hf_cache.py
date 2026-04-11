"""Unit tests for Hugging Face cache path configuration helpers."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fatterqwen.hf_cache import (
    configure_huggingface_cache,
    is_huggingface_offline_mode_enabled,
)


class HuggingFaceCacheTests(unittest.TestCase):
    """Verify that prefetch and runtime can be pointed at one shared hub cache path."""

    def test_configure_huggingface_cache_sets_consistent_environment_variables(self) -> None:
        """Ensure all relevant Hugging Face environment variables target one hub cache path.

        Usage:
            This test guards the offline Docker workflow by ensuring build-time
            prefetch and runtime `from_pretrained(...)` calls can be configured to
            resolve artifacts from the same cache directory.

        Parameters:
            None.

        Returns:
            None. The test asserts that the expected environment variables are set.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "hub"
            with patch.dict(os.environ, {}, clear=True):
                resolved_cache_dir = configure_huggingface_cache(cache_dir)

                self.assertEqual(resolved_cache_dir, cache_dir.resolve())
                self.assertEqual(os.environ["HF_HUB_CACHE"], str(cache_dir.resolve()))
                self.assertEqual(os.environ["HUGGINGFACE_HUB_CACHE"], str(cache_dir.resolve()))
                self.assertEqual(os.environ["TRANSFORMERS_CACHE"], str(cache_dir.resolve()))
                self.assertEqual(os.environ["HF_HOME"], str(cache_dir.resolve().parent))

    def test_is_huggingface_offline_mode_enabled_respects_both_env_flags(self) -> None:
        """Ensure offline-mode detection matches the Hugging Face environment contract.

        Usage:
            This test protects the local-files-only model-loading path by verifying
            that either supported offline environment variable enables the stricter
            runtime behavior.

        Parameters:
            None.

        Returns:
            None. The test asserts the expected boolean result for both flags.
        """
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}, clear=True):
            self.assertTrue(is_huggingface_offline_mode_enabled())

        with patch.dict(os.environ, {"TRANSFORMERS_OFFLINE": "true"}, clear=True):
            self.assertTrue(is_huggingface_offline_mode_enabled())

        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(is_huggingface_offline_mode_enabled())


if __name__ == "__main__":
    unittest.main()
