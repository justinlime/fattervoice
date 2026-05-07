"""Unit tests for Hugging Face cache environment helpers."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fattervoice.hf_cache import is_huggingface_offline_mode_enabled


class HuggingFaceCacheTests(unittest.TestCase):
    """Verify runtime inspection of Hugging Face offline environment flags."""

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
