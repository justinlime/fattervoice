"""Unit tests for model alias resolution helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fatterqwen.model_catalog import expand_model_selection, resolve_model_id


class ModelCatalogTests(unittest.TestCase):
    """Verify that built-in model aliases resolve to the expected upstream IDs."""

    def test_resolve_known_alias(self) -> None:
        """Ensure the required short alias resolves to the canonical Hugging Face model ID.

        Usage:
            This test guards the project contract that `1.7B` must remain a valid
            shorthand model selector.

        Parameters:
            None.

        Returns:
            None. The test asserts on the resolved model identifier.
        """
        self.assertEqual(
            resolve_model_id("1.7B"),
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        )

    def test_resolve_local_path(self) -> None:
        """Ensure a filesystem path is accepted as a direct model source.

        Usage:
            This test protects the offline and predownloaded-model workflows that
            rely on loading from a local directory instead of Hugging Face.

        Parameters:
            None.

        Returns:
            None. The test asserts the original path string is preserved.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            self.assertEqual(resolve_model_id(str(model_path)), str(model_path))

    def test_expand_all_models(self) -> None:
        """Ensure the prefetch helper expands `all` into both supported built-in models.

        Usage:
            This test protects Docker/offline workflows that warm both supported
            voice-clone models into the cache ahead of time.

        Parameters:
            None.

        Returns:
            None. The test asserts the expected model list length and contents.
        """
        expanded_models = expand_model_selection("all")
        self.assertEqual(len(expanded_models), 2)
        self.assertIn("Qwen/Qwen3-TTS-12Hz-0.6B-Base", expanded_models)
        self.assertIn("Qwen/Qwen3-TTS-12Hz-1.7B-Base", expanded_models)


if __name__ == "__main__":
    unittest.main()
