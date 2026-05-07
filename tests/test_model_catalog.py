"""Unit tests for model alias resolution helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fattervoice.model_catalog import (
    expand_model_selection,
    expand_prefetch_asset_ids,
    resolve_model_id,
)


class ModelCatalogTests(unittest.TestCase):
    """Verify that built-in OmniVoice aliases resolve to the expected model IDs."""

    def test_resolve_known_alias(self) -> None:
        """Ensure the built-in OmniVoice alias resolves to the canonical model ID.

        Usage:
            This test guards the project contract that `omnivoice` remains a
            valid shorthand model selector.

        Parameters:
            None.

        Returns:
            None. The test asserts on the resolved model identifier.
        """
        self.assertEqual(
            resolve_model_id("omnivoice"),
            "k2-fsa/OmniVoice",
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
        """Ensure the prefetch helper expands `all` into the supported built-in model.

        Usage:
            This test protects Docker/offline workflows that warm the built-in
            OmniVoice model into the cache ahead of time.

        Parameters:
            None.

        Returns:
            None. The test asserts the expected model list contents.
        """
        expanded_models = expand_model_selection("all")
        self.assertEqual(expanded_models, ["k2-fsa/OmniVoice"])

    def test_expand_prefetch_assets_includes_audio_tokenizer_dependency(self) -> None:
        """Ensure OmniVoice prefetch also downloads the required audio tokenizer repo.

        Usage:
            Offline OmniVoice loading may need the auxiliary Higgs audio tokenizer
            repository, so this test verifies the prefetch planner includes it.

        Parameters:
            None.

        Returns:
            None. The test asserts the expected ordered prefetch target list.
        """
        self.assertEqual(
            expand_prefetch_asset_ids("omnivoice"),
            ["k2-fsa/OmniVoice", "eustlb/higgs-audio-v2-tokenizer"],
        )


if __name__ == "__main__":
    unittest.main()
