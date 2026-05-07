"""Legacy placeholder for tests that depended on the removed vendored backend."""

from __future__ import annotations

import unittest


@unittest.skip("Graph-mask tests were specific to the removed vendored backend.")
class RemovedBackendGraphMaskTests(unittest.TestCase):
    """Document that vendor-internal graph-mask tests no longer apply after migration."""

    def test_removed_backend_graph_masks_are_no_longer_applicable(self) -> None:
        """Explain why the historical graph-mask tests are intentionally skipped.

        Usage:
            The repository previously vendored a CUDA-graph-optimized backend
            whose internal graph-mask behavior was unit tested directly. After
            migrating to OmniVoice, those backend-internal invariants are no
            longer part of `fattervoice`, so the old tests are retained only as a
            historical marker.

        Parameters:
            None.

        Returns:
            None. The enclosing class skip decorator prevents execution.
        """
        self.fail("This test should be skipped.")


if __name__ == "__main__":
    unittest.main()
