from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from qwen_tts.core.models.configuration_qwen3_tts import (
    Qwen3TTSTalkerCodePredictorConfig,
    Qwen3TTSTalkerConfig,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "faster-qwen3-tts"))

from faster_qwen3_tts.predictor_graph import PredictorGraph
from faster_qwen3_tts.talker_graph import TalkerGraph



def build_talker_config(attn_implementation: str) -> Qwen3TTSTalkerConfig:
    """Create a compact talker config for graph-mask regression tests.

    Usage:
        The FlashAttention regression tests use this helper to instantiate a
        minimal talker config whose masking behavior matches the production
        model contracts without requiring a full checkpoint load.

    Parameters:
        attn_implementation: The backend name that should be exposed through the
            config, such as `flash_attention_2` or `sdpa`.

    Returns:
        A `Qwen3TTSTalkerConfig` object with the requested attention backend and
        small dimensions suitable for unit tests.
    """
    config = Qwen3TTSTalkerConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=64,
        text_vocab_size=64,
        code_predictor_config=Qwen3TTSTalkerCodePredictorConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=64,
            vocab_size=64,
            num_code_groups=4,
        ),
    )
    config._attn_implementation = attn_implementation
    return config



def build_predictor_graph(attn_implementation: str) -> PredictorGraph:
    """Create a compact predictor graph instance for mask-table assertions.

    Usage:
        The predictor graph tests use this helper to construct the graph helper
        with a lightweight config and simple test doubles so the unit tests can
        inspect prepared masks without loading the full model.

    Parameters:
        attn_implementation: The backend name that should be exposed through the
            predictor config.

    Returns:
        A `PredictorGraph` instance configured for CPU-side mask preparation.
    """
    pred_config = Qwen3TTSTalkerCodePredictorConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=64,
        num_code_groups=4,
    )
    pred_config._attn_implementation = attn_implementation
    predictor_stub = types.SimpleNamespace(
        small_to_mtp_projection=Mock(),
        model=types.SimpleNamespace(
            config=pred_config,
            codec_embedding=[Mock() for _ in range(pred_config.num_code_groups - 1)],
        ),
        lm_head=[Mock() for _ in range(pred_config.num_code_groups - 1)],
    )

    with patch("faster_qwen3_tts.predictor_graph.torch.cuda.current_device", return_value=0):
        return PredictorGraph(
            predictor_stub,
            pred_config,
            talker_hidden_size=128,
            device="cpu",
            dtype=torch.float32,
        )


class GraphMaskRegressionTests(unittest.TestCase):
    """Verify backend-aware graph preparation for FlashAttention 2."""

    def test_talker_graph_builds_flash_attention_varlen_metadata(self) -> None:
        """Ensure talker replay mutates one capture-stable FA2 metadata buffer.

        Usage:
            This regression test protects the exact failure mode flagged during
            review: the talker graph must not swap Python dicts after capture in
            order to change FlashAttention sequence lengths. Instead it should
            keep one stable metadata tensor and mutate it in place per replay
            position.

        Parameters:
            None.

        Returns:
            None. The test asserts the prepared cumulative-length tensor and the
            fixed max-length configuration.
        """
        talker_config = build_talker_config("flash_attention_2")
        talker_model = types.SimpleNamespace(config=talker_config)

        with patch("faster_qwen3_tts.talker_graph.torch.cuda.current_device", return_value=0):
            talker_graph = TalkerGraph(
                talker_model,
                talker_config,
                device="cpu",
                dtype=torch.float32,
                max_seq_len=8,
            )

        talker_graph._build_attention_masks()
        initial_tensor = talker_graph.flash_attn_cu_seq_lens_k

        self.assertIsNone(talker_graph.attn_mask)
        self.assertTrue(torch.equal(
            talker_graph.flash_attn_cu_seq_lens_q,
            torch.tensor([0, 1], dtype=torch.int32),
        ))
        self.assertTrue(torch.equal(
            talker_graph.flash_attn_cu_seq_lens_k,
            torch.tensor([0, 1], dtype=torch.int32),
        ))
        self.assertEqual(talker_graph.flash_attn_max_length_q, 1)
        self.assertEqual(talker_graph.flash_attn_max_length_k, 8)

        talker_graph._set_attention_mask(3)

        self.assertIs(talker_graph.flash_attn_cu_seq_lens_k, initial_tensor)
        self.assertTrue(torch.equal(
            talker_graph.flash_attn_cu_seq_lens_k,
            torch.tensor([0, 4], dtype=torch.int32),
        ))

    def test_talker_graph_rejects_padded_flash_attention_masks(self) -> None:
        """Ensure talker FA2 graph replay fails fast for padded prompt masks.

        Usage:
            The current FlashAttention graph path is validated for the batch-1
            contiguous-prefix server flow. This test protects the explicit guard
            that rejects padded talker masks instead of silently replaying an
            incorrect prefix layout.

        Parameters:
            None.

        Returns:
            None. The test asserts that padded masks are rejected.
        """
        talker_config = build_talker_config("flash_attention_2")
        talker_model = types.SimpleNamespace(config=talker_config)

        with patch("faster_qwen3_tts.talker_graph.torch.cuda.current_device", return_value=0):
            talker_graph = TalkerGraph(
                talker_model,
                talker_config,
                device="cpu",
                dtype=torch.float32,
                max_seq_len=8,
            )

        prompt_mask = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
        with self.assertRaises(ValueError):
            talker_graph._build_attention_masks(prompt_mask)

    def test_predictor_graph_builds_flash_attention_varlen_metadata(self) -> None:
        """Ensure predictor replay caches live-prefix FlashAttention metadata.

        Usage:
            The predictor graph previously relied on FlashAttention's dynamic
            unpadding path during capture. This regression test verifies that the
            prefill and decode steps now carry explicit varlen metadata while the
            attention-mask mapping itself stays `None`.

        Parameters:
            None.

        Returns:
            None. The test asserts the prepared prefill and first-decode
            metadata.
        """
        predictor_graph = build_predictor_graph("flash_attention_2")
        predictor_graph._build_attention_masks()

        self.assertIsNone(predictor_graph.prefill_attn["full_attention"])
        self.assertTrue(torch.equal(
            predictor_graph.prefill_flash_attn_kwargs["cu_seq_lens_q"],
            torch.tensor([0, 2], dtype=torch.int32),
        ))
        self.assertTrue(torch.equal(
            predictor_graph.prefill_flash_attn_kwargs["cu_seq_lens_k"],
            torch.tensor([0, 2], dtype=torch.int32),
        ))
        self.assertIsNone(predictor_graph.decode_attn[0]["full_attention"])
        self.assertTrue(torch.equal(
            predictor_graph.decode_flash_attn_kwargs[0]["cu_seq_lens_k"],
            torch.tensor([0, 3], dtype=torch.int32),
        ))
        self.assertEqual(predictor_graph.decode_flash_attn_kwargs[0]["max_length_q"], 1)
        self.assertEqual(predictor_graph.decode_flash_attn_kwargs[0]["max_length_k"], 3)


if __name__ == "__main__":
    unittest.main()
