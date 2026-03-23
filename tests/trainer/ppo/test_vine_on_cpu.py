# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
"""CPU-only unit tests for VinePPO helpers."""

import pytest
import torch


class FakeTokenizer:
    """Minimal tokenizer for tests: each token id maps to a fixed string; decode joins them."""

    def __init__(self, id_to_str: dict[int, str]):
        self._map = id_to_str

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self._map.get(int(i), "") for i in ids)


def test_find_step_boundaries_basic():
    from verl.trainer.ppo.vine import find_step_boundaries

    # Tokens: "step1", "\n", "step2", "\n"
    # Two boundaries: after each "\n" → positions 2 and 4.
    tok = FakeTokenizer({0: "step1", 1: "\n", 2: "step2", 3: "\n"})
    boundaries = find_step_boundaries([0, 1, 2, 3], tok, separators=["\n"])
    assert boundaries == [2, 4]


def test_find_step_boundaries_no_separator():
    from verl.trainer.ppo.vine import find_step_boundaries

    tok = FakeTokenizer({0: "abc", 1: "def"})
    boundaries = find_step_boundaries([0, 1], tok, separators=["\n"])
    assert boundaries == []


def test_find_step_boundaries_multiple_separators():
    from verl.trainer.ppo.vine import find_step_boundaries

    # Tokens: "a", "\n\n", "b", ".", "c"
    # Separators ["\n", "."] → boundaries after token 1 and token 3.
    tok = FakeTokenizer({0: "a", 1: "\n\n", 2: "b", 3: ".", 4: "c"})
    boundaries = find_step_boundaries([0, 1, 2, 3, 4], tok, separators=["\n", "."])
    assert boundaries == [2, 4]


def test_broadcast_value_to_tokens_basic():
    from verl.trainer.ppo.vine import broadcast_value_to_tokens

    # boundaries=[3, 7, 10], v_hat=[1.0, 2.0, 3.0], response_length=12, fallback=0.5.
    # Step assignment: tokens 0..2 → fallback (no preceding boundary)
    #                  tokens 3..6 → v_hat[0] (after boundary 3)
    #                  tokens 7..9 → v_hat[1] (after boundary 7)
    #                  tokens 10..11 → v_hat[2] (after boundary 10)
    out = broadcast_value_to_tokens(
        boundaries=[3, 7, 10],
        v_hat=[1.0, 2.0, 3.0],
        response_length=12,
        fallback=0.5,
    )
    expected = torch.tensor([0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0])
    assert torch.allclose(out, expected)


def test_broadcast_value_to_tokens_no_boundaries():
    from verl.trainer.ppo.vine import broadcast_value_to_tokens

    out = broadcast_value_to_tokens(boundaries=[], v_hat=[], response_length=5, fallback=0.7)
    expected = torch.full((5,), 0.7)
    assert torch.allclose(out, expected)


def test_broadcast_value_to_tokens_length_mismatch_raises():
    from verl.trainer.ppo.vine import broadcast_value_to_tokens

    with pytest.raises(ValueError):
        broadcast_value_to_tokens(boundaries=[3], v_hat=[1.0, 2.0], response_length=5, fallback=0.0)
