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
