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


def test_compute_advantage_dispatches_to_vine():
    from verl.protocol import DataProto
    from verl.trainer.ppo.core_algos import AdvantageEstimator
    from verl.trainer.ppo.ray_trainer import compute_advantage

    bs, T = 2, 4
    R = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    v_hat = torch.tensor([[0.0, 0.5, 0.5, 0.5], [0.0, 0.5, 0.5, 0.5]])
    mask = torch.ones(bs, T)
    data = DataProto.from_single_dict(
        {
            "token_level_rewards": R,
            "response_mask": mask,
            "vine_v_hat_per_token": v_hat,
            "responses": torch.zeros(bs, T, dtype=torch.long),
            "attention_mask": torch.ones(bs, 2 * T, dtype=torch.long),
        }
    )
    out = compute_advantage(data, adv_estimator=AdvantageEstimator.VINE, config=None)
    assert "advantages" in out.batch
    assert "returns" in out.batch
    assert out.batch["advantages"].shape == (bs, T)


def test_compute_vine_advantage_basic():
    from verl.trainer.ppo.vine import compute_vine_advantage

    # Two rollouts, response_length=4.
    # token_level_rewards: outcome reward placed at the last masked token (verl convention).
    # Rollout 0: R=1.0, V_hat per token = [0.0, 0.5, 0.5, 0.5]
    # Rollout 1: R=0.0, V_hat per token = [0.0, 0.5, 0.5, 0.5]
    R = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    v_hat = torch.tensor([[0.0, 0.5, 0.5, 0.5], [0.0, 0.5, 0.5, 0.5]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0]])
    advantages, returns = compute_vine_advantage(R, v_hat, mask)

    # advantages = (R_per_rollout - v_hat) * mask
    # R_per_rollout: [1.0, 0.0]
    # Rollout 0: [1-0, 1-0.5, 1-0.5, 1-0.5] * [1,1,1,1] = [1.0, 0.5, 0.5, 0.5]
    # Rollout 1: [0-0, 0-0.5, 0-0.5, 0-0.5] * [1,1,1,0] = [0.0, -0.5, -0.5, 0.0]
    expected_adv = torch.tensor([[1.0, 0.5, 0.5, 0.5], [0.0, -0.5, -0.5, 0.0]])
    assert torch.allclose(advantages, expected_adv)
    # returns = R_per_rollout broadcast * mask
    expected_ret = torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(returns, expected_ret)
