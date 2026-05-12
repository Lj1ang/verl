# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""VinePPO: step-boundary detection, V-hat broadcast, branch-rollout dispatch."""

import time
from copy import deepcopy
from typing import Sequence

import torch

from verl.protocol import DataProto


def find_step_boundaries(
    token_ids: Sequence[int],
    tokenizer,
    separators: Sequence[str],
) -> list[int]:
    """Return token positions where the decoded suffix matches any separator.

    A "boundary" is a position b such that decoding tokens[:b] ends with a separator
    string. The branch rollout at boundary b uses tokens[:b] as the prefix.

    Args:
        token_ids: Response token ids (no prompt prefix).
        tokenizer: Object with `.decode(ids) -> str`.
        separators: Suffix strings that mark a reasoning-step end.

    Returns:
        Sorted list of boundary positions in [1, len(token_ids)].
    """
    boundaries: list[int] = []
    prev_decoded = ""
    for i in range(1, len(token_ids) + 1):
        decoded = tokenizer.decode(token_ids[:i])
        if decoded == prev_decoded:
            # No new characters emitted (e.g. byte-level token completing a unicode char).
            continue
        if any(decoded.endswith(sep) for sep in separators):
            boundaries.append(i)
        prev_decoded = decoded
    return boundaries


def broadcast_value_to_tokens(
    boundaries: Sequence[int],
    v_hat: Sequence[float],
    response_length: int,
    fallback: float,
) -> torch.Tensor:
    """Broadcast per-boundary V-hat values to per-token values.

    Token t at position p in the response is assigned V-hat[k] where k is the
    largest index with boundaries[k] <= p. Tokens before the first boundary use
    `fallback` (typically the GRPO group-mean baseline).

    Args:
        boundaries: Sorted token positions in [1, response_length] marking step ends.
        v_hat: V-hat values, one per boundary; len(v_hat) must equal len(boundaries).
        response_length: Total response length in tokens.
        fallback: Value for tokens before the first boundary (or all tokens if no boundaries).

    Returns:
        1-D float tensor of length `response_length`.
    """
    if len(boundaries) != len(v_hat):
        raise ValueError(
            f"len(boundaries)={len(boundaries)} != len(v_hat)={len(v_hat)}"
        )
    out = torch.full((response_length,), float(fallback))
    for k, b in enumerate(boundaries):
        # Tokens at positions [b, b_{k+1}) get v_hat[k].
        end = boundaries[k + 1] if k + 1 < len(boundaries) else response_length
        if b < response_length:
            out[b:end] = float(v_hat[k])
    return out


def compute_vine_advantage(
    token_level_rewards: torch.Tensor,
    v_hat_per_token: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute VinePPO per-token advantages and returns.

    Args:
        token_level_rewards: shape (bs, response_length). Outcome reward placed at
            the last response token; other positions zero (verl convention).
        v_hat_per_token: shape (bs, response_length). Per-token V-hat from
            broadcast_value_to_tokens, with fallback applied.
        response_mask: shape (bs, response_length). 1 for valid response tokens,
            0 for padding.

    Returns:
        advantages: (bs, response_length) — (R - V_hat) * mask, broadcast across tokens.
        returns: (bs, response_length) — R * mask broadcast.
    """
    # Sum to get the scalar outcome reward per rollout.
    R_per_rollout = token_level_rewards.sum(dim=-1, keepdim=True)  # (bs, 1)
    advantages = (R_per_rollout - v_hat_per_token) * response_mask
    returns = R_per_rollout * response_mask
    return advantages, returns


def _subsample_boundaries(boundaries: list[int], cap: int) -> list[int]:
    """Uniformly subsample boundaries down to `cap`, preserving order."""
    if len(boundaries) <= cap:
        return boundaries
    indices = torch.linspace(0, len(boundaries) - 1, cap).round().long().tolist()
    seen = set()
    out: list[int] = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            out.append(boundaries[i])
    return out


def _strip_left_padding(ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """Return ids with leading pad tokens removed."""
    nonpad = (ids != pad_token_id).nonzero(as_tuple=True)[0]
    if len(nonpad) == 0:
        return ids[:0]
    start = int(nonpad[0])
    return ids[start:]


def build_branch_prompts(
    main_rollouts: DataProto,
    tokenizer,
    step_separators: Sequence[str],
    max_branches_per_rollout: int,
    num_branches: int,
) -> tuple[DataProto, list[list[int]], list[int]]:
    """Build a flat batch of prefix-conditioned prompts for branch sampling.

    For each main rollout i with response token ids r_i, find step boundaries in r_i
    via find_step_boundaries. For each boundary b_k, the branch prefix is
    prompt_i ++ r_i[:b_k]. Each (i, k) is tiled by num_branches.

    Returns:
        branch_batch: DataProto where batch["prompts"] is a left-padded tensor of
            (sum_i M_i * num_branches, max_prefix_len). meta_info copied from
            main_rollouts but with do_sample=True forced.
        boundaries_per_rollout: boundary positions for each main rollout (post-cap).
        rollout_index: for each row in branch_batch, the index of the source main rollout.
    """
    pad_id = tokenizer.pad_token_id
    bs = main_rollouts.batch["prompts"].shape[0]
    prompt_len = main_rollouts.batch["prompts"].shape[1]

    boundaries_per_rollout: list[list[int]] = []
    flat_prefixes: list[torch.Tensor] = []
    rollout_index: list[int] = []

    for i in range(bs):
        prompt_ids = _strip_left_padding(main_rollouts.batch["prompts"][i], pad_id)
        # Use response_mask to find true response length (excludes right-pad).
        resp_mask = main_rollouts.batch["response_mask"][i]
        valid_len = int(resp_mask.sum().item())
        response_ids = main_rollouts.batch["responses"][i, :valid_len]

        boundaries = find_step_boundaries(
            response_ids.tolist(), tokenizer, separators=step_separators
        )
        boundaries = _subsample_boundaries(boundaries, max_branches_per_rollout)
        boundaries_per_rollout.append(boundaries)

        for b in boundaries:
            prefix = torch.cat([prompt_ids, response_ids[:b]], dim=0)
            for _ in range(num_branches):
                flat_prefixes.append(prefix)
                rollout_index.append(i)

    if not flat_prefixes:
        empty = torch.zeros((0, prompt_len), dtype=main_rollouts.batch["prompts"].dtype)
        empty_mask = torch.zeros((0, prompt_len), dtype=torch.long)
        branch_batch = DataProto.from_single_dict(
            {"prompts": empty, "attention_mask": empty_mask}
        )
        branch_batch.meta_info = deepcopy(main_rollouts.meta_info)
        return branch_batch, boundaries_per_rollout, rollout_index

    max_prefix_len = max(len(p) for p in flat_prefixes)
    branch_prompts = torch.full(
        (len(flat_prefixes), max_prefix_len),
        pad_id,
        dtype=main_rollouts.batch["prompts"].dtype,
    )
    for j, p in enumerate(flat_prefixes):
        branch_prompts[j, max_prefix_len - len(p) :] = p

    branch_attention_mask = (branch_prompts != pad_id).long()

    branch_batch = DataProto.from_single_dict(
        {"prompts": branch_prompts, "attention_mask": branch_attention_mask}
    )
    branch_batch.meta_info = deepcopy(main_rollouts.meta_info)
    branch_batch.meta_info["do_sample"] = True
    return branch_batch, boundaries_per_rollout, rollout_index


def run_branch_rollouts(
    main_rollouts: DataProto,
    tokenizer,
    rollout_dispatch_fn,
    reward_score_fn,
    num_branches: int,
    step_separators: Sequence[str],
    max_branches_per_rollout: int,
) -> tuple[torch.Tensor, list[list[int]], dict]:
    """Run branch rollouts and aggregate per-(rollout, boundary) V-hat.

    Args:
        main_rollouts: DataProto from the main rollout (post-generate, has responses).
        tokenizer: HuggingFace tokenizer.
        rollout_dispatch_fn: callable(DataProto) -> DataProto. Either
            self.actor_rollout_wg.generate_sequences or
            self.async_rollout_manager.generate_sequences.
        reward_score_fn: callable(DataProto) -> torch.Tensor of shape (n,). Wrap
            self._compute_or_extract_reward with sum_reward=True.
        num_branches: K' branches per boundary.
        step_separators: suffix strings marking step ends.
        max_branches_per_rollout: cap M per rollout.

    Returns:
        v_hat: (bs, M_max) tensor; v_hat[i, k] = mean reward over K' branches at
            boundary k of rollout i. Padded entries are 0.0.
        boundaries_per_rollout: list of lists of boundary positions.
        metrics: dict with v_hat_mean, v_hat_std, branches_per_rollout,
            boundaries_per_rollout, branch_rollout_time_s.
    """
    t0 = time.time()
    branch_batch, boundaries_per_rollout, rollout_index = build_branch_prompts(
        main_rollouts=main_rollouts,
        tokenizer=tokenizer,
        step_separators=step_separators,
        max_branches_per_rollout=max_branches_per_rollout,
        num_branches=num_branches,
    )

    bs = main_rollouts.batch["prompts"].shape[0]
    M_max = max((len(b) for b in boundaries_per_rollout), default=0)
    v_hat = torch.zeros(bs, max(M_max, 1))

    metrics = {
        "boundaries_per_rollout_mean": float(
            sum(len(b) for b in boundaries_per_rollout) / max(bs, 1)
        ),
        "boundaries_per_rollout_max": float(M_max),
        "branches_per_rollout_mean": (
            float(branch_batch.batch["prompts"].shape[0] / max(bs, 1))
            if branch_batch.batch["prompts"].shape[0] > 0
            else 0.0
        ),
    }

    if branch_batch.batch["prompts"].shape[0] == 0:
        metrics["v_hat_mean"] = 0.0
        metrics["v_hat_std"] = 0.0
        metrics["branch_rollout_time_s"] = time.time() - t0
        return v_hat, boundaries_per_rollout, metrics

    branch_output = rollout_dispatch_fn(branch_batch)
    branch_rewards = reward_score_fn(branch_output)  # shape (n_branches,)

    boundary_index: list[int] = []
    for i, bnds in enumerate(boundaries_per_rollout):
        for k in range(len(bnds)):
            boundary_index.extend([k] * num_branches)
    assert len(boundary_index) == len(rollout_index) == branch_rewards.numel()

    sums = torch.zeros(bs, max(M_max, 1))
    counts = torch.zeros(bs, max(M_max, 1))
    for n, (i, k) in enumerate(zip(rollout_index, boundary_index)):
        sums[i, k] += float(branch_rewards[n])
        counts[i, k] += 1.0
    safe_counts = counts.clamp(min=1.0)
    v_hat = sums / safe_counts

    metrics["v_hat_mean"] = float(v_hat[counts > 0].mean()) if (counts > 0).any() else 0.0
    metrics["v_hat_std"] = float(v_hat[counts > 0].std()) if (counts > 0).sum() > 1 else 0.0
    metrics["branch_rollout_time_s"] = time.time() - t0
    return v_hat, boundaries_per_rollout, metrics
