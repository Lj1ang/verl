# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""VinePPO: step-boundary detection, V-hat broadcast, branch-rollout dispatch."""

from typing import Sequence

import torch


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
