# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Profiling log manager for SGLang multi-turn rollout (same format as PrinsYin/verl multiturn_profile_log)."""

import atexit
import json
import os
from datetime import datetime
from typing import Optional


# Module-level step and rank for logging (set by trainer/rollout at start of each step).
_current_step = 0
_current_rank = 0


def get_sglang_log_dir() -> str:
    """Base directory for profiling JSONL. Uses SGLANG_PROFILE_LOG_ROOT if set, else logs/<EXPERIMENT_NAME>."""
    root = os.getenv("SGLANG_PROFILE_LOG_ROOT", "")
    name = os.getenv("EXPERIMENT_NAME", "multiturn_log_dir")
    if root:
        return os.path.join(root, name)
    return os.path.join("logs", name)


def get_sglang_step() -> int:
    """Current training/rollout step (for log paths)."""
    return _current_step


def get_sglang_rank() -> int:
    """Current worker rank (for log paths)."""
    return _current_rank


def set_sglang_rollout_step(step: int) -> None:
    """Set current step (call at start of each rollout step)."""
    global _current_step
    _current_step = step


def set_sglang_rollout_rank(rank: int) -> None:
    """Set current rank for this process."""
    global _current_rank
    _current_rank = rank


def build_profile_log_path(log_dir: str, step: int, rank: int) -> str:
    """Build step/worker profiling log path: log_dir/step_<step>/worker_<rank>.jsonl."""
    return os.path.join(log_dir, f"step_{step}", f"worker_{rank}.jsonl")


def get_sglang_log_path(
    step: Optional[int] = None, rank: Optional[int] = None, log_dir: Optional[str] = None
) -> str:
    """Path for worker JSONL: log_dir/step_<step>/worker_<rank>.jsonl.
    Uses get_sglang_log_dir() if log_dir is not provided.
    """
    step_val = step if step is not None else _current_step
    rank_val = rank if rank is not None else _current_rank
    base = log_dir if log_dir is not None else get_sglang_log_dir()
    return build_profile_log_path(base, step_val, rank_val)


class SGLangLogManager:
    """Logging for SGLang multi-turn rollout profiling (request/turn/engine/tool timings)."""

    def __init__(self):
        self.file_handles = {}
        atexit.register(self.close_all)

    def get_handle(self, log_path: str):
        if log_path not in self.file_handles:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self.file_handles[log_path] = open(log_path, "a", buffering=1)
        return self.file_handles[log_path]

    def log(
        self,
        log_path: str,
        event: str,
        duration: Optional[float] = None,
        extra: Optional[dict] = None,
        workid: Optional[int] = None,
        step: Optional[int] = None,
        **extra_keys,
    ) -> None:
        handle = self.get_handle(log_path)
        # Step/worker first for easy filtering and sorting (step/worker manner)
        step_val = step if step is not None else _current_step
        worker_val = workid if workid is not None else _current_rank
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step_val,
            "worker": worker_val,
            "event": event,
        }
        if duration is not None:
            log_entry["duration_sec"] = duration
        if extra is not None:
            log_entry["extra"] = extra
        if extra_keys:
            for key in extra_keys:
                log_entry[key] = extra_keys[key]
        ordered_keys = ["timestamp", "step", "worker", "event", "duration_sec"] + [
            k for k in log_entry if k not in ("timestamp", "step", "worker", "event", "duration_sec")
        ]
        ordered_entry = {k: log_entry[k] for k in ordered_keys if k in log_entry}
        handle.write(json.dumps(ordered_entry) + "\n")
        handle.flush()

    def close_all(self) -> None:
        for handle in self.file_handles.values():
            handle.close()


# Singleton for use across rollout/trainer
_log_manager: Optional[SGLangLogManager] = None


def get_sglang_log_manager() -> SGLangLogManager:
    """Get or create the global SGLangLogManager instance."""
    global _log_manager
    if _log_manager is None:
        _log_manager = SGLangLogManager()
    return _log_manager
