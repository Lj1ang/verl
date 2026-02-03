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
    """Base directory for profiling JSONL: logs/<EXPERIMENT_NAME>."""
    return "logs/" + os.getenv("EXPERIMENT_NAME", "multiturn_log_dir")


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


def get_sglang_log_path(step: Optional[int] = None, rank: Optional[int] = None) -> str:
    """Path for worker JSONL: logs/<EXPERIMENT_NAME>/step_<step>/worker_<rank>.jsonl."""
    step = step if step is not None else _current_step
    rank = rank if rank is not None else _current_rank
    return os.path.join(get_sglang_log_dir(), f"step_{step}", f"worker_{rank}.jsonl")


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
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
        }
        if duration is not None:
            log_entry["duration_sec"] = duration
        if extra is not None:
            log_entry["extra"] = extra
        if workid is not None:
            log_entry["workid"] = workid
        if step is not None:
            log_entry["step"] = step
        if extra_keys:
            for key in extra_keys:
                log_entry[key] = extra_keys[key]
        ordered_keys = ["timestamp", "event", "duration_sec"] + [
            k for k in log_entry if k not in ("timestamp", "event", "duration_sec")
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
