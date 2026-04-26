# Lj1ang's contribution: SGLang multi-turn rollout profiling logs

This document covers the work introduced by Lei Jiang (`Lj1ang` on GitHub) on the
`sglang-log` branch, on top of upstream `5cef3ca8`. It describes what each commit does,
how the pieces fit together, how to enable the instrumentation, and what the resulting
log layout looks like.

## TL;DR

The branch adds **per-step / per-worker profiling logs for SGLang multi-turn rollout** to
verl's experimental agent loop. When enabled, every async-rollout worker writes a
JSONL file recording the duration of each phase of a step:

```
<log_root>/<EXPERIMENT_NAME>/step_<N>/worker_<R>.jsonl
```

A separate `worker_-1.jsonl` records reward computation time on the trainer driver, so
all phases of a training step land under the same `step_<N>/` directory and can be
merged into one timeline.

A small number of secondary changes ride along: explicit `Tracking.finish()` to suppress
wandb shutdown noise, an SGLang engine `log_level` lifted from `error` to `INFO` for
throughput visibility, and two example scripts that demonstrate how to launch with
profiling on.

## Commits in dependency order

| Commit     | Title                              | What it does                                                               |
| ---------- | ---------------------------------- | -------------------------------------------------------------------------- |
| `6be75b35` | sglang log                         | Adds `log_manager.py` and the `Tracking.finish()` machinery.               |
| `97e6353d` | improve log                        | Refines the logger; tweaks an SGLang launch script.                        |
| `c84ccb61` | step and worker level logging      | Wires the trainer + agent_loop to call `set_step` / `set_rank` / `log()`.  |
| `6847516e` | fix step0 problem                  | Threads `worker_rank` into `AgentLoopWorker` so step 0 lands on rank N.    |
| `19f55f9b` | trainer reward log                 | Adds `worker_-1` reward-duration line emitted from `RayPPOTrainer.fit()`.  |

## File-by-file map

### Core: the log manager

**`verl/workers/rollout/sglang_rollout/log_manager.py`** (new, ~140 lines)

Public API:

- `get_sglang_log_dir()` — resolves the log root: `SGLANG_PROFILE_LOG_ROOT/<EXPERIMENT_NAME>`
  or `logs/<EXPERIMENT_NAME>`.
- `set_sglang_rollout_step(step)` / `get_sglang_step()` — module-level **per-process** step
  counter. Each Ray actor process imports this module independently, so the stamp must
  be set inside every process that wants to write logs.
- `set_sglang_rollout_rank(rank)` / `get_sglang_rank()` — same, but for the worker rank.
- `build_profile_log_path(log_dir, step, rank)` — formats `log_dir/step_<N>/worker_<R>.jsonl`.
- `get_sglang_log_path([step], [rank], [log_dir])` — convenience wrapper that fills missing
  args from the module-level state.
- `SGLangLogManager.log(log_path, event, duration, ...)` — appends a JSONL line. Handles
  are line-buffered, kept open for the lifetime of the worker, and registered with `atexit`
  so unfinalized lines are still flushed after a Ray worker shutdown.
- `get_sglang_log_manager()` — singleton accessor.

JSONL schema (one line per event):

```json
{
  "timestamp": "2025-...T...",
  "step": 42,
  "worker": 0,
  "event": "engine_async_generate",
  "duration_sec": 1.234,
  "...": "any extra kwargs the caller passed"
}
```

The fixed leading columns (`timestamp`, `step`, `worker`, `event`, `duration_sec`) make
the file grep- and sort-friendly.

**`verl/workers/rollout/sglang_rollout/__init__.py`** — re-exports the log_manager
symbols so callers in other layers can soft-import them as
`from verl.workers.rollout.sglang_rollout import ...`.

### Trainer-side wiring

**`verl/trainer/ppo/ray_trainer.py`**

Three additions:

1. **Soft import** of the log helpers at module load. If sglang isn't installed, both
   names become `None` and every call site below gates on `is not None`.
2. **Step stamping** before each async rollout: the trainer driver calls
   `set_sglang_rollout_step(self.global_steps)` before `async_rollout_manager.generate_sequences`
   (and again on the REMAX baseline path). This only stamps the **driver** process — workers
   re-stamp their own state inside `AgentLoopWorker.generate_sequences` because Ray actors
   are separate Python processes.
3. **Reward duration logging**: after computing rewards, if `EXPERIMENT_NAME` is set the
   driver writes a `reward_duration` event to `step_<N>/worker_-1.jsonl`. Rank `-1` is the
   sentinel for "trainer driver, not a worker".
4. **Explicit `logger.finish()`** before both early returns in `fit()` (val-only path and
   final return). This drives `Tracking.finish()` (see below).

### Agent-loop instrumentation

**`verl/experimental/agent_loop/agent_loop.py`**

The bulk of the instrumentation lives here. Three layers of timing are emitted per step,
all gated by a single `_log` flag:

| Phase                  | When measured                                                | Event name                  |
| ---------------------- | ------------------------------------------------------------ | --------------------------- |
| Per-turn engine call   | Each `await server.generate.remote(...)` in `AsyncLLMServerManager.generate()` | `engine_async_generate` |
| Step preprocessing     | Method entry → just before `asyncio.gather` on per-sample loops | `preprocessing_duration`   |
| Step generation fan-out| The `asyncio.gather` of all per-sample agent loops           | `async_generate_duration`   |
| End-to-end step        | Method entry → after `_postprocess`                          | `total_step_duration`       |

Critical detail: each engine-side timer brackets the await with `torch.cuda.synchronize()`
when CUDA is available, so `time.perf_counter()` bounds the actual GPU duration rather than
a stale dispatch.

`AgentLoopWorker.__init__` accepts a new `worker_rank` parameter and stamps it into
the log_manager once per actor (the rank is stable for the lifetime of the run because
each Ray actor is created exactly once in `AgentLoopManager._init_agent_loop_workers`).
The remote-create call passes `i` as `worker_rank`.

**`verl/workers/rollout/sglang_rollout/async_sglang_server.py`**

One line: the SGLang engine `log_level` is flipped from a hard-coded `"error"` to
`engine_kwargs.pop("log_level", "INFO")`. Result: per-decode-batch throughput numbers
print to stdout, which is useful for sanity-checking the JSONL durations against
SGLang's own reported throughput.

### Tracking shutdown

**`verl/utils/tracking.py`**

Adds `Tracking.finish()` and reroutes `__del__` through it. The existing `__del__`
pattern was unsafe because:

- During interpreter shutdown, wandb's background pipe may already be torn down.
- `__del__` running then surfaces as `BrokenPipeError` tracebacks.
- `Tracking` aggregates many backends (wandb, swanlab, vemlp_wandb, tensorboard, clearml,
  trackio, file) — one failing finalizer should not stop the rest.

`finish()` runs each backend's finalizer in its own try/except, is idempotent
(`_finish_called` flag), and is invoked explicitly by the trainer at every exit path so
shutdown happens while sockets are still healthy.

### Examples

**`examples/sglang_multiturn/run_qwen2.5-1.5b_gsm8k_multiturn_4xgpu.sh`** (new, 77 lines)

End-to-end recipe for a 4×H100 GRPO + SGLang multi-turn run on Qwen2.5-1.5B / GSM8K.
Wires three log destinations:

1. `stdout/stderr` → `$LOG_DIR/qwen2.5-1.5b_multiturn_4xgpu_<timestamp>.log`
2. verl `FileLogger` metrics → `$VERL_FILE_LOGGER_ROOT/<project>/<exp>.jsonl`
3. SGLang per-step/per-worker profiling → `$SGLANG_PROFILE_LOG_ROOT/$EXPERIMENT_NAME/step_*/worker_*.jsonl`

`EXPERIMENT_NAME` is the gate for the per-worker profile JSONL — without it the log
manager doesn't write.

**`examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh`** — minor: switches to
`nohup … &` background run + log file, exports `EXPERIMENT_NAME`, drops wandb from
`trainer.logger`.

## How to enable

Profiling is gated by **two** runtime conditions:

1. The sglang submodule has to import successfully (the soft-imports in `ray_trainer.py`
   and `agent_loop.py`).
2. `EXPERIMENT_NAME` must be set in the environment.

Optional knobs (env vars):

- `SGLANG_PROFILE_LOG_ROOT` — root for the per-step/per-worker JSONLs. Defaults to `logs/`.

In a typical launch:

```bash
export EXPERIMENT_NAME=my_exp_name
export SGLANG_PROFILE_LOG_ROOT=$PWD/logs
python -m verl.trainer.main_ppo --config-path=... --config-name=...
```

After a run, `logs/my_exp_name/step_*/worker_*.jsonl` contains the per-phase JSONL
events; they can be tailed live or post-processed into a per-step timeline.

## Process model recap

- The **trainer driver** stamps the step once per gen call and writes
  `worker_-1.jsonl` for reward duration.
- Each **AgentLoopWorker** is a separate Ray actor process. It re-stamps the step from
  `batch.meta_info["global_steps"]` on every `generate_sequences` call (because the
  driver's stamp does not propagate across processes), and stamps its rank exactly once
  at construction.
- The **SGLang engine actors** are also separate processes. They don't currently write
  profile logs; the per-turn `engine_async_generate` is timed on the *caller* side
  (`AsyncLLMServerManager.generate()`), so it captures end-to-end engine round-trip
  including the network hop.

## Files changed (cumulative)

```
examples/sglang_multiturn/run_qwen2.5-1.5b_gsm8k_multiturn_4xgpu.sh   |  77 ++++  (new)
examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh           |  11 +-
verl/experimental/agent_loop/agent_loop.py                            |  95 +++++++++++-
verl/trainer/ppo/ray_trainer.py                                       |  34 +++++
verl/utils/tracking.py                                                |  57 ++++---
verl/workers/rollout/sglang_rollout/__init__.py                       |  24 ++++
verl/workers/rollout/sglang_rollout/async_sglang_server.py            |   5 +-
verl/workers/rollout/sglang_rollout/log_manager.py                    | 139 +++++++++  (new)
```

8 files, ~423 insertions / 19 deletions.
