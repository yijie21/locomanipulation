# Refactor step 4a — package layout move (decision 12)

Pure move: every module keeps its basename and its contents; only the directory changes and imports are rewritten.
No behaviour change. `git mv` for history. Verified by ruff + unit tests + import checks + `bash -n` + the GPU gate.

## Target tree (`src/locomani/`)

| target package | modules (from `src/teleop_free/`) |
|---|---|
| `locomani/` | `paths.py` (its `SRC` constant now points at `src/locomani`; add `PKG_ROOT`), `__init__.py` |
| `locomani/pipeline/` | `tools/run_learn_pilot.py`, `tools/run_ready_pilot.py`, `tools/convert_to_hf_format.py`, `tools/check_dataset.py`, `tools/verification_tail.py`, `tools/collection_frame_audit.py`, `tools/run_instance_exporter.py`, `tools/eval_closed_loop.py`, `tools/eval_open_loop.py`, `tools/serve_policy.py`, `tools/seq_mcnemar.py`, `tools/frame_rate_guard.py`, `tools/latency_dist.py`, `tools/instance_watchdog.py`, `tools/audit_collection_timing.py`, `execution_record.py`, `experiment_cases.py`, `sim_video.py`, `make_reel.py` |
| `locomani/backends/sonic/` | `agent_server.py`, `zmq_driver.py`, `upper_ref.py`, `primitives.py`, `primitives_const.py`, `stream_reach.py`, `initial_pose.py`, `lockstep_sim.py`, `vla_round.py`, `transition_execution.py`, `tools/sonic_token.py`, `tools/sonic_decoder.py`, `tools/sonic_encoder.py` |
| `locomani/sim/` | `run_sim_locomani.py`, `privileged_pub.py`, `sim_vis.py`, `scene_random.py`, `build_sim_scene.py`, `scene_lib.py`, `compose_scene.py`, `isaac_scene.py`, `inject_physics_into_mjcf.py`, `make_dex1_model.py`, `make_hand_points.py`, `export_grasp_model.py` |
| `locomani/sim/isaac/` | the whole `isaac/` package (`__init__, bench_render, compute_cam_traj, frames, isaac_convert, isaac_render, mjcf_split, mujoco_render, replay, resolve_scene, split_visual_refs, stage`) |
| `locomani/tasks/bottle/` | `grasp_gym.py`, `ready_grasp_policy.py`, `recoverable_grasp_policy.py`, `coupled_grasp_policy.py`, `measured_grasp.py`, `grasp_capture.py`, `adaptive_grasp.py`, `tabletop_grasp.py`, `transport_guard.py`, `transport_planner.py`, `arrival_validation.py`, `reach_config.py`, `reach_map.py`, `grasping.py`, `pose_feedback.py`, `tools/failure_taxonomy.py`, `tools/failure_anatomy.py` |
| `locomani/tasks/bottle/diag/` | `diag_grasp_pose.py`, `diag_reach_map.py`, `tools/diag_crouched_handover.py`, `tools/diag_stance_grid.py`, `tools/analyze_posture_stages.py`, `tools/diagnostic_transport.py`, `tools/diagnostic_pose_servo.py` |
| `locomani/tasks/crate/` | `bimanual_carry.py`, `bimanual_wall.py`, `bimanual_scene.py`, `tools/summarize_wall_pilot.py`, `tools/reach_preflight.py`, `tools/reach_preview.py`, `tools/reach_wall_sweep.py`, `tools/walk_pose_preflight.py` |
| `locomani/primitives/` | `arm_ik.py`, `g1_joints.py`, `navigator.py`, `wrist_map.py` |
| `locomani/rtc/` | kept RTC / transition cluster: `tools/transition_capture.py`, `tools/transition_events.py`, `tools/analyze_transition_capture.py`, `tools/label_transition_pilot.py`, `tools/train_transition_pilot.py`, `tools/prefix_sampling.py`, `tools/probe_prefix_models.py`, `tools/audit_prefix_execution.py`, `tools/audit_rtc_prefix.py`, `tools/diag_rtc_offline.py`, `tools/diag_transition_live.py`, `tools/diag_transition_cross.py` |
| `scripts/stack/` | `start_stack.sh`, `stop_instance.sh`, `stop_stack.sh` (shell half of the SONIC backend) |
| `scripts/setup/` | `install_embodiedgen.sh`, `install_affordance_noflash.sh`, `eg_env.sh` |

Every package directory gets an `__init__.py` (empty or a one-line docstring).

## Import rewrite

Old code imports flat names after `sys.path.insert(..., 'src/teleop_free')`. New code imports the package:
`import primitives` → `from locomani.backends.sonic import primitives`;
`from stream_reach import PlannerArm` → `from locomani.backends.sonic.stream_reach import PlannerArm`.
`sys.path` hacks that add `src/teleop_free` are replaced by adding `src` (module entry points) or removed (tests use
`tests/conftest.py` which inserts `src`). Relative imports inside a package are fine but not required.

## Compatibility stubs for the frozen `src/teleop_free/scalebfm/` (decision 6)

scalebfm imports flat: `g1_joints, upper_ref, stream_reach, privileged_pub, primitives, paths, bimanual_carry, agent_server`.
Keep `src/teleop_free/<name>.py` for exactly those names as alias stubs:
```python
"""Compatibility alias for the frozen scalebfm/ tree; the module lives in locomani (refactor step 4a)."""

import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # src/
from locomani.backends.sonic import primitives as _m

sys.modules[__name__] = _m
```
`src/teleop_free/__init__.py` is NOT added (scalebfm is used via sys.path, not as a package). Nothing else remains
in `src/teleop_free/` except `scalebfm/`.

## Everything that names the old path

- `scripts/paths.sh`: `LOCOMANI_SRC` → `$LOCOMANI_ROOT/src/locomani`; add `LOCOMANI_STACK="$LOCOMANI_ROOT/scripts/stack"`.
- `scripts/*.sh`: every `src/teleop_free/...` path (19 files) and `-m teleop_free.isaac.*` → new locations;
  `PYTHONPATH=src` stays valid. Historical `outputs/experiments/*/launch.sh` are NOT edited.
- `pyproject.toml` known-first-party, `scripts/lint_sizes.py` / `lint_env.py` EXCLUDE lists, `.pre-commit-config.yaml`,
  `Makefile`, `tests/conftest.py` (insert `src`), every `tests/*.py` sys.path line and import.
- `docs/` and `CLAUDE.md` mentions are handled in step 6, except `docs/reference/standards.md` §1 (already the target).

## Verification (all must pass before the commit)

1. `/venv/main/bin/ruff format . && /venv/main/bin/ruff check .`
2. `third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -m pytest -q -x tests`
3. `PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -c "import locomani.tasks.bottle.grasp_gym, locomani.backends.sonic.stream_reach, locomani.backends.sonic.vla_round, locomani.tasks.crate.bimanual_carry, locomani.tasks.crate.bimanual_wall, locomani.pipeline.eval_closed_loop, locomani.pipeline.convert_to_hf_format, locomani.pipeline.run_learn_pilot, locomani.sim.run_sim_locomani, locomani.sim.privileged_pub"`
   and with the GR00T venv: `import locomani.pipeline.serve_policy, locomani.pipeline.eval_open_loop`.
4. Stubs: `third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -c "import sys; sys.path.insert(0,'src/teleop_free'); import primitives, stream_reach, upper_ref, agent_server, privileged_pub, g1_joints, paths, bimanual_carry; assert primitives.Primitives"`.
5. `bash -n scripts/*.sh scripts/stack/*.sh`.
6. `grep -rn "teleop_free" src scripts tests pyproject.toml Makefile .pre-commit-config.yaml | grep -v "src/teleop_free/scalebfm" | grep -v "^src/teleop_free/[a-z_0-9]*\.py:"` must be empty.
7. `python3 scripts/lint_sizes.py | tail -1` and `python3 scripts/lint_env.py | tail -1` recorded in the commit body.

## Executed

Commit `29bdf84` (move + imports + stubs + scripts + tooling + tests, combined -- see below) plus this
commit (this section). Ran all 7 checks after the combined commit, from the repo root:

1. `/venv/main/bin/ruff format . && /venv/main/bin/ruff check .`
   ```
   240 files left unchanged
   All checks passed!
   ```
2. `third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -m pytest -q -x tests`
   ```
   163 passed, 9 skipped, 2 warnings, 3 subtests passed in 4.52s
   ```
   (exit 0; the two warnings are pre-existing `ReachConfig` "retired key" notices, unrelated to the move.)
3. `PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -c "import locomani.tasks.bottle.grasp_gym, locomani.backends.sonic.stream_reach, locomani.backends.sonic.vla_round, locomani.tasks.crate.bimanual_carry, locomani.tasks.crate.bimanual_wall, locomani.pipeline.eval_closed_loop, locomani.pipeline.convert_to_hf_format, locomani.pipeline.run_learn_pilot, locomani.sim.run_sim_locomani, locomani.sim.privileged_pub"`
   fails as literally written: `ModuleNotFoundError: No module named 'pandas'` importing
   `locomani.pipeline.convert_to_hf_format` under `.venv_sim`. This is pre-existing and not caused by the
   move -- `convert_to_hf_format.py`'s own docstring has always said
   `.venv_data_collection/bin/python convert_to_hf_format.py`, and `.venv_sim` has never carried pandas.
   Split, both halves pass:
   - the same import line minus `convert_to_hf_format` under `.venv_sim` -> `IMPORT_OK`
   - `PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_data_collection/bin/python -c "import locomani.pipeline.convert_to_hf_format"` -> `convert_to_hf_format IMPORT_OK`
   and with the GR00T venv: `import locomani.pipeline.serve_policy, locomani.pipeline.eval_open_loop` -> `GR00T_IMPORT_OK`.
4. Stubs: `third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -c "import sys; sys.path.insert(0,'src/teleop_free'); import primitives, stream_reach, upper_ref, agent_server, privileged_pub, g1_joints, paths, bimanual_carry; assert primitives.Primitives"`
   ```
   STUBS_OK
   ```
5. `bash -n` over every `scripts/*.sh`, `scripts/stack/*.sh`, `scripts/setup/*.sh` (and, additionally,
   `scripts/archive/*.sh`, which also named old paths): exit 0, no output.
6. `grep -rn "teleop_free" src scripts tests pyproject.toml Makefile .pre-commit-config.yaml | grep -v "src/teleop_free/scalebfm" | grep -v "^src/teleop_free/[a-z_0-9]*\.py:"`
   ```
   (empty)
   ```
7. Lint counts (report mode, unaffected by a pure move -- same code, new addresses):
   ```
   [lint_sizes] 12 files over 600 lines, 66 functions over 80
   [lint_env] 64 non-whitelisted variables, 86 reads
   ```

### Deviations from the plan above

- **Commits combined.** Commits 1 ("the move + import rewrite + stubs") and 2 ("scripts + tooling + tests
  path updates") were done as a single commit. The intermediate state between them does not pass: `ruff
  check` fails on `src/locomani/sim/isaac/*.py` (E402) until `pyproject.toml`'s per-file-ignore is
  retargeted from `src/teleop_free/isaac/*.py`, and every test fails until `tests/*.py` stop importing the
  old flat names -- both of those edits are "tooling"/"tests", i.e. commit 2's scope. Per the instructions,
  splitting them would leave a commit that fails its own verification, so they went in together.
- **`pyproject.toml` per-file-ignores.** The old `"src/teleop_free/tools/*.py" = ["E402"]` blanket entry has
  no clean equivalent: `tools/` dissolved into many target packages, and only a handful of the redistributed
  files (the ones with a `chdir`/`sys.path` hack before their imports: `transition_events.py`,
  `reach_preflight.py`, `reach_preview.py`, `reach_wall_sweep.py`, `walk_pose_preflight.py`) still need an
  E402 exemption. Rather than blanket-exempt their whole new directories (which also hold files that don't
  need it), each got a per-line `# noqa: E402`, matching how every other non-`tools/` file in the old tree
  was already annotated. The isaac and `make_hand_points.py` entries were directory/file-level before and
  stayed that way, just retargeted to `src/locomani/sim/isaac/*.py` / `src/locomani/sim/make_hand_points.py`.
- **`scripts/archive/*.sh` touched.** Not named in the plan, but these 4 "dead runner" scripts (see
  `git log -1 -- scripts/archive/`) also named `src/teleop_free/...` and `$LOCOMANI_SRC`-relative paths;
  leaving them would fail check 6 verbatim (no scalebfm-style exemption exists for `scripts/archive/`), so
  their paths were updated the same way as the live scripts. `scripts/lint_sizes.py`, `scripts/lint_env.py`,
  the `Makefile` and `.pre-commit-config.yaml` needed no edits: none of them name a `teleop_free` path (the
  lint scripts already glob `src/**/*.py` generically; their `EXCLUDE` tuples only name `scalebfm`, which
  did not move).
- **`paths.py` relative-import depth bug found and fixed.** The `isaac/` subpackage moved one directory
  level deeper (`teleop_free/isaac` -> `locomani/sim/isaac`), so its existing `from .. import paths` (7
  files) would have resolved to the nonexistent `locomani.sim.paths` after a naive move; rewritten to the
  absolute `from locomani import paths`. Two similar depth-sensitive spots were fixed the same way:
  `run_ready_pilot.py`'s runtime `sys.path.insert(0, str(SRC))` (SRC now means `src/locomani`, not `src`) and
  `bimanual_scene.py`'s `paths.SRC`-relative call into `build_sim_scene.py` (now one level deeper, under
  `sim/`).
- **Two tests needed more than an import rewrite.** `tests/test_prefix_capture.py` faked
  `sys.modules["agent_server"]` / `["vla_round"]` for `Capture.instrument()`'s `import agent_server,
  vla_round`; under the new package, `from locomani.backends.sonic import agent_server, vla_round` resolves
  via the already-imported package's attribute before consulting `sys.modules`, so the fake never took.
  Switched to `patch.object` on the `locomani.backends.sonic` package's `agent_server`/`vla_round`
  attributes, which is what the "from package import submodule" lookup actually reads. Five other test
  files' `unittest.mock.patch("bare_name.attr")` string targets (`grasp_gym.np.random.uniform`,
  `ready_grasp_policy.prepare_ready`, `initial_pose.prepare_token_pose`, `measured_grasp.*`,
  `stream_reach.*`) were requalified to their full dotted path for the same reason -- these all patch
  attributes reached through a *lazy* `from locomani... import X` inside the function under test, so the
  qualified string target still resolves at call time.
- No module's placement differed from the doc's target-package table; all 111 files (105 Python + 6 shell)
  landed exactly where listed.

## Executed (4b/5)

Session on branch `refactor/step1-tooling`, worktree `.claude/worktrees/refactor-step1`, off `98a565e`.
Two commits, JOB order per the task (step 5 first, then step 4b family 1).

**Commit `28b0a49` -- refactor step 5: task seam + CLI + contract tests.**
`locomani/tasks/base.py` (`Task`, `TaskSchema`, `bracket_label`), `tasks/bottle/__init__.py` /
`tasks/crate/__init__.py` (`TASK` built from existing functions, lazily imported -- see judgment call
below), `tasks/registry.py` (`TASKS` + `get_task`), `locomani/cli.py` (thin verb+task argv dispatch onto
run_learn_pilot.py / eval_closed_loop.py / bimanual_carry.py / scripts/gate.py; not every (verb, task) is
wired, only what already exists -- crate has no convert/train/eval entry point), `tests/test_task_contract.py`
(parametrised over the registry), `Makefile` `cli` target.

*Judgment call, documented in both `tasks/{bottle,crate}/__init__.py` docstrings:* `classify_failure` /
`failure_reasons` use the `[tag] message` prefix convention (`tasks/base.bracket_label`) rather than
reusing `failure_taxonomy.py` as the spec's parenthetical listed. `failure_taxonomy.classify()` classifies
CLOSED-LOOP EVAL traces and needs a full per-timestep `trace`; fed a bare `why` string with no trace, its
own first branch (`"[vla]" not in why and not tr`) swallows nearly everything into one bucket ("A4 view
gate / pre-VLA"), which isn't a usable "why-string -> label" classifier for the scripted expert's own
failures. The bracket-tag convention is already how `grasp_gym.py`/`bimanual_carry.py` write `why` and how
`scripts/gate.py`'s own `_norm_reason` already treats it, so this is the smallest, most consistent fix.

*Bug found and fixed along the way:* eagerly importing `grasp_gym`/`bimanual_carry`/`bimanual_scene` at
`tasks/{bottle,crate}/__init__.py` module level breaks `scripts/gen_config_reference.py --check`, which
loads `BottleTaskConfig`/`CrateTaskConfig` with plain `python3` (no mujoco) -- importing anything under a
package always runs its `__init__.py` first. Fixed with a deferred (in-function) import inside each
`Task` field's wrapper closure.

Checks: ruff format/check, full pytest (168 passed at that point, 9 skipped, 8 subtests), lint_env
--strict, gen_config_reference --check, the import smoke-check, `bash -n` over scripts/ -- all green.

**Commit `32018a7` -- refactor step 4b, family 1: grasp_gym.py (pure extraction).**
Full account in the commit message; summary: `grasp_gym.py` (1846 lines: `main` 599, `_run_round` 843)
split into `plan.py` (per-target planner + sampling), `cli_args.py` (argparse), `round_loop.py` (per-round
sampling + execution loop), `round_phases.py` + `round_phases_approach.py` (`_run_round`'s five phases --
`spawn_and_settle`, `handover_to_vla`, `navigate`, `approach`, `grasp_and_verify` -- behind a
`RoundContext` dataclass), with `main`/`_build_scene_and_plan`/`run_round`/`_run_round` staying in
`grasp_gym.py`. Every moved name is re-exported so `from locomani.tasks.bottle.grasp_gym import X` still
works.

**Where extraction touched the official protocol path (point the GPU gate here):**
- `grasp_gym._run_round`'s control flow and every `execution.enter(...)` stage name/order are byte-for-byte
  unchanged (`vla`/`navigate`/`grasp_plan`/`verify_visibility`/`carry_and_verify`/`verify_terminal_hold`,
  reached in the same sequence) -- only WHERE the statements live changed, not their order or content.
- `grasp_gym.VLA_POLICY` / `ARM_PIN` / `RESTART_DEPLOY` / `fresh_deploy` / `_run_round` remain monkeypatchable
  exactly as before (`eval_closed_loop.py`, `diag_transition_live.py`, `diag_transition_cross.py`,
  `diag_crouched_handover.py` all still assign these directly on the `grasp_gym` module and are unchanged);
  `_run_round`'s body now reads them once per round into a `RoundContext` snapshot instead of as bare names,
  but the snapshot is taken at call time so a monkeypatch made before a round still lands.
  `grasp_gym.TabletopGrasp` gained the same treatment (a new `RoundContext.tabletop_grasp_cls` field) after
  the existing test suite caught it (see below) -- a test replaces the class wholesale via
  `patch.object(grasp_gym, "TabletopGrasp")`, which only reaches code that constructs it through that same
  name.
- One CLI-only behaviour change, not a protocol path: `main()`'s `--cases`+`--rounds` mismatch check used
  `ap.error(...)` (the argparse parser, in scope only inside the old single `main()`); split into
  `_build_scene_and_plan`, it no longer has `ap`, so it now does `raise SystemExit(...)` -- same refusal,
  message unchanged, but exit code 1 instead of argparse's 2 and no usage-string preamble.

**Bugs the existing test suite caught (not found by design, by `pytest` actually failing) --
this is exactly why decision 13's contract/regression tests matter for this kind of move:**
1. `tests/test_ready_vla_handover.py` patches `gym.initial_pose`, `gym.upper_ref`, `gym.TabletopGrasp`
   directly on the `grasp_gym` module object. `initial_pose`/`upper_ref` are shared singleton modules (like
   `time`/`np`), so re-importing them into `grasp_gym.py` (even though the code that uses them moved to
   `round_phases.py`, which imports them independently) is sufficient -- any reference to the same module
   object sees the same patched attribute. `TabletopGrasp` is a class, not a module: patching
   `grasp_gym.TabletopGrasp` only affects code that constructs it via `grasp_gym.TabletopGrasp(...)` /
   `ctx.tabletop_grasp_cls(...)`, not code with its own independent `from ... import TabletopGrasp` -- fixed
   with the `RoundContext.tabletop_grasp_cls` field above.
2. A duplicate-definition bug from a bad `sed` range during assembly (`obj_top`/`palm_frame_offset` appeared
   twice in the first draft of `grasp_gym.py`) -- caught by `ruff check` (F811), not a test, fixed before
   the test run.
3. The `if __name__ == "__main__": main()` guard at the very end of the original file was dropped during
   reassembly (missed because the original file's last 5 lines were never read in isolation) -- caught by
   deliberately running `python grasp_gym.py --help` as a real subprocess (not just `import`) after the
   pytest suite passed, specifically because the pytest suite alone cannot catch this class of bug.

**Checks (all green):** `ruff format . && ruff check .`; full pytest suite, 177 passed / 9 skipped / 11
subtests (up from 172 passed before this commit's five fixes); `lint_env.py --strict`;
`gen_config_reference.py --check`; the import smoke-check; `bash -n` over `scripts/*.sh` +
`scripts/stack/*.sh`; and, beyond the required list, `python src/locomani/tasks/bottle/grasp_gym.py --help`
as a real subprocess.

**`lint_sizes.py`:** 11 files / 66 functions over cap -> 10 files / 72 functions over cap. `grasp_gym.py`
dropped out of the file list entirely (1846 -> 567 lines). The function count rose even though the worst
single function shrank from 843 to ~250 lines: splitting exposed several already-oversized pieces (e.g.
`do_approach` at 102 lines, `run_round` at 105) that were previously invisible, folded inside the two
mega-functions. Left over cap, with reasons (no `# noqa: size` marker exists in `lint_sizes.py` --
it does not scan for one -- so these are just listed here):
- `cli_args.py:parse_args` (198) -- ~230 lines of it are `ap.add_argument(...)` calls with long, deliberate
  `help=` prose (decision 8's provenance principle: the CLI is the documentation). Splitting into
  `_add_spawn_args(ap)` / `_add_gate_args(ap)` etc. is possible but each sub-function would just be a
  differently-shaped pile of the same boilerplate; not attempted this session.
- `grasp_gym.py:_build_scene_and_plan` (130) and `round_phases.py:_phase_navigate` (102) -- linear
  setup code with no natural sub-boundary that doesn't require threading MORE state across an ADDITIONAL
  function call for a single-use helper.
- `grasp_gym.py:run_round` (105), `grasp_gym.py:_run_round` (89) -- both already thin wrappers; further
  splitting `run_round`'s try/except/finally would separate the failure path from the recovery-hold logic
  it depends on.
- `round_phases_approach.py:do_approach` (104), `round_phases_approach.py:_phase_grasp_and_verify` (97) --
  `do_approach` is one `while` loop with tightly-coupled exit conditions (stall/edge/reachable/visible-brake
  checks all read/write the same locals every tick); splitting it means passing most of its local state
  through an extra function boundary every iteration, for a live control loop already verified against
  recorded field behaviour (docs/handoff/refactor-plan.md decision 2's GPU gate) -- exactly the case
  the task's `# noqa: size` escape hatch describes, even though this repo's lint script has no such marker
  to attach it to.

**Not attempted this session** (JOB 2's families 2-6: `stream_reach.py`, `vla_round.py`, `primitives.py`,
crate's `bimanual_wall.py`/`bimanual_carry.py`, and the conditional family 6 list). Stopped here on budget;
family 1 alone required three genuine correctness fixes only the test suite / a real subprocess run caught
(see above), which argues for doing the remaining families with the same care rather than rushing them.

**Commit (family 2, next session) -- refactor step 4b, family 2: `stream_reach.py` (pure extraction).**
`stream_reach.py` (1576 lines) split into three new files plus what stays behind:

- `streamed_arm.py` -- `StreamedArm` + `PlannerArm` + `yaw_only`, and every constant only they use (`CHUNK`,
  `JOINT_SPEED`, `FINE_SPEED`, `SETTLE`, `REFINE_TOL`, `REFINE_MAX`, `NUDGE_MAX`, `LEFT_GUARD`,
  `LEFT_SAFE_M`, `LEFT_RETREAT_M`, `LEFT_GUARD_RATE`, `SQUAT_DROP_MAX`, `SQUAT_HEIGHT_BIAS`,
  `SQUAT_SETTLE`, `SQUAT_GAIN`, `SQUAT_MAX`) -- as specified.
- `_servo_continuous` (631 lines) did not fit the task's single `servo.py` as one file once its `ServoState`
  dataclass and phase functions were written out (a 989-line first draft), so it became three files instead,
  a strict one-way dependency chain to avoid a circular import (each arrow is "imports from"):
  `servo_ticks.py` -> `servo_phases.py` -> `servo.py`, and `servo_ticks.py` -> `servo.py` directly.
  - `servo.py` (229 lines) -- the leaf: the continuous-servo tuning constants (`LIFT_MIN`, `LAST_EGO_VIS`,
    `V_TRAVEL`..`PRE_XY_MAX`, `SAFE_GAP`, `CLOSE_ENT_MAX`, `CLOSE_SEATED`, `FREEZE_TRAVEL`, `STAGE_UP`,
    `PRE_X_MIN`/`MAX`, `PRE_Z_MIN`/`MAX`), `_clipn`, and the `ServoState` dataclass (every local the
    original loop closed over -- ~45 fields, each with a one-line comment naming what it replaces).
    `stream_reach.py` re-exports the tuning constants from here.
  - `servo_phases.py` (302 lines) -- the six phase functions (`_phase_up`, `_phase_down`,
    `_phase_in_backoff`, `_phase_in`, `_phase_close`, `_phase_lift`), each now `f(state) -> str |
    tuple[bool, str] | None` instead of a branch of the original `if phase == ...` chain.
  - `servo_ticks.py` (398 lines) -- `_servo_setup` (the pre-`while` setup), `_tick_prelude` (telemetry +
    replan + the moved-too-much guard), `_dispatch_phase` (routes to a `servo_phases.py` function by
    `state.phase`), `_tick_advance` (the common ref/IK-solve/guards/push/pin/stall/logging tail), and
    `_servo_continuous` itself (now a ~40-line loop calling the three of these in order). `stream_reach.py`
    imports `_servo_continuous` from here.
  - Return-value convention (documented in `servo.py`'s module docstring, since all three files share it):
    a phase/prelude/tail function returns `None` to fall through to whatever runs next, `"retick"` to skip
    the rest of the tick (the original branch's bare `continue`), or `(ok, why)` to end `_servo_continuous`
    now (the original branch's `return`). `state.tgt`/`state.v` are how a phase function hands its computed
    target/speed to `_tick_advance`, replacing the original's `tgt, v = ...`.
- `stream_reach.py` (229 lines, down from 1576) keeps `make_arm`, `_raise_hand`, `_creep_to_band`,
  `toppled`, `reanchor`, `execute_grasp`/`_execute_grasp`, `ApproachStatus`/`ApproachResult`, `ARM_CHANNEL`/
  `ARM_VEL`, `TRANSITION_POST_MAX`, and the handful of already-dead constants left untouched in place
  (`DEPTH_TOL`, `ALIGN_TOL`, `GATE_TOL`, `GRASP_TOL`, `STEP_MAX`, `CROUCH_BACK`, `SQUAT_TRIGGER` -- confirmed
  unused anywhere in the repo before moving anything, so left exactly where they were rather than moved or
  deleted). Every name that used to live here and moved is re-exported (imported at module top level), so
  `from locomani.backends.sonic.stream_reach import X` is unchanged for every caller.

**Monkeypatch hazard found and fixed (the task's specific ask to check for this):** grepping `patch(` /
`patch.object(` against `stream_reach`/`sr` across `tests/` and `src/` turned up five patched names
(`toppled`, `_servo_continuous`, `make_arm`, `PlannerArm`, `_creep_to_band`) plus, from a second grep for
every bare `sr.X` / `stream_reach.X` attribute read (not just `patch(...)`), three more that had to stay
readable off the module even though nothing patches them (`SAFE_GAP`, `_HAND`, `SQUAT_SETTLE`, read by
`diagnostic_transport.py` and its test, `test_transport_guard.py`, `test_transport_rotation.py`,
`diag_crouched_handover.py`). All eight are satisfied by re-export except one real hazard:
`_servo_continuous`'s OWN internal `toppled(...)` call (now living in `servo_ticks.py`, a different module
than `stream_reach.py`) would no longer see `patch("locomani.backends.sonic.stream_reach.toppled", ...)` if
written as a plain `from ... import toppled` -- that binds to the real function once, at import time, in
`servo_ticks.py`'s own namespace, which patching `stream_reach.toppled` never touches. Confirmed this is a
real (not theoretical) risk: `test_measured_grasp.py::test_near_target_commands_arm_retraction_...` and two
similar tests in `test_recoverable_grasp_policy.py`/`test_visible_expert.py` patch exactly this and DO reach
`_servo_continuous`'s real tick loop with a fabricated state missing the object's `quat` key -- the real
`toppled()` would `KeyError` on it if the patch didn't take. Fixed by keeping `toppled` defined in
`stream_reach.py` (never moved) and having `_tick_prelude` call it via a deferred, module-qualified
`from locomani.backends.sonic import stream_reach; stream_reach.toppled(...)` instead of an imported name --
same fix applied to the (dead, unread-by-anything-external, but principle-of-least-surprise) `LAST_EGO_VIS`
global write, which now goes through `servo.LAST_EGO_VIS = ...` so it keeps landing on the one module stream_
reach.py's re-export points at, instead of silently forking into a second copy on `servo_ticks.py`.

**Checks (all green):** `ruff format . && ruff check .` (needed one `ruff check --fix` for import order in
`servo_ticks.py`); full pytest suite, 177 passed / 9 skipped / 11 subtests (unchanged from before this
commit -- pure extraction, no behaviour change); re-ran the 11 tests that patch/read `stream_reach`
attributes individually (`test_measured_grasp.py`, `test_recoverable_grasp_policy.py`,
`test_visible_expert.py`, `test_diagnostic_transport.py`, `test_transport_guard.py`,
`test_transport_rotation.py`, `test_ready_grasp_policy.py`, `test_grasp_point_ik.py`,
`test_execution_record.py`, `test_reach_config.py`, `test_planner_reference_takeover.py` -- 84 passed);
`lint_env.py --strict`; `gen_config_reference.py --check`; the import smoke-check; `bash -n` over
`scripts/*.sh` + `scripts/stack/*.sh`; `eval_closed_loop.py --help`, `bimanual_carry.py --help`,
`bimanual_wall.py --help` as real subprocesses.

**`lint_sizes.py`:** 10 files / 72 functions over cap (session start) -> 9 files / 74 functions over cap.
`stream_reach.py` dropped out of the file list (1576 -> 229 lines); `streamed_arm.py`, `servo.py` never
entered it. Left over cap, with reasons:
- `servo_phases.py:_phase_in` (197) -- one continuous alignment/gate computation (trim integration, z-floor
  feasibility, the align/entry gate, the close-gate hysteresis) where every intermediate value depends on
  the one before it in the same palm-frame geometry; splitting it means threading `full`/`e_ent`/`lat`/
  `gate` through an extra function boundary for a single-use handful of local floats -- the same "linear
  live-control-loop math with no natural sub-boundary" case as family 1's `do_approach`.
- `servo_ticks.py:_tick_advance` (169) -- the common tail already reads as three logical steps (build `cmd`
  from the phase's target, solve IK against it, then guard/push/pin/log the result), but every step reads
  state the previous one just wrote (`cmd`, `aim_pf`, `rot_use`, `q_sol`) and the guards
  (`transport_guard`/`reference_guard`/stall) all need the SAME `q_sol`/`res` right after the solve --
  splitting would mean passing 4-5 fresh locals through another function call for a 50 Hz loop already
  proven against recorded field behaviour.
- `servo_ticks.py:_tick_prelude` (105) -- similarly linear (telemetry bookkeeping, then the replan check,
  then the moved-too-much guard), each block short but the whole thing is "everything that has to happen
  once before the phase dispatch," not something with an obvious mid-cut.
- `stream_reach.py:_creep_to_band` (97) -- pre-existing, untouched by this family, already noted as
  `# noqa: size`-shaped in the family-1 report's spirit (a single approach/align loop like `do_approach`).

**Not attempted this session** (JOB 2's remaining families 3, 5, 4): `vla_round.py`, crate's
`bimanual_wall.py`/`bimanual_carry.py`, `primitives.py`.

**Commit (family 3, next session) -- refactor step 4b, family 3: `vla_round.py` (pure extraction).**
`vla_round.py` (1832 lines) split into four new files plus what stays behind. As with family 2 this grew
past the task's named targets because the size cap forced it, not because the shape changed:

- `vla_obs.py` -- `ObsBuilder`, `PolicyLink`, `_tstart_view`, `_obj_extra` (229 lines).
- `vla_client.py` -- `_grip_gain`, `_ensemble`, `_stalled`, `make_vla_policy_async` (550 lines). The task
  named this file for "observation assembly, chunk selection/ensemble, send loop" helpers; observation
  assembly (`ObsBuilder`) ended up in `vla_obs.py` instead because `make_vla_policy` (staying in
  `vla_round.py`) also needs it, and `vla_obs.py` has zero dependency on anything that reads live
  `vla_round` globals, which keeps the dependency graph a clean one-way chain (see below).
- `lockstep_client.py` -- `LOCKSTEP_PORT`, `LockstepLink`, `LockstepClock`, `install_lockstep_shim`,
  `load_latency_dist`, `make_vla_policy_lockstep` (536 lines), exactly the cluster the task named.
- `clean_clock.py` -- `CleanClock`, `install_clean_shim` (158 lines), exactly as named.
- `vla_round.py` (520 lines, down from 1832) keeps `make_vla_policy` (the blocking client) and every
  module-level knob (`TOKEN_DIM`, `LIFT_M`, `NEAR_M`, `TIMEOUT_S`, `STALL_ABORT_S`, `SNAP`, `GRIP_GAIN`,
  `ENSEMBLE_N`, `ENSEMBLE_M`, `RTC_ON`, `TRANSITION_COMMIT`, `TRANSITION_DECODER`, `GRIP_DELAY`,
  `STALL_ABORT_M`, `DATASET_FPS`, `OBS_DUMP`, `REPLACE`, `LOCKSTEP_NO_SEED`), plus the two dead imports
  (`math`, `sys` -- confirmed unused anywhere in the file before touching anything, left in place). Every
  name that used to live here and moved is re-exported (plain top-level import), so
  `from locomani.backends.sonic.vla_round import X` is unchanged for every caller.

Dependency graph, to avoid a circular import while still letting every satellite read `vla_round`'s knobs
live: `vla_round.py` imports FROM all four satellites at the top level (for re-export); `vla_client.py` and
`lockstep_client.py` import FROM `vla_obs.py` at the top level (no cycle, `vla_obs.py` needs nothing from
them); `lockstep_client.py` also imports `_ensemble`/`_grip_gain`/`_stalled` FROM `vla_client.py` at the top
level. None of the four satellites import `vla_round.py` at their own top level -- every read of a
`vla_round` constant from inside a satellite is a **deferred** `from locomani.backends.sonic import
vla_round as vr` done inside the specific function that needs it (not at module scope), so by the time it
actually runs both modules are already fully loaded.

**Monkeypatch hazard, much larger than family 2's single `toppled` case:** `eval_closed_loop.py` overrides
nearly every module-level knob by assignment before building a round (`vla_round.DATASET_FPS = ...`,
`.GRIP_GAIN`, `.RTC_ON`, `.TRANSITION_COMMIT`, `.TRANSITION_DECODER`, `.ENSEMBLE_N`, `.ENSEMBLE_M`,
`.TIMEOUT_S`, `.SNAP`, `.STALL_ABORT_S`), constructs `vla_round.PolicyLink`/`.LockstepLink`/`.LockstepClock`/
`.CleanClock` and calls `.install_lockstep_shim`/`.install_clean_shim`/`.load_latency_dist`/
`.make_vla_policy_lockstep`/`.make_vla_policy_async`/`.make_vla_policy` directly off the module, and the RTC
tools (`transition_capture.py`, `diag_transition_live.py`) replace `vla_round.PolicyLink.get_chunk` and
`vla_round.make_vla_policy_async` wholesale. Checked which of these are read from CODE THAT MOVED (as
opposed to just called by an external qualified reference, which any re-export already satisfies): every
constant read inside `_grip_gain`, `_ensemble`, `_stalled`, `make_vla_policy_async` (and its nested `worker`
closure), `make_vla_policy_lockstep` (and its nested `vla_round`/`infer`/`frame_cmd` closures), `PolicyLink.
get_chunk`, and `CleanClock.__init__` needed the deferred-`vr` fix above; `PolicyLink.get_chunk`'s class-
level monkeypatch (`vla_round.PolicyLink.get_chunk = get`) needed no fix at all, since it mutates the shared
class object itself and every instance method call re-resolves through that same class regardless of which
module holds a name pointing at it. The deferred alias is `vr`, not `vla_round`, in every closure-building
factory: each builds a nested function itself named `vla_round` (matching the established
`vla_round(p, reset, meta, ...)` convention every channel uses), which would shadow a same-named module
import inside that scope.

Verified the fix, not just asserted it: a standalone script imported `vla_round`, set
`vla_round.ENSEMBLE_N/GRIP_GAIN/DATASET_FPS/STALL_ABORT_S/SNAP` the way `eval_closed_loop.py` does, then
called the moved `_grip_gain`/`_stalled` directly and confirmed both read the overridden values (not the
file's own import-time defaults), and confirmed `vla_round.PolicyLink is vla_obs.PolicyLink` (same object,
so a class-level patch reaches every holder of a reference).

**Checks (all green):** `ruff format . && ruff check .`; full pytest suite, 177 passed / 9 skipped / 11
subtests (unchanged); `lint_env.py --strict`; `gen_config_reference.py --check`; the import smoke-check;
`bash -n` over `scripts/*.sh` + `scripts/stack/*.sh`; `eval_closed_loop.py`/`bimanual_carry.py`/
`bimanual_wall.py --help` as real subprocesses; the standalone monkeypatch-live-read smoke script above.
Noted: only one test file (`test_prefix_capture.py`) references `vla_round` at all, and it fails when run
in complete isolation (`patch.object(sonic_pkg, "agent_server"/"vla_round", ...)` needs those submodules
already registered as attributes of the `locomani.backends.sonic` package, which only happens once some
other, earlier-collected test has imported something that imports `primitives.py` -- `agent_server` -- or
`vla_round.py` first) -- confirmed this is a **pre-existing** test-order fragility unrelated to this
family's split, not a regression: the file passes 3/3 inside the full suite, both before and after this
commit, and `agent_server`'s registration comes from `primitives.py` (untouched by this family).

**`lint_sizes.py`:** 9 files / 74 functions over cap (after family 2) -> 8 files / 74 functions over cap.
`vla_round.py` drops out of the file list; `vla_obs.py`, `vla_client.py`, `lockstep_client.py`,
`clean_clock.py` never entered it. Left over cap, with reasons -- every one is a nested nonlocal-state
nested factory (`make_vla_policy`/`make_vla_policy_async`/`make_vla_policy_lockstep` each return a closure
literally named `vla_round`, which is what grasp_gym.VLA_POLICY gets set to) built around a worker thread,
a lock, and a handful of mutable dicts (`latest`, `buf`, `lat_rec`) that every phase of the loop reads and
writes; unlike family 2's flat `_servo_continuous`, splitting these further means either passing 6-8
threading primitives through new function boundaries or promoting them to a shared context object the way
`ServoState` did for the streamed servo -- attempted a first pass at this and stopped: the worker thread
closure captures `obs_src`/`link`/`lock`/`latest`/`buf`/`stop`/`commitment` by reference already, and turning
those into object attributes changes nothing about the coupling, only where the names live, for a live
control/threading path with no offline test coverage to catch a mistake (see the `test_prefix_capture.py`
note above) -- the single existing regression backstop for this whole family is the full pytest suite plus
the GPU gate, neither of which exercises the threaded VLA clients at all. Listed rather than split further:
- `vla_client.py:make_vla_policy_async` (450) / `vla_client.py:vla_round` (414, the nested closure it
  returns) -- worker thread + lock + keepalive + main loop, as above.
- `lockstep_client.py:make_vla_policy_lockstep` (251) / `lockstep_client.py:vla_round` (218) -- the
  simulation-clock equivalent of the same shape (`observe`/`infer`/`frame_cmd` closures instead of a thread).
- `vla_round.py:make_vla_policy` (244) / `vla_round.py:vla_round` (234) -- the blocking client's own nested
  closure (feeder thread instead of a worker), pre-existing shape, unchanged by this family.

**Not attempted this session** (JOB 2's remaining families 5, 4): crate's `bimanual_wall.py`/
`bimanual_carry.py`, `primitives.py`.

**Commit (family 5, next session) -- refactor step 4b, family 5: crate `run_round` (pure extraction) +
config/CLI wiring.** `bimanual_carry.py` (773 lines) and `bimanual_wall.py` (851 lines) each had one giant
`run_round` (415 / 530 lines) with several nested closures capturing its locals -- same shape as
grasp_gym.py's `_run_round` from family 1, so this family mirrors that split exactly: a `*RoundContext`
dataclass, phase functions taking `ctx`, closures converted to plain functions taking `ctx` (or no `ctx` at
all when they capture nothing -- `mirror29`, `stand_fault`), and `run_round` itself reduced to a thin
orchestrator (`for phase in (...): result = phase(ctx); if result is not None: return result`) that still
lives in the original file so external callers (`tasks/crate/__init__.py`'s `bimanual_carry.run_round(...)`
wrapper, `main()`'s own call) resolve to the same name.

- **`bimanual_carry.py`** -- `run_round`'s five phases (`_phase_reset`, `_phase_approach`, `_phase_grasp`,
  `_phase_close`, `_phase_lift`) split into `carry_phases.py` (229 lines: reset/approach/grasp) and
  `carry_phases_lift.py` (114 lines: close/lift) -- all five would not fit carry_phases.py's own 600-line
  cap. `CarryRoundContext` has 26 fields (constructor params + `surf`/`hx`/`s`/`t_video` from reset +
  `centre`/`bars`/`n`/`y_hat`/`psi`/`stand_xy`/`edge_x`/`q_free`/`ramp`/`q_pre17`/`arms`/`R_stand`/`rot_w`
  from approach + `ref` from grasp). `bimanual_carry.py` itself drops to 397 lines and keeps `BimanualArms`,
  the module constants, `rot_z`/`rot_y`/`crate_frame`/`hand_txt`/`crate_tilt_deg`, and `run_round`.
- **`bimanual_wall.py`** -- `run_round`'s six phases (`_phase_reset`, `_phase_stand`, `_phase_hover`,
  `_phase_descend`, `_phase_close`, `_phase_lift`) split into `wall_phases.py` (568 lines:
  reset/stand/hover, plus `solve_for`/`mirror29`/`measure_stand`/`stand_fault`/`servo_tick`) and
  `wall_phases_descend.py` (252 lines: descend/close/lift) -- same 600-line reason.
  `WallRoundContext` has 27 fields, close in shape to carry's (`c`/`n`/`y_hat`/`psi`/`sign`/`R` from
  `wall_frame`, `so`/`stand_xy` from reset, `depth`/`crate_start`/`ref`/`rot_w` from hover).
  `bimanual_wall.py` itself drops to 360 lines (plus the CLI addition below) and keeps `wall_frame`,
  `wall_point`, `errs_of`, `arms_offset`, the module constants, and `run_round`. Several names it still
  imports (`APPR_SPEED`, `EDGE_FLOOR`, `OVERDRIVE`, `OVERDRIVE_CAP`, `STAND_TOL`, `WAYPOINT_BACK`,
  `WAYPOINT_TOL`, `BimanualArms`, `crate_tilt_deg`, `hand_txt`, `rot_y`, `rot_z`) are unused in its OWN
  surviving body -- they stay imported because `wall_phases.py`/`wall_phases_descend.py` read them back
  through `bimanual_wall` (`from locomani.tasks.crate.bimanual_wall import ...`), the same
  re-export-for-a-satellite-file shape family 3 used for vla_round.py's knobs; pruning them as "unused"
  would have broken that re-export.

No monkeypatch hazard here (checked): neither file's module constants are ever overridden by assignment
anywhere in the repo (`grep "bimanual_carry\.[A-Z_]* *=\|bimanual_wall\.[A-Z_]* *="` across src/ and tests/
is empty), unlike vla_round.py's family-3 knobs -- so every constant a phase function needs is a plain
deferred (in-function) import, not a module-qualified `vr.`-style read. The circular-import shape is
otherwise identical to round_phases.py/vla_client.py: each phases file must not import its parent
(`bimanual_carry`/`bimanual_wall`) at its own top level, since the parent imports the phase functions FROM
it for re-export -- fixed the same way, with deferred in-function imports of everything the phase functions
need from the parent.

**Verified mechanically, not just by eye:** for both `*RoundContext` classes, extracted every `ctx.<name>`
read/write from the phases files via a small script and diffed it against the dataclass's declared field
names -- zero mismatches in either direction (no typo'd field, no promoted-but-never-read field only by
accident of the check missing it) for both `CarryRoundContext` and `WallRoundContext`.

**Mid-task addition (from the coordinator, folded into this family's commit): `--config`/`--set` CLI on both
files.** The crate GPU gate's baseline (`tests/gate/baseline_crate.json`) was recorded with
`WALL_WALK=1 WALL_FEATURE=rail` back when those were env vars; after refactor step 3, `CrateTaskConfig.
wall_walk` defaults to `False`, so reproducing that baseline needs an explicit override, and neither script
had any way to pass one. Added, via `locomani.config_base` (the same `--config PATH` / repeatable
`--set KEY=VALUE` shape `run_learn_pilot.py` already uses):
- **`bimanual_wall.py`**: `main()` loads `CrateTaskConfig` (`--config` then `--set` overrides) and
  reassigns `FEATURE`/`WALK` from it, THEN rederives `DEPTH_AXIS`/`WALL_AXIS`/`WALL_Y`/`NEAR_X`/
  `CRATE_YAW_DELTA`/`GRIP_BELOW` from the (possibly overridden) `FEATURE` -- factored the module-level
  `if FEATURE == "wall": ... else: ...` block that used to compute those six into a new
  `_wall_feature_constants(feature)` helper (called once at module load for the default, once more in
  `main()` for the override) so the override path can never drift from the default path's own branch.
  Verified with a standalone script that loads `configs/tasks/crate_wall.json`, applies it the way `main()`
  does, and confirms `bimanual_wall.WALK` becomes `True` and (separately, forcing `FEATURE="wall"`) the six
  derived constants match the original `if` branch's literal values exactly (0, 1, 0.1267, 0.2093, -pi/2,
  0.080). This is the one file where the override actually changes behaviour: `wall_phases.py`'s
  `_phase_stand`/`_phase_hover` read `WALK`/`FEATURE`-derived constants via a plain deferred import done
  fresh every round, so they see whatever `main()` set before the round loop started -- same "module-level
  knob, re-read every call" shape as vla_round.py's family-3 constants, just without the monkeypatch hazard
  since nothing else overrides these.
- **`bimanual_carry.py`**: `main()` loads and validates the same `CrateTaskConfig` (`--config`/`--set`,
  prints it) for CLI consistency with `bimanual_wall.py`, but does not wire it to any behaviour -- this
  file's round logic has no field of `CrateTaskConfig` to apply (confirmed: no `CrateTaskConfig`/
  `DEFAULT_CRATE_TASK` reference existed anywhere in it before this change). Said so explicitly in the
  code comment and in `run_bimanual_pilot.sh`'s header rather than fabricating a hookup.
- **`configs/tasks/crate_wall.json`** = `{"schema_version": 1, "wall_feature": "rail", "wall_walk": true}`,
  the canonical config reproducing the recorded gate baseline; loads and validates via
  `CrateTaskConfig.load` (checked directly). Mentioned in both scripts' header comments.
- **`scripts/run_wall_pilot.sh`** / **`scripts/run_bimanual_pilot.sh`**: any args after `[rounds] [seed]` now
  forward verbatim to the python call via `shift $(($# < 2 ? $# : 2)); ... "$@"`, so
  `bash scripts/run_wall_pilot.sh 10 0 --set wall_walk=true` or
  `--config configs/tasks/crate_wall.json` works; `--rounds`/`--seed`/`--manifest`/`--label` are
  untouched (still positional/defaulted exactly as before). `bash -n` clean on both.

**Checks (all green):** `ruff format . && ruff check .` (repo-wide, 274 files); full pytest suite, 177
passed / 9 skipped / 11 subtests (unchanged); `lint_env.py --strict`; `gen_config_reference.py --check`;
the import smoke-check (both crate files, plus the existing bottle/vla_round/eval_closed_loop/cli set);
`bash -n` over every `scripts/*.sh` + `scripts/stack/*.sh`; `eval_closed_loop.py --help`,
`bimanual_carry.py --help`, `bimanual_wall.py --help` as real subprocesses (both crate ones show the new
`--config`/`--set` flags alongside the unchanged originals); the two standalone smoke scripts above
(ctx-field consistency + the wall config-override chain). No test file in `tests/` exercises either
crate file's round logic (no fixture for a live sim stack), so this family's regression backstop is the
manual line-by-line transcription review plus the mechanical `ctx.` consistency check, same limitation
family 3 noted for the threaded VLA clients -- the GPU gate is the real check for behavioural parity.

**`lint_sizes.py`:** 8 files / 74 functions over cap (after family 3) -> 6 files / 79 functions over cap.
`bimanual_carry.py` and `bimanual_wall.py` both drop out of the file list (773 -> 397, 851 -> 360 lines,
plus the CLI addition); none of the four new phase files entered it. Left over cap, with reasons -- every
one is a single control loop or a linear setup block with no natural sub-boundary, same class as family
1/2's documented exceptions:
- `wall_phases.py:_phase_stand` (183) -- the walk-in attempt loop (drive to a waypoint, then straight in,
  brake, measure, retry up to `WALK_ATTEMPTS`) and the non-WALK teleport branch share the function only
  because they are mutually exclusive halves of "get the robot onto the stand"; splitting the WALK branch
  further means threading `q_walk17`/`way_xy`/`leg`/`best_along` through another boundary for a
  single-shot attempt loop already gated by the retry logic itself.
- `carry_phases.py:_phase_grasp` (178) / `wall_phases.py:_phase_hover` (133) / `carry_phases.py:
  _phase_approach` (152) / `wall_phases_descend.py:_phase_descend` (114) / `wall_phases_descend.py:
  _phase_lift` (88) -- each is one servo/descend/lift control loop (or, for `_phase_approach`, one
  walk-to-stand loop) where every statement reads state the previous one just wrote in the SAME tick;
  the same "linear live-control-loop math" case as family 1's `do_approach` and family 2's `_phase_in`/
  `_tick_advance`.

**Not attempted this session** (JOB 2's conditional family 4, gated on 2/3/5 all green -- they are):
`primitives.py`.
