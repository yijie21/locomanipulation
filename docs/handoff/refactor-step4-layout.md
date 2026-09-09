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
