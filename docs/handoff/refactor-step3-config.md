# Refactor step 3 — knobs into config objects (decision 8)

Rule: a knob is a value that changes what the robot does or what gets recorded. Knobs live in dataclasses; each field
carries a one-line comment (meaning + where it was decided). Code reads config objects, never `os.environ`, except the
whitelist in `scripts/lint_env.py` (devices, ports, paths). Every run writes the full config snapshot into its output
directory; that file is the provenance.

## The four config objects

| object | module | fields (source today) |
|---|---|---|
| `BottleTaskConfig` | `locomani/tasks/bottle/config.py` | `reach: ReachConfig` (unchanged, nested); `target="water_bottle"`, `prompt="pick up the water bottle"`; spawn: `edge_range=(1.1,1.3)`, `face_object=True`, `yaw_err_deg=20`, `lateral=0.20`, `edge_jitter=0.20` (run_learn_pilot `SPAWN`, grasp_gym argparse); `stand_retries=0`; gym knobs: `restart_deploy` (GYM_RESTART_DEPLOY), `arm_pin` (GYM_ARM_PIN), `pose_sidecar` (LOCOMANI_POSE_SIDECAR), `video_include_init` (SIM_VIDEO_INCLUDE_INIT), `init_token_file` (TF_INIT_TOKEN_FILE / TF_INIT_TOKEN), `arm_channel`, `arm_vel`, `left_guard` (GYM_ARM_CHANNEL / GYM_ARM_VEL / GYM_LEFT_GUARD in stream_reach), `one_shot` if still a CLI flag |
| `CrateTaskConfig` | `locomani/tasks/crate/config.py` | `wall_feature` (WALL_FEATURE), `wall_walk` (WALL_WALK), `gripper_gear` (GRIPPER_GEAR), `couple_fingers` (COUPLE_FINGERS), `crate_setback` (CRATE_SETBACK), `scene_preset` (SCENE_PRESET), rounds/seed/label (argparse) |
| `SonicBackendConfig` | `locomani/backends/sonic/config.py` | `no_encoder=True` (NO_ENCODER), `hand` (HAND in agent_server), `upper_slew_deg` (GYM_UPPER_SLEW_DEG), `grip_delay` (TF_GRIP_DELAY), `obs_dump` (TF_OBS_DUMP), `replace` (LOCOMANI_REPLACE), `wrist_cams` (WRIST_CAMS), `sim_vis` (SIM_VIS), `scene_random` (SCENE_RANDOM), `dataset_fps=25` (TF_DATASET_FPS), lockstep: `ack_timeout, warmup_ticks, cmd_delay, jitter, jitter_seed, no_seed` (LOCKSTEP_*); deploy root/bin stay env (whitelist, the shell reads them) |
| `EvalConfig` | `locomani/pipeline/eval_config.py` | everything `eval_closed_loop.py` takes today: `rounds`, `seed`, `client` (async/blocking/lockstep), `budget_s`, `replan_interval`, `execution_horizon`, `ensemble_n`, `ensemble_m`, `rtc*`, `transition_commit*`, `policy_port`, `stand_retries`, `parallel` (EVAL_PARALLEL), `latency_dist`, trace/out paths are CLI, not config |
| `PilotConfig` | `locomani/pipeline/pilot_config.py` | run_learn_pilot argparse: `successes, batch_rounds, max_batches, holdout, steps, save_steps, train_gpu, eval_rounds, eval_seed, seed0, stack_env, squat_max (SQUAT_MAX)` |

`SceneBuildConfig` (SCENE_SEED, SCENE_TABLE_TOP, SCENE_PRESET in build_sim_scene / scene_lib) becomes argparse flags of
`build_sim_scene.py` with the same defaults; the scene builder is a one-shot tool, not a runtime knob.

Diagnostic tools (`tasks/crate/reach_preflight.py` and friends: PITCH0, SAG, PW, UP, OVER, OUT, WARM, ROT, SAVE,
POSE_LEFT_DEG, SCENE_DIR, VIA_CHAIN; `pipeline/instance_watchdog.py`: WATCHDOG_*, LOCOMANI_TMP) convert their env reads
to argparse flags with the same names and defaults. `RANK` in train_transition_pilot is torch's, whitelist it.

## Loading and overriding

- Each config is a frozen dataclass with `to_dict()/from_dict()/load(path)/dump(path)`; JSON on disk, `schema_version`.
- CLIs keep only: `--config <path>` (task / eval / pilot as applicable), `--out <dir>`, and `--set key=value` repeated
  (typed by the dataclass field). All other semantic argparse flags are removed; their defaults move into the dataclass.
- `ReachConfig.load` keeps `RETIRED_KEYS` (introduced in step 2) so historical reach_config.json files still load.
- The pipeline (pilot / eval) writes `<out>/config_snapshot.json` = {task, backend, eval, pilot, git sha, started}.
  `provenance.json` is replaced by it (keep the source-file digests in the same file).

## Shell boundary

`scripts/stack/start_stack.sh` still reads env (INSTANCE, ports, NO_ENCODER, SIM_VIDEO_DIR, LOCOMANI_SIM_DIR, ...). The
ONLY Python that sets those is `locomani/pipeline/stack.py` (`start_stack(backend_cfg, task_cfg, instance, gpu,
video_dir)` / `stop_instance(instance, port)`), which turns config into the env dict for the shell. run_learn_pilot and
eval_closed_loop call it instead of building env dicts inline.

## Order

1. `SonicBackendConfig` + `pipeline/stack.py` (no behaviour change: defaults = today's env defaults).
2. `BottleTaskConfig` (nesting ReachConfig) + grasp_gym/stream_reach read it; run_learn_pilot passes it.
3. `EvalConfig` + eval_closed_loop CLI shrink.
4. `PilotConfig` + run_learn_pilot CLI shrink + config_snapshot.json.
5. `CrateTaskConfig`; preflight/watchdog argparse conversion.
6. `scripts/lint_env.py --strict` passes; flip pre-commit to strict.

Verification per step: ruff, unit tests, import checks, and for 1-4 the GPU gate (bottle) once before merging step 3
as a whole. The official protocol under the new configs must reproduce `configs/collection/ready_grasp_visible.json`
+ the launch.sh argv of ready_visible200 exactly (write `configs/tasks/bottle_visible.json`, `configs/eval/closed_loop_100.json`,
`configs/pilot/learn_200.json` as the canonical files and delete `configs/collection/` once step 6 archives the verdicts).

## Executed

2026-09-09, one commit per item, all off `c1f73dd`:

| step | commit | lint_env before -> after |
|---|---|---|
| 1 SonicBackendConfig + stack.py | `4902f0a` | 64 vars / 86 reads -> 48 / 64 |
| 2 BottleTaskConfig | `8587a2a` | 48 / 64 -> 41 / 53 |
| 3 EvalConfig + eval_closed_loop CLI shrink | `2081484` | 41 / 53 -> 36 / 43 |
| 4 PilotConfig + run_learn_pilot CLI shrink | `053531a` | 36 / 43 -> 35 / 42 |
| 5 CrateTaskConfig; preflight/watchdog argparse | `da6e2d0` | 35 / 42 -> 14 / 18 |
| 6 whitelist + strict + config-doc generator | (this commit) | 14 / 18 -> 0 / 0, `--strict` exits 0 |

Every commit passed: `ruff format --check` + `ruff check`, the full unit test suite (`pytest -q -x tests`), the import
smoke-check (`grasp_gym`/`stream_reach`/`vla_round`/`bimanual_carry`/`eval_closed_loop`/`run_learn_pilot`/
`run_sim_locomani`), and `bash -n` over every `scripts/*.sh` + `scripts/stack/*.sh`. The GPU gate itself was **not**
run (no live stack/GPU available in this session) -- see "GPU gate coverage" below for exactly what it must cover
before this lands on `main`.

### CLI mapping (ready_visible200)

`outputs/experiments/ready_visible200_20260909/launch.sh`'s argv:

```
--out $E --reach-config configs/collection/ready_grasp_visible.json
--scene outputs/experiments/ready07d_200_20260908/scene
--successes 200 --holdout 10 --max-batches 40 --seed0 30000
--steps 3000 --save-steps 3000 --train-gpu 0 --eval-rounds 100 --eval-seed 6000
--gym-env GYM_SHUFFLE_BAND=0
--stack-env CUDA_VISIBLE_DEVICES=1 --stack-env MUJOCO_EGL_DEVICE_ID=1
```

is now:

```
python src/locomani/pipeline/run_learn_pilot.py \
  --config configs/pilot/learn_200.json \
  --task configs/tasks/bottle_visible.json \
  --out $E --scene outputs/experiments/ready07d_200_20260908/scene \
  --set train_gpu=0 \
  --set gym_env=GYM_SHUFFLE_BAND=0 \
  --set stack_env=CUDA_VISIBLE_DEVICES=1,MUJOCO_EGL_DEVICE_ID=1
```

`--reach-config` is gone: `--task` loads a `BottleTaskConfig` whose nested `reach` field carries exactly what
`ready_grasp_visible.json` did (`tests/test_configs.py::BottleVisibleTaskTests` asserts the two are equal, retired keys
included, and that `visible_brake_depth == 0.31`). `configs/pilot/learn_200.json` already carries
successes/holdout/max-batches/seed0/steps/save-steps/eval-rounds/eval-seed at exactly this launch's values, so only
`train_gpu` (machine-specific) and `gym_env`/`stack_env` (this specific GPU-sharing arrangement) need `--set`.
`tests/test_configs.py::LearnPilotReproductionTests` asserts the loaded-plus-`--set` config reproduces every one of
these values, plus that fields the old argv never touched (`batch_rounds`, `squat_max`, `through`) keep their
argparse-default values.

### Ambiguous defaults and judgment calls

- **HAND, GYM_UPPER_SLEW_DEG, TF_GRIP_DELAY, LOCOMANI_REPLACE, the six LOCKSTEP_* tuning fields, WRIST_CAMS,
  CRATE_SETBACK, GYM_RESTART_DEPLOY, LOCOMANI_POSE_SIDECAR, GYM_ARM_PIN, GYM_ARM_CHANNEL/VEL, GYM_LEFT_GUARD,
  GRIPPER_GEAR, COUPLE_FINGERS, WALL_FEATURE, WALL_WALK**: none of these is ever set to a non-default value in any
  committed script or config (grepped `scripts/`, `configs/`, `src/` for `NAME=` before touching each one). Converted
  to plain module-level defaults seeded from the config dataclass, with no further plumbing to make them
  live-overridable end-to-end -- doing that fully would mean threading a config object through files (`primitives.py`,
  `primitives_const.py`, `bimanual_wall.py`'s cascading geometry constants) that this step's scope did not otherwise
  touch, for knobs nobody currently exercises. If one of these needs a real live override later, the field already
  exists in the dataclass; only the plumbing from config to that specific call site is missing.
- **TF_DATASET_FPS and SIM_VIDEO_INCLUDE_INIT were NOT in that list** -- both are actively set (`run_learn_pilot.py`/
  `run_ready_pilot.py` set `TF_DATASET_FPS=25`; `run_learn_pilot.py`'s collect stage sets `SIM_VIDEO_INCLUDE_INIT=1`).
  This was caught only because step 3's "never overridden" check was re-run against `src/` (not just `scripts/` +
  `configs/`, which is what step 1's own check used and is too narrow) -- see the `2081484` commit message for the
  full account. `vla_round.DATASET_FPS` is now set explicitly by `eval_closed_loop.py` from
  `cfg.backend.dataset_fps`, same guard pattern as `GRIP_GAIN`. `grasp_gym.py` gained a real `--video-include-init`
  CLI flag (it runs as its own subprocess from the collect stage, unlike `eval_closed_loop.py`'s in-process call) and
  `run_learn_pilot.py` passes it instead of exporting env. The one accepted, non-numeric regression: `eval_closed_loop.py`'s
  in-process `grasp_gym.main()` call previously inherited `SIM_VIDEO_INCLUDE_INIT=1` incidentally through the shared
  `base_env`; it now defaults to `False` there, changing only whether the eval stage's diagnostic mp4 includes the
  ready-posture prelude -- not any recorded/measured number.
- **SIM_VIS, SCENE_RANDOM, SONIC_DDS_DOMAIN, the bare LOCKSTEP toggle**: these ARE actively set by real scripts, but
  the code that reads them (`sim_vis.py`, `privileged_pub.py`, `run_sim_locomani.py`, `lockstep_sim.py`) runs inside
  the MuJoCo sim, a separate OS process `start_stack.sh` launches with `os.environ` as its only channel in
  (upstream's `SimLoopConfig`/tyro CLI is not ours to extend). These stay as `os.environ` reads at that one receiving
  end, whitelisted in `scripts/lint_env.py` with the reasoning spelled out there; `locomani/pipeline/stack.py` is the
  sole writer for the Python side (`run_learn_pilot.py`/`eval_closed_loop.py` no longer build these env dicts inline).
- **Whitelist additions beyond "genuinely process-level"**: `RANK` (asked for explicitly) is the only strictly
  process-level addition. `SIMPLE_ROOT`/`SIMPLE_PIN`/`RENDER_PY`/`ISAAC_ASSETS` (`paths.py`) and `SCENE_META`/
  `SCENE_XML` (`isaac_scene.py`) are paths in files outside this step's scope, the same category as the
  already-whitelisted `LOCOMANI_ROOT`/`LOCOMANI_SIM_DIR`. `TF_FRAME_AUDIT_PATH` (`run_instance_exporter.py`) is
  likewise a path. `TF_DATASET_FPS`'s one remaining read (`latency_dist.py`, an out-of-scope calibration tool) is its
  own `--fps` flag's default fallback, not a control-path read. `GYM_CREEP_OBS` (`rtc/diag_transition_cross.py`,
  out of scope) is written into a diagnostic event log as metadata, never read back to gate anything. None of these
  seven is a knob that changes robot behaviour; all are either paths or inert diagnostics in files this step's
  explicit scope did not include.
- **EvalConfig.spawn_edge / task / target are independent of BottleTaskConfig's own spawn/target/prompt.** Both
  `eval_closed_loop.py` and `grasp_gym.py` had their own bare CLI defaults for these (0.7 / "water_bottle" / "pick up
  the water bottle" in both, coincidentally) before this refactor; `EvalConfig` keeps that as its own default rather
  than nesting `BottleTaskConfig`, matching `eval_closed_loop.py`'s pre-refactor behaviour exactly. The canonical
  `configs/eval/closed_loop_100.json` overrides `spawn_edge` to `(1.1, 1.3)` to match what `run_learn_pilot.py`'s eval
  stage actually passed.
- **`PilotConfig.save_steps` is a pre-existing dead flag, left dead.** The old `run_learn_pilot.py` argparse parsed
  `--save-steps` but the train stage always used `args.steps` for both `MAX_STEPS` and `SAVE_STEPS`. Decision 8 does
  not license fixing unrelated bugs found along the way, so `save_steps` stays parsed-but-unused; documented at its
  field definition and at the one place it would have been wired in.
- **`configs/collection/` was NOT deleted.** The doc's closing line asks for it once "step 6 archives the verdicts",
  but (a) several files there (`ready_grasp_visible.json`, `ready_grasp_measured_prediction.json`) are loaded directly,
  by path, from committed unit tests (`test_reach_config.py`, `test_squat_prediction.py`, `test_visible_expert.py`)
  and from `outputs/experiments/*/reach_config.json` provenance, so deleting the directory now would break the test
  suite; and (b) "archives the verdicts" is not specified anywhere in reach -- there is no ledger entry format for it
  and no indication of which of the seven files' verdicts are settled versus still-referenced history. Left in place;
  flagging for a deliberate, separately-reviewed follow-up rather than guessing at what "archived" should mean here.

### GPU gate coverage

None of steps 1-6 changed control flow or numeric behaviour anywhere the tests above can see (they run offline). The
one path this refactor touches that only a live stack exercises, and that the bottle GPU gate must specifically
re-confirm before merge:

- `grasp_gym.py`'s prelude branch that checks `initial_pose.INIT_TOKEN_FILE` truthiness (was
  `os.environ.get("TF_INIT_TOKEN_FILE")`) -- exercised by `tests/test_ready_vla_handover.py` with a patched module
  attribute, but the real cross-process path (`eval_closed_loop.py` setting `initial_pose.INIT_TOKEN_FILE` before
  calling `grasp_gym.main()` in the same interpreter) only actually runs end to end against a live deploy.
- `stack.export_env`/`stack.start_stack`'s env dict reaching `scripts/stack/start_stack.sh` and
  `scripts/restart_deploy.sh` correctly for all three clients (async/lockstep/blocking) -- `bash -n` only checks
  syntax, not that the right variables land with the right values against the real shell script's `${VAR:-default}`
  reads.
- `vla_round.DATASET_FPS`/`GRIP_DELAY` and the other module-global overrides actually taking effect mid-round (the
  unit tests exercise the assignment, not a live inference loop consuming it).
- The `--video-include-init` CLI flag on `grasp_gym.py`'s collect-stage subprocess invocation, and the accepted
  cosmetic regression on the eval stage's diagnostic video noted above.
