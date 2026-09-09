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
