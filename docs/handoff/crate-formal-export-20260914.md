# Crate visual expert: formal LeRobot export — 2026-09-14

Scope: the user chose to implement the bimanual line's formal export so a 200-episode crate
collection on GPU 1 can become a VLA training set. Usage contract:
[双臂采集 §4](../crate_collection.md#录制成正式训练数据) and [数据指南](../dataset.md).
Baseline `fcf23d9`; no deploy binary, weights, scene, gates or physics changed.

## What was blocking it

- `run_crate_pilot.py --record` was refused for the visual expert ("formal LeRobot export is
  not yet validated") and forced instance 0, the only instance `start_stack.sh` started an
  exporter on. Instance 0 / GPU 0 was busy with the single-arm 200-episode collection.
- The visual expert never drove the exporter (`c` / `c` / `x` keys) and its manifest rows had
  no `task`, no `target`, and the training boundary at the top level instead of under
  `execution.metrics.training_tail` where `convert_to_hf_format.py --trim-verification-tail`
  reads it.
- `check_dataset.py` judged the grasp with one wrist against the object centre and the
  object's rise in the pelvis frame, which under-reads a lift the torso takes part in (the
  same fallback the [bottle handoff](bottle-direct-20260914.md) flagged as unresolved).

## Implementation

1. `scripts/stack/start_stack.sh`: the exporter starts on instance 0, or on any instance when
   `RECORD_INSTANCE=1`. Its ports (`locomani.sim.ports`) and session root
   (`paths.session_root`: `$LOCOMANI_TMP/p03_i<N>/outputs`) were already per-instance, and
   `stop_instance.sh` already kills only the exporter serving that instance's camera port.
2. `run_crate_pilot.py`: `--record` is accepted for `visual`, no longer forces instance 0,
   passes `RECORD_INSTANCE=1` to the stack, and waits for the new exporter session dir after
   each `start_stack` (the single-arm pilot's rule) and stores it in `<out>/inputs.json`
   `sessions`, joined by order to `manifests`.
3. `tasks/crate/expert.py`: `--record` (refused without a live exporter via
   `ports.require_exporter`); `c` after the overview video starts so frame 0 is the task;
   `c` (save) or `x` (discard) after evaluation on every path that started recording,
   followed by a 2 s IDLE hold so the exporter has rows past the boundary. Manifest rows now
   carry `task` (the prompt), `target`, `produced`, and for successes
   `execution.metrics.training_tail` mirroring `training_tail`.
4. `pipeline/check_dataset.py`: the crate profile (`3cam_bimanual`: state 60, action 66, three
   cameras) is audited when the task text names the crate; proximity uses the midpoint of
   both wrists. New `--run-dir`: `pipeline/world_lift.py` reads the run's `inputs.json`, the
   manifest label of each produced row, the exporter's `priv_seq` and the sim's state trace,
   and returns the target object's world z for every converted row (exact seq match, no
   interpolation). The converter now imports the same 50 → 25 Hz row pick from that module,
   so the two cannot drift.
5. Tests: `test_crate_pilot` (visual records on its own instance), `test_crate_visual_execution`
   (key protocol `c`,`c` / `c`,`x`, `produced`, manifest fields), `test_world_lift`.
   Docs: crate, visual, dataset guides; `silent_failures.md`, `change-map-crate.md`.

## Validation (GPU 1 / instance 1, beside the bottle collection on GPU 0 / instance 0)

`outputs/crate_record_val_20260914`, seeds 7000–7009, `--record`:

| Step | Result |
|---|---|
| Stack | exporter started on instance 1, pose column `blue_crate`, session `/workspace/tmp/p03_i1/outputs/2026-09-14-12-37-17`; pids 2556 of 3840 with two recording stacks |
| Attempts | 7/10 OK; 9 produced (one `invalid_initial_distance` before recording); session has 9 episodes, discarded `[0, 7]` = the two `basket_tilt` failures, join exact |
| Convert | 7 episodes, 2,829 frames at 25 Hz, verification tails trimmed (e.g. 806/946 rows kept), task "pick up the crate", 21 video clips |
| Audit without `--run-dir` | 37 pass, 2 fail: wrist-to-object 3/7, pelvis-frame rise 2/7 |
| Audit with `--run-dir` | 39 pass, 0 fail; world rise after close median 21.6 cm, min 21.2 cm |
| Converter refactor | re-conversion of the same session is byte-identical on sampled parquet files |

The 260-attempt production run is `outputs/crate_vla200_record_20260914` (seed 8000, 26
batches, GPU 1 / instance 1). The earlier non-recording run `outputs/crate_vla200_20260914`
was stopped after 18 attempts (15 OK) because its output cannot be converted; its videos are
retained.

## Limits

- Format-ready is not learning-proof: both-gripper VLA inference and a closed-loop eval on
  crate data are still unvalidated.
- `--run-dir` needs a manifest `label` per produced row; the single-arm pilot's manifest has
  none, so its datasets still fall back to the pelvis-frame rise unless a root position is
  recorded. The exporter still writes no `observation.root_position`.
- The image/state 1–2 frame alignment question from the bottle handoff applies unchanged.

## Production run and dataset (later on 2026-09-14)

`outputs/crate_vla200_record_20260914`: 260 attempts, 219 successes (84%), 26 sessions under
`/workspace/tmp/p03_i1/outputs/`. Converted with `--holdout 10`: `train_dataset` 209 episodes /
85,721 frames, `val_dataset` 10 episodes / 4,089 frames, both trimmed at `training_tail`.
Stats computed with `configs/gr00t_sonic_3cam_bimanual.py`. Audit with `--run-dir`: 39/39 on
both sets (world rise after close: median 21.6 cm, min 19.4 cm). Two training episodes had
their first exporter row one tick before the trace start (the `c` key and the trace start
race across processes); `world_lift` maps up to two leading/trailing rows to the edge sample
and still refuses larger gaps. Training: `scripts/train_bottle.sh` with
`MODALITY=configs/gr00t_sonic_3cam_bimanual.py`, 10,000 steps on GPU 1 (~2 s/step).
Training finished: 10,000 steps in 5.1 h on GPU 1, mean train loss 0.095, last steps ≈0.05;
checkpoint `outputs/crate_vla200_record_20260914/model/checkpoint-10000` (24 GB; it was first
written to a stray `model/model` under the repo root because the chain script's `RUN=model`
prefix assignment leaked into `OUT_ROOT="$PWD/$RUN"`, and was moved).

## Closed-loop evaluation (2026-09-15)

Implemented as `pipeline/eval_crate_closed_loop.py` + `tasks/crate/vla_rounds.py` (usage and
per-round protocol: [双臂 Closed-loop](../evaluation.md#双臂-closed-loopsrclocomanipipelineeval_crate_closed_looppy)).
Supporting changes: `vla_obs.py` takes a `TaskSchema` (`ObsBuilder(schema=SONIC_CRATE)`,
`PolicyLink.get_chunk_bimanual`), `sonic_token.expand_hands_bimanual`, `initial_pose.load_token_file`
accepts the dict form with `arms`, and `configs/deployment/crate_initial_token.json` holds the
training data's first-frame token and arm posture. Ports: the policy server defaults to 5700
because 5591 is instance 1's privileged port; the run used instance 2 / GPU 1 while the
single-arm eval held 5590 on instance 0.

| Run | Rounds | Result | Failures |
|---|---|---|---|
| `outputs/crate_eval10_20260915c` (validation) | seeds 6000–6009 | 9/10 | 1 reach without a close |
| `outputs/crate_eval100_20260915` | seeds 6000–6099, 10 batches | **71/100**, Wilson 95% 0.61–0.79 | 20 `close: commanded a grasp but no lift` (18 end with neither hand touching, rise 0), 5 `reach: never commanded a close` (min distance ≈0.34 m), 4 `[stand] robot did not stabilise after reset` |

Successful rounds: rise median 15.2 cm, tilt median 7.4°, 18 s from the first chunk. The first
ten seeds scored 8/10 in the 100-round run against 9/10 in the validation run on the same
seeds and checkpoint, so single rounds are not repeatable across stack restarts. Start-edge
distance does not separate the close failures from successes (median 1.40 vs 1.35 m). The
dominant failure is a close commanded with the hands off the crate, i.e. reach precision at the
rim, not the walk. Videos: `videos_NN/crate_vla_KKK.mp4` per batch (round = 10·NN + KKK);
previews `previews/videos_NN_crate_vla_KKK.mp4`, 100 files ≤ 499,012 bytes, 38.2 MB total,
`previews/verification.json`.
