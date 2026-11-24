# Run switches for public benchmarks, and the SIMPLE port (TODO 49, 2026-11-24)

The user asked (2026-11-24) for a codebase where a run on a public benchmark is a set of switches: the benchmark
(SIMPLE, HumanoidArena, BiGym 2.0, LocoManiLab), whether grasp targets come from simulator truth or from perception,
and whether perception reads a stereo pair or one RGB-D camera. Decisions taken with the user: everything moves into
`src/locomani` (`bench/`, `perception/`); two more switches (self pose, the pelvis ScaleBFM is told); RGB-D without
noise by default; the simple_art loop paused until the loops run on the new code (TODO 49.6). Plan and status: TODO 49.

## How to run

```bash
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
PYTHONPATH=src $PY -m locomani.bench table
PYTHONPATH=src $PY -m locomani.bench show --bench simple --bench-set target=truth
```

The first prints every benchmark's capability table (which values exist, which run here, why one is missing); the
second resolves a run's switches (code defaults, `configs/bench/<bench>.yaml`, overrides) and prints the record every
row carries.

```bash
PYTHONPATH=src /venv/main/bin/python -m locomani.bench.simple run OpenOvenTeleop --ids ids.txt \
  --setting '{"feed_torque": true}' --out rows.jsonl --set kinds=P --bench-set sensor=rgbd
```

One SIMPLE task on the ids in `ids.txt` (one `<Task>/P/<n>` per line; `kinds=P` allows the development set, the
default `Q` the loop's pool; released episodes are always refused). It takes a slot (at most three Isaac servers on
GPU 0, shared with the esc bed's locks), starts the environment server and, for sensor=stereo, the stereo depth
service, makes the missing looks and GraspGen-X grasps, runs the episodes and writes one JSON row per id to
`rows.jsonl`; a row has the old bed's fields plus `bench` (the resolved switches), `simple_config`, `view` and
`depth_source`. SAM 3 must be up on port 18832 (`perception/sam3_server.py` in the sam3d env).

## What was checked (evidence under outputs/bench49_20261124/)

- 49.1 (commit 0246c3b8): switches, profiles, capability tables, interfaces; 17 tests.
- Backend merge (commit 2159195a): the esc branch's ScaleBFM changes merged onto main's; `motor_check loco` and
  `pick --walk` give identical results before and after, the 12 trace arrays of `pick --walk` differ by 0.
- SIMPLE port (commit 2159195a): inventory `simple_inventory.md`, spec `port_spec.md`, notes `port_notes.md`.
- Parity, first round (`parity1/compare.md`): 70 development episodes, old bed twice (A, B), new once. Agreement on
  compliant success A-B 0.914, A-new 0.929, B-new 0.871, which looked like noise, but the oven alone was old 46/50 against
  new 29/40 (Fisher p 0.021, `parity1/oven_followup.md`). Cause: the port had dropped `OffSim.reset` (the local
  default pose, then the environment's reset state), so the start hold ran on the default pose (pelvis 0.762 m instead
  of 0.797 m). Restored; regression test `tests/test_simple_client.py`.
- Parity after the fix (`parity2/`, run from a frozen snapshot): oven 42/50 against the old 46/50; over the 70 episodes
  compliant wins A 28, B 26, new 21, with paired counts A vs new 7:0, B vs new 7:2. Seventy noisy episodes could not
  settle this, so the comparison moved to deterministic checks (`parity2/equivalence.md`):
  - the start hold (50 steps) and the first look of all nine tasks: identical states, Belief tie, policy inputs;
  - perception on identical saved inputs (oven, trash can, door, faucet, XMovePick): bit-identical masks, points,
    targets, grasps and look clouds; SAM 3 is deterministic on an identical image;
  - whole episodes with perception frozen to an old run's outputs (oven, trash can, XMovePick; Handover and
    LocomotionPickBetweenTables, see equivalence.md): commands and states bit-identical at every step, old against old
    and old against new;
  - the run-to-run difference is the Isaac render (mean 0.4 grey levels, depth p95 about 6 cm, the same within the
    old bed), amplified by segmentation scores near the 0.3 floor.
- Step 6 (target=truth, sensor=rgbd, depth_noise=realsense; `step6/step6_notes.md`): smoke on XMovePickTeleop,
  CloseDoorTeleop and OpenOvenTeleop (P/0, P/10) for each combination, no errors; the rendered depth agrees with the
  stereo depth to 0.8-1.2 cm median, the overlay lines up (`step6/frames/`); realsense: 7 % holes, 0.8 cm median error.

## Self pose and the pelvis ScaleBFM is told (TODO 49.3, evidence under outputs/bench49_20261124/phase3/)

- `self_pose` (the skill's belief) and `controller_root` (ScaleBFM's root_source) are separate: `Belief(sim, mo, odo,
  root_odo)` builds Body, JawFeed and the stop-servo change only for odometry belief, OdoRoot only for an odometry
  root. The default (odometry, odometry) replays bit-identical to the committed code at every step (oven 13/26,
  XMovePick 89/0, perception frozen).
- Four combinations on 46 development episodes each (40 articulated, 6 XMovePick), compliant wins: odometry/odometry
  27, odometry/truth 25, truth/odometry 24, truth/truth 25: within same-code noise, not rankable. Larger than noise:
  with a true belief the stop error after the first walk is 1.3-2.3 cm instead of 2.9-5.7 cm; the mixed
  truth/odometry pair reaches worse on XMovePick (palm error 3.3 cm against 0.8), its two frames disagreeing.
- Truth still read on the odometry path (11 places, file:line in phase3_notes.md). The one that matters: the belief
  is re-tied to the true pelvis at the start and after every walk, so odometry error does not accumulate across walks
  as it would on a robot. Others: the true pose before the first walk, gravity from the true orientation (an IMU
  gives it), the reach settle ending when the true palm is within 1 cm (timing). A truth-free mode needs these
  replaced; it is not done.

## Gaps and limits

- Success rates on SIMPLE vary run to run by several episodes in 70 with identical code; compare versions on many
  repeats of the same ids, or with perception frozen, not on one pass of many ids.
- target=truth `art()` takes the moving part as a moving body's collision geometry of at most 0.5 m, ranked by its
  distance from the hinge; checked on the door and the oven only. Faucet, trash can and the five pick tasks other than
  XMovePickTeleop were not smoked under truth or rgbd.
- self_pose=truth and controller_root=truth were refused until TODO 49.3, now selectable. On the esc bed the loops always used leg
  odometry for both (OdoRoot is ScaleBFM's root_source, re-anchored at the true pelvis at each stand; the anchor
  cancels because goals go through Belief.to_sim). An earlier report of mine said the opposite; it was wrong.

## LocoManiLab port (TODO 49.4, 2026-11-24)

One bed for all 26 LocoManiLab tasks (the 20 practice tasks, 3 fridge-door, 3 low-drawer tasks) in `src/locomani/bench/locomanilab/`;
the three esc beds (escalation_20261116, lml_bed2_20261121, lml_bed3_20261121) were copies of one file set plus articulation code.
The engine is robosuite / robocasa on MuJoCo (not AI2-THOR); the environment server runs in LocoManiLab's own venv as a standalone
file and is started by the bed process (`client.ServerLink`), which it dies with.

```bash
PYTHONPATH=src /venv/main/bin/python -m locomani.bench.locomanilab run IThorKitchenApproachFridge_scene13 --ids ids.txt \
  --setting '{"feed_torque": true}' --out rows.jsonl --set kinds=P --bench-set depth_noise=realsense
```

`ids.txt`: one `<Task>/<kind>/<n>` per line. Kind `Q` is line n of the task's practice file (the loops' pool, default), `P` and `W` are
development episodes (`--set kinds=P`), the benchmark's own episodes (kind `E`, jitter seed 0) are refused in every layer: the CLI before
a slot is taken, `episodes.check_id`, `episodes.reset_msg` and the environment server (`seed_floor` as an argument; tests/test_lml_episodes.py).
SAM 3 must be up on port 18832. Switches: `target=perception` (rgbd, depth_noise none or realsense: no right-camera depth, hence no
occlusion shadows), `self_pose` and `controller_root` both values, `target=truth` for the 20 original tasks with `self_pose=truth`;
`sensor=stereo` stays unsupported (one head camera). Checks and results: `lml_port_notes.md` (outputs under /workspace/tmp/bench49/lml/).
