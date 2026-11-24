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

## Gaps and limits

- Success rates on SIMPLE vary run to run by several episodes in 70 with identical code; compare versions on many
  repeats of the same ids, or with perception frozen, not on one pass of many ids.
- target=truth `art()` takes the moving part as a moving body's collision geometry of at most 0.5 m, ranked by its
  distance from the hinge; checked on the door and the oven only. Faucet, trash can and the five pick tasks other than
  XMovePickTeleop were not smoked under truth or rgbd.
- self_pose=truth and controller_root=truth are refused until TODO 49.3. On the esc bed the loops always used leg
  odometry for both (OdoRoot is ScaleBFM's root_source, re-anchored at the true pelvis at each stand; the anchor
  cancels because goals go through Belief.to_sim). An earlier report of mine said the opposite; it was wrong.
