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

## BiGym 2.0 port (TODO 49.5, 2026-11-24)

The smallest useful adapter in `src/locomani/bench/bigym2/`: target=truth, controller=native (BiGym's frozen GR00T-WBC), the rgbd channel built.
The environment server runs in BiGym's own venv as a standalone file (`server.py`, no locomani imports) and is started by the bed process
(`client.ServerLink`), which it dies with; the machine's one-simulator lock is `session.Lock`.

Install (third_party is shared by the worktrees): `git clone https://github.com/swirl-uk/BiGym2 third_party/BiGym2` at commit
`35ec40923421aff26ac2a525adea428567732898` (2026-11-12, `fix(robots): snapshot the gripper slew targets`), then in that directory
`uv sync --frozen --python 3.12` -> `.venv` with Python 3.12.3, MuJoCo 3.8.1 (its `uv.lock`; substrate `bigym2-mj381-v1`), onnxruntime from the
lock. Nothing was installed into /venv/main. Licence: Apache-2.0; the GR00T weights fall under the NVIDIA Open Model License.

```bash
PYTHONPATH=src /venv/main/bin/python -m locomani.bench.bigym2 run drawer_top_open --ids drawer_top_open/P/0-4 --out rows.jsonl [--set depth=true]
```

`--ids` is a file or a spec `<task>/<pool>/<a>-<b>`. Pools: P practice (seed 1000+n), A acceptance (100000+n), S sealed (700000+n); `--set pools=PA`
opens more than P. BiGym's evaluation seeds [620000, 630000) are refused in five places (config, CLI, `check_id`, `reset_msg`, the server's own
check); the demonstration seeds 1-199 are never used. Tasks that ignore the seed (checked on all 20 by hashing the start state under three seeds):
`dishwasher_close`, `wall_cupboard_open`, `wall_cupboard_close` (their pools repeat one start); the other 17 draw from it, and a reset is repeatable.

- Truth view and judge: `privileged.py` (poses, scene joints with axes, the task's own facts, collision vertices), `truth.py` (`view.goal()`: the
  reach sphere, the top drawer's handle bar and pull axis), `judge.py` (BiGym's verdict plus: the hands, bodies of the Dex1 gripper, touched the
  task body on at least 5 steps and the body touched it on fewer than 3; the server records every robot-body / jointed-scene-body contact per step).
- rgbd: `--set depth=true` makes the server replace `bigym.loco.env.CameraConfig` at run time by one that sets `depth=True` at the render shape
  (their file hard-codes `depth=False`, env.py:669; no file of theirs is edited); only the head camera renders. `Cameras.rgbd()` returns the
  `Frame` (rgb, metric depth, K from the 60 degree vertical field of view, the camera in the pelvis frame as x right, y down, z forward). The head camera is
  `g1_29dof_with_dex1_1/head` on `torso_link`; at the settled pose it looks steeply down at the work surface. Physics is unchanged by depth (identical step counts).
- Seed package `bigym2/seed/` (skill.py, frozen `__init__.py`), guard `guard.py` (R1-R8, `python -I guard.py --selftest`).
- Evidence and numbers: `/workspace/tmp/bench49/bigym2_notes.md`.

## HumanoidArena port (TODO 49.5b, 2026-11-24)

An adapter in `src/locomani/bench/humanoidarena/` in the BiGym pattern: target=truth, controller=native (the benchmark's TWIST2 tracker), the rgbd channel built.
Tasks in scope: `pp_box` (P&PBox), `double_desk`, `open_door`. The environment server runs in the benchmark's own venv as a standalone file (`server.py`, no locomani imports)
and drives the benchmark's own evaluation worker (`script/eval_scripts/twist2/sim_eval_vla.py`) as a library: the same task configs (`base_test/*_twist2_test.yaml`), the TWIST2
ONNX the repository ships, the benchmark's reward and fall detector, physics on the CPU as its own run script sets it. Only the policy's HTTP client is replaced in process by
one that returns the action the bed sends; no file of the benchmark is edited. The bed starts the server (`client.ServerLink`, pinned with `taskset -c 28-31`,
`CUDA_VISIBLE_DEVICES=1`, Kit's `activeGpu=1`) and the machine's one-simulator lock is `session.Lock`.

Install (everything under `third_party/HumanoidArena/`, nothing in /venv/main or another venv; no sudo, no apt; sizes and commands in `/workspace/tmp/bench49/humanoidarena_notes.md`):
`repo/` is the mirror clone at `68479287a784a69be9ce6ad739311d2f11f75ef9` (2026-08-10); `venv/` Python 3.11.15 with torch 2.7.0+cu128, `isaacsim[all,extscache]==5.0.0`, Isaac Lab
`release/2.2.0` (`IsaacLab/`, commit `46dff135f44683f031edf346e544fcfd8456b2bb`, `uv pip install -e` of isaaclab, isaaclab_assets, isaaclab_tasks), the repository's
`requirements.txt`, `h5py`, `unitree_sdk2_python` (`814556d1...`, `--no-deps`) and `cyclonedds==0.10.5` built against a conda-forge `cyclonedds` prefix (`cdds/`; the server needs
`LD_LIBRARY_PATH=cdds/lib`). The asset package (Google Drive zip, 2.5 GB, sha256 `d0a0c6be...`) is unpacked into `repo/isaaclab_twist2_g1/assets`. About 26 GB in all.
Kit starts on the RTX 5090 (driver 580, sm_120) with Isaac Sim 5.0; the first start took about 5 minutes (shader cache), later ones about 30 s.

```bash
PYTHONPATH=src /venv/main/bin/python -m locomani.bench.humanoidarena run pp_box --ids pp_box/P/0-4 --out rows.jsonl [--set overlay_dir=D --set snap_dir=D]
```

- Episode ids `<task>/<pool>/<n>`. The benchmark's evaluation episodes are the 60 seeds `sha256("<gym task>|<g>|<r>")` for g in {0,1,2}, r in 0..19. A pool episode is
  (group, repeat) = (first group + n // 20, n % 20) with groups P 1000-1999, A 2000-2999, S 3000-3999, its seed from the same formula. The evaluation groups and any seed equal
  to one of the 60 are refused in six places (config, CLI, `check_id`, `reset_msg`, `collides`, the server's own derivation); `tests/test_humanoidarena_episodes.py`.
- Truth view and judge: `privileged.py`, `truth.py` (`view.goal()` for pp_box: the box, a spot the benchmark's own check accepts, the layer height, the table and shelf boxes), `judge.py`:
  the benchmark's checks need no hand contact, so a run counts (`success`) only when the benchmark's reward reached 1 without a fall, a hand link (Dex3 palm, fingers) pressed
  on the task object on at least 5 steps, links other than the hand pressed on it on fewer than 3, and it moved at least 0.10 m. The record is a Isaac Lab contact sensor on every
  robot link filtered to the task object (force > 1 N per physics step, history of the control step), added to the scene config at run time (physics unchanged). `open_door` has no
  object record, so its compliant verdict is `no_record` (never ok).
- rgbd: the front camera, 640x480, K from the benchmark's pinhole (f = 243 px), the renderer's exact depth, the camera in the pelvis frame; `--set overlay_dir=D` saves the first frame
  of an episode with the object's projected centre and depth for a person to look at, `--set snap_dir=D` saves the third-person camera at every mark of the package.
- Native action space: 40 floats (root step in the reference heading frame, root height, root rotation as rot6d, the 29 canonical joints, two hand values). The first two columns of the
  rotation matrix, row-major (identity is `[1 0 0 1 0 0]`; the first guess was the rows and the robot fell at once).
- Gait: the tracker needs the legs of its reference to show a gait; moving the reference root alone left the robot standing (400 steps at 0.35 m/s, the pelvis did not move).
  `gait_build.py` (benchmark venv, `python -I`) averages the near-straight strides of the ten walking clips the repository ships (`TWIST2/assets/example_motions`) into
  `gait_cycle.json` (6 strides, period 1.31 s, 0.80 m/s); `env.gait` hands it to a package. With it the robot walks about 0.4-0.5 m/s and the pelvis steers within 2-3 cm of a goal.
- Seed package `humanoidarena/seed/` (skill.py, frozen `__init__.py`), guard `guard.py` (R1-R8, `python -I guard.py --selftest`).
- Numbers and the smoke: see below and `/workspace/tmp/bench49/humanoidarena_notes.md`.

Checked on this machine (2026-11-24, GPU 1, 4 cores): reset and 60 hold steps headless on `pp_box`, `double_desk` and `open_door` with the native tracker; the rgbd frame (640x480, depth 0.10-2.3 m)
with the overlay (the box's projected centre lies on the box, its depth reads 0.98 m); the judge on fake outcomes and the contact record on real ones. The renderer shows the stage as spawned until
the physics has stepped, so the overlay is taken at the package's first mark, not at the reset.

Seed package (`pp_box`, target=truth, native action space): walk to the table's edge level with the box, raise both wrists to the box's sides, squeeze, lift, carry to the layer the benchmark's check names,
lower, let go. Chosen because the box is the one object two palms can hold, the success check is geometric (layer height 0.648 m, a rectangle) and the whole scene is static and known to the truth view;
`double_desk` needs a pinch grasp of a hammer with a binary Dex3 hand, `open_door` the latch and a walk through the frame and has no object record for the compliance check.

Smoke, 5 practice episodes `pp_box/P/0-4`, target=truth, cold start through the bed: **0/5 compliant, 0/5 by the benchmark's own check**; 3 falls (P/2, P/3, P/4), 2 standing to the 1450-step limit;
hand contact steps 0 / 2 / 0 / 6 / 27, body contact steps 0 / 0 / 0 / 0 / 11; the box was moved 0.005 / 0.18 / 0.40 / 0.85 / 1.76 m; 59-161 s per episode (545 s in all, one Kit start inside).
Every episode reached the table (the walk with the gait cycle works, 2-3 cm of the aimed pelvis position). The flow stops at the arms: with the legs held at the standing pose and only the arm joint
targets changing (even at 0.006 rad per step) this tracker does not stand still: the pelvis wandered 1-1.5 m and turned up to 50 degrees within 6 s. What would be needed next: a leg and waist
reference that balances with the arms out (a recorded whole-body pose from the benchmark's data, or the SONIC tracker, `nvidia/GEAR-SONIC`, not downloaded), then the squeeze. Nothing here is a statement about the
benchmark's policies; it is the seed of a scripted flow.
