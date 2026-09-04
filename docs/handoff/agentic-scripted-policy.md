# Handoff: agentic iteration on the scripted policy and scene generation

Written 2026-09-02 by the training/eval session ("99"). The new session gets
ONLY this file, so it states conclusions and points at the docs/memories that
hold the evidence. Nothing here is committed; the working tree is dirty (see §5).

## 0. What the user asked for (two readings)

Verbatim: "想以agent的方式修改之前的scripted policy和场景生成等过程".

- **Reading A (outer loop over code):** an agent (a Claude Code session, or an
  automated loop it writes) reads the outcomes of a collection/eval run
  (failure anatomy, `outputs/eval/*/closed_loop.json`, reels) and edits the
  scripted grasp expert and the scene sampler, re-runs, repeats. The human
  stops hand-editing `grasp_gym.py` / `scene_lib.py`.
- **Reading B (agent as the expert at runtime):** replace the fixed scripted
  policy with an LLM/VLM-driven agent that decides waypoints / grasp params
  per episode from observations, and likewise generates scenes on the fly.

Both are plausible. A is cheaper and matches the phrase "修改…过程"; B changes
the data distribution itself. **Confirm with the user which one (or both,
staged A -> B) before writing code.** The rest of this doc serves both.

## 1. Code map: scene sample -> scripted grasp -> SONIC stream -> dataset

Entry points (all under `src/teleop_free/` unless noted):

| file | role (one line) | safe to modify? |
|---|---|---|
| `start_stack.sh` / `stop_stack.sh` | bring up / tear down sim + C++ deploy (+ exporter); `INSTANCE=k` selects ports base+10k and DDS domain k | launcher only; keep the readiness markers and the `SIM_VIDEO_DIR`-before-start rule |
| `run_sim_locomani.py` | MuJoCo sim process: scene build, privileged publisher, sim-side arm pin hook, lockstep hook | scene/physics hooks OK; do not touch the LowState/LowCmd bridge |
| `scene_lib.py`, `scene_random.py`, `compose_scene.py`, `build_sim_scene.py`, `inject_physics_into_mjcf.py` | asset library, per-episode randomization, MJCF composition, physics injection. `SCENE_PRESET` is read in `scene_lib.py:849` | **yes, this is the scene-generation surface** |
| `grasp_gym.py` | the episode driver: sample scene -> reset+settle -> walk (planner) -> approach/reach/grasp (`grasping.py`, `adaptive_grasp.py`, `arm_ik.py`, `stream_reach.py`) -> lift -> verdict; writes the manifest and dumps samples. Flags: `--rounds --seed --spawn-edge --spawn-lateral --spawn-yaw-err --edge-jitter --until-ok --one-shot --mid-band --targets --dart-frac --dart-far --view-gate-x/y` | **yes, this is the scripted-policy surface** (`run_round`/`_run_round`) |
| `primitives.py`, `primitives_const.py`, `navigator.py`, `upper_ref.py`, `initial_pose.py`, `reach_map.py`, `wrist_map.py` | motion primitives (walk/turn/hold/reach), navigation to the table, upper-body references, standing token, measured reach map | tunables yes; the ZMQ seams they call are not |
| `agent_server.py` | `AgentCommander`: the ZMQ side of the deploy (planner commands, protocol-v4 token frames, streamed mode). **Owns the PUB bind on 5556+10k** | protocol seam — do not change message shapes or timing |
| `privileged_pub.py` | sim-side privileged state on :5581+10k, episode reset/control on :5582+10k, `ARM_PIN` | seam; adding fields is fine |
| `sonic_token.py` (tools) / `sonic_encoder.py` / `sonic_decoder.py` | FSQ token codebook, snap, encoder/decoder wrappers | do not change quantisation |
| `vla_round.py` | closed-loop VLA client (blocking / async / lockstep), ensemble, RTC flags, `ObsBuilder` | the eval line's file; coordinate before editing |
| `collect_episode.py`, `zmq_driver.py`, `run_batch.py` | older collection drivers (superseded by `grasp_gym.py`) | leave |
| `sim_video.py`, `sim_vis.py`, `make_reel.py` | third-person recording, overlays, reels | yes |
| `scripts/collect.sh`, `collect_dagger.sh`, `collect_pilot.sh`, `par_batch.sh` | collection wrappers (stack + gym + exporter) | yes, keep the ordering |
| `tools/convert_to_hf_format.py`, `tools/check_dataset.py`, `tools/failure_anatomy.py`, `tools/seq_mcnemar.py` | LeRobot conversion + audit, failure taxonomy, paired test | yes |
| `tools/eval_closed_loop.py`, `tools/serve_policy.py` | closed-loop eval driver and GR00T policy server | eval line's files |

Dataset write path: the deploy's exporter (`gear_sonic/scripts/run_data_exporter.py`,
started by `start_stack.sh`) records LowState/tokens/images per episode; the
gym marks episode start/stop through the :5582 control channel; conversion to
LeRobot is `tools/convert_to_hf_format.py` (see `docs/dataset.md`; the audit
pitfalls are in memory `lerobot-dataset-audit`).

Load-bearing seams (do not "clean up"):

- **ZMQ teleop seam**: four undocumented protocol details — memory
  `sonic-zmq-teleop-seam`; token path facts — `sonic-token-deployment-path`
  (launch with `NO_ENCODER=1`, prime before switching, 1/16-quantised token).
- **Wall-clock principle**: sim, deploy (50 Hz) and agent run on real time; the
  deploy's INIT ramp, the planner settle, and the 25 fps token cadence are all
  timed. Anything that pauses the agent while streaming freezes the intent.
- **Port ownership**: the commander binds PUB 5556+10k; sim binds 5555/5581;
  deploy binds 5557; eval server 5590 (clashes with INSTANCE=1's keyboard port
  — use 6590/6591). One policy server per GPU.
- **Streamed-motion channel is the real interface** (29-DoF pose + IK, ~1 cm
  palms); the planner wrist channel is dead — memory `sonic-streamed-motion-channel`.

## 2. Scripted policy: failure anatomy and levers (pointers)

Read these before touching the expert; each cost a day:

- Reach envelope: compliance/height fields are inert, palm reaches 0.50 m,
  wrist target not tracked 1:1 — `sonic-reach-envelope`; measured map
  `docs/reach_map.png`, `reach_map.py`.
- Elbow sign: elbow 0 = forearm level, positive extends DOWN — `g1-elbow-zero-is-level`.
- Sim-side arm pin (frozen-upper + :5582 `pin_arms` PD) — `sonic-arm-pin-sim-side`.
- Planner heading anchor rotates on every mode switch; compensated via
  `heading_offset` + `reanchor()` — `sonic-planner-heading-anchor`.
- Height only works in IDEL_SQUAT; commanded waist pitch is never delivered —
  `sonic-planner-depth-channel`.
- Ego camera visibility gates and the reach/visibility conflict (deeper
  stand-off impossible; reach map ends at x≈0.22 at can height) —
  `ego-camera-vla-visibility`; gate flags `--view-gate-x/y` in `grasp_gym.py`.
- Encoder sees only four inputs; token is pure intent — `sonic-encoder-sees-only-four-inputs`.
- Silent-failure table: `docs/reference.md` (read it whole).

Collection history that constrains the design (docs, not jsons):

- Continuous "walk then grasp in one go" collection (flow200): `--one-shot
  --mid-band 0.20`, hold after approach removed, `RAISE_FINISH_MAX_S 0.30`,
  spawn-edge 1.1–1.3 aligned with eval; 200/252 rounds OK. Recorded in
  `docs/collection.md` and the last commits (`git log -3`).
- The 2x2 verdict: DART-style perturbed data is necessary (dart200 73/100 vs
  walk200 8/20, p=0.021) — `docs/evaluation.md`; memory `vla-baseline-round1-results`.
- Dominant closed-loop failure today: "at the table, arms ready, never
  commands a close"; success is keyed to terminal pelvis-to-bottle distance
  ≤ ~0.27 m (2026-09-02 finding, `docs/lockstep.md` will carry it).

For Reading A the natural loop is: `tools/failure_anatomy.py` on the traces
-> decide a lever from the list above -> edit `grasp_gym.py`/`primitives.py`
-> smoke (§4) -> paired eval.

## 3. Scene generation today

- Randomized per episode (`scene_random.py` / `grasp_gym.py`): target object
  from `--targets` (default pool; `TARGETS=water_bottle` narrows it), object
  pose jitter on the table (`JITTER=0.20` in the flow200 runs — verify the env
  name in `scene_lib.py`/`start_stack.sh`), appearance seed, robot spawn
  (edge distance, lateral offset, yaw error via the `--spawn-*` flags),
  `--edge-jitter`, optional DART perturbations (`--dart-frac`, `--dart-far`).
- Fixed: table geometry/height (`table_top` in the manifest), camera intrinsics
  and head pitch 52 / fovy 70 (`ego-camera-vla-visibility`), physics injected
  by `inject_physics_into_mjcf.py`.
- Presets: `SCENE_PRESET=bottle_only` (single bottle, used for every eval and
  for flow200). Without it the multi-object table appears — this bit us once
  mid-session; always set `SCENE_PRESET`, `TARGETS`, `JITTER` explicitly.
- Assets: `scene_lib.py` (library + presets), `outputs/scene/`,
  `docs/rendering_survey.md`; Isaac re-render assets under `isaac_scene.py`
  belong to the rendering line (§5).
- Constraints on any new scene: the object must pass the two visibility gates
  at spawn AND lie inside the reach map at grasp height; stand-off deeper than
  the gate cannot be fixed by camera changes (memory above). New objects need
  the LeRobot audit (`lerobot-dataset-audit`) and a grip-height entry.

## 4. How to evaluate a change

Collection smoke (one instance, ~10 rounds):

```
export SIM_VIDEO_DIR=$PWD/outputs/videos/<name>   # BEFORE start_stack.sh
SCENE_PRESET=bottle_only TARGETS=water_bottle JITTER=0.20 \
  SONIC_DEPLOY_BIN=/workspace/deps/gear_sonic_deploy_ours_lockstep_nvdef/target/release/g1_deploy_onnx_ref \
  bash src/teleop_free/start_stack.sh "pick up the water bottle"
python src/teleop_free/grasp_gym.py --rounds 12 --seed 7000 --spawn-edge 1.1 1.3 \
  --one-shot --mid-band 0.20 --record --manifest outputs/<name>/manifest.jsonl
bash src/teleop_free/make_reel.py ...      # always export a reel (memory always-export-video-after-changes)
```

**Never set `NO_ENCODER=1` for collection** (found 2026-09-03 by the agentic session): the
planner-mode upper-body channel needs the encoder loaded; without it the deploy silently drops
the arm and crouch references (IK res 0, palm 8 cm off the reference, crouch 0 cm) and every
round that reaches the up phase stalls. `NO_ENCODER=1` is for token/VLA deployment only.

Closed-loop eval (`docs/evaluation.md`): start the stack, THEN the server
(`stop_stack.sh` kills `serve_policy.py` by name), then
`tools/eval_closed_loop.py --fresh N --seed 6000 --spawn-edge 1.1 1.3
--client async --ensemble 3 --ensemble-m 0.3 --policy-port <port>`.

Rules:

- Compare PAIRED on the same seeds with McNemar (`tools/seq_mcnemar.py`);
  50 rounds cannot resolve a 10-point difference; the view gate re-samples
  spawns, so "same seed" is not "same spawn" — pair on round index anyway and
  report discordant pairs.
- Verdicts are checkpoint-conditional (memory
  `deployment-verdicts-are-checkpoint-conditional`) and, as of today,
  deploy-binary-conditional (§5).
- Shell: never put `serve_policy.py` / `run_sim_locomani.py` literals on a
  command line that also runs `stop_stack.sh` — pkill self-match (memory
  `pkill-self-match-in-shell-chains`); run chains from script files.
- `push_videos.sh` clears `videos/`; disk: `/` is a nearly-full 32 GB
  overlay, keep TMPDIR/caches on `/workspace` (`vast-instance-disk-layout`).
- Ports/GPUs currently: nothing of this line is running at the time of
  writing (calibration stopped 2026-09-02 ~07:00). Another user's process
  holds ~3.2 GB on GPU1 (conda env `vg`, not ours). Rendering session: ask it
  which INSTANCE/ports it holds before launching a stack; default to
  `INSTANCE=2` for the new branch and source `scripts/paths.sh` AFTER
  exporting `INSTANCE`.

## 5. Boundaries with the two existing lines

Training/eval line (this session): `vla_round.py`, `tools/eval_closed_loop.py`,
`tools/serve_policy.py`, `tools/latency_dist.py`, `lockstep_sim.py`,
`lockstep_vendor/`, `start_stack.sh`/`scripts/restart_deploy.sh` (LOCKSTEP and
`SONIC_DEPLOY_BIN` branches), `src/gr00t/model/gr00t_n1d7/gr00t_n1d7.py`
(RTC-train patch), `third_party/GR00T-WholeBodyControl` (branch `lockstep-port`),
`/workspace/deps/gear_sonic_deploy_ours_lockstep`, `docs/lockstep.md`,
`docs/evaluation.md`. Do not edit without coordinating. Open finding that
affects everyone: the production deploy binary (Aug 17 build) executes walking
tokens at ~1/3 amplitude vs the lockstep-port build; which build collection
and eval should standardise on is undecided — do not rebuild
`gear_sonic_deploy/target/release` yourself.

Rendering line (session "6e"): `RENDER_BACKEND` migration, Isaac offline
re-render, live Isaac mirror — `isaac_scene.py`, `sim_video.py`/`sim_vis.py`
render paths, `docs/rendering_survey.md`, `outputs/renders*`. Coordinate on
`run_sim_locomani.py` camera/render code and on `scene_lib.py` asset paths.

Shared working tree is dirty and uncommitted (flow200 + RTC + lockstep work).
The new branch should start from its own worktree (`git worktree add`) on
`main`; do not commit or revert the shared tree's changes.

## 6. Questions for the user before starting

1. Reading A or B (§0)? If A: which outcome signals drive the loop (failure
   anatomy classes, success rate, reel review) and what may the agent change
   autonomously vs propose?
2. Success metric for the branch: collection OK-rate, or downstream closed-loop
   success of a model trained on the new data (that costs ~4 h per checkpoint)?
3. Which deploy binary to standardise on (production Aug 17 vs lockstep-port)?
   This changes every number.
4. Budget: an instance/GPU for the new branch, and whether it may run stacks
   in parallel with the other two lines (CPU/GPU contention changes wall-clock
   behaviour — one policy server per GPU).
5. Scope of scene generation: same table+bottle only, or new objects/tables
   (each needs the visibility/reach gates and the dataset audit)?
6. Whether the branch may modify `grasp_gym.py` in place or must fork it
   (`grasp_gym_agentic.py`) to keep flow200 reproducible.
