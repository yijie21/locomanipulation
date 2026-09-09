# Retired diagnostic/tool script catalogue

Refactor step 2, Group D family 3 (2026-09-09): these one-off diagnostic
scripts and analysis tools are deleted because nothing imports them and no
`scripts/*.sh` invokes them — each was independently re-verified against the
current tree (real `import`/`from ... import` statements and `.sh` process
references only; a prose mention of "`diag_x.py` measured ..." in a comment
or doc does not count as a dependency). Their one-line docstring description
and the first non-`docs/handoff/` doc citation found are recorded here so
that provenance survives the deletion. Per `docs/reference/standards.md`
("其余历史文档在 `docs/archive/`,只加索引不改内容"), this is an index, not a home
for the removed code -- `git show <commit>^:<path>` recovers any of them.

## `src/teleop_free/diag_*.py` (43 deleted)

| module | one-line description | doc citation |
|---|---|---|
| `diag_angled_grasp.py` | Show what a tilted, standing grasp would look like, before building it. | (none outside the refactor ledger) |
| `diag_can_watch.py` | Bystander: sample the privileged stream and log the can's life story. | (none outside the refactor ledger) |
| `diag_channel.py` | M1+M2: does the PLANNER channel deliver the arm better than the streamed one? | `docs/reference.md:215` |
| `diag_channel_report.py` | Read outputs/diag/channel.npz and print the channel comparison. | `docs/reference.md:215` |
| `diag_core_visit.py` | Wrist visits every dark-core (sweet-zone) voxel at standing height. | (none outside the refactor ledger) |
| `diag_core_zone.py` | Show the two-tone reach cloud: pale envelope + dark sweet-zone core. | (none outside the refactor ledger) |
| `diag_dex1.py` | Live Dex1 smoke: command open/close through the full chain, read back. | (none outside the refactor ledger) |
| `diag_ego_geom.py` | Where is the can visible from? Pure geometry, no sim needed. | `docs/ROADMAP.md:137` (+3 more) |
| `diag_hand.py` | How wide does this hand actually open? | (none outside the refactor ledger) |
| `diag_heading.py` | Measure the planner-mode heading offset, before and after a streamed toggle. | (none outside the refactor ledger) |
| `diag_height.py` | M3: is the planner message's `height` field actually honoured? | `docs/reference.md:218` |
| `diag_idle_drift.py` | How far does the robot wander while commanded to stand still? | (none outside the refactor ledger) |
| `diag_ik_grasp.py` | Pure-kinematics IK grasp bench: is the PLANNED hand pose any good? | `docs/reference.md:225` |
| `diag_lean.py` | M4: how much waist lean does the policy deliver, and what reach does it buy? | `docs/bimanual_collection_tutorial.md:244` (+2 more) |
| `diag_lockstep_stand.py` | Does the standing token hold / raise the robot under lockstep? | (none outside the refactor ledger) |
| `diag_lockstep_stand2.py` | Which bring-up order stands the robot under lockstep? | `docs/lockstep.md:96` |
| `diag_map.py` | Measure the command -> palm mapping over the region the grasp uses, and fit it. | (none outside the refactor ledger) |
| `diag_movement.py` | Which frame does the planner's `movement` vector actually walk in? | (none outside the refactor ledger) |
| `diag_nav.py` | Offline check of the map and the planner, with no sim running. | (none outside the refactor ledger) |
| `diag_pitch.py` | Can the palm be commanded to a given nose-down pitch, and how faithfully? | (none outside the refactor ledger) |
| `diag_plan_scan.py` | Visualize the JOINT squat-depth / stand-distance optimization. | (none outside the refactor ledger) |
| `diag_reach.py` | How accurately does a wrist target actually get tracked, and how far can it go? | (none outside the refactor ledger) |
| `diag_reach_error.py` | How far is SONIC's delivered gripper pose from the one IK asked for? | (none outside the refactor ledger) |
| `diag_reach_probe.py` | Probe the reach map against the real control stack. | `docs/reference.md:223` |
| `diag_reach_sweep.py` | Sweep the right palm across the body at GRASP height, on camera. | (none outside the refactor ledger) |
| `diag_reach_volume.py` | Show the 3D graspable volume riding down through a stepped squat. | (none outside the refactor ledger) |
| `diag_ready_walk.py` | Walk-only demo of the READY pose: forearms level, no grasping. | (none outside the refactor ledger) |
| `diag_repeat.py` | Is "command -> where the palm ends up" a function at all? | (none outside the refactor ledger) |
| `diag_settle.py` | Do the objects sit still when nothing touches them? | (none outside the refactor ledger) |
| `diag_shuffle.py` | Can SONIC shuffle? Step frequency, per-step advance, and STOP CARRY. | `docs/reference.md:221` |
| `diag_spawn_view.py` | Real spawns, real frame positions: where does the bottle sit in the ego view. | (none outside the refactor ledger) |
| `diag_speed.py` | What does the planner's `speed` field actually do? | (none outside the refactor ledger) |
| `diag_squat.py` | How low will the controller take the pelvis, and what makes it? | (none outside the refactor ledger) |
| `diag_squatmode.py` | Does LocomotionMode::IDEL_SQUAT give back the pelvis height the all-planner path commands? | `docs/ROADMAP.md:300` (+1 more) |
| `diag_stand.py` | Can the robot stand up from a reset? Watch pelvis height, second by second. | (none outside the refactor ledger) |
| `diag_station_keep.py` | Can planner mode hold a position at all, and what sets the floor? | (none outside the refactor ledger) |
| `diag_stream.py` | Can SONIC's STREAMED_MOTION channel place the palm where IK says? | (none outside the refactor ledger) |
| `diag_token_hold.py` | Net pelvis drift while the deploy holds ONE motion token. | `docs/evaluation.md:774` |
| `diag_token_replay.py` | Does the recorded 64-d token actually cause the motion we recorded? | `docs/ROADMAP.md:83` (+3 more) |
| `diag_turn.py` | How fast can the robot turn on the spot, and in which mode? | `docs/ROADMAP.md:200` (+4 more) |
| `diag_walk.py` | Measure how the deploy's `facing`/`movement` frame relates to MuJoCo world. | `docs/collection.md:159` |
| `diag_wrist.py` | Does the wrist *orientation* command do anything? | (none outside the refactor ledger) |
| `diag_wrist_cam.py` | Where should the wrist camera sit? Rendered from a real grasped frame. | `docs/ROADMAP.md:21` (+2 more) |

## `src/teleop_free/tools/*.py` (10 deleted)

| module | one-line description | doc citation |
|---|---|---|
| `analyze_pose_feedback.py` | Summarize measured fixed-world feedback outcomes, including early aborts. | (none outside the refactor ledger) |
| `audit_frame_sources.py` | Summarize source freshness and verify audit-to-dataset row correspondence. | (none outside the refactor ledger) |
| `audit_object_topple.py` | Separate pre-close object topples from later no-close timeouts. | (none outside the refactor ledger) |
| `probe_handover_geometry.py` | Read-only IK checks at recorded stalled/handover poses, not rollouts. | (none outside the refactor ledger) |
| `summarize_transport_trials.py` | Separate actual interventions from VLA prefix successes/timeouts. | (none outside the refactor ledger) |
| `motion_smoothness.py` | Compare the MOTION QUALITY of two LeRobot datasets, ours vs a reference. | `docs/ROADMAP.md:21` (+2 more) |
| `sweep_reach_feasibility.py` | Offline static approach grid; never a physical success-rate benchmark. | `docs/reach_feasibility.md:33` |
| `analyze_stance_grid.py` | Requested versus delivered stance, with failures kept in the denominator. | `docs/stance_validation.md:37` |
| `diag_arm_response.py` | Diagnostic only: fixed upper reference and wrist pulses in free space. | `docs/arm_response.md:7` |
| `analyze_arm_response.py` | Summarize fixed-reference/pulse diagnostics; gain is local and empirical. | `docs/arm_response.md:7` |

`probe_handover_geometry.py`'s own `from collect_episode import obj_top` was
already stale before this deletion (refactor step 2 family 2 removed
`collect_episode.py`); it is deleted here rather than fixed, since nothing
imports it.

## Named in the candidates doc but found to still be live -- NOT deleted

`docs/handoff/refactor-step2-candidates.md`'s row 79 ("unimported tools")
turned out to be stale for four of its entries; a fresh grep (real
`import`/`from ... import` statements, not prose mentions of the filename)
found real importers for each, so they were kept and are not part of this
family:

| module | real importer(s) |
|---|---|
| `src/teleop_free/tools/diag_crouched_handover.py` | `tests/test_continuous_handover.py`, `tests/test_planner_handover.py`, `src/teleop_free/tools/diag_stance_grid.py` |
| `src/teleop_free/tools/diag_stance_grid.py` | `tests/test_standing_settle.py` |
| `src/teleop_free/tools/analyze_posture_stages.py` | `tests/test_posture_analysis.py` |
| `src/teleop_free/tools/diagnostic_transport.py` | `src/teleop_free/tools/diag_crouched_handover.py`, `tests/test_diagnostic_transport.py` |

Two more `src/teleop_free/diag_*.py` files (not in the candidates doc's
named-orphan list, but part of "the other 23... in docs catalogues") were
independently found to have real importers and are also kept:

| module | real importer(s) |
|---|---|
| `src/teleop_free/diag_grasp_pose.py` | `src/teleop_free/export_grasp_model.py`, `src/teleop_free/make_hand_points.py` (both cited in `docs/collection.md:45`), `src/teleop_free/diag_angled_grasp.py` (deleted above -- not the reason it survives) |
| `src/teleop_free/diag_reach_map.py` | `src/teleop_free/diag_grasp_pose.py` (transitively live via the row above) |

The four crate-tutorial preflight tools named as exceptions in this task
(`reach_preflight.py`, `reach_preview.py`, `reach_wall_sweep.py`,
`walk_pose_preflight.py`) are also untouched and not re-listed here.
